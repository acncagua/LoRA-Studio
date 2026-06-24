from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from app import settings
from app.db import connect, create_job, fetch_one, latest_environment, utc_now
from app.services.command_builder import prepare_job_files
from app.services.training_runner import sd_scripts_subprocess_env


SMOKE_TARGET_PROFILES = {
    "adamw8bit_sdxl_balanced",
    "paged_adamw8bit_sdxl_balanced",
    "prodigy_sdxl_soft",
    "adafactor_sdxl_fixed",
    "adafactor_sdxl_auto",
    "lion_sdxl_soft",
    "dadaptadam_sdxl_auto",
    "dadaptlion_sdxl_auto",
}


def profile_validation_badge(profile: Any | None) -> dict[str, str]:
    status = (profile["validation_status"] if profile and "validation_status" in profile.keys() else None) or "untested"
    labels = {
        "untested": ("Untested", "warning"),
        "prepare_ok": ("Prepare OK", "ok"),
        "smoke_ok": ("Smoke OK", "ok"),
        "smoke_failed": ("Failed", "error"),
        "dependency_missing": ("Dependency Missing", "error"),
        "mini_pilot_ok": ("Mini Pilot OK", "ok"),
        "disabled": ("Disabled", "error"),
    }
    text, klass = labels.get(status, (status, "warning"))
    return {"status": status, "text": text, "class": klass}


def select_recipe_for_profile(profile_id: str, recipe_id: str | None = None) -> Any:
    if recipe_id:
        recipe = fetch_one("SELECT * FROM training_recipes_v2 WHERE id = ? AND optimizer_profile_id = ? AND is_active = 1", (recipe_id, profile_id))
        if recipe:
            return recipe
    recipe = fetch_one(
        """
        SELECT * FROM training_recipes_v2
        WHERE optimizer_profile_id = ? AND is_active = 1
        ORDER BY sort_order, display_name
        LIMIT 1
        """,
        (profile_id,),
    )
    if recipe is None:
        raise ValueError(f"Profile {profile_id} に対応するRecipeがありません。")
    return recipe


def preset_for_model_family(model_family: str) -> str:
    preset = fetch_one("SELECT id FROM presets WHERE model_family = ? ORDER BY id LIMIT 1", (model_family,))
    if preset:
        return preset["id"]
    fallback = fetch_one("SELECT id FROM presets ORDER BY id LIMIT 1")
    if fallback:
        return fallback["id"]
    raise RuntimeError("Presetが登録されていません。")


def smoke_params(recipe: Any) -> dict[str, Any]:
    params = json.loads(recipe["params_json"] or "{}")
    profile = fetch_one("SELECT smoke_params_json FROM optimizer_profiles_v2 WHERE id = ?", (recipe["optimizer_profile_id"],))
    if profile and profile["smoke_params_json"]:
        params.update(json.loads(profile["smoke_params_json"] or "{}"))
    params.update(
        {
            "max_train_steps": 2,
            "max_train_epochs": 1,
            "repeats": 1,
            "train_batch_size": 1,
            "network_dim": 4,
            "network_alpha": 2,
            "save_every_n_steps": 1,
            "sample_every_n_steps": 1,
            "sample_at_first": True,
            "generate_training_samples": True,
            "resolution": [512, 512],
        }
    )
    return params


def create_profile_test_job(
    optimizer_profile_id: str,
    *,
    recipe_id: str | None,
    dataset_id: int,
    base_model_path: str,
    test_type: str,
) -> tuple[int, Any]:
    profile = fetch_one("SELECT * FROM optimizer_profiles_v2 WHERE id = ?", (optimizer_profile_id,))
    if profile is None:
        raise ValueError(f"Optimizer Profile not found: {optimizer_profile_id}")
    recipe = select_recipe_for_profile(optimizer_profile_id, recipe_id)
    params = smoke_params(recipe)
    job_id = create_job(
        {
            "project_id": None,
            "name": f"optimizer_{test_type}_{optimizer_profile_id}",
            "dataset_id": dataset_id,
            "preset_id": preset_for_model_family(recipe["model_family"]),
            "recipe_v2_id": recipe["id"],
            "base_model_path": base_model_path,
            "output_name": f"optimizer_{test_type}_{optimizer_profile_id}",
            "params": params,
            "memo": "Optimizer Profile Validation用の一時Jobです。品質評価ではなく起動確認用です。",
            "post_training_review_mode": "manual",
            "max_auto_images": 0,
            "max_auto_runtime_minutes": 0,
        }
    )
    return job_id, recipe


def record_profile_test_result(
    optimizer_profile_id: str,
    *,
    recipe_id: str | None,
    test_type: str,
    status: str,
    test_job_id: int | None = None,
    command_path: str | None = None,
    log_path: str | None = None,
    return_code: int | None = None,
    elapsed_seconds: int | None = None,
    error_message: str | None = None,
    memo: str | None = None,
) -> int:
    now = utc_now()
    environment = latest_environment()
    profile = fetch_one("SELECT * FROM optimizer_profiles_v2 WHERE id = ?", (optimizer_profile_id,))
    optimizer_definition_id = profile["optimizer_definition_id"] if profile and "optimizer_definition_id" in profile.keys() else None
    validated_optimizer_type = profile["sd_scripts_optimizer_type"] if profile and "sd_scripts_optimizer_type" in profile.keys() else None
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO optimizer_profile_test_results(
                optimizer_definition_id, optimizer_profile_id, recipe_id, test_type, status, test_job_id,
                command_path, log_path, return_code, elapsed_seconds, error_message,
                sd_scripts_commit, torch_version, cuda_version, created_at, memo
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                optimizer_definition_id,
                optimizer_profile_id,
                recipe_id,
                test_type,
                status,
                test_job_id,
                command_path,
                log_path,
                return_code,
                elapsed_seconds,
                error_message,
                environment["sd_scripts_commit_hash"] if environment and "sd_scripts_commit_hash" in environment.keys() else None,
                environment["torch_version"] if environment and "torch_version" in environment.keys() else None,
                environment["torch_cuda_version"] if environment and "torch_cuda_version" in environment.keys() else None,
                now,
                memo,
            ),
        )
        result_id = int(cur.lastrowid)
        profile_status = {
            ("prepare", "ok"): "prepare_ok",
            ("smoke", "ok"): "smoke_ok",
            ("smoke", "failed"): "smoke_failed",
            ("mini_pilot", "ok"): "mini_pilot_ok",
            ("mini_pilot", "skipped"): None,
        }.get((test_type, status), "smoke_failed" if status == "failed" else "untested")
        conn.execute(
            """
            UPDATE optimizer_profiles_v2
            SET validation_status = COALESCE(?, validation_status),
                last_tested_at = ?, last_test_result_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (profile_status, now, result_id, now, optimizer_profile_id),
        )
        if optimizer_definition_id:
            conn.execute(
                """
                UPDATE optimizer_definitions_v2
                SET validation_status = COALESCE(?, validation_status),
                    validated_optimizer_type = COALESCE(?, validated_optimizer_type),
                    last_tested_at = ?, last_test_result_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (profile_status, validated_optimizer_type, now, result_id, now, optimizer_definition_id),
            )
    return result_id


def run_prepare_test(optimizer_profile_id: str, *, recipe_id: str | None, dataset_id: int, base_model_path: str) -> dict[str, Any]:
    job_id = None
    recipe = None
    try:
        job_id, recipe = create_profile_test_job(
            optimizer_profile_id,
            recipe_id=recipe_id,
            dataset_id=dataset_id,
            base_model_path=base_model_path,
            test_type="prepare",
        )
        job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
        dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        if job is None or dataset is None:
            raise RuntimeError("Prepare Test JobまたはDatasetを取得できません。")
        files = prepare_job_files(dict(job), dict(dataset))
        now = utc_now()
        with connect() as conn:
            conn.execute(
                "UPDATE training_jobs SET status = 'prepared', command_argv_json = ?, prompt_file_path = ?, updated_at = ? WHERE id = ?",
                (str(files["command_argv"]), str(files["sample_prompts"]), now, job_id),
            )
        result_id = record_profile_test_result(
            optimizer_profile_id,
            recipe_id=recipe["id"],
            test_type="prepare",
            status="ok",
            test_job_id=job_id,
            command_path=str(files["command_argv"]),
            log_path=str(Path(job["run_dir"]) / "logs" / "train.log"),
        )
        return {"ok": True, "result_id": result_id, "job_id": job_id, "recipe_id": recipe["id"]}
    except Exception as exc:
        result_id = record_profile_test_result(
            optimizer_profile_id,
            recipe_id=recipe["id"] if recipe else recipe_id,
            test_type="prepare",
            status="failed",
            test_job_id=job_id,
            error_message=str(exc),
        )
        return {"ok": False, "result_id": result_id, "job_id": job_id, "error": str(exc)}


def run_smoke_test(optimizer_profile_id: str, *, recipe_id: str | None, dataset_id: int, base_model_path: str) -> dict[str, Any]:
    prepared = run_prepare_test(optimizer_profile_id, recipe_id=recipe_id, dataset_id=dataset_id, base_model_path=base_model_path)
    if not prepared["ok"]:
        return prepared
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (prepared["job_id"],))
    if job is None:
        raise RuntimeError("Smoke Test Jobを取得できません。")
    argv_path = Path(job["run_dir"]) / "config" / "command_argv.json"
    argv = json.loads(argv_path.read_text(encoding="utf-8"))
    log_path = Path(job["run_dir"]) / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = latest_environment()
    if environment is None:
        raise RuntimeError("sd-scripts environment is not registered.")
    started = time.monotonic()
    return_code: int | None = None
    error_message: str | None = None
    with connect() as conn:
        now = utc_now()
        conn.execute("UPDATE training_jobs SET status = 'running', start_time = ?, updated_at = ? WHERE id = ?", (now, now, job["id"]))
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            argv,
            cwd=str(Path(environment["sd_scripts_path"])),
            stdout=handle,
            stderr=subprocess.STDOUT,
            shell=False,
            env=sd_scripts_subprocess_env(),
            check=False,
        )
        return_code = int(completed.returncode)
    elapsed = int(time.monotonic() - started)
    status = "ok" if return_code == 0 else "failed"
    if return_code != 0:
        error_message = f"Smoke Test failed with return_code={return_code}"
    end = utc_now()
    with connect() as conn:
        conn.execute(
            "UPDATE training_jobs SET status = ?, end_time = ?, elapsed_seconds = ?, return_code = ?, process_id = NULL, updated_at = ? WHERE id = ?",
            ("completed" if return_code == 0 else "failed", end, elapsed, return_code, end, job["id"]),
        )
    result_id = record_profile_test_result(
        optimizer_profile_id,
        recipe_id=prepared["recipe_id"],
        test_type="smoke",
        status=status,
        test_job_id=int(job["id"]),
        command_path=str(argv_path),
        log_path=str(log_path),
        return_code=return_code,
        elapsed_seconds=elapsed,
        error_message=error_message,
    )
    return {"ok": status == "ok", "result_id": result_id, "job_id": int(job["id"]), "return_code": return_code, "elapsed_seconds": elapsed}


def record_mini_pilot_skipped(optimizer_profile_id: str, *, recipe_id: str | None = None) -> dict[str, Any]:
    result_id = record_profile_test_result(
        optimizer_profile_id,
        recipe_id=recipe_id,
        test_type="mini_pilot",
        status="skipped",
        error_message="Mini PilotはPhase 12.3では基盤のみです。必要時に明示実行してください。",
    )
    return {"ok": True, "result_id": result_id}
