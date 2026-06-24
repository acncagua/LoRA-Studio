from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from app import settings
from app.db import connect, fetch_all, fetch_one, latest_environment, utc_now
from app.services.optimizer_profile_validation import (
    SMOKE_TARGET_PROFILES,
    record_profile_test_result,
    run_prepare_test,
    run_smoke_test,
    select_recipe_for_profile,
)
from app.services.output_collector import collect_job_results, safe_sha256_file
from app.services.training_runner import sd_scripts_subprocess_env
from app.services.validation_generation import common_gen_img_args


MASTER_CHECK_PROFILES = [
    "adamw8bit_sdxl_balanced",
    "paged_adamw8bit_sdxl_balanced",
    "prodigy_sdxl_soft",
    "adafactor_sdxl_auto",
    "adafactor_sdxl_fixed",
    "lion_sdxl_soft",
    "lion_sdxl_balanced_experimental",
    "dadaptadam_sdxl_auto",
    "dadaptlion_sdxl_auto",
]

RUN_STATUSES = {"planned", "running", "completed", "failed", "stopped"}
ITEM_STATUSES = {
    "planned",
    "prepare_ok",
    "smoke_ok",
    "smoke_failed",
    "image_smoke_ok",
    "image_smoke_warning",
    "failed",
    "skipped",
}


def classify_failure(message: str | None) -> str:
    text = (message or "").lower()
    if not text:
        return "unknown"
    if "modulenotfounderror" in text or "no module named" in text or "dependency" in text:
        return "missing_dependency"
    if "optimizer_type" in text and ("not found" in text or "unknown" in text or "unsupported" in text):
        return "sd_scripts_unsupported"
    if "unexpected keyword" in text or "invalid optimizer" in text or "optimizer_args" in text:
        return "master_parameter"
    if "learning_rate" in text and ("none" in text or "null" in text):
        return "lora_studio_logic"
    if "sqlite" in text or "database is locked" in text or "cuda out of memory" in text or "oom" in text:
        return "environment"
    if "command_argv" in text or "dataset_config" in text or "sample_prompts" in text:
        return "lora_studio_logic"
    if "nan" in text or "inf" in text:
        return "master_parameter"
    return "unknown"


def suggested_action(category: str, message: str | None = None) -> str:
    if category == "missing_dependency":
        return "Optimizer依存ライブラリをsd-scripts側venvへ導入してください。"
    if category == "master_parameter":
        return "optimizer_type / optimizer_args / scheduler / learning_rateのprofile値を見直してください。"
    if category == "sd_scripts_unsupported":
        return "sd-scriptsで受け付けるoptimizer名を確認し、validated_optimizer_typeを更新してください。"
    if category == "lora_studio_logic":
        return "LoRA-Studioのcommand生成またはnull LR処理を確認してください。"
    if category == "environment":
        return "GPUメモリ、SQLite lock、path、OneDrive同期など実行環境を確認してください。"
    return "ログを確認し、依存・パラメータ・環境の順に切り分けてください。"


def profile_targets(scope: str = "all_builtin", selected_profile_ids: list[str] | None = None) -> list[Any]:
    if scope == "selected_profiles" and selected_profile_ids:
        placeholders = ",".join("?" for _ in selected_profile_ids)
        return fetch_all(
            f"""
            SELECT p.*, od.display_name AS optimizer_display_name
            FROM optimizer_profiles_v2 p
            LEFT JOIN optimizer_definitions_v2 od ON od.id = p.optimizer_definition_id
            WHERE p.id IN ({placeholders}) AND p.is_active = 1
            ORDER BY od.smoke_test_priority, p.id
            """,
            tuple(selected_profile_ids),
        )
    if scope == "single_profile" and selected_profile_ids:
        return profile_targets("selected_profiles", selected_profile_ids[:1])
    return fetch_all(
        """
        SELECT p.*, od.display_name AS optimizer_display_name
        FROM optimizer_profiles_v2 p
        LEFT JOIN optimizer_definitions_v2 od ON od.id = p.optimizer_definition_id
        WHERE p.is_active = 1
          AND p.id IN ({})
        ORDER BY od.smoke_test_priority, p.id
        """.format(",".join("?" for _ in MASTER_CHECK_PROFILES)),
        tuple(MASTER_CHECK_PROFILES),
    )


def create_master_check_run(scope: str = "all_builtin", selected_profile_ids: list[str] | None = None) -> int:
    now = utc_now()
    targets = profile_targets(scope, selected_profile_ids)
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO optimizer_master_check_runs(
                status, target_scope, total_count, created_at
            ) VALUES ('planned', ?, ?, ?)
            """,
            (scope, len(targets), now),
        )
        run_id = int(cur.lastrowid)
        for profile in targets:
            try:
                recipe = select_recipe_for_profile(profile["id"])
                recipe_id = recipe["id"]
            except Exception:
                recipe_id = None
            conn.execute(
                """
                INSERT INTO optimizer_master_check_items(
                    check_run_id, optimizer_definition_id, optimizer_profile_id, recipe_id,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'planned', ?, ?)
                """,
                (run_id, profile["optimizer_definition_id"], profile["id"], recipe_id, now, now),
            )
    write_master_check_report(run_id)
    return run_id


def latest_master_check_run() -> Any | None:
    return fetch_one("SELECT * FROM optimizer_master_check_runs ORDER BY id DESC LIMIT 1")


def master_check_run_detail(run_id: int) -> dict[str, Any]:
    run = fetch_one("SELECT * FROM optimizer_master_check_runs WHERE id = ?", (run_id,))
    if run is None:
        raise ValueError(f"Optimizer Master Check Run not found: {run_id}")
    items = fetch_all(
        """
        SELECT i.*, p.display_name AS profile_display_name, p.validation_status AS profile_validation_status,
               od.display_name AS optimizer_display_name, r.short_label AS recipe_short_label,
               r.display_name AS recipe_display_name
        FROM optimizer_master_check_items i
        LEFT JOIN optimizer_profiles_v2 p ON p.id = i.optimizer_profile_id
        LEFT JOIN optimizer_definitions_v2 od ON od.id = i.optimizer_definition_id
        LEFT JOIN training_recipes_v2 r ON r.id = i.recipe_id
        WHERE i.check_run_id = ?
        ORDER BY i.id
        """,
        (run_id,),
    )
    return {"run": run, "items": items}


def update_run_counts(run_id: int) -> None:
    rows = fetch_all("SELECT status, failure_category, prepare_job_id FROM optimizer_master_check_items WHERE check_run_id = ?", (run_id,))
    prepare_ok = sum(
        1
        for row in rows
        if row["prepare_job_id"] or row["status"] in {"prepare_ok", "smoke_ok", "image_smoke_ok", "image_smoke_warning"}
    )
    smoke_ok = sum(1 for row in rows if row["status"] in {"smoke_ok", "image_smoke_ok", "image_smoke_warning"})
    failed = sum(1 for row in rows if row["status"] in {"failed", "smoke_failed"})
    dependency_missing = sum(1 for row in rows if row["failure_category"] == "missing_dependency")
    with connect() as conn:
        conn.execute(
            """
            UPDATE optimizer_master_check_runs
            SET prepare_ok_count = ?, smoke_ok_count = ?, failed_count = ?,
                dependency_missing_count = ?
            WHERE id = ?
            """,
            (prepare_ok, smoke_ok, failed, dependency_missing, run_id),
        )


def mark_profile_dependency_missing(profile_id: str, result_id: int | None = None) -> None:
    profile = fetch_one("SELECT optimizer_definition_id FROM optimizer_profiles_v2 WHERE id = ?", (profile_id,))
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE optimizer_profiles_v2
            SET validation_status = 'dependency_missing',
                last_tested_at = ?, last_test_result_id = COALESCE(?, last_test_result_id),
                updated_at = ?
            WHERE id = ?
            """,
            (now, result_id, now, profile_id),
        )
        if profile:
            conn.execute(
                """
                UPDATE optimizer_definitions_v2
                SET validation_status = 'dependency_missing',
                    last_tested_at = ?, last_test_result_id = COALESCE(?, last_test_result_id),
                    updated_at = ?
                WHERE id = ?
                """,
                (now, result_id, now, profile["optimizer_definition_id"]),
            )


def _mark_run_started(run_id: int) -> float:
    started = time.monotonic()
    now = utc_now()
    with connect() as conn:
        conn.execute("UPDATE optimizer_master_check_runs SET status = 'running', started_at = COALESCE(started_at, ?) WHERE id = ?", (now, run_id))
    return started


def _mark_run_finished(run_id: int, started: float, status: str = "completed") -> None:
    ended = utc_now()
    elapsed = int(time.monotonic() - started)
    write_master_check_report(run_id)
    run = fetch_one("SELECT report_path FROM optimizer_master_check_runs WHERE id = ?", (run_id,))
    with connect() as conn:
        conn.execute(
            """
            UPDATE optimizer_master_check_runs
            SET status = ?, ended_at = ?, elapsed_seconds = ?, report_path = COALESCE(report_path, ?)
            WHERE id = ?
            """,
            (status, ended, elapsed, run["report_path"] if run else None, run_id),
        )


def run_prepare_checks(run_id: int, *, dataset_id: int, base_model_path: str) -> dict[str, Any]:
    started = _mark_run_started(run_id)
    items = fetch_all("SELECT * FROM optimizer_master_check_items WHERE check_run_id = ? ORDER BY id", (run_id,))
    for item in items:
        if item["status"] not in {"planned", "failed", "skipped"}:
            continue
        result = run_prepare_test(
            item["optimizer_profile_id"],
            recipe_id=item["recipe_id"],
            dataset_id=dataset_id,
            base_model_path=base_model_path,
        )
        now = utc_now()
        if result["ok"]:
            status = "prepare_ok"
            failure_category = None
            error_message = None
        else:
            status = "failed"
            error_message = result.get("error") or "Prepare Check failed"
            failure_category = classify_failure(error_message)
        with connect() as conn:
            conn.execute(
                """
                UPDATE optimizer_master_check_items
                SET status = ?, prepare_job_id = ?, failure_category = ?,
                    error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, result.get("job_id"), failure_category, error_message, now, item["id"]),
            )
    update_run_counts(run_id)
    _mark_run_finished(run_id, started)
    return {"ok": True, "run_id": run_id}


def run_smoke_checks(run_id: int, *, dataset_id: int, base_model_path: str, profile_limit: int | None = None) -> dict[str, Any]:
    started = _mark_run_started(run_id)
    items = fetch_all("SELECT * FROM optimizer_master_check_items WHERE check_run_id = ? ORDER BY id", (run_id,))
    executed = 0
    for item in items:
        if profile_limit is not None and executed >= profile_limit:
            break
        if item["status"] in {"smoke_ok", "image_smoke_ok", "image_smoke_warning"}:
            continue
        result = run_smoke_test(
            item["optimizer_profile_id"],
            recipe_id=item["recipe_id"],
            dataset_id=dataset_id,
            base_model_path=base_model_path,
        )
        executed += 1
        now = utc_now()
        if result["ok"]:
            status = "smoke_ok"
            failure_category = None
            error_message = None
            artifact = latest_lora_artifact_for_job(int(result["job_id"]))
            artifact_status = artifact["status"]
        else:
            status = "smoke_failed"
            log_text = ""
            if result.get("job_id"):
                job = fetch_one("SELECT run_dir FROM training_jobs WHERE id = ?", (int(result["job_id"]),))
                log_path = Path(job["run_dir"]) / "logs" / "train.log" if job else None
                if log_path and log_path.exists():
                    log_text = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            error_message = result.get("error") or f"Smoke failed rc={result.get('return_code')}"
            failure_category = classify_failure(log_text + "\n" + error_message)
            if failure_category == "missing_dependency":
                mark_profile_dependency_missing(item["optimizer_profile_id"], result.get("result_id"))
            artifact = {}
            artifact_status = None
        with connect() as conn:
            conn.execute(
                """
                UPDATE optimizer_master_check_items
                SET status = ?, smoke_job_id = ?, output_lora_path = ?,
                    output_lora_sha256 = ?, output_lora_file_size = ?,
                    safetensors_check_status = ?, failure_category = ?,
                    error_message = ?, log_path = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    result.get("job_id"),
                    artifact.get("path"),
                    artifact.get("sha256"),
                    artifact.get("file_size"),
                    artifact_status,
                    failure_category,
                    error_message,
                    artifact.get("log_path"),
                    now,
                    item["id"],
                ),
            )
    update_run_counts(run_id)
    _mark_run_finished(run_id, started)
    return {"ok": True, "run_id": run_id, "executed": executed}


def latest_lora_artifact_for_job(job_id: int) -> dict[str, Any]:
    collect_job_results(job_id)
    output = fetch_one(
        """
        SELECT * FROM training_outputs
        WHERE job_id = ? AND file_type = 'model'
        ORDER BY id DESC LIMIT 1
        """,
        (job_id,),
    )
    if output is None:
        job = fetch_one("SELECT run_dir FROM training_jobs WHERE id = ?", (job_id,))
        log_path = str(Path(job["run_dir"]) / "logs" / "train.log") if job else None
        return {"status": "missing", "log_path": log_path}
    check = check_lora_artifact(output["file_path"])
    return {
        "path": output["file_path"],
        "sha256": check.get("sha256") or output["sha256"],
        "file_size": check.get("file_size") or output["file_size"],
        "status": check["status"],
        "log_path": str(Path(fetch_one("SELECT run_dir FROM training_jobs WHERE id = ?", (job_id,))["run_dir"]) / "logs" / "train.log"),
    }


def check_lora_artifact(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"status": "missing", "error": "LoRA path is empty"}
    path = Path(path_value)
    if not path.exists():
        return {"status": "missing", "error": f"file not found: {path}"}
    size = path.stat().st_size
    sha256, hash_error = safe_sha256_file(path)
    if size <= 0:
        return {"status": "failed", "file_size": size, "sha256": sha256, "error": "file size is zero"}
    result = {"status": "ok", "file_size": size, "sha256": sha256, "error": hash_error}
    if path.suffix.lower() != ".safetensors":
        result["status"] = "warning"
        result["error"] = "not a safetensors file"
        return result
    try:
        from safetensors import safe_open

        tensor_count = 0
        expected_prefix = False
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor_count += 1
                if key.startswith(("lora_", "network_", "diffusion_model.", "text_model.")):
                    expected_prefix = True
                tensor = handle.get_tensor(key)
                if bool(tensor.isnan().any()) or bool(tensor.isinf().any()):
                    return {**result, "status": "failed", "tensor_count": tensor_count, "error": f"NaN/Inf tensor: {key}"}
                try:
                    if float(tensor.abs().max()) == 0.0:
                        result["status"] = "warning"
                        result["error"] = "some tensors are all zero"
                except Exception:
                    pass
        if tensor_count <= 0:
            return {**result, "status": "failed", "tensor_count": 0, "error": "no tensors"}
        if not expected_prefix:
            result["status"] = "warning"
            result["error"] = "expected key prefix not found"
        result["tensor_count"] = tensor_count
        return result
    except ModuleNotFoundError:
        external_check = _check_safetensors_with_environment(path)
        if external_check:
            return {**result, **external_check}
        return {**result, "status": "warning", "error": "safetensors package is not available"}
    except Exception as exc:
        return {**result, "status": "failed", "error": f"safetensors read failed: {exc}"}


def _check_safetensors_with_environment(path: Path) -> dict[str, Any] | None:
    environment = latest_environment()
    if environment is None or not environment["venv_python_path"]:
        return None
    python_path = Path(environment["venv_python_path"])
    if not python_path.exists():
        return None
    code = r"""
import json
import sys

result = {"status": "ok"}
try:
    from safetensors import safe_open
    tensor_count = 0
    expected_prefix = False
    failed = False
    with safe_open(sys.argv[1], framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_count += 1
            if key.startswith(("lora_", "network_", "diffusion_model.", "text_model.")):
                expected_prefix = True
            tensor = handle.get_tensor(key)
            if bool(tensor.isnan().any()) or bool(tensor.isinf().any()):
                result = {"status": "failed", "tensor_count": tensor_count, "error": f"NaN/Inf tensor: {key}"}
                failed = True
                break
            try:
                if float(tensor.abs().max()) == 0.0 and result.get("status") == "ok":
                    result = {"status": "warning", "error": "some tensors are all zero"}
            except Exception:
                pass
    if not failed:
        if tensor_count <= 0:
            result = {"status": "failed", "tensor_count": 0, "error": "no tensors"}
        elif not expected_prefix:
            result = {"status": "warning", "tensor_count": tensor_count, "error": "expected key prefix not found"}
        else:
            result["tensor_count"] = tensor_count
except Exception as exc:
    result = {"status": "failed", "error": f"safetensors read failed: {exc}"}
print(json.dumps(result, ensure_ascii=False))
"""
    completed = subprocess.run(
        [str(python_path), "-c", code, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        return {"status": "warning", "error": "safetensors package is not available"}
    try:
        return json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {"status": "failed", "error": completed.stderr.strip() or completed.stdout.strip()}


def check_image_smoke(weight0_path: str | None, weight1_path: str | None) -> dict[str, Any]:
    checks = [check_png_image(weight0_path), check_png_image(weight1_path)]
    if any(item["status"] == "failed" for item in checks):
        return {"status": "failed", "difference_score": None, "checks": checks}
    difference = image_difference_score(weight0_path, weight1_path)
    status = "image_smoke_ok" if all(item["status"] == "ok" for item in checks) else "image_smoke_warning"
    return {"status": status, "difference_score": difference, "checks": checks}


def check_png_image(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"status": "failed", "error": "image path is empty"}
    path = Path(path_value)
    if not path.exists():
        return {"status": "failed", "error": f"image not found: {path}"}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            stat = ImageStat.Stat(rgb)
            variance = sum(stat.var) / len(stat.var)
            extrema = rgb.getextrema()
            flat = all(high - low < 3 for low, high in extrema)
            if variance < 5.0 or flat:
                return {"status": "failed", "width": rgb.width, "height": rgb.height, "variance": variance, "error": "image is too flat"}
            return {"status": "ok", "width": rgb.width, "height": rgb.height, "variance": variance}
    except Exception as exc:
        return {"status": "failed", "error": f"image read failed: {exc}"}


def image_difference_score(left_path: str | None, right_path: str | None) -> float | None:
    if not left_path or not right_path:
        return None
    try:
        with Image.open(left_path) as left, Image.open(right_path) as right:
            left = left.convert("RGB").resize((64, 64))
            right = right.convert("RGB").resize((64, 64))
            left_hist = left.histogram()
            right_hist = right.histogram()
            diff = sum(abs(a - b) for a, b in zip(left_hist, right_hist))
            return round(diff / max(1, sum(left_hist)), 6)
    except Exception:
        return None


def record_image_smoke_result(item_id: int, weight0_path: str, weight1_path: str) -> dict[str, Any]:
    result = check_image_smoke(weight0_path, weight1_path)
    failure_category = None if result["status"] in {"image_smoke_ok", "image_smoke_warning"} else "environment"
    error_message = None
    if failure_category:
        error_message = json.dumps(result.get("checks", []), ensure_ascii=False)
    with connect() as conn:
        conn.execute(
            """
            UPDATE optimizer_master_check_items
            SET status = ?, generated_weight0_image_path = ?,
                generated_weight1_image_path = ?, image_smoke_status = ?,
                difference_score = ?, failure_category = COALESCE(?, failure_category),
                error_message = COALESCE(?, error_message), updated_at = ?
            WHERE id = ?
            """,
            (
                result["status"],
                weight0_path,
                weight1_path,
                result["status"],
                result.get("difference_score"),
                failure_category,
                error_message,
                utc_now(),
                item_id,
            ),
        )
    return result


def run_image_smoke_for_item(item_id: int, *, base_model_path: str | None = None) -> dict[str, Any]:
    item = fetch_one("SELECT * FROM optimizer_master_check_items WHERE id = ?", (item_id,))
    if item is None:
        raise ValueError(f"Optimizer Master Check item not found: {item_id}")
    if not item["output_lora_path"] or not Path(item["output_lora_path"]).exists():
        raise ValueError("Smoke OKのLoRA artifactが見つかりません。先に2-step Smokeを実行してください。")
    environment = latest_environment()
    if environment is None:
        raise RuntimeError("sd-scripts environment is not registered.")
    sd_scripts_path = Path(environment["sd_scripts_path"])
    venv_python = Path(environment["venv_python_path"])
    gen_img = sd_scripts_path / "gen_img.py"
    if not gen_img.exists():
        raise RuntimeError(f"gen_img.py が存在しません: {gen_img}")
    if not venv_python.exists():
        raise RuntimeError(f"venv python が存在しません: {venv_python}")
    smoke_dir = settings.RUNS_DIR / "optimizer_master_checks" / f"run_{int(item['check_run_id']):06d}" / f"item_{item_id:06d}"
    out_dir = smoke_dir / "images"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = "zho, 1girl, upper body, looking at viewer, simple background"
    negative = "low quality, worst quality, blurry, bad anatomy"
    baseline_prompt = smoke_dir / "weight0_prompt.txt"
    lora_prompt = smoke_dir / "weight1_prompt.txt"
    baseline_name = "weight0_baseline.png"
    lora_name = "weight1_lora.png"
    baseline_prompt.write_text(
        f"{prompt} --n {negative} --d 111111 --w 512 --h 512 --s 20 --l 7 --f {baseline_name}\n",
        encoding="utf-8",
    )
    lora_prompt.write_text(
        f"{prompt} --n {negative} --d 111111 --w 512 --h 512 --s 20 --l 7 --am 1 --f {lora_name}\n",
        encoding="utf-8",
    )
    condition = {"width": 512, "height": 512, "cfg_scale": 7, "steps": 20, "sampler": "euler_a"}
    model_path = Path(base_model_path) if base_model_path else None
    if not model_path or not model_path.exists():
        smoke_job = fetch_one("SELECT base_model_path FROM training_jobs WHERE id = ?", (item["smoke_job_id"],)) if item["smoke_job_id"] else None
        model_path = Path(smoke_job["base_model_path"]) if smoke_job and smoke_job["base_model_path"] else None
    if not model_path or not model_path.exists():
        raise RuntimeError("base model pathが見つかりません。")
    base_args = common_gen_img_args(
        venv_python=venv_python,
        gen_img=gen_img,
        base_model_path=model_path,
        out_dir=out_dir,
        model_family="SDXL",
        mixed_precision=str(environment["mixed_precision"] or "bf16"),
        condition=condition,
    )
    commands = [
        [*base_args, "--from_file", str(baseline_prompt)],
        [
            *base_args,
            "--from_file",
            str(lora_prompt),
            "--network_module",
            "networks.lora",
            "--network_weights",
            str(item["output_lora_path"]),
        ],
    ]
    log_path = smoke_dir / "image_smoke.log"
    started = time.monotonic()
    with log_path.open("wb") as handle:
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=str(sd_scripts_path),
                stdout=handle,
                stderr=subprocess.STDOUT,
                shell=False,
                env=sd_scripts_subprocess_env(),
                check=False,
            )
            if int(completed.returncode) != 0:
                error = f"Image Smoke failed with return_code={completed.returncode}"
                category = classify_failure(log_path.read_text(encoding="utf-8", errors="replace")[-4000:] + "\n" + error)
                with connect() as conn:
                    conn.execute(
                        """
                        UPDATE optimizer_master_check_items
                        SET image_smoke_status = 'failed', failure_category = ?,
                            error_message = ?, log_path = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (category, error, str(log_path), utc_now(), item_id),
                    )
                return {"ok": False, "status": "failed", "error": error, "log_path": str(log_path)}
    weight0_path = out_dir / baseline_name
    weight1_path = out_dir / lora_name
    result = record_image_smoke_result(item_id, str(weight0_path), str(weight1_path))
    with connect() as conn:
        conn.execute("UPDATE optimizer_master_check_items SET log_path = ?, updated_at = ? WHERE id = ?", (str(log_path), utc_now(), item_id))
    result["elapsed_seconds"] = int(time.monotonic() - started)
    result["log_path"] = str(log_path)
    return result


def write_master_check_report(run_id: int) -> dict[str, str]:
    detail = master_check_run_detail(run_id)
    run = detail["run"]
    items = detail["items"]
    timestamp = (run["created_at"] or utc_now()).replace(":", "").replace("-", "").replace("+", "_").replace("T", "_")
    reports_dir = settings.ROOT_DIR / "reports" / "optimizer_master_checks"
    logs_dir = settings.ROOT_DIR / "logs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"optimizer_master_check_{timestamp}_run_{run_id}.md"
    json_path = logs_dir / f"optimizer_master_check_{timestamp}_run_{run_id}.json"
    payload = {
        "run": dict(run),
        "items": [dict(item) for item in items],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Optimizer Master Check #{run_id}",
        "",
        f"- status: {run['status']}",
        f"- target_scope: {run['target_scope']}",
        f"- total: {run['total_count']}",
        f"- prepare_ok: {run['prepare_ok_count']}",
        f"- smoke_ok: {run['smoke_ok_count']}",
        f"- failed: {run['failed_count']}",
        f"- dependency_missing: {run['dependency_missing_count']}",
        "",
        "| Profile | Recipe | Status | Artifact | Image Smoke | Failure | Suggested action |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in items:
        category = item["failure_category"] or ""
        lines.append(
            "| {profile} | {recipe} | {status} | {artifact} | {image} | {failure} | {action} |".format(
                profile=item["optimizer_profile_id"],
                recipe=item["recipe_short_label"] or item["recipe_id"] or "-",
                status=item["status"],
                artifact=item["safetensors_check_status"] or "-",
                image=item["image_smoke_status"] or "-",
                failure=category or "-",
                action=suggested_action(category, item["error_message"]) if category else "-",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with connect() as conn:
        conn.execute(
            "UPDATE optimizer_master_check_runs SET report_path = ?, log_path = ? WHERE id = ?",
            (str(md_path), str(json_path), run_id),
        )
    return {"report_path": str(md_path), "log_path": str(json_path)}
