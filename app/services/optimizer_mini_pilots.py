from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from app import settings
from app.db import connect, create_job, fetch_all, fetch_one, latest_environment, utc_now
from app.services.command_builder import prepare_job_files
from app.services.optimizer_master_checks import check_lora_artifact, classify_failure, suggested_action
from app.services.optimizer_profile_validation import (
    preset_for_model_family,
    record_profile_test_result,
    select_recipe_for_profile,
)
from app.services.output_collector import collect_job_results
from app.services.training_runner import sd_scripts_subprocess_env


MINI_PILOT_PROFILES = [
    "adamw8bit_sdxl_balanced",
    "paged_adamw8bit_sdxl_balanced",
    "prodigy_sdxl_soft",
    "adafactor_sdxl_fixed",
    "adafactor_sdxl_auto",
    "lion_sdxl_soft",
    "lion_sdxl_balanced_experimental",
    "dadaptadam_sdxl_auto",
    "dadaptlion_sdxl_auto",
]

ACCEPTANCE_MINI_PILOT_PROFILES = [
    "adamw8bit_sdxl_balanced",
    "prodigy_sdxl_soft",
    "dadaptlion_sdxl_auto",
    "adafactor_sdxl_auto",
    "lion_sdxl_soft",
]


def mini_pilot_targets(scope: str = "all_smoke_ok", selected_profile_ids: list[str] | None = None) -> list[Any]:
    if scope in {"selected_profiles", "single_profile"} and selected_profile_ids:
        ids = selected_profile_ids[:1] if scope == "single_profile" else selected_profile_ids
    elif scope == "acceptance_minimum":
        ids = ACCEPTANCE_MINI_PILOT_PROFILES
    else:
        ids = MINI_PILOT_PROFILES
    placeholders = ",".join("?" for _ in ids)
    return fetch_all(
        f"""
        SELECT p.*, od.display_name AS optimizer_display_name
        FROM optimizer_profiles_v2 p
        LEFT JOIN optimizer_definitions_v2 od ON od.id = p.optimizer_definition_id
        WHERE p.id IN ({placeholders}) AND p.is_active = 1
        ORDER BY od.smoke_test_priority, p.id
        """,
        tuple(ids),
    )


def create_mini_pilot_run(
    scope: str = "all_smoke_ok",
    selected_profile_ids: list[str] | None = None,
    *,
    dataset_id: int | None = None,
    base_model_path: str | None = None,
    source_job_id: int | None = None,
    provider: str = "default",
    steps: int = 300,
    fast_mode: bool = False,
) -> int:
    now = utc_now()
    targets = mini_pilot_targets(scope, selected_profile_ids)
    source_job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (source_job_id,)) if source_job_id else None
    if dataset_id is None and source_job:
        dataset_id = int(source_job["dataset_id"])
    if base_model_path is None and source_job:
        base_model_path = source_job["base_model_path"]
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO optimizer_mini_pilot_runs(
                status, target_scope, dataset_id, base_model_path, source_job_id,
                provider, total_count, created_at, updated_at
            ) VALUES ('planned', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (scope, dataset_id, base_model_path, source_job_id, provider, len(targets), now, now),
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
                INSERT INTO optimizer_mini_pilot_items(
                    mini_pilot_run_id, optimizer_definition_id, optimizer_profile_id,
                    recipe_id, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'planned', ?, ?)
                """,
                (run_id, profile["optimizer_definition_id"], profile["id"], recipe_id, now, now),
            )
    write_mini_pilot_report(run_id, extra={"steps": steps, "fast_mode": fast_mode})
    return run_id


def latest_mini_pilot_run() -> Any | None:
    return fetch_one("SELECT * FROM optimizer_mini_pilot_runs ORDER BY id DESC LIMIT 1")


def mini_pilot_run_detail(run_id: int) -> dict[str, Any]:
    run = fetch_one("SELECT * FROM optimizer_mini_pilot_runs WHERE id = ?", (run_id,))
    if run is None:
        raise ValueError(f"Optimizer Mini Pilot Run not found: {run_id}")
    items = fetch_all(
        """
        SELECT i.*, p.display_name AS profile_display_name,
               p.mini_pilot_status AS profile_mini_pilot_status,
               p.validation_status AS profile_validation_status,
               od.display_name AS optimizer_display_name,
               r.short_label AS recipe_short_label,
               r.display_name AS recipe_display_name
        FROM optimizer_mini_pilot_items i
        LEFT JOIN optimizer_profiles_v2 p ON p.id = i.optimizer_profile_id
        LEFT JOIN optimizer_definitions_v2 od ON od.id = i.optimizer_definition_id
        LEFT JOIN training_recipes_v2 r ON r.id = i.recipe_id
        WHERE i.mini_pilot_run_id = ?
        ORDER BY i.id
        """,
        (run_id,),
    )
    return {"run": run, "items": items}


def mini_pilot_params(recipe: Any, *, steps: int = 300, fast_mode: bool = False) -> dict[str, Any]:
    params = json.loads(recipe["params_json"] or "{}")
    params.update(
        {
            "max_train_steps": int(steps),
            "max_train_epochs": 1,
            "save_every_n_steps": max(1, min(100, int(steps))),
            "sample_every_n_steps": max(1, min(100, int(steps))),
            "save_every_n_epochs": 1,
            "sample_every_n_epochs": 1,
            "sample_at_first": False,
            "generate_training_samples": True,
            "no_metadata": True,
            "save_model_as": "safetensors",
            "train_batch_size": 1,
        }
    )
    if fast_mode:
        params.update(
            {
                "max_train_steps": min(100, int(steps)),
                "network_dim": min(16, int(params.get("network_dim") or 16)),
                "network_alpha": min(8, int(params.get("network_alpha") or 8)),
            }
        )
    return params


def create_mini_pilot_job(
    optimizer_profile_id: str,
    *,
    recipe_id: str | None,
    dataset_id: int,
    base_model_path: str,
    source_job_id: int | None = None,
    steps: int = 300,
    fast_mode: bool = False,
) -> tuple[int, Any]:
    recipe = select_recipe_for_profile(optimizer_profile_id, recipe_id)
    params = mini_pilot_params(recipe, steps=steps, fast_mode=fast_mode)
    job_id = create_job(
        {
            "project_id": None,
            "name": f"mini_pilot_{optimizer_profile_id}",
            "dataset_id": dataset_id,
            "preset_id": preset_for_model_family(recipe["model_family"]),
            "recipe_v2_id": recipe["id"],
            "base_model_path": base_model_path,
            "output_name": f"mini_pilot_{optimizer_profile_id}",
            "params": params,
            "parent_job_id": source_job_id,
            "memo": "Optimizer Practical Mini Pilot用の一時Jobです。品質保証ではなく実用候補の軽量確認です。",
            "post_training_review_mode": "manual",
            "max_auto_images": 0,
            "max_auto_runtime_minutes": 0,
        }
    )
    return job_id, recipe


def prepare_mini_pilot_job(job_id: int) -> dict[str, str]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise RuntimeError("Mini Pilot Jobを取得できません。")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
    if dataset is None:
        raise RuntimeError("Datasetを取得できません。")
    files = prepare_job_files(dict(job), dict(dataset))
    now = utc_now()
    with connect() as conn:
        conn.execute(
            "UPDATE training_jobs SET status = 'prepared', command_line = ?, command_argv_json = ?, prompt_file_path = ?, updated_at = ? WHERE id = ?",
            (files["command"], str(files["command_argv"]), str(files["sample_prompts"]), now, job_id),
        )
    return {key: str(value) for key, value in files.items()}


def run_mini_pilot_job(job_id: int) -> dict[str, Any]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise RuntimeError("Mini Pilot Jobを取得できません。")
    argv_path = Path(job["run_dir"]) / "config" / "command_argv.json"
    argv = json.loads(argv_path.read_text(encoding="utf-8"))
    environment = latest_environment()
    if environment is None:
        raise RuntimeError("sd-scripts environment is not registered.")
    log_path = Path(job["run_dir"]) / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    now = utc_now()
    with connect() as conn:
        conn.execute("UPDATE training_jobs SET status = 'running', start_time = ?, updated_at = ? WHERE id = ?", (now, now, job_id))
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
    elapsed = int(time.monotonic() - started)
    return_code = int(completed.returncode)
    end = utc_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET status = ?, end_time = ?, elapsed_seconds = ?, return_code = ?,
                process_id = NULL, updated_at = ?
            WHERE id = ?
            """,
            ("completed" if return_code == 0 else "failed", end, elapsed, return_code, end, job_id),
        )
    try:
        collect_job_results(job_id)
    except Exception:
        pass
    return {"return_code": return_code, "elapsed_seconds": elapsed, "log_path": str(log_path)}


LOSS_RE = re.compile(r"(?:loss|loss:)\s*[=:]?\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.IGNORECASE)


def loss_summary_from_log(log_path: str | None) -> dict[str, Any]:
    if not log_path or not Path(log_path).exists():
        return {"loss_status": "unknown"}
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    values: list[float] = []
    nan_count = len(re.findall(r"\bnan\b", text, flags=re.IGNORECASE))
    inf_count = len(re.findall(r"\binf\b|infinity", text, flags=re.IGNORECASE))
    for match in LOSS_RE.finditer(text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    if nan_count or inf_count:
        loss_status = "failed"
    elif values:
        loss_status = "ok"
    else:
        loss_status = "unknown"
    tail = values[-20:]
    return {
        "initial_loss": values[0] if values else None,
        "final_loss": values[-1] if values else None,
        "min_loss": min(values) if values else None,
        "max_loss": max(values) if values else None,
        "moving_avg_final_loss": round(sum(tail) / len(tail), 6) if tail else None,
        "nan_loss_count": nan_count,
        "inf_loss_count": inf_count,
        "loss_status": loss_status,
    }


def latest_artifact_for_job(job_id: int) -> dict[str, Any]:
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
        return {"artifact_status": "missing", "output_model_count": 0}
    check = check_lora_artifact(output["file_path"])
    count = fetch_one("SELECT COUNT(*) AS count FROM training_outputs WHERE job_id = ? AND file_type = 'model'", (job_id,))
    return {
        "output_lora_path": output["file_path"],
        "output_lora_sha256": check.get("sha256") or output["sha256"],
        "output_lora_file_size": check.get("file_size") or output["file_size"],
        "output_model_count": int(count["count"] if count else 1),
        "artifact_status": check["status"],
        "artifact_error": check.get("error"),
    }


def sample_image_summary(job_id: int) -> dict[str, int]:
    count = fetch_one("SELECT COUNT(*) AS count FROM sample_images WHERE job_id = ?", (job_id,))
    return {"sample_image_count": int(count["count"] if count else 0)}


def update_mini_pilot_counts(run_id: int) -> None:
    rows = fetch_all("SELECT status FROM optimizer_mini_pilot_items WHERE mini_pilot_run_id = ?", (run_id,))
    ok = sum(1 for row in rows if row["status"] == "mini_pilot_ok")
    warning = sum(1 for row in rows if row["status"] == "mini_pilot_warning")
    failed = sum(1 for row in rows if row["status"] == "failed")
    with connect() as conn:
        conn.execute(
            """
            UPDATE optimizer_mini_pilot_runs
            SET mini_pilot_ok_count = ?, warning_count = ?, failed_count = ?, updated_at = ?
            WHERE id = ?
            """,
            (ok, warning, failed, utc_now(), run_id),
        )


def run_mini_pilot_items(run_id: int, *, steps: int = 300, fast_mode: bool = False, profile_limit: int | None = None) -> dict[str, Any]:
    run = fetch_one("SELECT * FROM optimizer_mini_pilot_runs WHERE id = ?", (run_id,))
    if run is None:
        raise ValueError(f"Mini Pilot Run not found: {run_id}")
    if not run["dataset_id"] or not run["base_model_path"]:
        raise RuntimeError("Datasetとbase model pathを指定してから実行してください。")
    started = time.monotonic()
    now = utc_now()
    with connect() as conn:
        conn.execute("UPDATE optimizer_mini_pilot_runs SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ? WHERE id = ?", (now, now, run_id))
    items = fetch_all("SELECT * FROM optimizer_mini_pilot_items WHERE mini_pilot_run_id = ? ORDER BY id", (run_id,))
    executed = 0
    for item in items:
        if profile_limit is not None and executed >= profile_limit:
            break
        if item["status"] in {"mini_pilot_ok", "mini_pilot_warning"}:
            continue
        item_started = utc_now()
        with connect() as conn:
            conn.execute("UPDATE optimizer_mini_pilot_items SET status = 'running', updated_at = ? WHERE id = ?", (item_started, item["id"]))
        job_id: int | None = None
        recipe_id = item["recipe_id"]
        status = "failed"
        failure_category = None
        error_message = None
        log_path = None
        return_code: int | None = None
        elapsed = None
        try:
            job_id, recipe = create_mini_pilot_job(
                item["optimizer_profile_id"],
                recipe_id=recipe_id,
                dataset_id=int(run["dataset_id"]),
                base_model_path=str(run["base_model_path"]),
                source_job_id=int(run["source_job_id"]) if run["source_job_id"] else None,
                steps=steps,
                fast_mode=fast_mode,
            )
            recipe_id = recipe["id"]
            files = prepare_mini_pilot_job(job_id)
            result = run_mini_pilot_job(job_id)
            return_code = int(result["return_code"])
            elapsed = int(result["elapsed_seconds"])
            log_path = result["log_path"]
            losses = loss_summary_from_log(log_path)
            artifact = latest_artifact_for_job(job_id)
            samples = sample_image_summary(job_id)
            if return_code != 0:
                status = "failed"
                error_message = f"Mini Pilot failed with return_code={return_code}"
                log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")[-6000:] if log_path and Path(log_path).exists() else ""
                failure_category = classify_failure(log_text + "\n" + error_message)
            elif losses.get("loss_status") == "failed":
                status = "failed"
                error_message = "Mini Pilot loss contains NaN/Inf"
                failure_category = "optimizer_instability"
            elif artifact.get("artifact_status") == "failed":
                status = "failed"
                error_message = artifact.get("artifact_error") or "LoRA artifact check failed"
                failure_category = "optimizer_instability"
            elif artifact.get("artifact_status") in {"warning"} or losses.get("loss_status") in {"unknown", "warning"}:
                status = "mini_pilot_warning"
            else:
                status = "mini_pilot_ok"
            profile_status = "ok" if status == "mini_pilot_ok" else ("warning" if status == "mini_pilot_warning" else "failed")
            result_id = record_profile_test_result(
                item["optimizer_profile_id"],
                recipe_id=recipe_id,
                test_type="mini_pilot",
                status=profile_status,
                test_job_id=job_id,
                command_path=files.get("command_argv"),
                log_path=log_path,
                return_code=return_code,
                elapsed_seconds=elapsed,
                error_message=error_message,
                memo="Practical Mini Pilot: short training stability and artifact check.",
            )
            with connect() as conn:
                conn.execute(
                    """
                    UPDATE optimizer_mini_pilot_items
                    SET status = ?, test_job_id = ?, recipe_id = ?,
                        output_lora_path = ?, output_lora_sha256 = ?, output_lora_file_size = ?,
                        output_model_count = ?, sample_image_count = ?,
                        mini_validation_image_count = 0,
                        initial_loss = ?, final_loss = ?, min_loss = ?, max_loss = ?,
                        moving_avg_final_loss = ?, nan_loss_count = ?, inf_loss_count = ?,
                        loss_status = ?, artifact_status = ?, image_smoke_status = 'skipped',
                        machine_review_status = 'skipped',
                        failure_category = ?, error_message = ?, log_path = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        status,
                        job_id,
                        recipe_id,
                        artifact.get("output_lora_path"),
                        artifact.get("output_lora_sha256"),
                        artifact.get("output_lora_file_size"),
                        artifact.get("output_model_count", 0),
                        samples.get("sample_image_count", 0),
                        losses.get("initial_loss"),
                        losses.get("final_loss"),
                        losses.get("min_loss"),
                        losses.get("max_loss"),
                        losses.get("moving_avg_final_loss"),
                        losses.get("nan_loss_count", 0),
                        losses.get("inf_loss_count", 0),
                        losses.get("loss_status"),
                        artifact.get("artifact_status"),
                        failure_category,
                        error_message,
                        log_path,
                        utc_now(),
                        item["id"],
                    ),
                )
            _ = result_id
        except Exception as exc:
            error_message = str(exc)
            failure_category = classify_failure(error_message)
            if job_id:
                record_profile_test_result(
                    item["optimizer_profile_id"],
                    recipe_id=recipe_id,
                    test_type="mini_pilot",
                    status="failed",
                    test_job_id=job_id,
                    return_code=return_code,
                    elapsed_seconds=elapsed,
                    error_message=error_message,
                )
            with connect() as conn:
                conn.execute(
                    """
                    UPDATE optimizer_mini_pilot_items
                    SET status = 'failed', test_job_id = COALESCE(?, test_job_id),
                        failure_category = ?, error_message = ?, log_path = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (job_id, failure_category, error_message, log_path, utc_now(), item["id"]),
                )
        executed += 1
        update_mini_pilot_counts(run_id)
        write_mini_pilot_report(run_id, extra={"steps": steps, "fast_mode": fast_mode})
    update_mini_pilot_counts(run_id)
    detail = mini_pilot_run_detail(run_id)
    rows = detail["items"]
    final_status = "failed" if rows and all(row["status"] == "failed" for row in rows) else "completed"
    ended = utc_now()
    elapsed_total = int(time.monotonic() - started)
    paths = write_mini_pilot_report(run_id, extra={"steps": steps, "fast_mode": fast_mode})
    with connect() as conn:
        conn.execute(
            """
            UPDATE optimizer_mini_pilot_runs
            SET status = ?, ended_at = ?, elapsed_seconds = ?,
                report_path = ?, json_report_path = ?, log_path = ?, updated_at = ?
            WHERE id = ?
            """,
            (final_status, ended, elapsed_total, paths["report_path"], paths["json_report_path"], paths["json_report_path"], ended, run_id),
        )
    return {"ok": True, "run_id": run_id, "executed": executed}


def write_mini_pilot_report(run_id: int, *, extra: dict[str, Any] | None = None) -> dict[str, str]:
    detail = mini_pilot_run_detail(run_id)
    run = detail["run"]
    items = detail["items"]
    timestamp = (run["created_at"] or utc_now()).replace(":", "").replace("-", "").replace("+", "_").replace("T", "_")
    reports_dir = settings.ROOT_DIR / "reports" / "optimizer_mini_pilots"
    logs_dir = settings.ROOT_DIR / "logs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"optimizer_mini_pilot_{timestamp}_run_{run_id}.md"
    json_path = logs_dir / f"optimizer_mini_pilot_{timestamp}_run_{run_id}.json"
    payload = {"run": dict(run), "items": [dict(item) for item in items], "extra": extra or {}}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Optimizer Mini Pilot #{run_id}",
        "",
        f"- status: {run['status']}",
        f"- target_scope: {run['target_scope']}",
        f"- dataset_id: {run['dataset_id'] or '-'}",
        f"- base_model_path: {run['base_model_path'] or '-'}",
        f"- total: {run['total_count'] or 0}",
        f"- mini_pilot_ok: {run['mini_pilot_ok_count'] or 0}",
        f"- warning: {run['warning_count'] or 0}",
        f"- failed: {run['failed_count'] or 0}",
        "",
        "Mini Pilot OK is a practical short-run signal, not a final quality guarantee.",
        "",
        "| Profile | Recipe | Status | Job | Final loss | Artifact | Samples | Failure | Suggested action |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in items:
        category = item["failure_category"] or ""
        lines.append(
            "| {profile} | {recipe} | {status} | {job} | {loss} | {artifact} | {samples} | {failure} | {action} |".format(
                profile=item["optimizer_profile_id"],
                recipe=item["recipe_short_label"] or item["recipe_display_name"] or item["recipe_id"] or "-",
                status=item["status"],
                job=f"#{item['test_job_id']}" if item["test_job_id"] else "-",
                loss=item["final_loss"] if item["final_loss"] is not None else "-",
                artifact=item["artifact_status"] or "-",
                samples=item["sample_image_count"] or 0,
                failure=category or "-",
                action=suggested_action(category, item["error_message"]) if category else "-",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with connect() as conn:
        conn.execute(
            "UPDATE optimizer_mini_pilot_runs SET report_path = ?, json_report_path = ?, log_path = ?, updated_at = ? WHERE id = ?",
            (str(md_path), str(json_path), str(json_path), utc_now(), run_id),
        )
    return {"report_path": str(md_path), "json_report_path": str(json_path)}
