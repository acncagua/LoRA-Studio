from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.optimizer_master_checks import (
    classify_failure,
    create_master_check_run,
    latest_lora_artifact_for_job,
    mark_profile_dependency_missing,
    record_image_smoke_result,
    run_image_smoke_for_item,
    run_prepare_checks,
    suggested_action,
    update_run_counts,
    write_master_check_report,
)
from app.services.optimizer_profile_validation import run_smoke_test
from app.db import connect, fetch_one, utc_now


SMOKE_ORDER = [
    "adamw8bit_sdxl_balanced",
    "paged_adamw8bit_sdxl_balanced",
    "prodigy_sdxl_soft",
    "adafactor_sdxl_fixed",
    "lion_sdxl_soft",
    "dadaptadam_sdxl_auto",
    "adafactor_sdxl_auto",
    "dadaptlion_sdxl_auto",
    "lion_sdxl_balanced_experimental",
]


def log(message: str) -> None:
    print(message, flush=True)


def update_item_after_smoke(item: dict, result: dict) -> None:
    now = utc_now()
    if result.get("ok"):
        artifact = latest_lora_artifact_for_job(int(result["job_id"]))
        with connect() as conn:
            conn.execute(
                """
                UPDATE optimizer_master_check_items
                SET status = 'smoke_ok', smoke_job_id = ?, output_lora_path = ?,
                    output_lora_sha256 = ?, output_lora_file_size = ?,
                    safetensors_check_status = ?, failure_category = NULL,
                    error_message = NULL, log_path = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    result.get("job_id"),
                    artifact.get("path"),
                    artifact.get("sha256"),
                    artifact.get("file_size"),
                    artifact.get("status"),
                    artifact.get("log_path"),
                    now,
                    item["id"],
                ),
            )
        return

    log_text = ""
    log_path = None
    if result.get("job_id"):
        job = fetch_one("SELECT run_dir FROM training_jobs WHERE id = ?", (int(result["job_id"]),))
        if job:
            path = Path(job["run_dir"]) / "logs" / "train.log"
            log_path = str(path)
            if path.exists():
                log_text = path.read_text(encoding="utf-8", errors="replace")[-8000:]
    error_message = result.get("error") or f"Smoke failed rc={result.get('return_code')}"
    category = classify_failure(log_text + "\n" + error_message)
    if category == "missing_dependency":
        mark_profile_dependency_missing(item["optimizer_profile_id"], result.get("result_id"))
    with connect() as conn:
        conn.execute(
            """
            UPDATE optimizer_master_check_items
            SET status = 'smoke_failed', smoke_job_id = ?, failure_category = ?,
                error_message = ?, log_path = ?, updated_at = ?
            WHERE id = ?
            """,
            (result.get("job_id"), category, error_message, log_path, now, item["id"]),
        )


def mark_run_running(run_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE optimizer_master_check_runs SET status = 'running', ended_at = NULL, elapsed_seconds = NULL WHERE id = ?",
            (run_id,),
        )


def mark_run_completed(run_id: int) -> None:
    run = fetch_one("SELECT started_at FROM optimizer_master_check_runs WHERE id = ?", (run_id,))
    ended_at = utc_now()
    elapsed_seconds = None
    if run and run["started_at"]:
        from datetime import datetime

        started = datetime.fromisoformat(str(run["started_at"]))
        ended = datetime.fromisoformat(ended_at)
        elapsed_seconds = max(0, int((ended - started).total_seconds()))
    with connect() as conn:
        conn.execute(
            """
            UPDATE optimizer_master_check_runs
            SET status = 'completed', ended_at = ?, elapsed_seconds = ?
            WHERE id = ?
            """,
            (ended_at, elapsed_seconds, run_id),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--skip-image-smoke", action="store_true")
    args = parser.parse_args()

    log("[MasterCheck] create run")
    run_id = create_master_check_run("selected_profiles", SMOKE_ORDER)
    log(f"[MasterCheck] run_id={run_id}")

    log("[MasterCheck] Prepare Check all profiles")
    run_prepare_checks(run_id, dataset_id=args.dataset_id, base_model_path=args.base_model_path)
    mark_run_running(run_id)
    write_master_check_report(run_id)
    log("[MasterCheck] Prepare Check completed")

    for profile_id in SMOKE_ORDER:
        item_row = fetch_one(
            "SELECT * FROM optimizer_master_check_items WHERE check_run_id = ? AND optimizer_profile_id = ?",
            (run_id, profile_id),
        )
        if item_row is None:
            log(f"[MasterCheck] {profile_id}: item missing")
            continue
        item = dict(item_row)
        if item["status"] not in {"prepare_ok", "planned"}:
            log(f"[MasterCheck] {profile_id}: skip smoke, status={item['status']} category={item['failure_category']}")
            continue
        log(f"[MasterCheck] {profile_id}: 2-step Smoke start")
        try:
            result = run_smoke_test(
                profile_id,
                recipe_id=item["recipe_id"],
                dataset_id=args.dataset_id,
                base_model_path=args.base_model_path,
            )
        except Exception as exc:
            result = {"ok": False, "error": str(exc), "job_id": None, "return_code": None}
            traceback.print_exc()
        update_item_after_smoke(item, result)
        update_run_counts(run_id)
        write_master_check_report(run_id)
        item_after = fetch_one("SELECT * FROM optimizer_master_check_items WHERE id = ?", (item["id"],))
        log(
            f"[MasterCheck] {profile_id}: smoke status={item_after['status']} "
            f"artifact={item_after['safetensors_check_status']} category={item_after['failure_category']}"
        )

        if not args.skip_image_smoke and item_after["status"] == "smoke_ok":
            log(f"[MasterCheck] {profile_id}: Image Smoke start")
            try:
                image_result = run_image_smoke_for_item(int(item_after["id"]), base_model_path=args.base_model_path)
                log(
                    f"[MasterCheck] {profile_id}: image smoke status={image_result.get('status')} "
                    f"diff={image_result.get('difference_score')} elapsed={image_result.get('elapsed_seconds')}"
                )
            except Exception as exc:
                category = classify_failure(str(exc))
                with connect() as conn:
                    conn.execute(
                        """
                        UPDATE optimizer_master_check_items
                        SET image_smoke_status = 'failed', failure_category = COALESCE(?, failure_category),
                            error_message = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (category, str(exc), utc_now(), item_after["id"]),
                    )
                log(f"[MasterCheck] {profile_id}: image smoke failed category={category} error={exc}")
            update_run_counts(run_id)
            write_master_check_report(run_id)

    update_run_counts(run_id)
    mark_run_completed(run_id)
    paths = write_master_check_report(run_id)
    log(f"[MasterCheck] completed run_id={run_id}")
    log(f"[MasterCheck] report={paths['report_path']}")
    log(f"[MasterCheck] json={paths['log_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
