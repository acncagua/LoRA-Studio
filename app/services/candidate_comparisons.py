from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app import settings
from app.db import connect, fetch_all, fetch_one, utc_now
from app.services.review_candidates import ensure_epoch_candidates
from app.services.validation_generation import (
    build_epoch_cross_matrix_html,
    start_validation_assist_sequence,
    start_validation_generation_sequence,
)
from app.services.validation_runs import create_validation_run


STANDARD_PRESET_ID = "standard_validation_v1"
CANDIDATE_LABEL_ORDER = ("primary", "secondary", "check")


def db_retry(operation: Callable[[], Any], *, attempts: int = 6, base_sleep: float = 0.25) -> Any:
    for index in range(attempts):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or index == attempts - 1:
                raise
            time.sleep(base_sleep * (2**index))


def candidate_standard_epochs(job_id: int) -> list[int]:
    rows = ensure_epoch_candidates(job_id)
    by_label: dict[str, list[int]] = {label: [] for label in CANDIDATE_LABEL_ORDER}
    fallback: list[int] = []
    for row in rows:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        label = str(row.get("candidate_label") or "")
        if label in by_label:
            by_label[label].append(int(epoch))
        elif label != "low_priority":
            fallback.append(int(epoch))
    epochs: list[int] = []
    for label in CANDIDATE_LABEL_ORDER:
        for epoch in sorted(by_label[label]):
            if epoch not in epochs:
                epochs.append(epoch)
    for epoch in fallback:
        if epoch not in epochs:
            epochs.append(epoch)
    if len(epochs) < 3:
        for row in rows:
            epoch = row.get("epoch")
            if epoch is not None and int(epoch) not in epochs:
                epochs.append(int(epoch))
            if len(epochs) >= 3:
                break
    return epochs[:3]


def candidate_standard_estimate(job_id: int) -> dict[str, Any]:
    epochs = candidate_standard_epochs(job_id)
    preset = fetch_one("SELECT * FROM validation_presets WHERE id = ?", (STANDARD_PRESET_ID,))
    per_epoch = int((preset["expected_image_count"] if preset else 45) or 45)
    image_count = per_epoch * len(epochs)
    # Conservative, intentionally visible as an estimate: recent SDXL 45-image standard runs are often around 10 minutes.
    estimated_minutes = int(round((image_count / 45) * 10))
    estimated_gb = round((image_count * 2.0) / 1024, 2)
    return {
        "candidate_epochs": epochs,
        "epoch_count": len(epochs),
        "images_per_epoch": per_epoch,
        "expected_total_images": image_count,
        "estimated_runtime_minutes": estimated_minutes,
        "estimated_storage_gb": estimated_gb,
        "preset_id": STANDARD_PRESET_ID,
        "preset_name": preset["name"] if preset else "Standard Validation v1",
    }


def latest_candidate_comparison_groups(job_id: int) -> list[dict[str, Any]]:
    rows = fetch_all(
        """
        SELECT *
        FROM candidate_comparison_groups
        WHERE job_id = ?
        ORDER BY id DESC
        LIMIT 8
        """,
        (job_id,),
    )
    return [decorate_group(dict(row)) for row in rows]


def decorate_group(group: dict[str, Any]) -> dict[str, Any]:
    run_ids = json.loads(group.get("validation_run_ids_json") or "[]")
    epochs = json.loads(group.get("candidate_epochs_json") or "[]")
    runs = []
    if run_ids:
        placeholders = ",".join("?" for _ in run_ids)
        runs = [
            dict(row)
            for row in fetch_all(
                f"""
                SELECT vr.*, vg.status AS generation_status, vg.elapsed_seconds AS generation_elapsed_seconds
                FROM validation_runs vr
                LEFT JOIN validation_generation_runs vg ON vg.id = (
                    SELECT id FROM validation_generation_runs
                    WHERE validation_run_id = vr.id
                    ORDER BY id DESC LIMIT 1
                )
                WHERE vr.id IN ({placeholders})
                ORDER BY vr.selected_epoch
                """,
                tuple(run_ids),
            )
        ]
    group["candidate_epochs"] = epochs
    group["validation_run_ids"] = run_ids
    group["runs"] = runs
    group["matrix_ready"] = bool(group.get("matrix_path") and Path(str(group["matrix_path"])).exists())
    return group


def ensure_candidate_standard_comparison_group(job_id: int, *, force: bool = False) -> dict[str, Any]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    epochs = candidate_standard_epochs(job_id)
    if not epochs:
        raise ValueError("loss候補epochが見つかりません。学習結果を取り込んでから実行してください。")
    estimate = candidate_standard_estimate(job_id)
    epochs_json = json.dumps(epochs, ensure_ascii=False)
    if not force:
        existing = fetch_one(
            """
            SELECT *
            FROM candidate_comparison_groups
            WHERE job_id = ? AND preset_id = ? AND candidate_epochs_json = ?
            ORDER BY id DESC LIMIT 1
            """,
            (job_id, STANDARD_PRESET_ID, epochs_json),
        )
        if existing:
            return decorate_group(dict(existing))

    run_ids = [find_or_create_standard_run(job, epoch) for epoch in epochs]
    now = utc_now()

    def op() -> int:
        with connect() as conn:
            if force:
                conn.execute(
                    """
                    DELETE FROM candidate_comparison_groups
                    WHERE job_id = ? AND preset_id = ? AND candidate_epochs_json = ?
                    """,
                    (job_id, STANDARD_PRESET_ID, epochs_json),
                )
            cur = conn.execute(
                """
                INSERT INTO candidate_comparison_groups(
                    job_id, project_id, preset_id, name, candidate_epochs_json,
                    validation_run_ids_json, expected_total_images, status,
                    estimate_json, created_at, updated_at, memo
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?, ?)
                ON CONFLICT(job_id, preset_id, candidate_epochs_json) DO UPDATE SET
                    validation_run_ids_json = excluded.validation_run_ids_json,
                    expected_total_images = excluded.expected_total_images,
                    estimate_json = excluded.estimate_json,
                    updated_at = excluded.updated_at
                """,
                (
                    job_id,
                    job["project_id"] if "project_id" in job.keys() else None,
                    STANDARD_PRESET_ID,
                    f"Job #{job_id} Standard Candidate Comparison",
                    epochs_json,
                    json.dumps(run_ids, ensure_ascii=False),
                    estimate["expected_total_images"],
                    json.dumps(estimate, ensure_ascii=False),
                    now,
                    now,
                    "loss候補epochのStandard Validation v1一括比較",
                ),
            )
            return int(cur.lastrowid or conn.execute(
                """
                SELECT id FROM candidate_comparison_groups
                WHERE job_id = ? AND preset_id = ? AND candidate_epochs_json = ?
                """,
                (job_id, STANDARD_PRESET_ID, epochs_json),
            ).fetchone()["id"])

    group_id = int(db_retry(op))
    return decorate_group(dict(fetch_one("SELECT * FROM candidate_comparison_groups WHERE id = ?", (group_id,))))


def find_or_create_standard_run(job: Any, epoch: int) -> int:
    output = fetch_one(
        """
        SELECT *
        FROM training_outputs
        WHERE job_id = ? AND epoch = ? AND file_type = 'model'
          AND COALESCE(deleted_at, '') = ''
        ORDER BY step DESC, id DESC
        LIMIT 1
        """,
        (job["id"], epoch),
    )
    if output is None:
        raise ValueError(f"epoch {epoch} のLoRA出力が見つかりません。")
    existing = fetch_one(
        """
        SELECT id
        FROM validation_runs
        WHERE job_id = ? AND selected_output_id = ? AND validation_preset_id = ?
          AND validation_run_kind = 'candidate_standard_comparison'
        ORDER BY id DESC LIMIT 1
        """,
        (job["id"], output["id"], STANDARD_PRESET_ID),
    )
    if existing:
        return int(existing["id"])
    run_id = create_validation_run(
        int(job["id"]),
        STANDARD_PRESET_ID,
        "",
        "",
        f"Standard Candidate Comparison: Job #{job['id']} epoch {epoch}",
        selected_output_id=int(output["id"]),
    )
    with connect() as conn:
        conn.execute(
            """
            UPDATE validation_runs
            SET validation_run_kind = 'candidate_standard_comparison',
                source_training_job_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (job["id"], utc_now(), run_id),
        )
    return run_id


def start_candidate_standard_comparison(group_id: int) -> dict[str, Any]:
    group = fetch_one("SELECT * FROM candidate_comparison_groups WHERE id = ?", (group_id,))
    if group is None:
        raise ValueError(f"Candidate comparison group not found: {group_id}")
    if group["status"] in {"running", "generating_images", "embedding_images", "machine_reviewing", "building_matrix"}:
        raise RuntimeError("Standard Candidate Comparisonは既に実行中です。")
    thread = threading.Thread(target=_comparison_worker, args=(group_id,), daemon=True)
    thread.start()
    return {"group_id": group_id, "status": "started"}


def _set_group_status(group_id: int, status: str, **values: Any) -> None:
    now = utc_now()

    def op() -> None:
        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status, now]
        for key, value in values.items():
            assignments.append(f"{key} = ?")
            params.append(value)
        params.append(group_id)
        with connect() as conn:
            conn.execute(f"UPDATE candidate_comparison_groups SET {', '.join(assignments)} WHERE id = ?", tuple(params))

    db_retry(op)


def _comparison_worker(group_id: int) -> None:
    started = utc_now()
    group = fetch_one("SELECT * FROM candidate_comparison_groups WHERE id = ?", (group_id,))
    if group is None:
        return
    run_ids = [int(value) for value in json.loads(group["validation_run_ids_json"] or "[]")]
    timings: dict[str, Any] = {"pipeline_start": started}
    try:
        _set_group_status(group_id, "generating_images", started_at=started, error_message="")
        timings["generation_start"] = utc_now()
        start_validation_generation_sequence(run_ids)
        _wait_for_generation_runs(run_ids)
        timings["generation_end"] = utc_now()
        _refresh_group_counts(group_id)

        _set_group_status(group_id, "embedding_images")
        timings["assist_start"] = utc_now()
        start_validation_assist_sequence(run_ids)
        _wait_for_assist(run_ids)
        timings["assist_end"] = utc_now()
        _refresh_group_counts(group_id)

        _set_group_status(group_id, "building_matrix")
        timings["matrix_start"] = utc_now()
        matrix_path = write_group_cross_matrix(group_id)
        timings["matrix_end"] = utc_now()
        ended = utc_now()
        timings["pipeline_end"] = ended
        elapsed = elapsed_seconds(started, ended)
        _refresh_group_counts(group_id)
        _set_group_status(
            group_id,
            "completed",
            matrix_path=matrix_path,
            stage_timing_json=json.dumps(timings, ensure_ascii=False),
            ended_at=ended,
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        ended = utc_now()
        timings["pipeline_end"] = ended
        _set_group_status(
            group_id,
            "failed",
            stage_timing_json=json.dumps(timings, ensure_ascii=False),
            ended_at=ended,
            elapsed_seconds=elapsed_seconds(started, ended),
            error_message=str(exc),
        )


def _wait_for_generation_runs(run_ids: list[int], timeout_seconds: int = 60 * 60 * 8) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        rows = fetch_all(
            f"""
            SELECT validation_run_id, status
            FROM validation_generation_runs
            WHERE id IN (
                SELECT MAX(id)
                FROM validation_generation_runs
                WHERE validation_run_id IN ({','.join('?' for _ in run_ids)})
                GROUP BY validation_run_id
            )
            """,
            tuple(run_ids),
        )
        status_by_run = {int(row["validation_run_id"]): str(row["status"]) for row in rows}
        if all(status_by_run.get(run_id) == "completed" for run_id in run_ids):
            return
        if any(status_by_run.get(run_id) in {"failed", "stopped"} for run_id in run_ids):
            raise RuntimeError(f"Validation画像生成が失敗または停止しました: {status_by_run}")
        time.sleep(5)
    raise RuntimeError("Validation画像生成がタイムアウトしました。")


def _wait_for_assist(run_ids: list[int], timeout_seconds: int = 60 * 60 * 8) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        expected = _expected_total(run_ids)
        scores = _score_total(run_ids)
        running_embedding = fetch_one("SELECT id FROM embedding_jobs WHERE status = 'running' LIMIT 1")
        running_review = fetch_one("SELECT id FROM machine_review_jobs WHERE status = 'running' LIMIT 1")
        if scores >= expected and not running_embedding and not running_review:
            return
        time.sleep(5)
    raise RuntimeError("Embedding / Machine Reviewがタイムアウトしました。")


def _expected_total(run_ids: list[int]) -> int:
    if not run_ids:
        return 0
    return int(fetch_one(
        f"SELECT COALESCE(SUM(expected_image_count), 0) AS count FROM validation_runs WHERE id IN ({','.join('?' for _ in run_ids)})",
        tuple(run_ids),
    )["count"] or 0)


def _score_total(run_ids: list[int]) -> int:
    if not run_ids:
        return 0
    return int(fetch_one(
        f"""
        SELECT COUNT(*) AS count
        FROM machine_review_scores
        WHERE source_type = 'validation_image'
          AND validation_run_id IN ({','.join('?' for _ in run_ids)})
        """,
        tuple(run_ids),
    )["count"] or 0)


def _refresh_group_counts(group_id: int) -> None:
    group = fetch_one("SELECT validation_run_ids_json FROM candidate_comparison_groups WHERE id = ?", (group_id,))
    if group is None:
        return
    run_ids = [int(value) for value in json.loads(group["validation_run_ids_json"] or "[]")]
    if not run_ids:
        return
    placeholders = ",".join("?" for _ in run_ids)
    registered = int(fetch_one(f"SELECT COUNT(*) AS count FROM validation_images WHERE validation_run_id IN ({placeholders})", tuple(run_ids))["count"] or 0)
    embedding_ready = int(fetch_one(f"SELECT COUNT(*) AS count FROM image_embeddings WHERE validation_run_id IN ({placeholders}) AND status = 'ready'", tuple(run_ids))["count"] or 0)
    scores = _score_total(run_ids)

    def op() -> None:
        with connect() as conn:
            conn.execute(
                """
                UPDATE candidate_comparison_groups
                SET registered_image_count = ?, embedding_ready_count = ?,
                    machine_review_score_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (registered, embedding_ready, scores, utc_now(), group_id),
            )

    db_retry(op)


def write_group_cross_matrix(group_id: int) -> str:
    group = fetch_one("SELECT * FROM candidate_comparison_groups WHERE id = ?", (group_id,))
    if group is None:
        raise ValueError(f"Candidate comparison group not found: {group_id}")
    run_ids = [int(value) for value in json.loads(group["validation_run_ids_json"] or "[]")]
    path = settings.EXPORTS_DIR / "validation_runs" / f"candidate_comparison_group_{group_id:06d}_matrix.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_epoch_cross_matrix_html(int(group["job_id"]), run_ids), encoding="utf-8")
    return str(path)


def elapsed_seconds(start: str, end: str) -> int:
    try:
        return int((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
    except ValueError:
        return 0
