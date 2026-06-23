from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.services.training_runner import decode_log_bytes, process_exists


LOG_STALE_WARNING_SECONDS = 10 * 60


def tail_file(path_text: str | None, max_lines: int = 12) -> str:
    if not path_text:
        return ""
    path = Path(str(path_text))
    if not path.exists():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return "\n".join(decode_log_bytes(data).splitlines()[-max_lines:])


def file_mtime(path_text: str | None) -> datetime | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if not path.exists():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def age_label(moment: datetime | None, now: datetime | None = None) -> str:
    if moment is None:
        return "-"
    now = now or datetime.now(timezone.utc)
    seconds = max(0, int((now - moment).total_seconds()))
    if seconds < 60:
        return f"{seconds} sec ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    return f"{hours} h ago"


def elapsed_label(started_at: str | None, now: datetime | None = None) -> str:
    started = parse_iso_datetime(started_at)
    if started is None:
        return "-"
    now = now or datetime.now(timezone.utc)
    seconds = max(0, int((now - started).total_seconds()))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def format_duration(seconds: int | float | None) -> str:
    if seconds is None:
        return "-"
    try:
        value = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "-"
    minutes, sec = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def parse_duration_fragment(value: str | None) -> int | None:
    if not value:
        return None
    parts = str(value).strip().split(":")
    if not all(part.isdigit() for part in parts):
        return None
    numbers = [int(part) for part in parts]
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 1:
        return numbers[0]
    return None


def completion_eta_label(remaining_seconds: int | None, now: datetime | None = None) -> str:
    if remaining_seconds is None:
        return "-"
    now = now or datetime.now(timezone.utc)
    eta = now.astimezone() + timedelta(seconds=max(0, int(remaining_seconds)))
    return eta.strftime("%Y-%m-%d %H:%M:%S")


def completion_eta_from_seconds_label(total_remaining_seconds: int | None, now: datetime | None = None) -> str:
    return completion_eta_label(total_remaining_seconds, now)


def parse_training_tqdm_timing(log_tail: str) -> dict[str, Any]:
    matches = list(
        re.finditer(
            r"steps:\s+.*?\|\s*(?P<current>\d+)/(?P<total>\d+)\s+\[(?P<elapsed>[^<\]]+)<(?P<remaining>[^,\]]+)(?:,\s*(?P<rate>[^\]]+))?\]",
            log_tail,
        )
    )
    if not matches:
        return {}
    match = matches[-1]
    elapsed_seconds = parse_duration_fragment(match.group("elapsed"))
    remaining_seconds = parse_duration_fragment(match.group("remaining"))
    current = int(match.group("current"))
    total = int(match.group("total"))
    rate = (match.group("rate") or "").strip().split(",", 1)[0].strip()
    return {
        "current": current,
        "total": total,
        "stage_elapsed_seconds": elapsed_seconds,
        "estimated_remaining_seconds": remaining_seconds,
        "estimated_total_seconds": (elapsed_seconds + remaining_seconds) if elapsed_seconds is not None and remaining_seconds is not None else None,
        "rate_label": rate,
    }


def operation_timing_summary(
    *,
    started_at: str | None,
    progress_current: int | None,
    progress_total: int | None,
    log_tail: str = "",
    operation_type: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    started = parse_iso_datetime(started_at)
    elapsed_seconds = max(0, int((now - started).total_seconds())) if started else None
    timing: dict[str, Any] = {}
    if operation_type == "training":
        timing = parse_training_tqdm_timing(log_tail)
        progress_current = timing.get("current", progress_current)
        progress_total = timing.get("total", progress_total)

    estimated_remaining = timing.get("estimated_remaining_seconds")
    estimated_total = timing.get("estimated_total_seconds")
    stage_elapsed = timing.get("stage_elapsed_seconds")
    rate_label = timing.get("rate_label") or ""

    if estimated_remaining is None and elapsed_seconds is not None and progress_current and progress_total:
        estimated_total = max(elapsed_seconds, int(round(elapsed_seconds * int(progress_total) / max(1, int(progress_current)))))
        estimated_remaining = max(0, estimated_total - elapsed_seconds)
    if stage_elapsed is None:
        stage_elapsed = elapsed_seconds
    if estimated_total is None and elapsed_seconds is not None and estimated_remaining is not None:
        estimated_total = elapsed_seconds + estimated_remaining

    return {
        "elapsed_seconds": elapsed_seconds,
        "elapsed_label": format_duration(elapsed_seconds),
        "stage_elapsed_seconds": stage_elapsed,
        "stage_elapsed_label": format_duration(stage_elapsed),
        "estimated_remaining_seconds": estimated_remaining,
        "estimated_remaining_label": format_duration(estimated_remaining),
        "stage_remaining_label": format_duration(estimated_remaining),
        "estimated_total_seconds": estimated_total,
        "estimated_total_label": format_duration(estimated_total),
        "completion_eta_label": completion_eta_label(estimated_remaining, now),
        "rate_label": rate_label or "-",
    }


def log_warning(log_updated_at: datetime | None, now: datetime | None = None) -> str:
    if log_updated_at is None:
        return ""
    now = now or datetime.now(timezone.utc)
    if (now - log_updated_at).total_seconds() < LOG_STALE_WARNING_SECONDS:
        return ""
    return "ログ更新が一定時間ありません。処理が停止しているとは限りませんが、必要ならFull Logやシステム状態を確認してください。"


def training_progress_from_log(log_tail: str) -> tuple[int | None, int | None, str]:
    matches = list(re.finditer(r"steps:\s+.*?\|\s*(\d+)/(\d+)\s+\[[^\]]+\]", log_tail))
    if not matches:
        return None, None, ""
    match = matches[-1]
    current = int(match.group(1))
    total = int(match.group(2))
    return current, total, f"{current} / {total}"


def operation_monitor(
    *,
    operation_type: str,
    type_label: str,
    status: str | None,
    stage: str | None = None,
    started_at: str | None = None,
    pid: int | None = None,
    return_code: int | None = None,
    progress_current: int | None = None,
    progress_total: int | None = None,
    log_path: str | None = None,
    stop_action: str | None = None,
    full_log_anchor: str | None = None,
    status_url: str | None = None,
    message: str | None = None,
    followup_estimate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    short_tail = tail_file(log_path, 12)
    full_tail = tail_file(log_path, 80)
    parse_tail = tail_file(log_path, 500) if operation_type == "training" else full_tail
    log_updated_at = file_mtime(log_path)
    if operation_type == "training" and progress_current is None:
        progress_current, progress_total, parsed_progress = training_progress_from_log(parse_tail)
    else:
        parsed_progress = ""
    if progress_current is not None and progress_total is not None:
        progress_label = f"{progress_current} / {progress_total}"
    else:
        progress_label = parsed_progress
    timing = operation_timing_summary(
        started_at=started_at,
        progress_current=progress_current,
        progress_total=progress_total,
        log_tail=parse_tail,
        operation_type=operation_type,
        now=now,
    )
    if pid and status == "running" and not process_exists(int(pid)):
        stage = stage or "プロセス未確認"
    if followup_estimate:
        followup_seconds = int(followup_estimate.get("estimated_seconds") or 0)
        current_remaining = timing["estimated_remaining_seconds"]
        if current_remaining is not None:
            followup_estimate["pc_total_remaining_seconds"] = int(current_remaining) + followup_seconds
            followup_estimate["pc_total_remaining_label"] = format_duration(followup_estimate["pc_total_remaining_seconds"])
            followup_estimate["pc_total_completion_eta_label"] = completion_eta_from_seconds_label(followup_estimate["pc_total_remaining_seconds"], now)
        else:
            followup_estimate.setdefault("pc_total_remaining_seconds", None)
            followup_estimate.setdefault("pc_total_remaining_label", "-")
            followup_estimate.setdefault("pc_total_completion_eta_label", "-")
        followup_estimate["estimated_label"] = format_duration(followup_seconds)
        for key in ("generation_seconds", "import_seconds", "embedding_seconds", "machine_review_seconds", "matrix_seconds"):
            followup_estimate[f"{key}_label"] = format_duration(followup_estimate.get(key))
    return {
        "operation_type": operation_type,
        "type_label": type_label,
        "status": status or "-",
        "stage": stage or "-",
        "started_at": started_at or "-",
        "elapsed_label": timing["elapsed_label"],
        "elapsed_seconds": timing["elapsed_seconds"],
        "stage_elapsed_label": timing["stage_elapsed_label"],
        "stage_elapsed_seconds": timing["stage_elapsed_seconds"],
        "estimated_remaining_label": timing["estimated_remaining_label"],
        "estimated_remaining_seconds": timing["estimated_remaining_seconds"],
        "stage_remaining_label": timing["stage_remaining_label"],
        "estimated_total_label": timing["estimated_total_label"],
        "estimated_total_seconds": timing["estimated_total_seconds"],
        "completion_eta_label": timing["completion_eta_label"],
        "rate_label": timing["rate_label"],
        "pid": pid,
        "return_code": return_code,
        "progress_label": progress_label,
        "last_log_update_label": age_label(log_updated_at, now),
        "log_warning": log_warning(log_updated_at, now),
        "log_path": log_path or "",
        "log_tail_short": short_tail,
        "log_tail_full": full_tail,
        "stop_action": stop_action or "",
        "full_log_anchor": full_log_anchor or "",
        "status_url": status_url or "",
        "message": message or "",
        "is_running": status == "running",
        "followup_estimate": followup_estimate or {},
    }


def running_training_monitor(job: dict[str, Any], followup_estimate: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if job.get("status") != "running":
        return None
    log_path = str(Path(job["run_dir"]) / "logs" / "train.log")
    expected_total = job.get("expected_total_steps") or job.get("expected_total_steps_at_creation")
    return operation_monitor(
        operation_type="training",
        type_label="学習ジョブ",
        status=job.get("status"),
        stage="学習中",
        started_at=job.get("start_time"),
        pid=job.get("process_id"),
        return_code=job.get("return_code"),
        progress_total=int(expected_total) if expected_total else None,
        log_path=log_path,
        stop_action=f"/jobs/{job['id']}/stop",
        full_log_anchor="#technical-log",
        status_url=f"/jobs/{job['id']}/log-tail/status",
        message="処理は実行中です。ログ末尾で進行状況を確認できます。",
        followup_estimate=followup_estimate,
    )


def review_session_monitor(session: dict[str, Any]) -> dict[str, Any] | None:
    if session.get("status") not in {"running", "generating_images", "embedding_images", "machine_reviewing", "building_matrix"}:
        return None
    current = int(session.get("generated_image_count") or session.get("imported_image_count") or session.get("scored_image_count") or 0)
    total = int(session.get("expected_image_count") or 0)
    return operation_monitor(
        operation_type="review_generation",
        type_label="レビューセッション",
        status=session.get("status"),
        stage=review_stage_label(session),
        started_at=session.get("started_at"),
        pid=session.get("generation_process_id"),
        return_code=session.get("return_code"),
        progress_current=current,
        progress_total=total,
        log_path=session.get("log_path"),
        stop_action=f"/jobs/{session['job_id']}/review-sessions/{session['id']}/stop" if session.get("job_id") else "",
        full_log_anchor="#active-operation-monitor",
        status_url=f"/jobs/{session['job_id']}/review-sessions/{session['id']}/status" if session.get("job_id") else "",
        message="レビューセッションの処理は実行中です。ログ末尾で進行状況を確認できます。",
    )


def review_stage_label(session: dict[str, Any]) -> str:
    status = session.get("status")
    if status == "generating_images":
        return "Generating images"
    if status == "embedding_images":
        return "Computing embeddings"
    if status == "machine_reviewing":
        return "機械補助レビュー"
    if status == "building_matrix":
        return "Building matrix"
    generated = int(session.get("generated_image_count") or 0)
    imported = int(session.get("imported_image_count") or 0)
    scored = int(session.get("scored_image_count") or 0)
    if generated == 0:
        return "Generating images"
    if imported < generated:
        return "Importing images"
    if scored < imported:
        return "機械補助レビュー"
    return "Building matrix"


def validation_generation_monitor(generation: dict[str, Any] | None, run_id: int) -> dict[str, Any] | None:
    if not generation or generation.get("status") != "running":
        return None
    return operation_monitor(
        operation_type="validation_generation",
        type_label="検証画像生成",
        status=generation.get("status"),
        stage="Generating images",
        started_at=generation.get("started_at"),
        pid=generation.get("process_id"),
        return_code=generation.get("return_code"),
        progress_current=generation.get("generated_image_count"),
        progress_total=None,
        log_path=generation.get("log_path"),
        stop_action=f"/validation-runs/{run_id}/generation/stop",
        full_log_anchor="#active-operation-monitor",
        status_url=f"/validation-runs/{run_id}/generation/status",
        message="検証画像生成は実行中です。ログ末尾で進行状況を確認できます。",
    )


def validation_pipeline_monitor(run: dict[str, Any]) -> dict[str, Any] | None:
    status = run.get("pipeline_status")
    if status not in {"generating_images", "importing_images", "embedding_images", "machine_reviewing", "building_matrix"}:
        return None
    run_id = int(run["id"])
    log_path = Path("exports") / "validation_runs" / f"validation_run_{run_id:06d}" / "generation" / "weight_calibration_pipeline.log"
    progress_current = int(run.get("actual_image_count") or 0)
    progress_total = int(run.get("expected_image_count") or 0)
    return operation_monitor(
        operation_type="weight_calibration",
        type_label="Weight Calibration",
        status=status,
        stage=validation_pipeline_stage_label(status),
        started_at=run.get("updated_at"),
        pid=None,
        return_code=None,
        progress_current=progress_current,
        progress_total=progress_total,
        log_path=str(log_path),
        stop_action=f"/validation-runs/{run_id}/pipeline/stop",
        full_log_anchor="#active-operation-monitor",
        status_url="",
        message="Weight Calibration Pipelineを実行中です。画像生成、Embedding、機械補助レビュー、Matrix作成を順に進めます。",
    )


def validation_pipeline_stage_label(status: str | None) -> str:
    labels = {
        "generating_images": "Generating images",
        "importing_images": "Importing images",
        "embedding_images": "Computing embeddings",
        "machine_reviewing": "機械補助レビュー",
        "building_matrix": "Building matrix",
    }
    return labels.get(status or "", status or "-")


def embedding_monitor(job: dict[str, Any]) -> dict[str, Any] | None:
    if not job or job.get("status") != "running":
        return None
    return operation_monitor(
        operation_type="embedding",
        type_label="Embedding",
        status=job.get("status"),
        stage=f"{job.get('job_type') or '-'}",
        started_at=job.get("started_at"),
        pid=job.get("process_id"),
        return_code=job.get("return_code"),
        progress_current=job.get("processed_count"),
        progress_total=job.get("total_count"),
        log_path=job.get("log_path"),
        full_log_anchor="#machine-assist-readiness",
        status_url=f"/embeddings/jobs/{job['id']}/status" if job.get("id") else "",
        message="Embedding計算は実行中です。ログ末尾で進行状況を確認できます。",
    )


def machine_review_monitor(job: dict[str, Any]) -> dict[str, Any] | None:
    if not job or job.get("status") != "running":
        return None
    return operation_monitor(
        operation_type="machine_review",
        type_label="機械補助レビュー",
        status=job.get("status"),
        stage=f"{job.get('target_type') or '-'}",
        started_at=job.get("started_at"),
        pid=job.get("process_id"),
        return_code=job.get("return_code"),
        progress_current=job.get("processed_count"),
        progress_total=job.get("total_count"),
        log_path=job.get("log_path"),
        full_log_anchor="#machine-assist",
        status_url=f"/machine-review/jobs/{job['id']}/status" if job.get("id") else "",
        message="機械補助レビューは実行中です。ログ末尾で進行状況を確認できます。",
    )
