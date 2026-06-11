from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from app.db import connect, fetch_all, fetch_one, utc_now

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def collect_job_metrics(job_id: int) -> dict[str, Any]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    run_dir = Path(job["run_dir"])

    metrics = read_tensorboard_metrics(run_dir / "metrics")
    source = "tensorboard"
    if not metrics:
        metrics = read_train_log_metrics(run_dir / "logs" / "train.log")
        source = "train_log"

    if metrics:
        replace_metrics(job_id, metrics, source)

    summary = summarize_metrics(job_id)
    consistency = update_step_consistency(job_id)
    return {"metrics": len(metrics), "source": source if metrics else None, "summary": summary, "consistency": consistency}


def read_tensorboard_metrics(metrics_dir: Path) -> list[dict[str, Any]]:
    event_files = sorted(metrics_dir.rglob("events.out.tfevents.*")) if metrics_dir.exists() else []
    if not event_files:
        return []
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return []

    by_step: dict[int, dict[str, Any]] = {}
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
        except Exception:
            continue
        for tag in accumulator.Tags().get("scalars", []):
            tag_lower = tag.lower()
            if not is_loss_tag(tag_lower) and not is_lr_tag(tag_lower):
                continue
            for scalar in accumulator.Scalars(tag):
                row = by_step.setdefault(
                    int(scalar.step),
                    {"step": int(scalar.step), "epoch": None, "loss": None, "learning_rate": None, "raw_tag": tag},
                )
                if is_loss_tag(tag_lower) and row["loss"] is None:
                    row["loss"] = float(scalar.value)
                    row["raw_tag"] = tag
                elif is_lr_tag(tag_lower) and row["learning_rate"] is None:
                    row["learning_rate"] = float(scalar.value)
    return [row for row in sorted(by_step.values(), key=lambda item: item["step"]) if row["loss"] is not None or row["learning_rate"] is not None]


def is_loss_tag(tag: str) -> bool:
    return "loss" in tag and "val" not in tag and "validation" not in tag


def is_lr_tag(tag: str) -> bool:
    return tag in {"lr", "learning_rate"} or tag.endswith("/lr") or "learning_rate" in tag


def read_train_log_metrics(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[int, dict[str, Any]] = {}
    pattern = re.compile(r"steps:\s*\d+%.*?\|\s*(\d+)/(\d+).*?avr_loss=([0-9.eE+-]+)")
    for match in pattern.finditer(text):
        step = int(match.group(1))
        metrics[step] = {
            "step": step,
            "epoch": None,
            "loss": float(match.group(3)),
            "learning_rate": None,
            "raw_tag": "avr_loss",
        }
    return [metrics[key] for key in sorted(metrics)]


def replace_metrics(job_id: int, metrics: list[dict[str, Any]], source: str) -> None:
    now = utc_now()
    with connect() as conn:
        conn.execute("DELETE FROM training_metrics WHERE job_id = ?", (job_id,))
        conn.executemany(
            """
            INSERT INTO training_metrics(
                job_id, step, epoch, loss, learning_rate, source, raw_tag, raw_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, step, raw_tag) DO UPDATE SET
                epoch = excluded.epoch,
                loss = excluded.loss,
                learning_rate = excluded.learning_rate,
                source = excluded.source,
                raw_json = excluded.raw_json,
                created_at = excluded.created_at
            """,
            [
                (
                    job_id,
                    item.get("step"),
                    item.get("epoch"),
                    item.get("loss"),
                    item.get("learning_rate"),
                    source,
                    item.get("raw_tag") or source,
                    json.dumps(item, ensure_ascii=False),
                    now,
                )
                for item in metrics
            ],
        )


def summarize_metrics(job_id: int) -> dict[str, Any]:
    rows = fetch_all(
        "SELECT * FROM training_metrics WHERE job_id = ? AND loss IS NOT NULL ORDER BY step",
        (job_id,),
    )
    now = utc_now()
    if len(rows) < 2:
        summary = {
            "initial_loss": rows[0]["loss"] if rows else None,
            "final_loss": rows[-1]["loss"] if rows else None,
            "min_loss": rows[0]["loss"] if rows else None,
            "min_loss_step": rows[0]["step"] if rows else None,
            "max_loss": rows[0]["loss"] if rows else None,
            "loss_drop_rate": None,
            "loss_volatility": None,
            "spike_count": 0,
            "late_stage_slope": None,
            "health_label": "UNKNOWN",
            "health_message": "Metric count is too small to judge loss health.",
        }
    else:
        losses = [float(row["loss"]) for row in rows]
        steps = [int(row["step"]) for row in rows]
        initial = losses[0]
        final = losses[-1]
        min_index = min(range(len(losses)), key=lambda index: losses[index])
        deltas = [losses[index] - losses[index - 1] for index in range(1, len(losses))]
        spike_count = sum(1 for index in range(1, len(losses)) if losses[index] > losses[index - 1] * 1.5)
        tail = max(2, len(losses) // 3)
        late_stage_slope = (losses[-1] - losses[-tail]) / max(1, steps[-1] - steps[-tail])
        loss_drop_rate = (initial - final) / initial if initial else None
        volatility = statistics.pstdev(deltas) if len(deltas) > 1 else abs(deltas[0])
        health_label, health_message = judge_health(losses, spike_count, late_stage_slope, loss_drop_rate)
        summary = {
            "initial_loss": initial,
            "final_loss": final,
            "min_loss": losses[min_index],
            "min_loss_step": steps[min_index],
            "max_loss": max(losses),
            "loss_drop_rate": loss_drop_rate,
            "loss_volatility": volatility,
            "spike_count": spike_count,
            "late_stage_slope": late_stage_slope,
            "health_label": health_label,
            "health_message": health_message,
        }

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO training_metric_summaries(
                job_id, initial_loss, final_loss, min_loss, min_loss_step, max_loss,
                loss_drop_rate, loss_volatility, spike_count, late_stage_slope,
                health_label, health_message, summary_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                initial_loss = excluded.initial_loss,
                final_loss = excluded.final_loss,
                min_loss = excluded.min_loss,
                min_loss_step = excluded.min_loss_step,
                max_loss = excluded.max_loss,
                loss_drop_rate = excluded.loss_drop_rate,
                loss_volatility = excluded.loss_volatility,
                spike_count = excluded.spike_count,
                late_stage_slope = excluded.late_stage_slope,
                health_label = excluded.health_label,
                health_message = excluded.health_message,
                summary_json = excluded.summary_json,
                updated_at = excluded.updated_at
            """,
            (
                job_id,
                summary["initial_loss"],
                summary["final_loss"],
                summary["min_loss"],
                summary["min_loss_step"],
                summary["max_loss"],
                summary["loss_drop_rate"],
                summary["loss_volatility"],
                summary["spike_count"],
                summary["late_stage_slope"],
                summary["health_label"],
                summary["health_message"],
                json.dumps(summary, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.execute(
            "UPDATE training_jobs SET loss_health_label = ?, updated_at = ? WHERE id = ?",
            (summary["health_label"], now, job_id),
        )
    return summary


def judge_health(
    losses: list[float],
    spike_count: int,
    late_stage_slope: float | None,
    loss_drop_rate: float | None,
) -> tuple[str, str]:
    if len(losses) < 2:
        return "UNKNOWN", "Metric count is too small to judge loss health."
    if late_stage_slope is not None and late_stage_slope > max(0.01, abs(losses[-1]) * 0.1):
        return "DANGER", "Loss rises sharply in the late stage."
    if loss_drop_rate is not None and loss_drop_rate <= 0:
        return "WARNING", "Loss did not decrease from the first measured point."
    if spike_count >= max(2, len(losses) // 3):
        return "WARNING", "Loss has repeated spikes."
    return "OK", "Loss decreases without repeated large spikes."


def update_step_consistency(job_id: int) -> dict[str, Any]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    expected = expected_total_steps(dict(job))
    actual = actual_max_step(dict(job))
    metric_count = fetch_one("SELECT COUNT(*) AS count FROM training_metrics WHERE job_id = ?", (job_id,))["count"]
    output_count = fetch_one("SELECT COUNT(*) AS count FROM training_outputs WHERE job_id = ? AND file_type = 'model'", (job_id,))["count"]
    sample_count = fetch_one("SELECT COUNT(*) AS count FROM sample_images WHERE job_id = ?", (job_id,))["count"]
    label, message = judge_step_consistency(job["status"], expected, actual)
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET expected_total_steps = ?, actual_max_step = ?, actual_metric_count = ?,
                output_model_count = ?, sample_image_count = ?,
                step_consistency_label = ?, step_consistency_message = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (expected, actual, metric_count, output_count, sample_count, label, message, now, job_id),
        )
    return {
        "expected_total_steps": expected,
        "actual_max_step": actual,
        "actual_metric_count": metric_count,
        "output_model_count": output_count,
        "sample_image_count": sample_count,
        "step_consistency_label": label,
        "step_consistency_message": message,
    }


def expected_total_steps(job: dict[str, Any]) -> int | None:
    params = json.loads(job["params_json"])
    if params.get("max_train_steps"):
        return int(params["max_train_steps"])
    config_path = Path(job["run_dir"]) / "config" / "dataset_config.toml"
    batch_size = max(1, int(params.get("train_batch_size") or parse_batch_size(config_path) or 1))
    epoch_count = int(params.get("max_train_epochs") or 1)
    total_images = parse_dataset_image_repeats(config_path)
    if total_images <= 0:
        dataset = fetch_one("SELECT image_count FROM datasets WHERE id = ?", (job["dataset_id"],))
        total_images = int(dataset["image_count"] or 0) * int(params.get("repeats") or 1) if dataset else 0
    if total_images <= 0:
        return None
    return max(1, math.ceil(total_images * epoch_count / batch_size))


def parse_batch_size(config_path: Path) -> int | None:
    if not config_path.exists():
        return None
    match = re.search(r"batch_size\s*=\s*(\d+)", config_path.read_text(encoding="utf-8", errors="replace"))
    return int(match.group(1)) if match else None


def parse_dataset_image_repeats(config_path: Path) -> int:
    if not config_path.exists():
        return 0
    text = config_path.read_text(encoding="utf-8", errors="replace")
    total = 0
    for block in re.split(r"\[\[datasets\.subsets\]\]", text)[1:]:
        image_match = re.search(r'image_dir\s*=\s*"([^"]+)"', block)
        repeat_match = re.search(r"num_repeats\s*=\s*(\d+)", block)
        if not image_match:
            continue
        image_dir = Path(image_match.group(1))
        repeats = int(repeat_match.group(1)) if repeat_match else 1
        image_count = sum(1 for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        total += image_count * repeats
    return total


def actual_max_step(job: dict[str, Any]) -> int | None:
    metric = fetch_one("SELECT MAX(step) AS step FROM training_metrics WHERE job_id = ?", (job["id"],))
    if metric and metric["step"] is not None:
        return int(metric["step"])
    log_path = Path(job["run_dir"]) / "logs" / "train.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    steps = [int(match.group(1)) for match in re.finditer(r"steps:\s*\d+%.*?\|\s*(\d+)/\d+", text)]
    return max(steps) if steps else None


def judge_step_consistency(status: str, expected: int | None, actual: int | None) -> tuple[str, str]:
    if expected is None:
        return "WARNING", "Expected total steps could not be calculated."
    if actual is None:
        return "WARNING", "Actual max step could not be read from metrics or train.log."
    tolerance = max(1, math.ceil(expected * 0.05))
    if abs(expected - actual) <= tolerance:
        return "OK", f"Actual max step {actual} is consistent with expected {expected}."
    if status == "completed" and actual < max(1, expected // 2):
        return "ERROR", f"Job completed at step {actual}, far below expected {expected}."
    return "WARNING", f"Actual max step {actual} differs from expected {expected}."
