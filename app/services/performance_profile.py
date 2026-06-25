from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app import settings
from app.db import connect, fetch_all, fetch_one, utc_now
from app.services.embedding_service import active_embedding_model, load_embedding_settings


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def mark_stage(table_name: str, row_id: int, key: str, value: str | None = None) -> None:
    update_timing(table_name, row_id, lambda data: data.__setitem__(key, value or utc_now()))


def reset_pipeline_timing(table_name: str, row_id: int, *, commands: list[dict[str, Any]] | None = None, output_dir: str = "") -> None:
    now = utc_now()
    payload: dict[str, Any] = {
        "pipeline_start": now,
        "commands": [],
        "output_dir": output_dir,
    }
    if commands:
        payload["commands"] = [
            {
                "command_index": index,
                "name": command.get("name") or f"command_{index + 1}",
                "lora_file": lora_file_from_command(command),
                "condition_count": int(command.get("condition_count") or 0),
            }
            for index, command in enumerate(commands)
        ]
    save_timing(table_name, row_id, payload)


def mark_command_start(table_name: str, row_id: int, command_index: int) -> str:
    started = utc_now()

    def mutate(data: dict[str, Any]) -> None:
        command = ensure_command(data, command_index)
        command["process_start"] = started

    update_timing(table_name, row_id, mutate)
    return started


def mark_command_end(table_name: str, row_id: int, command_index: int, *, output_dir: str, return_code: int | None = None) -> None:
    ended = utc_now()

    def mutate(data: dict[str, Any]) -> None:
        command = ensure_command(data, command_index)
        command["process_end"] = ended
        if return_code is not None:
            command["return_code"] = return_code
        start_ts = parse_timestamp(command.get("process_start"))
        end_ts = parse_timestamp(ended)
        if start_ts and end_ts:
            command["elapsed_seconds"] = max(0, int(round(end_ts - start_ts)))
        image_stats = image_mtime_stats(Path(output_dir), start_ts, end_ts)
        command.update(image_stats)

    update_timing(table_name, row_id, mutate)


def refresh_image_mtime_summary(table_name: str, row_id: int, output_dir: str) -> None:
    def mutate(data: dict[str, Any]) -> None:
        stats = image_mtime_stats(Path(output_dir), None, None)
        if stats.get("first_output_detected_at"):
            data["generation_first_image_time"] = stats["first_output_detected_at"]
        if stats.get("last_output_detected_at"):
            data["generation_last_image_time"] = stats["last_output_detected_at"]
        data["generated_output_count"] = stats.get("output_count", 0)

    update_timing(table_name, row_id, mutate)


def performance_summary(row: Any | None, *, output_dir: str = "", model_paths: list[str] | None = None) -> dict[str, Any]:
    if row is None:
        return empty_summary()
    timing = json_loads(row["stage_timing_json"] if "stage_timing_json" in row.keys() else None, {})
    if output_dir and not timing.get("generation_first_image_time"):
        stats = image_mtime_stats(Path(output_dir), None, None)
        timing.setdefault("generation_first_image_time", stats.get("first_output_detected_at"))
        timing.setdefault("generation_last_image_time", stats.get("last_output_detected_at"))
        timing.setdefault("generated_output_count", stats.get("output_count", 0))

    commands = timing.get("commands") if isinstance(timing.get("commands"), list) else []
    generated_count = int(timing.get("generated_output_count") or 0)
    if not generated_count and "generated_image_count" in row.keys() and row["generated_image_count"] is not None:
        generated_count = int(row["generated_image_count"] or 0)
    if not generated_count:
        generated_count = sum(int(command.get("output_count") or 0) for command in commands)
    generation_elapsed = seconds_between(timing.get("generation_start"), timing.get("generation_end"))
    if generation_elapsed is None:
        generation_elapsed = seconds_between(timing.get("generation_start"), timing.get("generation_last_image_time"))
    command_elapsed = sum(int(command.get("elapsed_seconds") or 0) for command in commands)
    image_span = seconds_between(timing.get("generation_first_image_time"), timing.get("generation_last_image_time")) or 0
    model_load_overhead = max(0, command_elapsed - image_span) if command_elapsed else None
    command_count = len(commands)
    avg_seconds = round(generation_elapsed / generated_count, 2) if generation_elapsed and generated_count else None
    import_elapsed = seconds_between(timing.get("import_start"), timing.get("import_end"))
    embedding_elapsed = seconds_between(timing.get("embedding_start"), timing.get("embedding_end"))
    machine_review_elapsed = seconds_between(timing.get("machine_review_start"), timing.get("machine_review_end"))
    matrix_elapsed = seconds_between(timing.get("matrix_start"), timing.get("matrix_end"))
    per_image = {
        "generation_seconds": avg_seconds,
        "import_seconds": round(import_elapsed / generated_count, 2) if import_elapsed and generated_count else None,
        "embedding_seconds": round(embedding_elapsed / generated_count, 2) if embedding_elapsed and generated_count else None,
        "machine_review_seconds": round(machine_review_elapsed / generated_count, 2) if machine_review_elapsed and generated_count else None,
        "matrix_seconds": round(matrix_elapsed / generated_count, 2) if matrix_elapsed and generated_count else None,
    }
    machine_detail = timing.get("machine_review_detail") if isinstance(timing.get("machine_review_detail"), dict) else {}
    one_drive_paths = [path for path in model_paths or [] if is_onedrive_path(path)]
    if output_dir and is_onedrive_path(output_dir):
        one_drive_paths.append(output_dir)
    for candidate in (settings.RUNS_DIR, settings.EXPORTS_DIR, settings.EMBEDDINGS_DIR, settings.DATA_DIR):
        if is_onedrive_path(str(candidate)):
            one_drive_paths.append(str(candidate))
    embedding_runtime = embedding_runtime_summary()

    warnings: list[str] = []
    if generated_count and command_count >= generated_count:
        warnings.append("gen_img.pyの起動回数が生成画像数以上です。18枚で18回起動している場合、モデルロード待ちが支配的な可能性があります。")
    elif command_count:
        warnings.append("gen_img.pyは条件をまとめて起動されています。起動回数は概ねOKです。")
    if one_drive_paths:
        warnings.append("OneDrive配下のmodel/runs/outputを使用しています。大容量モデルや生成画像を同期対象に置くと遅くなる可能性があります。")

    return {
        "stage_timing": timing,
        "total_elapsed_seconds": seconds_between(timing.get("pipeline_start"), timing.get("pipeline_end")) or row_elapsed(row),
        "generation_elapsed_seconds": generation_elapsed,
        "import_elapsed_seconds": import_elapsed,
        "embedding_elapsed_seconds": embedding_elapsed,
        "machine_review_elapsed_seconds": machine_review_elapsed,
        "matrix_elapsed_seconds": matrix_elapsed,
        "generation_process_count": command_count,
        "generated_output_count": generated_count,
        "avg_seconds_per_image": avg_seconds,
        "per_image_seconds": per_image,
        "machine_review_detail": machine_detail,
        "estimated_model_load_overhead_seconds": model_load_overhead,
        "commands": commands,
        "warnings": warnings,
        "one_drive_paths": sorted(set(one_drive_paths)),
        "embedding_runtime": embedding_runtime,
    }


def format_seconds(value: Any) -> str:
    if value in (None, ""):
        return "-"
    try:
        seconds = int(round(float(value)))
    except (TypeError, ValueError):
        return "-"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def historical_training_seconds_per_step(limit: int = 10) -> float | None:
    rows = fetch_all(
        """
        SELECT elapsed_seconds, COALESCE(actual_max_step, expected_total_steps, expected_total_steps_at_creation) AS steps
        FROM training_jobs
        WHERE status = 'completed'
          AND elapsed_seconds IS NOT NULL
          AND elapsed_seconds > 0
          AND COALESCE(actual_max_step, expected_total_steps, expected_total_steps_at_creation) > 0
        ORDER BY end_time DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )
    values = [float(row["elapsed_seconds"]) / max(1, float(row["steps"])) for row in rows]
    return round(sum(values) / len(values), 4) if values else None


def historical_validation_seconds_per_image(limit: int = 10) -> dict[str, Any]:
    rows = fetch_all(
        """
        SELECT elapsed_seconds, generated_image_count
        FROM validation_generation_runs
        WHERE status = 'completed'
          AND elapsed_seconds IS NOT NULL
          AND elapsed_seconds > 0
          AND generated_image_count > 0
        ORDER BY ended_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )
    values = [float(row["elapsed_seconds"]) / max(1, float(row["generated_image_count"])) for row in rows]
    return {"seconds_per_image": round(sum(values) / len(values), 2) if values else None, "sample_count": len(values)}


def training_duration_estimate(total_steps: int | None) -> dict[str, Any]:
    seconds_per_step = historical_training_seconds_per_step()
    if not total_steps or not seconds_per_step:
        return {"seconds_per_step": seconds_per_step, "estimated_seconds": None, "basis": "過去の完了Job実績が不足しています。"}
    estimated = int(round(float(total_steps) * seconds_per_step))
    return {
        "seconds_per_step": seconds_per_step,
        "estimated_seconds": estimated,
        "estimated_label": format_seconds(estimated),
        "basis": f"直近完了Jobの平均 {seconds_per_step} 秒/step から概算しています。",
    }


def validation_duration_estimate(image_count: int | None) -> dict[str, Any]:
    stats = historical_validation_seconds_per_image()
    seconds_per_image = stats["seconds_per_image"]
    if not image_count or not seconds_per_image:
        return {**stats, "estimated_seconds": None, "basis": "過去の検証画像生成実績が不足しています。"}
    estimated = int(round(float(image_count) * seconds_per_image))
    return {
        **stats,
        "estimated_seconds": estimated,
        "estimated_label": format_seconds(estimated),
        "basis": f"直近検証生成の平均 {seconds_per_image} 秒/枚から概算しています。",
    }


def update_timing(table_name: str, row_id: int, mutate: Callable[[dict[str, Any]], None]) -> None:
    row = fetch_one(f"SELECT stage_timing_json FROM {table_name} WHERE id = ?", (row_id,))
    data = json_loads(row["stage_timing_json"] if row else None, {})
    mutate(data)
    save_timing(table_name, row_id, data)


def save_timing(table_name: str, row_id: int, data: dict[str, Any]) -> None:
    with connect() as conn:
        conn.execute(
            f"UPDATE {table_name} SET stage_timing_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), utc_now(), row_id),
        )


def ensure_command(data: dict[str, Any], command_index: int) -> dict[str, Any]:
    commands = data.setdefault("commands", [])
    while len(commands) <= command_index:
        commands.append({"command_index": len(commands)})
    command = commands[command_index]
    command.setdefault("command_index", command_index)
    return command


def lora_file_from_command(command: dict[str, Any]) -> str:
    argv = [str(part) for part in command.get("argv") or []]
    for index, part in enumerate(argv):
        if part == "--network_weights" and index + 1 < len(argv):
            return argv[index + 1]
    return ""


def image_mtime_stats(output_dir: Path, start_ts: float | None, end_ts: float | None) -> dict[str, Any]:
    paths: list[Path] = []
    if output_dir.exists():
        for path in output_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if start_ts is not None and mtime + 1 < start_ts:
                continue
            if end_ts is not None and mtime - 1 > end_ts:
                continue
            paths.append(path)
    mtimes = sorted(path.stat().st_mtime for path in paths)
    return {
        "first_output_detected_at": iso_from_timestamp(mtimes[0]) if mtimes else None,
        "last_output_detected_at": iso_from_timestamp(mtimes[-1]) if mtimes else None,
        "output_count": len(paths),
    }


def parse_timestamp(value: Any) -> float | None:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError):
        return None


def iso_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def seconds_between(start: Any, end: Any) -> int | None:
    start_ts = parse_timestamp(start)
    end_ts = parse_timestamp(end)
    if start_ts is None or end_ts is None:
        return None
    return max(0, int(round(end_ts - start_ts)))


def row_elapsed(row: Any) -> int | None:
    if "elapsed_seconds" in row.keys() and row["elapsed_seconds"] is not None:
        return int(row["elapsed_seconds"])
    if "started_at" in row.keys() and "ended_at" in row.keys():
        return seconds_between(row["started_at"], row["ended_at"])
    return None


def is_onedrive_path(value: str) -> bool:
    lowered = str(value or "").lower()
    return "onedrive" in lowered or "onlinestrage" in lowered


def json_loads(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, type(default)) else default
    except (TypeError, json.JSONDecodeError):
        return default


def empty_summary() -> dict[str, Any]:
    return {
        "stage_timing": {},
        "total_elapsed_seconds": None,
        "generation_elapsed_seconds": None,
        "import_elapsed_seconds": None,
        "embedding_elapsed_seconds": None,
        "machine_review_elapsed_seconds": None,
        "matrix_elapsed_seconds": None,
        "generation_process_count": 0,
        "generated_output_count": 0,
        "avg_seconds_per_image": None,
        "estimated_model_load_overhead_seconds": None,
        "commands": [],
        "warnings": [],
        "one_drive_paths": [],
        "embedding_runtime": {},
    }


def embedding_runtime_summary() -> dict[str, Any]:
    try:
        settings_row = load_embedding_settings()
        model = active_embedding_model()
        return {
            "provider": model.get("provider") or "-",
            "model_id": model.get("id") or "-",
            "device": settings_row.get("device") or "auto",
            "dtype": settings_row.get("dtype") or "fp32",
            "batch_size": int(settings_row.get("batch_size") or 1),
        }
    except Exception:
        return {}
