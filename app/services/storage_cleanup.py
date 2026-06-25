from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app import settings
from app.db import connect, fetch_all, fetch_one, utc_now
from app.services.output_collector import safe_sha256_file
from app.services.storage_paths import exports_root as runtime_exports_root, runs_root as runtime_runs_root, storage_status

MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
REPORT_CONFIG_LOG_DIRS = {"reports", "logs", "config", "metrics"}


@dataclass
class CleanupFile:
    id: int
    path: str
    kind: str
    selected: bool = False
    size: int = 0
    exists: bool = False
    sha256: str | None = None


def storage_root_warning() -> str:
    paths = [str(settings.ROOT_DIR), str(settings.RUNS_DIR), str(runtime_runs_root()), str(runtime_exports_root())]
    if any("onedrive" in path.lower() for path in paths):
        return "このフォルダはOneDrive配下です。削除するとクラウド側にも反映される可能性があります。大容量のrunsをOneDrive配下に置くと同期負荷が高くなる可能性があります。"
    return ""


def format_bytes(size: int | None) -> str:
    value = float(size or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return "0 B"


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def file_size(path_value: str | None) -> int:
    if not path_value:
        return 0
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def storage_usage() -> dict[str, Any]:
    jobs = job_storage_rows()
    review_session_size = review_session_images_size()
    totals = {
        "runs": path_size(runtime_runs_root()),
        "exports": path_size(runtime_exports_root()),
        "backups": path_size(settings.ROOT_DIR / "backups"),
        "trash": path_size(trash_root()),
        "model_outputs": sum(job["model_size"] for job in jobs),
        "sample_images": sum(job["sample_size"] for job in jobs),
        "review_session_images": review_session_size,
        "reports_logs_config": sum(job["support_size"] for job in jobs),
    }
    return {
        "totals": {key: {"bytes": value, "label": format_bytes(value)} for key, value in totals.items()},
        "jobs": jobs,
        "trash": trash_entries(),
        "onedrive_warning": storage_status().get("onedrive_warning") or storage_root_warning(),
    }


def job_storage_rows(project_id: int | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT j.*, p.name AS preset_name,
               EXISTS(SELECT 1 FROM training_outputs o WHERE o.job_id = j.id AND o.selected = 1 AND o.deleted_at IS NULL) AS has_selected_output,
               EXISTS(SELECT 1 FROM training_outputs o WHERE o.job_id = j.id AND o.external_copy_path IS NOT NULL) AS has_exported_output
        FROM training_jobs j
        LEFT JOIN presets p ON p.id = j.preset_id
        WHERE j.deleted_at IS NULL
    """
    params: list[Any] = []
    if project_id is not None:
        sql += " AND j.project_id = ?"
        params.append(project_id)
    sql += " ORDER BY j.id DESC"
    rows = fetch_all(sql, tuple(params))
    result = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        model_size = sum(file_size(item["file_path"]) for item in fetch_all("SELECT file_path FROM training_outputs WHERE job_id = ? AND file_type = 'model' AND deleted_at IS NULL", (row["id"],)))
        sample_size = sum(file_size(item["image_path"]) for item in fetch_all("SELECT image_path FROM sample_images WHERE job_id = ? AND deleted_at IS NULL", (row["id"],)))
        support_size = sum(path_size(run_dir / name) for name in REPORT_CONFIG_LOG_DIRS)
        cleanup_candidate = is_storage_cleanup_candidate(row)
        result.append(
            {
                **dict(row),
                "model_size": model_size,
                "sample_size": sample_size,
                "support_size": support_size,
                "total_size": model_size + sample_size + support_size,
                "model_size_label": format_bytes(model_size),
                "sample_size_label": format_bytes(sample_size),
                "support_size_label": format_bytes(support_size),
                "total_size_label": format_bytes(model_size + sample_size + support_size),
                "has_selected_output": bool(row["has_selected_output"]),
                "has_exported_output": bool(row["has_exported_output"]),
                "cleanup_candidate": cleanup_candidate,
                "cleanup_candidate_size": cleanup_candidate_size(row["id"]),
                "cleanup_candidate_size_label": format_bytes(cleanup_candidate_size(row["id"])),
            }
        )
    return result


def review_session_images_size(project_id: int | None = None) -> int:
    sql = """
        SELECT rsi.image_path
        FROM review_session_images rsi
        JOIN review_sessions rs ON rs.id = rsi.review_session_id
        WHERE rsi.deleted_at IS NULL
    """
    params: list[Any] = []
    if project_id is not None:
        sql += " AND rs.project_id = ?"
        params.append(project_id)
    return sum(file_size(row["image_path"]) for row in fetch_all(sql, tuple(params)))


def is_storage_cleanup_candidate(job: Any) -> bool:
    if job["status"] in {"failed", "stopped"}:
        selected = fetch_one("SELECT 1 FROM training_outputs WHERE job_id = ? AND selected = 1 AND deleted_at IS NULL LIMIT 1", (job["id"],))
        output = fetch_one("SELECT 1 FROM training_outputs WHERE job_id = ? AND file_type = 'model' AND deleted_at IS NULL LIMIT 1", (job["id"],))
        return selected is None and output is not None
    unselected = fetch_one("SELECT 1 FROM training_outputs WHERE job_id = ? AND file_type = 'model' AND selected = 0 AND deleted_at IS NULL LIMIT 1", (job["id"],))
    return unselected is not None


def cleanup_candidate_size(job_id: int) -> int:
    rows = fetch_all("SELECT file_path FROM training_outputs WHERE job_id = ? AND file_type = 'model' AND selected = 0 AND deleted_at IS NULL", (job_id,))
    return sum(file_size(row["file_path"]) for row in rows)


def project_storage_summary(project_id: int) -> dict[str, Any]:
    jobs = job_storage_rows(project_id)
    model_size = sum(job["model_size"] for job in jobs)
    sample_size = sum(job["sample_size"] for job in jobs)
    review_session_size = review_session_images_size(project_id)
    support_size = sum(job["support_size"] for job in jobs)
    cleanup_size = sum(job["cleanup_candidate_size"] for job in jobs)
    return {
        "jobs": jobs,
        "total_size": model_size + sample_size + review_session_size + support_size,
        "model_size": model_size,
        "sample_size": sample_size,
        "review_session_size": review_session_size,
        "support_size": support_size,
        "cleanup_candidate_size": cleanup_size,
        "total_size_label": format_bytes(model_size + sample_size + review_session_size + support_size),
        "model_size_label": format_bytes(model_size),
        "sample_size_label": format_bytes(sample_size),
        "review_session_size_label": format_bytes(review_session_size),
        "support_size_label": format_bytes(support_size),
        "cleanup_candidate_size_label": format_bytes(cleanup_size),
        "onedrive_warning": storage_root_warning(),
    }


def unselected_model_preview(job_id: int, include_selected: bool = False) -> dict[str, Any]:
    where_selected = "" if include_selected else "AND selected = 0"
    rows = fetch_all(
        f"""
        SELECT * FROM training_outputs
        WHERE job_id = ? AND file_type = 'model' {where_selected}
          AND deleted_at IS NULL
        ORDER BY selected, epoch, step, id
        """,
        (job_id,),
    )
    files = [output_file(row) for row in rows]
    selected_count = int(fetch_one("SELECT COUNT(*) AS count FROM training_outputs WHERE job_id = ? AND selected = 1 AND deleted_at IS NULL", (job_id,))["count"])
    return preview_payload(job_id, "unselected_models", files, "未採用LoRAファイル削除Preview", selected_count)


def failed_outputs_preview(job_id: int) -> dict[str, Any]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None or job["status"] not in {"failed", "stopped"}:
        files: list[CleanupFile] = []
    else:
        has_selected = fetch_one("SELECT 1 FROM training_outputs WHERE job_id = ? AND selected = 1 AND deleted_at IS NULL LIMIT 1", (job_id,))
        files = [] if has_selected else [output_file(row) for row in fetch_all("SELECT * FROM training_outputs WHERE job_id = ? AND file_type = 'model' AND deleted_at IS NULL ORDER BY epoch, id", (job_id,))]
    return preview_payload(job_id, "failed_outputs", files, "失敗/停止ジョブ出力削除Preview", 0)


def exported_selected_preview(job_id: int) -> dict[str, Any]:
    row = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1 AND deleted_at IS NULL", (job_id,))
    files = [output_file(row)] if row else []
    payload = preview_payload(job_id, "exported_selected", files, "Export済み採用LoRA runs側削除Preview", 1 if row else 0)
    payload["can_execute"] = bool(row and selected_export_verified(row))
    payload["blocked_reason"] = "" if payload["can_execute"] else "export先の存在とsha256一致が確認済みの場合のみ削除できます。"
    return payload


def sample_cleanup_preview(job_id: int, action: str = "delete_individual") -> dict[str, Any]:
    rows = fetch_all("SELECT * FROM sample_images WHERE job_id = ? AND deleted_at IS NULL ORDER BY epoch, prompt_id, id", (job_id,))
    files = [sample_file(row) for row in rows]
    payload = preview_payload(job_id, action, files, "Sample Cleanup Preview", 0)
    payload["contact_sheet"] = contact_sheet_path(job_id)
    payload["contact_sheet_exists"] = bool(payload["contact_sheet"] and Path(payload["contact_sheet"]).exists())
    return payload


def preview_payload(job_id: int, action: str, files: list[CleanupFile], title: str, selected_count: int) -> dict[str, Any]:
    total_size = sum(item.size for item in files if item.exists)
    return {
        "job": fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,)),
        "action": action,
        "title": title,
        "files": files,
        "total_size": total_size,
        "total_size_label": format_bytes(total_size),
        "selected_count": selected_count,
        "onedrive_warning": storage_root_warning(),
        "can_execute": True,
        "blocked_reason": "",
    }


def output_file(row: Any) -> CleanupFile:
    path = Path(row["file_path"])
    exists = path.exists()
    size = file_size(row["file_path"])
    return CleanupFile(id=int(row["id"]), path=str(path), kind="model", selected=bool(row["selected"]), size=size, exists=exists, sha256=row["sha256"])


def sample_file(row: Any) -> CleanupFile:
    path = Path(row["image_path"])
    exists = path.exists()
    return CleanupFile(id=int(row["id"]), path=str(path), kind="sample", selected=False, size=file_size(row["image_path"]), exists=exists, sha256=None)


def selected_export_verified(row: Any) -> bool:
    export_path = row["external_copy_path"]
    if not export_path or not row["export_verified_at"]:
        return False
    source_sha = row["sha256"]
    export_file = Path(export_path)
    if not source_sha or not export_file.exists():
        return False
    export_sha, _ = safe_sha256_file(export_file)
    return export_sha == source_sha


def cleanup_outputs(job_id: int, mode: str) -> dict[str, Any]:
    if mode == "unselected_models":
        preview = unselected_model_preview(job_id)
    elif mode == "failed_outputs":
        preview = failed_outputs_preview(job_id)
    elif mode == "exported_selected":
        preview = exported_selected_preview(job_id)
        if not preview["can_execute"]:
            raise ValueError(preview["blocked_reason"])
    else:
        raise ValueError(f"Unknown cleanup mode: {mode}")
    moved = []
    now = utc_now()
    for item in preview["files"]:
        if not item.exists:
            mark_output_missing(item.id, "missing during cleanup")
            continue
        trash_path = move_file_to_trash(Path(item.path), job_id, None, mode)
        record_history(job_id, None, "move_to_trash", item.path, str(trash_path), item.size, item.sha256, mode)
        with connect() as conn:
            conn.execute(
                """
                UPDATE training_outputs
                SET deleted_at = ?, cleanup_status = 'deleted', delete_reason = ?
                WHERE id = ?
                """,
                (now, mode, item.id),
            )
            if mode == "exported_selected":
                output = fetch_one("SELECT external_copy_path FROM training_outputs WHERE id = ?", (item.id,))
                if output and output["external_copy_path"]:
                    conn.execute(
                        """
                        UPDATE training_jobs
                        SET adopted_model_path = CASE WHEN adopted_model_path = ? THEN ? ELSE adopted_model_path END,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (item.path, output["external_copy_path"], now, job_id),
                    )
        moved.append(trash_path)
    update_job_counts(job_id)
    return {"moved": len(moved), "bytes": preview["total_size"], "bytes_label": preview["total_size_label"]}


def cleanup_samples(job_id: int, action: str = "delete_individual") -> dict[str, Any]:
    if action not in {"delete_individual", "delete_all"}:
        raise ValueError(f"Unknown sample cleanup action: {action}")
    preview = sample_cleanup_preview(job_id, action)
    moved = []
    now = utc_now()
    for item in preview["files"]:
        if item.exists:
            trash_path = move_file_to_trash(Path(item.path), job_id, None, action)
            record_history(job_id, None, "move_to_trash", item.path, str(trash_path), item.size, item.sha256, action)
            moved.append(trash_path)
        with connect() as conn:
            conn.execute(
                "UPDATE sample_images SET deleted_at = ?, cleanup_status = ? WHERE id = ?",
                (now, "deleted" if item.exists else "missing", item.id),
            )
    update_job_counts(job_id)
    return {"moved": len(moved), "bytes": preview["total_size"], "bytes_label": preview["total_size_label"]}


def mark_output_missing(output_id: int, reason: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE training_outputs SET cleanup_status = 'missing', delete_reason = COALESCE(delete_reason, ?) WHERE id = ?",
            (reason, output_id),
        )


def move_file_to_trash(path: Path, job_id: int | None, project_id: int | None, action: str) -> Path:
    destination_dir = trash_root() / timestamp() / (f"job_{job_id:06d}" if job_id else "files") / action
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = unique_destination(destination_dir / path.name)
    shutil.move(str(path), str(destination))
    return destination


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique trash path for {path}")


def record_history(job_id: int | None, project_id: int | None, action: str, original_path: str, trash_path: str | None, file_size_value: int, sha256: str | None, memo: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO file_cleanup_history(job_id, project_id, action, original_path, trash_path, file_size, sha256, created_at, memo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, project_id, action, original_path, trash_path, file_size_value, sha256, utc_now(), memo),
        )


def update_job_counts(job_id: int) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET output_model_count = (SELECT COUNT(*) FROM training_outputs WHERE job_id = ? AND file_type = 'model' AND deleted_at IS NULL),
                sample_image_count = (SELECT COUNT(*) FROM sample_images WHERE job_id = ? AND deleted_at IS NULL),
                updated_at = ?
            WHERE id = ?
            """,
            (job_id, job_id, utc_now(), job_id),
        )


def trash_root() -> Path:
    root = settings.ROOT_DIR / "trash" / "files"
    root.mkdir(parents=True, exist_ok=True)
    return root


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def trash_entries() -> list[dict[str, Any]]:
    trash_root()
    history = fetch_all("SELECT * FROM file_cleanup_history ORDER BY id DESC LIMIT 100")
    entries: list[dict[str, Any]] = []
    for row in history:
        trash_path = row["trash_path"]
        trash_exists = bool(trash_path and Path(trash_path).exists())
        if not trash_exists:
            continue
        entries.append(
            {
                **dict(row),
                "file_size_label": format_bytes(row["file_size"]),
                "trash_exists": trash_exists,
            }
        )
    return entries


def purge_trash() -> int:
    root = trash_root()
    size = path_size(root)
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
    return size


def contact_sheet_path(job_id: int) -> str:
    path = settings.RUNS_DIR / f"job_{job_id:06d}" / "reports" / f"contact_sheet_job_{job_id:06d}.html"
    return str(path) if path.exists() else ""


def project_cleanup_preview(project_id: int, mode: str) -> dict[str, Any]:
    if mode not in {"unselected_models", "failed_outputs"}:
        raise ValueError(f"Unknown project cleanup mode: {mode}")
    files: list[CleanupFile] = []
    jobs = fetch_all("SELECT id FROM training_jobs WHERE project_id = ? AND deleted_at IS NULL", (project_id,))
    for job in jobs:
        preview = failed_outputs_preview(job["id"]) if mode == "failed_outputs" else unselected_model_preview(job["id"])
        files.extend(preview["files"])
    project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (project_id,))
    total = sum(item.size for item in files if item.exists)
    return {
        "project": project,
        "mode": mode,
        "title": "Project Cleanup Preview",
        "files": files,
        "total_size": total,
        "total_size_label": format_bytes(total),
        "onedrive_warning": storage_root_warning(),
    }


def cleanup_project_outputs(project_id: int, mode: str) -> dict[str, Any]:
    preview = project_cleanup_preview(project_id, mode)
    moved = 0
    for item in preview["files"]:
        if not item.exists:
            mark_output_missing(item.id, "missing during project cleanup")
            continue
        trash_path = move_file_to_trash(Path(item.path), None, project_id, mode)
        record_history(None, project_id, "move_to_trash", item.path, str(trash_path), item.size, item.sha256, f"project:{mode}")
        with connect() as conn:
            conn.execute(
                "UPDATE training_outputs SET deleted_at = ?, cleanup_status = 'deleted', delete_reason = ? WHERE id = ?",
                (utc_now(), f"project:{mode}", item.id),
            )
        moved += 1
    for row in fetch_all("SELECT id FROM training_jobs WHERE project_id = ?", (project_id,)):
        update_job_counts(row["id"])
    return {"moved": moved, "bytes": preview["total_size"], "bytes_label": preview["total_size_label"]}
