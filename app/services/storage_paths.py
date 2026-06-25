from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app import settings
from app.db import connect, fetch_all, fetch_one, utc_now


RUNTIME_KEYS = {
    "runtime_root",
    "runs_root",
    "exports_root",
    "embedding_cache_root",
    "reports_root",
    "trash_root",
    "logs_root",
}


def default_runtime_root() -> Path:
    return settings.ROOT_DIR


def active_storage_locations() -> dict[str, str]:
    rows = fetch_all("SELECT key, path FROM storage_locations WHERE is_active = 1")
    return {str(row["key"]): str(row["path"]) for row in rows}


def runtime_root() -> Path:
    value = active_storage_locations().get("runtime_root", "").strip()
    return Path(value) if value else default_runtime_root()


def runs_root() -> Path:
    return configured_child("runs_root", "runs")


def exports_root() -> Path:
    return configured_child("exports_root", "exports")


def embedding_cache_root() -> Path:
    return configured_child("embedding_cache_root", "data/embeddings")


def reports_root() -> Path:
    return configured_child("reports_root", "reports")


def trash_root() -> Path:
    return configured_child("trash_root", "trash")


def logs_root() -> Path:
    return configured_child("logs_root", "logs")


def configured_child(key: str, relative: str) -> Path:
    locations = active_storage_locations()
    explicit = locations.get(key, "").strip()
    if explicit:
        return Path(explicit)
    root = locations.get("runtime_root", "").strip()
    if root:
        return Path(root) / Path(relative)
    if key == "runs_root":
        return settings.RUNS_DIR
    if key == "exports_root":
        return settings.EXPORTS_DIR
    if key == "embedding_cache_root":
        return settings.EMBEDDINGS_DIR
    if key == "logs_root":
        return settings.LOGS_DIR
    if key == "reports_root":
        return settings.ROOT_DIR / "reports"
    if key == "trash_root":
        return settings.ROOT_DIR / "trash"
    return settings.ROOT_DIR / relative


def ensure_runtime_dirs() -> None:
    for path in [runs_root(), exports_root(), embedding_cache_root(), reports_root(), trash_root(), logs_root()]:
        path.mkdir(parents=True, exist_ok=True)


def is_onedrive_path(value: str | Path) -> bool:
    text = str(value).lower()
    return "onedrive" in text or "onlinestrage" in text


def validate_storage_path(path_value: str) -> dict[str, Any]:
    path_text = (path_value or "").strip()
    if not path_text:
        return {"status": "error", "message": "pathが空です。"}
    path = Path(path_text)
    warnings: list[str] = []
    if not path.is_absolute():
        warnings.append("絶対パスを推奨します。")
        path = (settings.ROOT_DIR / path).resolve()
    if str(path) in {str(path.anchor), str(Path.home())}:
        warnings.append("rootやユーザーホーム直下は避けてください。")
    if path.exists() and not path.is_dir():
        return {"status": "error", "message": "既存ファイルが指定されています。ディレクトリを指定してください。", "path": str(path)}
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".lora_studio_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception as exc:
        return {"status": "error", "message": f"書き込み確認に失敗しました: {exc}", "path": str(path)}
    if is_onedrive_path(path):
        warnings.append("OneDrive配下です。大量I/Oでは同期負荷が出る可能性があります。")
    return {"status": "warning" if warnings else "ok", "message": " / ".join(warnings) if warnings else "OK", "path": str(path)}


def set_storage_location(key: str, path_value: str) -> dict[str, Any]:
    if key not in RUNTIME_KEYS:
        raise ValueError(f"Unsupported storage key: {key}")
    result = validate_storage_path(path_value)
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO storage_locations(key, path, is_active, created_at, updated_at, last_checked_at, check_status, error_message)
            VALUES (?, ?, 1, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                path = excluded.path,
                is_active = 1,
                updated_at = excluded.updated_at,
                last_checked_at = excluded.last_checked_at,
                check_status = excluded.check_status,
                error_message = excluded.error_message
            """,
            (key, result.get("path") or path_value, now, now, now, result["status"], result["message"]),
        )
    ensure_runtime_dirs()
    return result


def reset_storage_location(key: str = "runtime_root") -> None:
    with connect() as conn:
        conn.execute("DELETE FROM storage_locations WHERE key = ?", (key,))


def storage_status() -> dict[str, Any]:
    ensure_runtime_dirs()
    roots = {
        "app_root": str(settings.ROOT_DIR),
        "runtime_root": str(runtime_root()),
        "runs_root": str(runs_root()),
        "exports_root": str(exports_root()),
        "embedding_cache_root": str(embedding_cache_root()),
        "reports_root": str(reports_root()),
        "trash_root": str(trash_root()),
        "logs_root": str(logs_root()),
    }
    warnings = []
    for label, path in roots.items():
        if is_onedrive_path(path):
            warnings.append(f"{label} がOneDrive配下です。")
    return {
        "roots": roots,
        "locations": active_storage_locations(),
        "warnings": warnings,
        "onedrive_warning": " ".join(warnings),
    }


def allowed_serving_roots() -> list[Path]:
    return [
        settings.ROOT_DIR,
        settings.RUNS_DIR,
        settings.EXPORTS_DIR,
        settings.DATA_DIR,
        runs_root(),
        exports_root(),
        embedding_cache_root(),
        reports_root(),
        trash_root(),
        logs_root(),
    ]

