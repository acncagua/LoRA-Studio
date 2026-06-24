from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from app import settings
from app.db import connect, fetch_all, fetch_one, latest_environment, utc_now
from app.services.training_runner import sd_scripts_subprocess_env


def dependency_rows() -> list[Any]:
    return fetch_all("SELECT * FROM optional_optimizer_dependencies ORDER BY package_name")


def dependency_status_summary() -> dict[str, Any]:
    rows = dependency_rows()
    counts: dict[str, int] = {}
    for row in rows:
        status = row["status"] or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return {"rows": rows, "counts": counts}


def check_all_dependencies() -> list[dict[str, Any]]:
    return [check_dependency(row["id"]) for row in dependency_rows()]


def check_dependency(dependency_id: str) -> dict[str, Any]:
    row = fetch_one("SELECT * FROM optional_optimizer_dependencies WHERE id = ?", (dependency_id,))
    if row is None:
        raise ValueError(f"Optional optimizer dependency not found: {dependency_id}")
    python_path = _sd_scripts_python()
    module = row["import_check_module"]
    command = [str(python_path), "-c", f"import {module}; print(getattr({module}, '__file__', 'ok'))"]
    completed = subprocess.run(
        command,
        cwd=str(Path(python_path).parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=sd_scripts_subprocess_env(),
        check=False,
    )
    status = "installed" if completed.returncode == 0 else "missing"
    output = (completed.stdout or completed.stderr or "").strip()
    _update_dependency_status(dependency_id, status=status, error_message=None if status == "installed" else output, checked=True)
    return {"id": dependency_id, "status": status, "output": output, "return_code": completed.returncode}


def install_dependency(dependency_id: str) -> dict[str, Any]:
    row = fetch_one("SELECT * FROM optional_optimizer_dependencies WHERE id = ?", (dependency_id,))
    if row is None:
        raise ValueError(f"Optional optimizer dependency not found: {dependency_id}")
    python_path = _sd_scripts_python()
    install_args = json.loads(row["install_command_json"] or "[]")
    if not install_args:
        install_args = ["-m", "pip", "install", row["package_name"]]
    command = [str(python_path), *[str(item) for item in install_args]]
    log_dir = settings.ROOT_DIR / "logs" / "optimizer_dependencies"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{dependency_id}_{utc_now().replace(':', '').replace('+', '_')}.log"
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=sd_scripts_subprocess_env(),
        check=False,
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        message = f"install failed rc={completed.returncode}; log={log_path}"
        _update_dependency_status(dependency_id, status="install_failed", error_message=message, installed=True, checked=True)
        return {"id": dependency_id, "status": "install_failed", "return_code": completed.returncode, "log_path": str(log_path)}
    checked = check_dependency(dependency_id)
    if checked["status"] != "installed":
        message = f"install finished but import failed; log={log_path}; {checked.get('output') or ''}"
        _update_dependency_status(dependency_id, status="install_failed", error_message=message, installed=True, checked=True)
        return {"id": dependency_id, "status": "install_failed", "return_code": completed.returncode, "log_path": str(log_path)}
    with connect() as conn:
        conn.execute(
            "UPDATE optional_optimizer_dependencies SET last_install_at = ?, error_message = NULL, updated_at = ? WHERE id = ?",
            (utc_now(), utc_now(), dependency_id),
        )
    return {"id": dependency_id, "status": "installed", "return_code": completed.returncode, "log_path": str(log_path), "output": checked["output"]}


def install_all_missing_dependencies() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in dependency_rows():
        checked = check_dependency(row["id"])
        if checked["status"] != "installed":
            results.append(install_dependency(row["id"]))
        else:
            results.append(checked)
    return results


def _sd_scripts_python() -> Path:
    environment = latest_environment()
    if environment is None:
        raise RuntimeError("sd-scripts environment is not registered.")
    python_path = Path(environment["venv_python_path"])
    if not python_path.exists():
        raise RuntimeError(f"sd-scripts venv python does not exist: {python_path}")
    return python_path


def _update_dependency_status(
    dependency_id: str,
    *,
    status: str,
    error_message: str | None,
    checked: bool = False,
    installed: bool = False,
) -> None:
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE optional_optimizer_dependencies
            SET status = ?,
                last_checked_at = CASE WHEN ? THEN ? ELSE last_checked_at END,
                last_install_at = CASE WHEN ? THEN ? ELSE last_install_at END,
                error_message = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (status, 1 if checked else 0, now, 1 if installed else 0, now, error_message, now, dependency_id),
        )
