from __future__ import annotations

import subprocess
from dataclasses import dataclass

APP_VERSION = "v0.5.1-beta"
DB_SCHEMA_VERSION = "2026.06.12-beta"


@dataclass(frozen=True)
class AppVersionInfo:
    app_version: str
    git_commit: str
    db_schema_version: str


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except Exception:
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def app_version_info(db_schema_version: str | None = None) -> AppVersionInfo:
    return AppVersionInfo(
        app_version=APP_VERSION,
        git_commit=get_git_commit(),
        db_schema_version=db_schema_version or DB_SCHEMA_VERSION,
    )
