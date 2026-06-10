from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import settings
from app.services.preset_seed import preset_rows


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA_SQL)
        seed_app_settings(conn)
        seed_presets(conn)


def seed_app_settings(conn: sqlite3.Connection) -> None:
    now = utc_now()
    values = {
        "app_name": settings.APP_NAME,
        "sd_scripts_release_tag": settings.SD_SCRIPTS_RELEASE_TAG,
        "sd_scripts_release_commit": settings.SD_SCRIPTS_RELEASE_COMMIT,
        "sd_scripts_repo_url": settings.SD_SCRIPTS_REPO_URL,
    }
    conn.executemany(
        """
        INSERT INTO app_settings(key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        [(key, value, now) for key, value in values.items()],
    )


def seed_presets(conn: sqlite3.Connection) -> None:
    conn.executemany(
        """
        INSERT INTO presets(
            id, name, model_family, training_script, purpose, params_json,
            recommended_dataset_json, expected_behavior, risk_note, source_basis,
            is_builtin, parent_preset_id, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name, model_family=excluded.model_family,
            training_script=excluded.training_script, purpose=excluded.purpose,
            params_json=excluded.params_json,
            recommended_dataset_json=excluded.recommended_dataset_json,
            expected_behavior=excluded.expected_behavior, risk_note=excluded.risk_note,
            source_basis=excluded.source_basis, is_builtin=excluded.is_builtin,
            updated_at=excluded.updated_at
        """,
        list(preset_rows(utc_now())),
    )


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with connect() as conn:
        return list(conn.execute(query, params).fetchall())


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(query, params).fetchone()


def insert_dataset(name: str, path: str, model_family: str, trigger_word: str, class_token: str, memo: str) -> int:
    from app.services.dataset_scanner import scan_dataset

    now = utc_now()
    scan = scan_dataset(Path(path))
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO datasets(
                name, path, model_family, trigger_word, class_token, image_count,
                caption_count, missing_caption_count, resolution_summary_json,
                tag_summary_json, scan_status, memo, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name, path, model_family, trigger_word, class_token,
                scan["image_count"], scan["caption_count"], scan["missing_caption_count"],
                json.dumps(scan["resolution_summary"], ensure_ascii=False),
                json.dumps(scan["tag_summary"], ensure_ascii=False),
                scan["status"], memo, now, now,
            ),
        )
        return int(cur.lastrowid)


def create_job(data: dict[str, Any]) -> int:
    now = utc_now()
    preset = fetch_one("SELECT * FROM presets WHERE id = ?", (data["preset_id"],))
    if preset is None:
        raise ValueError(f"Preset not found: {data['preset_id']}")
    params = json.loads(preset["params_json"])
    output_name = data.get("output_name") or data["name"].replace(" ", "_")
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO training_jobs(
                name, dataset_id, preset_id, environment_id, status, model_family,
                training_script, base_model_path, vae_path, output_name, output_dir,
                run_dir, params_json, memo, created_at, updated_at
            ) VALUES (?, ?, ?, NULL, 'draft', ?, ?, ?, ?, ?, '', '', ?, ?, ?, ?)
            """,
            (
                data["name"], int(data["dataset_id"]), data["preset_id"],
                preset["model_family"], preset["training_script"], data["base_model_path"],
                data.get("vae_path") or None, output_name,
                json.dumps(params, ensure_ascii=False, indent=2), data.get("memo") or "", now, now,
            ),
        )
        job_id = int(cur.lastrowid)
        run_dir = settings.RUNS_DIR / f"job_{job_id:06d}"
        output_dir = run_dir / "models"
        for subdir in ("config", "logs", "models", "samples", "metrics", "exports"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
        conn.execute("UPDATE training_jobs SET run_dir = ?, output_dir = ?, updated_at = ? WHERE id = ?", (str(run_dir), str(output_dir), now, job_id))
        return job_id


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS environments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, sd_scripts_path TEXT NOT NULL,
    venv_python_path TEXT NOT NULL, venv_accelerate_path TEXT, hf_home TEXT, cuda_profile TEXT,
    mixed_precision TEXT, python_version TEXT, torch_version TEXT, torch_cuda_version TEXT,
    cuda_available INTEGER, gpu_name TEXT, sd_scripts_commit_hash TEXT, requirements_hash TEXT,
    status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS environment_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER, snapshot_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS presets (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, model_family TEXT NOT NULL, training_script TEXT NOT NULL,
    purpose TEXT NOT NULL, params_json TEXT NOT NULL, recommended_dataset_json TEXT,
    expected_behavior TEXT, risk_note TEXT, source_basis TEXT, is_builtin INTEGER NOT NULL DEFAULT 0,
    parent_preset_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, path TEXT NOT NULL, model_family TEXT,
    trigger_word TEXT, class_token TEXT, image_count INTEGER, caption_count INTEGER,
    missing_caption_count INTEGER, resolution_summary_json TEXT, tag_summary_json TEXT, scan_status TEXT,
    memo TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, dataset_id INTEGER, preset_id TEXT,
    environment_id INTEGER, status TEXT NOT NULL, model_family TEXT NOT NULL, training_script TEXT NOT NULL,
    base_model_path TEXT NOT NULL, vae_path TEXT, output_name TEXT NOT NULL, output_dir TEXT NOT NULL,
    run_dir TEXT NOT NULL, params_json TEXT NOT NULL, command_line TEXT, process_id INTEGER,
    return_code INTEGER, start_time TEXT, end_time TEXT, elapsed_seconds INTEGER, adopted_epoch INTEGER,
    adopted_model_path TEXT, image_rating INTEGER, loss_health_label TEXT, memo TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, step INTEGER, epoch REAL, loss REAL,
    lr REAL, unet_lr REAL, text_encoder_lr REAL, raw_json TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_metric_summaries (
    job_id INTEGER PRIMARY KEY, initial_loss REAL, final_loss REAL, min_loss REAL, min_loss_epoch REAL,
    loss_drop_rate REAL, loss_volatility REAL, spike_count INTEGER, late_stage_slope REAL,
    health_label TEXT, health_score INTEGER, summary_json TEXT, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sample_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER, name TEXT NOT NULL, prompt TEXT NOT NULL,
    negative_prompt TEXT, width INTEGER, height INTEGER, seed INTEGER, cfg_scale REAL, steps INTEGER,
    sort_order INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sample_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, prompt_id INTEGER, epoch INTEGER,
    step INTEGER, image_path TEXT NOT NULL, prompt TEXT, negative_prompt TEXT, seed INTEGER,
    width INTEGER, height INTEGER, cfg_scale REAL, steps INTEGER, rating INTEGER, memo TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, epoch INTEGER, step INTEGER,
    file_path TEXT NOT NULL, file_type TEXT NOT NULL, file_size INTEGER, sha256 TEXT,
    selected INTEGER NOT NULL DEFAULT 0, memo TEXT, created_at TEXT NOT NULL
);
"""
