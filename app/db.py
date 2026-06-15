from __future__ import annotations

import json
import sqlite3
import subprocess
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from app.app_version import APP_VERSION, DB_SCHEMA_VERSION
from app import settings
from app.services.preset_seed import preset_rows


INTEGRATION_SMOKE_PRESET_ID = "integration_smoke_sdxl"
SMOKE_STEP_LIMIT_KEYS = {"max_train_steps", "save_every_n_steps", "sample_every_n_steps"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        result = super().__exit__(exc_type, exc_value, traceback)
        self.close()
        return result


def connect() -> sqlite3.Connection:
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH, factory=ClosingConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_session() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA_SQL)
        run_migrations(conn)
        seed_app_settings(conn)
        seed_presets(conn)
        seed_sample_prompt_templates(conn)
        seed_evaluation_rubrics(conn)
        seed_validation_presets(conn)
        seed_embedding_models(conn)
        import_latest_environment(conn)
    from app.services.validation_runs import backfill_validation_runs

    backfill_validation_runs()


def run_migrations(conn: sqlite3.Connection) -> None:
    ensure_columns(
        conn,
        "lora_projects",
        {
            "selected_lora_profile_id": "INTEGER",
            "recommended_weight_min": "REAL",
            "recommended_weight_max": "REAL",
            "default_reference_set_id": "INTEGER",
            "default_reference_set_version_id": "INTEGER",
            "memo": "TEXT",
            "archived_at": "TEXT",
            "deleted_at": "TEXT",
            "archive_reason": "TEXT",
            "delete_reason": "TEXT",
        },
    )
    ensure_columns(
        conn,
        "environments",
        {
            "venv_accelerate_path": "TEXT",
            "hf_home": "TEXT",
            "cuda_profile": "TEXT",
            "mixed_precision": "TEXT",
            "python_version": "TEXT",
            "torch_version": "TEXT",
            "torch_cuda_version": "TEXT",
            "cuda_available": "INTEGER",
            "gpu_name": "TEXT",
            "sd_scripts_commit_hash": "TEXT",
            "requirements_hash": "TEXT",
            "updated_at": "TEXT",
        },
    )
    ensure_columns(
        conn,
        "training_jobs",
        {
            "command_line": "TEXT",
            "process_id": "INTEGER",
            "return_code": "INTEGER",
            "start_time": "TEXT",
            "end_time": "TEXT",
            "elapsed_seconds": "INTEGER",
            "adopted_epoch": "INTEGER",
            "adopted_model_path": "TEXT",
            "image_rating": "INTEGER",
            "loss_health_label": "TEXT",
            "expected_total_steps": "INTEGER",
            "actual_max_step": "INTEGER",
            "actual_metric_count": "INTEGER",
            "output_model_count": "INTEGER",
            "sample_image_count": "INTEGER",
            "step_consistency_label": "TEXT",
            "step_consistency_message": "TEXT",
            "parent_job_id": "INTEGER",
            "sample_prompt_template_id": "TEXT",
            "config_dirty": "INTEGER NOT NULL DEFAULT 0",
            "trigger_word_at_creation": "TEXT",
            "trigger_occurrence_count_at_creation": "INTEGER",
            "trigger_occurrence_rate_at_creation": "REAL",
            "trigger_consistency_label_at_creation": "TEXT",
            "dataset_version_id": "INTEGER",
            "project_id": "INTEGER",
            "updated_at": "TEXT",
            "archived_at": "TEXT",
            "deleted_at": "TEXT",
            "archived_reason": "TEXT",
            "delete_reason": "TEXT",
        },
    )
    ensure_columns(
        conn,
        "training_metrics",
        {
            "learning_rate": "REAL",
            "source": "TEXT",
            "raw_tag": "TEXT",
        },
    )
    ensure_columns(
        conn,
        "training_metric_summaries",
        {
            "min_loss_step": "INTEGER",
            "max_loss": "REAL",
            "moving_avg_final_loss": "REAL",
            "raw_loss_label": "TEXT",
            "smoothed_loss_label": "TEXT",
            "epoch_trend_label": "TEXT",
            "health_message": "TEXT",
            "created_at": "TEXT",
        },
    )
    ensure_columns(
        conn,
        "training_outputs",
        {
            "selected": "INTEGER NOT NULL DEFAULT 0",
            "memo": "TEXT",
            "metadata_error": "TEXT",
            "deleted_at": "TEXT",
            "archived_at": "TEXT",
            "export_verified_at": "TEXT",
            "external_copy_path": "TEXT",
            "cleanup_status": "TEXT",
            "delete_reason": "TEXT",
        },
    )
    ensure_columns(
        conn,
        "sample_images",
        {
            "rating": "INTEGER",
            "rating_face": "INTEGER",
            "rating_costume": "INTEGER",
            "rating_style": "INTEGER",
            "rating_stability": "INTEGER",
            "rating_overall": "INTEGER",
            "strength_label": "TEXT",
            "overfit_level": "TEXT",
            "adoption_label": "TEXT",
            "failure_tags_json": "TEXT",
            "rubric_version": "TEXT",
            "memo": "TEXT",
            "deleted_at": "TEXT",
            "cleanup_status": "TEXT",
            "rating_flexibility": "INTEGER",
            "review_priority": "TEXT",
            "auto_review_label": "TEXT",
            "auto_review_reason": "TEXT",
        },
    )
    ensure_columns(conn, "sample_prompts", {"prompt_role": "TEXT"})
    for row in conn.execute("SELECT id, name, prompt FROM sample_prompts WHERE prompt_role IS NULL OR prompt_role = ''").fetchall():
        conn.execute(
            "UPDATE sample_prompts SET prompt_role = ? WHERE id = ?",
            (infer_prompt_role(row["name"], row["prompt"]), row["id"]),
        )
    ensure_columns(
        conn,
        "validation_images",
        {
            "expected_condition_id": "INTEGER",
            "validation_run_id": "INTEGER",
            "validation_preset_id": "TEXT",
            "prompt_key": "TEXT",
            "seed": "INTEGER",
            "lora_weight": "REAL",
            "grid_image_flag": "INTEGER NOT NULL DEFAULT 0",
            "image_role": "TEXT NOT NULL DEFAULT 'individual'",
            "condition_hash": "TEXT",
            "rating_flexibility": "INTEGER",
            "ignored": "INTEGER NOT NULL DEFAULT 0",
            "strength_label": "TEXT",
            "overfit_level": "TEXT",
            "adoption_label": "TEXT",
            "failure_tags_json": "TEXT",
            "rubric_version": "TEXT",
        },
    )
    ensure_columns(
        conn,
        "validation_weight_reviews",
        {
            "validation_run_id": "INTEGER",
            "validation_preset_id": "TEXT",
            "hires_enabled": "INTEGER NOT NULL DEFAULT 0",
            "rating_flexibility": "INTEGER",
            "strength_label": "TEXT",
            "overfit_level": "TEXT",
            "adoption_label": "TEXT",
            "failure_tags_json": "TEXT",
            "rubric_version": "TEXT",
        },
    )
    conn.execute("UPDATE sample_images SET rating_overall = rating WHERE rating_overall IS NULL AND rating IS NOT NULL")
    ensure_columns(
        conn,
        "selected_lora_profiles",
        {
            "default_validation_preset_id": "TEXT",
            "last_validation_preset_id": "TEXT",
            "validation_policy_memo": "TEXT",
            "reference_set_id": "INTEGER",
            "reference_set_version_id": "INTEGER",
            "project_id": "INTEGER",
        },
    )
    ensure_columns(
        conn,
        "validation_runs",
        {
            "suggested_weight_min": "REAL",
            "suggested_weight_max": "REAL",
            "suggested_light_weight": "REAL",
            "suggested_strong_weight": "REAL",
            "suggested_weight_reason": "TEXT",
            "profile_applied_at": "TEXT",
            "preset_snapshot_json": "TEXT",
            "project_id": "INTEGER",
            "reference_set_id": "INTEGER",
            "reference_set_version_id": "INTEGER",
        },
    )
    ensure_columns(
        conn,
        "reference_sets",
        {
            "project_id": "INTEGER",
            "dataset_id": "INTEGER",
            "current_dataset_version_id": "INTEGER",
            "current_version_id": "INTEGER",
            "reference_type": "TEXT NOT NULL DEFAULT 'character'",
            "selection_mode": "TEXT NOT NULL DEFAULT 'manual'",
            "trigger_word": "TEXT",
            "description": "TEXT",
            "is_default": "INTEGER NOT NULL DEFAULT 0",
            "is_archived": "INTEGER NOT NULL DEFAULT 0",
            "memo": "TEXT",
        },
    )
    ensure_columns(
        conn,
        "reference_images",
        {
            "reference_set_version_id": "INTEGER",
            "dataset_id": "INTEGER",
            "dataset_version_id": "INTEGER",
            "source_type": "TEXT NOT NULL DEFAULT 'manual'",
            "source_image_path": "TEXT",
            "image_role": "TEXT NOT NULL DEFAULT 'other'",
            "prompt_role_hint": "TEXT",
            "caption": "TEXT",
            "caption_snapshot": "TEXT",
            "tags_json": "TEXT",
            "width": "INTEGER",
            "height": "INTEGER",
            "file_size": "INTEGER",
            "sha256": "TEXT",
            "include_in_machine_review": "INTEGER NOT NULL DEFAULT 1",
            "exclude_reason": "TEXT",
            "memo": "TEXT",
            "updated_at": "TEXT",
        },
    )
    ensure_columns(conn, "experiment_recommendations", {"project_id": "INTEGER"})
    ensure_columns(
        conn,
        "validation_expected_conditions",
        {
            "preset_version": "TEXT",
            "prompt": "TEXT",
            "webui_prompt": "TEXT",
            "negative_prompt": "TEXT",
            "trigger_word": "TEXT",
            "lora_filename": "TEXT",
            "base_model": "TEXT",
        },
    )
    ensure_columns(
        conn,
        "dataset_analysis",
        {
            "trigger_consistency_label": "TEXT",
            "trigger_consistency_message": "TEXT",
            "trigger_candidates_json": "TEXT",
        },
    )
    conn.executescript(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_training_outputs_job_path
            ON training_outputs(job_id, file_path);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sample_images_job_path
            ON sample_images(job_id, image_path);
        CREATE INDEX IF NOT EXISTS idx_file_cleanup_history_job
            ON file_cleanup_history(job_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_epoch_candidates_job_epoch
            ON training_epoch_candidate_summaries(job_id, epoch);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_training_metrics_job_step_tag
            ON training_metrics(job_id, step, raw_tag);
        CREATE INDEX IF NOT EXISTS idx_training_jobs_status
            ON training_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_training_jobs_project
            ON training_jobs(project_id);
        CREATE INDEX IF NOT EXISTS idx_lora_projects_status
            ON lora_projects(status);
        CREATE INDEX IF NOT EXISTS idx_validation_results_job
            ON validation_results(job_id);
        CREATE INDEX IF NOT EXISTS idx_validation_images_job
            ON validation_images(job_id);
        CREATE INDEX IF NOT EXISTS idx_validation_images_run
            ON validation_images(validation_run_id);
        CREATE INDEX IF NOT EXISTS idx_validation_images_expected_condition
            ON validation_images(expected_condition_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_validation_expected_conditions_hash
            ON validation_expected_conditions(validation_run_id, condition_hash);
        CREATE INDEX IF NOT EXISTS idx_validation_expected_conditions_run
            ON validation_expected_conditions(validation_run_id);
        CREATE INDEX IF NOT EXISTS idx_validation_weight_reviews_job
            ON validation_weight_reviews(job_id);
        CREATE INDEX IF NOT EXISTS idx_validation_weight_reviews_run
            ON validation_weight_reviews(validation_run_id);
        CREATE INDEX IF NOT EXISTS idx_validation_runs_job
            ON validation_runs(job_id);
        CREATE INDEX IF NOT EXISTS idx_validation_runs_project
            ON validation_runs(project_id);
        CREATE INDEX IF NOT EXISTS idx_validation_generation_runs_run
            ON validation_generation_runs(validation_run_id);
        CREATE INDEX IF NOT EXISTS idx_validation_generation_runs_status
            ON validation_generation_runs(status);
        CREATE INDEX IF NOT EXISTS idx_reference_images_set
            ON reference_images(reference_set_id);
        CREATE INDEX IF NOT EXISTS idx_reference_images_version
            ON reference_images(reference_set_version_id);
        CREATE INDEX IF NOT EXISTS idx_reference_set_versions_set
            ON reference_set_versions(reference_set_id);
        CREATE INDEX IF NOT EXISTS idx_image_embeddings_source
            ON image_embeddings(source_type, source_id, embedding_model_id);
        CREATE INDEX IF NOT EXISTS idx_image_embeddings_status
            ON image_embeddings(status);
        CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status
            ON embedding_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_embedding_job_items_job
            ON embedding_job_items(embedding_job_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_selected_lora_profiles_job_output
            ON selected_lora_profiles(job_id, selected_output_id);
        CREATE INDEX IF NOT EXISTS idx_selected_lora_profiles_project
            ON selected_lora_profiles(project_id);
        CREATE INDEX IF NOT EXISTS idx_experiment_recommendations_job
            ON experiment_recommendations(source_job_id, status);
        CREATE INDEX IF NOT EXISTS idx_experiment_recommendations_project
            ON experiment_recommendations(project_id);
        """
    )
    backfill_project_ids(conn)
    backfill_reference_set_versions(conn)


def ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def backfill_project_ids(conn: sqlite3.Connection) -> None:
    """Best-effort migration for existing beta data without changing user-owned rows destructively."""
    now = utc_now()
    rows = conn.execute("SELECT id FROM training_jobs WHERE project_id IS NULL ORDER BY id").fetchall()
    for row in rows:
        job_id = int(row["id"])
        project_id = infer_project_id_for_job(conn, job_id)
        if project_id is not None:
            conn.execute("UPDATE training_jobs SET project_id = ?, updated_at = ? WHERE id = ?", (project_id, now, job_id))

    for table, job_column in (
        ("selected_lora_profiles", "job_id"),
        ("validation_runs", "job_id"),
        ("experiment_recommendations", "source_job_id"),
    ):
        rows = conn.execute(
            f"""
            SELECT t.id, j.project_id
            FROM {table} t
            LEFT JOIN training_jobs j ON j.id = t.{job_column}
            WHERE t.project_id IS NULL AND j.project_id IS NOT NULL
            """
        ).fetchall()
        for row in rows:
            conn.execute(f"UPDATE {table} SET project_id = ? WHERE id = ?", (row["project_id"], row["id"]))


def infer_project_id_for_job(conn: sqlite3.Connection, job_id: int) -> int | None:
    job = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
    if job is None:
        return None
    parent_id = job["parent_job_id"] if "parent_job_id" in job.keys() else None
    if parent_id:
        parent = conn.execute("SELECT project_id FROM training_jobs WHERE id = ?", (parent_id,)).fetchone()
        if parent and parent["project_id"]:
            return int(parent["project_id"])

    dataset_id = job["dataset_id"]
    trigger_word = job["trigger_word_at_creation"] or ""
    base_model_path = job["base_model_path"] or ""
    existing = conn.execute(
        """
        SELECT id FROM lora_projects
        WHERE dataset_id IS ? AND COALESCE(trigger_word, '') = ? AND COALESCE(base_model_path, '') = ?
        ORDER BY id LIMIT 1
        """,
        (dataset_id, trigger_word, base_model_path),
    ).fetchone()
    if existing:
        return int(existing["id"])

    dataset = conn.execute("SELECT name, trigger_word FROM datasets WHERE id = ?", (dataset_id,)).fetchone() if dataset_id else None
    selected_output = conn.execute("SELECT id, epoch FROM training_outputs WHERE job_id = ? AND selected = 1 ORDER BY id DESC LIMIT 1", (job_id,)).fetchone()
    selected_profile = conn.execute("SELECT id, recommended_weight_min, recommended_weight_max FROM selected_lora_profiles WHERE job_id = ? ORDER BY updated_at DESC, id DESC LIMIT 1", (job_id,)).fetchone()
    name_parts = []
    if dataset:
        name_parts.append(f"Dataset #{dataset_id} {dataset['name']}")
    else:
        name_parts.append(f"Job #{job_id}")
    if trigger_word:
        name_parts.append(trigger_word)
    name = " ".join(name_parts).strip()[:160]
    cur = conn.execute(
        """
        INSERT INTO lora_projects(
            name, description, dataset_id, current_dataset_version_id, trigger_word,
            base_model_path, status, selected_job_id, selected_output_id,
            selected_lora_profile_id, recommended_weight_min, recommended_weight_max,
            created_at, updated_at, memo
        )
        VALUES (?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '')
        """,
        (
            name,
            dataset_id,
            job["dataset_version_id"] if "dataset_version_id" in job.keys() else None,
            trigger_word or (dataset["trigger_word"] if dataset else ""),
            base_model_path,
            "selected" if selected_output else "draft",
            job_id if selected_output else None,
            selected_output["id"] if selected_output else None,
            selected_profile["id"] if selected_profile else None,
            selected_profile["recommended_weight_min"] if selected_profile else None,
            selected_profile["recommended_weight_max"] if selected_profile else None,
            utc_now(),
            utc_now(),
        ),
    )
    return int(cur.lastrowid)


def backfill_reference_set_versions(conn: sqlite3.Connection) -> None:
    """Create a v1 snapshot for legacy reference sets and connect existing images to it."""
    now = utc_now()
    for row in conn.execute("SELECT * FROM reference_sets ORDER BY id").fetchall():
        set_id = int(row["id"])
        dataset_version_id = None
        if "current_dataset_version_id" in row.keys():
            dataset_version_id = row["current_dataset_version_id"]
        if not dataset_version_id and "dataset_version_id" in row.keys():
            dataset_version_id = row["dataset_version_id"]
        if dataset_version_id and not row["current_dataset_version_id"]:
            conn.execute("UPDATE reference_sets SET current_dataset_version_id = ? WHERE id = ?", (dataset_version_id, set_id))

        reference_type = row["reference_type"] or "character"
        if reference_type not in {"character", "style", "mixed", "other"}:
            reference_type = "other"
            conn.execute("UPDATE reference_sets SET reference_type = ? WHERE id = ?", (reference_type, set_id))

        version = conn.execute(
            "SELECT * FROM reference_set_versions WHERE reference_set_id = ? ORDER BY version_no LIMIT 1",
            (set_id,),
        ).fetchone()
        if version is None:
            image_count = int(conn.execute("SELECT COUNT(*) AS count FROM reference_images WHERE reference_set_id = ?", (set_id,)).fetchone()["count"] or 0)
            roles = reference_role_counts(conn, set_id, None)
            label, message = reference_completeness(reference_type, image_count, roles)
            cur = conn.execute(
                """
                INSERT INTO reference_set_versions(
                    reference_set_id, version_no, dataset_id, dataset_version_id, trigger_word,
                    reference_type, image_count, roles_json, completeness_label,
                    completeness_message, locked_at, memo, created_at, updated_at
                )
                VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    set_id,
                    row["dataset_id"],
                    dataset_version_id,
                    row["trigger_word"] or "",
                    reference_type,
                    image_count,
                    json.dumps(roles, ensure_ascii=False, sort_keys=True),
                    label,
                    message,
                    now,
                    row["memo"] or "Legacy Reference Set v1",
                    row["created_at"] or now,
                    now,
                ),
            )
            version_id = int(cur.lastrowid)
            conn.execute("UPDATE reference_sets SET current_version_id = ?, updated_at = ? WHERE id = ?", (version_id, now, set_id))
        else:
            version_id = int(version["id"])
            if not row["current_version_id"]:
                conn.execute("UPDATE reference_sets SET current_version_id = ? WHERE id = ?", (version_id, set_id))

        conn.execute(
            """
            UPDATE reference_images
            SET reference_set_version_id = COALESCE(reference_set_version_id, ?),
                dataset_id = COALESCE(dataset_id, ?),
                dataset_version_id = COALESCE(dataset_version_id, ?),
                source_type = COALESCE(NULLIF(source_type, ''), 'manual'),
                image_role = COALESCE(NULLIF(image_role, ''), 'other'),
                include_in_machine_review = COALESCE(include_in_machine_review, 1),
                updated_at = COALESCE(updated_at, created_at, ?)
            WHERE reference_set_id = ?
            """,
            (version_id, row["dataset_id"], dataset_version_id, now, set_id),
        )


def reference_role_counts(conn: sqlite3.Connection, reference_set_id: int, version_id: int | None) -> dict[str, int]:
    if version_id is None:
        rows = conn.execute(
            "SELECT image_role, COUNT(*) AS count FROM reference_images WHERE reference_set_id = ? GROUP BY image_role",
            (reference_set_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT image_role, COUNT(*) AS count FROM reference_images WHERE reference_set_version_id = ? GROUP BY image_role",
            (version_id,),
        ).fetchall()
    return {str(row["image_role"] or "other"): int(row["count"] or 0) for row in rows}


def reference_completeness(reference_type: str, image_count: int, roles: dict[str, int]) -> tuple[str, str]:
    if image_count <= 0:
        return "ERROR", "リファレンス画像がありません。"
    if reference_type == "style":
        coverage = sum(1 for key in ("close_up", "upper_body", "full_body", "background_scene") if roles.get(key, 0) > 0)
        if image_count >= 6 and coverage >= 3:
            return "OK", "style確認に必要な役割が概ね揃っています。"
        if image_count <= 2:
            return "ERROR", "style確認には画像が少なすぎます。"
        return "WARNING", "style確認用の画像数または役割に偏りがあります。"
    if reference_type == "character":
        coverage = sum(1 for key in ("face_front", "upper_body", "full_body") if roles.get(key, 0) > 0)
        if image_count >= 3 and coverage >= 2:
            return "OK", "character確認に必要な役割が概ね揃っています。"
        return "WARNING", "顔・上半身・全身のいずれかが不足している可能性があります。"
    if reference_type == "mixed":
        if image_count >= 4:
            return "OK", "mixed用途として最低限の画像数があります。"
        return "WARNING", "mixed用途としては画像数が少なめです。"
    return "UNKNOWN", "reference_typeが未分類のため、手動確認してください。"


def seed_app_settings(conn: sqlite3.Connection) -> None:
    now = utc_now()
    values = {
        "app_name": settings.APP_NAME,
        "app_version": APP_VERSION,
        "db_schema_version": DB_SCHEMA_VERSION,
        "sd_scripts_release_tag": settings.SD_SCRIPTS_RELEASE_TAG,
        "sd_scripts_release_commit": settings.SD_SCRIPTS_RELEASE_COMMIT,
        "sd_scripts_repo_url": settings.SD_SCRIPTS_REPO_URL,
        "webui_api_enabled": "false",
        "webui_api_url": "http://127.0.0.1:7865",
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


def seed_sample_prompt_templates(conn: sqlite3.Connection) -> None:
    now = utc_now()
    prompts = [
        {
            "name": "basic_face",
            "prompt": "{trigger_word}, 1girl, upper body, looking at viewer, simple background",
            "negative_prompt": "low quality, worst quality, bad anatomy, bad hands",
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "steps": 28,
            "cfg_scale": 7,
        },
        {
            "name": "full_body",
            "prompt": "{trigger_word}, 1girl, full body, standing, outdoors",
            "negative_prompt": "low quality, worst quality, bad anatomy, bad hands",
            "width": 1024,
            "height": 1024,
            "seed": 43,
            "steps": 28,
            "cfg_scale": 7,
        },
        {
            "name": "expression_pose",
            "prompt": "{trigger_word}, 1girl, smile, dynamic pose, city background",
            "negative_prompt": "low quality, worst quality, bad anatomy, bad hands",
            "width": 1024,
            "height": 1024,
            "seed": 44,
            "steps": 28,
            "cfg_scale": 7,
        },
    ]
    conn.execute(
        """
        INSERT INTO sample_prompt_templates(
            id, name, purpose, prompts_json, is_builtin, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, 1, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            purpose = excluded.purpose,
            prompts_json = excluded.prompts_json,
            is_builtin = excluded.is_builtin,
            updated_at = excluded.updated_at
        """,
        (
            "sdxl_face_basic_3prompts",
            "SDXL Face Basic 3 Prompts",
            "顔LoRAの基本確認用",
            json.dumps(prompts, ensure_ascii=False, indent=2),
            now,
            now,
        ),
    )


def seed_evaluation_rubrics(conn: sqlite3.Connection) -> None:
    now = utc_now()
    schema = {
        "version": "1.0",
        "fields": {
            "strength_label": ["too_weak", "weak_but_usable", "recommended", "strong_but_usable", "too_strong", "broken"],
            "overfit_level": ["none", "slight", "moderate", "severe"],
            "adoption_label": ["reject", "candidate", "adopt"],
            "failure_tags": [
                "顔が弱い",
                "顔が変わる",
                "衣装が弱い",
                "衣装固定",
                "背景汚染",
                "構図固定",
                "表情固定",
                "手足破綻",
                "画風過多",
                "LoRA効果弱い",
                "LoRA効果強すぎ",
                "trigger反応弱い",
                "triggerなし暴発",
            ],
            "ratings": [
                "rating_face",
                "rating_costume",
                "rating_style",
                "rating_stability",
                "rating_flexibility",
                "rating_overall",
            ],
        },
    }
    conn.execute(
        """
        INSERT INTO evaluation_rubrics(id, name, version, description, schema_json, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, 1, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            version = excluded.version,
            description = excluded.description,
            schema_json = excluded.schema_json,
            is_active = excluded.is_active
        """,
        (
            "lora_visual_eval_v1",
            "LoRA Visual Evaluation Rubric",
            "1.0",
            "LoRAの強さ、過学習、採用判断、失敗タグを定型化する評価schema。",
            json.dumps(schema, ensure_ascii=False, indent=2),
            now,
        ),
    )


def validation_prompt_rows() -> list[dict[str, Any]]:
    return [
        {
            "prompt_key": "basic_face",
            "name": "Basic Face",
            "prompt": "{trigger_word}, 1girl, upper body, looking at viewer, simple background",
            "seed_offset": 0,
        },
        {
            "prompt_key": "full_body",
            "name": "Full Body",
            "prompt": "{trigger_word}, 1girl, full body, standing, outdoors",
            "seed_offset": 0,
        },
        {
            "prompt_key": "expression_pose",
            "name": "Expression / Pose",
            "prompt": "{trigger_word}, 1girl, smile, dynamic pose, city background",
            "seed_offset": 0,
        },
    ]


def seed_validation_presets(conn: sqlite3.Connection) -> None:
    now = utc_now()
    negative = "low quality, worst quality, bad anatomy, bad hands"
    rows = [
        {
            "id": "quick_validation_v1",
            "name": "Quick Validation v1",
            "version": "1.0",
            "description": "採用LoRAが使えそうかを短時間で確認する推奨weight帯の初期確認用。",
            "validation_level": "quick",
            "seeds": [111111],
            "weights": [0.4, 0.6, 0.8],
            "hires_modes": [False],
            "hires_scale": None,
            "hires_denoising_strength": None,
            "hires_upscaler": "",
            "memo": "3 prompts x 1 seed x 3 weights x no hires = 9 images.",
        },
        {
            "id": "standard_validation_v1",
            "name": "Standard Validation v1",
            "version": "1.0",
            "description": "採用候補LoRAの正式なweight比較。LoRA Libraryに残す標準Validation。",
            "validation_level": "standard",
            "seeds": [111111, 222222, 333333],
            "weights": [0, 0.4, 0.6, 0.8, 1.0],
            "hires_modes": [False],
            "hires_scale": None,
            "hires_denoising_strength": None,
            "hires_upscaler": "",
            "memo": "weight 0はベースモデルとの差分確認用。標準比較はHiresなしを基準にします。",
        },
        {
            "id": "extended_validation_v1",
            "name": "Extended Validation v1",
            "version": "1.0",
            "description": "採用候補LoRAの最終確認。Hiresあり/なしの見え方の差を確認する。",
            "validation_level": "extended",
            "seeds": [111111, 222222],
            "weights": [0.6, 0.8],
            "hires_modes": [False, True],
            "hires_scale": 2.0,
            "hires_denoising_strength": 0.4,
            "hires_upscaler": "Latent",
            "memo": "HiresはLoRAの素の比較ではなく、最終見栄え確認として扱います。",
        },
    ]
    prompts = validation_prompt_rows()
    for row in rows:
        expected = len(prompts) * len(row["seeds"]) * len(row["weights"]) * len(row["hires_modes"])
        conn.execute(
            """
            INSERT INTO validation_presets(
                id, name, version, description, validation_level, prompts_json,
                seeds_json, weights_json, width, height, hires_modes_json,
                hires_scale, hires_denoising_strength, hires_upscaler, sampler,
                steps, cfg_scale, negative_prompt, reference_set_id,
                expected_image_count, is_builtin, is_active, created_at,
                updated_at, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1024, 1024, ?, ?, ?, ?, 'Euler a',
                28, 7, ?, NULL, ?, 1, 1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                version = excluded.version,
                description = excluded.description,
                validation_level = excluded.validation_level,
                prompts_json = excluded.prompts_json,
                seeds_json = excluded.seeds_json,
                weights_json = excluded.weights_json,
                width = excluded.width,
                height = excluded.height,
                hires_modes_json = excluded.hires_modes_json,
                hires_scale = excluded.hires_scale,
                hires_denoising_strength = excluded.hires_denoising_strength,
                hires_upscaler = excluded.hires_upscaler,
                sampler = excluded.sampler,
                steps = excluded.steps,
                cfg_scale = excluded.cfg_scale,
                negative_prompt = excluded.negative_prompt,
                expected_image_count = excluded.expected_image_count,
                is_builtin = excluded.is_builtin,
                is_active = excluded.is_active,
                updated_at = excluded.updated_at,
                memo = excluded.memo
            """,
            (
                row["id"],
                row["name"],
                row["version"],
                row["description"],
                row["validation_level"],
                json.dumps(prompts, ensure_ascii=False, indent=2),
                json.dumps(row["seeds"], ensure_ascii=False),
                json.dumps(row["weights"], ensure_ascii=False),
                json.dumps(row["hires_modes"], ensure_ascii=False),
                row["hires_scale"],
                row["hires_denoising_strength"],
                row["hires_upscaler"],
                negative,
                expected,
                now,
                now,
                row["memo"],
            ),
        )


def seed_embedding_models(conn: sqlite3.Connection) -> None:
    now = utc_now()
    rows = [
        {
            "id": "mock_image_512",
            "name": "Mock Image 512",
            "provider": "mock",
            "model_name": "mock_image_512",
            "pretrained": "",
            "embedding_type": "image",
            "vector_dim": 512,
            "normalize": 1,
            "device_default": "cpu",
            "dtype_default": "fp32",
            "batch_size_default": 8,
            "allow_download": 0,
            "memo": "テスト用の決定論的な疑似embedding。外部モデルやdownloadは不要です。",
        },
        {
            "id": "open_clip_vit_b32",
            "name": "OpenCLIP ViT-B-32",
            "provider": "open_clip",
            "model_name": "ViT-B-32",
            "pretrained": "openai",
            "embedding_type": "image",
            "vector_dim": None,
            "normalize": 1,
            "device_default": "auto",
            "dtype_default": "fp32",
            "batch_size_default": 4,
            "allow_download": 0,
            "memo": "任意provider。open_clipが未導入の場合はWARNING表示になります。",
        },
        {
            "id": "transformers_clip_vit_base_patch32",
            "name": "Transformers CLIP ViT-B/32",
            "provider": "transformers_clip",
            "model_name": "openai/clip-vit-base-patch32",
            "pretrained": "",
            "embedding_type": "image",
            "vector_dim": None,
            "normalize": 1,
            "device_default": "auto",
            "dtype_default": "fp32",
            "batch_size_default": 4,
            "allow_download": 0,
            "memo": "任意provider。transformersが未導入の場合はWARNING表示になります。",
        },
    ]
    for row in rows:
        conn.execute(
            """
            INSERT INTO embedding_models(
                id, name, provider, model_name, pretrained, embedding_type,
                vector_dim, normalize, device_default, dtype_default,
                batch_size_default, allow_download, is_builtin, is_active,
                created_at, updated_at, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                provider = excluded.provider,
                model_name = excluded.model_name,
                pretrained = excluded.pretrained,
                embedding_type = excluded.embedding_type,
                vector_dim = COALESCE(embedding_models.vector_dim, excluded.vector_dim),
                normalize = excluded.normalize,
                device_default = excluded.device_default,
                dtype_default = excluded.dtype_default,
                batch_size_default = excluded.batch_size_default,
                allow_download = excluded.allow_download,
                is_builtin = excluded.is_builtin,
                updated_at = excluded.updated_at,
                memo = excluded.memo
            """,
            (
                row["id"],
                row["name"],
                row["provider"],
                row["model_name"],
                row["pretrained"],
                row["embedding_type"],
                row["vector_dim"],
                row["normalize"],
                row["device_default"],
                row["dtype_default"],
                row["batch_size_default"],
                row["allow_download"],
                now,
                now,
                row["memo"],
            ),
        )
    existing = conn.execute("SELECT id FROM embedding_settings ORDER BY id LIMIT 1").fetchone()
    if existing is None:
        conn.execute(
            """
            INSERT INTO embedding_settings(
                active_embedding_model_id, python_path, device, dtype, batch_size,
                cache_root, allow_model_download, max_image_size, num_workers,
                created_at, updated_at
            )
            VALUES ('mock_image_512', '', 'auto', 'fp32', 8, ?, 0, 1024, 1, ?, ?)
            """,
            (str(settings.EMBEDDINGS_DIR), now, now),
        )


def seed_legacy_validation_run(conn: sqlite3.Connection) -> None:
    now = utc_now()
    jobs = conn.execute(
        """
        SELECT DISTINCT job_id
        FROM (
            SELECT job_id FROM validation_images WHERE validation_run_id IS NULL
            UNION
            SELECT job_id FROM validation_weight_reviews WHERE validation_run_id IS NULL
        )
        """
    ).fetchall()
    for job in jobs:
        job_id = int(job["job_id"])
        existing = conn.execute(
            "SELECT id FROM validation_runs WHERE job_id = ? AND validation_preset_id IS NULL AND name LIKE 'Legacy Validation%'",
            (job_id,),
        ).fetchone()
        if existing:
            run_id = int(existing["id"])
        else:
            profile = conn.execute(
                "SELECT * FROM selected_lora_profiles WHERE job_id = ? ORDER BY updated_at DESC, id DESC LIMIT 1",
                (job_id,),
            ).fetchone()
            output_id = profile["selected_output_id"] if profile else None
            count = conn.execute("SELECT COUNT(*) AS count FROM validation_images WHERE job_id = ?", (job_id,)).fetchone()["count"]
            cur = conn.execute(
                """
                INSERT INTO validation_runs(
                    job_id, selected_output_id, selected_lora_profile_id, validation_preset_id,
                    name, validation_level, base_model, trigger_word, lora_filename,
                    recommended_weight_min, recommended_weight_max, expected_image_count,
                    actual_image_count, status, created_at, updated_at, memo
                )
                VALUES (?, ?, ?, NULL, ?, 'legacy', ?, ?, ?, ?, ?, NULL, ?, 'images_registered', ?, ?, ?)
                """,
                (
                    job_id,
                    output_id,
                    profile["id"] if profile else None,
                    f"Legacy Validation Job #{job_id}",
                    profile["base_model"] if profile else "",
                    profile["trigger_word"] if profile else "",
                    Path(profile["selected_model_path"]).name if profile and profile["selected_model_path"] else "",
                    profile["recommended_weight_min"] if profile else None,
                    profile["recommended_weight_max"] if profile else None,
                    count,
                    now,
                    now,
                    "legacy / preset unspecified",
                ),
            )
            run_id = int(cur.lastrowid)
        conn.execute("UPDATE validation_images SET validation_run_id = ? WHERE job_id = ? AND validation_run_id IS NULL", (run_id, job_id))
        conn.execute("UPDATE validation_weight_reviews SET validation_run_id = ? WHERE job_id = ? AND validation_run_id IS NULL", (run_id, job_id))


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with db_session() as conn:
        return list(conn.execute(query, params).fetchall())


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    with db_session() as conn:
        return conn.execute(query, params).fetchone()


def has_smoke_step_limit(params: dict[str, Any]) -> bool:
    return any(key in params for key in SMOKE_STEP_LIMIT_KEYS)


def normalize_job_params_for_preset(preset: sqlite3.Row, params: dict[str, Any] | None) -> dict[str, Any]:
    preset_params = json.loads(preset["params_json"])
    if params is None:
        return preset_params
    if preset["id"] != INTEGRATION_SMOKE_PRESET_ID and has_smoke_step_limit(params):
        return preset_params
    return params


def insert_dataset(name: str, path: str, model_family: str, trigger_word: str, class_token: str, memo: str) -> int:
    from app.services.dataset_scanner import scan_dataset

    now = utc_now()
    scan = scan_dataset(Path(path), trigger_word)
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
        dataset_id = int(cur.lastrowid)
        upsert_dataset_analysis(conn, dataset_id, scan)
        create_dataset_version(conn, dataset_id, scan, "Initial dataset registration")
        return dataset_id


def create_job(data: dict[str, Any]) -> int:
    now = utc_now()
    preset = fetch_one("SELECT * FROM presets WHERE id = ?", (data["preset_id"],))
    if preset is None:
        raise ValueError(f"Preset not found: {data['preset_id']}")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (int(data["dataset_id"]),))
    analysis = fetch_one("SELECT * FROM dataset_analysis WHERE dataset_id = ?", (int(data["dataset_id"]),))
    params = normalize_job_params_for_preset(preset, data.get("params"))
    output_name = data.get("output_name") or data["name"].replace(" ", "_")
    trigger_word = dataset["trigger_word"] if dataset else None
    trigger_count = analysis["trigger_word_count"] if analysis else None
    trigger_rate = analysis["trigger_word_rate"] if analysis else None
    trigger_label = analysis["trigger_consistency_label"] if analysis and "trigger_consistency_label" in analysis.keys() else None
    version = fetch_one("SELECT id FROM dataset_versions WHERE dataset_id = ? ORDER BY version_no DESC LIMIT 1", (int(data["dataset_id"]),))
    dataset_version_id = version["id"] if version else None
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO training_jobs(
                project_id, name, dataset_id, preset_id, environment_id, status, model_family,
                training_script, base_model_path, vae_path, output_name, output_dir,
                run_dir, params_json, memo, created_at, updated_at
                , parent_job_id, sample_prompt_template_id
                , trigger_word_at_creation, trigger_occurrence_count_at_creation,
                trigger_occurrence_rate_at_creation, trigger_consistency_label_at_creation,
                dataset_version_id
            ) VALUES (?, ?, ?, ?, NULL, 'draft', ?, ?, ?, ?, ?, '', '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("project_id"), data["name"], int(data["dataset_id"]), data["preset_id"],
                preset["model_family"], preset["training_script"], data["base_model_path"],
                data.get("vae_path") or None, output_name,
                json.dumps(params, ensure_ascii=False, indent=2), data.get("memo") or "", now, now,
                data.get("parent_job_id"), data.get("sample_prompt_template_id") or None,
                trigger_word, trigger_count, trigger_rate, trigger_label, dataset_version_id,
            ),
        )
        job_id = int(cur.lastrowid)
        run_dir = settings.RUNS_DIR / f"job_{job_id:06d}"
        output_dir = run_dir / "models"
        for subdir in ("config", "logs", "models", "samples", "metrics", "exports"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
        conn.execute("UPDATE training_jobs SET run_dir = ?, output_dir = ?, updated_at = ? WHERE id = ?", (str(run_dir), str(output_dir), now, job_id))
        return job_id


def upsert_dataset_analysis(conn: sqlite3.Connection, dataset_id: int, scan: dict[str, Any]) -> None:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO dataset_analysis(
            dataset_id, supported_image_count, unsupported_file_count, broken_image_count,
            empty_caption_count, unreadable_caption_count, caption_encoding_summary_json,
            image_size_summary_json, tag_summary_json, trigger_word_count, trigger_word_rate,
            trigger_consistency_label, trigger_consistency_message, trigger_candidates_json,
            missing_caption_images_json, caption_without_images_json, broken_images_json,
            unsupported_files_json, analysis_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id) DO UPDATE SET
            supported_image_count = excluded.supported_image_count,
            unsupported_file_count = excluded.unsupported_file_count,
            broken_image_count = excluded.broken_image_count,
            empty_caption_count = excluded.empty_caption_count,
            unreadable_caption_count = excluded.unreadable_caption_count,
            caption_encoding_summary_json = excluded.caption_encoding_summary_json,
            image_size_summary_json = excluded.image_size_summary_json,
            tag_summary_json = excluded.tag_summary_json,
            trigger_word_count = excluded.trigger_word_count,
            trigger_word_rate = excluded.trigger_word_rate,
            trigger_consistency_label = excluded.trigger_consistency_label,
            trigger_consistency_message = excluded.trigger_consistency_message,
            trigger_candidates_json = excluded.trigger_candidates_json,
            missing_caption_images_json = excluded.missing_caption_images_json,
            caption_without_images_json = excluded.caption_without_images_json,
            broken_images_json = excluded.broken_images_json,
            unsupported_files_json = excluded.unsupported_files_json,
            analysis_json = excluded.analysis_json,
            updated_at = excluded.updated_at
        """,
        (
            dataset_id,
            scan.get("supported_image_count", 0),
            scan.get("unsupported_file_count", 0),
            scan.get("broken_image_count", 0),
            scan.get("empty_caption_count", 0),
            scan.get("unreadable_caption_count", 0),
            json.dumps(scan.get("caption_encoding_summary") or {}, ensure_ascii=False),
            json.dumps(scan.get("image_size_summary") or {}, ensure_ascii=False),
            json.dumps(scan.get("tag_summary") or {}, ensure_ascii=False),
            scan.get("trigger_word_count", 0),
            scan.get("trigger_word_rate"),
            (scan.get("trigger_consistency") or {}).get("label"),
            (scan.get("trigger_consistency") or {}).get("message"),
            json.dumps(scan.get("trigger_candidates") or [], ensure_ascii=False),
            json.dumps(scan.get("missing_caption_images") or [], ensure_ascii=False),
            json.dumps(scan.get("caption_without_images") or [], ensure_ascii=False),
            json.dumps(scan.get("broken_images") or [], ensure_ascii=False),
            json.dumps(scan.get("unsupported_files") or [], ensure_ascii=False),
            json.dumps(scan, ensure_ascii=False),
            now,
            now,
        ),
    )


def create_dataset_version(
    conn: sqlite3.Connection,
    dataset_id: int,
    scan: dict[str, Any],
    memo: str,
) -> int:
    dataset = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    if dataset is None:
        raise ValueError(f"Dataset not found: {dataset_id}")
    row = conn.execute(
        "SELECT COALESCE(MAX(version_no), 0) + 1 AS next_no FROM dataset_versions WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchone()
    version_no = int(row["next_no"])
    now = utc_now()
    stats = {
        "resolution_summary": scan.get("resolution_summary") or {},
        "image_size_summary": scan.get("image_size_summary") or {},
        "tag_summary": scan.get("tag_summary") or {},
        "trigger_candidates": scan.get("trigger_candidates") or [],
        "trigger_consistency": scan.get("trigger_consistency") or {},
    }
    cur = conn.execute(
        """
        INSERT INTO dataset_versions(
            dataset_id, version_no, trigger_word, image_count, caption_count,
            missing_caption_count, broken_image_count, trigger_occurrence_count,
            trigger_occurrence_rate, trigger_consistency_label, image_manifest_hash,
            caption_manifest_hash, stats_json, created_at, memo
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset_id,
            version_no,
            dataset["trigger_word"],
            scan.get("image_count", 0),
            scan.get("caption_count", 0),
            scan.get("missing_caption_count", 0),
            scan.get("broken_image_count", 0),
            scan.get("trigger_word_count", 0),
            scan.get("trigger_word_rate"),
            (scan.get("trigger_consistency") or {}).get("label"),
            manifest_hash(Path(dataset["path"]), image_manifest_entries(Path(dataset["path"]))),
            manifest_hash(Path(dataset["path"]), caption_manifest_entries(Path(dataset["path"]))),
            json.dumps(stats, ensure_ascii=False),
            now,
            memo,
        ),
    )
    return int(cur.lastrowid)


def manifest_hash(dataset_path: Path, entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def image_manifest_entries(dataset_path: Path) -> list[dict[str, Any]]:
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    entries = []
    if not dataset_path.exists():
        return entries
    for path in sorted(p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in extensions):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append(
            {
                "path": str(path.relative_to(dataset_path)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return entries


def caption_manifest_entries(dataset_path: Path) -> list[dict[str, Any]]:
    entries = []
    if not dataset_path.exists():
        return entries
    for path in sorted(p for p in dataset_path.rglob("*.txt") if p.is_file()):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        entries.append(
            {
                "path": str(path.relative_to(dataset_path)),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return entries


def import_latest_environment(conn: sqlite3.Connection | None = None) -> None:
    owns_connection = conn is None
    if conn is None:
        conn = connect()
    try:
        result_path = settings.LOGS_DIR / "setup_sd_scripts_result.json"
        if not result_path.exists():
            return
        try:
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            return

        venv_python_path = result.get("venv_python_path") or ""
        runtime = inspect_sd_scripts_runtime(venv_python_path)
        now = utc_now()
        existing = conn.execute("SELECT id, created_at FROM environments WHERE name = ?", ("default",)).fetchone()
        values = {
            "name": "default",
            "sd_scripts_path": result.get("sd_scripts_path") or "",
            "venv_python_path": venv_python_path,
            "venv_accelerate_path": result.get("venv_accelerate_path"),
            "cuda_profile": result.get("cuda_profile"),
            "mixed_precision": result.get("mixed_precision"),
            "python_version": runtime.get("python_version"),
            "torch_version": runtime.get("torch_version"),
            "torch_cuda_version": runtime.get("torch_cuda_version"),
            "cuda_available": 1 if runtime.get("cuda_available") else 0 if "cuda_available" in runtime else None,
            "gpu_name": runtime.get("gpu_name"),
            "sd_scripts_commit_hash": result.get("commit"),
            "status": "ready" if result.get("commit") and Path(venv_python_path).exists() else "unknown",
            "updated_at": now,
        }
        if existing:
            conn.execute(
                """
                UPDATE environments SET
                    sd_scripts_path = :sd_scripts_path,
                    venv_python_path = :venv_python_path,
                    venv_accelerate_path = :venv_accelerate_path,
                    cuda_profile = :cuda_profile,
                    mixed_precision = :mixed_precision,
                    python_version = :python_version,
                    torch_version = :torch_version,
                    torch_cuda_version = :torch_cuda_version,
                    cuda_available = :cuda_available,
                    gpu_name = :gpu_name,
                    sd_scripts_commit_hash = :sd_scripts_commit_hash,
                    status = :status,
                    updated_at = :updated_at
                WHERE name = :name
                """,
                values,
            )
        else:
            values["created_at"] = now
            conn.execute(
                """
                INSERT INTO environments(
                    name, sd_scripts_path, venv_python_path, venv_accelerate_path,
                    cuda_profile, mixed_precision, python_version, torch_version,
                    torch_cuda_version, cuda_available, gpu_name, sd_scripts_commit_hash,
                    status, created_at, updated_at
                )
                VALUES (
                    :name, :sd_scripts_path, :venv_python_path, :venv_accelerate_path,
                    :cuda_profile, :mixed_precision, :python_version, :torch_version,
                    :torch_cuda_version, :cuda_available, :gpu_name, :sd_scripts_commit_hash,
                    :status, :created_at, :updated_at
                )
                """,
                values,
            )
    finally:
        if owns_connection:
            conn.commit()
            conn.close()


def inspect_sd_scripts_runtime(venv_python_path: str) -> dict[str, Any]:
    python_path = Path(venv_python_path)
    if not python_path.exists():
        return {}
    code = (
        "import json, sys, torch; "
        "print(json.dumps({"
        "'python_version': sys.version, "
        "'torch_version': torch.__version__, "
        "'torch_cuda_version': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), "
        "'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        "}, ensure_ascii=False))"
    )
    try:
        result = subprocess.run(
            [str(python_path), "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except Exception:
        return {}
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {}


def latest_environment() -> sqlite3.Row | None:
    import_latest_environment()
    return fetch_one("SELECT * FROM environments ORDER BY updated_at DESC, id DESC LIMIT 1")


def replace_sample_prompts(job_id: int, prompts: list[dict[str, Any]]) -> None:
    now = utc_now()
    with connect() as conn:
        conn.execute("DELETE FROM sample_prompts WHERE job_id = ?", (job_id,))
        conn.executemany(
            """
            INSERT INTO sample_prompts(
                job_id, name, prompt, negative_prompt, width, height,
                seed, cfg_scale, steps, sort_order, prompt_role, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    job_id,
                    item["name"],
                    item["prompt"],
                    item.get("negative_prompt"),
                    item.get("width"),
                    item.get("height"),
                    item.get("seed"),
                    item.get("cfg_scale"),
                    item.get("steps"),
                    index,
                    item.get("prompt_role") or infer_prompt_role(item.get("name", ""), item.get("prompt", "")),
                    now,
                )
                for index, item in enumerate(prompts, start=1)
            ],
        )


def infer_prompt_role(name: str, prompt: str = "") -> str:
    value = f"{name} {prompt}".lower()
    if "basic_face" in value or "face" in value or "upper body" in value:
        return "face"
    if "full_body" in value or "full body" in value or "standing" in value:
        return "full_body"
    if "expression" in value or "pose" in value or "dynamic" in value:
        return "expression_pose"
    if "clothes" in value or "costume" in value:
        return "clothes"
    if "background" in value:
        return "background"
    return "other"


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
CREATE TABLE IF NOT EXISTS dataset_analysis (
    dataset_id INTEGER PRIMARY KEY, supported_image_count INTEGER, unsupported_file_count INTEGER,
    broken_image_count INTEGER, empty_caption_count INTEGER, unreadable_caption_count INTEGER,
    caption_encoding_summary_json TEXT, image_size_summary_json TEXT, tag_summary_json TEXT,
    trigger_word_count INTEGER, trigger_word_rate REAL, trigger_consistency_label TEXT,
    trigger_consistency_message TEXT, trigger_candidates_json TEXT, missing_caption_images_json TEXT,
    caption_without_images_json TEXT, broken_images_json TEXT, unsupported_files_json TEXT,
    analysis_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id INTEGER NOT NULL, version_no INTEGER NOT NULL,
    trigger_word TEXT, image_count INTEGER, caption_count INTEGER, missing_caption_count INTEGER,
    broken_image_count INTEGER, trigger_occurrence_count INTEGER, trigger_occurrence_rate REAL,
    trigger_consistency_label TEXT, image_manifest_hash TEXT, caption_manifest_hash TEXT,
    stats_json TEXT, created_at TEXT NOT NULL, memo TEXT
);
CREATE TABLE IF NOT EXISTS sample_prompt_templates (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, purpose TEXT, prompts_json TEXT NOT NULL,
    is_builtin INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lora_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, description TEXT,
    dataset_id INTEGER, current_dataset_version_id INTEGER, trigger_word TEXT,
    base_model_path TEXT, status TEXT NOT NULL DEFAULT 'draft',
    selected_job_id INTEGER, selected_output_id INTEGER, selected_lora_profile_id INTEGER,
    recommended_weight_min REAL, recommended_weight_max REAL,
    archived_at TEXT, deleted_at TEXT, archive_reason TEXT, delete_reason TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, memo TEXT
);
CREATE TABLE IF NOT EXISTS training_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, name TEXT NOT NULL, dataset_id INTEGER, preset_id TEXT,
    environment_id INTEGER, status TEXT NOT NULL, model_family TEXT NOT NULL, training_script TEXT NOT NULL,
    base_model_path TEXT NOT NULL, vae_path TEXT, output_name TEXT NOT NULL, output_dir TEXT NOT NULL,
    run_dir TEXT NOT NULL, params_json TEXT NOT NULL, command_line TEXT, process_id INTEGER,
    return_code INTEGER, start_time TEXT, end_time TEXT, elapsed_seconds INTEGER, adopted_epoch INTEGER,
    adopted_model_path TEXT, image_rating INTEGER, loss_health_label TEXT, memo TEXT,
    parent_job_id INTEGER, sample_prompt_template_id TEXT,
    trigger_word_at_creation TEXT, trigger_occurrence_count_at_creation INTEGER,
    trigger_occurrence_rate_at_creation REAL, trigger_consistency_label_at_creation TEXT,
    dataset_version_id INTEGER, config_dirty INTEGER NOT NULL DEFAULT 0,
    archived_at TEXT, deleted_at TEXT, archived_reason TEXT, delete_reason TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, step INTEGER, epoch REAL, loss REAL,
    lr REAL, unet_lr REAL, text_encoder_lr REAL, raw_json TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_metric_summaries (
    job_id INTEGER PRIMARY KEY, initial_loss REAL, final_loss REAL, min_loss REAL, min_loss_epoch REAL,
    loss_drop_rate REAL, loss_volatility REAL, spike_count INTEGER, late_stage_slope REAL,
    moving_avg_final_loss REAL, raw_loss_label TEXT, smoothed_loss_label TEXT, epoch_trend_label TEXT,
    health_label TEXT, health_score INTEGER, summary_json TEXT, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_epoch_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, epoch INTEGER NOT NULL,
    step_start INTEGER, step_end INTEGER, metric_count INTEGER, avg_loss REAL,
    min_loss REAL, max_loss REAL, final_loss REAL, moving_avg_final_loss REAL,
    spike_count INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE(job_id, epoch)
);
CREATE TABLE IF NOT EXISTS sample_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER, name TEXT NOT NULL, prompt TEXT NOT NULL,
    negative_prompt TEXT, width INTEGER, height INTEGER, seed INTEGER, cfg_scale REAL, steps INTEGER,
    sort_order INTEGER NOT NULL DEFAULT 0, prompt_role TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sample_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, prompt_id INTEGER, epoch INTEGER,
    step INTEGER, image_path TEXT NOT NULL, prompt TEXT, negative_prompt TEXT, seed INTEGER,
    width INTEGER, height INTEGER, cfg_scale REAL, steps INTEGER, rating INTEGER,
    rating_face INTEGER, rating_costume INTEGER, rating_style INTEGER,
    rating_stability INTEGER, rating_flexibility INTEGER, rating_overall INTEGER, memo TEXT,
    review_priority TEXT, auto_review_label TEXT, auto_review_reason TEXT,
    deleted_at TEXT, cleanup_status TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_epoch_candidate_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, epoch INTEGER NOT NULL,
    candidate_rank INTEGER, candidate_label TEXT NOT NULL, score REAL,
    reason_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE(job_id, epoch)
);
CREATE TABLE IF NOT EXISTS training_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, epoch INTEGER, step INTEGER,
    file_path TEXT NOT NULL, file_type TEXT NOT NULL, file_size INTEGER, sha256 TEXT,
    selected INTEGER NOT NULL DEFAULT 0, memo TEXT, metadata_error TEXT,
    deleted_at TEXT, archived_at TEXT, export_verified_at TEXT, external_copy_path TEXT,
    cleanup_status TEXT, delete_reason TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS file_cleanup_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER, project_id INTEGER,
    action TEXT NOT NULL, original_path TEXT NOT NULL, trash_path TEXT,
    file_size INTEGER, sha256 TEXT, created_at TEXT NOT NULL, memo TEXT
);
CREATE TABLE IF NOT EXISTS validation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, selected_output_id INTEGER,
    prompt_type TEXT NOT NULL, lora_weight REAL NOT NULL,
    face_score INTEGER, costume_score INTEGER, stability_score INTEGER,
    flexibility_score INTEGER, overall_score INTEGER,
    memo TEXT, image_path TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluation_rubrics (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL,
    description TEXT, schema_json TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS validation_presets (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL,
    description TEXT, validation_level TEXT NOT NULL, prompts_json TEXT NOT NULL,
    seeds_json TEXT NOT NULL, weights_json TEXT NOT NULL, width INTEGER NOT NULL,
    height INTEGER NOT NULL, hires_modes_json TEXT NOT NULL, hires_scale REAL,
    hires_denoising_strength REAL, hires_upscaler TEXT, sampler TEXT,
    steps INTEGER, cfg_scale REAL, negative_prompt TEXT, reference_set_id INTEGER,
    expected_image_count INTEGER, is_builtin INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, memo TEXT
);
CREATE TABLE IF NOT EXISTS validation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, job_id INTEGER NOT NULL,
    selected_output_id INTEGER, selected_lora_profile_id INTEGER,
    validation_preset_id TEXT, name TEXT NOT NULL, validation_level TEXT,
    base_model TEXT, trigger_word TEXT, lora_filename TEXT,
    recommended_weight_min REAL, recommended_weight_max REAL,
    suggested_weight_min REAL, suggested_weight_max REAL,
    suggested_light_weight REAL, suggested_strong_weight REAL,
    suggested_weight_reason TEXT, profile_applied_at TEXT,
    preset_snapshot_json TEXT,
    expected_image_count INTEGER, actual_image_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'planned', created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, memo TEXT
);
CREATE TABLE IF NOT EXISTS validation_generation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, validation_run_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'planned', process_id INTEGER,
    command_argv_json TEXT, prompt_file_path TEXT, output_dir TEXT, log_path TEXT,
    started_at TEXT, ended_at TEXT, elapsed_seconds INTEGER, return_code INTEGER,
    generated_image_count INTEGER NOT NULL DEFAULT 0,
    imported_image_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS validation_expected_conditions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, validation_run_id INTEGER NOT NULL,
    validation_preset_id TEXT, prompt_key TEXT, seed INTEGER, lora_weight REAL,
    hires_enabled INTEGER NOT NULL DEFAULT 0, width INTEGER, height INTEGER,
    sampler TEXT, steps INTEGER, cfg_scale REAL, condition_hash TEXT NOT NULL,
    expected_order INTEGER NOT NULL, preset_version TEXT, prompt TEXT,
    webui_prompt TEXT, negative_prompt TEXT, trigger_word TEXT,
    lora_filename TEXT, base_model TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS validation_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, selected_output_id INTEGER,
    expected_condition_id INTEGER, validation_run_id INTEGER, validation_preset_id TEXT, prompt_key TEXT, seed INTEGER,
    lora_weight REAL, image_path TEXT NOT NULL, validation_type TEXT, prompt TEXT, negative_prompt TEXT,
    base_model TEXT, sampler TEXT, steps INTEGER, cfg_scale REAL, width INTEGER, height INTEGER,
    hires_enabled INTEGER NOT NULL DEFAULT 0, hires_scale REAL, lora_weights TEXT, seeds TEXT,
    grid_image_flag INTEGER NOT NULL DEFAULT 0, image_role TEXT NOT NULL DEFAULT 'individual',
    condition_hash TEXT, ignored INTEGER NOT NULL DEFAULT 0,
    rating_face INTEGER, rating_costume INTEGER, rating_style INTEGER,
    rating_stability INTEGER, rating_flexibility INTEGER, rating_overall INTEGER,
    strength_label TEXT, overfit_level TEXT, adoption_label TEXT,
    failure_tags_json TEXT, rubric_version TEXT,
    recommended_weight_min REAL, recommended_weight_max REAL,
    memo TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS validation_weight_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, selected_output_id INTEGER,
    validation_run_id INTEGER, validation_preset_id TEXT, hires_enabled INTEGER NOT NULL DEFAULT 0,
    lora_weight REAL NOT NULL, validation_type TEXT, rating_face INTEGER, rating_costume INTEGER,
    rating_style INTEGER, rating_stability INTEGER, rating_flexibility INTEGER, rating_overall INTEGER,
    strength_label TEXT, overfit_level TEXT, adoption_label TEXT,
    failure_tags_json TEXT, rubric_version TEXT,
    recommended_weight_min REAL, recommended_weight_max REAL, memo TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS selected_lora_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, job_id INTEGER NOT NULL, selected_output_id INTEGER,
    profile_name TEXT NOT NULL, trigger_word TEXT, selected_epoch INTEGER, selected_model_path TEXT,
    exported_model_path TEXT, base_model TEXT, recommended_weight_min REAL,
    recommended_weight_max REAL, light_weight REAL, strong_weight REAL,
    validation_memo TEXT, library_memo TEXT, default_validation_preset_id TEXT,
    last_validation_preset_id TEXT, validation_policy_memo TEXT, reference_set_id INTEGER,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reference_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, project_id INTEGER,
    dataset_id INTEGER, dataset_version_id INTEGER, current_dataset_version_id INTEGER,
    current_version_id INTEGER, reference_type TEXT NOT NULL DEFAULT 'character',
    selection_mode TEXT NOT NULL DEFAULT 'manual', trigger_word TEXT, description TEXT,
    is_default INTEGER NOT NULL DEFAULT 0, is_archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, memo TEXT
);
CREATE TABLE IF NOT EXISTS reference_set_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, reference_set_id INTEGER NOT NULL,
    version_no INTEGER NOT NULL, dataset_id INTEGER, dataset_version_id INTEGER,
    trigger_word TEXT, reference_type TEXT, image_count INTEGER NOT NULL DEFAULT 0,
    roles_json TEXT, completeness_label TEXT, completeness_message TEXT,
    locked_at TEXT, memo TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE(reference_set_id, version_no)
);
CREATE TABLE IF NOT EXISTS reference_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT, reference_set_id INTEGER NOT NULL,
    reference_set_version_id INTEGER, dataset_id INTEGER, dataset_version_id INTEGER,
    image_path TEXT NOT NULL, source_type TEXT NOT NULL DEFAULT 'manual',
    source_image_path TEXT, image_role TEXT NOT NULL DEFAULT 'other',
    prompt_role_hint TEXT, caption TEXT, caption_snapshot TEXT, tags_json TEXT,
    width INTEGER, height INTEGER, file_size INTEGER, sha256 TEXT,
    include_in_machine_review INTEGER NOT NULL DEFAULT 1, exclude_reason TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT, memo TEXT
);
CREATE TABLE IF NOT EXISTS embedding_models (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, provider TEXT NOT NULL,
    model_name TEXT, pretrained TEXT, embedding_type TEXT NOT NULL DEFAULT 'image',
    vector_dim INTEGER, normalize INTEGER NOT NULL DEFAULT 1,
    device_default TEXT NOT NULL DEFAULT 'auto', dtype_default TEXT NOT NULL DEFAULT 'fp32',
    batch_size_default INTEGER NOT NULL DEFAULT 8, allow_download INTEGER NOT NULL DEFAULT 0,
    is_builtin INTEGER NOT NULL DEFAULT 0, is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, memo TEXT
);
CREATE TABLE IF NOT EXISTS embedding_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, active_embedding_model_id TEXT,
    python_path TEXT, device TEXT NOT NULL DEFAULT 'auto', dtype TEXT NOT NULL DEFAULT 'fp32',
    batch_size INTEGER NOT NULL DEFAULT 8, cache_root TEXT NOT NULL,
    allow_model_download INTEGER NOT NULL DEFAULT 0, max_image_size INTEGER NOT NULL DEFAULT 1024,
    num_workers INTEGER NOT NULL DEFAULT 1, last_preflight_status TEXT,
    last_preflight_message TEXT, last_preflight_json TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS image_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source_type TEXT NOT NULL,
    source_id INTEGER, source_path TEXT NOT NULL, project_id INTEGER,
    dataset_id INTEGER, dataset_version_id INTEGER, reference_set_id INTEGER,
    reference_set_version_id INTEGER, job_id INTEGER, validation_run_id INTEGER,
    embedding_model_id TEXT NOT NULL, provider TEXT, model_name TEXT,
    embedding_type TEXT NOT NULL DEFAULT 'image', vector_dim INTEGER,
    normalized INTEGER NOT NULL DEFAULT 1, embedding_path TEXT,
    image_sha256 TEXT, image_file_size INTEGER, image_mtime REAL,
    width INTEGER, height INTEGER, status TEXT NOT NULL DEFAULT 'ready',
    error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embedding_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_type TEXT NOT NULL, target_id INTEGER,
    embedding_model_id TEXT, provider TEXT, status TEXT NOT NULL DEFAULT 'planned',
    total_count INTEGER NOT NULL DEFAULT 0, processed_count INTEGER NOT NULL DEFAULT 0,
    ready_count INTEGER NOT NULL DEFAULT 0, failed_count INTEGER NOT NULL DEFAULT 0,
    skipped_count INTEGER NOT NULL DEFAULT 0, log_path TEXT, process_id INTEGER,
    started_at TEXT, ended_at TEXT, elapsed_seconds INTEGER, return_code INTEGER,
    error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embedding_job_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT, embedding_job_id INTEGER NOT NULL,
    source_type TEXT NOT NULL, source_id INTEGER, source_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', embedding_id INTEGER,
    error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendation_rules (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, recommendation_type TEXT NOT NULL,
    priority TEXT NOT NULL, description TEXT, enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER, source_job_id INTEGER NOT NULL,
    source_profile_id INTEGER, recommendation_type TEXT NOT NULL, priority TEXT NOT NULL,
    title TEXT NOT NULL, summary TEXT, reason TEXT, suggested_params_json TEXT,
    expected_effect TEXT, risk_note TEXT, created_job_id INTEGER,
    status TEXT NOT NULL DEFAULT 'proposed', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS caption_edit_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id INTEGER NOT NULL, action TEXT NOT NULL,
    trigger_word TEXT, changed_count INTEGER, skipped_count INTEGER, backup_path TEXT,
    created_at TEXT NOT NULL, memo TEXT
);
"""
