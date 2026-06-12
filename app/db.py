from __future__ import annotations

import json
import sqlite3
import subprocess
import hashlib
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
        run_migrations(conn)
        seed_app_settings(conn)
        seed_presets(conn)
        seed_sample_prompt_templates(conn)
        seed_evaluation_rubrics(conn)
        seed_validation_presets(conn)
        seed_job12_validation_defaults(conn)
        seed_legacy_validation_run(conn)
        import_latest_environment(conn)


def run_migrations(conn: sqlite3.Connection) -> None:
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
            "trigger_word_at_creation": "TEXT",
            "trigger_occurrence_count_at_creation": "INTEGER",
            "trigger_occurrence_rate_at_creation": "REAL",
            "trigger_consistency_label_at_creation": "TEXT",
            "dataset_version_id": "INTEGER",
            "updated_at": "TEXT",
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
    ensure_columns(conn, "training_outputs", {"selected": "INTEGER NOT NULL DEFAULT 0", "memo": "TEXT", "metadata_error": "TEXT"})
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
        },
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
        CREATE UNIQUE INDEX IF NOT EXISTS idx_training_metrics_job_step_tag
            ON training_metrics(job_id, step, raw_tag);
        CREATE INDEX IF NOT EXISTS idx_training_jobs_status
            ON training_jobs(status);
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
        CREATE INDEX IF NOT EXISTS idx_reference_images_set
            ON reference_images(reference_set_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_selected_lora_profiles_job_output
            ON selected_lora_profiles(job_id, selected_output_id);
        CREATE INDEX IF NOT EXISTS idx_experiment_recommendations_job
            ON experiment_recommendations(source_job_id, status);
        """
    )


def ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def seed_app_settings(conn: sqlite3.Connection) -> None:
    now = utc_now()
    values = {
        "app_name": settings.APP_NAME,
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
            "ratings": ["rating_face", "rating_costume", "rating_style", "rating_stability", "rating_overall"],
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


def seed_job12_validation_defaults(conn: sqlite3.Connection) -> None:
    now = utc_now()
    preset = conn.execute("SELECT id FROM validation_presets WHERE id = 'standard_validation_v1'").fetchone()
    profile = conn.execute("SELECT id FROM selected_lora_profiles WHERE id = 1 AND job_id = 12").fetchone()
    if preset and profile:
        conn.execute(
            """
            UPDATE selected_lora_profiles
            SET default_validation_preset_id = COALESCE(default_validation_preset_id, 'standard_validation_v1'),
                validation_policy_memo = COALESCE(
                    NULLIF(validation_policy_memo, ''),
                    '通常比較はHiresなしのStandard Validationを基準にする。HiresありはExtended Validationで最終見栄え確認として扱う。'
                ),
                updated_at = ?
            WHERE id = 1 AND job_id = 12
            """,
            (now,),
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
    with connect() as conn:
        return list(conn.execute(query, params).fetchall())


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(query, params).fetchone()


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
    params = data.get("params") or json.loads(preset["params_json"])
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
                name, dataset_id, preset_id, environment_id, status, model_family,
                training_script, base_model_path, vae_path, output_name, output_dir,
                run_dir, params_json, memo, created_at, updated_at
                , parent_job_id, sample_prompt_template_id
                , trigger_word_at_creation, trigger_occurrence_count_at_creation,
                trigger_occurrence_rate_at_creation, trigger_consistency_label_at_creation,
                dataset_version_id
            ) VALUES (?, ?, ?, NULL, 'draft', ?, ?, ?, ?, ?, '', '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["name"], int(data["dataset_id"]), data["preset_id"],
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
                seed, cfg_scale, steps, sort_order, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    now,
                )
                for index, item in enumerate(prompts, start=1)
            ],
        )


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
CREATE TABLE IF NOT EXISTS training_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, dataset_id INTEGER, preset_id TEXT,
    environment_id INTEGER, status TEXT NOT NULL, model_family TEXT NOT NULL, training_script TEXT NOT NULL,
    base_model_path TEXT NOT NULL, vae_path TEXT, output_name TEXT NOT NULL, output_dir TEXT NOT NULL,
    run_dir TEXT NOT NULL, params_json TEXT NOT NULL, command_line TEXT, process_id INTEGER,
    return_code INTEGER, start_time TEXT, end_time TEXT, elapsed_seconds INTEGER, adopted_epoch INTEGER,
    adopted_model_path TEXT, image_rating INTEGER, loss_health_label TEXT, memo TEXT,
    parent_job_id INTEGER, sample_prompt_template_id TEXT,
    trigger_word_at_creation TEXT, trigger_occurrence_count_at_creation INTEGER,
    trigger_occurrence_rate_at_creation REAL, trigger_consistency_label_at_creation TEXT,
    dataset_version_id INTEGER,
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
    sort_order INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sample_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, prompt_id INTEGER, epoch INTEGER,
    step INTEGER, image_path TEXT NOT NULL, prompt TEXT, negative_prompt TEXT, seed INTEGER,
    width INTEGER, height INTEGER, cfg_scale REAL, steps INTEGER, rating INTEGER,
    rating_face INTEGER, rating_costume INTEGER, rating_style INTEGER,
    rating_stability INTEGER, rating_overall INTEGER, memo TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, epoch INTEGER, step INTEGER,
    file_path TEXT NOT NULL, file_type TEXT NOT NULL, file_size INTEGER, sha256 TEXT,
    selected INTEGER NOT NULL DEFAULT 0, memo TEXT, metadata_error TEXT, created_at TEXT NOT NULL
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
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL,
    selected_output_id INTEGER, selected_lora_profile_id INTEGER,
    validation_preset_id TEXT, name TEXT NOT NULL, validation_level TEXT,
    base_model TEXT, trigger_word TEXT, lora_filename TEXT,
    recommended_weight_min REAL, recommended_weight_max REAL,
    suggested_weight_min REAL, suggested_weight_max REAL,
    suggested_light_weight REAL, suggested_strong_weight REAL,
    suggested_weight_reason TEXT, profile_applied_at TEXT,
    expected_image_count INTEGER, actual_image_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'planned', created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, memo TEXT
);
CREATE TABLE IF NOT EXISTS validation_expected_conditions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, validation_run_id INTEGER NOT NULL,
    validation_preset_id TEXT, prompt_key TEXT, seed INTEGER, lora_weight REAL,
    hires_enabled INTEGER NOT NULL DEFAULT 0, width INTEGER, height INTEGER,
    sampler TEXT, steps INTEGER, cfg_scale REAL, condition_hash TEXT NOT NULL,
    expected_order INTEGER NOT NULL, created_at TEXT NOT NULL
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
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, selected_output_id INTEGER,
    profile_name TEXT NOT NULL, trigger_word TEXT, selected_epoch INTEGER, selected_model_path TEXT,
    exported_model_path TEXT, base_model TEXT, recommended_weight_min REAL,
    recommended_weight_max REAL, light_weight REAL, strong_weight REAL,
    validation_memo TEXT, library_memo TEXT, default_validation_preset_id TEXT,
    last_validation_preset_id TEXT, validation_policy_memo TEXT, reference_set_id INTEGER,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reference_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, dataset_id INTEGER,
    dataset_version_id INTEGER, trigger_word TEXT, description TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, memo TEXT
);
CREATE TABLE IF NOT EXISTS reference_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT, reference_set_id INTEGER NOT NULL,
    image_path TEXT NOT NULL, image_role TEXT NOT NULL DEFAULT 'other',
    caption TEXT, sort_order INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendation_rules (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, recommendation_type TEXT NOT NULL,
    priority TEXT NOT NULL, description TEXT, enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source_job_id INTEGER NOT NULL,
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
