from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import connect, utc_now


def seed_zuihou_project() -> None:
    """One-off local migration for the original zuihou beta data.

    This script intentionally keeps the Acncagua-local IDs outside normal
    init_db() so new installations never receive project/profile changes based
    on hard-coded Job/Dataset/Profile IDs.
    """
    with connect() as conn:
        job_ids = [10, 12, 13]
        jobs = conn.execute(
            f"SELECT * FROM training_jobs WHERE id IN ({','.join('?' for _ in job_ids)}) ORDER BY id",
            tuple(job_ids),
        ).fetchall()
        if not jobs:
            print("No zuihou beta jobs found. Nothing to do.")
            return
        selected_job = conn.execute("SELECT * FROM training_jobs WHERE id = 12").fetchone() or jobs[-1]
        selected_output = conn.execute(
            "SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1 ORDER BY id DESC LIMIT 1",
            (selected_job["id"],),
        ).fetchone()
        selected_profile = conn.execute("SELECT * FROM selected_lora_profiles WHERE id = 1").fetchone()
        dataset_id = 4
        version_id = 2
        dataset = conn.execute("SELECT id, name, trigger_word FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        existing = conn.execute(
            "SELECT id FROM lora_projects WHERE selected_job_id = 12 OR name IN ('zuihou_v1', 'Dataset #4 zuihou LoRA') ORDER BY id LIMIT 1"
        ).fetchone()
        now = utc_now()
        if existing:
            project_id = int(existing["id"])
            conn.execute(
                """
                UPDATE lora_projects
                SET name = COALESCE(NULLIF(name, ''), 'zuihou_v1'),
                    dataset_id = COALESCE(dataset_id, ?),
                    current_dataset_version_id = COALESCE(current_dataset_version_id, ?),
                    trigger_word = COALESCE(NULLIF(trigger_word, ''), 'zuihou'),
                    base_model_path = COALESCE(NULLIF(base_model_path, ''), ?),
                    status = CASE WHEN status = 'draft' THEN 'selected' ELSE status END,
                    selected_job_id = COALESCE(selected_job_id, ?),
                    selected_output_id = COALESCE(selected_output_id, ?),
                    selected_lora_profile_id = COALESCE(selected_lora_profile_id, ?),
                    recommended_weight_min = COALESCE(recommended_weight_min, ?),
                    recommended_weight_max = COALESCE(recommended_weight_max, ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    dataset_id,
                    version_id,
                    selected_job["base_model_path"] or "",
                    selected_job["id"],
                    selected_output["id"] if selected_output else None,
                    selected_profile["id"] if selected_profile else None,
                    selected_profile["recommended_weight_min"] if selected_profile else 0.6,
                    selected_profile["recommended_weight_max"] if selected_profile else 0.8,
                    now,
                    project_id,
                ),
            )
        else:
            cur = conn.execute(
                """
                INSERT INTO lora_projects(
                    name, description, dataset_id, current_dataset_version_id, trigger_word,
                    base_model_path, status, selected_job_id, selected_output_id,
                    selected_lora_profile_id, recommended_weight_min, recommended_weight_max,
                    created_at, updated_at, memo
                )
                VALUES (?, ?, ?, ?, ?, ?, 'selected', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "zuihou_v1" if dataset else "Dataset #4 zuihou LoRA",
                    "既存のPilot/Standard/Validationをまとめた移行Project",
                    dataset_id,
                    version_id,
                    "zuihou",
                    selected_job["base_model_path"] or "",
                    selected_job["id"],
                    selected_output["id"] if selected_output else None,
                    selected_profile["id"] if selected_profile else None,
                    selected_profile["recommended_weight_min"] if selected_profile else 0.6,
                    selected_profile["recommended_weight_max"] if selected_profile else 0.8,
                    now,
                    now,
                    "One-off zuihou beta migration",
                ),
            )
            project_id = int(cur.lastrowid)
        for jid in job_ids:
            conn.execute("UPDATE training_jobs SET project_id = ?, updated_at = ? WHERE id = ?", (project_id, now, jid))
        conn.execute("UPDATE selected_lora_profiles SET project_id = ? WHERE id = 1", (project_id,))
        conn.execute("UPDATE validation_runs SET project_id = ? WHERE id = 2", (project_id,))
        conn.execute("UPDATE experiment_recommendations SET project_id = ? WHERE source_job_id IN (10, 12, 13)", (project_id,))
        print(f"zuihou project migration complete: project_id={project_id}")


if __name__ == "__main__":
    seed_zuihou_project()
