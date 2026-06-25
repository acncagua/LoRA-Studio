from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
FIXTURE_DIR = ROOT_DIR / "demo" / "fixtures"
IMAGE_FIXTURE_DIR = FIXTURE_DIR / "images"
REPORT_FIXTURE_DIR = FIXTURE_DIR / "reports"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_fixture_image(target: Path, fixture_index: int) -> dict[str, Any]:
    source = IMAGE_FIXTURE_DIR / f"demo_image_{fixture_index:02d}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {"path": target, "size": target.stat().st_size, "sha256": sha256_file(target)}


def insert(conn: sqlite3.Connection, table: str, values: dict[str, Any]) -> int:
    columns = ", ".join(values.keys())
    placeholders = ", ".join("?" for _ in values)
    cursor = conn.execute(f"INSERT INTO {table}({columns}) VALUES ({placeholders})", tuple(values.values()))
    return int(cursor.lastrowid)


def jdump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def condition_hash(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def configure_demo_environment(output: Path) -> None:
    os.environ["LORA_STUDIO_DB"] = str(output.resolve())
    os.environ["LORA_STUDIO_DEMO_MODE"] = "1"


def init_empty_db(output: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{output}{suffix}")
        if path.exists():
            path.unlink()
    configure_demo_environment(output)
    from app import settings
    from app.db import init_db

    settings.DB_PATH = output.resolve()
    settings.DEMO_MODE = True
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db()


def create_demo_db(output: Path) -> dict[str, Any]:
    init_empty_db(output)

    from app.db import connect, utc_now

    data = json.loads((FIXTURE_DIR / "demo_data.json").read_text(encoding="utf-8"))
    now = utc_now()
    runtime_root = (ROOT_DIR / "demo" / "runtime").resolve()
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runs_root = runtime_root / "runs"
    exports_root = runtime_root / "exports"
    reports_root = runtime_root / "reports"
    validation_root = exports_root / "validation_runs"
    reference_root = exports_root / "reference_sets"
    job_run_dir = runs_root / "job_000001"
    job_models_dir = job_run_dir / "models"
    sample_dir = job_run_dir / "samples"
    reports_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPORT_FIXTURE_DIR / "review_matrix.html", reports_root / "review_matrix.html")
    shutil.copyfile(REPORT_FIXTURE_DIR / "validation_matrix.html", reports_root / "validation_matrix.html")

    with connect() as conn:
        for key, rel in {
            "runtime_root": runtime_root,
            "runs_root": runs_root,
            "exports_root": exports_root,
            "reports_root": reports_root,
        }.items():
            conn.execute(
                """
                INSERT INTO storage_locations(key, path, is_active, created_at, updated_at, check_status, error_message)
                VALUES (?, ?, 1, ?, ?, 'ok', 'Demo runtime')
                ON CONFLICT(key) DO UPDATE SET path = excluded.path, updated_at = excluded.updated_at, is_active = 1
                """,
                (key, str(rel), now, now),
            )

        dataset_id = insert(
            conn,
            "datasets",
            {
                "name": data["dataset"]["name"],
                "path": "<demo-runtime>/datasets/demo-character",
                "model_family": "SDXL",
                "trigger_word": data["dataset"]["trigger_word"],
                "class_token": "character",
                "image_count": data["dataset"]["image_count"],
                "caption_count": data["dataset"]["image_count"],
                "created_at": now,
                "updated_at": now,
            },
        )
        dataset_version_id = insert(
            conn,
            "dataset_versions",
            {
                "dataset_id": dataset_id,
                "version_no": 1,
                "trigger_word": data["dataset"]["trigger_word"],
                "image_count": data["dataset"]["image_count"],
                "caption_count": data["dataset"]["image_count"],
                "missing_caption_count": 0,
                "created_at": now,
            },
        )
        project_id = insert(
            conn,
            "lora_projects",
            {
                "name": data["project"]["name"],
                "description": data["project"]["description"],
                "dataset_id": dataset_id,
                "current_dataset_version_id": dataset_version_id,
                "trigger_word": data["dataset"]["trigger_word"],
                "base_model_path": "<demo-runtime>/models/demo_base_model.safetensors",
                "status": "completed",
                "recommended_weight_min": data["lora_profile"]["recommended_weight_min"],
                "recommended_weight_max": data["lora_profile"]["recommended_weight_max"],
                "post_training_review_mode": "standard_auto",
                "max_auto_images": 150,
                "max_auto_runtime_minutes": 240,
                "auto_review_provider": "demo",
                "include_neighbor_epochs": 1,
                "created_at": now,
                "updated_at": now,
                "memo": "Public screenshot demo project. All paths and images are synthetic.",
            },
        )
        ref_set_id = insert(
            conn,
            "reference_sets",
            {
                "name": data["reference_set"]["name"],
                "project_id": project_id,
                "dataset_id": dataset_id,
                "dataset_version_id": dataset_version_id,
                "current_dataset_version_id": dataset_version_id,
                "reference_type": "character",
                "selection_mode": "demo",
                "trigger_word": data["dataset"]["trigger_word"],
                "description": "Synthetic reference set for screenshots.",
                "is_default": 1,
                "created_at": now,
                "updated_at": now,
            },
        )
        ref_version_id = insert(
            conn,
            "reference_set_versions",
            {
                "reference_set_id": ref_set_id,
                "version_no": 1,
                "dataset_id": dataset_id,
                "dataset_version_id": dataset_version_id,
                "trigger_word": data["dataset"]["trigger_word"],
                "reference_type": "character",
                "image_count": 3,
                "roles_json": jdump(["face", "full_body", "expression"]),
                "completeness_label": "ok",
                "completeness_message": "Synthetic reference coverage is ready for screenshots.",
                "locked_at": now,
                "created_at": now,
                "updated_at": now,
            },
        )
        reference_image_ids: list[int] = []
        for index, role in enumerate(("face", "full_body", "expression"), start=1):
            copied = copy_fixture_image(reference_root / "reference_set_000001" / "images" / f"reference_{index:02d}.png", index)
            reference_image_ids.append(
                insert(
                    conn,
                    "reference_images",
                    {
                        "reference_set_id": ref_set_id,
                        "reference_set_version_id": ref_version_id,
                        "dataset_id": dataset_id,
                        "dataset_version_id": dataset_version_id,
                        "image_path": str(copied["path"]),
                        "source_type": "demo_fixture",
                        "image_role": role,
                        "caption": f"demochar synthetic {role} reference",
                        "width": 320,
                        "height": 320,
                        "file_size": copied["size"],
                        "sha256": copied["sha256"],
                        "sort_order": index,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
            )

        params = {
            "pretrained_model_name_or_path": "<demo-runtime>/models/demo_base_model.safetensors",
            "train_batch_size": 1,
            "max_train_epochs": 10,
            "repeats": 14,
            "network_dim": 32,
            "network_alpha": 16,
            "optimizer_type": "AdamW8bit",
            "learning_rate": 0.0001,
            "unet_lr": 0.0001,
            "text_encoder_lr": 0,
            "sample_every_n_epochs": 1,
            "save_every_n_epochs": 1,
        }
        step_snapshot = {
            "image_count": data["dataset"]["image_count"],
            "repeats": 14,
            "max_train_epochs": 10,
            "effective_batch_size": 1,
            "steps_per_epoch": 588,
            "expected_total_steps": 5880,
            "target_steps_recommended": 5000,
            "status": "OK",
        }
        job_id = insert(
            conn,
            "training_jobs",
            {
                "project_id": project_id,
                "name": data["job"]["name"],
                "dataset_id": dataset_id,
                "preset_id": "sdxl_2d_face_adamw8bit_standard",
                "status": "completed",
                "model_family": "SDXL",
                "training_script": "sdxl_train_network.py",
                "base_model_path": "<demo-runtime>/models/demo_base_model.safetensors",
                "output_name": "demo_character_lora",
                "output_dir": str(job_models_dir),
                "run_dir": str(job_run_dir),
                "params_json": jdump(params),
                "command_line": "python sdxl_train_network.py --demo",
                "return_code": 0,
                "start_time": now,
                "end_time": now,
                "elapsed_seconds": 7200,
                "adopted_epoch": data["job"]["selected_epoch"],
                "loss_health_label": "ok",
                "expected_total_steps_at_creation": step_snapshot["expected_total_steps"],
                "steps_per_epoch_at_creation": step_snapshot["steps_per_epoch"],
                "target_steps_recommended_at_creation": step_snapshot["target_steps_recommended"],
                "step_status_at_creation": "OK",
                "step_estimate_snapshot_json": jdump(step_snapshot),
                "repeats_auto_calculated": 1,
                "target_steps_source": "recipe",
                "post_training_review_mode": "standard_auto",
                "post_training_review_status": "completed",
                "post_training_review_message": "Demo standard candidate comparison completed.",
                "dataset_version_id": dataset_version_id,
                "recipe_v2_id": "sdxl_character_face_adamw8bit_balanced",
                "optimizer_definition_id": "adamw8bit",
                "optimizer_profile_id": "adamw8bit_sdxl_balanced",
                "network_type_id": "standard_lora",
                "training_purpose_id": "character_face",
                "recipe_snapshot_json": jdump({"display_name": "Demo Character Face / AdamW8bit Balanced"}),
                "params_snapshot_json": jdump(params),
                "user_overrides_json": jdump({}),
                "created_at": now,
                "updated_at": now,
            },
        )

        output_ids_by_epoch: dict[int, int] = {}
        for epoch in data["job"]["candidate_epochs"]:
            model_path = job_models_dir / f"demo_character_epoch_{epoch:06d}.safetensors"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(f"DEMO SAFETENSORS PLACEHOLDER epoch={epoch}\n".encode("utf-8"))
            output_ids_by_epoch[epoch] = insert(
                conn,
                "training_outputs",
                {
                    "job_id": job_id,
                    "epoch": epoch,
                    "step": epoch * step_snapshot["steps_per_epoch"],
                    "file_path": str(model_path),
                    "file_type": "model",
                    "file_size": model_path.stat().st_size,
                    "sha256": sha256_file(model_path),
                    "selected": 1 if epoch == data["job"]["selected_epoch"] else 0,
                    "memo": "Synthetic demo LoRA artifact placeholder.",
                    "created_at": now,
                },
            )
            conn.execute(
                """
                INSERT INTO training_epoch_candidate_summaries(job_id, epoch, candidate_rank, candidate_label, score, reason_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, epoch, data["job"]["candidate_epochs"].index(epoch) + 1, "candidate", 0.9 - abs(epoch - 6) * 0.05, jdump(["demo loss candidate"]), now, now),
            )
        selected_output_id = output_ids_by_epoch[data["job"]["selected_epoch"]]

        prompt_id = insert(
            conn,
            "sample_prompts",
            {
                "job_id": job_id,
                "name": "Demo prompt",
                "prompt": "demochar, portrait, clean studio lighting",
                "negative_prompt": "low quality, blurry",
                "width": 1024,
                "height": 1024,
                "seed": 111111,
                "cfg_scale": 7,
                "steps": 24,
                "sort_order": 1,
                "prompt_role": "basic_face",
                "created_at": now,
            },
        )
        for index, epoch in enumerate(data["job"]["candidate_epochs"], start=1):
            copied = copy_fixture_image(sample_dir / f"sample_epoch_{epoch:06d}.png", index)
            insert(
                conn,
                "sample_images",
                {
                    "job_id": job_id,
                    "prompt_id": prompt_id,
                    "epoch": epoch,
                    "step": epoch * step_snapshot["steps_per_epoch"],
                    "image_path": str(copied["path"]),
                    "prompt": "demochar, portrait, clean studio lighting",
                    "negative_prompt": "low quality, blurry",
                    "seed": 111111,
                    "width": 320,
                    "height": 320,
                    "cfg_scale": 7,
                    "steps": 24,
                    "rating_overall": 4 if epoch == 6 else 3,
                    "memo": "Synthetic sample for screenshots.",
                    "created_at": now,
                },
            )

        review_dir = exports_root / "review_sessions" / "review_session_000001"
        review_session_id = insert(
            conn,
            "review_sessions",
            {
                "job_id": job_id,
                "project_id": project_id,
                "reference_set_id": ref_set_id,
                "reference_set_version_id": ref_version_id,
                "dataset_id": dataset_id,
                "dataset_version_id": dataset_version_id,
                "embedding_model_id": "mock_image_512",
                "name": data["review_session"]["name"],
                "preset_id": "quick_candidate_review_v1",
                "preset_snapshot_json": jdump({"demo": True}),
                "candidate_epochs_json": jdump(data["job"]["candidate_epochs"]),
                "prompt_keys_json": jdump(["basic_face", "full_body", "expression_pose"]),
                "weights_json": jdump([0.6, 0.8]),
                "seed": 111111,
                "expected_image_count": data["review_session"]["expected_image_count"],
                "generated_image_count": data["review_session"]["expected_image_count"],
                "imported_image_count": data["review_session"]["expected_image_count"],
                "scored_image_count": data["review_session"]["expected_image_count"],
                "status": "completed",
                "run_dir": str(review_dir),
                "output_dir": str(review_dir / "images"),
                "matrix_path": str(reports_root / "review_matrix.html"),
                "log_path": str(review_dir / "review_preparation.log"),
                "review_plan_kind": "quick_auto",
                "automation_mode": "quick_auto",
                "automation_status": "completed",
                "machine_assist_summary_json": jdump({"confidence": "medium", "candidate_group": [6, 8], "human_rating_preferred": True}),
                "started_at": now,
                "ended_at": now,
                "elapsed_seconds": 420,
                "stage_timing_json": jdump({"generation_elapsed_seconds": 240, "embedding_elapsed_seconds": 60, "machine_review_elapsed_seconds": 45}),
                "created_at": now,
                "updated_at": now,
                "memo": "Synthetic candidate review for README screenshots.",
            },
        )
        prompt_keys = ["basic_face", "full_body", "expression_pose"]
        image_counter = 0
        for epoch in data["job"]["candidate_epochs"]:
            for prompt_key in prompt_keys:
                for weight in (0.6, 0.8):
                    image_counter += 1
                    out_id = output_ids_by_epoch[epoch]
                    lora_path = job_models_dir / f"demo_character_epoch_{epoch:06d}.safetensors"
                    cond_id = insert(
                        conn,
                        "review_session_conditions",
                        {
                            "review_session_id": review_session_id,
                            "job_id": job_id,
                            "epoch": epoch,
                            "output_id": out_id,
                            "lora_path": str(lora_path),
                            "prompt_key": prompt_key,
                            "prompt_role": prompt_key,
                            "prompt": f"demochar, {prompt_key}, screenshot-safe demo image",
                            "negative_prompt": "low quality, blurry",
                            "seed": 111111,
                            "lora_weight": weight,
                            "hires_enabled": 0,
                            "width": 1024,
                            "height": 1024,
                            "sampler": "euler_a",
                            "steps": 24,
                            "cfg_scale": 7,
                            "condition_hash": condition_hash("review", epoch, prompt_key, weight),
                            "expected_order": image_counter,
                            "status": "completed",
                            "created_at": now,
                            "updated_at": now,
                        },
                    )
                    copied = copy_fixture_image(review_dir / "images" / f"review_{image_counter:03d}.png", ((image_counter - 1) % 5) + 1)
                    image_id = insert(
                        conn,
                        "review_session_images",
                        {
                            "review_session_id": review_session_id,
                            "condition_id": cond_id,
                            "job_id": job_id,
                            "epoch": epoch,
                            "output_id": out_id,
                            "prompt_key": prompt_key,
                            "prompt_role": prompt_key,
                            "seed": 111111,
                            "lora_weight": weight,
                            "image_path": str(copied["path"]),
                            "file_size": copied["size"],
                            "sha256": copied["sha256"],
                            "width": 320,
                            "height": 320,
                            "rating_overall": 4 if epoch == 6 else 3,
                            "strength_label": "usable",
                            "overfit_level": "low",
                            "adoption_label": "candidate" if epoch == 6 else "compare",
                            "rubric_version": "demo",
                            "created_at": now,
                            "updated_at": now,
                        },
                    )
                    score_id = insert_machine_review_score(conn, image_id, project_id, job_id, None, dataset_id, dataset_version_id, ref_set_id, ref_version_id, prompt_key, epoch, weight, reference_image_ids[0], now)
                    conn.execute("UPDATE review_session_images SET machine_review_score_id = ? WHERE id = ?", (score_id, image_id))
                    conn.execute("UPDATE review_session_conditions SET image_id = ?, image_path = ? WHERE id = ?", (image_id, str(copied["path"]), cond_id))

        candidate_run_ids: list[int] = []
        all_validation_run_ids: list[int] = []
        for epoch in data["job"]["candidate_epochs"]:
            run_id = create_validation_run_rows(
                conn,
                project_id=project_id,
                job_id=job_id,
                output_id=output_ids_by_epoch[epoch],
                epoch=epoch,
                name=f"Standard Candidate Comparison epoch {epoch}",
                kind="candidate_standard_comparison",
                root=validation_root / f"validation_run_{len(all_validation_run_ids) + 1:06d}",
                reports_root=reports_root,
                now=now,
                reference_image_id=reference_image_ids[0],
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
                ref_set_id=ref_set_id,
                ref_version_id=ref_version_id,
                profile_id=None,
            )
            candidate_run_ids.append(run_id)
            all_validation_run_ids.append(run_id)

        profile_id = insert(
            conn,
            "selected_lora_profiles",
            {
                "project_id": project_id,
                "job_id": job_id,
                "selected_output_id": selected_output_id,
                "profile_name": data["lora_profile"]["profile_name"],
                "trigger_word": data["dataset"]["trigger_word"],
                "selected_epoch": data["job"]["selected_epoch"],
                "selected_model_path": str(job_models_dir / "demo_character_epoch_000006.safetensors"),
                "exported_model_path": "<demo-runtime>/exports/selected_loras/demo_character_lora.safetensors",
                "base_model": "<demo-runtime>/models/demo_base_model.safetensors",
                "recommended_weight_min": data["lora_profile"]["recommended_weight_min"],
                "recommended_weight_max": data["lora_profile"]["recommended_weight_max"],
                "light_weight": 0.6,
                "strong_weight": 0.8,
                "validation_memo": "Synthetic weight calibration recommends 0.6-0.8.",
                "library_memo": "Public demo LoRA profile.",
                "default_validation_preset_id": "standard_validation_v1",
                "last_validation_preset_id": "standard_validation_v1",
                "validation_policy_memo": "Use for screenshot/demo only.",
                "reference_set_id": ref_set_id,
                "created_at": now,
                "updated_at": now,
            },
        )
        weight_run_id = create_validation_run_rows(
            conn,
            project_id=project_id,
            job_id=job_id,
            output_id=selected_output_id,
            epoch=data["job"]["selected_epoch"],
            name=data["validation_run"]["name"],
            kind="weight_calibration",
            root=validation_root / f"validation_run_{len(all_validation_run_ids) + 1:06d}",
            reports_root=reports_root,
            now=now,
            reference_image_id=reference_image_ids[0],
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            ref_set_id=ref_set_id,
            ref_version_id=ref_version_id,
            profile_id=profile_id,
        )
        all_validation_run_ids.append(weight_run_id)
        for weight in (0.0, 0.4, 0.6, 0.8, 1.0):
            insert(
                conn,
                "validation_weight_reviews",
                {
                    "job_id": job_id,
                    "selected_output_id": selected_output_id,
                    "validation_run_id": weight_run_id,
                    "validation_preset_id": "standard_validation_v1",
                    "hires_enabled": 0,
                    "lora_weight": weight,
                    "validation_type": "weight_calibration",
                    "rating_overall": 4 if weight in (0.6, 0.8) else 3,
                    "strength_label": "recommended" if weight in (0.6, 0.8) else "compare",
                    "overfit_level": "low",
                    "rubric_version": "demo",
                    "recommended_weight_min": 0.6,
                    "recommended_weight_max": 0.8,
                    "memo": "Synthetic demo weight review.",
                    "created_at": now,
                    "updated_at": now,
                },
            )

        group_id = insert(
            conn,
            "candidate_comparison_groups",
            {
                "job_id": job_id,
                "project_id": project_id,
                "preset_id": "standard_validation_v1",
                "name": data["candidate_comparison"]["name"],
                "candidate_epochs_json": jdump(data["job"]["candidate_epochs"]),
                "validation_run_ids_json": jdump(candidate_run_ids),
                "expected_total_images": data["candidate_comparison"]["expected_total_images"],
                "registered_image_count": data["candidate_comparison"]["expected_total_images"],
                "embedding_ready_count": data["candidate_comparison"]["expected_total_images"],
                "machine_review_score_count": data["candidate_comparison"]["expected_total_images"],
                "status": "completed",
                "matrix_path": str(reports_root / "validation_matrix.html"),
                "estimate_json": jdump({"estimated_minutes": 60, "storage_mb": 180}),
                "stage_timing_json": jdump({"generation_elapsed_seconds": 3600, "embedding_elapsed_seconds": 240, "machine_review_elapsed_seconds": 180}),
                "started_at": now,
                "ended_at": now,
                "elapsed_seconds": 4200,
                "created_at": now,
                "updated_at": now,
                "memo": "Synthetic standard comparison group.",
            },
        )

        conn.execute(
            """
            UPDATE lora_projects
            SET selected_job_id = ?, selected_output_id = ?, selected_lora_profile_id = ?,
                default_reference_set_id = ?, default_reference_set_version_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (job_id, selected_output_id, profile_id, ref_set_id, ref_version_id, now, project_id),
        )
        conn.execute("UPDATE validation_runs SET selected_lora_profile_id = ? WHERE id = ?", (profile_id, weight_run_id))
        conn.execute(
            """
            UPDATE training_jobs
            SET adopted_model_path = ?, adopted_epoch = ?, updated_at = ?
            WHERE id = ?
            """,
            (str(job_models_dir / "demo_character_epoch_000006.safetensors"), data["job"]["selected_epoch"], now, job_id),
        )

        for profile_key, status in data["optimizer_statuses"].items():
            mapped_status = "smoke_ok" if status == "smoke_ok" else "smoke_failed"
            result_id = insert(
                conn,
                "optimizer_profile_test_results",
                {
                    "optimizer_profile_id": profile_key,
                    "recipe_id": None,
                    "test_type": "image_smoke",
                    "status": "ok" if status == "smoke_ok" else "failed",
                    "return_code": 0 if status == "smoke_ok" else 1,
                    "elapsed_seconds": 180,
                    "error_message": "Demo warning: tune before production use." if status != "smoke_ok" else "",
                    "sd_scripts_commit": "demo",
                    "torch_version": "demo",
                    "cuda_version": "demo",
                    "created_at": now,
                    "memo": "Synthetic optimizer status for screenshots.",
                },
            )
            conn.execute(
                """
                UPDATE optimizer_profiles_v2
                SET validation_status = ?, last_tested_at = ?, last_test_result_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (mapped_status, now, result_id, now, profile_key),
            )

        conn.commit()

    return {
        "db": str(output.resolve()),
        "runtime_root": str(runtime_root),
        "project_id": project_id,
        "job_id": job_id,
        "review_session_id": review_session_id,
        "candidate_comparison_group_id": group_id,
        "candidate_validation_run_ids": candidate_run_ids,
        "weight_calibration_run_id": weight_run_id,
        "lora_profile_id": profile_id,
    }


def insert_machine_review_score(
    conn: sqlite3.Connection,
    source_id: int,
    project_id: int,
    job_id: int,
    validation_run_id: int | None,
    dataset_id: int,
    dataset_version_id: int,
    ref_set_id: int,
    ref_version_id: int,
    prompt_key: str,
    epoch: int,
    weight: float,
    reference_image_id: int,
    now: str,
) -> int:
    score = 0.82 - abs(epoch - 6) * 0.03 + (0.02 if weight in (0.6, 0.8) else 0)
    return insert(
        conn,
        "machine_review_scores",
        {
            "source_type": "validation_image" if validation_run_id else "review_session_image",
            "source_id": source_id,
            "project_id": project_id,
            "job_id": job_id,
            "validation_run_id": validation_run_id,
            "reference_set_id": ref_set_id,
            "reference_set_version_id": ref_version_id,
            "dataset_id": dataset_id,
            "dataset_version_id": dataset_version_id,
            "embedding_model_id": "mock_image_512",
            "provider": "demo",
            "prompt_key": prompt_key,
            "prompt_role": prompt_key,
            "epoch": epoch,
            "lora_weight": weight,
            "reference_similarity_avg": round(score, 3),
            "reference_similarity_max": round(score + 0.05, 3),
            "nearest_reference_image_id": reference_image_id,
            "nearest_reference_similarity": round(score + 0.05, 3),
            "dataset_similarity_avg": round(score - 0.04, 3),
            "nearest_dataset_similarity": round(score - 0.02, 3),
            "dataset_top1_margin": 0.08,
            "overfit_risk_label": "low",
            "assist_score": round(score, 3),
            "assist_label": "candidate" if epoch == 6 else "compare",
            "confidence_label": "medium",
            "reason_json": jdump(["Synthetic demo score", "Human review remains preferred"]),
            "created_at": now,
            "updated_at": now,
        },
    )


def create_validation_run_rows(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    job_id: int,
    output_id: int,
    epoch: int,
    name: str,
    kind: str,
    root: Path,
    reports_root: Path,
    now: str,
    reference_image_id: int,
    dataset_id: int,
    dataset_version_id: int,
    ref_set_id: int,
    ref_version_id: int,
    profile_id: int | None,
) -> int:
    image_dir = root / "images"
    run_id = insert(
        conn,
        "validation_runs",
        {
            "project_id": project_id,
            "job_id": job_id,
            "selected_output_id": output_id,
            "selected_lora_profile_id": profile_id,
            "validation_run_kind": kind,
            "source_training_job_id": job_id,
            "selected_epoch": epoch,
            "pipeline_status": "completed",
            "matrix_path": str(reports_root / "validation_matrix.html"),
            "validation_preset_id": "standard_validation_v1",
            "name": name,
            "validation_level": "standard",
            "base_model": "<demo-runtime>/models/demo_base_model.safetensors",
            "trigger_word": "demochar",
            "lora_filename": f"demo_character_epoch_{epoch:06d}.safetensors",
            "recommended_weight_min": 0.6,
            "recommended_weight_max": 0.8,
            "suggested_weight_min": 0.6,
            "suggested_weight_max": 0.8,
            "suggested_light_weight": 0.6,
            "suggested_strong_weight": 0.8,
            "suggested_weight_reason": "Synthetic demo calibration.",
            "stage_timing_json": jdump({"generation_elapsed_seconds": 900, "import_elapsed_seconds": 5, "embedding_elapsed_seconds": 60, "machine_review_elapsed_seconds": 45}),
            "expected_image_count": 45,
            "actual_image_count": 45,
            "status": "completed",
            "created_at": now,
            "updated_at": now,
            "memo": "Synthetic validation run for screenshots.",
        },
    )
    prompt_keys = ["basic_face", "full_body", "expression_pose"]
    seeds = [111111, 222222, 333333]
    weights = [0.0, 0.4, 0.6, 0.8, 1.0]
    order = 0
    for prompt_key in prompt_keys:
        for seed in seeds:
            for weight in weights:
                order += 1
                hash_value = condition_hash("validation", run_id, prompt_key, seed, weight)
                cond_id = insert(
                    conn,
                    "validation_expected_conditions",
                    {
                        "validation_run_id": run_id,
                        "validation_preset_id": "standard_validation_v1",
                        "prompt_key": prompt_key,
                        "seed": seed,
                        "lora_weight": weight,
                        "hires_enabled": 0,
                        "width": 1024,
                        "height": 1024,
                        "sampler": "euler_a",
                        "steps": 24,
                        "cfg_scale": 7,
                        "condition_hash": hash_value,
                        "expected_order": order,
                        "preset_version": "v1",
                        "prompt": f"demochar, {prompt_key}, validation screenshot image",
                        "webui_prompt": f"demochar, {prompt_key}, validation screenshot image",
                        "negative_prompt": "low quality, blurry",
                        "trigger_word": "demochar",
                        "lora_filename": f"demo_character_epoch_{epoch:06d}.safetensors",
                        "base_model": "<demo-runtime>/models/demo_base_model.safetensors",
                        "created_at": now,
                    },
                )
                copied = copy_fixture_image(image_dir / f"validation_{order:03d}.png", ((order - 1) % 5) + 1)
                image_id = insert(
                    conn,
                    "validation_images",
                    {
                        "job_id": job_id,
                        "selected_output_id": output_id,
                        "expected_condition_id": cond_id,
                        "validation_run_id": run_id,
                        "validation_preset_id": "standard_validation_v1",
                        "prompt_key": prompt_key,
                        "seed": seed,
                        "lora_weight": weight,
                        "image_path": str(copied["path"]),
                        "validation_type": kind,
                        "prompt": f"demochar, {prompt_key}, validation screenshot image",
                        "negative_prompt": "low quality, blurry",
                        "base_model": "<demo-runtime>/models/demo_base_model.safetensors",
                        "sampler": "euler_a",
                        "steps": 24,
                        "cfg_scale": 7,
                        "width": 320,
                        "height": 320,
                        "hires_enabled": 0,
                        "condition_hash": hash_value,
                        "rating_overall": 4 if weight in (0.6, 0.8) else 3,
                        "strength_label": "recommended" if weight in (0.6, 0.8) else "compare",
                        "overfit_level": "low",
                        "adoption_label": "recommended" if weight in (0.6, 0.8) and epoch == 6 else "compare",
                        "rubric_version": "demo",
                        "recommended_weight_min": 0.6,
                        "recommended_weight_max": 0.8,
                        "memo": "sd-scripts generated / demo fixture",
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                score_id = insert_machine_review_score(conn, image_id, project_id, job_id, run_id, dataset_id, dataset_version_id, ref_set_id, ref_version_id, prompt_key, epoch, weight, reference_image_id, now)
                conn.execute("UPDATE validation_images SET memo = memo || ' / score #' || ? WHERE id = ?", (score_id, image_id))
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sanitized LoRA-Studio demo database for screenshots.")
    parser.add_argument("--output", default="demo/demo.sqlite", help="Output SQLite database path.")
    args = parser.parse_args()
    output = (ROOT_DIR / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    summary = create_demo_db(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
