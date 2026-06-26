from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from app import settings
from app.db import connect, fetch_all, fetch_one, init_db, utc_now
from app.services.lora_artifacts import resolve_lora_artifact
from app.services.lora_comparisons import (
    build_lora_comparison_matrix_html,
    create_lora_comparison_session,
    load_lora_comparison_session,
    refresh_lora_comparison_session,
    run_parity_gate,
    save_lora_comparison_decision,
)
from app.services.storage_cleanup import exported_selected_preview


class Phase1253ComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp.name)
        settings.DATA_DIR = self.root / "data"
        settings.DB_PATH = settings.DATA_DIR / "app.db"
        settings.EXPORTS_DIR = self.root / "exports"
        settings.RUNS_DIR = self.root / "runs"
        settings.LOGS_DIR = self.root / "logs"
        settings.EMBEDDINGS_DIR = settings.DATA_DIR / "embeddings"
        for directory in (settings.DATA_DIR, settings.EXPORTS_DIR, settings.RUNS_DIR, settings.LOGS_DIR, settings.EMBEDDINGS_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        init_db()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_model(self, name: str, payload: bytes = b"lora") -> tuple[Path, str]:
        path = self.root / name
        path.write_bytes(payload)
        return path, hashlib.sha256(payload).hexdigest()

    def create_fixture_pair(self, with_images: bool = True) -> tuple[list[int], list[int], list[int]]:
        now = utc_now()
        standard_path, standard_sha = self.write_model("standard.safetensors", b"standard")
        c3_path, c3_sha = self.write_model("c3lier.safetensors", b"c3lier")
        with connect() as conn:
            project_id = int(
                conn.execute(
                    """
                    INSERT INTO lora_projects(name, status, base_model_path, trigger_word, created_at, updated_at)
                    VALUES('Demo Project', 'draft', 'base.safetensors', 'demo', ?, ?)
                    """,
                    (now, now),
                ).lastrowid
            )
            job_ids = []
            output_ids = []
            profile_ids = []
            for name, network_type, model_path, sha_value in [
                ("Standard LoRA", "standard_lora", standard_path, standard_sha),
                ("LoRA-C3Lier（セリア）", "lora_c3lier", c3_path, c3_sha),
            ]:
                job_id = int(
                    conn.execute(
                        """
                        INSERT INTO training_jobs(
                            project_id, name, dataset_id, status, model_family, training_script,
                            base_model_path, output_name, output_dir, run_dir, params_json,
                            dataset_version_id, optimizer_definition_id, optimizer_profile_id,
                            network_type_id, created_at, updated_at
                        )
                        VALUES(?, ?, 1, 'completed', 'SDXL', 'sdxl_train_network.py',
                               'base.safetensors', ?, ?, ?, '{}', 1, 'AdamW8bit',
                               'adamw8bit_sdxl_balanced', ?, ?, ?)
                        """,
                        (project_id, name, name, str(self.root), str(self.root), network_type, now, now),
                    ).lastrowid
                )
                output_id = int(
                    conn.execute(
                        """
                        INSERT INTO training_outputs(job_id, epoch, step, file_path, file_type, file_size, sha256, selected, created_at)
                        VALUES(?, 10, 1000, ?, 'model', ?, ?, 1, ?)
                        """,
                        (job_id, str(model_path), model_path.stat().st_size, sha_value, now),
                    ).lastrowid
                )
                profile_id = int(
                    conn.execute(
                        """
                        INSERT INTO selected_lora_profiles(
                            project_id, job_id, selected_output_id, profile_name, trigger_word,
                            selected_epoch, selected_model_path, base_model,
                            default_validation_preset_id, created_at, updated_at
                        )
                        VALUES(?, ?, ?, ?, 'demo', 10, ?, 'base.safetensors',
                               'standard_validation_v1', ?, ?)
                        """,
                        (project_id, job_id, output_id, name, str(model_path), now, now),
                    ).lastrowid
                )
                run_id = int(
                    conn.execute(
                        """
                        INSERT INTO validation_runs(
                            project_id, job_id, selected_output_id, selected_lora_profile_id,
                            validation_run_kind, source_training_job_id, selected_epoch,
                            pipeline_status, validation_preset_id, name, validation_level,
                            base_model, trigger_word, lora_filename, expected_image_count,
                            actual_image_count, status, preset_snapshot_json, created_at, updated_at, memo
                        )
                        VALUES(?, ?, ?, ?, 'weight_calibration', ?, 10, 'completed',
                               'standard_validation_v1', ?, 'standard', 'base.safetensors',
                               'demo', ?, 1, ?, 'images_registered', '{}', ?, ?, '')
                        """,
                        (project_id, job_id, output_id, profile_id, job_id, f"{name} Validation", model_path.name, 1 if with_images else 0, now, now),
                    ).lastrowid
                )
                condition_id = int(
                    conn.execute(
                        """
                        INSERT INTO validation_expected_conditions(
                            validation_run_id, validation_preset_id, prompt_key, seed,
                            lora_weight, hires_enabled, width, height, sampler, steps,
                            cfg_scale, condition_hash, expected_order, preset_version,
                            prompt, webui_prompt, negative_prompt, trigger_word,
                            lora_filename, base_model, created_at
                        )
                        VALUES(?, 'standard_validation_v1', 'basic_face', 111111, 0.8, 0,
                               1024, 1024, 'Euler a', 28, 7.0, ?, 1, 'v1',
                               'demo portrait', 'demo portrait', '', 'demo', ?, 'base.safetensors', ?)
                        """,
                        (run_id, f"hash-{run_id}", model_path.name, now),
                    ).lastrowid
                )
                if with_images:
                    conn.execute(
                        """
                        INSERT INTO validation_images(
                            job_id, selected_output_id, expected_condition_id,
                            validation_run_id, validation_preset_id, prompt_key, seed,
                            lora_weight, image_path, validation_type, prompt,
                            base_model, sampler, steps, cfg_scale, width, height,
                            condition_hash, created_at, updated_at
                        )
                        VALUES(?, ?, ?, ?, 'standard_validation_v1', 'basic_face', 111111,
                               0.8, ?, 'standard', 'demo portrait', 'base.safetensors',
                               'Euler a', 28, 7.0, 1024, 1024, ?, ?, ?)
                        """,
                        (job_id, output_id, condition_id, run_id, str(self.root / f"{run_id}.png"), f"hash-{run_id}", now, now),
                    )
                job_ids.append(job_id)
                output_ids.append(output_id)
                profile_ids.append(profile_id)
        return profile_ids, output_ids, job_ids

    def test_schema_has_lora_comparison_tables_and_validation_snapshots(self) -> None:
        tables = {row["name"] for row in fetch_all("SELECT name FROM sqlite_master WHERE type = 'table'")}
        self.assertIn("lora_comparison_sessions", tables)
        self.assertIn("lora_comparison_candidates", tables)
        cols = {row["name"] for row in fetch_all("PRAGMA table_info(validation_runs)")}
        self.assertIn("artifact_path_snapshot", cols)
        self.assertIn("artifact_sha256_snapshot", cols)

    def test_artifact_resolver_falls_back_to_external_copy_and_rejects_mismatch(self) -> None:
        profile_ids, output_ids, _ = self.create_fixture_pair()
        external, sha_value = self.write_model("exported.safetensors", b"exported")
        with connect() as conn:
            conn.execute(
                """
                UPDATE training_outputs
                SET file_path = ?, external_copy_path = ?, export_verified_at = ?, sha256 = ?
                WHERE id = ?
                """,
                (str(self.root / "missing.safetensors"), str(external), utc_now(), sha_value, output_ids[0]),
            )
        resolved = resolve_lora_artifact(profile_id=profile_ids[0])
        self.assertEqual(resolved.source_kind, "training_outputs.external_copy_path")
        with connect() as conn:
            conn.execute("UPDATE training_outputs SET sha256 = 'bad' WHERE id = ?", (output_ids[0],))
        with self.assertRaises(ValueError):
            resolve_lora_artifact(profile_id=profile_ids[0])

    def test_create_session_reuses_validation_runs_and_uses_real_candidate_names(self) -> None:
        profile_ids, _, _ = self.create_fixture_pair()
        session_id = create_lora_comparison_session(
            profile_ids=profile_ids,
            name="Standard vs C3Lier",
            comparison_mode="controlled",
            comparison_axis="network_type",
            validation_preset_id="standard_validation_v1",
            allow_warnings=True,
        )
        session, candidates = load_lora_comparison_session(session_id)
        self.assertEqual(session["parity_status"], "warning")
        self.assertEqual({candidate["validation_run_source"] for candidate in candidates}, {"reused"})
        summary = refresh_lora_comparison_session(session_id)
        self.assertEqual(summary["logical_image_count"], 2)
        self.assertEqual(summary["registered_image_count"], 2)
        html = build_lora_comparison_matrix_html(session_id)
        self.assertIn("Standard LoRA", html)
        self.assertIn("LoRA-C3Lier（セリア）", html)
        self.assertNotIn("Candidate A", html)

    def test_create_session_creates_missing_validation_runs(self) -> None:
        profile_ids, _, _ = self.create_fixture_pair(with_images=False)
        with connect() as conn:
            conn.execute("DELETE FROM validation_images")
            conn.execute("DELETE FROM validation_expected_conditions")
            conn.execute("DELETE FROM validation_runs")
        session_id = create_lora_comparison_session(
            profile_ids=profile_ids,
            name="Create runs",
            comparison_mode="controlled",
            comparison_axis="network_type",
            validation_preset_id="standard_validation_v1",
            allow_warnings=True,
        )
        _, candidates = load_lora_comparison_session(session_id)
        self.assertEqual({candidate["validation_run_source"] for candidate in candidates}, {"created"})
        self.assertEqual(len(fetch_all("SELECT * FROM validation_runs WHERE validation_run_kind = 'lora_comparison'")), 2)

    def test_decision_rejects_preferred_candidate_when_images_missing(self) -> None:
        profile_ids, _, _ = self.create_fixture_pair(with_images=False)
        session_id = create_lora_comparison_session(
            profile_ids=profile_ids,
            name="Missing images",
            comparison_mode="controlled",
            comparison_axis="network_type",
            validation_preset_id="standard_validation_v1",
            allow_warnings=True,
        )
        _, candidates = load_lora_comparison_session(session_id)
        with self.assertRaises(ValueError):
            save_lora_comparison_decision(
                session_id,
                decision_status="candidate_preferred",
                preferred_candidate_id=int(candidates[0]["id"]),
                decision_reason="not allowed",
            )

    def test_cleanup_blocks_comparison_output_without_verified_fallback(self) -> None:
        profile_ids, output_ids, job_ids = self.create_fixture_pair()
        create_lora_comparison_session(
            profile_ids=profile_ids,
            name="Cleanup guard",
            comparison_mode="controlled",
            comparison_axis="network_type",
            validation_preset_id="standard_validation_v1",
            allow_warnings=True,
        )
        preview = exported_selected_preview(job_ids[0])
        self.assertFalse(preview["can_execute"])
        self.assertIn("LoRA比較Session", preview["blocked_reason"])

    def test_parity_hard_fails_different_projects(self) -> None:
        profile_ids, _, _ = self.create_fixture_pair()
        first = {
            "profile": {"project_id": 1},
            "artifact": type("A", (), {"verified": True})(),
            "snapshot": {"job": {"model_family": "SDXL", "base_model_path": "base", "network_type_id": "standard_lora"}},
        }
        second = {
            "profile": {"project_id": 2},
            "artifact": type("A", (), {"verified": True})(),
            "snapshot": {"job": {"model_family": "SDXL", "base_model_path": "base", "network_type_id": "lora_c3lier"}},
        }
        report = run_parity_gate([first, second], "controlled", "network_type")
        self.assertEqual(report["status"], "fail")
