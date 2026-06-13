from __future__ import annotations

import sys
import tempfile
import types
import unittest
import hashlib
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from PIL import Image

from app import settings
from app.db import connect, fetch_all, fetch_one, init_db, utc_now
from app.main import build_loss_chart, build_metric_table


class IsolatedDbTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp.name)
        settings.DATA_DIR = self.root / "data"
        settings.DB_PATH = settings.DATA_DIR / "app.db"
        settings.EXPORTS_DIR = self.root / "exports"
        settings.RUNS_DIR = self.root / "runs"
        settings.LOGS_DIR = self.root / "logs"
        for directory in (settings.DATA_DIR, settings.EXPORTS_DIR, settings.RUNS_DIR, settings.LOGS_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        init_db()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_png(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(20, 120, 80)).save(path)
        return path


class MetricDisplayTests(unittest.TestCase):
    def make_metrics(self, count: int) -> list[dict[str, object]]:
        return [
            {
                "step": index + 1,
                "loss": 0.1 + (index % 17) * 0.001,
                "learning_rate": None,
                "source": "tensorboard",
                "raw_tag": "loss/current",
            }
            for index in range(count)
        ]

    def test_metric_table_limits_large_runs(self) -> None:
        table = build_metric_table(self.make_metrics(18900))
        self.assertTrue(table["limited"])
        self.assertEqual(table["total"], 18900)
        self.assertEqual(table["shown"], 205)
        self.assertEqual(table["omitted"], 18695)
        self.assertEqual(table["rows"][0]["step"], 1)
        self.assertEqual(table["rows"][4]["step"], 5)
        self.assertEqual(table["rows"][5]["step"], 18701)
        self.assertEqual(table["rows"][-1]["step"], 18900)

    def test_loss_chart_downsamples_large_runs(self) -> None:
        chart = build_loss_chart(self.make_metrics(18900))
        self.assertIsNotNone(chart)
        assert chart is not None
        self.assertEqual(chart["source_count"], 18900)
        self.assertLessEqual(chart["point_count"], 1200)
        self.assertEqual(chart["min_step"], 1)
        self.assertEqual(chart["max_step"], 18900)


class Phase107StabilizationTests(IsolatedDbTest):
    def create_project_fixture(self, base_model: str = "D:/models/base.safetensors") -> tuple[int, int, int]:
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO datasets(
                    name, path, model_family, trigger_word, class_token, image_count,
                    caption_count, missing_caption_count, resolution_summary_json,
                    tag_summary_json, scan_status, memo, created_at, updated_at
                )
                VALUES ('dataset', 'D:/datasets/test', 'SDXL', 'testchar', 'person', 50, 50, 0, '{}', '{}', 'ok', '', ?, ?)
                """,
                (now, now),
            )
            dataset_id = int(cur.lastrowid)
            cur = conn.execute(
                """
                INSERT INTO dataset_versions(dataset_id, version_no, trigger_word, image_count, caption_count, created_at, memo)
                VALUES (?, 1, 'testchar', 50, 50, ?, 'test')
                """,
                (dataset_id, now),
            )
            version_id = int(cur.lastrowid)
            cur = conn.execute(
                """
                INSERT INTO lora_projects(
                    name, dataset_id, current_dataset_version_id, trigger_word,
                    base_model_path, status, created_at, updated_at, memo
                )
                VALUES ('project', ?, ?, 'testchar', ?, 'active', ?, ?, '')
                """,
                (dataset_id, version_id, base_model, now, now),
            )
            project_id = int(cur.lastrowid)
        return project_id, dataset_id, version_id

    def add_completed_job(
        self,
        dataset_id: int,
        version_id: int,
        preset_id: str,
        base_model: str = "D:/models/base.safetensors",
        project_id: int | None = None,
    ) -> int:
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_jobs(
                    project_id, name, dataset_id, preset_id, status, model_family, training_script,
                    base_model_path, output_name, output_dir, run_dir, params_json,
                    return_code, output_model_count, dataset_version_id, created_at, updated_at
                )
                VALUES (?, 'completed job', ?, ?, 'completed', 'SDXL', 'sdxl_train_network.py',
                    ?, 'out', 'outdir', 'rundir', '{}', 0, 1, ?, ?, ?)
                """,
                (project_id, dataset_id, preset_id, base_model, version_id, now, now),
            )
            return int(cur.lastrowid)

    def add_job_with_status(self, status: str, project_id: int | None = None, name: str = "cleanup test") -> int:
        _, dataset_id, version_id = self.create_project_fixture() if project_id is None else (project_id, 1, 1)
        now = utc_now()
        with connect() as conn:
            if project_id is not None:
                dataset_row = conn.execute("SELECT dataset_id, current_dataset_version_id FROM lora_projects WHERE id = ?", (project_id,)).fetchone()
                dataset_id = dataset_row["dataset_id"]
                version_id = dataset_row["current_dataset_version_id"]
            cur = conn.execute(
                """
                INSERT INTO training_jobs(
                    project_id, name, dataset_id, preset_id, status, model_family, training_script,
                    base_model_path, output_name, output_dir, run_dir, params_json,
                    dataset_version_id, created_at, updated_at
                )
                VALUES (?, ?, ?, 'sdxl_2d_face_adamw8bit_standard', ?, 'SDXL', 'sdxl_train_network.py',
                    'D:/models/base.safetensors', 'out', 'outdir', ?, '{}', ?, ?, ?)
                """,
                (project_id, name, dataset_id, status, str(self.root / "runs" / name), version_id, now, now),
            )
            return int(cur.lastrowid)

    def test_init_db_and_validation_presets(self) -> None:
        count = fetch_one("SELECT COUNT(*) AS count FROM validation_presets")["count"]
        standard = fetch_one("SELECT expected_image_count FROM validation_presets WHERE id = 'standard_validation_v1'")
        self.assertGreaterEqual(count, 3)
        self.assertEqual(standard["expected_image_count"], 45)

    def test_job_archive_restore_and_delete_draft(self) -> None:
        from app.main import job_archive, job_delete, job_restore

        job_id = self.add_job_with_status("draft", name="draft cleanup")
        job_archive(job_id, "整理")
        archived = fetch_one("SELECT archived_at, archived_reason FROM training_jobs WHERE id = ?", (job_id,))
        self.assertTrue(archived["archived_at"])
        self.assertEqual(archived["archived_reason"], "整理")

        job_restore(job_id)
        restored = fetch_one("SELECT archived_at FROM training_jobs WHERE id = ?", (job_id,))
        self.assertIsNone(restored["archived_at"])

        job_delete(job_id, "未実行のため削除", "db_only")
        deleted = fetch_one("SELECT deleted_at, delete_reason FROM training_jobs WHERE id = ?", (job_id,))
        self.assertTrue(deleted["deleted_at"])
        self.assertEqual(deleted["delete_reason"], "未実行のため削除")

    def test_selected_completed_job_delete_is_rejected(self) -> None:
        project_id, dataset_id, version_id = self.create_project_fixture()
        job_id = self.add_completed_job(dataset_id, version_id, "sdxl_2d_face_standard_6epoch", project_id=project_id)
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO training_outputs(job_id, file_path, file_type, selected, created_at)
                VALUES (?, 'D:/runs/job/model.safetensors', 'model', 1, ?)
                """,
                (job_id, now),
            )
        from app.main import job_delete, job_delete_preview

        preview = job_delete_preview(job_id)
        self.assertTrue(preview["selected_links"])
        self.assertFalse(preview["can_delete_db"])
        with self.assertRaises(HTTPException):
            job_delete(job_id, "danger", "db_only")

    def test_project_archive_restore_with_jobs(self) -> None:
        project_id, _, _ = self.create_project_fixture()
        job_id = self.add_job_with_status("draft", project_id=project_id, name="project child")
        from app.main import project_archive, project_restore

        project_archive(project_id, "Project整理", "1")
        project = fetch_one("SELECT archived_at, archive_reason FROM lora_projects WHERE id = ?", (project_id,))
        job = fetch_one("SELECT archived_at FROM training_jobs WHERE id = ?", (job_id,))
        self.assertTrue(project["archived_at"])
        self.assertEqual(project["archive_reason"], "Project整理")
        self.assertTrue(job["archived_at"])

        project_restore(project_id)
        restored = fetch_one("SELECT archived_at FROM lora_projects WHERE id = ?", (project_id,))
        self.assertIsNone(restored["archived_at"])

    def test_cleanup_candidates_include_unexecuted_jobs(self) -> None:
        job_id = self.add_job_with_status("prepared", name="old prepared")
        from app.main import cleanup_job_candidates

        ids = {row["id"] for row in cleanup_job_candidates(limit=20)}
        self.assertIn(job_id, ids)

    def test_stale_running_job_is_marked_stopped(self) -> None:
        project_id, _, _ = self.create_project_fixture()
        job_id = self.add_job_with_status("running", project_id=project_id, name="stale running")
        with connect() as conn:
            conn.execute("UPDATE training_jobs SET process_id = 999999, start_time = ? WHERE id = ?", (utc_now(), job_id))
        from app.services import training_runner

        with mock.patch.object(training_runner, "process_exists", return_value=False), \
            mock.patch.object(training_runner, "job_marker_process_exists", return_value=False), \
            mock.patch.object(training_runner, "collect_job_results", return_value={"models": 0, "samples": 0}):
            fixed = training_runner.reconcile_stale_running_jobs()

        self.assertIn(job_id, fixed)
        job = fetch_one("SELECT status, process_id, return_code FROM training_jobs WHERE id = ?", (job_id,))
        self.assertEqual(job["status"], "stopped")
        self.assertIsNone(job["process_id"])
        self.assertEqual(job["return_code"], 4294967295)

    def test_pilot_recommendation_required_without_real_completed_job(self) -> None:
        project_id, _, _ = self.create_project_fixture()
        from app.main import pilot_recommendation

        project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (project_id,))
        guidance = pilot_recommendation(project)
        self.assertEqual(guidance["label"], "REQUIRED")

    def test_pilot_recommendation_skippable_with_same_standard_completed(self) -> None:
        project_id, dataset_id, version_id = self.create_project_fixture()
        self.add_completed_job(dataset_id, version_id, "sdxl_2d_face_standard_6epoch", project_id=project_id)
        from app.main import pilot_recommendation

        project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (project_id,))
        guidance = pilot_recommendation(project)
        self.assertEqual(guidance["label"], "SKIPPABLE")

    def test_project_standard_creation_saves_pilot_skip_reason(self) -> None:
        project_id, dataset_id, version_id = self.create_project_fixture()
        self.add_completed_job(dataset_id, version_id, "sdxl_2d_face_pilot_3epoch", project_id=project_id)
        from app.main import create_project_preset_job

        reason = "Preflight OK、同一base modelで完走実績ありのためPilotスキップ"
        job_id = create_project_preset_job(project_id, "sdxl_2d_face_standard_6epoch", reason)
        job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
        project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (project_id,))
        self.assertEqual(job["preset_id"], "sdxl_2d_face_standard_6epoch")
        self.assertEqual(job["dataset_version_id"], version_id)
        self.assertIn(reason, project["memo"])

    def test_smoke_step_limit_resets_to_selected_preset_params(self) -> None:
        from app.main import params_for_selected_preset

        preset = fetch_one("SELECT * FROM presets WHERE id = 'sdxl_2d_face_adamw8bit_standard'")
        smoke_params = {
            "network_dim": 4,
            "network_alpha": 2,
            "train_batch_size": 1,
            "repeats": 1,
            "max_train_steps": 2,
            "save_every_n_steps": 1,
            "sample_every_n_steps": 1,
            "resolution": [512, 512],
        }
        params, reset = params_for_selected_preset("sdxl_2d_face_adamw8bit_standard", preset, smoke_params)

        self.assertTrue(reset)
        self.assertNotIn("max_train_steps", params)
        self.assertNotIn("save_every_n_steps", params)
        self.assertNotIn("sample_every_n_steps", params)
        self.assertEqual(params["network_dim"], 32)
        self.assertEqual(params["network_alpha"], 16)
        self.assertEqual(params["train_batch_size"], 2)
        self.assertEqual(params["max_train_epochs"], 10)

    def test_create_job_resets_smoke_step_limit_for_normal_preset(self) -> None:
        import json

        from app.db import create_job

        _, dataset_id, _ = self.create_project_fixture()
        job_id = create_job(
            {
                "name": "from smoke clone",
                "dataset_id": dataset_id,
                "preset_id": "sdxl_2d_face_adamw8bit_generalize",
                "base_model_path": "D:/models/base.safetensors",
                "output_name": "from_smoke_clone",
                "params": {
                    "network_dim": 4,
                    "network_alpha": 2,
                    "train_batch_size": 1,
                    "repeats": 1,
                    "max_train_steps": 2,
                    "save_every_n_steps": 1,
                    "sample_every_n_steps": 1,
                    "resolution": [512, 512],
                },
            }
        )
        job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
        params = json.loads(job["params_json"])

        self.assertNotIn("max_train_steps", params)
        self.assertNotIn("save_every_n_steps", params)
        self.assertNotIn("sample_every_n_steps", params)
        self.assertEqual(params["network_dim"], 16)
        self.assertEqual(params["network_alpha"], 8)
        self.assertEqual(params["train_batch_size"], 2)
        self.assertEqual(params["max_train_epochs"], 8)

    def test_train_log_tail_decodes_cp932_logs(self) -> None:
        from app.services.training_runner import read_log_tail

        run_dir = self.root / "runs" / "job_000001"
        log_path = run_dir / "logs" / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes("エラー: メモリ不足\nRuntimeError: bad allocation\n".encode("cp932"))

        tail = read_log_tail({"run_dir": str(run_dir)})

        self.assertIn("エラー: メモリ不足", tail)
        self.assertIn("RuntimeError: bad allocation", tail)

    def test_validation_image_outside_allowed_root_is_403(self) -> None:
        outside = self.make_png(self.root / "outside.png")
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO validation_images(job_id, image_path, validation_type, created_at, updated_at)
                VALUES (1, ?, 'test', ?, ?)
                """,
                (str(outside), now, now),
            )
            image_id = int(cur.lastrowid)
        from app.main import validation_image_file

        with self.assertRaises(HTTPException) as raised:
            validation_image_file(image_id)
        self.assertEqual(raised.exception.status_code, 403)

    def test_standard_validation_run_creates_fixed_expected_conditions(self) -> None:
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_jobs(
                    name, status, model_family, training_script, base_model_path,
                    output_name, output_dir, run_dir, params_json,
                    trigger_word_at_creation, created_at, updated_at
                )
                VALUES ('test job', 'completed', 'SDXL', 'sdxl_train_network.py',
                    'D:/models/base.safetensors', 'test', 'out', 'run', '{}',
                    'testchar', ?, ?)
                """,
                (now, now),
            )
            job_id = int(cur.lastrowid)
        from app.services.validation_runs import create_validation_run

        run_id = create_validation_run(job_id, "standard_validation_v1", "base", "testchar", "")
        run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
        count = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
        first = fetch_one("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order LIMIT 1", (run_id,))
        self.assertTrue(run["preset_snapshot_json"])
        self.assertEqual(count["count"], 45)
        self.assertEqual(first["trigger_word"], "testchar")
        self.assertTrue(first["prompt"])
        self.assertTrue(first["webui_prompt"])
        self.assertEqual(first["preset_version"], "1.0")

    def test_reference_image_is_copied_to_managed_directory(self) -> None:
        source = self.make_png(self.root / "source" / "ref.png")
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO reference_sets(name, created_at, updated_at)
                VALUES ('test refs', ?, ?)
                """,
                (now, now),
            )
            set_id = int(cur.lastrowid)
        from app.main import reference_image_add

        reference_image_add(set_id, str(source), "face", "", 0)
        image = fetch_one("SELECT * FROM reference_images WHERE reference_set_id = ?", (set_id,))
        managed = Path(image["image_path"]).resolve()
        self.assertTrue(managed.exists())
        self.assertTrue(managed.is_relative_to((settings.EXPORTS_DIR / "reference_sets").resolve()))

    def create_basic_job(self) -> int:
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_jobs(
                    name, status, model_family, training_script, base_model_path,
                    output_name, output_dir, run_dir, params_json,
                    trigger_word_at_creation, created_at, updated_at
                )
                VALUES ('test job', 'completed', 'SDXL', 'sdxl_train_network.py',
                    'D:/models/base.safetensors', 'test', 'out', 'run', '{}',
                    'testchar', ?, ?)
                """,
                (now, now),
            )
            return int(cur.lastrowid)

    def test_matching_validation_run_image_links_expected_condition_and_serves(self) -> None:
        source = self.make_png(self.root / "source" / "validation.png")
        job_id = self.create_basic_job()
        from app.main import register_validation_run_image, validation_image_file
        from app.services.validation_runs import create_validation_run

        run_id = create_validation_run(job_id, "standard_validation_v1", "base", "testchar", "")
        first = fetch_one("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order LIMIT 1", (run_id,))
        register_validation_run_image(
            run_id,
            str(source),
            "individual",
            first["prompt_key"],
            int(first["seed"]),
            float(first["lora_weight"]),
            bool(first["hires_enabled"]),
            "",
        )
        image = fetch_one("SELECT * FROM validation_images WHERE validation_run_id = ?", (run_id,))
        self.assertEqual(image["expected_condition_id"], first["id"])
        self.assertTrue(Path(image["image_path"]).is_relative_to((settings.EXPORTS_DIR / "validation_runs").resolve()))
        response = validation_image_file(int(image["id"]))
        self.assertEqual(response.status_code, 200)

    def test_legacy_external_validation_image_is_copied_and_served(self) -> None:
        source = self.make_png(self.root / "source" / "legacy.png")
        job_id = self.create_basic_job()
        from app.main import job_add_validation_image, validation_image_file

        job_add_validation_image(
            job_id,
            str(source),
            validation_type="external",
            prompt="",
            negative_prompt="",
            base_model="",
            sampler="",
            steps="",
            cfg_scale="",
            width="",
            height="",
            hires_enabled="",
            hires_scale="",
            lora_weights="0.6",
            seeds="123",
            rating_face=0,
            rating_costume=0,
            rating_style=0,
            rating_stability=0,
            rating_overall=0,
            strength_label="",
            overfit_level="",
            adoption_label="",
            failure_tags=[],
            recommended_weight_min="",
            recommended_weight_max="",
            memo="",
        )
        image = fetch_one("SELECT * FROM validation_images WHERE job_id = ?", (job_id,))
        self.assertTrue(Path(image["image_path"]).is_relative_to((settings.EXPORTS_DIR / "validation_runs").resolve()))
        response = validation_image_file(int(image["id"]))
        self.assertEqual(response.status_code, 200)

    def test_quoted_legacy_external_validation_path_is_normalized(self) -> None:
        source = self.make_png(self.root / "source" / "quoted legacy.png")
        job_id = self.create_basic_job()
        from app.main import job_add_validation_image

        job_add_validation_image(
            job_id,
            f'  "{source}"  ',
            validation_type="external",
            prompt="",
            negative_prompt="",
            base_model="",
            sampler="",
            steps="",
            cfg_scale="",
            width="",
            height="",
            hires_enabled="",
            hires_scale="",
            lora_weights="0.6",
            seeds="123",
            rating_face=0,
            rating_costume=0,
            rating_style=0,
            rating_stability=0,
            rating_overall=0,
            strength_label="",
            overfit_level="",
            adoption_label="",
            failure_tags=[],
            recommended_weight_min="",
            recommended_weight_max="",
            memo="",
        )
        image = fetch_one("SELECT * FROM validation_images WHERE job_id = ?", (job_id,))
        self.assertTrue(Path(image["image_path"]).exists())

    def test_validation_image_delivery_rejects_broken_managed_image(self) -> None:
        managed_dir = settings.EXPORTS_DIR / "validation_runs" / "broken" / "images"
        managed_dir.mkdir(parents=True, exist_ok=True)
        broken = managed_dir / "broken.png"
        broken.write_text("not an image", encoding="utf-8")
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO validation_images(job_id, image_path, validation_type, created_at, updated_at)
                VALUES (1, ?, 'test', ?, ?)
                """,
                (str(broken), now, now),
            )
            image_id = int(cur.lastrowid)
        from app.main import validation_image_file

        with self.assertRaises(HTTPException) as raised:
            validation_image_file(image_id)
        self.assertEqual(raised.exception.status_code, 404)

    def test_validation_result_can_be_saved_without_image_path(self) -> None:
        job_id = self.create_basic_job()
        from app.main import job_add_validation_result

        job_add_validation_result(
            job_id,
            "basic_face",
            0.6,
            face_score=0,
            costume_score=0,
            stability_score=0,
            flexibility_score=0,
            overall_score=3,
            memo="no image",
            image_path="",
        )
        row = fetch_one("SELECT * FROM validation_results WHERE job_id = ?", (job_id,))
        self.assertEqual(row["image_path"], "")

    def test_quoted_validation_result_image_path_is_normalized(self) -> None:
        source = self.make_png(self.root / "source" / "quoted result.png")
        job_id = self.create_basic_job()
        from app.main import job_add_validation_result

        job_add_validation_result(
            job_id,
            "basic_face",
            0.6,
            face_score=0,
            costume_score=0,
            stability_score=0,
            flexibility_score=0,
            overall_score=3,
            memo="quoted image",
            image_path=f" '{source}' ",
        )
        row = fetch_one("SELECT * FROM validation_results WHERE job_id = ?", (job_id,))
        self.assertTrue(Path(row["image_path"]).exists())

    def test_backfills_legacy_expected_condition_columns_without_hash_change(self) -> None:
        job_id = self.create_basic_job()
        from app.services.validation_runs import backfill_validation_runs, create_validation_run, write_validation_prompt_pack

        run_id = create_validation_run(job_id, "standard_validation_v1", "base", "testchar", "")
        first = fetch_one("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order LIMIT 1", (run_id,))
        original_hash = first["condition_hash"]
        with connect() as conn:
            conn.execute(
                """
                UPDATE validation_runs SET preset_snapshot_json = NULL WHERE id = ?
                """,
                (run_id,),
            )
            conn.execute(
                """
                UPDATE validation_expected_conditions
                SET prompt = NULL, webui_prompt = NULL, preset_version = NULL,
                    negative_prompt = NULL, trigger_word = NULL, lora_filename = NULL, base_model = NULL
                WHERE id = ?
                """,
                (first["id"],),
            )
        backfill_validation_runs()
        after = fetch_one("SELECT * FROM validation_expected_conditions WHERE id = ?", (first["id"],))
        run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
        self.assertEqual(after["condition_hash"], original_hash)
        self.assertTrue(after["prompt"])
        self.assertTrue(after["webui_prompt"])
        self.assertTrue(after["preset_version"])
        self.assertTrue(run["preset_snapshot_json"])
        paths = write_validation_prompt_pack(run_id)
        prompt_pack = Path(paths["prompts_md"]).read_text(encoding="utf-8")
        self.assertIn("testchar", prompt_pack)
        self.assertIn("Prompt:", prompt_pack)

    def test_existing_validation_run_with_images_preserves_condition_mismatch(self) -> None:
        source = self.make_png(self.root / "source" / "validation mismatch.png")
        job_id = self.create_basic_job()
        from app.main import register_validation_run_image
        from app.services.validation_runs import create_validation_run, ensure_expected_conditions, load_validation_run_bundle

        run_id = create_validation_run(job_id, "standard_validation_v1", "base", "testchar", "")
        first = fetch_one("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order LIMIT 1", (run_id,))
        register_validation_run_image(
            run_id,
            str(source),
            "individual",
            first["prompt_key"],
            int(first["seed"]),
            float(first["lora_weight"]),
            bool(first["hires_enabled"]),
            "",
        )
        with connect() as conn:
            conn.execute(
                """
                DELETE FROM validation_expected_conditions
                WHERE validation_run_id = ?
                  AND id != ?
                  AND expected_order = 45
                """,
                (run_id, first["id"]),
            )

        rows = ensure_expected_conditions(run_id)
        self.assertEqual(len(rows), 44)
        hashes = [row["condition_hash"] for row in fetch_all("SELECT condition_hash FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))]
        self.assertIn(first["condition_hash"], hashes)
        image = fetch_one("SELECT * FROM validation_images WHERE validation_run_id = ?", (run_id,))
        self.assertEqual(image["expected_condition_id"], first["id"])
        bundle = load_validation_run_bundle(run_id)
        self.assertIn("Expected Condition count mismatch", bundle["condition_warning"])

    def test_reference_image_delivery_rejects_broken_managed_image(self) -> None:
        managed_dir = settings.EXPORTS_DIR / "reference_sets" / "reference_set_000001" / "images"
        managed_dir.mkdir(parents=True, exist_ok=True)
        broken = managed_dir / "broken.png"
        broken.write_text("not an image", encoding="utf-8")
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO reference_sets(name, created_at, updated_at)
                VALUES ('test refs', ?, ?)
                """,
                (now, now),
            )
            set_id = int(cur.lastrowid)
            cur = conn.execute(
                """
                INSERT INTO reference_images(reference_set_id, image_path, image_role, created_at)
                VALUES (?, ?, 'face', ?)
                """,
                (set_id, str(broken), now),
            )
            image_id = int(cur.lastrowid)
        from app.main import reference_image_file

        with self.assertRaises(HTTPException) as raised:
            reference_image_file(image_id)
        self.assertEqual(raised.exception.status_code, 404)


class StorageCleanupTests(IsolatedDbTest):
    def make_job_with_files(self, status: str = "completed", selected_exported: bool = False) -> int:
        now = utc_now()
        dataset = self.root / "datasets" / "set"
        dataset.mkdir(parents=True, exist_ok=True)
        run_dir = settings.RUNS_DIR / "job_000001"
        model_dir = run_dir / "models"
        sample_dir = run_dir / "samples"
        log_dir = run_dir / "logs"
        for directory in (model_dir, sample_dir, log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        selected = model_dir / "selected.safetensors"
        unselected = model_dir / "unselected.safetensors"
        sample = sample_dir / "sample.png"
        selected.write_bytes(b"selected model")
        unselected.write_bytes(b"unselected model")
        self.make_png(sample)
        (log_dir / "train.log").write_text("log", encoding="utf-8")
        selected_sha = hashlib.sha256(selected.read_bytes()).hexdigest()
        export_path = ""
        export_verified_at = None
        if selected_exported:
            export_path = str(settings.EXPORTS_DIR / "selected_loras" / "job_000001" / selected.name)
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            Path(export_path).write_bytes(selected.read_bytes())
            export_verified_at = now
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO datasets(name, path, model_family, trigger_word, image_count, caption_count, created_at, updated_at)
                VALUES ('dataset', ?, 'SDXL', 'testchar', 1, 1, ?, ?)
                """,
                (str(dataset), now, now),
            )
            dataset_id = int(cur.lastrowid)
            cur = conn.execute(
                """
                INSERT INTO training_jobs(
                    name, dataset_id, preset_id, status, model_family, training_script,
                    base_model_path, output_name, output_dir, run_dir, params_json,
                    output_model_count, sample_image_count, adopted_model_path, created_at, updated_at
                )
                VALUES ('storage job', ?, 'sdxl_2d_face_standard_6epoch', ?, 'SDXL', 'sdxl_train_network.py',
                    'D:/models/base.safetensors', 'out', ?, ?, '{}', 2, 1, ?, ?, ?)
                """,
                (dataset_id, status, str(model_dir), str(run_dir), str(selected), now, now),
            )
            job_id = int(cur.lastrowid)
            conn.execute(
                """
                INSERT INTO training_outputs(job_id, file_path, file_type, selected, file_size, sha256, external_copy_path, export_verified_at, created_at)
                VALUES (?, ?, 'model', 1, ?, ?, ?, ?, ?)
                """,
                (job_id, str(selected), selected.stat().st_size, selected_sha, export_path or None, export_verified_at, now),
            )
            conn.execute(
                """
                INSERT INTO training_outputs(job_id, file_path, file_type, selected, file_size, created_at)
                VALUES (?, ?, 'model', 0, ?, ?)
                """,
                (job_id, str(unselected), unselected.stat().st_size, now),
            )
            conn.execute(
                """
                INSERT INTO sample_images(job_id, image_path, created_at)
                VALUES (?, ?, ?)
                """,
                (job_id, str(sample), now),
            )
        return job_id

    def test_storage_usage_reports_job_sizes_and_onedrive_warning(self) -> None:
        from app.services.storage_cleanup import storage_root_warning, storage_usage

        self.make_job_with_files()
        usage = storage_usage()
        self.assertGreater(usage["totals"]["runs"]["bytes"], 0)
        self.assertGreater(usage["totals"]["model_outputs"]["bytes"], 0)
        self.assertGreater(usage["totals"]["sample_images"]["bytes"], 0)
        with mock.patch.object(settings, "ROOT_DIR", Path("C:/Users/test/OneDrive/LoRA-Studio")):
            self.assertIn("OneDrive", storage_root_warning())

    def test_unselected_cleanup_preview_excludes_selected_and_moves_to_trash(self) -> None:
        from app.services.storage_cleanup import cleanup_outputs, unselected_model_preview

        job_id = self.make_job_with_files()
        preview = unselected_model_preview(job_id)
        self.assertEqual(len(preview["files"]), 1)
        self.assertFalse(preview["files"][0].selected)
        result = cleanup_outputs(job_id, "unselected_models")
        self.assertEqual(result["moved"], 1)
        output = fetch_one("SELECT cleanup_status, deleted_at FROM training_outputs WHERE job_id = ? AND selected = 0", (job_id,))
        self.assertEqual(output["cleanup_status"], "deleted")
        self.assertTrue(output["deleted_at"])
        history = fetch_one("SELECT COUNT(*) AS count FROM file_cleanup_history WHERE job_id = ?", (job_id,))
        self.assertEqual(history["count"], 1)

    def test_exported_selected_cleanup_requires_verified_copy(self) -> None:
        from app.services.storage_cleanup import cleanup_outputs, exported_selected_preview

        job_id = self.make_job_with_files(selected_exported=True)
        preview = exported_selected_preview(job_id)
        self.assertTrue(preview["can_execute"])
        cleanup_outputs(job_id, "exported_selected")
        job = fetch_one("SELECT adopted_model_path FROM training_jobs WHERE id = ?", (job_id,))
        selected = fetch_one("SELECT cleanup_status, external_copy_path FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
        self.assertEqual(selected["cleanup_status"], "deleted")
        self.assertEqual(job["adopted_model_path"], selected["external_copy_path"])

    def test_sample_cleanup_preview_and_move_to_trash(self) -> None:
        from app.services.storage_cleanup import cleanup_samples, sample_cleanup_preview

        job_id = self.make_job_with_files()
        preview = sample_cleanup_preview(job_id)
        self.assertEqual(len(preview["files"]), 1)
        cleanup_samples(job_id, "delete_individual")
        sample = fetch_one("SELECT cleanup_status, deleted_at FROM sample_images WHERE job_id = ?", (job_id,))
        self.assertEqual(sample["cleanup_status"], "deleted")
        self.assertTrue(sample["deleted_at"])


class ReviewQueueTests(IsolatedDbTest):
    def make_review_job(self) -> int:
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO datasets(name, path, model_family, trigger_word, image_count, caption_count, created_at, updated_at)
                VALUES ('dataset', 'D:/datasets/test', 'SDXL', 'testchar', 3, 3, ?, ?)
                """,
                (now, now),
            )
            dataset_id = int(cur.lastrowid)
            cur = conn.execute(
                """
                INSERT INTO training_jobs(
                    name, dataset_id, preset_id, status, model_family, training_script,
                    base_model_path, output_name, output_dir, run_dir, params_json,
                    adopted_epoch, created_at, updated_at
                )
                VALUES ('review job', ?, 'sdxl_2d_face_standard_6epoch', 'completed', 'SDXL', 'sdxl_train_network.py',
                    'D:/models/base.safetensors', 'out', 'outdir', ?, '{}', 4, ?, ?)
                """,
                (dataset_id, str(self.root / "runs" / "job_000001"), now, now),
            )
            job_id = int(cur.lastrowid)
            for epoch, avg, moving in [
                (1, 0.18, 0.16),
                (2, 0.14, 0.13),
                (3, 0.08, 0.08),
                (4, 0.09, 0.07),
                (5, 0.12, 0.10),
                (6, 0.17, 0.15),
            ]:
                conn.execute(
                    """
                    INSERT INTO training_epoch_summaries(job_id, epoch, avg_loss, moving_avg_final_loss, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, epoch, avg, moving, now, now),
                )
                conn.execute(
                    """
                    INSERT INTO training_outputs(job_id, epoch, file_path, file_type, selected, created_at)
                    VALUES (?, ?, ?, 'model', ?, ?)
                    """,
                    (job_id, epoch, f"D:/runs/job/model-{epoch}.safetensors", 1 if epoch == 4 else 0, now),
                )
            prompts = [
                ("basic_face", "face"),
                ("full_body", "full_body"),
                ("expression_pose", "expression_pose"),
            ]
            prompt_ids = []
            for order, (name, role) in enumerate(prompts, start=1):
                cur = conn.execute(
                    """
                    INSERT INTO sample_prompts(job_id, name, prompt, sort_order, prompt_role, created_at)
                    VALUES (?, ?, 'prompt', ?, ?, ?)
                    """,
                    (job_id, name, order, role, now),
                )
                prompt_ids.append(int(cur.lastrowid))
            for epoch in range(1, 7):
                for prompt_id in prompt_ids:
                    conn.execute(
                        """
                        INSERT INTO sample_images(job_id, prompt_id, epoch, image_path, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (job_id, prompt_id, epoch, f"D:/runs/job/e{epoch}_p{prompt_id}.png", now),
                    )
        return job_id

    def test_candidate_epoch_generation_prefers_selected_neighbors(self) -> None:
        from app.services.review_candidates import regenerate_epoch_candidates

        job_id = self.make_review_job()
        rows = regenerate_epoch_candidates(job_id)
        labels = {row["epoch"]: row["candidate_label"] for row in rows}
        self.assertEqual(labels[4], "primary")
        self.assertEqual(labels[3], "secondary")
        self.assertEqual(labels[5], "check")
        self.assertEqual(labels[1], "low_priority")
        sample = fetch_one("SELECT review_priority, auto_review_reason FROM sample_images WHERE job_id = ? AND epoch = 4 LIMIT 1", (job_id,))
        self.assertEqual(sample["review_priority"], "high")
        self.assertIn("候補epoch", sample["auto_review_reason"])

    def test_group_samples_filters_candidates_and_marks_full_body_role(self) -> None:
        from app.main import group_samples
        from app.services.review_candidates import regenerate_epoch_candidates

        job_id = self.make_review_job()
        candidates = {row["epoch"]: row for row in regenerate_epoch_candidates(job_id)}
        prompts = fetch_all("SELECT * FROM sample_prompts WHERE job_id = ? ORDER BY sort_order", (job_id,))
        samples = fetch_all("SELECT * FROM sample_images WHERE job_id = ? ORDER BY id", (job_id,))
        groups = group_samples(prompts, samples, candidates, "candidates")
        epochs = {sample["epoch"] for group in groups for sample in group["samples"]}
        self.assertEqual(epochs, {3, 4, 5})
        full_body = next(group for group in groups if group["prompt_role"] == "full_body")
        self.assertIn("N/A", full_body["rubric"]["face"])

    def test_nullable_ratings_and_bulk_save(self) -> None:
        from app.main import job_review_samples_bulk, nullable_rating

        self.assertIsNone(nullable_rating(""))
        job_id = self.make_review_job()
        sample = fetch_one("SELECT id FROM sample_images WHERE job_id = ? LIMIT 1", (job_id,))
        class DummyRequest:
            async def json(self):
                return {"items": [{"id": sample["id"], "rating_face": "", "rating_costume": "4", "rating_style": "3", "rating_stability": "5", "rating_flexibility": "", "rating_overall": "4", "failure_tags": [], "memo": "bulk"}]}

        import asyncio
        asyncio.run(job_review_samples_bulk(DummyRequest(), job_id))
        row = fetch_one("SELECT rating_face, rating_costume, rating_flexibility, rating_overall, memo FROM sample_images WHERE id = ?", (sample["id"],))
        self.assertIsNone(row["rating_face"])
        self.assertEqual(row["rating_costume"], 4)
        self.assertIsNone(row["rating_flexibility"])
        self.assertEqual(row["rating_overall"], 4)
        self.assertEqual(row["memo"], "bulk")


class StartHelperTests(unittest.TestCase):
    def test_start_helper_does_not_release_7865_by_default(self) -> None:
        import start_lora_helper

        fake_uvicorn = types.SimpleNamespace(run=mock.Mock())
        with mock.patch.object(sys, "argv", ["start_lora_helper.py", "--port", "8768", "--no-browser", "--skip-sd-scripts-setup"]), \
            mock.patch.dict(sys.modules, {"uvicorn": fake_uvicorn}), \
            mock.patch.object(start_lora_helper, "init_db"), \
            mock.patch.object(start_lora_helper, "release_port") as release_port:
            start_lora_helper.main()
        release_port.assert_not_called()

    def test_start_helper_force_releases_only_app_port(self) -> None:
        import start_lora_helper

        fake_uvicorn = types.SimpleNamespace(run=mock.Mock())
        with mock.patch.object(sys, "argv", ["start_lora_helper.py", "--port", "8768", "--no-browser", "--skip-sd-scripts-setup", "--force-release-port"]), \
            mock.patch.dict(sys.modules, {"uvicorn": fake_uvicorn}), \
            mock.patch.object(start_lora_helper, "init_db"), \
            mock.patch.object(start_lora_helper, "release_port") as release_port:
            start_lora_helper.main()
        release_port.assert_called_once_with(8768)

    def test_release_port_does_not_kill_process_tree(self) -> None:
        import start_lora_helper

        with mock.patch.object(start_lora_helper, "find_listening_pids", return_value={1234}), \
            mock.patch.object(start_lora_helper.os, "getpid", return_value=9999), \
            mock.patch.object(start_lora_helper.subprocess, "run") as run:
            start_lora_helper.release_port(8768)

        args = run.call_args.args[0]
        self.assertEqual(args, ["taskkill", "/PID", "1234", "/F"])
        self.assertNotIn("/T", args)


class OneoffScriptTests(unittest.TestCase):
    def test_oneoff_scripts_import_as_direct_scripts(self) -> None:
        import importlib.util

        for script in [
            "scripts/oneoff_job12_validation_defaults.py",
            "scripts/oneoff_seed_legacy_validation_runs.py",
            "scripts/oneoff_migrate_managed_images.py",
        ]:
            spec = importlib.util.spec_from_file_location(Path(script).stem, script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertTrue(hasattr(module, "main"))


if __name__ == "__main__":
    unittest.main()
