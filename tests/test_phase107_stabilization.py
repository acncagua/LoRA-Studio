from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from PIL import Image

from app import settings
from app.db import connect, fetch_all, fetch_one, init_db, utc_now


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

    def test_init_db_and_validation_presets(self) -> None:
        count = fetch_one("SELECT COUNT(*) AS count FROM validation_presets")["count"]
        standard = fetch_one("SELECT expected_image_count FROM validation_presets WHERE id = 'standard_validation_v1'")
        self.assertGreaterEqual(count, 3)
        self.assertEqual(standard["expected_image_count"], 45)

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
