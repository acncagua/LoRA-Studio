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
from app.db import connect, fetch_one, init_db, utc_now


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


class Phase1071Tests(IsolatedDbTest):
    def test_init_db_and_validation_presets(self) -> None:
        count = fetch_one("SELECT COUNT(*) AS count FROM validation_presets")["count"]
        standard = fetch_one("SELECT expected_image_count FROM validation_presets WHERE id = 'standard_validation_v1'")
        self.assertGreaterEqual(count, 3)
        self.assertEqual(standard["expected_image_count"], 45)

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


if __name__ == "__main__":
    unittest.main()
