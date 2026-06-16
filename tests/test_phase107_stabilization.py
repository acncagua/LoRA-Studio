from __future__ import annotations

import sys
import tempfile
import types
import unittest
import hashlib
import json
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from PIL import Image

from app import settings
from app.db import connect, fetch_all, fetch_one, init_db, utc_now
from app.main import build_loss_chart, build_metric_table, reference_set_detail, reference_sets
from app.services.reference_sets import (
    add_reference_image,
    create_reference_set,
    export_reference_artifacts,
    reference_detail,
    set_project_default,
)
from app.services.validation_generation import (
    common_gen_img_args,
    normalize_sd_scripts_sampler,
    reconcile_stale_validation_generations,
)
from app.services.validation_runs import image_is_reviewed


class IsolatedDbTest(unittest.TestCase):
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
        self.assertEqual(table["shown"], 43)
        self.assertEqual(table["omitted"], 18857)
        self.assertEqual(table["rows"][0]["step"], 1)
        self.assertEqual(table["rows"][2]["step"], 3)
        self.assertEqual(table["rows"][3]["step"], 18861)
        self.assertEqual(table["rows"][-1]["step"], 18900)

    def test_loss_chart_downsamples_large_runs(self) -> None:
        chart = build_loss_chart(self.make_metrics(18900))
        self.assertIsNotNone(chart)
        assert chart is not None
        self.assertEqual(chart["source_count"], 18900)
        self.assertLessEqual(chart["point_count"], 1200)
        self.assertEqual(chart["min_step"], 1)
        self.assertEqual(chart["max_step"], 18900)


class ReviewSemanticsTests(IsolatedDbTest):
    def test_empty_failure_tags_do_not_mark_validation_image_reviewed(self) -> None:
        image = {
            "rating_face": None,
            "rating_costume": None,
            "rating_style": None,
            "rating_stability": None,
            "rating_flexibility": None,
            "rating_overall": None,
            "strength_label": "",
            "overfit_level": "",
            "adoption_label": "",
            "failure_tags_json": "[]",
        }
        self.assertFalse(image_is_reviewed(image))
        image["failure_tags_json"] = json.dumps(["顔が弱い"], ensure_ascii=False)
        self.assertTrue(image_is_reviewed(image))

    def test_rubric_schema_includes_flexibility_rating(self) -> None:
        row = fetch_one("SELECT schema_json FROM evaluation_rubrics WHERE id = ?", ("lora_visual_eval_v1",))
        self.assertIsNotNone(row)
        schema = json.loads(row["schema_json"])
        self.assertIn("rating_flexibility", schema["fields"]["ratings"])

    def test_sd_scripts_validation_generation_uses_condition_defaults(self) -> None:
        args = common_gen_img_args(
            venv_python=Path("python.exe"),
            gen_img=Path("gen_img.py"),
            base_model_path=Path("base.safetensors"),
            out_dir=Path("out"),
            model_family="SDXL",
            mixed_precision="bf16",
            condition={"width": 768, "height": 1152, "cfg_scale": 6.5, "steps": 32, "sampler": "ddim"},
        )
        self.assertEqual(args[args.index("--W") + 1], "768")
        self.assertEqual(args[args.index("--H") + 1], "1152")
        self.assertEqual(args[args.index("--scale") + 1], "6.5")
        self.assertEqual(args[args.index("--steps") + 1], "32")
        self.assertEqual(args[args.index("--sampler") + 1], "ddim")

    def test_sd_scripts_validation_generation_normalizes_sampler_display_names(self) -> None:
        self.assertEqual(normalize_sd_scripts_sampler("Euler a"), "euler_a")
        self.assertEqual(normalize_sd_scripts_sampler("Euler_A"), "euler_a")
        self.assertEqual(normalize_sd_scripts_sampler("k euler a"), "k_euler_a")
        args = common_gen_img_args(
            venv_python=Path("python.exe"),
            gen_img=Path("gen_img.py"),
            base_model_path=Path("base.safetensors"),
            out_dir=Path("out"),
            model_family="SDXL",
            mixed_precision="bf16",
            condition={"sampler": "Euler a"},
        )
        self.assertEqual(args[args.index("--sampler") + 1], "euler_a")

    def test_init_db_does_not_seed_zuihou_project_from_hardcoded_ids(self) -> None:
        source = Path("app/db.py").read_text(encoding="utf-8")
        self.assertNotIn("seed_zuihou_project(conn)", source)


class ReferenceSetPhase111Tests(IsolatedDbTest):
    def create_dataset_fixture(self) -> tuple[int, int, Path]:
        now = utc_now()
        dataset_dir = self.root / "dataset"
        dataset_dir.mkdir()
        image_path = self.make_png(dataset_dir / "face_front.png")
        image_path.with_suffix(".txt").write_text("testchar, 1girl, face, upper body", encoding="utf-8")
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO datasets(
                    name, path, model_family, trigger_word, class_token, image_count,
                    caption_count, missing_caption_count, resolution_summary_json,
                    tag_summary_json, scan_status, memo, created_at, updated_at
                )
                VALUES ('ref dataset', ?, 'SDXL', 'testchar', 'person', 1, 1, 0, '{}', '{}', 'ok', '', ?, ?)
                """,
                (str(dataset_dir), now, now),
            )
            dataset_id = int(cur.lastrowid)
            cur = conn.execute(
                """
                INSERT INTO dataset_versions(dataset_id, version_no, trigger_word, image_count, caption_count, created_at, memo)
                VALUES (?, 1, 'testchar', 1, 1, ?, 'v1')
                """,
                (dataset_id, now),
            )
            version_id = int(cur.lastrowid)
        return dataset_id, version_id, image_path

    def test_legacy_reference_set_gets_v1_version(self) -> None:
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                "INSERT INTO reference_sets(name, trigger_word, created_at, updated_at, memo) VALUES ('legacy', 'testchar', ?, ?, '')",
                (now, now),
            )
            set_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO reference_images(reference_set_id, image_path, image_role, caption, sort_order, created_at) VALUES (?, ?, 'face_front', '', 0, ?)",
                (set_id, str(self.make_png(self.root / "legacy.png")), now),
            )
        init_db()
        ref = fetch_one("SELECT current_version_id FROM reference_sets WHERE id = ?", (set_id,))
        self.assertIsNotNone(ref["current_version_id"])
        image = fetch_one("SELECT reference_set_version_id, include_in_machine_review FROM reference_images WHERE reference_set_id = ?", (set_id,))
        self.assertEqual(image["reference_set_version_id"], ref["current_version_id"])
        self.assertEqual(image["include_in_machine_review"], 1)

    def test_create_character_reference_set_add_image_and_completeness_warning(self) -> None:
        dataset_id, version_id, image_path = self.create_dataset_fixture()
        set_id = create_reference_set(
            name="character ref",
            reference_type="character",
            dataset_id=dataset_id,
            dataset_version_id=version_id,
            trigger_word="testchar",
        )
        add_reference_image(reference_set_id=set_id, image_path=str(image_path), image_role="face_front", source_type="dataset")
        detail = reference_detail(set_id)
        self.assertEqual(detail["reference_set"]["reference_type"], "character")
        self.assertEqual(detail["reference_set"]["completeness_label"], "WARNING")
        self.assertEqual(len(detail["images"]), 1)
        self.assertIn("testchar", detail["images"][0]["caption_snapshot"])

    def test_style_reference_set_can_reach_ok_completeness(self) -> None:
        dataset_id, version_id, _ = self.create_dataset_fixture()
        set_id = create_reference_set(name="style ref", reference_type="style", dataset_id=dataset_id, dataset_version_id=version_id)
        roles = ["close_up", "upper_body", "full_body", "background_scene", "color_palette", "lighting"]
        for index, role in enumerate(roles):
            image_path = self.make_png(self.root / f"style_{index}.png")
            add_reference_image(reference_set_id=set_id, image_path=str(image_path), image_role=role)
        detail = reference_detail(set_id)
        self.assertEqual(detail["reference_set"]["completeness_label"], "OK")

    def test_reference_set_pages_render(self) -> None:
        dataset_id, version_id, image_path = self.create_dataset_fixture()
        set_id = create_reference_set(name="render ref", reference_type="character", dataset_id=dataset_id, dataset_version_id=version_id)
        add_reference_image(reference_set_id=set_id, image_path=str(image_path), image_role="face_front")
        listing = reference_sets(request=None)
        detail = reference_set_detail(request=None, set_id=set_id)
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(detail.status_code, 200)
        self.assertIn("Completeness", detail.body.decode("utf-8"))

    def test_project_profile_validation_link_reference_version(self) -> None:
        dataset_id, version_id, image_path = self.create_dataset_fixture()
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO lora_projects(name, dataset_id, current_dataset_version_id, trigger_word, base_model_path, status, created_at, updated_at, memo)
                VALUES ('project', ?, ?, 'testchar', 'base.safetensors', 'active', ?, ?, '')
                """,
                (dataset_id, version_id, now, now),
            )
            project_id = int(cur.lastrowid)
        set_id = create_reference_set(name="project ref", reference_type="character", dataset_id=dataset_id, dataset_version_id=version_id, project_id=project_id)
        add_reference_image(reference_set_id=set_id, image_path=str(image_path), image_role="face_front")
        set_project_default(set_id)
        ref = fetch_one("SELECT current_version_id FROM reference_sets WHERE id = ?", (set_id,))
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_jobs(project_id, name, dataset_id, preset_id, status, model_family, training_script, base_model_path, output_name, output_dir, run_dir, params_json, dataset_version_id, created_at, updated_at)
                VALUES (?, 'job', ?, 'sdxl_2d_face_adamw8bit_standard', 'completed', 'SDXL', 'sdxl_train_network.py', 'base.safetensors', 'out', 'out', 'run', '{}', ?, ?, ?)
                """,
                (project_id, dataset_id, version_id, now, now),
            )
            job_id = int(cur.lastrowid)
            cur = conn.execute(
                """
                INSERT INTO training_outputs(job_id, epoch, file_path, file_type, selected, created_at)
                VALUES (?, 1, 'model.safetensors', 'model', 1, ?)
                """,
                (job_id, now),
            )
            output_id = int(cur.lastrowid)
            cur = conn.execute(
                """
                INSERT INTO selected_lora_profiles(project_id, job_id, selected_output_id, profile_name, selected_model_path, default_validation_preset_id, reference_set_id, reference_set_version_id, created_at, updated_at)
                VALUES (?, ?, ?, 'profile', 'model.safetensors', 'standard_validation_v1', ?, ?, ?, ?)
                """,
                (project_id, job_id, output_id, set_id, ref["current_version_id"], now, now),
            )
            profile_id = int(cur.lastrowid)
        from app.services.validation_runs import create_validation_run

        run_id = create_validation_run(job_id, "standard_validation_v1", "base", "testchar", "", profile_id=profile_id)
        run = fetch_one("SELECT reference_set_id, reference_set_version_id FROM validation_runs WHERE id = ?", (run_id,))
        self.assertEqual(run["reference_set_id"], set_id)
        self.assertEqual(run["reference_set_version_id"], ref["current_version_id"])

    def test_reference_contact_sheet_and_report_export(self) -> None:
        dataset_id, version_id, image_path = self.create_dataset_fixture()
        set_id = create_reference_set(name="export ref", reference_type="character", dataset_id=dataset_id, dataset_version_id=version_id)
        add_reference_image(reference_set_id=set_id, image_path=str(image_path), image_role="face_front")
        paths = export_reference_artifacts(set_id)
        self.assertTrue(Path(paths["html"]).exists())
        self.assertTrue(Path(paths["markdown"]).exists())


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

    def test_sd_scripts_subprocess_env_strips_app_pythonpath(self) -> None:
        from app.services.training_runner import sd_scripts_subprocess_env

        with mock.patch.dict("os.environ", {"PYTHONPATH": "D:/app/.venv/Lib/site-packages", "PYTHONHOME": "D:/bad"}, clear=False):
            env = sd_scripts_subprocess_env()

        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertEqual(env["PYTHONUTF8"], "1")
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8:replace")

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

    def test_validation_generation_busy_redirects_back_to_job(self) -> None:
        project_id, dataset_id, version_id = self.create_project_fixture()
        job_id = self.add_completed_job(dataset_id, version_id, "sdxl_2d_face_adamw8bit_standard", project_id=project_id)
        from app.main import job_validation_generation_run
        from app.services.validation_runs import create_validation_run

        run_id = create_validation_run(job_id, "standard_validation_v1", "base", "testchar", "")
        with mock.patch("app.main.start_validation_generation", side_effect=RuntimeError("Validation生成 #8 が実行中です。")):
            response = job_validation_generation_run(job_id, run_id)
        self.assertEqual(response.status_code, 303)
        self.assertIn(f"/jobs/{job_id}?generation_error=", response.headers["location"])
        self.assertTrue(response.headers["location"].endswith("#validation-runs"))

    def test_stale_validation_generation_completed_log_is_reconciled(self) -> None:
        project_id, dataset_id, version_id = self.create_project_fixture()
        job_id = self.add_completed_job(dataset_id, version_id, "sdxl_2d_face_adamw8bit_standard", project_id=project_id)
        from app.services.validation_runs import create_validation_run

        run_id = create_validation_run(job_id, "standard_validation_v1", "base", "testchar", "")
        condition = fetch_one("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order LIMIT 1", (run_id,))
        output_dir = self.root / "exports" / "validation_runs" / f"validation_run_{run_id:06d}" / "generation" / "images"
        log_path = output_dir.parent / "generation.log"
        image_path = output_dir / f"vr{run_id:06d}_ec{int(condition['id']):06d}_{condition['condition_hash'][:12]}_test.png"
        self.make_png(image_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("100%|done\nINFO done!\n", encoding="utf-8")
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO validation_generation_runs(
                    validation_run_id, status, process_id, output_dir, log_path,
                    started_at, created_at, updated_at
                )
                VALUES (?, 'running', 999999, ?, ?, ?, ?, ?)
                """,
                (run_id, str(output_dir), str(log_path), now, now, now),
            )
            generation_id = int(cur.lastrowid)
        fixed = reconcile_stale_validation_generations()
        generation = fetch_one("SELECT * FROM validation_generation_runs WHERE id = ?", (generation_id,))
        imported = fetch_one("SELECT COUNT(*) AS count FROM validation_images WHERE validation_run_id = ?", (run_id,))
        self.assertEqual(fixed, 1)
        self.assertEqual(generation["status"], "completed")
        self.assertIsNone(generation["process_id"])
        self.assertEqual(generation["return_code"], 0)
        self.assertEqual(generation["generated_image_count"], 1)
        self.assertEqual(generation["imported_image_count"], 1)
        self.assertEqual(imported["count"], 1)

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

    def create_validation_generation_fixture(self, preset_id: str = "standard_validation_v1") -> tuple[int, int]:
        base_model = self.root / "models" / "base.safetensors"
        lora_model = self.root / "runs" / "job_000001" / "models" / "selected.safetensors"
        sd_scripts = self.root / "external" / "sd-scripts"
        for path in (base_model, lora_model, sd_scripts / "gen_img.py"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("stub", encoding="utf-8")
        project_id, dataset_id, version_id = self.create_project_fixture(str(base_model))
        job_id = self.add_completed_job(
            dataset_id,
            version_id,
            "sdxl_2d_face_standard_6epoch",
            base_model=str(base_model),
            project_id=project_id,
        )
        now = utc_now()
        with connect() as conn:
            output = conn.execute(
                """
                INSERT INTO training_outputs(job_id, epoch, file_path, file_type, selected, file_size, sha256, created_at)
                VALUES (?, 4, ?, 'model', 1, ?, 'abc123', ?)
                """,
                (job_id, str(lora_model), lora_model.stat().st_size, now),
            )
            selected_output_id = int(output.lastrowid)
            conn.execute(
                """
                INSERT INTO environments(
                    name, sd_scripts_path, venv_python_path, mixed_precision,
                    sd_scripts_commit_hash, status, created_at, updated_at
                )
                VALUES ('default', ?, ?, 'bf16', 'test', 'ok', ?, ?)
                """,
                (str(sd_scripts), sys.executable, now, now),
            )
        from app.services.validation_runs import create_validation_run

        run_id = create_validation_run(job_id, preset_id, str(base_model), "testchar", "")
        with connect() as conn:
            conn.execute(
                "UPDATE validation_runs SET selected_output_id = ? WHERE id = ?",
                (selected_output_id, run_id),
            )
        return run_id, selected_output_id

    def test_sd_scripts_generation_prepare_splits_baseline_and_lora_commands(self) -> None:
        from app.services.validation_generation import prepare_validation_generation

        run_id, _ = self.create_validation_generation_fixture()
        result = prepare_validation_generation(run_id)

        command_path = Path(result["command_argv"])
        payload = json.loads(command_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["skipped_hires_count"], 0)
        self.assertEqual(len(payload["commands"]), 2)
        baseline, lora = payload["commands"]
        self.assertEqual(baseline["name"], "baseline_weight_0_no_lora")
        self.assertNotIn("--network_weights", baseline["argv"])
        self.assertNotIn("--network_module", baseline["argv"])
        self.assertIn("--network_weights", lora["argv"])
        self.assertIn("--network_module", lora["argv"])
        baseline_prompt = Path(baseline["prompt_file"]).read_text(encoding="utf-8")
        lora_prompt = Path(lora["prompt_file"]).read_text(encoding="utf-8")
        self.assertNotIn("--am", baseline_prompt)
        self.assertIn("--am", lora_prompt)
        self.assertIn(".png", baseline_prompt)
        self.assertIn(".png", lora_prompt)
        self.assertRegex(baseline_prompt, r"--f\s+\S+\.png")
        expected = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
        self.assertEqual(len(Path(payload["all_prompt_file"]).read_text(encoding="utf-8").splitlines()), expected["count"])

    def test_validation_run_can_target_specific_output_epoch(self) -> None:
        from app.services.validation_runs import create_validation_run

        run_id, _ = self.create_validation_generation_fixture()
        run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
        job_id = int(run["job_id"])
        second_lora = self.root / "runs" / "job_000001" / "models" / "epoch7.safetensors"
        second_lora.parent.mkdir(parents=True, exist_ok=True)
        second_lora.write_text("stub", encoding="utf-8")
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_outputs(job_id, epoch, file_path, file_type, selected, file_size, sha256, created_at)
                VALUES (?, 7, ?, 'model', 0, ?, 'def456', ?)
                """,
                (job_id, str(second_lora), second_lora.stat().st_size, now),
            )
            output_id = int(cur.lastrowid)

        new_run_id = create_validation_run(job_id, "standard_validation_v1", "", "", "epoch 7 check", selected_output_id=output_id)

        new_run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (new_run_id,))
        self.assertEqual(new_run["selected_output_id"], output_id)
        self.assertIn("epoch 7", new_run["name"])
        self.assertEqual(new_run["lora_filename"], second_lora.name)

    def test_sd_scripts_generated_image_import_links_expected_condition_and_matrix(self) -> None:
        from app.services.validation_generation import (
            build_epoch_cross_matrix_html,
            generation_output_dir,
            import_generated_images,
            output_stem,
            prepare_validation_generation,
            write_validation_matrix,
        )
        from app.services.validation_runs import create_validation_run

        run_id, _ = self.create_validation_generation_fixture()
        prepare_validation_generation(run_id)
        condition = dict(fetch_one("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order LIMIT 1", (run_id,)))
        image_path = generation_output_dir(run_id) / f"{output_stem(run_id, condition)}.png"
        self.make_png(image_path)

        imported = import_generated_images(run_id)

        self.assertEqual(imported, 1)
        image = fetch_one("SELECT * FROM validation_images WHERE validation_run_id = ?", (run_id,))
        self.assertEqual(image["expected_condition_id"], condition["id"])
        self.assertEqual(image["condition_hash"], condition["condition_hash"])
        matrix_path = Path(write_validation_matrix(run_id))
        self.assertTrue(matrix_path.exists())
        matrix_html = matrix_path.read_text(encoding="utf-8")
        self.assertIn(f"/validation-images/{image['id']}", matrix_html)
        self.assertIn(f"/validation-runs/{run_id}/images/{image['id']}/matrix-review", matrix_html)
        self.assertIn(str(condition["id"]), matrix_html)

        run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
        job_id = int(run["job_id"])
        second_lora = self.root / "runs" / "job_000001" / "models" / "epoch7.safetensors"
        second_lora.parent.mkdir(parents=True, exist_ok=True)
        second_lora.write_text("stub", encoding="utf-8")
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_outputs(job_id, epoch, file_path, file_type, selected, file_size, sha256, created_at)
                VALUES (?, 7, ?, 'model', 0, ?, 'def456', ?)
                """,
                (job_id, str(second_lora), second_lora.stat().st_size, now),
            )
            output_id = int(cur.lastrowid)
        second_run_id = create_validation_run(job_id, "standard_validation_v1", "", "", "epoch 7 compare", selected_output_id=output_id)
        second_condition = dict(
            fetch_one(
                """
                SELECT * FROM validation_expected_conditions
                WHERE validation_run_id = ? AND prompt_key = ? AND seed = ? AND lora_weight = ? AND hires_enabled = ?
                LIMIT 1
                """,
                (second_run_id, condition["prompt_key"], condition["seed"], condition["lora_weight"], condition["hires_enabled"]),
            )
        )
        second_image_path = generation_output_dir(second_run_id) / f"{output_stem(second_run_id, second_condition)}.png"
        self.make_png(second_image_path)
        from app.main import register_validation_run_image

        register_validation_run_image(
            run_id=second_run_id,
            source_path=str(second_image_path),
            image_role="individual",
            prompt_key=second_condition["prompt_key"],
            seed=int(second_condition["seed"]),
            lora_weight=float(second_condition["lora_weight"]),
            hires_enabled=bool(second_condition["hires_enabled"]),
            memo="epoch 7",
        )
        cross_html = build_epoch_cross_matrix_html(job_id, [run_id, second_run_id])
        self.assertIn("Epoch横断Matrix", cross_html)
        self.assertIn("Epoch 4", cross_html)
        self.assertIn("Epoch 7", cross_html)
        self.assertIn(f"/validation-images/{image['id']}", cross_html)
        second_image = fetch_one("SELECT * FROM validation_images WHERE validation_run_id = ?", (second_run_id,))
        self.assertIn(f"/validation-images/{second_image['id']}", cross_html)

    def test_sd_scripts_generation_stop_without_process_marks_stopped(self) -> None:
        from app.services.validation_generation import stop_validation_generation

        run_id, _ = self.create_validation_generation_fixture()
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO validation_generation_runs(
                    validation_run_id, status, process_id, started_at,
                    created_at, updated_at
                )
                VALUES (?, 'running', NULL, ?, ?, ?)
                """,
                (run_id, now, now, now),
            )

        stop_validation_generation(run_id)

        generation = fetch_one("SELECT * FROM validation_generation_runs WHERE validation_run_id = ?", (run_id,))
        self.assertEqual(generation["status"], "stopped")
        self.assertIsNone(generation["process_id"])


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


class EmbeddingPhase112Tests(IsolatedDbTest):
    def make_dataset_version(self) -> tuple[int, int, Path]:
        dataset_dir = self.root / "dataset"
        self.make_png(dataset_dir / "img001.png")
        (dataset_dir / "img001.txt").write_text("testchar, portrait", encoding="utf-8")
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO datasets(name, path, model_family, trigger_word, class_token, image_count, caption_count, created_at, updated_at)
                VALUES ('Embedding Dataset', ?, 'SDXL', 'testchar', 'person', 1, 1, ?, ?)
                """,
                (str(dataset_dir), now, now),
            )
            dataset_id = int(cur.lastrowid)
            cur = conn.execute(
                """
                INSERT INTO dataset_versions(dataset_id, version_no, trigger_word, image_count, caption_count, created_at)
                VALUES (?, 1, 'testchar', 1, 1, ?)
                """,
                (dataset_id, now),
            )
            version_id = int(cur.lastrowid)
        return dataset_id, version_id, dataset_dir

    def make_sample_job(self, dataset_id: int, version_id: int) -> int:
        now = utc_now()
        image = self.make_png(self.root / "runs" / "job_000001" / "samples" / "sample.png")
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_jobs(name, dataset_id, dataset_version_id, preset_id, status, model_family, training_script,
                    base_model_path, output_name, output_dir, run_dir, params_json, created_at, updated_at)
                VALUES ('Embedding Job', ?, ?, 'preset', 'completed', 'SDXL', 'sdxl_train_network.py',
                    'model.safetensors', 'out', 'out', 'run', '{}', ?, ?)
                """,
                (dataset_id, version_id, now, now),
            )
            job_id = int(cur.lastrowid)
            conn.execute("INSERT INTO sample_images(job_id, image_path, created_at) VALUES (?, ?, ?)", (job_id, str(image), now))
        return job_id

    def make_validation_run_with_image(self, job_id: int) -> int:
        from app.services.validation_runs import create_validation_run

        run_id = create_validation_run(job_id, "quick_validation_v1", "base", "testchar", "embedding test")
        image = self.make_png(settings.EXPORTS_DIR / "validation_runs" / f"validation_run_{run_id:06d}" / "images" / "val.png")
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO validation_images(job_id, validation_run_id, image_path, image_role, created_at, updated_at)
                VALUES (?, ?, ?, 'individual', ?, ?)
                """,
                (job_id, run_id, str(image), now, now),
            )
        return run_id

    def run_embedding_job_sync(self, job_type: str, target_id: int) -> int:
        from app.services.embedding_service import create_embedding_job
        from app.services.embedding_worker import run_embedding_job

        embedding_job_id = create_embedding_job(job_type, target_id)
        self.assertEqual(run_embedding_job(embedding_job_id), 0)
        return embedding_job_id

    def test_mock_model_and_settings_are_seeded_and_preflight_ok(self) -> None:
        from app.services.embedding_service import load_embedding_settings, provider_preflight

        model = fetch_one("SELECT * FROM embedding_models WHERE id = 'mock_image_512'")
        self.assertIsNotNone(model)
        self.assertEqual(model["provider"], "mock")
        settings_row = load_embedding_settings()
        self.assertEqual(settings_row["active_embedding_model_id"], "mock_image_512")
        result = provider_preflight("mock_image_512")
        self.assertEqual(result["status"], "OK")

    def test_transformers_clip_model_is_seeded_without_implicit_download(self) -> None:
        from app.services.embedding_service import provider_preflight

        model = fetch_one("SELECT * FROM embedding_models WHERE id = 'transformers_clip_vit_base_patch32'")
        self.assertIsNotNone(model)
        self.assertEqual(model["provider"], "transformers_clip")
        self.assertEqual(model["model_name"], "openai/clip-vit-base-patch32")
        self.assertEqual(model["vector_dim"], 512)
        result = provider_preflight("transformers_clip_vit_base_patch32", deep=False)
        self.assertFalse(result["download_allowed"])
        self.assertTrue(any(check["name"] == "model download" for check in result["checks"]))

    def test_open_clip_model_is_seeded_without_implicit_download(self) -> None:
        from app.services.embedding_service import provider_preflight

        model = fetch_one("SELECT * FROM embedding_models WHERE id = 'open_clip_vit_b32_laion2b'")
        self.assertIsNotNone(model)
        self.assertEqual(model["provider"], "open_clip")
        self.assertEqual(model["model_name"], "ViT-B-32")
        self.assertEqual(model["pretrained"], "laion2b_s34b_b79k")
        self.assertEqual(model["vector_dim"], 512)
        self.assertEqual(model["dtype_default"], "fp16")
        self.assertEqual(model["batch_size_default"], 8)
        self.assertEqual(model["allow_download"], 1)
        result = provider_preflight("open_clip_vit_b32_laion2b", deep=False)
        self.assertFalse(result["download_allowed"])
        self.assertEqual(result["provider"], "open_clip")
        self.assertTrue(any(check["name"] == "model download" for check in result["checks"]))

    def test_transformers_clip_device_dtype_resolution(self) -> None:
        from app.services.embedding_worker import resolve_torch_device_and_dtype

        class FakeCuda:
            def __init__(self, available: bool) -> None:
                self.available = available

            def is_available(self) -> bool:
                return self.available

        class FakeTorch:
            float32 = "float32"
            float16 = "float16"
            bfloat16 = "bfloat16"

            def __init__(self, available: bool) -> None:
                self.cuda = FakeCuda(available)

            def device(self, name: str) -> str:
                return name

        device, dtype, device_name, dtype_name = resolve_torch_device_and_dtype(FakeTorch(True), "auto", "fp16")
        self.assertEqual((device, dtype, device_name, dtype_name), ("cuda", "float16", "cuda", "fp16"))
        device, dtype, device_name, dtype_name = resolve_torch_device_and_dtype(FakeTorch(False), "cuda", "fp16")
        self.assertEqual((device, dtype, device_name, dtype_name), ("cpu", "float32", "cpu", "fp32"))

    def test_real_embedding_provider_waits_for_training_or_generation(self) -> None:
        from app.services.embedding_service import assert_embedding_can_start

        dataset_id, version_id, _ = self.make_dataset_version()
        job_id = self.make_sample_job(dataset_id, version_id)
        with connect() as conn:
            conn.execute("UPDATE training_jobs SET status = 'running' WHERE id = ?", (job_id,))
        with self.assertRaisesRegex(RuntimeError, "学習ジョブ"):
            assert_embedding_can_start({"provider": "transformers_clip"})
        with self.assertRaisesRegex(RuntimeError, "学習ジョブ"):
            assert_embedding_can_start({"provider": "open_clip"})
        assert_embedding_can_start({"provider": "mock"})

    def test_dataset_reference_sample_and_validation_embeddings_are_created(self) -> None:
        from app.services.embedding_service import embedding_coverage

        dataset_id, version_id, dataset_dir = self.make_dataset_version()
        reference_set_id = create_reference_set(name="Embedding Reference", reference_type="character", dataset_id=dataset_id, dataset_version_id=version_id, trigger_word="testchar")
        ref_image_id = add_reference_image(reference_set_id=reference_set_id, image_path=str(dataset_dir / "img001.png"), image_role="face_front", source_type="dataset")
        reference = fetch_one("SELECT current_version_id FROM reference_sets WHERE id = ?", (reference_set_id,))
        job_id = self.make_sample_job(dataset_id, version_id)
        run_id = self.make_validation_run_with_image(job_id)

        dataset_job_id = self.run_embedding_job_sync("dataset_version", version_id)
        reference_job_id = self.run_embedding_job_sync("reference_set_version", reference["current_version_id"])
        sample_job_id = self.run_embedding_job_sync("training_job_samples", job_id)
        validation_job_id = self.run_embedding_job_sync("validation_run", run_id)

        for embedding_job_id in [dataset_job_id, reference_job_id, sample_job_id, validation_job_id]:
            job = fetch_one("SELECT status, ready_count FROM embedding_jobs WHERE id = ?", (embedding_job_id,))
            self.assertEqual(job["status"], "completed")
            self.assertGreaterEqual(job["ready_count"], 1)

        embeddings = fetch_all("SELECT * FROM image_embeddings ORDER BY id")
        self.assertEqual({row["source_type"] for row in embeddings}, {"dataset_image", "reference_image", "sample_image", "validation_image"})
        for row in embeddings:
            self.assertTrue(Path(row["embedding_path"]).exists())
            self.assertEqual(row["vector_dim"], 512)
            self.assertEqual(row["status"], "ready")

        self.assertEqual(embedding_coverage("dataset_version", version_id)["ready"], 1)
        self.assertEqual(embedding_coverage("reference_set_version", reference["current_version_id"])["ready"], 1)
        self.assertEqual(embedding_coverage("training_job_samples", job_id)["ready"], 1)
        self.assertEqual(embedding_coverage("validation_run", run_id)["ready"], 1)
        self.assertEqual(ref_image_id, fetch_one("SELECT source_id FROM image_embeddings WHERE source_type = 'reference_image'")["source_id"])

    def test_dataset_image_embeddings_do_not_collide_between_versions(self) -> None:
        now = utc_now()
        version_ids = []
        for index in [1, 2]:
            dataset_dir = self.root / f"dataset_{index}"
            self.make_png(dataset_dir / "img001.png")
            (dataset_dir / "img001.txt").write_text(f"testchar, version {index}", encoding="utf-8")
            with connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO datasets(name, path, model_family, trigger_word, class_token, image_count, caption_count, created_at, updated_at)
                    VALUES (?, ?, 'SDXL', 'testchar', 'person', 1, 1, ?, ?)
                    """,
                    (f"Embedding Dataset {index}", str(dataset_dir), now, now),
                )
                dataset_id = int(cur.lastrowid)
                cur = conn.execute(
                    """
                    INSERT INTO dataset_versions(dataset_id, version_no, trigger_word, image_count, caption_count, created_at)
                    VALUES (?, 1, 'testchar', 1, 1, ?)
                    """,
                    (dataset_id, now),
                )
                version_ids.append(int(cur.lastrowid))

        for version_id in version_ids:
            self.run_embedding_job_sync("dataset_version", version_id)

        rows = fetch_all(
            """
            SELECT source_id, dataset_version_id, source_path, embedding_path
            FROM image_embeddings
            WHERE source_type = 'dataset_image'
            ORDER BY dataset_version_id
            """
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["source_id"] for row in rows}, {1})
        self.assertEqual({row["dataset_version_id"] for row in rows}, set(version_ids))
        self.assertEqual(len({row["source_path"] for row in rows}), 2)
        self.assertEqual(len({row["embedding_path"] for row in rows}), 2)

    def test_stale_detection_and_recompute_stale(self) -> None:
        from app.services.embedding_service import create_embedding_job, embedding_coverage
        from app.services.embedding_worker import run_embedding_job

        _, version_id, dataset_dir = self.make_dataset_version()
        self.run_embedding_job_sync("dataset_version", version_id)
        image_path = dataset_dir / "img001.png"
        Image.new("RGB", (9, 9), color=(200, 10, 40)).save(image_path)
        self.assertEqual(embedding_coverage("dataset_version", version_id)["stale"], 1)
        stale_job_id = create_embedding_job("dataset_version", version_id, recompute="stale")
        run_embedding_job(stale_job_id)
        self.assertEqual(embedding_coverage("dataset_version", version_id)["ready"], 1)

    def test_embedding_pages_render(self) -> None:
        from app.main import dataset_detail, embedding_settings_page, reference_set_detail as reference_page, validation_run_detail as validation_page

        dataset_id, version_id, dataset_dir = self.make_dataset_version()
        reference_set_id = create_reference_set(name="Embedding Reference", reference_type="character", dataset_id=dataset_id, dataset_version_id=version_id)
        add_reference_image(reference_set_id=reference_set_id, image_path=str(dataset_dir / "img001.png"), image_role="face_front")
        job_id = self.make_sample_job(dataset_id, version_id)
        run_id = self.make_validation_run_with_image(job_id)

        self.assertIn("Embedding設定", embedding_settings_page(request=None).body.decode("utf-8"))
        self.assertIn("Embedding Coverage", dataset_detail(request=None, dataset_id=dataset_id).body.decode("utf-8"))
        self.assertIn("Reference Embedding Coverage", reference_page(request=None, set_id=reference_set_id).body.decode("utf-8"))
        self.assertIn("Validation Image Embedding Coverage", validation_page(request=None, run_id=run_id).body.decode("utf-8"))

    def test_machine_review_scores_samples_and_validation_images(self) -> None:
        from app.services.embedding_service import embedding_coverage
        from app.services.machine_review import load_machine_review_settings, reference_set_readiness, run_machine_review
        from app.services.recommendations import regenerate_recommendations

        dataset_id, version_id, dataset_dir = self.make_dataset_version()
        self.make_png(dataset_dir / "img002.png")
        (dataset_dir / "img002.txt").write_text("testchar, full body", encoding="utf-8")
        reference_set_id = create_reference_set(
            name="Machine Review Reference",
            reference_type="character",
            dataset_id=dataset_id,
            dataset_version_id=version_id,
            trigger_word="testchar",
        )
        add_reference_image(reference_set_id=reference_set_id, image_path=str(dataset_dir / "img001.png"), image_role="face_front", source_type="dataset")
        reference = fetch_one("SELECT current_version_id FROM reference_sets WHERE id = ?", (reference_set_id,))
        job_id = self.make_sample_job(dataset_id, version_id)
        run_id = self.make_validation_run_with_image(job_id)

        self.run_embedding_job_sync("dataset_version", version_id)
        self.run_embedding_job_sync("reference_set_version", reference["current_version_id"])
        self.run_embedding_job_sync("training_job_samples", job_id)
        self.run_embedding_job_sync("validation_run", run_id)

        settings_row = load_machine_review_settings()
        self.assertEqual(settings_row["active_embedding_model_id"], "mock_image_512")
        sample_result = run_machine_review("training_job_samples", job_id)
        validation_result = run_machine_review("validation_run_images", run_id)
        self.assertEqual(sample_result["scored"], 1)
        self.assertEqual(validation_result["scored"], 1)

        sample_score = fetch_one("SELECT * FROM machine_review_scores WHERE source_type = 'sample_image' AND job_id = ?", (job_id,))
        validation_score = fetch_one("SELECT * FROM machine_review_scores WHERE source_type = 'validation_image' AND validation_run_id = ?", (run_id,))
        self.assertIsNotNone(sample_score)
        self.assertIsNotNone(validation_score)
        self.assertEqual(sample_score["provider"], "mock")
        self.assertEqual(sample_score["confidence_label"], "low")
        self.assertEqual(sample_score["assist_label"], "low_confidence")
        self.assertEqual(sample_score["overfit_risk_label"], "unknown")
        self.assertIsNotNone(sample_score["nearest_reference_image_id"])
        self.assertIsNotNone(sample_score["nearest_dataset_image_id"])
        self.assertIsNotNone(sample_score["nearest_dataset_similarity"])
        self.assertIsNotNone(sample_score["dataset_top1_margin"])
        self.assertIsNotNone(validation_score["nearest_dataset_image_id"])

        coverage = embedding_coverage("reference_set_version", reference["current_version_id"])
        readiness = reference_set_readiness(reference_detail(reference_set_id)["reference_set"], coverage)
        self.assertIn(readiness["label"], {"OK", "WARNING"})
        recommendations = regenerate_recommendations(job_id)
        self.assertTrue(any(row["recommendation_type"] == "machine_review_notice" for row in recommendations))

    def test_machine_review_pages_render(self) -> None:
        from app.main import embedding_settings_page, job_detail, reference_set_detail as reference_page, validation_run_detail as validation_page

        dataset_id, version_id, dataset_dir = self.make_dataset_version()
        reference_set_id = create_reference_set(name="Machine Review Reference", reference_type="character", dataset_id=dataset_id, dataset_version_id=version_id)
        add_reference_image(reference_set_id=reference_set_id, image_path=str(dataset_dir / "img001.png"), image_role="face_front")
        job_id = self.make_sample_job(dataset_id, version_id)
        run_id = self.make_validation_run_with_image(job_id)

        self.assertIn("機械補助レビュー設定", embedding_settings_page(request=None).body.decode("utf-8"))
        self.assertIn("機械補助レビュー", job_detail(request=None, job_id=job_id).body.decode("utf-8"))
        self.assertIn("Machine Review readiness", reference_page(request=None, set_id=reference_set_id).body.decode("utf-8"))
        self.assertIn("機械補助レビュー", validation_page(request=None, run_id=run_id).body.decode("utf-8"))

    def test_machine_review_skips_stale_or_missing_embeddings(self) -> None:
        from app.services.machine_review import run_machine_review

        dataset_id, version_id, dataset_dir = self.make_dataset_version()
        reference_set_id = create_reference_set(
            name="Machine Review Reference",
            reference_type="character",
            dataset_id=dataset_id,
            dataset_version_id=version_id,
        )
        add_reference_image(reference_set_id=reference_set_id, image_path=str(dataset_dir / "img001.png"), image_role="face_front")
        reference = fetch_one("SELECT current_version_id FROM reference_sets WHERE id = ?", (reference_set_id,))
        job_id = self.make_sample_job(dataset_id, version_id)
        self.run_embedding_job_sync("dataset_version", version_id)
        self.run_embedding_job_sync("reference_set_version", reference["current_version_id"])
        self.run_embedding_job_sync("training_job_samples", job_id)

        sample = fetch_one("SELECT image_path FROM sample_images WHERE job_id = ? LIMIT 1", (job_id,))
        Path(sample["image_path"]).unlink()
        result = run_machine_review("training_job_samples", job_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["scored"], 0)
        self.assertEqual(fetch_one("SELECT skipped_count, failed_count FROM machine_review_jobs WHERE id = ?", (result["job_id"],))["skipped_count"], 1)
        self.assertEqual(fetch_one("SELECT failed_count FROM machine_review_jobs WHERE id = ?", (result["job_id"],))["failed_count"], 0)

    def test_machine_review_background_start_and_readiness_actions(self) -> None:
        from app.services.machine_review import create_machine_review_job, machine_review_readiness, start_machine_review_job

        dataset_id, version_id, dataset_dir = self.make_dataset_version()
        reference_set_id = create_reference_set(
            name="Machine Review Reference",
            reference_type="character",
            dataset_id=dataset_id,
            dataset_version_id=version_id,
        )
        add_reference_image(reference_set_id=reference_set_id, image_path=str(dataset_dir / "img001.png"), image_role="face_front")
        job_id = self.make_sample_job(dataset_id, version_id)
        reference = fetch_one("SELECT current_version_id FROM reference_sets WHERE id = ?", (reference_set_id,))

        readiness = machine_review_readiness("training_job_samples", job_id)
        self.assertTrue(any("Reference画像Embedding" in item for item in readiness["next_actions"]))
        self.assertTrue(any("Dataset画像Embedding" in item for item in readiness["next_actions"]))
        self.assertTrue(any("サンプル画像Embedding" in item for item in readiness["next_actions"]))

        self.run_embedding_job_sync("dataset_version", version_id)
        self.run_embedding_job_sync("reference_set_version", reference["current_version_id"])
        self.run_embedding_job_sync("training_job_samples", job_id)
        readiness = machine_review_readiness("training_job_samples", job_id)
        self.assertTrue(any("機械補助レビュー" in item for item in readiness["next_actions"]))

        machine_review_job_id = create_machine_review_job("training_job_samples", job_id)
        fake_proc = types.SimpleNamespace(pid=4242)
        with mock.patch("app.services.machine_review.subprocess.Popen", return_value=fake_proc) as popen:
            start_machine_review_job(machine_review_job_id)
        row = fetch_one("SELECT status, process_id, log_path FROM machine_review_jobs WHERE id = ?", (machine_review_job_id,))
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["process_id"], 4242)
        self.assertIn("machine_review_job_", row["log_path"])
        popen.assert_called_once()


class StartHelperTests(unittest.TestCase):
    def test_start_lora_studio_does_not_release_7865_by_default(self) -> None:
        import start_lora_studio

        fake_uvicorn = types.SimpleNamespace(run=mock.Mock())
        with mock.patch.object(sys, "argv", ["start_lora_studio.py", "--port", "8768", "--no-browser", "--skip-sd-scripts-setup"]), \
            mock.patch.dict(sys.modules, {"uvicorn": fake_uvicorn}), \
            mock.patch.object(start_lora_studio, "init_db"), \
            mock.patch.object(start_lora_studio, "release_port") as release_port:
            start_lora_studio.main()
        release_port.assert_not_called()

    def test_start_lora_studio_force_releases_only_app_port(self) -> None:
        import start_lora_studio

        fake_uvicorn = types.SimpleNamespace(run=mock.Mock())
        with mock.patch.object(sys, "argv", ["start_lora_studio.py", "--port", "8768", "--no-browser", "--skip-sd-scripts-setup", "--force-release-port"]), \
            mock.patch.dict(sys.modules, {"uvicorn": fake_uvicorn}), \
            mock.patch.object(start_lora_studio, "init_db"), \
            mock.patch.object(start_lora_studio, "release_port") as release_port:
            start_lora_studio.main()
        release_port.assert_called_once_with(8768)

    def test_release_port_does_not_kill_process_tree(self) -> None:
        import start_lora_studio

        with mock.patch.object(start_lora_studio, "find_listening_pids", return_value={1234}), \
            mock.patch.object(start_lora_studio.os, "getpid", return_value=9999), \
            mock.patch.object(start_lora_studio.subprocess, "run") as run:
            start_lora_studio.release_port(8768)

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
