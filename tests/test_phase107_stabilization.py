from __future__ import annotations

import sys
import tempfile
import types
import unittest
import hashlib
import json
import os
import subprocess
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
from app.services.review_sessions import ensure_candidate_review_plan, reconcile_stale_review_sessions, review_session_summary, write_review_matrix
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

    def make_tree_old(self, path: Path, timestamp: int = 1_577_836_800) -> None:
        if path.is_file():
            os.utime(path, (timestamp, timestamp))
            return
        for child in path.rglob("*"):
            if child.exists():
                os.utime(child, (timestamp, timestamp))
        if path.exists():
            os.utime(path, (timestamp, timestamp))


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


class ReviewSessionSummaryTests(IsolatedDbTest):
    def add_session(self, *, status: str, epochs: list[int], expected: int, imported: int, scored: int, matrix: bool) -> int:
        now = utc_now()
        matrix_path = ""
        if matrix:
            matrix_file = self.root / "exports" / "review_sessions" / f"review_session_{status}_{expected}" / "review_matrix.html"
            matrix_file.parent.mkdir(parents=True, exist_ok=True)
            matrix_file.write_text("<html></html>", encoding="utf-8")
            matrix_path = str(matrix_file)
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO review_sessions(
                    job_id, project_id, name, preset_id, candidate_epochs_json,
                    expected_image_count, imported_image_count, scored_image_count,
                    status, matrix_path, created_at, updated_at
                )
                VALUES (21, 1, ?, 'candidate_epoch_review_v1', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"session {status}",
                    json.dumps(epochs),
                    expected,
                    imported,
                    scored,
                    status,
                    matrix_path,
                    now,
                    now,
                ),
            )
            session_id = int(cur.lastrowid)
            for index in range(expected):
                conn.execute(
                    """
                    INSERT INTO review_session_conditions(
                        review_session_id, job_id, epoch, output_id, lora_path,
                        prompt_key, prompt_role, prompt, negative_prompt, seed,
                        lora_weight, width, height, sampler, steps, cfg_scale,
                        condition_hash, expected_order, created_at, updated_at
                    )
                    VALUES (?, 21, ?, NULL, '', 'basic_face', 'face', 'prompt', '', 111111,
                            0.6, 1024, 1024, 'euler_a', 28, 7.0, ?, ?, ?, ?)
                    """,
                    (session_id, epochs[index % len(epochs)], f"hash-{session_id}-{index}", index + 1, now, now),
                )
            for index in range(imported):
                image_path = self.make_png(self.root / "exports" / "review_sessions" / f"review_session_{session_id:06d}" / "images" / f"{index}.png")
                conn.execute(
                    """
                    INSERT INTO review_session_images(
                        review_session_id, job_id, epoch, prompt_key, seed, lora_weight,
                        image_path, created_at, updated_at
                    )
                    VALUES (?, 21, ?, 'basic_face', 111111, 0.6, ?, ?, ?)
                    """,
                    (session_id, epochs[index % len(epochs)], str(image_path), now, now),
                )
        return session_id

    def test_summary_uses_one_current_session_without_mixing(self) -> None:
        planned_id = self.add_session(status="planned", epochs=[4, 5, 6, 7, 8, 9, 10], expected=42, imported=0, scored=0, matrix=False)
        completed_id = self.add_session(status="completed", epochs=[2, 3, 4, 5, 6], expected=30, imported=30, scored=30, matrix=True)

        summary = review_session_summary(21)

        self.assertEqual(summary["current"]["session_id"], completed_id)
        self.assertEqual(summary["current"]["candidate_epochs"], [2, 3, 4, 5, 6])
        self.assertEqual(summary["current"]["condition_count"], 30)
        self.assertEqual(summary["current"]["registered_image_count"], 30)
        self.assertEqual(summary["current"]["machine_review_count"], 30)
        self.assertTrue(summary["current"]["can_open_matrix"])
        self.assertEqual(summary["other_sessions"][0]["session_id"], planned_id)
        self.assertEqual(summary["other_sessions"][0]["condition_count"], 42)
        self.assertEqual(summary["other_sessions"][0]["registered_image_count"], 0)

    def test_summary_can_select_planned_session_as_current(self) -> None:
        planned_id = self.add_session(status="planned", epochs=[4, 5, 6], expected=18, imported=0, scored=0, matrix=False)
        completed_id = self.add_session(status="completed", epochs=[6, 7], expected=12, imported=12, scored=12, matrix=True)

        summary = review_session_summary(21, current_session_id=planned_id)

        self.assertEqual(summary["current"]["session_id"], planned_id)
        self.assertEqual(summary["current"]["condition_count"], 18)
        self.assertFalse(summary["current"]["can_open_matrix"])
        self.assertEqual(summary["other_sessions"][0]["session_id"], completed_id)

    def test_summary_uses_live_generated_count_when_db_count_is_stale(self) -> None:
        session_id = self.add_session(status="stopped", epochs=[4], expected=6, imported=0, scored=0, matrix=False)
        output_dir = self.root / "exports" / "review_sessions" / f"review_session_{session_id:06d}" / "images"
        self.make_png(output_dir / "orphan_generated.png")
        with connect() as conn:
            conn.execute(
                "UPDATE review_sessions SET output_dir = ?, generated_image_count = 0 WHERE id = ?",
                (str(output_dir), session_id),
            )

        summary = review_session_summary(21, current_session_id=session_id)

        self.assertEqual(summary["current"]["registered_image_count"], 0)
        self.assertEqual(summary["current"]["generated_image_count"], 1)
        self.assertEqual(summary["current"]["primary_action"], "retry")
        self.assertTrue(summary["current"]["retry_available"])


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

    def test_init_db_does_not_seed_legacy_project_from_hardcoded_ids(self) -> None:
        source = Path("app/db.py").read_text(encoding="utf-8")
        self.assertNotIn("seed_legacy_project(conn)", source)


class ReviewPreparationPhase116Tests(IsolatedDbTest):
    def create_completed_job_with_outputs(self) -> int:
        now = utc_now()
        dataset_dir = self.root / "dataset"
        dataset_dir.mkdir()
        with connect() as conn:
            dataset_id = int(
                conn.execute(
                    """
                    INSERT INTO datasets(
                        name, path, model_family, trigger_word, class_token, image_count,
                        caption_count, missing_caption_count, resolution_summary_json,
                        tag_summary_json, scan_status, memo, created_at, updated_at
                    )
                    VALUES ('review dataset', ?, 'SDXL', 'testchar', 'person', 3, 3, 0, '{}', '{}', 'ok', '', ?, ?)
                    """,
                    (str(dataset_dir), now, now),
                ).lastrowid
            )
            version_id = int(
                conn.execute(
                    """
                    INSERT INTO dataset_versions(dataset_id, version_no, trigger_word, image_count, caption_count, created_at, memo)
                    VALUES (?, 1, 'testchar', 3, 3, ?, 'v1')
                    """,
                    (dataset_id, now),
                ).lastrowid
            )
            job_id = int(
                conn.execute(
                    """
                    INSERT INTO training_jobs(
                        name, dataset_id, preset_id, status, model_family, training_script,
                        base_model_path, output_name, output_dir, run_dir, params_json,
                        adopted_epoch, dataset_version_id, created_at, updated_at
                    )
                    VALUES ('review job', ?, 'sdxl_2d_face_adamw8bit_standard', 'completed',
                            'SDXL', 'sdxl_train_network.py', 'base.safetensors',
                            'review_job', 'out', 'run', '{}', 2, ?, ?, ?)
                    """,
                    (dataset_id, version_id, now, now),
                ).lastrowid
            )
            for epoch, loss in [(1, 0.15), (2, 0.10), (3, 0.12)]:
                model_path = self.root / "runs" / f"job_{job_id:06d}" / "models" / f"review-{epoch:06d}.safetensors"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(f"model-{epoch}".encode("utf-8"))
                conn.execute(
                    """
                    INSERT INTO training_outputs(job_id, epoch, file_path, file_type, file_size, sha256, selected, created_at)
                    VALUES (?, ?, ?, 'model', ?, ?, ?, ?)
                    """,
                    (job_id, epoch, str(model_path), model_path.stat().st_size, hashlib.sha256(model_path.read_bytes()).hexdigest(), 1 if epoch == 2 else 0, now),
                )
                conn.execute(
                    """
                    INSERT INTO training_epoch_summaries(
                        job_id, epoch, step_start, step_end, metric_count, avg_loss,
                        min_loss, max_loss, final_loss, moving_avg_final_loss,
                        spike_count, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, 10, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (job_id, epoch, epoch * 10 - 9, epoch * 10, loss, loss, loss + 0.01, loss, loss, now, now),
                )
        return job_id

    def test_candidate_review_plan_creates_expected_conditions(self) -> None:
        job_id = self.create_completed_job_with_outputs()
        session = ensure_candidate_review_plan(job_id, force=True)
        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session["preset_id"], "candidate_epoch_review_v1")
        self.assertEqual(session["expected_image_count"], 18)
        conditions = fetch_all("SELECT * FROM review_session_conditions WHERE review_session_id = ?", (session["id"],))
        self.assertEqual(len(conditions), 18)
        self.assertEqual(sorted({row["prompt_key"] for row in conditions}), ["basic_face", "expression_pose", "full_body"])
        self.assertEqual(sorted({float(row["lora_weight"]) for row in conditions}), [0.6, 0.8])

    def test_standard_candidate_comparison_group_creates_standard_runs(self) -> None:
        from app.services.candidate_comparisons import candidate_standard_estimate, ensure_candidate_standard_comparison_group

        job_id = self.create_completed_job_with_outputs()
        estimate = candidate_standard_estimate(job_id)
        group = ensure_candidate_standard_comparison_group(job_id)

        self.assertEqual(group["status"], "planned")
        self.assertEqual(group["expected_total_images"], 135)
        self.assertEqual(estimate["logical_image_count"], 135)
        self.assertEqual(estimate["physical_generation_count"], 117)
        self.assertEqual(estimate["shared_baseline_count"], 9)
        self.assertEqual(estimate["saved_image_count"], 18)
        self.assertEqual(len(group["candidate_epochs"]), 3)
        self.assertEqual(len(group["validation_run_ids"]), 3)
        runs = fetch_all("SELECT * FROM validation_runs WHERE id IN ({}) ORDER BY selected_epoch".format(",".join("?" for _ in group["validation_run_ids"])), tuple(group["validation_run_ids"]))
        self.assertEqual(len(runs), 3)
        self.assertEqual({row["validation_run_kind"] for row in runs}, {"candidate_standard_comparison"})
        self.assertEqual({row["validation_preset_id"] for row in runs}, {"standard_validation_v1"})
        self.assertEqual({int(row["expected_image_count"]) for row in runs}, {45})
        for run in runs:
            count = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ?", (run["id"],))
            self.assertEqual(count["count"], 45)

    def test_standard_candidate_comparison_group_is_idempotent(self) -> None:
        from app.services.candidate_comparisons import ensure_candidate_standard_comparison_group

        job_id = self.create_completed_job_with_outputs()
        first = ensure_candidate_standard_comparison_group(job_id)
        second = ensure_candidate_standard_comparison_group(job_id)

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["validation_run_ids"], second["validation_run_ids"])
        group_count = fetch_one("SELECT COUNT(*) AS count FROM candidate_comparison_groups WHERE job_id = ?", (job_id,))
        self.assertEqual(group_count["count"], 1)

    def test_review_matrix_html_is_written(self) -> None:
        job_id = self.create_completed_job_with_outputs()
        session = ensure_candidate_review_plan(job_id, force=True)
        assert session is not None
        condition = fetch_one("SELECT * FROM review_session_conditions WHERE review_session_id = ? ORDER BY id LIMIT 1", (session["id"],))
        image_path = self.make_png(self.root / "exports" / "review_sessions" / f"review_session_{int(session['id']):06d}" / "images" / "sample.png")
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO review_session_images(
                    review_session_id, condition_id, job_id, epoch, output_id,
                    prompt_key, prompt_role, seed, lora_weight, image_path,
                    file_size, sha256, width, height, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 8, 8, ?, ?)
                """,
                (
                    session["id"],
                    condition["id"],
                    job_id,
                    condition["epoch"],
                    condition["output_id"],
                    condition["prompt_key"],
                    condition["prompt_role"],
                    condition["seed"],
                    condition["lora_weight"],
                    str(image_path),
                    image_path.stat().st_size,
                    hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    now,
                    now,
                ),
            )
        matrix_path = Path(write_review_matrix(int(session["id"])))
        self.assertTrue(matrix_path.exists())
        html = matrix_path.read_text(encoding="utf-8")
        self.assertIn("候補epochレビューMatrix", html)
        self.assertIn("機械補助レビュー", html)
        self.assertIn("sample.png", html)

    def test_planned_review_session_keeps_validation_next_collapsed(self) -> None:
        from app.main import review_session_detail

        job_id = self.create_completed_job_with_outputs()
        session = ensure_candidate_review_plan(job_id, force=True)
        assert session is not None

        body = review_session_detail(request=None, session_id=int(session["id"])).body.decode("utf-8")

        self.assertIn("このReview Sessionはまだ未生成です。正式検証は候補epochを確認して採用LoRAを選んだ後に実行します。", body)
        self.assertIn("<details id=\"validation-next\"", body)
        self.assertNotIn("<section id=\"validation-next\" class=\"action-panel\">", body)
        self.assertNotIn("採用epochの検証Runを作成", body)

    def test_completed_review_session_shows_validation_next_panel(self) -> None:
        from app.main import review_session_detail

        job_id = self.create_completed_job_with_outputs()
        session = ensure_candidate_review_plan(job_id, force=True)
        assert session is not None
        matrix_path = self.root / "exports" / "review_sessions" / f"review_session_{int(session['id']):06d}" / "review_matrix.html"
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_path.write_text("<html></html>", encoding="utf-8")
        now = utc_now()
        with connect() as conn:
            conn.execute(
                "UPDATE review_sessions SET status = 'completed', matrix_path = ?, updated_at = ? WHERE id = ?",
                (str(matrix_path), now, session["id"]),
            )

        body = review_session_detail(request=None, session_id=int(session["id"])).body.decode("utf-8")

        self.assertIn("<section id=\"validation-next\" class=\"action-panel\">", body)
        self.assertIn("正式検証へ進む", body)
        self.assertIn("採用epochの検証Runを作成", body)

    def test_selected_lora_job_primary_action_links_back_to_review_matrix(self) -> None:
        from app.main import ensure_selected_lora_profile, job_detail

        job_id = self.create_completed_job_with_outputs()
        ensure_selected_lora_profile(job_id)
        session = ensure_candidate_review_plan(job_id, force=True)
        assert session is not None
        matrix_path = self.root / "exports" / "review_sessions" / f"review_session_{int(session['id']):06d}" / "review_matrix.html"
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_path.write_text("<html></html>", encoding="utf-8")
        now = utc_now()
        with connect() as conn:
            conn.execute(
                "UPDATE review_sessions SET status = 'completed', matrix_path = ?, updated_at = ? WHERE id = ?",
                (str(matrix_path), now, session["id"]),
            )

        body = job_detail(request=None, job_id=job_id).body.decode("utf-8")

        self.assertIn("weight検証Runを作成", body)
        self.assertIn("採用LoRAは選択済みです。次の全体工程はweight検証です。", body)
        self.assertIn(f"/jobs/{job_id}/review-sessions/{int(session['id'])}/matrix", body)
        self.assertIn("Review Matrixを開く", body)

    def test_stale_review_session_imports_partial_generated_images_without_completing(self) -> None:
        job_id = self.create_completed_job_with_outputs()
        session = ensure_candidate_review_plan(job_id, force=True)
        assert session is not None
        session_id = int(session["id"])
        output_dir = self.root / "exports" / "review_sessions" / f"review_session_{session_id:06d}" / "images"
        log_path = output_dir.parent / "review_preparation.log"
        conditions = fetch_all(
            "SELECT * FROM review_session_conditions WHERE review_session_id = ? ORDER BY id LIMIT 2",
            (session_id,),
        )
        for condition in conditions:
            self.make_png(output_dir / f"rs{session_id:06d}_rc{int(condition['id']):06d}_{condition['condition_hash'][:12]}.png")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("INFO done!\n", encoding="utf-8")
        old = "2020-01-01T00:00:00+00:00"
        self.make_tree_old(output_dir.parent)
        with connect() as conn:
            conn.execute(
                """
                UPDATE review_sessions
                SET status = 'running', generation_process_id = 999999,
                    output_dir = ?, log_path = ?, started_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(output_dir), str(log_path), old, old, session_id),
            )

        with mock.patch("app.services.review_sessions.process_exists", return_value=False):
            fixed = reconcile_stale_review_sessions()

        row = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
        imported = fetch_one("SELECT COUNT(*) AS count FROM review_session_images WHERE review_session_id = ?", (session_id,))
        self.assertEqual(fixed, 1)
        self.assertEqual(row["status"], "stopped")
        self.assertEqual(row["generated_image_count"], 2)
        self.assertEqual(row["imported_image_count"], 2)
        self.assertEqual(imported["count"], 2)

    def test_stale_review_session_keeps_recent_activity_running(self) -> None:
        job_id = self.create_completed_job_with_outputs()
        session = ensure_candidate_review_plan(job_id, force=True)
        assert session is not None
        session_id = int(session["id"])
        output_dir = self.root / "exports" / "review_sessions" / f"review_session_{session_id:06d}" / "images"
        log_path = output_dir.parent / "review_preparation.log"
        condition = fetch_one(
            "SELECT * FROM review_session_conditions WHERE review_session_id = ? ORDER BY id LIMIT 1",
            (session_id,),
        )
        self.make_png(output_dir / f"rs{session_id:06d}_rc{int(condition['id']):06d}_{condition['condition_hash'][:12]}.png")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("INFO still switching epochs\n", encoding="utf-8")
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                UPDATE review_sessions
                SET status = 'running', generation_process_id = 999999,
                    output_dir = ?, log_path = ?, started_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(output_dir), str(log_path), now, now, session_id),
            )

        with mock.patch("app.services.review_sessions.process_exists", return_value=False):
            fixed = reconcile_stale_review_sessions()

        row = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
        self.assertEqual(fixed, 0)
        self.assertEqual(row["status"], "running")

    def test_stopped_review_session_refreshes_imported_counts_from_disk(self) -> None:
        job_id = self.create_completed_job_with_outputs()
        session = ensure_candidate_review_plan(job_id, force=True)
        assert session is not None
        session_id = int(session["id"])
        output_dir = self.root / "exports" / "review_sessions" / f"review_session_{session_id:06d}" / "images"
        log_path = output_dir.parent / "review_preparation.log"
        condition = fetch_one(
            "SELECT * FROM review_session_conditions WHERE review_session_id = ? ORDER BY id LIMIT 1",
            (session_id,),
        )
        self.make_png(output_dir / f"rs{session_id:06d}_rc{int(condition['id']):06d}_{condition['condition_hash'][:12]}.png")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("INFO done!\n", encoding="utf-8")
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                UPDATE review_sessions
                SET status = 'stopped', output_dir = ?, log_path = ?,
                    generated_image_count = 0, imported_image_count = 0,
                    started_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(output_dir), str(log_path), now, now, session_id),
            )

        fixed = reconcile_stale_review_sessions()

        row = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
        imported = fetch_one("SELECT COUNT(*) AS count FROM review_session_images WHERE review_session_id = ?", (session_id,))
        self.assertEqual(fixed, 1)
        self.assertEqual(row["status"], "stopped")
        self.assertEqual(row["generated_image_count"], 1)
        self.assertEqual(row["imported_image_count"], 1)
        self.assertEqual(imported["count"], 1)


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
        self.assertIn("構成チェック", detail.body.decode("utf-8"))

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

    def add_review_ready_job(
        self,
        mode: str = "plan_only",
        max_auto_images: int = 18,
        epochs: tuple[int, ...] = (1, 2, 3),
    ) -> int:
        project_id, dataset_id, version_id = self.create_project_fixture()
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                UPDATE lora_projects
                SET post_training_review_mode = ?, max_auto_images = ?, include_neighbor_epochs = 0
                WHERE id = ?
                """,
                (mode, max_auto_images, project_id),
            )
        job_id = self.add_completed_job(dataset_id, version_id, "sdxl_2d_face_adamw8bit_standard", project_id=project_id)
        with connect() as conn:
            for epoch in epochs:
                model_path = self.root / "runs" / f"job_{job_id:06d}" / "models" / f"model-{epoch:06d}.safetensors"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_text("model", encoding="utf-8")
                conn.execute(
                    """
                    INSERT INTO training_outputs(job_id, epoch, file_path, file_type, selected, created_at)
                    VALUES (?, ?, ?, 'model', 0, ?)
                    """,
                    (job_id, epoch, str(model_path), now),
                )
                conn.execute(
                    """
                    INSERT INTO training_epoch_summaries(job_id, epoch, avg_loss, moving_avg_final_loss, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, epoch, 0.2 - epoch * 0.01, 0.2 - epoch * 0.01, now, now),
                )
        return job_id

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

        response = job_delete(job_id, "未実行のため削除", "db_only", "/jobs?view=active")
        deleted = fetch_one("SELECT deleted_at, delete_reason FROM training_jobs WHERE id = ?", (job_id,))
        self.assertTrue(deleted["deleted_at"])
        self.assertEqual(deleted["delete_reason"], "未実行のため削除")
        self.assertEqual(response.headers["location"], "/jobs?view=active")

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
        self.assertEqual(params["train_batch_size"], 1)
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
        self.assertEqual(params["train_batch_size"], 1)
        self.assertEqual(params["max_train_epochs"], 8)

    def test_builtin_training_presets_default_to_batch_size_one(self) -> None:
        import json

        rows = fetch_all("SELECT id, params_json FROM presets")
        self.assertGreater(len(rows), 0)
        for row in rows:
            params = json.loads(row["params_json"] or "{}")
            self.assertEqual(params.get("train_batch_size"), 1, row["id"])

    def test_train_log_tail_decodes_cp932_logs(self) -> None:
        from app.services.training_runner import read_log_tail

        run_dir = self.root / "runs" / "job_000001"
        log_path = run_dir / "logs" / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes("エラー: メモリ不足\nRuntimeError: bad allocation\n".encode("cp932"))

        tail = read_log_tail({"run_dir": str(run_dir)})

        self.assertIn("エラー: メモリ不足", tail)
        self.assertIn("RuntimeError: bad allocation", tail)

    def test_training_operation_timing_uses_tqdm_progress(self) -> None:
        from datetime import datetime, timezone

        from app.services.operation_monitor import operation_timing_summary

        timing = operation_timing_summary(
            started_at="2026-06-22T21:23:52+00:00",
            progress_current=None,
            progress_total=None,
            log_tail="steps:  20%|██        | 1030/5180 [36:02<2:25:10,  2.10s/it, avr_loss=0.1]",
            operation_type="training",
            now=datetime(2026, 6, 22, 22, 12, 28, tzinfo=timezone.utc),
        )

        self.assertEqual(timing["stage_elapsed_seconds"], 2162)
        self.assertEqual(timing["estimated_remaining_seconds"], 8710)
        self.assertEqual(timing["estimated_total_seconds"], 10872)
        self.assertEqual(timing["rate_label"], "2.10s/it")
        self.assertEqual(timing["elapsed_label"], "48m 36s")

    def test_command_builder_skips_zero_sdxl_text_encoder_lr(self) -> None:
        from app.services.command_builder import build_command_argv

        now = utc_now()
        sd_scripts = self.root / "external" / "sd-scripts"
        sd_scripts.mkdir(parents=True, exist_ok=True)
        with connect() as conn:
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
        job = {
            "params_json": json.dumps(
                {
                    "optimizer_type": "AdamW8bit",
                    "text_encoder_lr1": 0,
                    "text_encoder_lr2": 0,
                    "network_train_unet_only": True,
                    "cache_text_encoder_outputs": True,
                    "train_batch_size": 2,
                    "resolution": [1024, 1024],
                }
            ),
            "training_script": "sdxl_train_network.py",
            "base_model_path": "D:/models/base.safetensors",
            "output_dir": str(self.root / "runs" / "job_000001" / "models"),
            "output_name": "test",
            "run_dir": str(self.root / "runs" / "job_000001"),
            "vae_path": None,
        }

        argv = build_command_argv(job, self.root / "dataset.toml", self.root / "sample.txt")

        self.assertNotIn("--text_encoder_lr", argv)

    def test_command_builder_keeps_active_sdxl_text_encoder_lr(self) -> None:
        from app.services.command_builder import build_command_argv

        now = utc_now()
        sd_scripts = self.root / "external" / "sd-scripts"
        sd_scripts.mkdir(parents=True, exist_ok=True)
        with connect() as conn:
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
        job = {
            "params_json": json.dumps({"text_encoder_lr1": 0.000005, "text_encoder_lr2": 0.000005}),
            "training_script": "sdxl_train_network.py",
            "base_model_path": "D:/models/base.safetensors",
            "output_dir": str(self.root / "runs" / "job_000001" / "models"),
            "output_name": "test",
            "run_dir": str(self.root / "runs" / "job_000001"),
            "vae_path": None,
        }

        argv = build_command_argv(job, self.root / "dataset.toml", self.root / "sample.txt")

        index = argv.index("--text_encoder_lr")
        self.assertEqual(argv[index + 1 : index + 3], ["5e-06", "5e-06"])

    def test_command_builder_omits_sample_prompts_when_training_samples_disabled(self) -> None:
        from app.services.command_builder import build_command_argv

        now = utc_now()
        sd_scripts = self.root / "external" / "sd-scripts"
        sd_scripts.mkdir(parents=True, exist_ok=True)
        with connect() as conn:
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
        job = {
            "params_json": json.dumps(
                {
                    "generate_training_samples": False,
                    "sample_every_n_epochs": 1,
                    "sample_at_first": True,
                }
            ),
            "training_script": "sdxl_train_network.py",
            "base_model_path": "D:/models/base.safetensors",
            "output_dir": str(self.root / "runs" / "job_000001" / "models"),
            "output_name": "test",
            "run_dir": str(self.root / "runs" / "job_000001"),
            "vae_path": None,
        }

        argv = build_command_argv(job, self.root / "dataset.toml", self.root / "sample.txt")

        self.assertNotIn("--sample_prompts", argv)
        self.assertNotIn("--sample_every_n_epochs", argv)
        self.assertNotIn("--sample_at_first", argv)

    def test_command_builder_renders_optimizer_bool_args_as_python_literals(self) -> None:
        from app.services.command_builder import build_command_argv

        now = utc_now()
        sd_scripts = self.root / "external" / "sd-scripts"
        sd_scripts.mkdir(parents=True, exist_ok=True)
        with connect() as conn:
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
        job = {
            "params_json": json.dumps(
                {
                    "optimizer_type": "Prodigy",
                    "optimizer_args": {
                        "decouple": True,
                        "use_bias_correction": False,
                        "weight_decay": 0.01,
                    },
                }
            ),
            "training_script": "sdxl_train_network.py",
            "base_model_path": "D:/models/base.safetensors",
            "output_dir": str(self.root / "runs" / "job_000001" / "models"),
            "output_name": "test",
            "run_dir": str(self.root / "runs" / "job_000001"),
            "vae_path": None,
        }

        argv = build_command_argv(job, self.root / "dataset.toml", self.root / "sample.txt")

        optimizer_args_index = argv.index("--optimizer_args")
        optimizer_args = []
        for part in argv[optimizer_args_index + 1 :]:
            if part.startswith("--"):
                break
            optimizer_args.append(part)
        self.assertIn("decouple=True", optimizer_args)
        self.assertIn("use_bias_correction=False", optimizer_args)
        self.assertIn("weight_decay=0.01", optimizer_args)
        self.assertEqual(argv.count("--optimizer_args"), 1)

    def test_phase123_command_builder_handles_optimizer_profile_params(self) -> None:
        from app.services.command_builder import build_command_argv

        sd_scripts = self.root / "external" / "sd-scripts"
        sd_scripts.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        with connect() as conn:
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

        def argv_for_profile(profile_id: str) -> list[str]:
            profile = fetch_one("SELECT command_params_json FROM optimizer_profiles_v2 WHERE id = ?", (profile_id,))
            self.assertIsNotNone(profile)
            params = json.loads(profile["command_params_json"] or "{}")
            job = {
                "params_json": json.dumps(params),
                "training_script": "sdxl_train_network.py",
                "base_model_path": "D:/models/base.safetensors",
                "output_dir": str(self.root / "runs" / f"job_{profile_id}" / "models"),
                "output_name": profile_id,
                "run_dir": str(self.root / "runs" / f"job_{profile_id}"),
                "vae_path": None,
            }
            return build_command_argv(job, self.root / f"{profile_id}.toml", self.root / f"{profile_id}.txt")

        prodigy_argv = argv_for_profile("prodigy_sdxl_soft")
        self.assertIn("Prodigy", prodigy_argv)
        self.assertEqual(prodigy_argv.count("--optimizer_args"), 1)
        prodigy_arg_index = prodigy_argv.index("--optimizer_args")
        self.assertIn("d_coef=0.5", prodigy_argv[prodigy_arg_index + 1 : prodigy_arg_index + 5])

        adafactor_auto_argv = argv_for_profile("adafactor_sdxl_auto")
        self.assertIn("Adafactor", adafactor_auto_argv)
        self.assertNotIn("--learning_rate", adafactor_auto_argv)
        self.assertNotIn("--unet_lr", adafactor_auto_argv)
        self.assertIn("adafactor", adafactor_auto_argv)

        adafactor_fixed_argv = argv_for_profile("adafactor_sdxl_fixed")
        self.assertIn("--learning_rate", adafactor_fixed_argv)
        self.assertIn("0.0001", adafactor_fixed_argv)
        self.assertIn("--max_grad_norm", adafactor_fixed_argv)
        self.assertIn("0.0", adafactor_fixed_argv)

        for profile_id, optimizer_type in {
            "paged_adamw8bit_sdxl_balanced": "PagedAdamW8bit",
            "lion_sdxl_soft": "Lion",
            "dadaptadam_sdxl_auto": "DAdaptAdam",
            "dadaptlion_sdxl_auto": "DAdaptLion",
        }.items():
            argv = argv_for_profile(profile_id)
            self.assertIn(optimizer_type, argv)

    def test_training_failure_diagnosis_explains_access_violation(self) -> None:
        from app.services.training_runner import training_failure_diagnosis

        diagnosis = training_failure_diagnosis(3221225477)

        self.assertIn("0xC0000005", diagnosis)
        self.assertIn("access violation", diagnosis)

    def test_sd_scripts_subprocess_env_strips_app_pythonpath(self) -> None:
        from app.services.training_runner import sd_scripts_subprocess_env

        with mock.patch.dict("os.environ", {"PYTHONPATH": "D:/app/.venv/Lib/site-packages", "PYTHONHOME": "D:/bad", "PYTHONUTF8": "1", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}, clear=False):
            env = sd_scripts_subprocess_env()

        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTHONUTF8", env)
        self.assertNotIn("LANG", env)
        self.assertNotIn("LC_ALL", env)
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

    def test_weight_calibration_run_metadata_and_preflight(self) -> None:
        from app.services.validation_generation import weight_calibration_preflight

        run_id, selected_output_id = self.create_validation_generation_fixture()
        run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
        conditions = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
        preflight = weight_calibration_preflight(run_id)

        self.assertEqual(run["validation_run_kind"], "weight_calibration")
        self.assertEqual(run["source_training_job_id"], run["job_id"])
        self.assertEqual(run["selected_output_id"], selected_output_id)
        self.assertEqual(run["pipeline_status"], "planned")
        self.assertEqual(conditions["count"], 45)
        self.assertFalse(preflight["has_errors"])
        self.assertTrue(any(row["key"] == "reference_set" for row in preflight["checks"]))

    def test_weight_calibration_pipeline_start_sets_status(self) -> None:
        from app.services.validation_generation import start_weight_calibration_pipeline

        run_id, _ = self.create_validation_generation_fixture()
        with mock.patch("app.services.validation_generation.threading.Thread") as thread_mock:
            result = start_weight_calibration_pipeline(run_id, force_warnings=True)

        run = fetch_one("SELECT pipeline_status FROM validation_runs WHERE id = ?", (run_id,))
        self.assertEqual(result["status"], "started")
        self.assertEqual(run["pipeline_status"], "generating_images")
        thread_mock.return_value.start.assert_called_once()

    def test_post_training_review_plan_only_creates_plan_without_generation(self) -> None:
        from app.services.review_sessions import handle_post_training_review_automation

        job_id = self.add_review_ready_job(mode="plan_only")
        with mock.patch("app.services.review_sessions.start_review_preparation") as start_mock:
            result = handle_post_training_review_automation(job_id)

        session = fetch_one("SELECT * FROM review_sessions WHERE job_id = ?", (job_id,))
        self.assertEqual(result["status"], "planned")
        self.assertIsNotNone(session)
        self.assertEqual(session["status"], "planned")
        self.assertEqual(session["expected_image_count"], 18)
        start_mock.assert_not_called()

    def test_post_training_review_manual_mode_does_not_create_plan(self) -> None:
        from app.services.review_sessions import handle_post_training_review_automation

        job_id = self.add_review_ready_job(mode="manual")
        result = handle_post_training_review_automation(job_id)

        session = fetch_one("SELECT * FROM review_sessions WHERE job_id = ?", (job_id,))
        job = fetch_one("SELECT post_training_review_status FROM training_jobs WHERE id = ?", (job_id,))
        self.assertEqual(result["status"], "manual")
        self.assertIsNone(session)
        self.assertEqual(job["post_training_review_status"], "manual")

    def test_post_training_review_plan_only_is_idempotent(self) -> None:
        from app.services.review_sessions import handle_post_training_review_automation

        job_id = self.add_review_ready_job(mode="plan_only")
        first = handle_post_training_review_automation(job_id)
        second = handle_post_training_review_automation(job_id)

        count = fetch_one("SELECT COUNT(*) AS count FROM review_sessions WHERE job_id = ?", (job_id,))
        self.assertEqual(first["session_id"], second["session_id"])
        self.assertEqual(count["count"], 1)

    def test_post_training_review_quick_auto_starts_pipeline(self) -> None:
        from app.services.review_sessions import handle_post_training_review_automation

        job_id = self.add_review_ready_job(mode="quick_auto")
        with mock.patch("app.services.review_sessions.review_gpu_task_busy", return_value=False), mock.patch(
            "app.services.review_sessions.start_review_preparation",
            return_value=4321,
        ) as start_mock:
            result = handle_post_training_review_automation(job_id)

        session = fetch_one("SELECT * FROM review_sessions WHERE job_id = ?", (job_id,))
        job = fetch_one("SELECT post_training_review_status FROM training_jobs WHERE id = ?", (job_id,))
        self.assertEqual(result["status"], "auto_started")
        self.assertEqual(session["expected_image_count"], 18)
        self.assertEqual(job["post_training_review_status"], "auto_started")
        start_mock.assert_called_once_with(session["id"])

    def test_post_training_review_quick_auto_waits_when_gpu_busy(self) -> None:
        from app.services.review_sessions import handle_post_training_review_automation

        job_id = self.add_review_ready_job(mode="quick_auto")
        with mock.patch("app.services.review_sessions.review_gpu_task_busy", return_value=True), mock.patch(
            "app.services.review_sessions.start_review_preparation",
        ) as start_mock:
            result = handle_post_training_review_automation(job_id)

        job = fetch_one("SELECT post_training_review_status FROM training_jobs WHERE id = ?", (job_id,))
        self.assertEqual(result["status"], "planned_waiting")
        self.assertEqual(job["post_training_review_status"], "planned_waiting")
        start_mock.assert_not_called()

    def test_post_training_review_quick_auto_does_not_restart_completed_session(self) -> None:
        from app.services.review_sessions import handle_post_training_review_automation

        job_id = self.add_review_ready_job(mode="quick_auto")
        with mock.patch("app.services.review_sessions.review_gpu_task_busy", return_value=False), mock.patch(
            "app.services.review_sessions.start_review_preparation",
            return_value=4321,
        ) as first_start:
            first = handle_post_training_review_automation(job_id)
        with connect() as conn:
            conn.execute(
                "UPDATE review_sessions SET status = 'completed', automation_status = 'auto_started' WHERE id = ?",
                (first["session_id"],),
            )
        with mock.patch("app.services.review_sessions.start_review_preparation") as second_start:
            second = handle_post_training_review_automation(job_id)

        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["session_id"], first["session_id"])
        first_start.assert_called_once()
        second_start.assert_not_called()

    def test_post_training_review_max_auto_images_blocks_auto_start(self) -> None:
        from app.services.review_sessions import handle_post_training_review_automation

        job_id = self.add_review_ready_job(mode="quick_auto", max_auto_images=12)
        with mock.patch("app.services.review_sessions.start_review_preparation") as start_mock:
            result = handle_post_training_review_automation(job_id)

        job = fetch_one("SELECT post_training_review_status FROM training_jobs WHERE id = ?", (job_id,))
        self.assertEqual(result["status"], "waiting_confirmation")
        self.assertEqual(job["post_training_review_status"], "waiting_confirmation")
        start_mock.assert_not_called()

    def test_post_training_review_standard_auto_respects_max_auto_images(self) -> None:
        from app.services.review_sessions import handle_post_training_review_automation

        job_id = self.add_review_ready_job(mode="standard_auto", max_auto_images=12, epochs=(1, 2, 3))
        with mock.patch("app.services.review_sessions.start_candidate_standard_comparison") as start_mock:
            result = handle_post_training_review_automation(job_id)

        group = fetch_one("SELECT * FROM candidate_comparison_groups WHERE job_id = ?", (job_id,))
        self.assertEqual(result["status"], "waiting_confirmation")
        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(result["comparison_group_id"], group["id"])
        self.assertEqual(group["expected_total_images"], 135)
        self.assertGreater(group["expected_total_images"], 12)
        start_mock.assert_not_called()

    def test_post_training_review_standard_auto_starts_candidate_comparison(self) -> None:
        from app.services.review_sessions import handle_post_training_review_automation

        job_id = self.add_review_ready_job(mode="standard_auto", max_auto_images=150, epochs=(1, 2, 3))
        with mock.patch("app.services.review_sessions.start_candidate_standard_comparison", return_value={"group_id": 123, "status": "started"}) as start_mock:
            result = handle_post_training_review_automation(job_id)

        group = fetch_one("SELECT * FROM candidate_comparison_groups WHERE job_id = ?", (job_id,))
        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(result["status"], "auto_started")
        self.assertEqual(result["comparison_group_id"], group["id"])
        self.assertEqual(result["start_result"]["status"], "started")
        self.assertEqual(group["expected_total_images"], 135)
        start_mock.assert_called_once_with(group["id"])

    def test_machine_assist_no_clear_winner_summary(self) -> None:
        from app.services.review_sessions import review_machine_candidate_summary

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2))
        from app.services.review_sessions import create_candidate_review_plan

        session = create_candidate_review_plan(job_id, include_neighbor_epochs=False, force=True, max_candidate_epochs=2)
        now = utc_now()
        with connect() as conn:
            for epoch, value in ((1, 0.801), (2, 0.795)):
                image_path = self.make_png(self.root / "exports" / "review_sessions" / f"review_session_{session['id']:06d}" / f"e{epoch}.png")
                cur = conn.execute(
                    """
                    INSERT INTO review_session_images(review_session_id, job_id, epoch, image_path, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session["id"], job_id, epoch, str(image_path), now, now),
                )
                image_id = int(cur.lastrowid)
                conn.execute(
                    """
                    INSERT INTO machine_review_scores(
                        source_type, source_id, job_id, embedding_model_id, provider,
                        epoch, reference_similarity_avg, reference_similarity_max,
                        overfit_risk_label, assist_score, assist_label, confidence_label,
                        reason_json, created_at, updated_at
                    )
                    VALUES ('review_session_image', ?, ?, 'mock', 'mock', ?, ?, ?, 'unknown', ?, 'candidate', 'medium', '[]', ?, ?)
                    """,
                    (image_id, job_id, epoch, value, value, value, now, now),
                )

        summary = review_machine_candidate_summary(session["id"])
        self.assertEqual(summary["confidence"], "no_clear_winner")
        self.assertEqual(summary["primary_candidate"], None)
        self.assertEqual(summary["candidate_group"], [1, 2])

    def test_retry_signal_detects_step_shortage(self) -> None:
        from app.services.retry_signal import retry_signal_for_job

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2, 3))
        with connect() as conn:
            conn.execute(
                """
                UPDATE training_jobs
                SET expected_total_steps_at_creation = 1200,
                    target_steps_recommended_at_creation = 5000,
                    step_status_at_creation = 'LOW'
                WHERE id = ?
                """,
                (job_id,),
            )

        summary = retry_signal_for_job(job_id)
        self.assertEqual(summary["retry_signal_label"], "UNDERTRAINED_STEP_SHORTAGE")
        self.assertIn("目標Step", " ".join(summary["reasons"]))

    def test_retry_signal_uses_no_clear_winner_review_summary(self) -> None:
        from app.services.retry_signal import retry_signal_for_review_session
        from app.services.review_sessions import create_candidate_review_plan

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2))
        session = create_candidate_review_plan(job_id, include_neighbor_epochs=False, force=True, max_candidate_epochs=2)
        now = utc_now()
        with connect() as conn:
            conn.execute("UPDATE review_sessions SET status = 'completed' WHERE id = ?", (session["id"],))
            for epoch, value in ((1, 0.801), (2, 0.795)):
                image_path = self.make_png(self.root / "exports" / "review_sessions" / f"retry_signal_{epoch}.png")
                cur = conn.execute(
                    """
                    INSERT INTO review_session_images(review_session_id, job_id, epoch, image_path, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session["id"], job_id, epoch, str(image_path), now, now),
                )
                conn.execute(
                    """
                    INSERT INTO machine_review_scores(
                        source_type, source_id, job_id, embedding_model_id, provider,
                        epoch, reference_similarity_avg, reference_similarity_max,
                        overfit_risk_label, assist_score, assist_label, confidence_label,
                        reason_json, created_at, updated_at
                    )
                    VALUES ('review_session_image', ?, ?, 'mock', 'mock', ?, ?, ?, 'unknown', ?, 'candidate', 'medium', '[]', ?, ?)
                    """,
                    (int(cur.lastrowid), job_id, epoch, value, value, value, now, now),
                )

        summary = retry_signal_for_review_session(session["id"])
        self.assertEqual(summary["retry_signal_label"], "NO_CLEAR_WINNER")
        self.assertEqual(summary["confidence"], "medium")

    def test_retry_signal_detects_weight_strength_from_profile(self) -> None:
        from app.services.retry_signal import retry_signal_for_profile

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2, 3))
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO selected_lora_profiles(
                    job_id, profile_name, selected_model_path,
                    recommended_weight_min, recommended_weight_max,
                    created_at, updated_at
                )
                VALUES (?, 'test profile', 'D:/models/test.safetensors', 0.2, 0.4, ?, ?)
                """,
                (job_id, now, now),
            )
            profile_id = int(cur.lastrowid)

        summary = retry_signal_for_profile(profile_id)
        self.assertEqual(summary["retry_signal_label"], "PARAMETER_TOO_STRONG")
        self.assertIn("0.4", " ".join(summary["reasons"]))

    def test_retry_signal_detects_still_improving_loss(self) -> None:
        from app.services.retry_signal import retry_signal_for_job

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2, 3))
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO training_metric_summaries(job_id, epoch_trend_label, health_label, updated_at)
                VALUES (?, 'STILL_IMPROVING', 'OK', ?)
                """,
                (job_id, now),
            )

        summary = retry_signal_for_job(job_id)
        self.assertEqual(summary["retry_signal_label"], "UNDERTRAINED_STILL_IMPROVING")
        self.assertIn("Loss傾向", " ".join(summary["reasons"]))

    def test_retry_signal_detects_overtrained_loss(self) -> None:
        from app.services.retry_signal import retry_signal_for_job

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2, 3))
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO training_metric_summaries(job_id, epoch_trend_label, health_label, updated_at)
                VALUES (?, 'OVERTRAINING', 'DANGER', ?)
                """,
                (job_id, now),
            )

        summary = retry_signal_for_job(job_id)
        self.assertEqual(summary["retry_signal_label"], "OVERTRAINED")
        self.assertIn("DANGER", " ".join(summary["reasons"]))

    def test_retry_signal_detects_parameter_too_weak_from_profile(self) -> None:
        from app.services.retry_signal import retry_signal_for_profile

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2, 3))
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO selected_lora_profiles(
                    job_id, profile_name, selected_model_path,
                    recommended_weight_min, recommended_weight_max,
                    created_at, updated_at
                )
                VALUES (?, 'weak profile', 'D:/models/test.safetensors', 0.9, 1.0, ?, ?)
                """,
                (job_id, now, now),
            )
            profile_id = int(cur.lastrowid)

        summary = retry_signal_for_profile(profile_id)
        self.assertEqual(summary["retry_signal_label"], "PARAMETER_TOO_WEAK")
        self.assertIn("0.9", " ".join(summary["reasons"]))

    def test_retry_signal_detects_dataset_or_caption_issue(self) -> None:
        from app.services.retry_signal import retry_signal_for_job

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2, 3))
        with connect() as conn:
            conn.execute(
                "UPDATE training_jobs SET trigger_consistency_label_at_creation = 'WARNING' WHERE id = ?",
                (job_id,),
            )

        summary = retry_signal_for_job(job_id)
        self.assertEqual(summary["retry_signal_label"], "DATASET_OR_CAPTION_ISSUE")
        self.assertEqual(summary["confidence"], "high")

    def test_retry_signal_accepts_completed_profile_and_weight_calibration(self) -> None:
        from app.services.retry_signal import retry_signal_for_profile

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2, 3))
        now = utc_now()
        with connect() as conn:
            conn.execute(
                "UPDATE training_epoch_summaries SET moving_avg_final_loss = 0.2, avg_loss = 0.2 WHERE job_id = ?",
                (job_id,),
            )
            cur = conn.execute(
                """
                INSERT INTO selected_lora_profiles(
                    job_id, profile_name, selected_model_path,
                    recommended_weight_min, recommended_weight_max,
                    created_at, updated_at
                )
                VALUES (?, 'accepted profile', 'D:/models/test.safetensors', 0.6, 0.8, ?, ?)
                """,
                (job_id, now, now),
            )
            profile_id = int(cur.lastrowid)
            conn.execute(
                """
                INSERT INTO validation_runs(
                    job_id, selected_lora_profile_id, validation_preset_id, name,
                    status, recommended_weight_min, recommended_weight_max,
                    expected_image_count, actual_image_count, created_at, updated_at
                )
                VALUES (?, ?, 'standard_validation_v1', 'accepted validation',
                        'completed', 0.6, 0.8, 45, 45, ?, ?)
                """,
                (job_id, profile_id, now, now),
            )

        summary = retry_signal_for_profile(profile_id)
        self.assertEqual(summary["retry_signal_label"], "ACCEPTABLE")
        self.assertEqual(summary["confidence"], "high")
        self.assertIn("採用可能", " ".join(summary["recommended_next_actions"]))

    def test_retry_signal_planned_review_does_not_force_no_clear_winner(self) -> None:
        from app.services.retry_signal import retry_signal_for_review_session
        from app.services.review_sessions import create_candidate_review_plan

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2))
        session = create_candidate_review_plan(job_id, include_neighbor_epochs=False, force=True, max_candidate_epochs=2)

        summary = retry_signal_for_review_session(session["id"])
        self.assertNotEqual(summary["retry_signal_label"], "NO_CLEAR_WINNER")
        self.assertFalse(summary["evidence"]["review_signal"]["evidence"]["review_ready"])

    def test_retry_signal_human_rating_overrides_no_clear_machine_assist(self) -> None:
        from app.services.retry_signal import retry_signal_for_review_session
        from app.services.review_sessions import create_candidate_review_plan

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2))
        session = create_candidate_review_plan(job_id, include_neighbor_epochs=False, force=True, max_candidate_epochs=2)
        now = utc_now()
        with connect() as conn:
            conn.execute("UPDATE review_sessions SET status = 'completed' WHERE id = ?", (session["id"],))
            for epoch, value, rating in ((1, 0.801, 5), (2, 0.795, 3)):
                image_path = self.make_png(self.root / "exports" / "review_sessions" / f"human_retry_signal_{epoch}.png")
                cur = conn.execute(
                    """
                    INSERT INTO review_session_images(review_session_id, job_id, epoch, image_path, rating_overall, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session["id"], job_id, epoch, str(image_path), rating, now, now),
                )
                conn.execute(
                    """
                    INSERT INTO machine_review_scores(
                        source_type, source_id, job_id, embedding_model_id, provider,
                        epoch, reference_similarity_avg, reference_similarity_max,
                        overfit_risk_label, assist_score, assist_label, confidence_label,
                        reason_json, created_at, updated_at
                    )
                    VALUES ('review_session_image', ?, ?, 'mock', 'mock', ?, ?, ?, 'unknown', ?, 'candidate', 'medium', '[]', ?, ?)
                    """,
                    (int(cur.lastrowid), job_id, epoch, value, value, value, now, now),
                )

        summary = retry_signal_for_review_session(session["id"])
        self.assertNotEqual(summary["retry_signal_label"], "NO_CLEAR_WINNER")
        self.assertEqual(summary["evidence"]["review_signal"]["evidence"]["human_top_epoch"], 1)

    def test_performance_profile_records_stage_timing_and_process_count(self) -> None:
        from app.services.performance_profile import mark_command_end, mark_command_start, mark_stage, performance_summary, reset_pipeline_timing
        from app.services.review_sessions import create_candidate_review_plan

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2))
        session = create_candidate_review_plan(job_id, include_neighbor_epochs=False, force=True, max_candidate_epochs=2)
        output_dir = self.root / "exports" / "review_sessions" / f"review_session_{session['id']:06d}" / "images"
        image_path = self.make_png(output_dir / "generated.png")
        commands = [{"name": "epoch_1", "condition_count": 1, "argv": ["python", "gen_img.py", "--network_weights", "D:/model/lora.safetensors"]}]

        with connect() as conn:
            conn.execute("UPDATE review_sessions SET output_dir = ? WHERE id = ?", (str(output_dir), session["id"]))
        reset_pipeline_timing("review_sessions", session["id"], commands=commands, output_dir=str(output_dir))
        mark_stage("review_sessions", session["id"], "generation_start")
        mark_command_start("review_sessions", session["id"], 0)
        mark_command_end("review_sessions", session["id"], 0, output_dir=str(output_dir), return_code=0)
        mark_stage("review_sessions", session["id"], "generation_end")

        row = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session["id"],))
        self.assertTrue(row["stage_timing_json"])
        summary = performance_summary(row, output_dir=str(output_dir))
        self.assertEqual(summary["generation_process_count"], 1)
        self.assertEqual(summary["commands"][0]["output_count"], 1)
        self.assertEqual(Path(image_path).suffix, ".png")

    def test_performance_profile_handles_missing_timing(self) -> None:
        from app.services.performance_profile import performance_summary
        from app.services.review_sessions import create_candidate_review_plan

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2))
        session = create_candidate_review_plan(job_id, include_neighbor_epochs=False, force=True, max_candidate_epochs=2)
        row = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session["id"],))

        summary = performance_summary(row, output_dir="")
        self.assertEqual(summary["generation_process_count"], 0)
        self.assertEqual(summary["commands"], [])

    def test_performance_profile_can_save_validation_run_timing(self) -> None:
        from app.services.performance_profile import mark_stage, reset_pipeline_timing

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2))
        now = utc_now()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO validation_runs(
                    job_id, validation_preset_id, name, expected_image_count,
                    status, created_at, updated_at
                )
                VALUES (?, 'standard_validation_v1', 'timing test', 1, 'planned', ?, ?)
                """,
                (job_id, now, now),
            )
            run_id = int(cur.lastrowid)

        reset_pipeline_timing("validation_runs", run_id, commands=[{"name": "lora", "condition_count": 1, "argv": []}], output_dir="")
        mark_stage("validation_runs", run_id, "pipeline_end")

        row = fetch_one("SELECT stage_timing_json FROM validation_runs WHERE id = ?", (run_id,))
        self.assertIn("pipeline_end", row["stage_timing_json"])

    def test_expanded_neighbor_review_session_uses_neighbor_epochs(self) -> None:
        from app.services.review_sessions import create_candidate_review_plan, create_neighbor_review_session

        job_id = self.add_review_ready_job(mode="plan_only", epochs=(1, 2, 3))
        parent = create_candidate_review_plan(job_id, include_neighbor_epochs=False, force=True, max_candidate_epochs=2)
        session = create_neighbor_review_session(job_id, center_epoch=2, radius=1, parent_review_session_id=parent["id"])

        self.assertIsNotNone(session)
        self.assertEqual(session["status"], "planned")
        self.assertEqual(session["review_plan_kind"], "expanded_neighbor_review")
        self.assertEqual(session["parent_review_session_id"], parent["id"])
        self.assertEqual(json.loads(session["candidate_epochs_json"]), [1, 2, 3])
        self.assertEqual(json.loads(session["weights_json"]), [0.6, 0.8])
        self.assertEqual(session["seed"], 111111)
        self.assertEqual(session["expected_image_count"], 18)

    def test_step_estimator_basic_formula_and_warning(self) -> None:
        from app.services.step_estimator import estimate_steps

        estimate = estimate_steps(
            image_count=39,
            params={"repeats": 10, "max_train_epochs": 10, "train_batch_size": 2},
            target={"target_steps_min": 2500, "target_steps_recommended": 5000, "target_steps_max": 8000, "target_checkpoint_count": 6},
        )

        self.assertEqual(estimate["steps_per_epoch"], 195)
        self.assertEqual(estimate["total_steps"], 1950)
        self.assertEqual(estimate["status"], "LOW")
        self.assertIn("少なめ", estimate["message"])

    def test_step_target_catalog_priority(self) -> None:
        from app.services.step_estimator import target_config_from_catalog

        target = target_config_from_catalog(
            training_recipe={
                "name": "Recipe Target",
                "target_steps_min": 100,
                "target_steps_recommended": 200,
                "target_steps_max": 300,
                "target_checkpoint_count": 3,
                "note": "recipe",
            },
            optimizer_profile={
                "target_steps_min": 1000,
                "target_steps_recommended": 2000,
                "target_steps_max": 3000,
                "target_checkpoint_count": 6,
            },
            optimizer_definition={
                "name": "AdamW8bit",
                "lr_meaning": "normal_lr",
                "category": "stable",
                "target_steps_min": 2500,
                "target_steps_recommended": 5000,
                "target_steps_max": 8000,
            },
        )

        self.assertEqual(target["target_steps_recommended"], 200)
        self.assertEqual(target["target_checkpoint_count"], 3)
        self.assertEqual(target["step_target_source"], "recipe")
        self.assertEqual(target["recipe_target_steps_recommended"], 200)
        self.assertEqual(target["optimizer_target_steps_recommended"], 2000)
        self.assertEqual(target["optimizer_lr_meaning"], "normal_lr")

    def test_step_target_catalog_falls_back_to_optimizer_definition(self) -> None:
        from app.services.step_estimator import target_config_from_catalog

        target = target_config_from_catalog(
            optimizer_definition={
                "name": "Prodigy",
                "lr_meaning": "auto_lr_multiplier",
                "category": "advanced",
                "target_steps_min": 1800,
                "target_steps_recommended": 3500,
                "target_steps_max": 6500,
            },
        )

        self.assertEqual(target["target_steps_recommended"], 3500)
        self.assertEqual(target["step_target_source"], "optimizer_definition")
        self.assertEqual(target["optimizer_target_steps_recommended"], 3500)
        self.assertEqual(target["optimizer_category"], "advanced")

    def test_step_estimator_multiple_subsets_and_gradient_accumulation(self) -> None:
        from app.services.step_estimator import estimate_steps

        estimate = estimate_steps(
            image_count=0,
            params={"repeats": 1, "max_train_epochs": 5, "train_batch_size": 2, "gradient_accumulation_steps": 2},
            subsets=[{"image_count": 10, "num_repeats": 3}, {"image_count": 5, "num_repeats": 4}],
        )

        self.assertEqual(estimate["weighted_image_count"], 50)
        self.assertEqual(estimate["effective_batch_size"], 4)
        self.assertEqual(estimate["steps_per_epoch"], 13)
        self.assertEqual(estimate["total_steps"], 65)

    def test_target_step_assistant_suggests_candidates_and_intervals(self) -> None:
        from app.services.step_estimator import calculate_required_repeats, estimate_steps, suggest_target_steps

        params = {"repeats": 10, "max_train_epochs": 10, "train_batch_size": 2}
        auto = calculate_required_repeats(image_count=39, params=params, target_steps=5000)
        suggestions = suggest_target_steps(
            image_count=39,
            params=params,
            target_steps=5000,
            target={"target_checkpoint_count": 6},
        )
        high_epoch = estimate_steps(image_count=39, params={**params, "max_train_epochs": 26}, target={"target_checkpoint_count": 6})

        self.assertEqual(auto["required_repeats"], 26)
        self.assertEqual(auto["expected_total_steps"], 5070)
        self.assertTrue(any(row["expected_total_steps"] >= 4990 for row in suggestions))
        self.assertTrue(any(row["save_every_n_epochs_proposal"] > 1 for row in suggestions))
        self.assertEqual(high_epoch["save_every_n_epochs_proposal"], 5)
        self.assertTrue(any("毎epoch保存" in warning for warning in high_epoch["warnings"]))

    def test_target_step_assistant_uses_actual_steps_per_epoch_rounding(self) -> None:
        from app.services.step_estimator import calculate_required_repeats

        auto = calculate_required_repeats(
            image_count=37,
            params={"max_train_epochs": 10, "train_batch_size": 2, "gradient_accumulation_steps": 1},
            target_steps=5000,
        )

        self.assertEqual(auto["required_repeats"], 27)
        self.assertEqual(auto["expected_total_steps"], 5000)

    def test_target_step_assistant_handles_multiple_subsets_with_fixed_repeats(self) -> None:
        from app.services.step_estimator import calculate_required_repeats, suggest_target_steps

        subsets = [{"image_count": 10, "num_repeats": 3}, {"image_count": 5, "num_repeats": 4}]
        params = {"repeats": 1, "max_train_epochs": 5, "train_batch_size": 4, "gradient_accumulation_steps": 1}
        auto = calculate_required_repeats(image_count=0, params=params, target_steps=100, subsets=subsets)
        suggestions = suggest_target_steps(image_count=0, params=params, target_steps=100, subsets=subsets)

        self.assertEqual(auto["required_repeats"], 6)
        self.assertEqual(auto["steps_per_epoch"], 23)
        self.assertEqual(auto["expected_total_steps"], 115)
        self.assertTrue(any(row["repeats"] == 6 and row["expected_total_steps"] == 115 for row in suggestions))

    def test_job_creation_saves_step_estimate_snapshot(self) -> None:
        from app.db import create_job

        project_id, dataset_id, version_id = self.create_project_fixture()
        job_id = create_job(
            {
                "project_id": project_id,
                "name": "step snapshot",
                "dataset_id": dataset_id,
                "dataset_version_id": version_id,
                "preset_id": "sdxl_2d_face_adamw8bit_standard",
                "base_model_path": "D:/models/base.safetensors",
                "output_name": "step_snapshot",
                "memo": "",
            }
        )
        job = fetch_one(
            """
            SELECT expected_total_steps_at_creation, steps_per_epoch_at_creation,
                   target_steps_recommended_at_creation, step_status_at_creation,
                   repeats_auto_calculated, target_steps_source, step_estimate_snapshot_json
            FROM training_jobs WHERE id = ?
            """,
            (job_id,),
        )

        self.assertEqual(job["steps_per_epoch_at_creation"], 500)
        self.assertEqual(job["expected_total_steps_at_creation"], 5000)
        self.assertEqual(job["target_steps_recommended_at_creation"], 5000)
        self.assertEqual(job["step_status_at_creation"], "OK")
        self.assertEqual(job["repeats_auto_calculated"], 0)
        self.assertEqual(job["target_steps_source"], "recipe")
        self.assertIn("total_steps", job["step_estimate_snapshot_json"])
        self.assertIn("recipe_target_steps_recommended", job["step_estimate_snapshot_json"])

    def test_phase121_optimizer_recipe_v2_seed(self) -> None:
        optimizers = {row["id"] for row in fetch_all("SELECT id FROM optimizer_definitions_v2")}
        for optimizer_id in {"AdamW8bit", "PagedAdamW8bit", "Adafactor", "Lion", "DAdaptAdam", "DAdaptLion", "Prodigy"}:
            self.assertIn(optimizer_id, optimizers)

        recipes = {row["id"] for row in fetch_all("SELECT id FROM training_recipes_v2")}
        for recipe_id in {
            "sdxl_character_face_adamw8bit_smoke",
            "sdxl_character_face_adamw8bit_pilot_3epoch",
            "sdxl_character_face_adamw8bit_soft",
            "sdxl_character_face_adamw8bit_balanced",
            "sdxl_character_face_adamw8bit_strong",
            "sdxl_character_face_adamw8bit_standard_6epoch",
            "sdxl_character_face_adamw8bit_standard_10epoch",
            "sdxl_character_face_adamw8bit_generalize",
            "sdxl_character_face_paged_adamw8bit_balanced",
            "sdxl_character_face_lion_soft_experimental",
            "sdxl_character_face_lion_balanced_experimental",
            "sdxl_character_face_adafactor_auto_advanced",
            "sdxl_character_face_adafactor_fixed_advanced",
            "sdxl_character_face_dadapt_adam_auto_advanced",
            "sdxl_character_face_dadapt_lion_auto_experimental",
            "sdxl_character_face_prodigy_soft_advanced",
            "sdxl_style_adamw8bit_soft",
            "sdxl_style_adamw8bit_balanced",
            "sdxl_style_prodigy_soft_advanced",
            "sdxl_style_adafactor_auto_advanced",
            "sdxl_style_lion_experimental",
            "sdxl_costume_adamw8bit_balanced",
            "sdxl_costume_adamw8bit_strong",
            "sdxl_costume_prodigy_soft_advanced",
            "sd15_character_face_adamw8bit_balanced",
            "sd15_character_face_adamw8bit_strong",
            "sd15_style_adamw8bit_balanced",
        }:
            self.assertIn(recipe_id, recipes)

        profiles = {row["id"] for row in fetch_all("SELECT id FROM optimizer_profiles_v2")}
        for profile_id in {
            "adamw8bit_sdxl_balanced",
            "paged_adamw8bit_sdxl_balanced",
            "prodigy_sdxl_soft",
            "adafactor_sdxl_auto",
            "adafactor_sdxl_fixed",
            "lion_sdxl_soft",
            "lion_sdxl_balanced_experimental",
            "dadaptadam_sdxl_auto",
            "dadaptlion_sdxl_auto",
        }:
            self.assertIn(profile_id, profiles)

        prodigy = fetch_one("SELECT * FROM optimizer_definitions_v2 WHERE id = 'Prodigy'")
        self.assertEqual(prodigy["sd_scripts_optimizer_type"], "Prodigy")
        self.assertIn("prodigyopt", prodigy["required_dependencies_json"])
        self.assertIn("倍率", prodigy["lr_semantics_help"])

        adafactor = fetch_one("SELECT * FROM optimizer_definitions_v2 WHERE id = 'Adafactor'")
        self.assertIn("AdaFactor", adafactor["aliases_json"])

        adafactor_auto = fetch_one("SELECT * FROM optimizer_profiles_v2 WHERE id = 'adafactor_sdxl_auto'")
        auto_command = json.loads(adafactor_auto["command_params_json"])
        auto_smoke = json.loads(adafactor_auto["smoke_params_json"])
        self.assertNotIn("learning_rate", auto_command)
        self.assertNotIn("unet_lr", auto_command)
        self.assertEqual(auto_smoke["max_train_steps"], 2)

        adafactor_fixed = fetch_one("SELECT * FROM optimizer_profiles_v2 WHERE id = 'adafactor_sdxl_fixed'")
        fixed_command = json.loads(adafactor_fixed["command_params_json"])
        self.assertEqual(fixed_command["learning_rate"], 0.0001)
        self.assertEqual(fixed_command["max_grad_norm"], 0.0)

    def test_phase1221_recipe_seed_supports_purpose_and_optimizer_entry_filters(self) -> None:
        character_face_rows = fetch_all(
            """
            SELECT DISTINCT optimizer_definition_id
            FROM training_recipes_v2
            WHERE model_family = 'SDXL' AND training_purpose_id = 'character_face' AND is_active = 1
            """
        )
        self.assertGreaterEqual(
            len({row["optimizer_definition_id"] for row in character_face_rows}),
            5,
        )

        prodigy_rows = fetch_all(
            """
            SELECT optimizer_definition_id, training_purpose_id
            FROM training_recipes_v2
            WHERE model_family = 'SDXL' AND optimizer_definition_id = 'Prodigy' AND is_active = 1
            """
        )
        self.assertTrue(prodigy_rows)
        self.assertTrue(all(row["optimizer_definition_id"] == "Prodigy" for row in prodigy_rows))
        self.assertIn("style", {row["training_purpose_id"] for row in prodigy_rows})
        self.assertIn("costume", {row["training_purpose_id"] for row in prodigy_rows})

        lion_rows = fetch_all(
            """
            SELECT optimizer_definition_id, recipe_type
            FROM training_recipes_v2
            WHERE model_family = 'SDXL' AND optimizer_definition_id = 'Lion' AND is_active = 1
            """
        )
        self.assertTrue(lion_rows)
        self.assertTrue(all(row["optimizer_definition_id"] == "Lion" for row in lion_rows))
        self.assertTrue(all(row["recipe_type"] == "experimental" for row in lion_rows))

        style_rows = fetch_all(
            """
            SELECT id
            FROM training_recipes_v2
            WHERE model_family = 'SDXL' AND training_purpose_id = 'style' AND is_active = 1
            """
        )
        self.assertGreaterEqual(len(style_rows), 5)

    def test_phase1221_recipe_display_labels_are_seeded(self) -> None:
        recipe = fetch_one(
            """
            SELECT display_name, short_label, full_label, card_subtitle,
                   direct_select_label, difficulty_label, recommended_badge
            FROM training_recipes_v2
            WHERE id = 'sdxl_character_face_adamw8bit_balanced'
            """
        )
        self.assertEqual(recipe["short_label"], "顔キャラ・標準")
        self.assertIn("[SDXL]", recipe["full_label"])
        self.assertIn("[SDXL]", recipe["direct_select_label"])
        self.assertIn("AdamW8bit", recipe["card_subtitle"])
        self.assertEqual(recipe["difficulty_label"], "stable")
        self.assertEqual(recipe["recommended_badge"], "おすすめ")

        prodigy = fetch_one(
            "SELECT short_label, card_subtitle, difficulty_label FROM training_recipes_v2 WHERE id = 'sdxl_character_face_prodigy_soft_advanced'"
        )
        self.assertEqual(prodigy["short_label"], "顔キャラ・Prodigy弱め")
        self.assertIn("Auto-LR", prodigy["card_subtitle"])
        self.assertEqual(prodigy["difficulty_label"], "advanced")

    def test_phase123_optimizer_profile_compatibility_warnings(self) -> None:
        from app.services.recipe_optimizer_catalog import compatibility_check

        prodigy = compatibility_check({"optimizer_type": "Prodigy", "lr_scheduler": "cosine", "learning_rate": 0.5})
        self.assertTrue(any("lr_scheduler=constant" in message for message in prodigy["warnings"]))
        self.assertTrue(any("1.0" in message for message in prodigy["warnings"]))

        adafactor_auto = compatibility_check(
            {
                "optimizer_type": "Adafactor",
                "lr_scheduler": "constant",
                "learning_rate": 0.0001,
                "optimizer_args": {"relative_step": True},
            }
        )
        self.assertTrue(any("learning_rate" in message for message in adafactor_auto["warnings"]))
        self.assertTrue(any("lr_scheduler=adafactor" in message for message in adafactor_auto["warnings"]))

        adafactor_fixed = compatibility_check(
            {
                "optimizer_type": "Adafactor",
                "lr_scheduler": "constant",
                "learning_rate": 0.0001,
                "optimizer_args": {"relative_step": False},
                "max_grad_norm": 1.0,
            }
        )
        self.assertTrue(any("constant_with_warmup" in message for message in adafactor_fixed["warnings"]))
        self.assertTrue(any("max_grad_norm=0.0" in message for message in adafactor_fixed["warnings"]))

        lion = compatibility_check({"optimizer_type": "Lion", "lr_scheduler": "constant", "learning_rate": 0.0002})
        self.assertTrue(any("Experimental" in message for message in lion["warnings"]))
        self.assertTrue(any("0.0001" in message for message in lion["warnings"]))

        dadapt = compatibility_check({"optimizer_type": "DAdaptAdam", "lr_scheduler": "constant", "learning_rate": 0.5})
        self.assertTrue(any("1.0" in message for message in dadapt["warnings"]))

    def test_phase123_optimizer_profile_validation_result_updates_profile(self) -> None:
        from app.services.optimizer_profile_validation import record_profile_test_result

        prepare_result_id = record_profile_test_result(
            "adamw8bit_sdxl_balanced",
            recipe_id="sdxl_character_face_adamw8bit_balanced",
            test_type="prepare",
            status="ok",
            test_job_id=123,
            command_path="D:/tmp/command_argv.json",
            log_path="D:/tmp/train.log",
        )
        profile = fetch_one("SELECT validation_status, last_test_result_id FROM optimizer_profiles_v2 WHERE id = ?", ("adamw8bit_sdxl_balanced",))
        self.assertEqual(profile["validation_status"], "prepare_ok")
        self.assertEqual(profile["last_test_result_id"], prepare_result_id)

        smoke_result_id = record_profile_test_result(
            "adamw8bit_sdxl_balanced",
            recipe_id="sdxl_character_face_adamw8bit_balanced",
            test_type="smoke",
            status="ok",
            test_job_id=124,
            return_code=0,
            elapsed_seconds=2,
        )
        profile = fetch_one("SELECT validation_status, last_test_result_id FROM optimizer_profiles_v2 WHERE id = ?", ("adamw8bit_sdxl_balanced",))
        definition = fetch_one("SELECT validation_status, last_test_result_id, validated_optimizer_type FROM optimizer_definitions_v2 WHERE id = ?", ("AdamW8bit",))
        result = fetch_one("SELECT * FROM optimizer_profile_test_results WHERE id = ?", (smoke_result_id,))
        self.assertEqual(profile["validation_status"], "smoke_ok")
        self.assertEqual(profile["last_test_result_id"], smoke_result_id)
        self.assertEqual(definition["validation_status"], "smoke_ok")
        self.assertEqual(definition["last_test_result_id"], smoke_result_id)
        self.assertEqual(definition["validated_optimizer_type"], "AdamW8bit")
        self.assertEqual(result["optimizer_definition_id"], "AdamW8bit")
        self.assertEqual(result["test_type"], "smoke")
        self.assertEqual(result["status"], "ok")

        record_profile_test_result(
            "adamw8bit_sdxl_balanced",
            recipe_id="sdxl_character_face_adamw8bit_balanced",
            test_type="mini_pilot",
            status="skipped",
            error_message="manual skip",
        )
        profile = fetch_one("SELECT validation_status FROM optimizer_profiles_v2 WHERE id = ?", ("adamw8bit_sdxl_balanced",))
        self.assertEqual(profile["validation_status"], "smoke_ok")

    def test_phase123_recipe_cards_include_profile_validation_badge(self) -> None:
        from app.main import job_new, training_recipes_library
        from app.services.optimizer_profile_validation import record_profile_test_result

        record_profile_test_result(
            "adamw8bit_sdxl_balanced",
            recipe_id="sdxl_character_face_adamw8bit_balanced",
            test_type="smoke",
            status="ok",
            return_code=0,
        )

        body = job_new(request=None, project_id="", mode="purpose").body.decode("utf-8")
        recipes_body = training_recipes_library(request=None, model_family="", purpose="", optimizer="", optimizer_category="", network_type="", source="", recipe_type="").body.decode("utf-8")
        js_source = Path("app/static/js/app.js").read_text(encoding="utf-8")
        self.assertIn('"optimizer_profile_validation_status": "smoke_ok"', body)
        self.assertIn("Smoke OK", js_source)
        self.assertIn("smoke_ok", recipes_body)

    def test_phase123x_optimizer_master_check_run_and_prepare_are_saved(self) -> None:
        from app.services.optimizer_master_checks import create_master_check_run, master_check_run_detail, run_prepare_checks

        project_id, dataset_id, _version_id = self.create_project_fixture()
        sd_scripts = self.root / "external" / "sd-scripts"
        sd_scripts.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        with connect() as conn:
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

        run_id = create_master_check_run("selected_profiles", ["adamw8bit_sdxl_balanced"])
        detail = master_check_run_detail(run_id)
        self.assertEqual(detail["run"]["total_count"], 1)
        self.assertEqual(len(detail["items"]), 1)
        self.assertEqual(detail["items"][0]["optimizer_profile_id"], "adamw8bit_sdxl_balanced")

        result = run_prepare_checks(run_id, dataset_id=dataset_id, base_model_path="D:/models/base.safetensors")
        self.assertTrue(result["ok"])
        item = fetch_one("SELECT * FROM optimizer_master_check_items WHERE check_run_id = ?", (run_id,))
        run = fetch_one("SELECT * FROM optimizer_master_check_runs WHERE id = ?", (run_id,))
        self.assertEqual(item["status"], "prepare_ok")
        self.assertIsNotNone(item["prepare_job_id"])
        self.assertEqual(run["prepare_ok_count"], 1)
        self.assertTrue(run["report_path"])

    def test_phase123x_optimizer_master_artifact_and_image_checks(self) -> None:
        from app.services.optimizer_master_checks import check_image_smoke, check_lora_artifact

        try:
            import numpy as np
            from safetensors.numpy import save_file
        except Exception as exc:
            self.skipTest(f"safetensors/numpy unavailable: {exc}")

        valid_model = self.root / "model.safetensors"
        save_file({"lora_unet_down.weight": np.array([1.0, 2.0], dtype=np.float32)}, str(valid_model))
        valid_check = check_lora_artifact(str(valid_model))
        self.assertEqual(valid_check["status"], "ok")
        self.assertGreater(valid_check["file_size"], 0)
        self.assertTrue(valid_check["sha256"])

        empty_model = self.root / "empty.safetensors"
        empty_model.write_bytes(b"")
        failed_check = check_lora_artifact(str(empty_model))
        self.assertEqual(failed_check["status"], "failed")

        def make_varied(path: Path, offset: int) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (16, 16))
            for x in range(16):
                for y in range(16):
                    img.putpixel((x, y), ((x * 13 + offset) % 255, (y * 17 + offset) % 255, ((x + y) * 11 + offset) % 255))
            img.save(path)

        left = self.root / "left.png"
        right = self.root / "right.png"
        make_varied(left, 0)
        make_varied(right, 20)
        image_check = check_image_smoke(str(left), str(right))
        self.assertEqual(image_check["status"], "image_smoke_ok")
        self.assertIsNotNone(image_check["difference_score"])

    def test_phase123x_optimizer_master_failure_classification(self) -> None:
        from app.services.optimizer_master_checks import classify_failure, suggested_action

        self.assertEqual(classify_failure("ModuleNotFoundError: No module named 'prodigyopt'"), "missing_dependency")
        self.assertEqual(classify_failure("unexpected keyword argument decouple"), "master_parameter")
        self.assertEqual(classify_failure("optimizer_type not found: DAdaptLion"), "sd_scripts_unsupported")
        self.assertEqual(classify_failure("database is locked"), "environment")
        self.assertIn("依存", suggested_action("missing_dependency"))

    def test_phase123x_optimizer_master_check_page_renders(self) -> None:
        from app.main import optimizer_master_checks
        from app.services.optimizer_master_checks import create_master_check_run

        run_id = create_master_check_run("selected_profiles", ["adamw8bit_sdxl_balanced"])
        body = optimizer_master_checks(request=None, run_id=run_id).body.decode("utf-8")
        self.assertIn("Optimizer Master Validation / Smoke Matrix", body)
        self.assertIn("adamw8bit_sdxl_balanced", body)
        self.assertIn("Run Prepare Check", body)

    def test_phase1231_optimizer_mini_pilot_run_and_report_are_saved(self) -> None:
        from app.main import optimizer_mini_pilots
        from app.services.optimizer_mini_pilots import create_mini_pilot_run, mini_pilot_run_detail, write_mini_pilot_report

        _project_id, dataset_id, _version_id = self.create_project_fixture()
        run_id = create_mini_pilot_run(
            "selected_profiles",
            ["adamw8bit_sdxl_balanced"],
            dataset_id=dataset_id,
            base_model_path="D:/models/base.safetensors",
            steps=300,
        )
        detail = mini_pilot_run_detail(run_id)
        self.assertEqual(detail["run"]["total_count"], 1)
        self.assertEqual(detail["items"][0]["optimizer_profile_id"], "adamw8bit_sdxl_balanced")
        paths = write_mini_pilot_report(run_id)
        self.assertTrue(Path(paths["report_path"]).exists())
        self.assertTrue(Path(paths["json_report_path"]).exists())

        body = optimizer_mini_pilots(request=None, run_id=run_id).body.decode("utf-8")
        self.assertIn("Optimizer Practical Mini Pilot", body)
        self.assertIn("adamw8bit_sdxl_balanced", body)
        self.assertIn("Run Mini Pilot", body)

    def test_phase1231_mini_pilot_result_updates_profile_badge(self) -> None:
        from app.services.optimizer_profile_validation import profile_validation_badge, record_profile_test_result

        result_id = record_profile_test_result(
            "adamw8bit_sdxl_balanced",
            recipe_id="sdxl_character_face_adamw8bit_balanced",
            test_type="mini_pilot",
            status="ok",
            test_job_id=321,
            return_code=0,
            elapsed_seconds=12,
        )
        profile = fetch_one(
            """
            SELECT validation_status, mini_pilot_status, last_mini_pilot_result_id
            FROM optimizer_profiles_v2 WHERE id = ?
            """,
            ("adamw8bit_sdxl_balanced",),
        )
        self.assertEqual(profile["validation_status"], "mini_pilot_ok")
        self.assertEqual(profile["mini_pilot_status"], "mini_pilot_ok")
        self.assertEqual(profile["last_mini_pilot_result_id"], result_id)
        badge = profile_validation_badge(profile)
        self.assertEqual(badge["text"], "Mini Pilot OK")

    def test_phase121_recipe_v2_job_creation_saves_snapshots(self) -> None:
        from app.db import create_job

        project_id, dataset_id, version_id = self.create_project_fixture()
        job_id = create_job(
            {
                "project_id": project_id,
                "name": "recipe v2 job",
                "dataset_id": dataset_id,
                "dataset_version_id": version_id,
                "preset_id": "sdxl_2d_face_adamw8bit_standard",
                "recipe_v2_id": "sdxl_character_face_adamw8bit_standard_10epoch",
                "base_model_path": "D:/models/base.safetensors",
                "output_name": "recipe_v2_job",
                "params": {"repeats": 10, "max_train_epochs": 10, "train_batch_size": 2},
                "user_overrides": {"train_batch_size": 2},
            }
        )
        job = fetch_one(
            """
            SELECT recipe_v2_id, optimizer_definition_id, optimizer_profile_id,
                   network_type_id, training_purpose_id, recipe_snapshot_json,
                   params_snapshot_json, user_overrides_json,
                   expected_total_steps_at_creation, target_steps_source
            FROM training_jobs WHERE id = ?
            """,
            (job_id,),
        )

        self.assertEqual(job["recipe_v2_id"], "sdxl_character_face_adamw8bit_standard_10epoch")
        self.assertEqual(job["optimizer_definition_id"], "AdamW8bit")
        self.assertEqual(job["network_type_id"], "standard_lora")
        self.assertEqual(job["training_purpose_id"], "character_face")
        self.assertIn("optimizer_definition", job["recipe_snapshot_json"])
        self.assertIn('"train_batch_size": 2', job["params_snapshot_json"])
        self.assertIn('"train_batch_size": 2', job["user_overrides_json"])
        self.assertEqual(job["expected_total_steps_at_creation"], 2500)
        self.assertEqual(job["target_steps_source"], "recipe")

    def test_phase121_compatibility_check_flags_te_cache_conflict(self) -> None:
        from app.services.recipe_optimizer_catalog import compatibility_check

        result = compatibility_check(
            {
                "optimizer_type": "AdamW8bit",
                "cache_text_encoder_outputs": True,
                "network_train_unet_only": True,
                "text_encoder_lr1": 0.00001,
                "text_encoder_lr2": 0,
            },
            network_type={"id": "standard_lora", "display_name": "Standard LoRA", "availability": "available"},
            optimizer_definition={"id": "AdamW8bit"},
        )

        self.assertFalse(result["ok"])
        self.assertTrue(any("cache_text_encoder_outputs" in message for message in result["errors"]))
        self.assertTrue(any("network_train_unet_only" in message for message in result["errors"]))

    def test_phase125_lora_c3lier_network_type_seed(self) -> None:
        import json

        from app.db import fetch_one

        network_type = fetch_one("SELECT * FROM network_type_definitions WHERE id = ?", ("lora_c3lier",))
        self.assertIsNotNone(network_type)
        self.assertEqual(network_type["display_name"], "LoRA-C3Lier")
        self.assertEqual(network_type["network_module"], "networks.lora")
        self.assertEqual(network_type["availability"], "available")
        schema = json.loads(network_type["params_schema_json"])
        self.assertEqual(schema["conv_dim"], "int")
        self.assertEqual(schema["conv_alpha"], "int")
        self.assertIn("LoCon-like", schema["alias"])
        self.assertIn("LyCORIS LoConとは別実装", network_type["description"])

        legacy = fetch_one("SELECT * FROM network_type_definitions WHERE id = ?", ("locon",))
        if legacy is not None:
            self.assertEqual(legacy["is_active"], 0)
            self.assertEqual(legacy["availability"], "unsupported")

    def test_phase125_job_wizard_network_type_choices(self) -> None:
        from app.main import job_new

        body = job_new(request=None, project_id="", mode="custom").body.decode("utf-8")

        self.assertIn('value="standard_lora"', body)
        self.assertIn('value="lora_c3lier"', body)
        self.assertIn("LoRA-C3Lier / available", body)
        self.assertIn('value="lycoris_locon"', body)
        self.assertIn("LyCORIS LoCon / planned", body)
        self.assertRegex(body, r'<option value="lycoris_locon"[^>]*disabled')
        self.assertNotIn('value="locon"', body)
        self.assertIn("networks.loraにconv_dim / conv_alpha", body)
        self.assertIn("LyCORIS LoConとは別実装", body)
        self.assertIn('data-network-param="lora_c3lier" hidden', body)

    def test_phase125_lora_c3lier_recipe_seed_and_compatibility(self) -> None:
        import json

        from app.services.recipe_optimizer_catalog import compatibility_check

        recipes = fetch_all("SELECT * FROM training_recipes_v2 WHERE network_type_id = ? ORDER BY id", ("lora_c3lier",))
        recipe_ids = {row["id"] for row in recipes}
        self.assertIn("sdxl_character_face_lora_c3lier_adamw8bit_balanced", recipe_ids)
        self.assertIn("sdxl_costume_lora_c3lier_adamw8bit_balanced", recipe_ids)
        self.assertIn("sdxl_style_lora_c3lier_adamw8bit_soft", recipe_ids)
        for row in recipes:
            params = json.loads(row["params_json"])
            self.assertEqual(params["network_dim"], 32)
            self.assertEqual(params["network_alpha"], 16)
            self.assertEqual(params["conv_dim"], 8)
            self.assertEqual(params["conv_alpha"], 4)
            self.assertEqual(params["optimizer_type"], "AdamW8bit")
            self.assertEqual(params["learning_rate"], 0.0001)
            self.assertEqual(row["target_steps_recommended"], 5000)

        network_type = fetch_one("SELECT * FROM network_type_definitions WHERE id = ?", ("lora_c3lier",))
        ok = compatibility_check({"conv_dim": 8, "conv_alpha": 4}, network_type=network_type)
        self.assertTrue(ok["ok"])
        missing = compatibility_check({"conv_alpha": 4}, network_type=network_type)
        self.assertFalse(missing["ok"])
        self.assertTrue(any("conv_dim" in message for message in missing["errors"]))
        high_alpha = compatibility_check({"conv_dim": 4, "conv_alpha": 8}, network_type=network_type)
        self.assertTrue(high_alpha["ok"])
        self.assertTrue(any("conv_alpha" in message for message in high_alpha["warnings"]))

    def test_phase125_lora_c3lier_command_uses_network_args(self) -> None:
        import json
        from pathlib import Path

        import app.services.command_builder as command_builder

        original_latest_environment = command_builder.latest_environment
        command_builder.latest_environment = lambda: {
            "sd_scripts_path": "D:/sd-scripts",
            "venv_python_path": "D:/sd-scripts/venv/Scripts/python.exe",
        }
        job = {
            "training_script": "sdxl_train_network.py",
            "base_model_path": "D:/models/base.safetensors",
            "output_dir": "D:/out",
            "output_name": "c3lier_test",
            "run_dir": "D:/runs/job_1",
            "vae_path": "",
            "params_json": json.dumps(
                {
                    "network_module": "networks.lora",
                    "network_dim": 32,
                    "network_alpha": 16,
                    "conv_dim": 12,
                    "conv_alpha": 6,
                    "network_args": {"conv_dim": 8, "conv_alpha": 4},
                    "optimizer_type": "AdamW8bit",
                    "generate_training_samples": False,
                }
            ),
        }

        try:
            argv = command_builder.build_command_argv(job, Path("D:/dataset.toml"), Path("D:/sample.txt"))
        finally:
            command_builder.latest_environment = original_latest_environment

        self.assertIn("--network_module", argv)
        self.assertIn("networks.lora", argv)
        self.assertIn("--network_dim", argv)
        self.assertIn("32", argv)
        self.assertIn("--network_alpha", argv)
        self.assertIn("16", argv)
        self.assertNotIn("--conv_dim", argv)
        self.assertNotIn("--conv_alpha", argv)
        network_args_index = argv.index("--network_args")
        self.assertIn("conv_dim=12", argv[network_args_index + 1 :])
        self.assertIn("conv_alpha=6", argv[network_args_index + 1 :])

    def test_phase125_smoke_params_use_max_steps_without_epoch_limit(self) -> None:
        from app.services.optimizer_profile_validation import smoke_params

        recipe = fetch_one(
            "SELECT * FROM training_recipes_v2 WHERE id = ?",
            ("sdxl_character_face_lora_c3lier_adamw8bit_balanced",),
        )

        params = smoke_params(recipe)

        self.assertEqual(params["max_train_steps"], 2)
        self.assertIsNone(params["max_train_epochs"])
        self.assertIsNone(params["save_every_n_epochs"])
        self.assertIsNone(params["sample_every_n_epochs"])
        self.assertEqual(params["conv_dim"], 8)
        self.assertEqual(params["conv_alpha"], 4)

    def test_phase121_create_job_rejects_planned_network_type_recipe(self) -> None:
        from app.db import create_job

        now = utc_now()
        project_id, dataset_id, version_id = self.create_project_fixture()
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO training_recipes_v2 (
                    id, name, display_name, model_family, training_purpose_id,
                    optimizer_definition_id, optimizer_profile_id, network_type_id,
                    recipe_type, params_json, basic_params_json, advanced_params_json,
                    raw_args_json, compatibility_rules_json, target_steps_min,
                    target_steps_recommended, target_steps_max, target_checkpoint_count,
                    expected_behavior, risk_note, sort_order, is_builtin, is_active,
                    created_at, updated_at
                )
                SELECT
                    'test_lycoris_locon_planned_recipe', 'test_lycoris_locon_planned_recipe', 'Test LyCORIS LoCon Planned',
                    model_family, training_purpose_id, optimizer_definition_id, optimizer_profile_id,
                    'lycoris_locon', recipe_type, params_json, basic_params_json, advanced_params_json,
                    raw_args_json, compatibility_rules_json, target_steps_min,
                    target_steps_recommended, target_steps_max, target_checkpoint_count,
                    expected_behavior, 'planned network type test', 9999, 0, 1, ?, ?
                FROM training_recipes_v2
                WHERE id = 'sdxl_character_face_adamw8bit_standard_10epoch'
                """,
                (now, now),
            )

        with self.assertRaises(ValueError) as raised:
            create_job(
                {
                    "project_id": project_id,
                    "name": "planned network recipe",
                    "dataset_id": dataset_id,
                    "dataset_version_id": version_id,
                    "preset_id": "sdxl_2d_face_adamw8bit_standard",
                    "recipe_v2_id": "test_lycoris_locon_planned_recipe",
                    "base_model_path": "D:/models/base.safetensors",
                    "output_name": "planned_network_recipe",
                    "params": {"repeats": 10, "max_train_epochs": 10, "train_batch_size": 1},
                }
            )

        self.assertIn("planned", str(raised.exception))
        self.assertIn("実行できません", str(raised.exception))

    def test_phase121_recipe_and_optimizer_pages_render(self) -> None:
        from app.main import optimizers_library, training_recipes_library

        recipes_body = training_recipes_library(request=None, model_family="", purpose="", optimizer="", recipe_type="").body.decode("utf-8")
        optimizers_body = optimizers_library(request=None).body.decode("utf-8")

        self.assertIn("Training Recipe Library", recipes_body)
        self.assertIn("顔キャラ・標準10epoch", recipes_body)
        self.assertIn("Profile検証", recipes_body)
        self.assertIn("Optimizer Master", optimizers_body)
        self.assertIn("Prodigy", optimizers_body)

    def test_phase122_recipe_wizard_pages_render(self) -> None:
        from app.main import job_new, optimizer_detail, training_recipe_detail

        body = job_new(request=None, project_id="", mode="optimizer").body.decode("utf-8")
        recipe_body = training_recipe_detail(request=None, recipe_id="sdxl_character_face_adamw8bit_standard_10epoch").body.decode("utf-8")
        optimizer_body = optimizer_detail(request=None, optimizer_id="Prodigy").body.decode("utf-8")

        self.assertIn("Recipe Wizard UX", body)
        self.assertIn("Parameter Editor v2", body)
        self.assertIn("Compatibility Check", body)
        self.assertIn("data-mode-summary", body)
        self.assertIn("data-mode-change-button", body)
        self.assertIn("data-selected-recipe-panel", body)
        self.assertIn("data-recipe-result-count", body)
        self.assertIn("data-optimizer-info-panel", body)
        self.assertIn("Recipeを直接選択", body)
        self.assertIn("[SDXL] 顔キャラ・標準 / AdamW8bit", body)
        self.assertNotIn("<strong>SDXL Character Face / AdamW8bit Balanced</strong>", body)
        self.assertIn("この条件に合うRecipeはまだ登録されていません。", body)
        self.assertIn("/jobs/new?mode=custom", body)
        self.assertIn("顔キャラ・標準10epoch", recipe_body)
        self.assertIn("[SDXL] 顔キャラ・標準10epoch / AdamW8bit", recipe_body)
        self.assertIn("learning_rate=1.0", optimizer_body)
        self.assertIn("Optimizer Profile Validation", optimizer_body)
        self.assertIn("Run 2-step Smoke Test", optimizer_body)

    def test_phase122_structured_user_override_diff_is_saved(self) -> None:
        from app.db import create_job

        project_id, dataset_id, version_id = self.create_project_fixture()
        job_id = create_job(
            {
                "project_id": project_id,
                "name": "phase122 override diff",
                "dataset_id": dataset_id,
                "dataset_version_id": version_id,
                "preset_id": "sdxl_2d_face_adamw8bit_standard",
                "recipe_v2_id": "sdxl_character_face_adamw8bit_standard_10epoch",
                "base_model_path": "D:/models/base.safetensors",
                "output_name": "phase122_override_diff",
                "params": {"repeats": 12, "max_train_epochs": 10, "train_batch_size": 1},
                "user_overrides": {"repeats": 12},
                "user_overrides_detail": {
                    "repeats": {
                        "from": 10,
                        "to": 12,
                        "reason": "target step assistant",
                    }
                },
            }
        )
        job = fetch_one("SELECT user_overrides_json FROM training_jobs WHERE id = ?", (job_id,))
        overrides = json.loads(job["user_overrides_json"])

        self.assertEqual(overrides["repeats"]["from"], 10)
        self.assertEqual(overrides["repeats"]["to"], 12)
        self.assertEqual(overrides["repeats"]["reason"], "target step assistant")

    def test_phase122_edit_step_estimate_uses_recipe_v2_target(self) -> None:
        from app.db import create_job
        from app.main import job_edit

        project_id, dataset_id, version_id = self.create_project_fixture()
        job_id = create_job(
            {
                "project_id": project_id,
                "name": "prodigy edit estimate",
                "dataset_id": dataset_id,
                "dataset_version_id": version_id,
                "preset_id": "sdxl_2d_face_adamw8bit_standard",
                "recipe_v2_id": "sdxl_character_face_prodigy_soft_advanced",
                "base_model_path": "D:/models/base.safetensors",
                "output_name": "prodigy_edit_estimate",
                "params": {"repeats": 11, "max_train_epochs": 10, "train_batch_size": 1},
            }
        )

        body = job_edit(request=None, job_id=job_id).body.decode("utf-8")

        self.assertIn('name="recipe_v2_id" value="sdxl_character_face_prodigy_soft_advanced"', body)
        self.assertIn('data-step-field="optimizer_name">Prodigy', body)
        self.assertIn('data-step-field="recipe_name">sdxl_character_face_prodigy_soft_advanced', body)
        self.assertIn('data-step-field="target_source">recipe', body)

    def test_weight_calibration_pipeline_start_recreates_missing_conditions(self) -> None:
        from app.services.validation_generation import start_weight_calibration_pipeline

        run_id, _ = self.create_validation_generation_fixture()
        with connect() as conn:
            conn.execute("DELETE FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
        with mock.patch("app.services.validation_generation.threading.Thread"):
            start_weight_calibration_pipeline(run_id, force_warnings=True)

        count = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
        self.assertEqual(count["count"], 45)

    def test_weight_calibration_planned_primary_action_and_detail_actions(self) -> None:
        from app.main import validation_run_detail

        run_id, _ = self.create_validation_generation_fixture()
        body = validation_run_detail(request=None, run_id=run_id).body.decode("utf-8")

        self.assertIn("Weight検証を開始してください。45枚の検証画像を生成し、Embedding・Machine Review・Matrix作成まで自動で実行します。", body)
        self.assertIn('href="/validation-runs/1/pipeline/run"', body)
        self.assertEqual(body.count('href="/validation-runs/1/pipeline/run"'), 1)
        self.assertNotIn("Prepare Weight Calibration", body)
        self.assertIn("詳細操作", body)
        self.assertIn("Weight検証を準備", body)
        self.assertIn("画像生成＋不足レビュー計算", body)
        self.assertIn("Reimport", body)
        self.assertIn("検証レポート出力", body)
        self.assertIn("Matrix再作成", body)

    def test_weight_calibration_start_confirmation_page_renders_execution_plan(self) -> None:
        from app.main import validation_pipeline_run_confirm

        run_id, _ = self.create_validation_generation_fixture()
        with connect() as conn:
            conn.execute("DELETE FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))

        body = validation_pipeline_run_confirm(request=None, run_id=run_id).body.decode("utf-8")
        count = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))

        self.assertEqual(count["count"], 45)
        self.assertIn("Weight検証開始の確認", body)
        self.assertIn("selected Job", body)
        self.assertIn("selected epoch", body)
        self.assertIn("selected LoRA", body)
        self.assertIn("validation preset", body)
        self.assertIn("expected image count", body)
        self.assertIn("prompts数", body)
        self.assertIn("seeds数", body)
        self.assertIn("weights", body)
        self.assertIn("Hires有無", body)
        self.assertIn("conditions作成", body)
        self.assertIn("sd-scripts画像生成", body)
        self.assertIn("machine review", body)
        self.assertIn("matrix作成", body)

    def test_weight_calibration_primary_action_changes_by_state(self) -> None:
        from app.main import validation_run_detail

        run_id, _ = self.create_validation_generation_fixture()
        matrix_path = settings.EXPORTS_DIR / "validation_runs" / "validation_run_000001" / "validation_matrix.html"
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_path.write_text("<html>matrix</html>", encoding="utf-8")
        with connect() as conn:
            conn.execute(
                "UPDATE validation_runs SET pipeline_status = 'ready_for_review', matrix_path = ? WHERE id = ?",
                (str(matrix_path), run_id),
            )
        ready_body = validation_run_detail(request=None, run_id=run_id).body.decode("utf-8")
        self.assertIn("Weight Review Matrixを開く", ready_body)

        with connect() as conn:
            conn.execute(
                "UPDATE validation_runs SET status = 'completed', pipeline_status = 'ready_for_review', profile_applied_at = NULL WHERE id = ?",
                (run_id,),
            )
        completed_unapplied_body = validation_run_detail(request=None, run_id=run_id).body.decode("utf-8")
        self.assertIn("Profileへ反映", completed_unapplied_body)

        with connect() as conn:
            conn.execute("UPDATE validation_runs SET profile_applied_at = ? WHERE id = ?", (utc_now(), run_id))
        completed_applied_body = validation_run_detail(request=None, run_id=run_id).body.decode("utf-8")
        self.assertIn("Weight Review Matrixを開く", completed_applied_body)

    def test_weight_calibration_pipeline_stop_sets_stopped_status(self) -> None:
        from app.services.validation_generation import stop_weight_calibration_pipeline

        run_id, _ = self.create_validation_generation_fixture()
        with connect() as conn:
            conn.execute("UPDATE validation_runs SET pipeline_status = 'generating_images' WHERE id = ?", (run_id,))
        stop_weight_calibration_pipeline(run_id)

        run = fetch_one("SELECT status, pipeline_status, memo FROM validation_runs WHERE id = ?", (run_id,))
        self.assertEqual(run["status"], "stopped")
        self.assertEqual(run["pipeline_status"], "stopped")
        self.assertIn("stopped by user", run["memo"])

    def test_weight_calibration_suggestion_keeps_strong_usable_outside_recommended_range(self) -> None:
        from app.main import register_validation_run_image
        from app.services.validation_runs import load_validation_run_bundle

        run_id, _ = self.create_validation_generation_fixture()
        labels = {
            0.4: ("weak_but_usable", "candidate", 3),
            0.6: ("recommended", "adopt", 4),
            0.8: ("recommended", "adopt", 4),
            1.0: ("strong_but_usable", "candidate", 3),
        }
        for weight, (strength, adoption, rating) in labels.items():
            condition = fetch_one(
                """
                SELECT * FROM validation_expected_conditions
                WHERE validation_run_id = ? AND lora_weight = ?
                ORDER BY expected_order LIMIT 1
                """,
                (run_id, weight),
            )
            source = self.make_png(self.root / "generated" / f"w{weight}.png")
            register_validation_run_image(
                run_id,
                str(source),
                "individual",
                condition["prompt_key"],
                int(condition["seed"]),
                weight,
                bool(condition["hires_enabled"]),
                "",
            )
            with connect() as conn:
                conn.execute(
                    """
                    UPDATE validation_images
                    SET rating_overall = ?, strength_label = ?, adoption_label = ?
                    WHERE validation_run_id = ? AND expected_condition_id = ?
                    """,
                    (rating, strength, adoption, run_id, condition["id"]),
                )

        suggestion = load_validation_run_bundle(run_id)["suggestion"]

        self.assertEqual(suggestion["suggested_weight_min"], 0.6)
        self.assertEqual(suggestion["suggested_weight_max"], 0.8)
        self.assertEqual(suggestion["suggested_light_weight"], 0.4)
        self.assertEqual(suggestion["suggested_strong_weight"], 1.0)

    def test_weight_review_matrix_updates_run_matrix_path(self) -> None:
        from app.services.validation_generation import write_validation_matrix

        run_id, _ = self.create_validation_generation_fixture()
        matrix_path = write_validation_matrix(run_id)
        run = fetch_one("SELECT matrix_path FROM validation_runs WHERE id = ?", (run_id,))

        self.assertEqual(run["matrix_path"], matrix_path)
        self.assertTrue(Path(matrix_path).exists())

    def test_bulk_validation_generation_persists_queued_runs(self) -> None:
        from app.main import current_running_validation_generation
        from app.services.validation_generation import start_validation_generation_sequence
        from app.services.validation_runs import create_validation_run

        first_run_id, selected_output_id = self.create_validation_generation_fixture()
        first_run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (first_run_id,))
        second_run_id = create_validation_run(first_run["job_id"], "standard_validation_v1", str(self.root / "models" / "base.safetensors"), "testchar", "")
        with connect() as conn:
            conn.execute(
                "UPDATE validation_runs SET selected_output_id = ? WHERE id = ?",
                (selected_output_id, second_run_id),
            )

        with mock.patch("app.services.validation_generation.start_validation_generation", return_value=12345) as start_mock, mock.patch(
            "app.services.validation_generation.threading.Thread"
        ) as thread_mock:
            count = start_validation_generation_sequence([first_run_id, second_run_id])

        queued = fetch_one("SELECT * FROM validation_generation_runs WHERE validation_run_id = ? ORDER BY id DESC LIMIT 1", (second_run_id,))
        active = current_running_validation_generation()
        self.assertEqual(count, 2)
        start_mock.assert_called_once_with(first_run_id)
        thread_mock.return_value.start.assert_called_once()
        self.assertEqual(queued["status"], "queued")
        self.assertEqual(active["status"], "queued")
        self.assertEqual(active["validation_run_id"], second_run_id)

    def test_superseded_queued_validation_generation_is_not_active(self) -> None:
        from app.main import current_running_validation_generation
        from app.services.validation_generation import reconcile_stale_validation_generations

        run_id, _ = self.create_validation_generation_fixture()
        now = utc_now()
        with connect() as conn:
            old_id = int(
                conn.execute(
                    """
                    INSERT INTO validation_generation_runs(
                        validation_run_id, status, created_at, updated_at
                    )
                    VALUES (?, 'queued', ?, ?)
                    """,
                    (run_id, now, now),
                ).lastrowid
            )
            conn.execute(
                """
                INSERT INTO validation_generation_runs(
                    validation_run_id, status, generated_image_count,
                    imported_image_count, created_at, updated_at
                )
                VALUES (?, 'completed', 45, 45, ?, ?)
                """,
                (run_id, now, now),
            )

        self.assertIsNone(current_running_validation_generation())
        fixed = reconcile_stale_validation_generations()
        old_generation = fetch_one("SELECT * FROM validation_generation_runs WHERE id = ?", (old_id,))
        self.assertEqual(fixed, 1)
        self.assertEqual(old_generation["status"], "superseded")
        self.assertIsNone(current_running_validation_generation())

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
        self.assertIn("選択weightを追加生成して不足レビューも再計算", cross_html)
        self.assertIn("選択weightでMatrix表示を更新", cross_html)
        self.assertIn("display_weights", cross_html)
        self.assertIn(f"/jobs/{job_id}/validation-runs/epoch-matrix/weights", cross_html)
        self.assertIn('name="selected_weights"', cross_html)
        self.assertIn('value="0.9"', cross_html)
        self.assertIn("pollMatrixGeneration", cross_html)
        self.assertIn("matrix_message", cross_html)
        self.assertIn(f"/jobs/{job_id}/validation-runs/epoch-matrix/missing-review", cross_html)
        self.assertIn(f'name="run_ids" value="{run_id}"', cross_html)
        self.assertIn(f'name="run_ids" value="{second_run_id}"', cross_html)
        self.assertIn("Epoch 4", cross_html)
        self.assertIn("Epoch 7", cross_html)
        self.assertIn(f"/validation-images/{image['id']}", cross_html)
        second_image = fetch_one("SELECT * FROM validation_images WHERE validation_run_id = ?", (second_run_id,))
        self.assertIn(f"/validation-images/{second_image['id']}", cross_html)

        filtered_html = build_epoch_cross_matrix_html(job_id, [run_id, second_run_id], display_weights=["0.4"])
        self.assertIn("seed 111111 / weight 0.4", filtered_html)
        self.assertNotIn("seed 111111 / weight 0.6", filtered_html)

    def test_validation_matrix_can_add_weight_conditions(self) -> None:
        from app.services.validation_generation import write_validation_matrix
        from app.services.validation_runs import add_validation_run_weights

        run_id, _ = self.create_validation_generation_fixture()
        before = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
        summary = add_validation_run_weights(run_id, ["0.8", "0.9", "1.0", "bad", "2.1"])
        after = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
        added_weight = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ? AND lora_weight = 0.9", (run_id,))
        run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))

        self.assertEqual(summary["selected_weights"], [0.8, 0.9, 1.0])
        self.assertEqual(summary["added_weights"], [0.9])
        self.assertEqual(after["count"], before["count"] + 9)
        self.assertEqual(added_weight["count"], 9)
        self.assertEqual(run["expected_image_count"], after["count"])

        matrix_path = Path(write_validation_matrix(run_id))
        matrix_html = matrix_path.read_text(encoding="utf-8")
        self.assertIn("weight選択", matrix_html)
        self.assertIn('name="selected_weights" value="0.9" checked', matrix_html)
        self.assertIn("1.1〜2.0を表示", matrix_html)
        self.assertIn("選択weightを追加生成して不足レビューも再計算", matrix_html)
        self.assertIn("不足レビューだけ再計算", matrix_html)
        self.assertIn(f"/validation-runs/{run_id}/matrix/missing-review", matrix_html)
        self.assertIn(f'name="run_ids" value="{run_id}"', matrix_html)

    def test_validation_matrix_weight_post_adds_conditions_and_starts_missing_generation(self) -> None:
        from app.main import validation_matrix_add_weights

        run_id, _ = self.create_validation_generation_fixture()
        with mock.patch("app.main.start_validation_generation", return_value=12345) as start_mock:
            response = validation_matrix_add_weights(run_id, ["0.9"])

        added_weight = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ? AND lora_weight = 0.9", (run_id,))
        self.assertEqual(response.status_code, 303)
        self.assertIn(f"/validation-runs/{run_id}/matrix?", response.headers["location"])
        self.assertIn("display_weights=0.9", response.headers["location"])
        self.assertIn("weight_notice=weights_started", response.headers["location"])
        self.assertEqual(added_weight["count"], 9)
        start_mock.assert_called_once_with(run_id, run_missing_review_after=True)

    def test_validation_matrix_missing_review_post_starts_sequence(self) -> None:
        from app.main import validation_matrix_missing_review

        run_id, _ = self.create_validation_generation_fixture()
        with mock.patch("app.main.start_missing_validation_review_sequence", return_value={"status": "started"}) as start_mock:
            response = validation_matrix_missing_review(run_id)

        self.assertEqual(response.status_code, 303)
        self.assertIn(f"/validation-runs/{run_id}/matrix?weight_notice=missing_review_started", response.headers["location"])
        start_mock.assert_called_once_with(run_id)

    def test_epoch_cross_matrix_missing_review_post_starts_sequences(self) -> None:
        from app.main import validation_epoch_cross_matrix_missing_review

        run_id, _ = self.create_validation_generation_fixture()
        second_run_id, _ = self.create_validation_generation_fixture()
        with mock.patch("app.main.start_missing_validation_review_sequences", return_value={"status": "started"}) as start_mock:
            response = validation_epoch_cross_matrix_missing_review(1, [run_id, second_run_id, run_id])

        self.assertEqual(response.status_code, 303)
        location = response.headers["location"]
        self.assertIn("/jobs/1/validation-runs/epoch-matrix?", location)
        self.assertIn(f"run_ids={run_id}", location)
        self.assertIn(f"run_ids={second_run_id}", location)
        self.assertIn("matrix_notice=missing_review_started", location)
        start_mock.assert_called_once_with([run_id, second_run_id])

    def test_epoch_cross_matrix_weight_post_adds_conditions_and_starts_sequence(self) -> None:
        from app.main import validation_epoch_cross_matrix_add_weights
        from app.services.validation_runs import create_validation_run

        run_id, selected_output_id = self.create_validation_generation_fixture()
        run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
        second_run_id = create_validation_run(run["job_id"], "standard_validation_v1", run["base_model"], run["trigger_word"], "second")
        with connect() as conn:
            conn.execute(
                "UPDATE validation_runs SET selected_output_id = ? WHERE id = ?",
                (selected_output_id, second_run_id),
            )

        with mock.patch("app.main.start_validation_generation_sequence", return_value=2) as start_mock:
            response = validation_epoch_cross_matrix_add_weights(run["job_id"], [run_id, second_run_id, run_id], ["0.9"])

        self.assertEqual(response.status_code, 303)
        location = response.headers["location"]
        self.assertIn(f"/jobs/{run['job_id']}/validation-runs/epoch-matrix?", location)
        self.assertIn(f"run_ids={run_id}", location)
        self.assertIn(f"run_ids={second_run_id}", location)
        self.assertIn("display_weights=0.9", location)
        self.assertIn("matrix_notice=weights_started", location)
        added_first = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ? AND lora_weight = 0.9", (run_id,))
        added_second = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ? AND lora_weight = 0.9", (second_run_id,))
        self.assertEqual(added_first["count"], 9)
        self.assertEqual(added_second["count"], 9)
        from app.services.validation_generation import build_epoch_cross_matrix_html

        cross_html = build_epoch_cross_matrix_html(run["job_id"], [run_id, second_run_id])
        self.assertIn('name="selected_weights"', cross_html)
        self.assertIn('value="0.9" checked', cross_html)
        start_mock.assert_called_once_with(
            [run_id, second_run_id],
            run_missing_review_after=True,
            missing_review_run_ids=[run_id, second_run_id],
        )

    def test_epoch_cross_matrix_weight_post_preserves_display_selection(self) -> None:
        from app.main import validation_epoch_cross_matrix_add_weights
        from app.services.validation_runs import create_validation_run

        run_id, selected_output_id = self.create_validation_generation_fixture()
        run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
        second_run_id = create_validation_run(run["job_id"], "standard_validation_v1", run["base_model"], run["trigger_word"], "second")
        with connect() as conn:
            conn.execute(
                "UPDATE validation_runs SET selected_output_id = ? WHERE id = ?",
                (selected_output_id, second_run_id),
            )

        selected = ["0", "0.7", "0.8", "0.9", "1"]
        with mock.patch("app.main.start_validation_generation_sequence", return_value=2):
            response = validation_epoch_cross_matrix_add_weights(run["job_id"], [run_id, second_run_id], selected)

        location = response.headers["location"]
        self.assertIn("display_weights=0", location)
        self.assertIn("display_weights=0.7", location)
        self.assertIn("display_weights=0.8", location)
        self.assertIn("display_weights=0.9", location)
        self.assertIn("display_weights=1", location)
        self.assertNotIn("display_weights=0.4", location)
        self.assertNotIn("display_weights=0.6", location)

    def test_missing_machine_review_job_targets_only_unscored_validation_images(self) -> None:
        from app.main import register_validation_run_image
        from app.services.machine_review import create_machine_review_job

        run_id, _ = self.create_validation_generation_fixture()
        first_condition = dict(fetch_one("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order LIMIT 1", (run_id,)))
        second_condition = dict(fetch_one("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order LIMIT 1 OFFSET 1", (run_id,)))
        first_image_path = self.make_png(self.root / "generated" / "first.png")
        second_image_path = self.make_png(self.root / "generated" / "second.png")
        register_validation_run_image(
            run_id,
            str(first_image_path),
            "individual",
            first_condition["prompt_key"],
            int(first_condition["seed"]),
            float(first_condition["lora_weight"]),
            bool(first_condition["hires_enabled"]),
            "",
        )
        register_validation_run_image(
            run_id,
            str(second_image_path),
            "individual",
            second_condition["prompt_key"],
            int(second_condition["seed"]),
            float(second_condition["lora_weight"]),
            bool(second_condition["hires_enabled"]),
            "",
        )
        first_image = fetch_one("SELECT * FROM validation_images WHERE validation_run_id = ? ORDER BY id LIMIT 1", (run_id,))
        run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
        job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (run["job_id"],))
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO machine_review_scores(
                    source_type, source_id, project_id, job_id, validation_run_id,
                    reference_set_version_id, dataset_version_id, embedding_model_id, provider,
                    assist_label, confidence_label, created_at, updated_at
                )
                VALUES ('validation_image', ?, ?, ?, ?, ?, ?, 'mock_image_512', 'mock',
                        'low_confidence', 'low', ?, ?)
                """,
                (
                    first_image["id"],
                    run["project_id"],
                    run["job_id"],
                    run_id,
                    run["reference_set_version_id"],
                    job["dataset_version_id"] if job else None,
                    now,
                    now,
                ),
            )

        job_id = create_machine_review_job("validation_run_images_missing", run_id)
        job = fetch_one("SELECT * FROM machine_review_jobs WHERE id = ?", (job_id,))
        self.assertEqual(job["total_count"], 1)

    def test_validation_generation_skips_registered_existing_images(self) -> None:
        from app.services.validation_generation import (
            generation_output_dir,
            import_generated_images,
            output_stem,
            prepare_validation_generation,
        )

        run_id, _ = self.create_validation_generation_fixture()
        first_generation = prepare_validation_generation(run_id)
        condition = dict(fetch_one("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order LIMIT 1", (run_id,)))
        image_path = generation_output_dir(run_id) / f"{output_stem(run_id, condition)}.png"
        self.make_png(image_path)
        self.assertEqual(import_generated_images(run_id, int(first_generation["generation_id"])), 1)

        second_generation = prepare_validation_generation(run_id)
        commands = second_generation["commands"]
        condition_count = sum(int(command["condition_count"]) for command in commands)
        payload = json.loads(fetch_one("SELECT command_argv_json FROM validation_generation_runs WHERE id = ?", (second_generation["generation_id"],))["command_argv_json"])
        prompt_text = Path(second_generation["prompt_file"]).read_text(encoding="utf-8")

        self.assertEqual(condition_count, 44)
        self.assertEqual(payload["skipped_existing_count"], 1)
        self.assertNotIn(f"ec{int(condition['id']):06d}", prompt_text)

    def test_validation_generation_import_skips_duplicate_condition_hash(self) -> None:
        from app.services.validation_generation import (
            generation_output_dir,
            import_generated_images,
            output_stem,
            prepare_validation_generation,
        )

        run_id, _ = self.create_validation_generation_fixture()
        generation = prepare_validation_generation(run_id)
        condition = dict(fetch_one("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order LIMIT 1", (run_id,)))
        output_dir = generation_output_dir(run_id)
        image_path = output_dir / f"{output_stem(run_id, condition)}.png"
        self.make_png(image_path)
        self.assertEqual(import_generated_images(run_id, int(generation["generation_id"])), 1)

        duplicate_path = output_dir / f"copy_{condition['condition_hash'][:12]}.png"
        self.make_png(duplicate_path)
        self.assertEqual(import_generated_images(run_id, int(generation["generation_id"])), 0)
        count = fetch_one("SELECT COUNT(*) AS count FROM validation_images WHERE validation_run_id = ?", (run_id,))
        self.assertEqual(count["count"], 1)

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

    def test_epoch_candidates_refresh_when_adopted_epoch_changes(self) -> None:
        from app.services.review_candidates import ensure_epoch_candidates, regenerate_epoch_candidates

        job_id = self.make_review_job()
        regenerate_epoch_candidates(job_id)
        with connect() as conn:
            conn.execute("UPDATE training_jobs SET adopted_epoch = 5 WHERE id = ?", (job_id,))
            conn.execute("UPDATE training_outputs SET selected = CASE WHEN epoch = 5 THEN 1 ELSE 0 END WHERE job_id = ?", (job_id,))

        rows = ensure_epoch_candidates(job_id)
        reasons_by_epoch = {row["epoch"]: row["reasons"] for row in rows}
        self.assertNotIn("採用済みepoch", reasons_by_epoch[4])
        self.assertNotIn("現在の採用epoch", reasons_by_epoch[4])
        self.assertIn("現在の採用epoch", reasons_by_epoch[5])

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
        self.assertIn("データセット画像Embedding状況", dataset_detail(request=None, dataset_id=dataset_id).body.decode("utf-8"))
        self.assertIn("リファレンス画像Embedding状況", reference_page(request=None, set_id=reference_set_id).body.decode("utf-8"))
        self.assertIn("検証画像Embedding状況", validation_page(request=None, run_id=run_id).body.decode("utf-8"))

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
        sample_job = fetch_one("SELECT stage_timing_json FROM machine_review_jobs WHERE id = ?", (sample_result["job_id"],))
        validation_job = fetch_one("SELECT stage_timing_json FROM machine_review_jobs WHERE id = ?", (validation_result["job_id"],))
        self.assertIn("load_target_embeddings_seconds", sample_job["stage_timing_json"])
        self.assertIn("similarity_calculation_seconds", validation_job["stage_timing_json"])
        self.assertIn("db_write_seconds", validation_job["stage_timing_json"])

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
        self.assertIn("機械補助レビュー準備状況", reference_page(request=None, set_id=reference_set_id).body.decode("utf-8"))
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

    def test_stale_machine_review_job_is_reconciled(self) -> None:
        from app.services.machine_review import create_machine_review_job, reconcile_stale_machine_review_jobs

        dataset_id, version_id, _dataset_dir = self.make_dataset_version()
        job_id = self.make_sample_job(dataset_id, version_id)
        machine_review_job_id = create_machine_review_job("training_job_samples", job_id)
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                UPDATE machine_review_jobs
                SET status = 'running', process_id = ?, started_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (424242, now, now, machine_review_job_id),
            )

        with mock.patch("app.services.machine_review.process_exists", return_value=False):
            fixed = reconcile_stale_machine_review_jobs()

        row = fetch_one("SELECT status, process_id, return_code, error_message FROM machine_review_jobs WHERE id = ?", (machine_review_job_id,))
        self.assertEqual(fixed, 1)
        self.assertEqual(row["status"], "stopped")
        self.assertIsNone(row["process_id"])
        self.assertEqual(row["return_code"], -1)
        self.assertIn("Machine Review process was not found", row["error_message"])


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


class OptimizerDependencyTests(IsolatedDbTest):
    def add_environment(self) -> Path:
        fake_python = self.root / "external" / "sd-scripts" / "venv" / "Scripts" / "python.exe"
        fake_python.parent.mkdir(parents=True, exist_ok=True)
        fake_python.write_text("", encoding="utf-8")
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO environments(
                    name, sd_scripts_path, venv_python_path, venv_accelerate_path,
                    status, created_at, updated_at
                )
                VALUES ('test', ?, ?, ?, 'ready', ?, ?)
                """,
                (
                    str(self.root / "external" / "sd-scripts"),
                    str(fake_python),
                    str(fake_python.parent / "accelerate.exe"),
                    now,
                    now,
                ),
            )
        return fake_python

    def test_optional_optimizer_dependencies_are_seeded(self) -> None:
        rows = fetch_all("SELECT id, package_name, install_target FROM optional_optimizer_dependencies ORDER BY id")
        self.assertEqual([row["id"] for row in rows], ["dadaptation", "lion-pytorch", "prodigyopt"])
        self.assertTrue(all(row["install_target"] == "sd_scripts_venv" for row in rows))

    def test_dependency_check_uses_sd_scripts_python_and_saves_status(self) -> None:
        from app.services.optimizer_dependencies import check_dependency

        fake_python = self.add_environment()
        completed = subprocess.CompletedProcess([str(fake_python)], 0, stdout="module-path\n", stderr="")
        with mock.patch("app.services.optimizer_dependencies.subprocess.run", return_value=completed) as run:
            result = check_dependency("dadaptation")

        self.assertEqual(result["status"], "installed")
        self.assertEqual(run.call_args.args[0][0], str(fake_python))
        row = fetch_one("SELECT status, last_checked_at, error_message FROM optional_optimizer_dependencies WHERE id = 'dadaptation'")
        self.assertEqual(row["status"], "installed")
        self.assertIsNotNone(row["last_checked_at"])
        self.assertIsNone(row["error_message"])

    def test_dependency_install_failure_is_saved_without_raising(self) -> None:
        from app.services.optimizer_dependencies import install_dependency

        fake_python = self.add_environment()
        completed = subprocess.CompletedProcess([str(fake_python)], 1, stdout="install failed", stderr="")
        with mock.patch("app.services.optimizer_dependencies.subprocess.run", return_value=completed) as run:
            result = install_dependency("lion-pytorch")

        self.assertEqual(result["status"], "install_failed")
        self.assertEqual(run.call_args.args[0][0], str(fake_python))
        self.assertIn("-m", run.call_args.args[0])
        self.assertIn("pip", run.call_args.args[0])
        row = fetch_one("SELECT status, last_install_at, error_message FROM optional_optimizer_dependencies WHERE id = 'lion-pytorch'")
        self.assertEqual(row["status"], "install_failed")
        self.assertIsNotNone(row["last_install_at"])
        self.assertIn("install failed", row["error_message"])


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
