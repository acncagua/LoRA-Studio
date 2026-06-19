from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from app import settings
from app.db import connect, fetch_all, fetch_one, latest_environment, utc_now
from app.services.image_store import verify_image_file
from app.services.embedding_service import reconcile_stale_embedding_jobs
from app.services.output_collector import image_size, safe_sha256_file
from app.services.review_candidates import ensure_epoch_candidates
from app.services.training_runner import archive_existing_log, decode_log_bytes, elapsed_seconds, process_exists, sd_scripts_subprocess_env
from app.services.validation_generation import IMAGE_SUFFIXES, common_gen_img_args, count_generated_images, sanitize_filename
from app.services.validation_generation import matrix_display_controls, matrix_display_script, matrix_display_style, matrix_machine_score


PRESET_ID = "candidate_epoch_review_v1"
PRESET_NAME = "候補epochレビュー"
PROMPTS: list[dict[str, str]] = [
    {
        "key": "basic_face",
        "role": "face",
        "prompt": "{trigger}, 1girl, upper body, looking at viewer, simple background",
    },
    {
        "key": "full_body",
        "role": "full_body",
        "prompt": "{trigger}, 1girl, full body, standing, simple background",
    },
    {
        "key": "expression_pose",
        "role": "expression_pose",
        "prompt": "{trigger}, 1girl, looking at viewer, dynamic pose, expressive face, simple background",
    },
]
WEIGHTS = [0.6, 0.8]
SEED = 111111
WIDTH = 1024
HEIGHT = 1024
STEPS = 28
CFG_SCALE = 7.0
SAMPLER = "euler_a"
NEGATIVE_PROMPT = "low quality, worst quality, bad anatomy, extra fingers, missing fingers, blurry"


def preset_snapshot() -> dict[str, Any]:
    return {
        "id": PRESET_ID,
        "name": PRESET_NAME,
        "version": "1",
        "purpose": "採用前の候補epochを少数条件で横断比較するレビュー用preset。",
        "epoch_policy": "candidate_epoch_plus_minus_1",
        "prompts": PROMPTS,
        "seed": SEED,
        "weights": WEIGHTS,
        "hires_enabled": False,
        "width": WIDTH,
        "height": HEIGHT,
        "steps": STEPS,
        "cfg_scale": CFG_SCALE,
        "sampler": SAMPLER,
        "negative_prompt": NEGATIVE_PROMPT,
    }


ACTIVE_REVIEW_STATUSES = {"running", "generating_images", "embedding_images", "machine_reviewing", "building_matrix"}


def latest_review_session(job_id: int) -> dict[str, Any] | None:
    row = fetch_one(
        """
        SELECT * FROM review_sessions
        WHERE job_id = ? AND preset_id = ?
        ORDER BY
            CASE
                WHEN status IN ('running', 'generating_images', 'embedding_images', 'machine_reviewing', 'building_matrix') THEN 0
                WHEN status = 'completed' AND COALESCE(matrix_path, '') != '' THEN 1
                WHEN status = 'completed' THEN 2
                WHEN status IN ('failed', 'stopped') THEN 3
                WHEN status IN ('planned', 'prepared') THEN 4
                ELSE 5
            END,
            id DESC
        LIMIT 1
        """,
        (job_id, PRESET_ID),
    )
    return dict(row) if row else None


def review_session_priority(session: dict[str, Any]) -> tuple[int, int]:
    status = str(session.get("status") or "")
    if status in ACTIVE_REVIEW_STATUSES:
        rank = 0
    elif status == "completed" and session.get("matrix_path"):
        rank = 1
    elif status == "completed":
        rank = 2
    elif status in {"failed", "stopped"}:
        rank = 3
    elif status in {"planned", "prepared"}:
        rank = 4
    else:
        rank = 5
    return rank, -int(session.get("id") or 0)


def review_session_rows(job_id: int) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in fetch_all(
            """
            SELECT * FROM review_sessions
            WHERE job_id = ? AND preset_id = ?
            ORDER BY id DESC
            """,
            (job_id, PRESET_ID),
        )
    ]


def summarize_review_session(session: dict[str, Any]) -> dict[str, Any]:
    session_id = int(session["id"])
    condition_count = fetch_one(
        "SELECT COUNT(*) AS c FROM review_session_conditions WHERE review_session_id = ?",
        (session_id,),
    )
    image_count = fetch_one(
        "SELECT COUNT(*) AS c FROM review_session_images WHERE review_session_id = ? AND deleted_at IS NULL",
        (session_id,),
    )
    try:
        candidate_epochs = json.loads(session.get("candidate_epochs_json") or "[]")
    except json.JSONDecodeError:
        candidate_epochs = []
    matrix_path = session.get("matrix_path") or ""
    try:
        from app.services.embedding_service import embedding_coverage

        embedding = embedding_coverage("review_session", session_id)
    except Exception:
        embedding = None
    counted_conditions = int(condition_count["c"] if condition_count else 0)
    counted_images = int(image_count["c"] if image_count else 0)
    expected = int(session.get("expected_image_count") or counted_conditions)
    status = session.get("status") or "-"
    has_matrix_path = bool(matrix_path)
    can_open_matrix = bool(matrix_path and Path(matrix_path).exists())
    if status in {"running", "generating_images", "embedding_images", "machine_reviewing", "building_matrix"}:
        primary_action = "progress"
        primary_label = "進捗を確認"
    elif status == "completed" and can_open_matrix:
        primary_action = "open_matrix"
        primary_label = "レビューMatrixを開く"
    elif status == "completed":
        primary_action = "build_matrix"
        primary_label = "レビューMatrixを作成"
    elif status in {"planned", "prepared"}:
        primary_action = "start"
        primary_label = "候補レビューを開始"
    elif status in {"failed", "stopped"}:
        primary_action = "check_log"
        primary_label = "ログを確認"
    else:
        primary_action = "start"
        primary_label = "候補レビューを開始"

    if embedding is None:
        embedding_display = "未確認"
    elif int(embedding.get("total") or 0) <= 0:
        embedding_display = "対象外"
    else:
        embedding_display = f"{int(embedding.get('ready') or 0)} / {int(embedding.get('total') or 0)}"
    return {
        "session": session,
        "session_id": session_id,
        "status": status,
        "condition_count": counted_conditions,
        "image_count": counted_images,
        "registered_image_count": int(session.get("imported_image_count") or counted_images),
        "generated_image_count": int(session.get("generated_image_count") or 0),
        "machine_review_count": int(session.get("scored_image_count") or 0),
        "expected_image_count": expected,
        "generation_target_count": expected,
        "candidate_epochs": candidate_epochs,
        "matrix_path": matrix_path,
        "has_matrix": has_matrix_path,
        "can_open_matrix": can_open_matrix,
        "primary_action": primary_action,
        "primary_label": primary_label,
        "embedding_display": embedding_display,
        "embedding_coverage": embedding,
        "log_tail": review_session_log_tail(session, max_lines=20),
        "log_size": review_session_log_size(session),
    }


def review_session_summary(job_id: int, current_session_id: int | None = None) -> dict[str, Any]:
    sessions = review_session_rows(job_id)
    selected_session: dict[str, Any] | None = None
    if current_session_id is not None:
        selected_session = next((session for session in sessions if int(session["id"]) == current_session_id), None)
    if selected_session is None and sessions:
        selected_session = sorted(sessions, key=review_session_priority)[0]
    if selected_session is None:
        return {
            "session": None,
            "current": None,
            "other_sessions": [],
            "all_sessions": [],
            "condition_count": 0,
            "image_count": 0,
            "candidate_epochs": [],
            "matrix_path": "",
            "can_open_matrix": False,
            "primary_action": "create",
            "primary_label": "候補レビューを作成",
            "embedding_coverage": None,
        }
    current = summarize_review_session(selected_session)
    other_sessions = [summarize_review_session(session) for session in sessions if int(session["id"]) != int(selected_session["id"])]
    planned_sessions = [item for item in other_sessions if item["status"] in {"planned", "prepared"}]
    return {
        "session": current["session"],
        "current": current,
        "other_sessions": other_sessions,
        "planned_sessions": planned_sessions,
        "all_sessions": [current] + other_sessions,
        "condition_count": current["condition_count"],
        "image_count": current["image_count"],
        "candidate_epochs": current["candidate_epochs"],
        "matrix_path": current["matrix_path"],
        "can_open_matrix": current["can_open_matrix"],
        "primary_action": current["primary_action"],
        "primary_label": current["primary_label"],
        "embedding_coverage": current["embedding_coverage"],
        "log_tail": current["log_tail"],
        "log_size": current["log_size"],
    }


def review_session_dir(session_id: int) -> Path:
    return settings.ROOT_DIR / "exports" / "review_sessions" / f"review_session_{session_id:06d}"


def review_session_output_dir(session_id: int) -> Path:
    return review_session_dir(session_id) / "images"


def review_session_log_tail(session: dict[str, Any], max_lines: int = 20) -> str:
    log_path = session.get("log_path") or ""
    if not log_path:
        return ""
    path = Path(str(log_path))
    if not path.exists():
        return ""
    data = path.read_bytes()
    text = decode_log_bytes(data)
    return "\n".join(text.splitlines()[-max_lines:])


def review_session_log_size(session: dict[str, Any]) -> int:
    log_path = session.get("log_path") or ""
    if not log_path:
        return 0
    path = Path(str(log_path))
    return path.stat().st_size if path.exists() else 0


def prepare_review_generation(session_id: int) -> dict[str, Any]:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
    if session is None:
        raise ValueError(f"レビューセッションが見つかりません: {session_id}")
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (session["job_id"],))
    if job is None:
        raise ValueError(f"Job not found: {session['job_id']}")
    environment = latest_environment()
    if environment is None:
        raise RuntimeError("sd-scripts環境が登録されていません。")
    sd_scripts_path = Path(environment["sd_scripts_path"])
    gen_img = sd_scripts_path / "gen_img.py"
    venv_python = Path(environment["venv_python_path"])
    if not sd_scripts_path.exists():
        raise RuntimeError(f"sd-scripts path が存在しません: {sd_scripts_path}")
    if not gen_img.exists():
        raise RuntimeError(f"gen_img.py が存在しません: {gen_img}")
    if not venv_python.exists():
        raise RuntimeError(f"venv python が存在しません: {venv_python}")
    base_model_path = Path(str(job["base_model_path"]))
    if not base_model_path.exists():
        raise RuntimeError(f"ベースモデルが存在しません: {base_model_path}")

    conditions = [dict(row) for row in fetch_all("SELECT * FROM review_session_conditions WHERE review_session_id = ? ORDER BY epoch, prompt_key, lora_weight, id", (session_id,))]
    if not conditions:
        raise RuntimeError("レビューセッションに生成条件がありません。")
    run_dir = review_session_dir(session_id)
    out_dir = review_session_output_dir(session_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    for lora_path, rows in group_conditions_by_lora(conditions).items():
        lora_file = Path(lora_path)
        if not lora_file.exists():
            raise RuntimeError(f"候補epochのLoRAファイルが存在しません: {lora_file}")
        epoch = rows[0].get("epoch")
        prompt_path = run_dir / f"review_prompts_epoch_{epoch or 'unknown'}_output_{rows[0].get('output_id')}.txt"
        prompt_path.write_text("\n".join(review_prompt_line(session_id, row) for row in rows) + "\n", encoding="utf-8")
        base_args = common_gen_img_args(
            venv_python=venv_python,
            gen_img=gen_img,
            base_model_path=base_model_path,
            out_dir=out_dir,
            model_family=str(job["model_family"] or ""),
            mixed_precision=str(environment["mixed_precision"] or ""),
            condition=rows[0],
        )
        commands.append(
            {
                "name": f"epoch_{epoch}_output_{rows[0].get('output_id')}",
                "epoch": epoch,
                "output_id": rows[0].get("output_id"),
                "prompt_file": str(prompt_path),
                "condition_count": len(rows),
                "argv": [
                    *base_args,
                    "--from_file",
                    str(prompt_path),
                    "--network_module",
                    "networks.lora",
                    "--network_weights",
                    str(lora_file),
                ],
            }
        )

    payload = {"commands": commands, "output_dir": str(out_dir), "preset_id": PRESET_ID}
    command_argv_path = run_dir / "command_argv.json"
    command_txt_path = run_dir / "command.txt"
    prompt_all_path = run_dir / "review_prompts_all.txt"
    prompt_all_path.write_text("\n".join(review_prompt_line(session_id, row) for row in conditions) + "\n", encoding="utf-8")
    command_argv_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    command_txt_path.write_text(review_command_text(payload), encoding="utf-8")
    log_path = run_dir / "review_preparation.log"
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE review_sessions
            SET status = 'prepared', run_dir = ?, output_dir = ?, prompt_file_path = ?,
                command_argv_json = ?, log_path = ?, updated_at = ?
            WHERE id = ?
            """,
            (str(run_dir), str(out_dir), str(prompt_all_path), json.dumps(payload, ensure_ascii=False), str(log_path), now, session_id),
        )
    return {"session_id": session_id, "run_dir": str(run_dir), "output_dir": str(out_dir), "commands": commands}


def start_review_preparation(session_id: int) -> int:
    reject_if_review_gpu_busy()
    prepare_review_generation(session_id)
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
    if session is None:
        raise ValueError(f"レビューセッションが見つかりません: {session_id}")
    payload = json.loads(session["command_argv_json"] or "{}")
    commands = payload.get("commands") or []
    if not commands:
        raise RuntimeError("生成対象の条件がありません。")
    environment = latest_environment()
    if environment is None:
        raise RuntimeError("sd-scripts環境が登録されていません。")
    sd_scripts_path = Path(environment["sd_scripts_path"])
    log_path = Path(session["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    archive_existing_log(log_path)
    start_time = utc_now()
    env = sd_scripts_subprocess_env()
    with connect() as conn:
        conn.execute(
            """
            UPDATE review_sessions
            SET status = 'running', generation_process_id = NULL, started_at = ?,
                ended_at = NULL, elapsed_seconds = NULL, return_code = NULL,
                generated_image_count = 0, imported_image_count = 0,
                scored_image_count = 0, error_message = NULL, updated_at = ?
            WHERE id = ?
            """,
            (start_time, start_time, session_id),
        )
    append_log_note(log_path, f"レビュー準備を開始します。commands={len(commands)}")
    append_log_note(log_path, f"First command: {commands[0].get('name') or 'gen_img.py'}")
    log_handle = log_path.open("ab")
    try:
        first_process = subprocess.Popen(
            commands[0]["argv"],
            cwd=str(sd_scripts_path),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            shell=False,
            env=env,
        )
    except Exception as exc:
        log_handle.close()
        end_time = utc_now()
        append_log_note(log_path, f"レビュー準備の開始に失敗しました: {exc}")
        with connect() as conn:
            conn.execute(
                """
                UPDATE review_sessions
                SET status = 'failed', generation_process_id = NULL,
                    return_code = -1, ended_at = ?, elapsed_seconds = ?,
                    error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (end_time, elapsed_seconds(start_time, end_time), str(exc), end_time, session_id),
            )
        raise
    with connect() as conn:
        conn.execute(
            "UPDATE review_sessions SET generation_process_id = ?, updated_at = ? WHERE id = ?",
            (first_process.pid, utc_now(), session_id),
        )
    thread = threading.Thread(
        target=monitor_review_generation,
        args=(session_id, first_process, commands, 0, log_handle, start_time, log_path, sd_scripts_path, env),
        daemon=True,
    )
    thread.start()
    return first_process.pid


def stop_review_preparation(session_id: int) -> None:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
    if session is None:
        raise ValueError(f"レビューセッションが見つかりません: {session_id}")
    end_time = utc_now()
    elapsed = elapsed_seconds(session["started_at"], end_time) if session["started_at"] else None
    pid = session["generation_process_id"]
    if pid:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    with connect() as conn:
        conn.execute(
            """
            UPDATE review_sessions
            SET status = 'stopped', generation_process_id = NULL,
                ended_at = ?, elapsed_seconds = ?, return_code = COALESCE(return_code, 4294967295),
                updated_at = ?
            WHERE id = ?
            """,
            (end_time, elapsed, end_time, session_id),
        )
    log_path = Path(str(session["log_path"] or review_session_dir(session_id) / "review_preparation.log"))
    append_log_note(log_path, "レビュー準備はユーザー操作で停止されました。")


def reconcile_stale_review_sessions() -> int:
    rows = fetch_all("SELECT * FROM review_sessions WHERE status = 'running'")
    fixed = 0
    for row in rows:
        session = dict(row)
        pid = session.get("generation_process_id")
        if pid and process_exists(int(pid)):
            continue
        end_time = utc_now()
        elapsed = elapsed_seconds(session.get("started_at"), end_time) if session.get("started_at") else None
        with connect() as conn:
            conn.execute(
                """
                UPDATE review_sessions
                SET status = 'stopped', generation_process_id = NULL,
                    ended_at = COALESCE(ended_at, ?),
                    elapsed_seconds = COALESCE(elapsed_seconds, ?),
                    return_code = COALESCE(return_code, 4294967295),
                    error_message = COALESCE(error_message, 'レビュー準備プロセスが見つかりませんでした。'),
                    updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (end_time, elapsed, end_time, session["id"]),
            )
        log_path = Path(str(session.get("log_path") or review_session_dir(int(session["id"])) / "review_preparation.log"))
        append_log_note(log_path, "running status reconciled: review process was not found. Marking session stopped.")
        fixed += 1
    return fixed


def reject_if_review_gpu_busy() -> None:
    reconcile_stale_embedding_jobs()
    running_job = fetch_one("SELECT id FROM training_jobs WHERE status = 'running' LIMIT 1")
    if running_job:
        raise RuntimeError(f"学習ジョブ #{running_job['id']} が実行中です。")
    running_generation = fetch_one("SELECT id FROM validation_generation_runs WHERE status = 'running' LIMIT 1")
    if running_generation:
        raise RuntimeError(f"検証画像生成 #{running_generation['id']} が実行中です。")
    running_embedding = fetch_one("SELECT id FROM embedding_jobs WHERE status = 'running' LIMIT 1")
    if running_embedding:
        raise RuntimeError(f"Embedding Job #{running_embedding['id']} が実行中です。")
    running_review = fetch_one("SELECT id FROM review_sessions WHERE status = 'running' LIMIT 1")
    if running_review:
        raise RuntimeError(f"レビュー準備 #{running_review['id']} が実行中です。")


def monitor_review_generation(
    session_id: int,
    process: subprocess.Popen[bytes],
    commands: list[dict[str, Any]],
    command_index: int,
    log_handle,
    start_time_text: str,
    log_path: Path,
    sd_scripts_path: Path,
    env: dict[str, str],
) -> None:
    return_code = process.wait()
    current = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
    if current is not None and current["status"] == "stopped":
        log_handle.close()
        return
    if return_code == 0 and command_index + 1 < len(commands):
        next_process = subprocess.Popen(
            commands[command_index + 1]["argv"],
            cwd=str(sd_scripts_path),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            shell=False,
            env=env,
        )
        with connect() as conn:
            conn.execute("UPDATE review_sessions SET generation_process_id = ?, updated_at = ? WHERE id = ?", (next_process.pid, utc_now(), session_id))
        monitor_review_generation(session_id, next_process, commands, command_index + 1, log_handle, start_time_text, log_path, sd_scripts_path, env)
        return

    log_handle.close()
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
    if session is None:
        return
    end_time = utc_now()
    status = "completed" if return_code == 0 else "failed"
    generated = count_generated_images(Path(session["output_dir"]))
    imported = 0
    error_message = session["error_message"] or ""
    if status == "completed":
        try:
            import_review_session_images(session_id)
            imported = int(fetch_one("SELECT COUNT(*) AS count FROM review_session_images WHERE review_session_id = ?", (session_id,))["count"] or 0)
            auto_embedding_after_review_generation(session_id, log_path)
            scored = auto_machine_review_after_review_generation(session_id, log_path)
            matrix_path = write_review_matrix(session_id)
        except Exception as exc:
            status = "failed"
            error_message = f"{error_message}\nImport failed: {exc}".strip()
            append_log_note(log_path, f"generated image import failed: {exc}")
        else:
            with connect() as conn:
                conn.execute(
                    "UPDATE review_sessions SET scored_image_count = ?, matrix_path = ?, updated_at = ? WHERE id = ?",
                    (scored, matrix_path, utc_now(), session_id),
                )
    with connect() as conn:
        conn.execute(
            """
            UPDATE review_sessions
            SET status = ?, generation_process_id = NULL, return_code = ?,
                ended_at = ?, elapsed_seconds = ?, generated_image_count = ?,
                imported_image_count = ?, error_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, return_code, end_time, elapsed_seconds(start_time_text, end_time), generated, imported, error_message, end_time, session_id),
        )


def import_review_session_images(session_id: int) -> int:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
    if session is None:
        raise ValueError(f"レビューセッションが見つかりません: {session_id}")
    output_dir = Path(session["output_dir"]) if session["output_dir"] else review_session_output_dir(session_id)
    conditions = {int(row["id"]): dict(row) for row in fetch_all("SELECT * FROM review_session_conditions WHERE review_session_id = ?", (session_id,))}
    hashes = {str(row["condition_hash"]): dict(row) for row in conditions.values()}
    imported = 0
    now = utc_now()
    for path in sorted(output_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        try:
            verify_image_file(path)
        except ValueError:
            continue
        condition = condition_for_review_file(path, conditions, hashes)
        if condition is None:
            continue
        existing = fetch_one("SELECT id FROM review_session_images WHERE review_session_id = ? AND image_path = ?", (session_id, str(path)))
        if existing is not None:
            continue
        sha256, _metadata_error = safe_sha256_file(path)
        width, height = image_size(path)
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO review_session_images(
                    review_session_id, condition_id, job_id, epoch, output_id,
                    prompt_key, prompt_role, seed, lora_weight, image_path,
                    file_size, sha256, width, height, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    condition["id"],
                    session["job_id"],
                    condition["epoch"],
                    condition["output_id"],
                    condition["prompt_key"],
                    condition["prompt_role"],
                    condition["seed"],
                    condition["lora_weight"],
                    str(path),
                    path.stat().st_size,
                    sha256,
                    width,
                    height,
                    now,
                    now,
                ),
            )
            image_id = int(cur.lastrowid)
            conn.execute(
                """
                UPDATE review_session_conditions
                SET image_path = ?, image_id = ?, status = 'generated', updated_at = ?
                WHERE id = ?
                """,
                (str(path), image_id, now, condition["id"]),
            )
        imported += 1
    return imported


def wait_for_review_session_embedding_job(embedding_job_id: int, log_path: Path) -> None:
    deadline = time.monotonic() + 60 * 60
    while time.monotonic() < deadline:
        row = fetch_one(
            "SELECT status, ready_count, failed_count, processed_count, total_count, error_message FROM embedding_jobs WHERE id = ?",
            (embedding_job_id,),
        )
        if row is None:
            append_log_note(log_path, f"Auto embedding: job #{embedding_job_id} disappeared.")
            return
        if row["status"] not in {"planned", "running"}:
            append_log_note(
                log_path,
                "Auto embedding finished: "
                f"status={row['status']} ready={row['ready_count']} "
                f"failed={row['failed_count']} processed={row['processed_count']}/{row['total_count']} "
                f"error={row['error_message'] or ''}",
            )
            return
        time.sleep(2)
    append_log_note(log_path, f"Auto embedding timed out: job #{embedding_job_id}.")


def run_review_session_embedding_job(session_id: int, log_path: Path, reason: str) -> dict[str, Any]:
    from app.services.embedding_service import create_embedding_job, embedding_coverage, start_embedding_job

    embedding_job_id = create_embedding_job("review_session", session_id, recompute="missing")
    embedding_job = fetch_one("SELECT total_count FROM embedding_jobs WHERE id = ?", (embedding_job_id,))
    total = int(embedding_job["total_count"] or 0) if embedding_job else 0
    if total:
        append_log_note(log_path, f"Auto embedding ({reason}): review_session #{session_id}, {total} item(s).")
        start_embedding_job(embedding_job_id)
        wait_for_review_session_embedding_job(embedding_job_id, log_path)
    else:
        append_log_note(log_path, f"Auto embedding ({reason}): review_session #{session_id}, no missing/stale item.")
    coverage = embedding_coverage("review_session", session_id)
    append_log_note(
        log_path,
        "Embedding coverage: "
        f"ready={coverage.get('ready')} stale={coverage.get('stale')} "
        f"missing={coverage.get('missing')} not_computed={coverage.get('not_computed')} total={coverage.get('total')}.",
    )
    return coverage


def auto_embedding_after_review_generation(session_id: int, log_path: Path) -> None:
    try:
        run_review_session_embedding_job(session_id, log_path, "after generation")
    except Exception as exc:
        append_log_note(log_path, f"Embedding auto step failed: {exc}")


def auto_machine_review_after_review_generation(session_id: int, log_path: Path) -> int:
    try:
        from app.services.machine_review import run_machine_review
        from app.services.embedding_service import embedding_coverage
    except Exception as exc:
        append_log_note(log_path, f"Machine Review auto step skipped: import failed: {exc}")
        return 0
    try:
        coverage = embedding_coverage("review_session", session_id)
        if int(coverage.get("ready") or 0) < int(coverage.get("total") or 0):
            append_log_note(
                log_path,
                "Machine Review preflight: review_session embedding is not fully ready; recomputing stale/missing items.",
            )
            coverage = run_review_session_embedding_job(session_id, log_path, "before machine review")
        if int(coverage.get("ready") or 0) < int(coverage.get("total") or 0):
            append_log_note(
                log_path,
                "Machine Review auto step skipped: "
                f"embedding coverage ready={coverage.get('ready')} total={coverage.get('total')}.",
            )
            return 0
        result = run_machine_review("review_session_images", session_id)
        scored = int(result.get("scored") or 0)
        link_machine_scores_to_review_images(session_id)
        append_log_note(log_path, f"Auto machine review completed: scored={result.get('scored')} failed={result.get('failed')}.")
        return scored
    except Exception as exc:
        append_log_note(log_path, f"Machine Review auto step failed: {exc}")
        return 0


def link_machine_scores_to_review_images(session_id: int) -> None:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
    if session is None:
        return
    rows = fetch_all(
        """
        SELECT rsi.id AS image_id, mrs.id AS score_id
        FROM review_session_images rsi
        JOIN machine_review_scores mrs
          ON mrs.source_type = 'review_session_image'
         AND mrs.source_id = rsi.id
        WHERE rsi.review_session_id = ?
          AND mrs.job_id = ?
        ORDER BY mrs.updated_at DESC, mrs.id DESC
        """,
        (session_id, session["job_id"]),
    )
    now = utc_now()
    with connect() as conn:
        for row in rows:
            conn.execute(
                "UPDATE review_session_images SET machine_review_score_id = ?, updated_at = ? WHERE id = ?",
                (row["score_id"], now, row["image_id"]),
            )


def write_review_matrix(session_id: int) -> str:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
    if session is None:
        raise ValueError(f"レビューセッションが見つかりません: {session_id}")
    job = fetch_one("SELECT id, name, adopted_epoch FROM training_jobs WHERE id = ?", (session["job_id"],))
    conditions = [dict(row) for row in fetch_all("SELECT * FROM review_session_conditions WHERE review_session_id = ? ORDER BY prompt_key, lora_weight, seed, epoch, id", (session_id,))]
    images = [dict(row) for row in fetch_all("SELECT * FROM review_session_images WHERE review_session_id = ? AND deleted_at IS NULL", (session_id,))]
    scores = review_session_scores(session_id)
    by_condition = {int(row["condition_id"]): row for row in images if row.get("condition_id") is not None}
    prompt_keys = sorted({str(row["prompt_key"] or "-") for row in conditions})
    weights = sorted({float(row["lora_weight"] or 0) for row in conditions})
    epochs = sorted({int(row["epoch"] or 0) for row in conditions})
    selected_epoch = int(job["adopted_epoch"]) if job and job["adopted_epoch"] is not None else None
    matrix_path = review_session_dir(session_id) / "review_matrix.html"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    project_id = int(session["project_id"]) if "project_id" in session.keys() and session["project_id"] else None
    lines = [
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">",
        f"<title>レビューMatrix #{session_id}</title>",
        review_matrix_style(),
        "</head><body>",
        review_matrix_navigation(session_id, int(session["job_id"]), project_id),
        f"<h1>候補epochレビューMatrix #{session_id}</h1>",
        f"<p class=\"muted\">Job #{int(session['job_id'])} {html.escape(str(job['name'] if job else ''))}</p>",
        "<p class=\"notice\">採用前の候補epoch比較用Matrixです。機械補助レビューは補助情報であり、最終判断は人間評価を優先してください。</p>",
        matrix_display_controls(),
        "<div class=\"summary-grid\">",
        summary_card("候補epoch", ", ".join(str(epoch) for epoch in epochs) or "-"),
        summary_card("採用epoch", str(selected_epoch) if selected_epoch is not None else "-"),
        summary_card("条件数", str(len(conditions))),
        summary_card("登録画像", str(len(images))),
        summary_card("機械補助レビュー", str(len(scores))),
        "</div>",
    ]
    for prompt_key in prompt_keys:
        prompt_conditions = [row for row in conditions if str(row["prompt_key"] or "-") == prompt_key]
        prompt_text = prompt_conditions[0].get("prompt") if prompt_conditions else ""
        lines.append(f"<h2>{html.escape(prompt_key)}</h2>")
        lines.append(f"<p>{html.escape(str(prompt_text or ''))}</p>")
        for weight in weights:
            weight_conditions = [row for row in prompt_conditions if float(row["lora_weight"] or 0) == weight]
            if not weight_conditions:
                continue
            lines.append(f"<h3>weight {weight:g}</h3>")
            lines.append("<table><thead><tr><th>条件</th>")
            for epoch in epochs:
                marker = " <span class=\"selected-marker\">採用中</span>" if selected_epoch == epoch else ""
                selected_class = " class=\"selected-epoch\"" if selected_epoch == epoch else ""
                lines.append(f"<th{selected_class}>epoch {epoch}{marker}</th>")
            lines.append("</tr></thead><tbody>")
            lines.append("<tr>")
            lines.append(f"<th>{html.escape(prompt_key)}<br>seed {int(weight_conditions[0]['seed'])}<br>weight {weight:g}</th>")
            for epoch in epochs:
                condition = next((row for row in weight_conditions if int(row["epoch"] or 0) == epoch), None)
                if condition is None:
                    lines.append("<td class=\"missing\">条件なし</td>")
                    continue
                image = by_condition.get(int(condition["id"]))
                cell_class = "epoch-cell selected-epoch" if selected_epoch == epoch else "epoch-cell"
                lines.append(f"<td class=\"{cell_class}\">")
                if image is None:
                    lines.append("<div class=\"missing\">画像未登録</div>")
                    lines.append(f"<div class=\"muted\">条件 #{int(condition['id'])}</div>")
                else:
                    image_path = Path(str(image["image_path"]))
                    src = path_to_review_matrix_relative(matrix_path, image_path)
                    lines.append(f"<img class=\"matrix-image\" src=\"{html.escape(src)}\" alt=\"review image\">")
                    lines.append(review_image_caption(condition, image))
                    lines.append(matrix_machine_score(scores.get(int(image["id"]))))
                lines.append("</td>")
            lines.append("</tr></tbody></table>")
    lines.append(review_matrix_navigation(session_id, int(session["job_id"]), project_id))
    lines.append(matrix_display_script())
    lines.append("</body></html>")
    matrix_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(matrix_path)


def review_session_scores(session_id: int) -> dict[int, dict[str, Any]]:
    rows = fetch_all(
        """
        SELECT mrs.*
        FROM machine_review_scores mrs
        JOIN review_session_images rsi ON rsi.id = mrs.source_id
        WHERE mrs.source_type = 'review_session_image'
          AND rsi.review_session_id = ?
        ORDER BY mrs.updated_at DESC, mrs.id DESC
        """,
        (session_id,),
    )
    scores: dict[int, dict[str, Any]] = {}
    for row in rows:
        source_id = row["source_id"]
        if source_id is not None:
            scores.setdefault(int(source_id), dict(row))
    return scores


def review_image_caption(condition: dict[str, Any], image: dict[str, Any]) -> str:
    return (
        "<div class=\"muted\">"
        f"image #{int(image['id'])} / condition #{int(condition['id'])}<br>"
        f"epoch {int(condition['epoch'])} / seed {int(condition['seed'])} / weight {float(condition['lora_weight']):g}<br>"
        f"{html.escape(Path(str(image['image_path'])).name)}"
        "</div>"
    )


def review_matrix_navigation(session_id: int, job_id: int, project_id: int | None = None) -> str:
    parts = [
        "<div class=\"matrix-actions\">",
        f"<a class=\"button\" href=\"/review-sessions/{session_id}\">レビューセッションへ戻る</a>",
        f"<a class=\"button\" href=\"/jobs/{job_id}#review-preparation\">学習ジョブへ戻る</a>",
    ]
    if project_id:
        parts.append(f"<a class=\"button\" href=\"/projects/{project_id}\">Projectへ戻る</a>")
    parts.append("<button class=\"button secondary\" type=\"button\" onclick=\"window.close()\">閉じる</button>")
    parts.append("</div>")
    return "".join(parts)


def summary_card(label: str, value: str) -> str:
    return f"<div class=\"summary-card\"><div class=\"muted\">{html.escape(label)}</div><strong>{html.escape(value)}</strong></div>"


def review_matrix_style() -> str:
    return (
        "<style>"
        "body{font-family:'Segoe UI','Yu Gothic UI',sans-serif;background:#f6f7f4;color:#20231f;margin:24px;overflow-x:auto}"
        "table{border-collapse:collapse;width:max-content;min-width:100%;margin:12px 0 28px;background:white}"
        "th,td{border:1px solid #d8ddd4;padding:8px;vertical-align:top}th{background:#eef2eb;min-width:220px}"
        ".epoch-cell{min-width:300px}.missing{color:#8b4f39;font-weight:700}.notice{background:#eef5ef;border:1px solid #cfd8d1;border-radius:6px;padding:10px}"
        "img.matrix-image{width:auto;max-width:none;border-radius:6px;display:block;margin-bottom:6px;cursor:zoom-in}.muted{color:#657064;font-size:12px}"
        ".selected-epoch{background:#eef8f1;box-shadow:inset 0 0 0 2px #2f7668}.selected-marker{display:inline-flex;margin-left:6px;padding:2px 6px;border-radius:999px;background:#2f7668;color:white;font-size:11px;font-weight:700}"
        ".matrix-actions{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 18px}.button{display:inline-flex;align-items:center;min-height:34px;padding:7px 12px;border-radius:6px;background:#2f7668;color:white;text-decoration:none;font-weight:700;border:0;cursor:pointer;font:inherit}.button.secondary{background:#dce4df;color:#20231f}"
        ".summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:10px 0 20px}.summary-card{background:white;border:1px solid #d8ddd4;border-radius:6px;padding:10px}.summary-card strong{font-size:20px}"
        ".machine-score{display:grid;gap:4px;margin:8px 0;padding:8px;border:1px solid #d8ddd4;border-radius:6px;background:#f8faf7;font-size:13px}.machine-score .badges{display:flex;flex-wrap:wrap;gap:6px}.badge{display:inline-flex;align-items:center;justify-content:center;min-width:56px;padding:3px 8px;border-radius:6px;background:#dce4df;font-weight:700}.badge.low,.badge.low_confidence,.badge.unavailable,.badge.unknown{background:#dce4df}.badge.primary_candidate,.badge.secondary_candidate{background:#c6e7d8}.badge.possible_overfit,.badge.high{background:#f0c2c2}"
        + matrix_display_style()
        + "</style>"
    )


def path_to_review_matrix_relative(matrix_path: Path, image_path: Path) -> str:
    try:
        return os.path.relpath(image_path, start=matrix_path.parent).replace("\\", "/")
    except ValueError:
        return str(image_path)


def condition_for_review_file(path: Path, conditions: dict[int, dict[str, Any]], hashes: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    stem = path.stem
    id_match = re.search(r"rc(\d+)", stem)
    if id_match:
        condition = conditions.get(int(id_match.group(1)))
        if condition:
            return condition
    for condition_hash, condition in hashes.items():
        if condition_hash[:12] in stem or condition_hash in stem:
            return condition
    return None


def append_log_note(log_path: Path, message: str) -> None:
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(f"\n[LoRA-Studio] {message}\n")
    except OSError:
        pass


def group_conditions_by_lora(conditions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in conditions:
        grouped.setdefault(str(row["lora_path"]), []).append(row)
    return grouped


def review_prompt_line(session_id: int, row: dict[str, Any]) -> str:
    filename = review_output_filename(session_id, row)
    parts = [
        row.get("prompt") or "",
        "--n",
        row.get("negative_prompt") or "",
        "--d",
        str(int(row["seed"])),
        "--w",
        str(int(row["width"] or WIDTH)),
        "--h",
        str(int(row["height"] or HEIGHT)),
        "--s",
        str(int(row["steps"] or STEPS)),
        "--l",
        f"{float(row['cfg_scale'] or CFG_SCALE):g}",
        "--am",
        f"{float(row['lora_weight'] or 0):g}",
        "--f",
        filename,
    ]
    return " ".join(parts)


def review_output_filename(session_id: int, row: dict[str, Any]) -> str:
    prompt_key = sanitize_filename(str(row.get("prompt_key") or "prompt"))
    weight = f"{float(row['lora_weight'] or 0):g}".replace(".", "p").replace("-", "m")
    epoch = int(row["epoch"] or 0)
    return f"rs{session_id:06d}_rc{int(row['id']):06d}_{str(row['condition_hash'])[:12]}_e{epoch:06d}_{prompt_key}_seed{int(row['seed'])}_w{weight}_nohires.png"


def review_command_text(payload: dict[str, Any]) -> str:
    lines = ["# LoRA-Studio レビュー準備で生成", ""]
    for command in payload["commands"]:
        lines.append(f"## {command['name']} ({command['condition_count']} conditions)")
        lines.append(" ".join(json.dumps(part, ensure_ascii=False) for part in command["argv"]))
        lines.append("")
    return "\n".join(lines)


def ensure_candidate_review_plan(job_id: int, *, force: bool = False) -> dict[str, Any] | None:
    existing = latest_review_session(job_id)
    if existing and not force:
        return existing
    if existing and force and existing.get("status") == "running":
        raise RuntimeError("レビュー準備が実行中です。完了後に再作成してください。")

    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    outputs = model_outputs_by_epoch(job_id)
    if not outputs:
        return None
    epochs = candidate_review_epochs(job_id, set(outputs.keys()))
    if not epochs:
        return None

    project = project_context(job)
    reference_set_id = project.get("reference_set_id")
    reference_set_version_id = project.get("reference_set_version_id")
    snapshot = preset_snapshot()
    conditions = build_conditions(job, outputs, epochs, snapshot)
    now = utc_now()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO review_sessions(
                job_id, project_id, reference_set_id, reference_set_version_id,
                dataset_id, dataset_version_id, name, preset_id, preset_snapshot_json,
                candidate_epochs_json, prompt_keys_json, weights_json, seed,
                expected_image_count, status, created_at, updated_at, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?, '')
            """,
            (
                job_id,
                job["project_id"],
                reference_set_id,
                reference_set_version_id,
                job["dataset_id"],
                job["dataset_version_id"],
                f"Job #{job_id} {PRESET_NAME}",
                PRESET_ID,
                json.dumps(snapshot, ensure_ascii=False),
                json.dumps(epochs, ensure_ascii=False),
                json.dumps([prompt["key"] for prompt in PROMPTS], ensure_ascii=False),
                json.dumps(WEIGHTS, ensure_ascii=False),
                SEED,
                len(conditions),
                now,
                now,
            ),
        )
        session_id = int(cur.lastrowid)
        for order, condition in enumerate(conditions, start=1):
            conn.execute(
                """
                INSERT INTO review_session_conditions(
                    review_session_id, job_id, epoch, output_id, lora_path,
                    prompt_key, prompt_role, prompt, negative_prompt, seed,
                    lora_weight, hires_enabled, width, height, sampler, steps,
                    cfg_scale, condition_hash, expected_order, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    job_id,
                    condition["epoch"],
                    condition["output_id"],
                    condition["lora_path"],
                    condition["prompt_key"],
                    condition["prompt_role"],
                    condition["prompt"],
                    condition["negative_prompt"],
                    condition["seed"],
                    condition["lora_weight"],
                    condition["width"],
                    condition["height"],
                    condition["sampler"],
                    condition["steps"],
                    condition["cfg_scale"],
                    condition["condition_hash"],
                    order,
                    now,
                    now,
                ),
            )
    return latest_review_session(job_id)


def model_outputs_by_epoch(job_id: int) -> dict[int, dict[str, Any]]:
    rows = fetch_all(
        """
        SELECT * FROM training_outputs
        WHERE job_id = ? AND file_type = 'model' AND deleted_at IS NULL
          AND epoch IS NOT NULL
        ORDER BY epoch, selected DESC, id DESC
        """,
        (job_id,),
    )
    outputs: dict[int, dict[str, Any]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        path = Path(str(row["file_path"]))
        if epoch not in outputs and path.exists():
            outputs[epoch] = dict(row)
    return outputs


def candidate_review_epochs(job_id: int, output_epochs: set[int]) -> list[int]:
    candidates = ensure_epoch_candidates(job_id)
    base_epochs: set[int] = set()
    for row in candidates:
        if row.get("candidate_label") in {"primary", "secondary", "check"} and row.get("epoch") is not None:
            epoch = int(row["epoch"])
            base_epochs.update({epoch - 1, epoch, epoch + 1})
    if not base_epochs:
        job = fetch_one("SELECT adopted_epoch FROM training_jobs WHERE id = ?", (job_id,))
        if job and job["adopted_epoch"] is not None:
            epoch = int(job["adopted_epoch"])
            base_epochs.update({epoch - 1, epoch, epoch + 1})
    return sorted(epoch for epoch in base_epochs if epoch in output_epochs and epoch > 0)


def project_context(job: Any) -> dict[str, Any]:
    if not job["project_id"]:
        return {}
    row = fetch_one(
        "SELECT default_reference_set_id, default_reference_set_version_id FROM lora_projects WHERE id = ?",
        (job["project_id"],),
    )
    if not row:
        return {}
    reference_set_id = row["default_reference_set_id"]
    reference_set_version_id = row["default_reference_set_version_id"]
    if reference_set_id and not reference_set_version_id:
        version = fetch_one(
            "SELECT id FROM reference_set_versions WHERE reference_set_id = ? ORDER BY version_no DESC, id DESC LIMIT 1",
            (reference_set_id,),
        )
        reference_set_version_id = version["id"] if version else None
    return {"reference_set_id": reference_set_id, "reference_set_version_id": reference_set_version_id}


def build_conditions(job: Any, outputs: dict[int, dict[str, Any]], epochs: list[int], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    trigger = job["trigger_word_at_creation"] or ""
    if not trigger and job["dataset_id"]:
        dataset = fetch_one("SELECT trigger_word FROM datasets WHERE id = ?", (job["dataset_id"],))
        trigger = dataset["trigger_word"] if dataset and dataset["trigger_word"] else ""
    conditions: list[dict[str, Any]] = []
    for epoch in epochs:
        output = outputs[epoch]
        for prompt in snapshot["prompts"]:
            prompt_text = prompt["prompt"].format(trigger=trigger).strip(", ")
            for weight in snapshot["weights"]:
                condition = {
                    "epoch": epoch,
                    "output_id": output["id"],
                    "lora_path": output["file_path"],
                    "prompt_key": prompt["key"],
                    "prompt_role": prompt["role"],
                    "prompt": prompt_text,
                    "negative_prompt": snapshot["negative_prompt"],
                    "seed": snapshot["seed"],
                    "lora_weight": float(weight),
                    "width": snapshot["width"],
                    "height": snapshot["height"],
                    "sampler": snapshot["sampler"],
                    "steps": snapshot["steps"],
                    "cfg_scale": snapshot["cfg_scale"],
                }
                condition["condition_hash"] = condition_hash(job["id"], condition)
                conditions.append(condition)
    return conditions


def condition_hash(job_id: int, condition: dict[str, Any]) -> str:
    payload = {
        "job_id": job_id,
        "epoch": condition["epoch"],
        "output_id": condition["output_id"],
        "prompt_key": condition["prompt_key"],
        "seed": condition["seed"],
        "lora_weight": condition["lora_weight"],
        "hires_enabled": 0,
        "preset_id": PRESET_ID,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
