from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

from app import settings
from app.db import connect, fetch_all, fetch_one, latest_environment, utc_now
from app.services.image_store import verify_image_file
from app.services.output_collector import image_size, safe_sha256_file
from app.services.review_candidates import ensure_epoch_candidates
from app.services.training_runner import archive_existing_log, elapsed_seconds, sd_scripts_subprocess_env
from app.services.validation_generation import IMAGE_SUFFIXES, common_gen_img_args, count_generated_images, sanitize_filename


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


def latest_review_session(job_id: int) -> dict[str, Any] | None:
    row = fetch_one(
        """
        SELECT * FROM review_sessions
        WHERE job_id = ? AND preset_id = ?
        ORDER BY id DESC LIMIT 1
        """,
        (job_id, PRESET_ID),
    )
    return dict(row) if row else None


def review_session_summary(job_id: int) -> dict[str, Any]:
    session = latest_review_session(job_id)
    if session is None:
        return {
            "session": None,
            "condition_count": 0,
            "image_count": 0,
            "candidate_epochs": [],
            "matrix_path": "",
            "can_open_matrix": False,
            "embedding_coverage": None,
        }
    condition_count = fetch_one(
        "SELECT COUNT(*) AS c FROM review_session_conditions WHERE review_session_id = ?",
        (session["id"],),
    )
    image_count = fetch_one(
        "SELECT COUNT(*) AS c FROM review_session_images WHERE review_session_id = ? AND deleted_at IS NULL",
        (session["id"],),
    )
    try:
        candidate_epochs = json.loads(session.get("candidate_epochs_json") or "[]")
    except json.JSONDecodeError:
        candidate_epochs = []
    matrix_path = session.get("matrix_path") or ""
    try:
        from app.services.embedding_service import embedding_coverage

        embedding = embedding_coverage("review_session", int(session["id"]))
    except Exception:
        embedding = None
    return {
        "session": session,
        "condition_count": int(condition_count["c"] if condition_count else 0),
        "image_count": int(image_count["c"] if image_count else 0),
        "candidate_epochs": candidate_epochs,
        "matrix_path": matrix_path,
        "can_open_matrix": bool(matrix_path and Path(matrix_path).exists()),
        "embedding_coverage": embedding,
    }


def review_session_dir(session_id: int) -> Path:
    return settings.ROOT_DIR / "exports" / "review_sessions" / f"review_session_{session_id:06d}"


def review_session_output_dir(session_id: int) -> Path:
    return review_session_dir(session_id) / "images"


def prepare_review_generation(session_id: int) -> dict[str, Any]:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
    if session is None:
        raise ValueError(f"Review Session not found: {session_id}")
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
        raise RuntimeError("Review Sessionに生成条件がありません。")
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
        raise ValueError(f"Review Session not found: {session_id}")
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
    log_handle = log_path.open("ab")
    start_time = utc_now()
    env = sd_scripts_subprocess_env()
    first_process = subprocess.Popen(
        commands[0]["argv"],
        cwd=str(sd_scripts_path),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        shell=False,
        env=env,
    )
    with connect() as conn:
        conn.execute(
            """
            UPDATE review_sessions
            SET status = 'running', generation_process_id = ?, started_at = ?,
                ended_at = NULL, elapsed_seconds = NULL, return_code = NULL,
                generated_image_count = 0, imported_image_count = 0,
                scored_image_count = 0, error_message = NULL, updated_at = ?
            WHERE id = ?
            """,
            (first_process.pid, start_time, start_time, session_id),
        )
    thread = threading.Thread(
        target=monitor_review_generation,
        args=(session_id, first_process, commands, 0, log_handle, start_time, log_path, sd_scripts_path, env),
        daemon=True,
    )
    thread.start()
    return first_process.pid


def reject_if_review_gpu_busy() -> None:
    running_job = fetch_one("SELECT id FROM training_jobs WHERE status = 'running' LIMIT 1")
    if running_job:
        raise RuntimeError(f"学習ジョブ #{running_job['id']} が実行中です。")
    running_generation = fetch_one("SELECT id FROM validation_generation_runs WHERE status = 'running' LIMIT 1")
    if running_generation:
        raise RuntimeError(f"Validation生成 #{running_generation['id']} が実行中です。")
    running_embedding = fetch_one("SELECT id FROM embedding_jobs WHERE status = 'running' LIMIT 1")
    if running_embedding:
        raise RuntimeError(f"Embedding Job #{running_embedding['id']} が実行中です。")
    running_review = fetch_one("SELECT id FROM review_sessions WHERE status = 'running' LIMIT 1")
    if running_review:
        raise RuntimeError(f"Review Preparation #{running_review['id']} が実行中です。")


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
            imported = import_review_session_images(session_id)
            auto_embedding_after_review_generation(session_id, log_path)
            scored = auto_machine_review_after_review_generation(session_id, log_path)
        except Exception as exc:
            status = "failed"
            error_message = f"{error_message}\nImport failed: {exc}".strip()
            append_log_note(log_path, f"generated image import failed: {exc}")
        else:
            with connect() as conn:
                conn.execute("UPDATE review_sessions SET scored_image_count = ?, updated_at = ? WHERE id = ?", (scored, utc_now(), session_id))
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
        raise ValueError(f"Review Session not found: {session_id}")
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


def auto_embedding_after_review_generation(session_id: int, log_path: Path) -> None:
    try:
        from app.services.embedding_service import create_embedding_job, embedding_coverage
        from app.services.embedding_worker import run_embedding_job
    except Exception as exc:
        append_log_note(log_path, f"Embedding auto step skipped: import failed: {exc}")
        return
    try:
        embedding_job_id = create_embedding_job("review_session", session_id, recompute="missing")
        embedding_job = fetch_one("SELECT total_count FROM embedding_jobs WHERE id = ?", (embedding_job_id,))
        total = int(embedding_job["total_count"] or 0) if embedding_job else 0
        if total:
            append_log_note(log_path, f"Auto embedding: review_session #{session_id}, {total} item(s).")
            run_embedding_job(embedding_job_id)
        else:
            append_log_note(log_path, f"Auto embedding: review_session #{session_id}, no missing item.")
        coverage = embedding_coverage("review_session", session_id)
        append_log_note(log_path, f"Embedding coverage: ready={coverage.get('ready')} total={coverage.get('total')}.")
    except Exception as exc:
        append_log_note(log_path, f"Embedding auto step failed: {exc}")


def auto_machine_review_after_review_generation(session_id: int, log_path: Path) -> int:
    try:
        from app.services.machine_review import run_machine_review
    except Exception as exc:
        append_log_note(log_path, f"Machine Review auto step skipped: import failed: {exc}")
        return 0
    try:
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
    lines = ["# Generated by LoRA-Studio Review Preparation", ""]
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
        raise RuntimeError("Review Preparationが実行中です。完了後に再作成してください。")

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
