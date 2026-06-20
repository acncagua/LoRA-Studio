from __future__ import annotations

import html
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import settings
from app.db import connect, fetch_all, fetch_one, latest_environment, utc_now
from app.services.image_store import verify_image_file
from app.services.training_runner import process_exists, sd_scripts_subprocess_env
from app.services.validation_runs import (
    ensure_expected_conditions,
    update_validation_run_counts,
    validation_run_dir,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SD_SCRIPTS_SAMPLER_ALIASES = {
    "ddim": "ddim",
    "pndm": "pndm",
    "lms": "lms",
    "euler": "euler",
    "euler a": "euler_a",
    "euler_a": "euler_a",
    "heun": "heun",
    "dpm 2": "dpm_2",
    "dpm_2": "dpm_2",
    "dpm 2 a": "dpm_2_a",
    "dpm_2_a": "dpm_2_a",
    "dpmsolver": "dpmsolver",
    "dpmsolver++": "dpmsolver++",
    "dpmsingle": "dpmsingle",
    "k lms": "k_lms",
    "k_lms": "k_lms",
    "k euler": "k_euler",
    "k_euler": "k_euler",
    "k euler a": "k_euler_a",
    "k_euler_a": "k_euler_a",
    "k dpm 2": "k_dpm_2",
    "k_dpm_2": "k_dpm_2",
    "k dpm 2 a": "k_dpm_2_a",
    "k_dpm_2_a": "k_dpm_2_a",
}


def normalize_sd_scripts_sampler(value: Any) -> str:
    sampler = str(value or "").strip()
    if not sampler:
        return "euler_a"
    key = re.sub(r"\s+", " ", sampler.replace("-", " ").replace("_", " ")).lower()
    return SD_SCRIPTS_SAMPLER_ALIASES.get(key, sampler)


def generation_dir(run_id: int) -> Path:
    return validation_run_dir(run_id) / "generation"


def generation_output_dir(run_id: int) -> Path:
    return generation_dir(run_id) / "images"


def latest_generation_run(run_id: int) -> Any | None:
    return fetch_one(
        "SELECT * FROM validation_generation_runs WHERE validation_run_id = ? ORDER BY id DESC LIMIT 1",
        (run_id,),
    )


def prepare_validation_generation(run_id: int) -> dict[str, Any]:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None:
        raise ValueError(f"Validation Run not found: {run_id}")
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (run["job_id"],))
    if job is None:
        raise ValueError(f"Job not found: {run['job_id']}")
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

    selected_output = fetch_one("SELECT * FROM training_outputs WHERE id = ?", (run["selected_output_id"],)) if run["selected_output_id"] else None
    profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (run["selected_lora_profile_id"],)) if run["selected_lora_profile_id"] else None
    selected_lora_path = selected_output["file_path"] if selected_output else (profile["selected_model_path"] if profile else "")
    if not selected_lora_path or not Path(selected_lora_path).exists():
        raise RuntimeError("採用LoRAのファイルパスが見つかりません。")

    base_model_path = resolve_base_model_path(run, job, profile)
    if not base_model_path.exists():
        raise RuntimeError(f"ベースモデルが存在しません: {base_model_path}")

    conditions = [dict(row) for row in ensure_expected_conditions(run_id)]

    gen_dir = generation_dir(run_id)
    out_dir = generation_output_dir(run_id)
    gen_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_prompt_path = gen_dir / "validation_prompts_for_sd_scripts.txt"
    all_prompt_path.write_text(
        "\n".join(prompt_line(run_id, row, include_am=float(row["lora_weight"] or 0) != 0) for row in conditions)
        + ("\n" if conditions else ""),
        encoding="utf-8",
    )

    commands = []
    for hires_enabled in (False, True):
        group_rows = [row for row in conditions if bool(row["hires_enabled"]) == hires_enabled]
        if not group_rows:
            continue
        baseline = [row for row in group_rows if float(row["lora_weight"] or 0) == 0]
        lora_rows = [row for row in group_rows if float(row["lora_weight"] or 0) != 0]
        suffix = "hires" if hires_enabled else "nohires"
        baseline_name = "baseline_weight_0_hires" if hires_enabled else "baseline_weight_0_no_lora"
        lora_name = "lora_weights_hires" if hires_enabled else "lora_weights"
        base_args = common_gen_img_args(
            venv_python=venv_python,
            gen_img=gen_img,
            base_model_path=base_model_path,
            out_dir=out_dir,
            model_family=str(job["model_family"] or ""),
            mixed_precision=str(environment["mixed_precision"] or ""),
            condition=group_rows[0],
        )
        if baseline:
            prompt_path = gen_dir / f"validation_prompts_baseline_{suffix}_for_sd_scripts.txt"
            prompt_path.write_text(
                "\n".join(prompt_line(run_id, row, include_am=False) for row in baseline) + "\n",
                encoding="utf-8",
            )
            commands.append(
                {
                    "name": baseline_name,
                    "baseline_mode": "no_network_weights",
                    "prompt_file": str(prompt_path),
                    "condition_count": len(baseline),
                    "argv": [*base_args, "--from_file", str(prompt_path)],
                }
            )
        if lora_rows:
            prompt_path = gen_dir / f"validation_prompts_lora_{suffix}_for_sd_scripts.txt"
            prompt_path.write_text(
                "\n".join(prompt_line(run_id, row, include_am=True) for row in lora_rows) + "\n",
                encoding="utf-8",
            )
            commands.append(
                {
                    "name": lora_name,
                    "baseline_mode": "",
                    "prompt_file": str(prompt_path),
                    "condition_count": len(lora_rows),
                    "argv": [
                        *base_args,
                        "--from_file",
                        str(prompt_path),
                        "--network_module",
                        "networks.lora",
                        "--network_weights",
                        str(selected_lora_path),
                    ],
                }
            )

    command_payload = {
        "commands": commands,
        "baseline_mode": "no_network_weights",
        "skipped_hires_count": 0,
        "skipped_hires_message": "",
        "all_prompt_file": str(all_prompt_path),
        "output_dir": str(out_dir),
    }
    command_argv_path = gen_dir / "command_argv.json"
    command_txt_path = gen_dir / "command.txt"
    command_argv_path.write_text(json.dumps(command_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    command_txt_path.write_text(command_text(command_payload), encoding="utf-8")
    log_path = gen_dir / "generation.log"
    now = utc_now()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO validation_generation_runs(
                validation_run_id, status, command_argv_json, prompt_file_path,
                output_dir, log_path, generated_image_count, imported_image_count,
                error_message, created_at, updated_at
            )
            VALUES (?, 'planned', ?, ?, ?, ?, 0, 0, ?, ?, ?)
            """,
            (
                run_id,
                json.dumps(command_payload, ensure_ascii=False),
                str(all_prompt_path),
                str(out_dir),
                str(log_path),
                command_payload["skipped_hires_message"],
                now,
                now,
            ),
        )
        generation_id = int(cur.lastrowid)
    return {
        "generation_id": generation_id,
        "prompt_file": str(all_prompt_path),
        "command_argv": str(command_argv_path),
        "command_txt": str(command_txt_path),
        "output_dir": str(out_dir),
        "commands": commands,
        "skipped_hires_count": 0,
    }


def resolve_base_model_path(run: Any, job: Any, profile: Any | None) -> Path:
    candidates = [run["base_model"], profile["base_model"] if profile else "", job["base_model_path"]]
    for candidate in candidates:
        if candidate and Path(str(candidate)).exists():
            return Path(str(candidate))
    return Path(str(job["base_model_path"]))


def common_gen_img_args(
    venv_python: Path,
    gen_img: Path,
    base_model_path: Path,
    out_dir: Path,
    model_family: str,
    mixed_precision: str,
    condition: dict[str, Any] | None = None,
) -> list[str]:
    condition = condition or {}
    args = [
        str(venv_python),
        str(gen_img),
        "--ckpt",
        str(base_model_path),
        "--outdir",
        str(out_dir),
        "--W",
        str(int(condition.get("width") or 1024)),
        "--H",
        str(int(condition.get("height") or 1024)),
        "--scale",
        f"{float(condition.get('cfg_scale') or 7):g}",
        "--steps",
        str(int(condition.get("steps") or 28)),
        "--sampler",
        normalize_sd_scripts_sampler(condition.get("sampler")),
        "--no_preview",
    ]
    if model_family.upper() == "SDXL":
        args.append("--sdxl")
    if "bf16" in mixed_precision.lower():
        args.append("--bf16")
    else:
        args.append("--fp16")
    if condition.get("hires_enabled"):
        hires_scale = float(condition.get("hires_scale") or 2.0)
        first_stage_scale = 1.0 / hires_scale if hires_scale > 0 else 0.5
        args.extend(["--highres_fix_scale", f"{first_stage_scale:g}"])
        args.extend(["--highres_fix_steps", str(int(condition.get("steps") or 28))])
        strength = condition.get("hires_denoising_strength")
        if strength not in (None, ""):
            args.extend(["--highres_fix_strength", f"{float(strength):g}"])
        upscaler = str(condition.get("hires_upscaler") or "").strip()
        if upscaler.lower() == "latent":
            args.append("--highres_fix_latents_upscaling")
        elif upscaler:
            args.extend(["--highres_fix_upscaler", upscaler])
    return args


def prompt_line(run_id: int, row: dict[str, Any], include_am: bool) -> str:
    prompt = row.get("prompt") or row.get("webui_prompt") or ""
    negative = row.get("negative_prompt") or ""
    filename = output_filename(run_id, row)
    parts = [
        prompt,
        "--n",
        negative,
        "--d",
        str(int(row["seed"])),
        "--w",
        str(int(row["width"] or 1024)),
        "--h",
        str(int(row["height"] or 1024)),
        "--s",
        str(int(row["steps"] or 28)),
        "--l",
        f"{float(row['cfg_scale'] or 7):g}",
    ]
    if include_am:
        parts.extend(["--am", f"{float(row['lora_weight'] or 0):g}"])
    parts.extend(["--f", filename])
    return " ".join(parts)


def output_stem(run_id: int, row: dict[str, Any]) -> str:
    prompt_key = sanitize_filename(row.get("prompt_key") or "prompt")
    weight = f"{float(row['lora_weight'] or 0):g}".replace(".", "p").replace("-", "m")
    hires = "hires" if row.get("hires_enabled") else "nohires"
    return (
        f"vr{run_id:06d}_ec{int(row['id']):06d}_{str(row['condition_hash'])[:12]}_"
        f"{prompt_key}_seed{int(row['seed'])}_w{weight}_{hires}"
    )


def output_filename(run_id: int, row: dict[str, Any]) -> str:
    return f"{output_stem(run_id, row)}.png"


def sanitize_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("._-") or "prompt"


def command_text(payload: dict[str, Any]) -> str:
    lines = [
        "# Generated by LoRA-Studio",
        "# weight 0 baseline uses no network weights.",
        "# Hires conditions use sd-scripts highres fix. WebUI output is not expected to match exactly.",
        "",
    ]
    for command in payload["commands"]:
        lines.append(f"## {command['name']} ({command['condition_count']} conditions)")
        lines.append(" ".join(json.dumps(part, ensure_ascii=False) for part in command["argv"]))
        lines.append("")
    if payload.get("skipped_hires_count"):
        lines.append(f"# skipped hires conditions: {payload['skipped_hires_count']}")
    return "\n".join(lines)


def start_validation_generation(run_id: int) -> int:
    reject_if_gpu_busy()
    # 実行時に必ず最新のValidation条件・採用LoRA・出力ファイル名で生成ファイルを作り直す。
    # これにより、古いplanned生成Runや過去バージョンのpromptファイルを誤って再利用しない。
    prepare_validation_generation(run_id)
    generation = latest_generation_run(run_id)
    if generation is None:
        raise RuntimeError("Generation Runを作成できませんでした。")
    payload = json.loads(generation["command_argv_json"] or "{}")
    commands = payload.get("commands") or []
    if not commands:
        raise RuntimeError("生成対象の条件がありません。")
    environment = latest_environment()
    if environment is None:
        raise RuntimeError("sd-scripts環境が登録されていません。")
    sd_scripts_path = Path(environment["sd_scripts_path"])
    log_path = Path(generation["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    archive_existing_log(log_path)
    start_time = utc_now()
    log_handle = log_path.open("ab")
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
            UPDATE validation_generation_runs
            SET status = 'running', process_id = ?, started_at = ?, ended_at = NULL,
                elapsed_seconds = NULL, return_code = NULL, generated_image_count = 0,
                imported_image_count = 0, error_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                first_process.pid,
                start_time,
                payload.get("skipped_hires_message") or "",
                start_time,
                generation["id"],
            ),
        )
    thread = threading.Thread(
        target=monitor_generation,
        args=(int(generation["id"]), first_process, commands, 0, log_handle, start_time, log_path, sd_scripts_path, env),
        daemon=True,
    )
    thread.start()
    return first_process.pid


def start_validation_generation_sequence(run_ids: list[int]) -> int:
    unique_run_ids: list[int] = []
    seen: set[int] = set()
    for run_id in run_ids:
        if run_id in seen:
            continue
        seen.add(run_id)
        unique_run_ids.append(run_id)
    if not unique_run_ids:
        raise ValueError("画像生成する検証Runを選択してください。")
    start_validation_generation(unique_run_ids[0])
    remaining_run_ids = unique_run_ids[1:]
    if remaining_run_ids:
        thread = threading.Thread(
            target=_validation_generation_sequence_worker,
            args=(remaining_run_ids,),
            daemon=True,
        )
        thread.start()
    return len(unique_run_ids)


def start_validation_assist_sequence(run_ids: list[int]) -> int:
    unique_run_ids: list[int] = []
    seen: set[int] = set()
    for run_id in run_ids:
        if run_id in seen:
            continue
        seen.add(run_id)
        unique_run_ids.append(run_id)
    if not unique_run_ids:
        raise ValueError("Embedding計算する検証Runを選択してください。")
    if fetch_one("SELECT id FROM validation_generation_runs WHERE status = 'running' LIMIT 1"):
        raise RuntimeError("検証画像生成が実行中です。画像生成の完了後にEmbedding計算を開始してください。")
    if fetch_one("SELECT id FROM embedding_jobs WHERE status = 'running' LIMIT 1"):
        raise RuntimeError("Embedding Jobが実行中です。完了後に再実行してください。")
    if fetch_one("SELECT id FROM machine_review_jobs WHERE status = 'running' LIMIT 1"):
        raise RuntimeError("機械補助レビューJobが実行中です。完了後に再実行してください。")
    thread = threading.Thread(
        target=_validation_assist_sequence_worker,
        args=(unique_run_ids,),
        daemon=True,
    )
    thread.start()
    return len(unique_run_ids)


def _validation_assist_sequence_worker(run_ids: list[int]) -> None:
    for run_id in run_ids:
        log_path = validation_run_dir(run_id) / "generation" / "assist_sequence.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _run_validation_assist_for_run(run_id, log_path)
        except Exception as exc:
            append_generation_note(log_path, f"検証Run #{run_id} のEmbedding / 機械補助レビューに失敗しました: {exc}")


def _wait_for_background_job(table_name: str, job_id: int, timeout_seconds: int = 60 * 60 * 6) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        row = fetch_one(f"SELECT status FROM {table_name} WHERE id = ?", (job_id,))
        if row is None:
            return "missing"
        status = row["status"]
        if status in {"completed", "failed", "stopped"}:
            return status
        time.sleep(3)
    return "timeout"


def _run_validation_assist_for_run(run_id: int, log_path: Path) -> None:
    from app.services.embedding_service import create_embedding_job, start_embedding_job
    from app.services.machine_review import context_for_validation_run, create_machine_review_job, start_machine_review_job

    context = context_for_validation_run(run_id)
    targets: list[tuple[str, int]] = []
    if context.get("reference_set_version_id"):
        targets.append(("reference_set_version", int(context["reference_set_version_id"])))
    if context.get("dataset_version_id"):
        targets.append(("dataset_version", int(context["dataset_version_id"])))
    targets.append(("validation_run", run_id))

    append_generation_note(log_path, f"検証Run #{run_id}: Embedding計算を開始します。")
    for job_type, target_id in targets:
        embedding_job_id = create_embedding_job(job_type, target_id, recompute="missing")
        embedding_job = fetch_one("SELECT total_count FROM embedding_jobs WHERE id = ?", (embedding_job_id,))
        total = int(embedding_job["total_count"] or 0) if embedding_job else 0
        if not total:
            append_generation_note(log_path, f"Embedding: {job_type} #{target_id} は未計算対象がありません。")
            continue
        append_generation_note(log_path, f"Embedding Job #{embedding_job_id}: {job_type} #{target_id}, {total}件を開始します。")
        start_embedding_job(embedding_job_id)
        status = _wait_for_background_job("embedding_jobs", embedding_job_id)
        append_generation_note(log_path, f"Embedding Job #{embedding_job_id}: status={status}")
        if status != "completed":
            return

    append_generation_note(log_path, f"検証Run #{run_id}: 機械補助レビューを開始します。")
    review_job_id = create_machine_review_job("validation_run_images", run_id)
    start_machine_review_job(review_job_id)
    status = _wait_for_background_job("machine_review_jobs", review_job_id)
    append_generation_note(log_path, f"Machine Review Job #{review_job_id}: status={status}")


def _validation_generation_sequence_worker(run_ids: list[int]) -> None:
    for run_id in run_ids:
        while fetch_one("SELECT id FROM validation_generation_runs WHERE status = 'running' LIMIT 1"):
            time.sleep(5)
        try:
            start_validation_generation(run_id)
        except Exception as exc:
            log_path = validation_run_dir(run_id) / "generation" / "bulk_generation.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(f"[LoRA-Studio] 検証Run #{run_id} の画像生成を開始できませんでした: {exc}\n")
            continue
        while True:
            generation = latest_generation_run(run_id)
            if generation is None or generation["status"] in {"completed", "failed", "stopped"}:
                break
            time.sleep(5)


def reject_if_gpu_busy() -> None:
    running_job = fetch_one("SELECT id FROM training_jobs WHERE status = 'running' LIMIT 1")
    if running_job:
        raise RuntimeError(f"学習ジョブ #{running_job['id']} が実行中です。")
    running_generation = fetch_one("SELECT id FROM validation_generation_runs WHERE status = 'running' LIMIT 1")
    if running_generation:
        raise RuntimeError(f"Validation生成 #{running_generation['id']} が実行中です。")


def monitor_generation(
    generation_id: int,
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
    current = fetch_one("SELECT * FROM validation_generation_runs WHERE id = ?", (generation_id,))
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
            conn.execute(
                "UPDATE validation_generation_runs SET process_id = ?, updated_at = ? WHERE id = ?",
                (next_process.pid, utc_now(), generation_id),
            )
        monitor_generation(generation_id, next_process, commands, command_index + 1, log_handle, start_time_text, log_path, sd_scripts_path, env)
        return

    log_handle.close()
    generation = fetch_one("SELECT * FROM validation_generation_runs WHERE id = ?", (generation_id,))
    if generation is None:
        return
    run_id = int(generation["validation_run_id"])
    end_time = utc_now()
    elapsed = elapsed_seconds(start_time_text, end_time)
    status = "completed" if return_code == 0 else "failed"
    error_message = generation["error_message"] or ""
    imported = 0
    generated = count_generated_images(Path(generation["output_dir"]))
    if status == "completed":
        try:
            imported = import_generated_images(int(run_id), generation_id)
            auto_machine_review_after_generation(run_id, log_path)
            write_validation_matrix(run_id)
        except Exception as exc:
            status = "failed"
            error_message = f"{error_message}\nImport failed: {exc}".strip()
            with log_path.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(f"\n[LoRA-Studio] generated image import failed: {exc}\n")
    with connect() as conn:
        conn.execute(
            """
            UPDATE validation_generation_runs
            SET status = ?, process_id = NULL, return_code = ?, ended_at = ?,
                elapsed_seconds = ?, generated_image_count = ?,
                imported_image_count = ?, error_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, return_code, end_time, elapsed, generated, imported, error_message, end_time, generation_id),
        )
    update_validation_run_counts(run_id)


def stop_validation_generation(run_id: int) -> None:
    generation = fetch_one(
        "SELECT * FROM validation_generation_runs WHERE validation_run_id = ? AND status = 'running' ORDER BY id DESC LIMIT 1",
        (run_id,),
    )
    if generation is None:
        return
    end_time = utc_now()
    elapsed = elapsed_seconds(generation["started_at"], end_time)
    pid = generation["process_id"]
    if pid:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    with connect() as conn:
        conn.execute(
            """
            UPDATE validation_generation_runs
            SET status = 'stopped', process_id = NULL, ended_at = ?,
                elapsed_seconds = ?, return_code = COALESCE(return_code, 4294967295),
                updated_at = ?
            WHERE id = ?
            """,
            (end_time, elapsed, end_time, generation["id"]),
        )


def import_generated_images(run_id: int, generation_id: int | None = None) -> int:
    generation = fetch_one("SELECT * FROM validation_generation_runs WHERE id = ?", (generation_id,)) if generation_id else latest_generation_run(run_id)
    output_dir = Path(generation["output_dir"]) if generation else generation_output_dir(run_id)
    conditions = {int(row["id"]): dict(row) for row in fetch_all("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))}
    hashes = {row["condition_hash"]: dict(row) for row in conditions.values()}
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None:
        raise ValueError(f"Validation Run not found: {run_id}")
    imported = 0
    now = utc_now()
    for path in sorted(output_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        try:
            verify_image_file(path)
        except ValueError:
            continue
        condition = condition_for_generated_file(path, conditions, hashes)
        if condition is None:
            continue
        existing = fetch_one("SELECT id FROM validation_images WHERE validation_run_id = ? AND image_path = ?", (run_id, str(path)))
        if existing is not None:
            continue
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO validation_images(
                    job_id, selected_output_id, expected_condition_id,
                    validation_run_id, validation_preset_id, prompt_key, seed,
                    lora_weight, image_path, validation_type, prompt, negative_prompt,
                    base_model, sampler, steps, cfg_scale, width, height,
                    hires_enabled, hires_scale, lora_weights, seeds,
                    grid_image_flag, image_role, condition_hash, memo,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'sd_scripts_generated',
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'individual', ?, ?, ?, ?)
                """,
                (
                    run["job_id"],
                    run["selected_output_id"],
                    condition["id"],
                    run_id,
                    condition["validation_preset_id"],
                    condition["prompt_key"],
                    condition["seed"],
                    condition["lora_weight"],
                    str(path),
                    condition.get("prompt") or condition.get("webui_prompt") or "",
                    condition.get("negative_prompt") or "",
                    condition.get("base_model") or run["base_model"] or "",
                    condition["sampler"],
                    condition["steps"],
                    condition["cfg_scale"],
                    condition["width"],
                    condition["height"],
                    condition["hires_enabled"],
                    None,
                    str(condition["lora_weight"]),
                    str(condition["seed"]),
                    condition["condition_hash"],
                    f"sd-scripts generated / generation #{generation_id or '-'}",
                    now,
                    now,
                ),
            )
        imported += 1
    update_validation_run_counts(run_id)
    return imported


def reconcile_stale_validation_generations() -> int:
    rows = fetch_all("SELECT * FROM validation_generation_runs WHERE status = 'running' ORDER BY id")
    fixed = 0
    for row in rows:
        pid = row["process_id"]
        if pid and process_exists(int(pid)):
            continue
        run_id = int(row["validation_run_id"])
        generation_id = int(row["id"])
        output_dir = Path(row["output_dir"]) if row["output_dir"] else generation_output_dir(run_id)
        generated = count_generated_images(output_dir)
        log_tail = validation_generation_log_tail(run_id, max_lines=20)
        completed = generated > 0 and "done!" in log_tail.lower()
        imported = 0
        return_code = 0 if completed else -1
        status = "completed" if completed else "failed"
        now = utc_now()
        if completed:
            try:
                import_generated_images(run_id, generation_id)
                auto_machine_review_after_generation(run_id, Path(row["log_path"]) if row["log_path"] else None)
                write_validation_matrix(run_id)
            except Exception as exc:
                # Keep the generation completed if sd-scripts finished; record import trouble.
                with connect() as conn:
                    conn.execute(
                        """
                        UPDATE validation_generation_runs
                        SET error_message = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (f"Import after stale reconcile failed: {exc}", now, generation_id),
                    )
            imported_row = fetch_one(
                "SELECT COUNT(*) AS count FROM validation_images WHERE validation_run_id = ? AND image_role = 'individual'",
                (run_id,),
            )
            imported = int(imported_row["count"] if imported_row else 0)
        error_message = "" if completed else "Process disappeared before LoRA-Studio could observe completion."
        with connect() as conn:
            conn.execute(
                """
                UPDATE validation_generation_runs
                SET status = ?, process_id = NULL, return_code = ?,
                    generated_image_count = ?, imported_image_count = ?,
                    ended_at = COALESCE(ended_at, ?), updated_at = ?,
                    error_message = COALESCE(NULLIF(error_message, ''), ?)
                WHERE id = ?
                """,
                (status, return_code, generated, imported, now, now, error_message, generation_id),
            )
        update_validation_run_counts(run_id)
        fixed += 1
    return fixed


def condition_for_generated_file(path: Path, conditions: dict[int, dict[str, Any]], hashes: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    stem = path.stem
    id_match = re.search(r"ec(\d+)", stem)
    if id_match:
        condition = conditions.get(int(id_match.group(1)))
        if condition:
            return condition
    for condition_hash, condition in hashes.items():
        if condition_hash[:12] in stem or condition_hash in stem:
            return condition
    return None


def count_generated_images(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    return sum(1 for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def auto_machine_review_after_generation(run_id: int, log_path: Path | None = None) -> None:
    """Run the light mock-only assist path after sd-scripts validation generation."""
    try:
        from app.services.embedding_service import active_embedding_model, create_embedding_job
        from app.services.embedding_worker import run_embedding_job
        from app.services.machine_review import context_for_validation_run, run_machine_review
    except Exception as exc:
        append_generation_note(log_path, f"Machine assist auto step skipped: import failed: {exc}")
        return

    try:
        model = active_embedding_model()
        if (model.get("provider") or "mock") != "mock":
            append_generation_note(log_path, "Machine assist auto step skipped: active embedding provider is not mock.")
            return
        running_embedding = fetch_one("SELECT id FROM embedding_jobs WHERE status = 'running' LIMIT 1")
        if running_embedding:
            append_generation_note(log_path, f"Machine assist auto step skipped: Embedding Job #{running_embedding['id']} is running.")
            return

        context = context_for_validation_run(run_id)
        targets: list[tuple[str, int]] = []
        if context.get("reference_set_version_id"):
            targets.append(("reference_set_version", int(context["reference_set_version_id"])))
        targets.append(("validation_run", run_id))

        for job_type, target_id in targets:
            embedding_job_id = create_embedding_job(job_type, target_id, recompute="missing")
            embedding_job = fetch_one("SELECT total_count FROM embedding_jobs WHERE id = ?", (embedding_job_id,))
            total = int(embedding_job["total_count"] or 0) if embedding_job else 0
            if total:
                append_generation_note(log_path, f"Auto embedding: {job_type} #{target_id}, {total} item(s).")
                run_embedding_job(embedding_job_id)
            else:
                append_generation_note(log_path, f"Auto embedding: {job_type} #{target_id}, no missing item.")

        result = run_machine_review("validation_run_images", run_id)
        append_generation_note(log_path, f"Auto machine assist completed: scored={result.get('scored')} failed={result.get('failed')}.")
    except Exception as exc:
        append_generation_note(log_path, f"Machine assist auto step failed: {exc}")


def append_generation_note(log_path: Path | None, message: str) -> None:
    if not log_path:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(f"\n[LoRA-Studio] {message}\n")
    except OSError:
        pass


def write_validation_matrix(run_id: int) -> str:
    conditions = [dict(row) for row in fetch_all("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY prompt_key, hires_enabled, seed, lora_weight", (run_id,))]
    images = [dict(row) for row in fetch_all("SELECT * FROM validation_images WHERE validation_run_id = ? ORDER BY prompt_key, hires_enabled, seed, lora_weight, id", (run_id,))]
    score_by_image_id = validation_machine_score_map([run_id])
    images_by_condition: dict[int, list[dict[str, Any]]] = {}
    grid_images = []
    for image in images:
        if image["image_role"] == "grid":
            grid_images.append(image)
            continue
        if image["expected_condition_id"]:
            images_by_condition.setdefault(int(image["expected_condition_id"]), []).append(image)
    run_dir = validation_run_dir(run_id)
    matrix_path = run_dir / "validation_matrix.html"
    sections: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for condition in conditions:
        sections.setdefault((condition["prompt_key"] or "prompt", int(condition["hires_enabled"] or 0)), []).append(condition)
    has_hires_sections = any(key[1] for key in sections)
    lines = [
        "<!doctype html>",
        "<meta charset=\"utf-8\">",
        f"<title>検証Matrix #{run_id}</title>",
        "<style>body{font-family:'Segoe UI','Yu Gothic UI',sans-serif;background:#f6f7f4;color:#20231f;margin:24px;overflow-x:auto}table{border-collapse:collapse;width:max-content;min-width:100%;margin:16px 0;background:white}th,td{border:1px solid #d8ddd4;padding:8px;vertical-align:top}th{background:#eef2eb}.cell{min-width:300px}.missing{color:#8b4f39;font-weight:700}img.matrix-image{width:auto;max-width:none;border-radius:6px;display:block;margin-bottom:6px;cursor:zoom-in}.muted{color:#657064;font-size:12px}.matrix-actions{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 18px}.button{display:inline-flex;align-items:center;min-height:34px;padding:7px 12px;border-radius:6px;background:#2f7668;color:white;text-decoration:none;font-weight:700;border:0;cursor:pointer;font:inherit}.button.secondary{background:#dce4df;color:#20231f}.machine-score{display:grid;gap:4px;margin:8px 0;padding:8px;border:1px solid #d8ddd4;border-radius:6px;background:#f8faf7;font-size:13px}.machine-score .badges{display:flex;flex-wrap:wrap;gap:6px}.badge{display:inline-flex;align-items:center;justify-content:center;min-width:56px;padding:3px 8px;border-radius:6px;background:#dce4df;font-weight:700}.badge.low,.badge.low_confidence,.badge.unavailable,.badge.unknown{background:#dce4df}.badge.primary_candidate,.badge.secondary_candidate{background:#c6e7d8}.badge.possible_overfit,.badge.high{background:#f0c2c2}.matrix-review{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr)) auto auto;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid #d8ddd4;align-items:end}.matrix-review label{display:grid;gap:3px;font-size:12px;color:#4a554f}.matrix-review input,.matrix-review select,.matrix-review textarea{box-sizing:border-box;width:100%;border:1px solid #cfd8d1;border-radius:5px;padding:5px;font:inherit}.matrix-review textarea{min-height:34px;resize:vertical}.matrix-review button{border:0;border-radius:6px;background:#2f7668;color:white;font-weight:700;padding:7px 9px;cursor:pointer}.matrix-review button:disabled{background:#cfd8d1;cursor:wait}.matrix-review .save-status{font-size:12px;color:#2f7668;font-weight:700;min-height:16px}.matrix-review.saved{outline:2px solid #b8ded0;border-radius:6px}" + matrix_display_style() + "</style>",
        f"<h1>検証Matrix #{run_id}</h1>",
        matrix_navigation(run_id),
        matrix_display_controls(),
        matrix_hires_controls(has_hires_sections),
        "<p class=\"muted\">画像は50%表示を初期値にしています。必要に応じて25% / 50% / 75% / 100%を切り替えてください。画像クリックで100%表示を開けます。</p>",
    ]
    for (prompt_key, hires), rows in sections.items():
        seeds = sorted({int(row["seed"]) for row in rows})
        weights = sorted({float(row["lora_weight"]) for row in rows})
        lines.append(f"<section class=\"matrix-hires-section\" data-matrix-hires=\"{'hires' if hires else 'nohires'}\">")
        lines.append(f"<h2>{html.escape(prompt_key)} / {'Hiresあり' if hires else 'Hiresなし'}</h2>")
        lines.append("<table><thead><tr><th>seed \\ weight</th>")
        lines.extend(f"<th>{weight:g}</th>" for weight in weights)
        lines.append("</tr></thead><tbody>")
        for seed in seeds:
            lines.append(f"<tr><th>{seed}</th>")
            for weight in weights:
                condition = next((row for row in rows if int(row["seed"]) == seed and float(row["lora_weight"]) == weight), None)
                lines.append("<td class=\"cell\">")
                if condition:
                    linked = images_by_condition.get(int(condition["id"]), [])
                    if linked:
                        image = linked[-1]
                        lines.append(f"<img class=\"matrix-image\" src=\"/validation-images/{int(image['id'])}\" alt=\"validation image\">")
                        lines.append(matrix_machine_score(score_by_image_id.get(int(image['id']))))
                        lines.append(matrix_review_form(run_id, image))
                    else:
                        lines.append("<div class=\"missing\">missing</div>")
                    lines.append(f"<div class=\"muted\">prompt_key: {html.escape(str(condition['prompt_key']))}</div>")
                    lines.append(f"<div class=\"muted\">seed: {condition['seed']} / weight: {float(condition['lora_weight']):g}</div>")
                    lines.append(f"<div class=\"muted\">expected_condition_id: {condition['id']}</div>")
                    lines.append(f"<div class=\"muted\">hash: {html.escape(condition['condition_hash'][:12])}</div>")
                lines.append("</td>")
            lines.append("</tr>")
        lines.append("</tbody></table>")
        lines.append("</section>")
    if grid_images:
        lines.append("<h2>Grid画像</h2><div>")
        for image in grid_images:
            lines.append(f"<figure><img class=\"matrix-image\" src=\"/validation-images/{int(image['id'])}\" alt=\"grid image\"><figcaption>{html.escape(image['memo'] or '')}</figcaption></figure>")
        lines.append("</div>")
    lines.append(matrix_navigation(run_id))
    lines.append(matrix_display_script())
    lines.append(matrix_review_script())
    matrix_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(matrix_path)


def build_epoch_cross_matrix_html(job_id: int, run_ids: list[int]) -> str:
    unique_run_ids = list(dict.fromkeys(int(run_id) for run_id in run_ids if int(run_id) > 0))
    if len(unique_run_ids) < 2:
        raise ValueError("Epoch横断Matrixには検証Runを2件以上選択してください。")
    placeholders = ",".join("?" for _ in unique_run_ids)
    params: list[Any] = [job_id, *unique_run_ids]
    runs = [
        dict(row)
        for row in fetch_all(
            f"""
            SELECT vr.*, o.epoch AS selected_epoch, o.file_path AS selected_output_path
            FROM validation_runs vr
            LEFT JOIN training_outputs o ON o.id = vr.selected_output_id
            WHERE vr.job_id = ? AND vr.id IN ({placeholders})
            ORDER BY CASE WHEN o.epoch IS NULL THEN 999999 ELSE o.epoch END, vr.id
            """,
            tuple(params),
        )
    ]
    if len(runs) != len(unique_run_ids):
        raise ValueError("指定された検証Runの一部が見つからないか、このジョブに属していません。")
    ordered_run_ids = [int(run["id"]) for run in runs]
    placeholders = ",".join("?" for _ in ordered_run_ids)
    conditions = [
        dict(row)
        for row in fetch_all(
            f"""
            SELECT *
            FROM validation_expected_conditions
            WHERE validation_run_id IN ({placeholders})
            ORDER BY prompt_key, hires_enabled, seed, lora_weight, validation_run_id
            """,
            tuple(ordered_run_ids),
        )
    ]
    images = [
        dict(row)
        for row in fetch_all(
            f"""
            SELECT *
            FROM validation_images
            WHERE validation_run_id IN ({placeholders}) AND image_role = 'individual'
            ORDER BY validation_run_id, prompt_key, hires_enabled, seed, lora_weight, id
            """,
            tuple(ordered_run_ids),
        )
    ]
    score_by_image_id = validation_machine_score_map(ordered_run_ids)
    condition_by_key: dict[tuple[int, str, int, int, str], dict[str, Any]] = {}
    sections: dict[tuple[str, int], dict[str, set[Any]]] = {}
    for condition in conditions:
        prompt_key = str(condition["prompt_key"] or "prompt")
        hires = int(condition["hires_enabled"] or 0)
        seed = int(condition["seed"])
        weight_key = f"{float(condition['lora_weight'] or 0):g}"
        condition_by_key[(int(condition["validation_run_id"]), prompt_key, hires, seed, weight_key)] = condition
        section = sections.setdefault((prompt_key, hires), {"seeds": set(), "weights": set()})
        section["seeds"].add(seed)
        section["weights"].add(float(condition["lora_weight"] or 0))

    images_by_condition: dict[int, dict[str, Any]] = {}
    for image in images:
        if image["expected_condition_id"]:
            images_by_condition[int(image["expected_condition_id"])] = image

    title = f"Epoch横断Matrix Job #{job_id}"
    has_hires_sections = any(key[1] for key in sections)
    lines = [
        "<!doctype html>",
        "<meta charset=\"utf-8\">",
        f"<title>{html.escape(title)}</title>",
        cross_matrix_style(),
        f"<h1>{html.escape(title)}</h1>",
        cross_matrix_navigation(job_id),
        matrix_display_controls(),
        matrix_hires_controls(has_hires_sections),
        "<p class=\"muted\">prompt単位でまとめ、同じprompt / seed / weight / Hires条件をEpoch横断で横並び比較します。画像は50%表示を初期値にしています。画像クリックで100%表示を開けます。</p>",
        "<section class=\"run-summary\"><h2>比較対象</h2><div class=\"summary-grid\">",
    ]
    for run in runs:
        lines.append(
            "<div class=\"summary-card\">"
            f"<strong>検証Run #{int(run['id'])}</strong>"
            f"<div>Epoch: {html.escape(run_epoch_label(run))}</div>"
            f"<div>{html.escape(str(run['name'] or '-'))}</div>"
            f"<div class=\"muted\">LoRA: {html.escape(str(run['lora_filename'] or '-'))}</div>"
            "</div>"
        )
    lines.append("</div></section>")

    for (prompt_key, hires), section in sorted(sections.items(), key=lambda item: (item[0][0], item[0][1])):
        seeds = sorted(int(seed) for seed in section["seeds"])
        weights = sorted(float(weight) for weight in section["weights"])
        lines.append(f"<section class=\"matrix-hires-section\" data-matrix-hires=\"{'hires' if hires else 'nohires'}\">")
        lines.append(f"<h2>{html.escape(prompt_key)} / {'Hiresあり' if hires else 'Hiresなし'}</h2>")
        for seed in seeds:
            for weight in weights:
                weight_key = f"{weight:g}"
                lines.append(f"<h3>seed {seed} / weight {weight_key}</h3>")
                lines.append("<table><thead><tr>")
                for run in runs:
                    lines.append(f"<th>{html.escape(run_matrix_label(run))}</th>")
                lines.append("</tr></thead><tbody><tr>")
                for run in runs:
                    condition = condition_by_key.get((int(run["id"]), prompt_key, hires, seed, weight_key))
                    lines.append("<td class=\"epoch-cell\">")
                    if condition is None:
                        lines.append("<div class=\"missing\">条件なし</div>")
                    else:
                        image = images_by_condition.get(int(condition["id"]))
                        if image:
                            lines.append(f"<img class=\"matrix-image\" src=\"/validation-images/{int(image['id'])}\" alt=\"validation image\">")
                            lines.append(matrix_machine_score(score_by_image_id.get(int(image['id']))))
                            lines.append(matrix_review_form(int(run["id"]), image))
                        else:
                            lines.append("<div class=\"missing\">画像未登録</div>")
                        lines.append(f"<div class=\"muted\">検証Run: #{int(run['id'])} / Epoch: {html.escape(run_epoch_label(run))}</div>")
                        lines.append(f"<div class=\"muted\">expected_condition_id: {int(condition['id'])}</div>")
                        lines.append(f"<div class=\"muted\">hash: {html.escape(str(condition['condition_hash'])[:12])}</div>")
                    lines.append("</td>")
                lines.append("</tr></tbody></table>")
        lines.append("</section>")
    if not sections:
        lines.append("<p class=\"notice\">比較できる検証条件がありません。検証Runの生成ファイル作成または条件再生成を確認してください。</p>")
    lines.append(cross_matrix_navigation(job_id))
    lines.append(matrix_display_script())
    lines.append(matrix_review_script())
    return "\n".join(lines) + "\n"


def run_epoch_label(run: dict[str, Any]) -> str:
    return str(run["selected_epoch"]) if run.get("selected_epoch") is not None else "-"


def run_matrix_label(run: dict[str, Any]) -> str:
    epoch = run_epoch_label(run)
    return f"Epoch {epoch} / Run #{int(run['id'])}"


def cross_matrix_navigation(job_id: int) -> str:
    return (
        "<div class=\"matrix-actions\">"
        f"<a class=\"button\" href=\"/jobs/{job_id}#validation-runs\">ジョブへ戻る</a>"
        "<button class=\"button secondary\" type=\"button\" onclick=\"window.close()\">閉じる</button>"
        "</div>"
    )


def validation_machine_score_map(run_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not run_ids:
        return {}
    unique_run_ids = list(dict.fromkeys(int(run_id) for run_id in run_ids if int(run_id) > 0))
    if not unique_run_ids:
        return {}
    placeholders = ",".join("?" for _ in unique_run_ids)
    rows = fetch_all(
        f"""
        SELECT *
        FROM machine_review_scores
        WHERE source_type = 'validation_image'
          AND validation_run_id IN ({placeholders})
        ORDER BY updated_at DESC, id DESC
        """,
        tuple(unique_run_ids),
    )
    scores: dict[int, dict[str, Any]] = {}
    for row in rows:
        source_id = row["source_id"]
        if source_id is None:
            continue
        scores.setdefault(int(source_id), dict(row))
    return scores


def matrix_machine_score(score: dict[str, Any] | None) -> str:
    if not score:
        return "<div class=\"machine-score\"><div class=\"muted\">機械補助レビュー: 未計算</div></div>"
    confidence = str(score.get("confidence_label") or "unknown")
    assist = str(score.get("assist_label") or "unavailable")
    overfit = str(score.get("overfit_risk_label") or "unknown")
    reference = format_score_number(score.get("reference_similarity_max"))
    dataset = format_score_number(score.get("nearest_dataset_similarity"))
    nearest_ref = f"#{int(score['nearest_reference_image_id'])}" if score.get("nearest_reference_image_id") is not None else "-"
    nearest_dataset = f"#{int(score['nearest_dataset_image_id'])}" if score.get("nearest_dataset_image_id") is not None else "-"
    return (
        "<div class=\"machine-score\">"
        "<div class=\"badges\">"
        f"<span class=\"badge {html.escape(confidence)}\">信頼度 {html.escape(machine_label(confidence))}</span>"
        f"<span class=\"badge {html.escape(assist)}\">{html.escape(machine_label(assist))}</span>"
        f"<span class=\"badge {html.escape(overfit)}\">過学習 {html.escape(machine_label(overfit))}</span>"
        "</div>"
        f"<div>Reference {reference} ({html.escape(nearest_ref)}) / Dataset {dataset} ({html.escape(nearest_dataset)})</div>"
        "</div>"
    )


def format_score_number(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def machine_label(value: str) -> str:
    labels = {
        "high": "高",
        "medium": "中",
        "low": "低",
        "unknown": "不明",
        "unavailable": "利用不可",
        "low_confidence": "低信頼",
        "primary_candidate": "有力候補",
        "secondary_candidate": "候補",
        "check_manually": "要確認",
        "possible_overfit": "過学習注意",
    }
    return labels.get(value, value or "-")


def cross_matrix_style() -> str:
    return (
        "<style>"
        "body{font-family:'Segoe UI','Yu Gothic UI',sans-serif;background:#f6f7f4;color:#20231f;margin:24px;overflow-x:auto}"
        "table{border-collapse:collapse;width:max-content;min-width:100%;margin:12px 0 28px;background:white}"
        "th,td{border:1px solid #d8ddd4;padding:8px;vertical-align:top}th{background:#eef2eb;min-width:220px}"
        ".epoch-cell{min-width:300px}.missing{color:#8b4f39;font-weight:700}.notice{background:#f4ece6;border:1px solid #d7b79f;border-radius:6px;padding:10px}"
        "img.matrix-image{width:auto;max-width:none;border-radius:6px;display:block;margin-bottom:6px;cursor:zoom-in}.muted{color:#657064;font-size:12px}"
        ".matrix-actions{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 18px}.button{display:inline-flex;align-items:center;min-height:34px;padding:7px 12px;border-radius:6px;background:#2f7668;color:white;text-decoration:none;font-weight:700;border:0;cursor:pointer;font:inherit}.button.secondary{background:#dce4df;color:#20231f}"
        ".summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:10px 0 20px}.summary-card{background:white;border:1px solid #d8ddd4;border-radius:6px;padding:10px}"
        ".machine-score{display:grid;gap:4px;margin:8px 0;padding:8px;border:1px solid #d8ddd4;border-radius:6px;background:#f8faf7;font-size:13px}.machine-score .badges{display:flex;flex-wrap:wrap;gap:6px}.badge{display:inline-flex;align-items:center;justify-content:center;min-width:56px;padding:3px 8px;border-radius:6px;background:#dce4df;font-weight:700}.badge.low,.badge.low_confidence,.badge.unavailable,.badge.unknown{background:#dce4df}.badge.primary_candidate,.badge.secondary_candidate{background:#c6e7d8}.badge.possible_overfit,.badge.high{background:#f0c2c2}"
        ".matrix-review{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr)) auto auto;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid #d8ddd4;align-items:end}.matrix-review label{display:grid;gap:3px;font-size:12px;color:#4a554f}.matrix-review input,.matrix-review select,.matrix-review textarea{box-sizing:border-box;width:100%;border:1px solid #cfd8d1;border-radius:5px;padding:5px;font:inherit}.matrix-review textarea{min-height:34px;resize:vertical}.matrix-review button{border:0;border-radius:6px;background:#2f7668;color:white;font-weight:700;padding:7px 9px;cursor:pointer}.matrix-review button:disabled{background:#cfd8d1;cursor:wait}.matrix-review .save-status{font-size:12px;color:#2f7668;font-weight:700;min-height:16px}.matrix-review.saved{outline:2px solid #b8ded0;border-radius:6px}"
        + matrix_display_style()
        + "</style>"
    )


def matrix_display_style() -> str:
    return (
        ".matrix-scale-panel{position:sticky;top:0;z-index:20;display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:10px 0 14px;padding:10px;border:1px solid #d8ddd4;border-radius:8px;background:#f6f7f4cc;backdrop-filter:blur(4px)}"
        ".matrix-scale-panel strong{margin-right:2px}.matrix-scale-button{border:0;border-radius:6px;background:#dce4df;color:#20231f;font:inherit;font-weight:700;padding:7px 10px;cursor:pointer}.matrix-scale-button.active{background:#2f7668;color:white}.matrix-scale-note{color:#657064;font-size:12px}"
        ".matrix-hires-section.hidden{display:none}.matrix-hires-button{border:0;border-radius:6px;background:#dce4df;color:#20231f;font:inherit;font-weight:700;padding:7px 10px;cursor:pointer}.matrix-hires-button.active{background:#2f7668;color:white}"
        ".matrix-lightbox{position:fixed;inset:0;z-index:9999;display:none;background:rgba(20,24,21,.86);overflow:auto;padding:24px}.matrix-lightbox.open{display:block}.matrix-lightbox-inner{min-width:max-content}.matrix-lightbox img{display:block;width:auto;max-width:none;height:auto;border-radius:8px;background:white}.matrix-lightbox-close{position:sticky;top:0;left:0;margin-bottom:12px;border:0;border-radius:6px;background:#f6f7f4;color:#20231f;font:inherit;font-weight:700;padding:8px 12px;cursor:pointer}.matrix-lightbox-caption{color:white;font-weight:700;margin:0 0 8px}"
    )


def matrix_display_controls() -> str:
    buttons = "".join(
        f"<button class=\"matrix-scale-button{' active' if scale == 50 else ''}\" type=\"button\" data-matrix-scale=\"{scale}\">{scale}%</button>"
        for scale in (25, 50, 75, 100)
    )
    return (
        "<div class=\"matrix-scale-panel\" data-matrix-scale-panel>"
        "<strong>画像表示倍率</strong>"
        f"{buttons}"
        "<span class=\"matrix-scale-note\">初期値は50%。画像クリックで100%表示を開きます。</span>"
        "</div>"
    )


def matrix_hires_controls(has_hires_sections: bool) -> str:
    if not has_hires_sections:
        return ""
    return (
        "<div class=\"matrix-scale-panel\" data-matrix-hires-panel>"
        "<strong>Hires表示</strong>"
        "<button class=\"matrix-hires-button active\" type=\"button\" data-matrix-hires-mode=\"nohires\">Hiresなし</button>"
        "<button class=\"matrix-hires-button\" type=\"button\" data-matrix-hires-mode=\"hires\">Hiresあり</button>"
        "<span class=\"matrix-scale-note\">画像サイズ差による崩れを避けるため、初期表示はHiresなしです。</span>"
        "</div>"
    )


def matrix_navigation(run_id: int) -> str:
    return (
        "<div class=\"matrix-actions\">"
        f"<a class=\"button\" href=\"/validation-runs/{run_id}\">検証Runへ戻る</a>"
        "<button class=\"button secondary\" type=\"button\" onclick=\"window.close()\">閉じる</button>"
        "</div>"
    )


def matrix_review_form(run_id: int, image: dict[str, Any]) -> str:
    image_id = int(image["id"])
    overall = "" if image.get("rating_overall") is None else str(image["rating_overall"])
    strength = str(image.get("strength_label") or "")
    adoption = str(image.get("adoption_label") or "")
    memo = html.escape(str(image.get("memo") or ""))
    return "\n".join(
        [
            f"<form class=\"matrix-review\" method=\"post\" action=\"/validation-runs/{run_id}/images/{image_id}/matrix-review\">",
            "<label>総合点<input type=\"number\" min=\"0\" max=\"5\" name=\"rating_overall\" value=\"" + html.escape(overall) + "\"></label>",
            "<label>強さ<select name=\"strength_label\">",
            option_tag("", "未評価", strength),
            option_tag("too_weak", "弱すぎる", strength),
            option_tag("weak_but_usable", "弱いが使える", strength),
            option_tag("recommended", "推奨", strength),
            option_tag("strong_but_usable", "強いが使える", strength),
            option_tag("too_strong", "強すぎる", strength),
            option_tag("broken", "破綻", strength),
            "</select></label>",
            "<label>採用判断<select name=\"adoption_label\">",
            option_tag("", "未評価", adoption),
            option_tag("reject", "不採用", adoption),
            option_tag("candidate", "候補", adoption),
            option_tag("adopt", "採用", adoption),
            "</select></label>",
            f"<label>メモ<textarea name=\"memo\">{memo}</textarea></label>",
            "<button type=\"submit\">保存</button><span class=\"save-status\" aria-live=\"polite\"></span>",
            "</form>",
        ]
    )


def option_tag(value: str, label: str, current: str) -> str:
    selected = " selected" if value == current else ""
    return f"<option value=\"{html.escape(value)}\"{selected}>{html.escape(label)}</option>"


def matrix_review_script() -> str:
    return """
<script>
document.addEventListener("submit", async (event) => {
  const form = event.target.closest(".matrix-review");
  if (!form) return;
  event.preventDefault();
  const button = form.querySelector("button[type='submit']");
  const status = form.querySelector(".save-status");
  const original = button ? button.textContent : "";
  if (button) { button.disabled = true; button.textContent = "保存中"; }
  if (status) status.textContent = "";
  try {
    const response = await fetch(form.action, {
      method: "POST",
      body: new FormData(form),
      headers: {"X-Requested-With": "fetch", "Accept": "application/json"}
    });
    if (!response.ok) throw new Error(await response.text());
    form.classList.add("saved");
    if (status) status.textContent = "保存済み";
    window.setTimeout(() => {
      form.classList.remove("saved");
      if (status) status.textContent = "";
    }, 1800);
  } catch (error) {
    if (status) status.textContent = "保存失敗";
    alert("評価保存に失敗しました: " + error.message);
  } finally {
    if (button) { button.disabled = false; button.textContent = original; }
  }
});
</script>
"""


def matrix_display_script() -> str:
    return """
<script>
(() => {
  const DEFAULT_SCALE = 50;
  let currentScale = DEFAULT_SCALE;
  let currentHiresMode = "nohires";

  function scaledWidth(img) {
    const naturalWidth = img.naturalWidth || Number(img.dataset.naturalWidth) || 0;
    if (!naturalWidth) return "";
    return Math.max(1, Math.round(naturalWidth * currentScale / 100)) + "px";
  }

  function applyScale() {
    document.querySelectorAll("img.matrix-image").forEach((img) => {
      const width = scaledWidth(img);
      if (width) img.style.width = width;
    });
    document.querySelectorAll("[data-matrix-scale]").forEach((button) => {
      button.classList.toggle("active", Number(button.dataset.matrixScale) === currentScale);
    });
  }

  function applyHiresMode() {
    const sections = Array.from(document.querySelectorAll(".matrix-hires-section"));
    if (!sections.length) return;
    const modes = new Set(sections.map((section) => section.dataset.matrixHires || "nohires"));
    if (!modes.has(currentHiresMode)) currentHiresMode = modes.has("nohires") ? "nohires" : Array.from(modes)[0];
    sections.forEach((section) => {
      section.classList.toggle("hidden", (section.dataset.matrixHires || "nohires") !== currentHiresMode);
    });
    document.querySelectorAll("[data-matrix-hires-mode]").forEach((button) => {
      button.classList.toggle("active", button.dataset.matrixHiresMode === currentHiresMode);
    });
  }

  function ensureLightbox() {
    let box = document.querySelector(".matrix-lightbox");
    if (box) return box;
    box = document.createElement("div");
    box.className = "matrix-lightbox";
    box.innerHTML = '<button class="matrix-lightbox-close" type="button">閉じる</button><div class="matrix-lightbox-caption"></div><div class="matrix-lightbox-inner"><img alt=""></div>';
    box.querySelector(".matrix-lightbox-close").addEventListener("click", () => box.classList.remove("open"));
    box.addEventListener("click", (event) => {
      if (event.target === box) box.classList.remove("open");
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") box.classList.remove("open");
    });
    document.body.appendChild(box);
    return box;
  }

  document.querySelectorAll("img.matrix-image").forEach((img) => {
    if (img.complete) applyScale();
    img.addEventListener("load", applyScale, {once: true});
    img.addEventListener("click", () => {
      const box = ensureLightbox();
      const preview = box.querySelector("img");
      const caption = box.querySelector(".matrix-lightbox-caption");
      preview.src = img.currentSrc || img.src;
      preview.alt = img.alt || "matrix image";
      preview.style.width = (img.naturalWidth || Number(img.dataset.naturalWidth) || "") ? (img.naturalWidth || Number(img.dataset.naturalWidth)) + "px" : "auto";
      caption.textContent = "100%表示";
      box.classList.add("open");
    });
  });

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-matrix-scale]");
    if (!button) return;
    currentScale = Number(button.dataset.matrixScale) || DEFAULT_SCALE;
    applyScale();
  });

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-matrix-hires-mode]");
    if (!button) return;
    currentHiresMode = button.dataset.matrixHiresMode || "nohires";
    applyHiresMode();
    applyScale();
  });

  applyScale();
  applyHiresMode();
})();
</script>
"""


def path_to_matrix_relative(matrix_path: Path, image_path: Path) -> str:
    try:
        return os.path.relpath(image_path, start=matrix_path.parent).replace("\\", "/")
    except ValueError:
        return str(image_path)


def validation_generation_log_tail(run_id: int, max_lines: int = 80) -> str:
    generation = latest_generation_run(run_id)
    if generation is None or not generation["log_path"]:
        return ""
    path = Path(generation["log_path"])
    if not path.exists():
        return ""
    data = path.read_bytes()
    text = decode_log_bytes(data)
    return "\n".join(text.splitlines()[-max_lines:])


def validation_assist_log_path(run_id: int) -> Path:
    return validation_run_dir(run_id) / "generation" / "assist_sequence.log"


def validation_assist_log_tail(run_id: int, max_lines: int = 40) -> str:
    path = validation_assist_log_path(run_id)
    if not path.exists():
        return ""
    data = path.read_bytes()
    text = decode_log_bytes(data)
    return "\n".join(text.splitlines()[-max_lines:])


def validation_assist_log_state(run_id: int, max_lines: int = 8) -> dict[str, Any]:
    path = validation_assist_log_path(run_id)
    exists = path.exists()
    return {
        "exists": exists,
        "log_path": str(path),
        "log_preview": validation_assist_log_tail(run_id, max_lines=max_lines) if exists else "",
        "log_size": path.stat().st_size if exists else 0,
        "log_updated_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if exists else "",
    }


def generation_view_state(run_id: int) -> dict[str, Any]:
    generation = latest_generation_run(run_id)
    matrix_path = validation_run_dir(run_id) / "validation_matrix.html"
    output_dir = Path(generation["output_dir"]) if generation and generation["output_dir"] else generation_output_dir(run_id)
    log_path = Path(generation["log_path"]) if generation and generation["log_path"] else generation_dir(run_id) / "generation.log"
    log_size = log_path.stat().st_size if log_path.exists() else 0
    log_updated_at = datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if log_path.exists() else ""
    process_alive = bool(
        generation
        and generation["status"] == "running"
        and generation["process_id"]
        and process_exists(int(generation["process_id"]))
    )
    return {
        "generation": generation,
        "generation_log_tail": validation_generation_log_tail(run_id),
        "generation_log_preview": validation_generation_log_tail(run_id, max_lines=12),
        "generation_process_alive": process_alive,
        "generation_output_image_count": count_generated_images(output_dir),
        "generation_log_size": log_size,
        "generation_log_updated_at": log_updated_at,
        "matrix_path": str(matrix_path) if matrix_path.exists() else "",
        "matrix_exists": matrix_path.exists(),
    }


def archive_existing_log(log_path: Path) -> None:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = log_path.with_name(f"generation_{stamp}.log")
    try:
        log_path.replace(archive_path)
    except OSError:
        pass


def elapsed_seconds(start_time_text: str | None, end_time_text: str) -> int | None:
    if not start_time_text:
        return None
    try:
        start = datetime.fromisoformat(start_time_text)
        end = datetime.fromisoformat(end_time_text)
    except ValueError:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return max(0, int((end - start).total_seconds()))


def decode_log_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "cp932", "shift_jis"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")
