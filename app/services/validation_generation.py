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
from app.services.performance_profile import (
    mark_command_end,
    mark_command_start,
    mark_stage,
    refresh_image_mtime_summary,
    reset_pipeline_timing,
    update_timing,
)
from app.services.training_runner import process_exists, sd_scripts_subprocess_env
from app.services.validation_runs import (
    ensure_expected_conditions,
    json_loads,
    update_validation_run_counts,
    validation_run_dir,
    validation_preset_for_run,
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

    all_conditions = [dict(row) for row in ensure_expected_conditions(run_id)]
    conditions = missing_validation_generation_conditions(run_id, all_conditions)
    skipped_existing_count = len(all_conditions) - len(conditions)

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
        "skipped_existing_count": skipped_existing_count,
        "skipped_existing_message": f"既存画像登録済みの条件 {skipped_existing_count} 件をスキップしました。" if skipped_existing_count else "",
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
                "\n".join(part for part in [command_payload["skipped_hires_message"], command_payload["skipped_existing_message"]] if part),
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
        "skipped_existing_count": skipped_existing_count,
    }


def missing_validation_generation_conditions(run_id: int, conditions: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = conditions if conditions is not None else [dict(row) for row in ensure_expected_conditions(run_id)]
    if not rows:
        return []
    registered = fetch_all(
        """
        SELECT expected_condition_id, condition_hash
        FROM validation_images
        WHERE validation_run_id = ?
          AND image_role = 'individual'
          AND COALESCE(ignored, 0) = 0
        """,
        (run_id,),
    )
    registered_ids = {int(row["expected_condition_id"]) for row in registered if row["expected_condition_id"]}
    registered_hashes = {row["condition_hash"] for row in registered if row["condition_hash"]}
    return [
        row
        for row in rows
        if int(row["id"]) not in registered_ids and row["condition_hash"] not in registered_hashes
    ]


def missing_validation_generation_count(run_id: int) -> int:
    return len(missing_validation_generation_conditions(run_id))


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
    if payload.get("skipped_existing_count"):
        lines.append(f"# skipped existing registered images: {payload['skipped_existing_count']}")
    return "\n".join(lines)


def start_validation_generation(run_id: int, run_missing_review_after: bool = False) -> int:
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
        skipped_message = payload.get("skipped_existing_message") or ""
        if skipped_message:
            raise RuntimeError(f"生成対象の不足画像がありません。{skipped_message}")
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
    reset_pipeline_timing("validation_runs", run_id, commands=commands, output_dir=str(generation["output_dir"] or ""))
    mark_stage("validation_runs", run_id, "generation_start", start_time)
    mark_command_start("validation_runs", run_id, 0)
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
                "\n".join(part for part in [payload.get("skipped_hires_message") or "", payload.get("skipped_existing_message") or ""] if part),
                start_time,
                generation["id"],
            ),
        )
    thread = threading.Thread(
        target=monitor_generation,
        args=(
            int(generation["id"]),
            first_process,
            commands,
            0,
            log_handle,
            start_time,
            log_path,
            sd_scripts_path,
            env,
            run_missing_review_after,
        ),
        daemon=True,
    )
    thread.start()
    return first_process.pid


def start_validation_generation_sequence(
    run_ids: list[int],
    run_missing_review_after: bool = False,
    missing_review_run_ids: list[int] | None = None,
) -> int:
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
        now = utc_now()
        for run_id in remaining_run_ids:
            generation = prepare_validation_generation(run_id)
            with connect() as conn:
                conn.execute(
                    """
                    UPDATE validation_generation_runs
                    SET status = 'queued',
                        error_message = COALESCE(NULLIF(error_message, ''), '一括画像生成キューで待機中です。'),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, generation["generation_id"]),
                )
        thread = threading.Thread(
            target=_validation_generation_sequence_worker,
            args=(remaining_run_ids,),
            daemon=True,
        )
        thread.start()
    if run_missing_review_after:
        followup_run_ids = missing_review_run_ids or unique_run_ids
        thread = threading.Thread(
            target=_validation_generation_followup_review_worker,
            args=(unique_run_ids, followup_run_ids),
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


def start_missing_validation_review_sequence(run_id: int) -> dict[str, Any]:
    if fetch_one("SELECT id FROM validation_generation_runs WHERE status = 'running' LIMIT 1"):
        raise RuntimeError("検証画像生成が実行中です。完了後に不足レビューを開始してください。")
    if fetch_one("SELECT id FROM embedding_jobs WHERE status = 'running' LIMIT 1"):
        raise RuntimeError("Embedding Jobが実行中です。完了後に不足レビューを開始してください。")
    if fetch_one("SELECT id FROM machine_review_jobs WHERE status = 'running' LIMIT 1"):
        raise RuntimeError("機械補助レビューJobが実行中です。完了後に不足レビューを開始してください。")
    try:
        from app.services.embedding_service import active_embedding_model
    except Exception:
        active_embedding_model = lambda: {"provider": "mock"}  # type: ignore[assignment]
    embedding_model = active_embedding_model()
    if (embedding_model.get("provider") or "mock") != "mock":
        running_training = fetch_one("SELECT id FROM training_jobs WHERE status = 'running' LIMIT 1")
        if running_training:
            raise RuntimeError("学習ジョブが実行中です。GPUを使うEmbedding providerは、実行中の処理が終わってから開始してください。")
    thread = threading.Thread(
        target=_missing_validation_review_worker,
        args=(run_id,),
        daemon=True,
    )
    thread.start()
    return {"run_id": run_id, "status": "started"}


def start_missing_validation_review_sequences(run_ids: list[int]) -> dict[str, Any]:
    unique_run_ids: list[int] = []
    seen: set[int] = set()
    for run_id in run_ids:
        run_id = int(run_id)
        if run_id <= 0 or run_id in seen:
            continue
        seen.add(run_id)
        unique_run_ids.append(run_id)
    if not unique_run_ids:
        raise ValueError("不足レビューを再計算する検証Runを選択してください。")
    if fetch_one("SELECT id FROM validation_generation_runs WHERE status = 'running' LIMIT 1"):
        raise RuntimeError("検証画像生成が実行中です。完了後に不足レビューを開始してください。")
    if fetch_one("SELECT id FROM embedding_jobs WHERE status = 'running' LIMIT 1"):
        raise RuntimeError("Embedding Jobが実行中です。完了後に不足レビューを開始してください。")
    if fetch_one("SELECT id FROM machine_review_jobs WHERE status = 'running' LIMIT 1"):
        raise RuntimeError("機械補助レビューJobが実行中です。完了後に不足レビューを開始してください。")
    try:
        from app.services.embedding_service import active_embedding_model
    except Exception:
        active_embedding_model = lambda: {"provider": "mock"}  # type: ignore[assignment]
    embedding_model = active_embedding_model()
    if (embedding_model.get("provider") or "mock") != "mock":
        running_training = fetch_one("SELECT id FROM training_jobs WHERE status = 'running' LIMIT 1")
        if running_training:
            raise RuntimeError("学習ジョブが実行中です。GPUを使うEmbedding providerは、実行中の処理が終わってから開始してください。")
    thread = threading.Thread(
        target=_missing_validation_review_sequence_worker,
        args=(unique_run_ids,),
        daemon=True,
    )
    thread.start()
    return {"run_ids": unique_run_ids, "status": "started"}


def set_validation_pipeline_status(run_id: int, status: str, message: str = "") -> None:
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE validation_runs
            SET pipeline_status = ?,
                status = CASE
                    WHEN ? IN ('ready_for_review') THEN 'reviewed'
                    WHEN ? IN ('completed') THEN 'completed'
                    WHEN ? IN ('failed') THEN 'failed'
                    WHEN ? IN ('stopped') THEN 'stopped'
                    ELSE status
                END,
                memo = CASE
                    WHEN ? = '' THEN memo
                    WHEN memo IS NULL OR memo = '' THEN ?
                    ELSE memo || char(10) || ?
                END,
                updated_at = ?
            WHERE id = ?
            """,
            (status, status, status, status, status, message, message, message, now, run_id),
        )


def weight_calibration_preflight(run_id: int) -> dict[str, Any]:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None:
        raise ValueError(f"Validation Run not found: {run_id}")
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (run["job_id"],))
    profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (run["selected_lora_profile_id"],)) if run["selected_lora_profile_id"] else None
    output = fetch_one("SELECT * FROM training_outputs WHERE id = ?", (run["selected_output_id"],)) if run["selected_output_id"] else None
    preset = validation_preset_for_run(run) if run["validation_preset_id"] else None
    conditions = ensure_expected_conditions(run_id)
    checks: list[dict[str, str]] = []

    def add(level: str, key: str, message: str) -> None:
        checks.append({"level": level, "key": key, "message": message})

    def exists_file(value: Any) -> bool:
        return bool(value) and Path(str(value)).exists()

    if output is None and profile is None:
        add("ERROR", "selected_output", "採用LoRA / selected outputが見つかりません。")
    elif output is not None and not exists_file(output["file_path"]):
        exported = profile["exported_model_path"] if profile and "exported_model_path" in profile.keys() else ""
        if not exists_file(exported):
            add("ERROR", "selected_output_file", f"selected outputのfile_pathが存在しません: {output['file_path']}")
    lora_path = (output["file_path"] if output else "") or (profile["selected_model_path"] if profile else "")
    if not exists_file(lora_path):
        add("ERROR", "selected_lora_path", f"採用LoRAファイルが存在しません: {lora_path or '-'}")
    try:
        base_model_path = resolve_base_model_path(run, job, profile)
        if not base_model_path.exists():
            add("ERROR", "base_model", f"base modelが存在しません: {base_model_path}")
    except Exception as exc:
        add("ERROR", "base_model", str(exc))
    if job is None or not job["dataset_version_id"]:
        add("WARNING", "dataset_version", "dataset_version_idがありません。Dataset近傍比較が弱くなります。")
    if not (run["trigger_word"] or "").strip():
        add("ERROR", "trigger_word", "trigger_wordが未設定です。")
    if preset is None:
        add("ERROR", "validation_preset", "validation_preset_idが見つかりません。")
    if not conditions:
        add("ERROR", "expected_conditions", "Expected Conditionsが生成されていません。")
    environment = latest_environment()
    if environment is None:
        add("ERROR", "sd_scripts", "sd-scripts環境が登録されていません。")
    else:
        sd_scripts_path = Path(environment["sd_scripts_path"])
        venv_python = Path(environment["venv_python_path"])
        if not sd_scripts_path.exists() or not (sd_scripts_path / "gen_img.py").exists() or not venv_python.exists():
            add("ERROR", "sd_scripts", "sd-scripts path / gen_img.py / venv pythonのいずれかが存在しません。")
    try:
        from app.services.embedding_service import active_embedding_model, embedding_coverage, provider_preflight
        from app.services.machine_review import context_for_validation_run

        provider = active_embedding_model()
        provider_state = provider_preflight(provider.get("id"))
        if provider_state.get("status") not in {"OK"}:
            messages = "; ".join(str(check.get("message") or check.get("name") or "") for check in provider_state.get("checks", []) if check.get("status") in {"WARNING", "ERROR"})
            add("WARNING", "embedding_provider", messages or "Active embedding providerの準備状態を確認してください。")
        context = context_for_validation_run(run_id)
        if not context.get("reference_set_version_id"):
            add("WARNING", "reference_set", "Reference Setが未設定です。機械補助レビューの根拠が弱くなります。")
        else:
            ref_cov = embedding_coverage("reference_set_version", int(context["reference_set_version_id"]))
            if ref_cov and int(ref_cov.get("ready") or 0) < int(ref_cov.get("total") or 0):
                add("WARNING", "reference_embedding", f"Reference Embeddingが未完了です: {ref_cov.get('ready')} / {ref_cov.get('total')}")
        if context.get("dataset_version_id"):
            dataset_cov = embedding_coverage("dataset_version", int(context["dataset_version_id"]))
            if dataset_cov and int(dataset_cov.get("ready") or 0) < int(dataset_cov.get("total") or 0):
                add("WARNING", "dataset_embedding", f"Dataset Embeddingが未完了です: {dataset_cov.get('ready')} / {dataset_cov.get('total')}")
    except Exception as exc:
        add("WARNING", "embedding_provider", f"Embedding provider確認を完了できませんでした: {exc}")
    if fetch_one("SELECT id FROM training_jobs WHERE status = 'running' LIMIT 1"):
        add("ERROR", "gpu_busy", "学習ジョブが実行中です。")
    if fetch_one("SELECT id FROM review_sessions WHERE status IN ('running', 'generating_images', 'embedding_images', 'machine_reviewing', 'building_matrix') LIMIT 1"):
        add("ERROR", "review_busy", "Review Session処理が実行中です。")
    if fetch_one("SELECT id FROM validation_generation_runs WHERE status = 'running' LIMIT 1"):
        add("ERROR", "validation_generation_busy", "検証画像生成が実行中です。")
    if fetch_one("SELECT id FROM embedding_jobs WHERE status = 'running' LIMIT 1"):
        add("ERROR", "embedding_busy", "Embedding Jobが実行中です。")
    if fetch_one("SELECT id FROM machine_review_jobs WHERE status = 'running' LIMIT 1"):
        add("ERROR", "machine_review_busy", "機械補助レビューJobが実行中です。")
    has_errors = any(row["level"] == "ERROR" for row in checks)
    return {"run_id": run_id, "checks": checks, "has_errors": has_errors, "error_count": sum(row["level"] == "ERROR" for row in checks), "warning_count": sum(row["level"] == "WARNING" for row in checks)}


def start_weight_calibration_pipeline(run_id: int, force_warnings: bool = False) -> dict[str, Any]:
    preflight = weight_calibration_preflight(run_id)
    if preflight["has_errors"]:
        raise RuntimeError("Weight Calibration PreflightでERRORがあります。内容を確認してください。")
    if preflight["warning_count"] and not force_warnings:
        raise RuntimeError("Weight Calibration PreflightにWARNINGがあります。確認してから再実行してください。")
    if fetch_one("SELECT id FROM validation_runs WHERE pipeline_status IN ('generating_images', 'importing_images', 'embedding_images', 'machine_reviewing', 'building_matrix') LIMIT 1"):
        raise RuntimeError("別のWeight Calibration Pipelineが実行中です。")
    set_validation_pipeline_status(run_id, "generating_images")
    thread = threading.Thread(target=_weight_calibration_pipeline_worker, args=(run_id,), daemon=True)
    thread.start()
    return {"run_id": run_id, "status": "started"}


def stop_weight_calibration_pipeline(run_id: int) -> None:
    stop_validation_generation(run_id)
    set_validation_pipeline_status(run_id, "stopped", "Weight Calibration Pipeline was stopped by user.")


def _weight_calibration_pipeline_worker(run_id: int) -> None:
    log_path = validation_run_dir(run_id) / "generation" / "weight_calibration_pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    append_generation_note(log_path, f"Weight Calibration Pipeline started at {started}")
    try:
        set_validation_pipeline_status(run_id, "generating_images")
        append_generation_note(log_path, "Stage: generating_images")
        try:
            start_validation_generation(run_id)
            generation_status = _wait_for_latest_validation_generation(run_id)
            append_generation_note(log_path, f"Generation status: {generation_status}")
            if generation_status not in {"completed"}:
                set_validation_pipeline_status(run_id, "failed", f"画像生成が {generation_status} で終了しました。")
                return
        except RuntimeError as exc:
            if "生成対象の不足画像がありません" not in str(exc):
                raise
            append_generation_note(log_path, str(exc))

        set_validation_pipeline_status(run_id, "importing_images")
        append_generation_note(log_path, "Stage: importing_images")
        generation = latest_generation_run(run_id)
        mark_stage("validation_runs", run_id, "import_start")
        imported = import_generated_images(run_id, int(generation["id"])) if generation else import_generated_images(run_id)
        mark_stage("validation_runs", run_id, "import_end")
        append_generation_note(log_path, f"Imported images: {imported}")

        set_validation_pipeline_status(run_id, "embedding_images")
        append_generation_note(log_path, "Stage: embedding_images")
        mark_stage("validation_runs", run_id, "embedding_start")
        _run_weight_calibration_embeddings(run_id, log_path)
        mark_stage("validation_runs", run_id, "embedding_end")

        set_validation_pipeline_status(run_id, "machine_reviewing")
        append_generation_note(log_path, "Stage: machine_reviewing")
        mark_stage("validation_runs", run_id, "machine_review_start")
        _run_weight_calibration_machine_review(run_id, log_path)
        mark_stage("validation_runs", run_id, "machine_review_end")

        set_validation_pipeline_status(run_id, "building_matrix")
        append_generation_note(log_path, "Stage: building_matrix")
        mark_stage("validation_runs", run_id, "matrix_start")
        matrix_path = write_validation_matrix(run_id)
        mark_stage("validation_runs", run_id, "matrix_end")
        suggestion = _persist_weight_calibration_suggestion(run_id)
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                UPDATE validation_runs
                SET matrix_path = ?, pipeline_status = 'ready_for_review',
                    status = 'reviewed',
                    suggested_weight_min = ?, suggested_weight_max = ?,
                    suggested_light_weight = ?, suggested_strong_weight = ?,
                    suggested_weight_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    matrix_path,
                    suggestion["suggested_weight_min"],
                    suggestion["suggested_weight_max"],
                    suggestion["suggested_light_weight"],
                    suggestion["suggested_strong_weight"],
                    suggestion["suggested_weight_reason"],
                    now,
                    run_id,
                ),
            )
        append_generation_note(log_path, f"Matrix: {matrix_path}")
        append_generation_note(log_path, "Weight Calibration Pipeline completed: ready_for_review")
        mark_stage("validation_runs", run_id, "pipeline_end")
    except Exception as exc:
        append_generation_note(log_path, f"Weight Calibration Pipeline failed: {exc}")
        mark_stage("validation_runs", run_id, "pipeline_end")
        set_validation_pipeline_status(run_id, "failed", f"Weight Calibration Pipeline failed: {exc}")


def _wait_for_latest_validation_generation(run_id: int, timeout_seconds: int = 60 * 60 * 6) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        generation = latest_generation_run(run_id)
        if generation is None:
            return "missing"
        if generation["status"] in {"completed", "failed", "stopped", "superseded"}:
            return str(generation["status"])
        time.sleep(3)
    return "timeout"


def _run_weight_calibration_embeddings(run_id: int, log_path: Path) -> None:
    from app.services.embedding_service import create_embedding_job, start_embedding_job
    from app.services.machine_review import context_for_validation_run

    context = context_for_validation_run(run_id)
    targets: list[tuple[str, int]] = []
    if context.get("reference_set_version_id"):
        targets.append(("reference_set_version", int(context["reference_set_version_id"])))
    if context.get("dataset_version_id"):
        targets.append(("dataset_version", int(context["dataset_version_id"])))
    targets.append(("validation_run", run_id))
    for job_type, target_id in targets:
        embedding_job_id = create_embedding_job(job_type, target_id, recompute="missing")
        job = fetch_one("SELECT total_count FROM embedding_jobs WHERE id = ?", (embedding_job_id,))
        total = int(job["total_count"] or 0) if job else 0
        append_generation_note(log_path, f"Embedding Job #{embedding_job_id}: {job_type} #{target_id}, total={total}")
        if total:
            start_embedding_job(embedding_job_id)
            status_text = _wait_for_background_job("embedding_jobs", embedding_job_id)
        else:
            complete_empty_background_job("embedding_jobs", embedding_job_id)
            status_text = "completed"
        if status_text != "completed":
            raise RuntimeError(f"Embedding Job #{embedding_job_id} ended with {status_text}")


def _run_weight_calibration_machine_review(run_id: int, log_path: Path) -> None:
    from app.services.machine_review import create_machine_review_job, start_machine_review_job

    review_job_id = create_machine_review_job("validation_run_images", run_id)
    job = fetch_one("SELECT total_count FROM machine_review_jobs WHERE id = ?", (review_job_id,))
    total = int(job["total_count"] or 0) if job else 0
    append_generation_note(log_path, f"Machine Review Job #{review_job_id}: total={total}")
    if total:
        start_machine_review_job(review_job_id)
        status_text = _wait_for_background_job("machine_review_jobs", review_job_id)
    else:
        complete_empty_background_job("machine_review_jobs", review_job_id)
        status_text = "completed"
    if status_text != "completed":
        raise RuntimeError(f"Machine Review Job #{review_job_id} ended with {status_text}")


def _persist_weight_calibration_suggestion(run_id: int) -> dict[str, Any]:
    from app.services.validation_runs import persist_suggestion

    return persist_suggestion(run_id)


def _missing_validation_review_worker(run_id: int) -> None:
    log_path = validation_run_dir(run_id) / "generation" / "missing_review_sequence.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from app.services.embedding_service import create_embedding_job, start_embedding_job
        from app.services.machine_review import context_for_validation_run, create_machine_review_job, start_machine_review_job

        context = context_for_validation_run(run_id)
        targets: list[tuple[str, int]] = []
        if context.get("reference_set_version_id"):
            targets.append(("reference_set_version", int(context["reference_set_version_id"])))
        if context.get("dataset_version_id"):
            targets.append(("dataset_version", int(context["dataset_version_id"])))
        targets.append(("validation_run", run_id))

        append_generation_note(log_path, f"検証Run #{run_id}: 不足Embedding計算を開始します。")
        for job_type, target_id in targets:
            embedding_job_id = create_embedding_job(job_type, target_id, recompute="missing")
            embedding_job = fetch_one("SELECT total_count FROM embedding_jobs WHERE id = ?", (embedding_job_id,))
            total = int(embedding_job["total_count"] or 0) if embedding_job else 0
            if not total:
                append_generation_note(log_path, f"Embedding: {job_type} #{target_id} は不足対象がありません。")
                complete_empty_background_job("embedding_jobs", embedding_job_id)
                continue
            append_generation_note(log_path, f"Embedding Job #{embedding_job_id}: {job_type} #{target_id}, {total}件を開始します。")
            start_embedding_job(embedding_job_id)
            status = _wait_for_background_job("embedding_jobs", embedding_job_id)
            append_generation_note(log_path, f"Embedding Job #{embedding_job_id}: status={status}")
            if status != "completed":
                return

        machine_review_job_id = create_machine_review_job("validation_run_images_missing", run_id)
        machine_review_job = fetch_one("SELECT total_count FROM machine_review_jobs WHERE id = ?", (machine_review_job_id,))
        total = int(machine_review_job["total_count"] or 0) if machine_review_job else 0
        if not total:
            append_generation_note(log_path, "Machine Review: 不足レビュー対象はありません。")
            complete_empty_background_job("machine_review_jobs", machine_review_job_id)
            write_validation_matrix(run_id)
            return
        append_generation_note(log_path, f"Machine Review Job #{machine_review_job_id}: 不足レビュー {total}件を開始します。")
        start_machine_review_job(machine_review_job_id)
        status = _wait_for_background_job("machine_review_jobs", machine_review_job_id)
        append_generation_note(log_path, f"Machine Review Job #{machine_review_job_id}: status={status}")
        if status == "completed":
            write_validation_matrix(run_id)
    except Exception as exc:
        append_generation_note(log_path, f"不足レビューの開始に失敗しました: {exc}")


def _missing_validation_review_sequence_worker(run_ids: list[int]) -> None:
    for run_id in run_ids:
        while fetch_one("SELECT id FROM embedding_jobs WHERE status = 'running' LIMIT 1"):
            time.sleep(5)
        while fetch_one("SELECT id FROM machine_review_jobs WHERE status = 'running' LIMIT 1"):
            time.sleep(5)
        _missing_validation_review_worker(run_id)


def _validation_assist_sequence_worker(run_ids: list[int]) -> None:
    for run_id in run_ids:
        log_path = validation_run_dir(run_id) / "generation" / "assist_sequence.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _run_validation_assist_for_run(run_id, log_path)
        except Exception as exc:
            append_generation_note(log_path, f"検証Run #{run_id} のEmbedding / 機械補助レビューに失敗しました: {exc}")


def complete_empty_background_job(table_name: str, job_id: int) -> None:
    now = utc_now()
    with connect() as conn:
        conn.execute(
            f"""
            UPDATE {table_name}
            SET status = 'completed',
                started_at = COALESCE(started_at, ?),
                ended_at = COALESCE(ended_at, ?),
                elapsed_seconds = COALESCE(elapsed_seconds, 0),
                return_code = COALESCE(return_code, 0),
                updated_at = ?
            WHERE id = ?
            """,
            (now, now, now, job_id),
        )


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


def _validation_generation_followup_review_worker(generation_run_ids: list[int], review_run_ids: list[int]) -> None:
    unique_generation_run_ids = list(dict.fromkeys(int(run_id) for run_id in generation_run_ids if int(run_id) > 0))
    unique_review_run_ids = list(dict.fromkeys(int(run_id) for run_id in review_run_ids if int(run_id) > 0))
    if not unique_generation_run_ids or not unique_review_run_ids:
        return
    while True:
        active = False
        for run_id in unique_generation_run_ids:
            generation = latest_generation_run(run_id)
            if generation is not None and generation["status"] in {"running", "queued"}:
                active = True
                break
        if not active:
            break
        time.sleep(5)
    completed_generation_run_ids: set[int] = set()
    for run_id in unique_generation_run_ids:
        generation = latest_generation_run(run_id)
        if generation is not None and generation["status"] == "completed":
            completed_generation_run_ids.add(run_id)
        else:
            log_path = validation_run_dir(run_id) / "generation" / "missing_review_sequence.log"
            append_generation_note(log_path, "画像生成が完了していないため、不足レビュー再計算の自動開始をスキップしました。")
    reviewable_run_ids = [run_id for run_id in unique_review_run_ids if run_id not in unique_generation_run_ids or run_id in completed_generation_run_ids]
    if reviewable_run_ids:
        _missing_validation_review_sequence_worker(reviewable_run_ids)


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
    run_missing_review_after: bool = False,
) -> None:
    return_code = process.wait()
    current_generation = fetch_one("SELECT validation_run_id, output_dir FROM validation_generation_runs WHERE id = ?", (generation_id,))
    run_id_for_timing = int(current_generation["validation_run_id"]) if current_generation else 0
    output_dir_for_timing = str(current_generation["output_dir"] or "") if current_generation else ""
    if run_id_for_timing:
        mark_command_end("validation_runs", run_id_for_timing, command_index, output_dir=output_dir_for_timing, return_code=return_code)
        refresh_image_mtime_summary("validation_runs", run_id_for_timing, output_dir_for_timing)
    current = fetch_one("SELECT * FROM validation_generation_runs WHERE id = ?", (generation_id,))
    if current is not None and current["status"] == "stopped":
        log_handle.close()
        return
    if return_code == 0 and command_index + 1 < len(commands):
        if run_id_for_timing:
            mark_command_start("validation_runs", run_id_for_timing, command_index + 1)
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
        monitor_generation(
            generation_id,
            next_process,
            commands,
            command_index + 1,
            log_handle,
            start_time_text,
            log_path,
            sd_scripts_path,
            env,
            run_missing_review_after,
        )
        return

    log_handle.close()
    generation = fetch_one("SELECT * FROM validation_generation_runs WHERE id = ?", (generation_id,))
    if generation is None:
        return
    run_id = int(generation["validation_run_id"])
    generation_end_time = utc_now()
    mark_stage("validation_runs", run_id, "generation_end", generation_end_time)
    status = "completed" if return_code == 0 else "failed"
    error_message = generation["error_message"] or ""
    imported = 0
    generated = count_generated_images(Path(generation["output_dir"]))
    if status == "completed":
        try:
            mark_stage("validation_runs", run_id, "import_start")
            imported = import_generated_images(int(run_id), generation_id)
            mark_stage("validation_runs", run_id, "import_end")
            auto_machine_review_after_generation(run_id, log_path)
            mark_stage("validation_runs", run_id, "matrix_start")
            write_validation_matrix(run_id)
            mark_stage("validation_runs", run_id, "matrix_end")
        except Exception as exc:
            status = "failed"
            error_message = f"{error_message}\nImport failed: {exc}".strip()
            with log_path.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(f"\n[LoRA-Studio] generated image import failed: {exc}\n")
    final_end_time = utc_now()
    elapsed = elapsed_seconds(start_time_text, final_end_time)
    with connect() as conn:
        conn.execute(
            """
            UPDATE validation_generation_runs
            SET status = ?, process_id = NULL, return_code = ?, ended_at = ?,
                elapsed_seconds = ?, generated_image_count = ?,
                imported_image_count = ?, error_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, return_code, final_end_time, elapsed, generated, imported, error_message, final_end_time, generation_id),
        )
    update_validation_run_counts(run_id)
    mark_stage("validation_runs", run_id, "pipeline_end", final_end_time)
    if status == "completed" and run_missing_review_after:
        append_generation_note(log_path, "画像生成完了後の不足Embedding / 不足Machine Reviewを自動で再計算します。")
        _missing_validation_review_worker(run_id)


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
    detail = {
        "file_scan_seconds": 0.0,
        "duplicate_check_seconds": 0.0,
        "condition_hash_match_seconds": 0.0,
        "image_open_dimension_seconds": 0.0,
        "sha256_seconds": 0.0,
        "db_write_seconds": 0.0,
        "scanned_files": 0,
        "image_files": 0,
        "duplicates": 0,
        "matched": 0,
        "inserted": 0,
    }
    scan_started = time.perf_counter()
    paths = sorted(output_dir.rglob("*")) if output_dir.exists() else []
    detail["file_scan_seconds"] = round(time.perf_counter() - scan_started, 3)
    detail["scanned_files"] = len(paths)
    for path in paths:
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        detail["image_files"] += 1
        try:
            verify_image_file(path)
        except ValueError:
            continue
        match_started = time.perf_counter()
        condition = condition_for_generated_file(path, conditions, hashes)
        detail["condition_hash_match_seconds"] += time.perf_counter() - match_started
        if condition is None:
            continue
        detail["matched"] += 1
        duplicate_started = time.perf_counter()
        existing = fetch_one("SELECT id FROM validation_images WHERE validation_run_id = ? AND image_path = ?", (run_id, str(path)))
        detail["duplicate_check_seconds"] += time.perf_counter() - duplicate_started
        if existing is not None:
            detail["duplicates"] += 1
            continue
        sha_started = time.perf_counter()
        # SHA256 is currently implicit for validation images; keep the timer slot for parity with Review Session import.
        detail["sha256_seconds"] += time.perf_counter() - sha_started
        dim_started = time.perf_counter()
        width = condition["width"]
        height = condition["height"]
        detail["image_open_dimension_seconds"] += time.perf_counter() - dim_started
        db_started = time.perf_counter()
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
                    width,
                    height,
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
        detail["db_write_seconds"] += time.perf_counter() - db_started
        imported += 1
        detail["inserted"] += 1
    for key, value in list(detail.items()):
        if key.endswith("_seconds"):
            detail[key] = round(float(value), 3)
    update_timing("validation_runs", run_id, lambda data: data.__setitem__("import_detail", detail))
    update_validation_run_counts(run_id)
    return imported


def reconcile_stale_validation_generations() -> int:
    now = utc_now()
    with connect() as conn:
        superseded = conn.execute(
            """
            UPDATE validation_generation_runs
            SET status = 'superseded',
                ended_at = COALESCE(ended_at, ?),
                updated_at = ?,
                error_message = COALESCE(
                    NULLIF(error_message, ''),
                    'A newer generation run superseded this queued entry.'
                )
            WHERE status = 'queued'
              AND EXISTS (
                  SELECT 1
                  FROM validation_generation_runs newer
                  WHERE newer.validation_run_id = validation_generation_runs.validation_run_id
                    AND newer.id > validation_generation_runs.id
              )
            """,
            (now, now),
        ).rowcount
    rows = fetch_all("SELECT * FROM validation_generation_runs WHERE status = 'running' ORDER BY id")
    fixed = int(superseded or 0)
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
                mark_stage("validation_runs", run_id, "embedding_start")
                append_generation_note(log_path, f"Auto embedding: {job_type} #{target_id}, {total} item(s).")
                run_embedding_job(embedding_job_id)
                mark_stage("validation_runs", run_id, "embedding_end")
            else:
                append_generation_note(log_path, f"Auto embedding: {job_type} #{target_id}, no missing item.")

        mark_stage("validation_runs", run_id, "machine_review_start")
        result = run_machine_review("validation_run_images", run_id)
        mark_stage("validation_runs", run_id, "machine_review_end")
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
        "<style>body{font-family:'Segoe UI','Yu Gothic UI',sans-serif;background:#f6f7f4;color:#20231f;margin:24px;overflow-x:auto}table{border-collapse:collapse;width:max-content;min-width:100%;margin:16px 0;background:white}th,td{border:1px solid #d8ddd4;padding:8px;vertical-align:top}th{background:#eef2eb}.cell{min-width:300px}.missing{color:#8b4f39;font-weight:700}.notice{background:#f4ece6;border:1px solid #d7b79f;border-radius:6px;padding:10px;margin:0 0 12px}img.matrix-image{width:auto;max-width:none;border-radius:6px;display:block;margin-bottom:6px;cursor:zoom-in}.muted{color:#657064;font-size:12px}.matrix-actions{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 18px}.button{display:inline-flex;align-items:center;min-height:34px;padding:7px 12px;border-radius:6px;background:#2f7668;color:white;text-decoration:none;font-weight:700;border:0;cursor:pointer;font:inherit}.button.secondary{background:#dce4df;color:#20231f}.machine-score{display:grid;gap:4px;margin:8px 0;padding:8px;border:1px solid #d8ddd4;border-radius:6px;background:#f8faf7;font-size:13px}.machine-score .badges{display:flex;flex-wrap:wrap;gap:6px}.badge{display:inline-flex;align-items:center;justify-content:center;min-width:56px;padding:3px 8px;border-radius:6px;background:#dce4df;font-weight:700}.badge.low,.badge.low_confidence,.badge.unavailable,.badge.unknown{background:#dce4df}.badge.primary_candidate,.badge.secondary_candidate{background:#c6e7d8}.badge.possible_overfit,.badge.high{background:#f0c2c2}.matrix-review{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr)) auto auto;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid #d8ddd4;align-items:end}.matrix-review label{display:grid;gap:3px;font-size:12px;color:#4a554f}.matrix-review input,.matrix-review select,.matrix-review textarea{box-sizing:border-box;width:100%;border:1px solid #cfd8d1;border-radius:5px;padding:5px;font:inherit}.matrix-review textarea{min-height:34px;resize:vertical}.matrix-review button{border:0;border-radius:6px;background:#2f7668;color:white;font-weight:700;padding:7px 9px;cursor:pointer}.matrix-review button:disabled{background:#cfd8d1;cursor:wait}.matrix-review .save-status{font-size:12px;color:#2f7668;font-weight:700;min-height:16px}.matrix-review.saved{outline:2px solid #b8ded0;border-radius:6px}" + matrix_display_style() + "</style>",
        f"<h1>検証Matrix #{run_id}</h1>",
        matrix_navigation(run_id),
        matrix_display_controls(),
        matrix_hires_controls(has_hires_sections),
        matrix_weight_controls(run_id),
        "<p class=\"muted\">画像は50%表示を初期値にしています。必要に応じて25% / 50% / 75% / 100%を切り替えてください。画像クリックで100%表示を開けます。</p>",
    ]
    for (prompt_key, hires), rows in sections.items():
        seeds = sorted({int(row["seed"]) for row in rows})
        weights = sorted({float(row["lora_weight"]) for row in rows})
        lines.append(f"<section class=\"matrix-hires-section\" data-matrix-hires=\"{'hires' if hires else 'nohires'}\">")
        lines.append(f"<h2>{html.escape(prompt_key)} / {'Hiresあり' if hires else 'Hiresなし'}</h2>")
        lines.append("<table><thead><tr><th>seed \\ weight</th>")
        lines.extend(f"<th>{weight:g}{' (baseline)' if float(weight) == 0 else ''}</th>" for weight in weights)
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
    with connect() as conn:
        conn.execute("UPDATE validation_runs SET matrix_path = ?, updated_at = ? WHERE id = ?", (str(matrix_path), utc_now(), run_id))
    return str(matrix_path)


def build_epoch_cross_matrix_html(job_id: int, run_ids: list[int], display_weights: list[Any] | None = None) -> str:
    unique_run_ids = list(dict.fromkeys(int(run_id) for run_id in run_ids if int(run_id) > 0))
    if len(unique_run_ids) < 2:
        raise ValueError("Epoch横断Matrixには検証Runを2件以上選択してください。")
    selected_display_weights = normalize_matrix_weights(display_weights or [])
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
        cross_matrix_weight_controls(job_id, runs, selected_display_weights),
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
        if selected_display_weights:
            weights = [weight for weight in weights if round(float(weight), 1) in selected_display_weights]
            if not weights:
                continue
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


def normalize_matrix_weights(values: list[Any]) -> list[float]:
    weights: list[float] = []
    for value in values:
        try:
            weight = round(float(value), 1)
        except (TypeError, ValueError):
            continue
        if 0 <= weight <= 2.0:
            weights.append(weight)
    return sorted(set(weights))


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
        ".matrix-weight-panel{position:sticky;top:58px;z-index:19;display:grid;gap:10px;margin:10px 0 14px;padding:12px;border:1px solid #d8ddd4;border-radius:8px;background:#f6f7f4cc;backdrop-filter:blur(4px)}.matrix-weight-head{display:flex;flex-wrap:wrap;align-items:center;gap:8px}.matrix-weight-options{display:flex;flex-wrap:wrap;gap:6px}.matrix-weight-option{display:inline-flex;align-items:center;gap:5px;min-height:30px;padding:4px 8px;border:1px solid #cfd8d1;border-radius:6px;background:white;font-weight:700}.matrix-weight-option input{margin:0}.matrix-weight-extra.hidden{display:none}.matrix-weight-submit{justify-self:start;border:0;border-radius:6px;background:#2f7668;color:white;font:inherit;font-weight:700;padding:8px 12px;cursor:pointer}.matrix-weight-toggle{border:0;border-radius:6px;background:#dce4df;color:#20231f;font:inherit;font-weight:700;padding:7px 10px;cursor:pointer}"
        ".matrix-review-panel{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:10px 0 14px;padding:10px;border:1px solid #d8ddd4;border-radius:8px;background:#eef2eb}.matrix-review-panel form{display:inline-flex}.matrix-review-submit{border:0;border-radius:6px;background:#2f7668;color:white;font:inherit;font-weight:700;padding:8px 12px;cursor:pointer}"
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


def matrix_weight_controls(run_id: int) -> str:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    preset = validation_preset_for_run(run) if run else None
    preset_weights = set()
    if preset:
        preset_weights = {round(float(weight), 1) for weight in json_loads(preset["weights_json"], [])}
    existing_weights = {
        round(float(row["lora_weight"] or 0), 1)
        for row in fetch_all("SELECT DISTINCT lora_weight FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
    }
    selected_weights = existing_weights or preset_weights or {0, 0.4, 0.6, 0.8, 1.0}

    def option(weight: float) -> str:
        checked = " checked" if round(weight, 1) in selected_weights else ""
        exists_label = " data-existing=\"1\"" if round(weight, 1) in existing_weights else ""
        return (
            f"<label class=\"matrix-weight-option\"{exists_label}>"
            f"<input type=\"checkbox\" name=\"selected_weights\" value=\"{weight:g}\"{checked} form=\"matrix-weight-form\">"
            f"<span>{weight:g}</span>"
            "</label>"
        )

    default_options = "".join(option(index / 10) for index in range(0, 11))
    extra_options = "".join(option(index / 10) for index in range(11, 21))
    return (
        f"<form id=\"matrix-weight-form\" method=\"post\" action=\"/validation-runs/{run_id}/matrix/weights\"></form>"
        f"<form id=\"matrix-missing-review-form\" method=\"post\" action=\"/validation-runs/{run_id}/matrix/missing-review\"></form>"
        "<div class=\"matrix-weight-panel\">"
        "<div class=\"matrix-weight-head\">"
        "<strong>weight選択</strong>"
        "<span class=\"matrix-scale-note\">チェックしたweightをMatrix操作の対象にします。追加生成は不足画像だけを作成し、既存画像はスキップします。</span>"
        "</div>"
        f"<div class=\"matrix-weight-options\">{default_options}</div>"
        "<div class=\"matrix-weight-extra hidden\" data-matrix-extra-weights>"
        f"<div class=\"matrix-weight-options\">{extra_options}</div>"
        "</div>"
        "<div class=\"matrix-weight-head\">"
        "<button class=\"matrix-weight-toggle\" type=\"button\" data-matrix-toggle-extra-weights>1.1〜2.0を表示</button>"
        "<button class=\"matrix-weight-submit\" type=\"submit\" form=\"matrix-weight-form\">選択weightを追加生成して不足レビューも再計算</button>"
        "<button class=\"matrix-weight-submit\" type=\"submit\" form=\"matrix-missing-review-form\">不足レビューだけ再計算</button>"
        "</div>"
        "</div>"
    )


def matrix_missing_review_controls(run_id: int) -> str:
    return (
        "<div class=\"matrix-review-panel\">"
        "<strong>機械補助レビュー</strong>"
        "<span class=\"matrix-scale-note\">画像生成済みの未計算Embedding / Machine Reviewだけを実行します。</span>"
        f"<form method=\"post\" action=\"/validation-runs/{run_id}/matrix/missing-review\">"
        "<button class=\"matrix-review-submit\" type=\"submit\">不足レビューだけ再計算</button>"
        "</form>"
        "</div>"
    )


def cross_matrix_weight_controls(job_id: int, runs: list[dict[str, Any]], display_weights: list[float] | None = None) -> str:
    run_ids = [int(run["id"]) for run in runs]
    hidden_runs = "".join(f"<input type=\"hidden\" name=\"run_ids\" value=\"{run_id}\">" for run_id in run_ids)
    display_hidden_runs = "".join(f"<input type=\"hidden\" form=\"cross-matrix-display-form\" name=\"run_ids\" value=\"{run_id}\">" for run_id in run_ids)
    review_hidden_runs = "".join(f"<input type=\"hidden\" form=\"cross-matrix-missing-review-form\" name=\"run_ids\" value=\"{run_id}\">" for run_id in run_ids)
    selected_weights: set[float] = set()
    existing_weights: set[float] = set()
    if run_ids:
        placeholders = ",".join("?" for _ in run_ids)
        existing_weights = {
            round(float(row["lora_weight"] or 0), 1)
            for row in fetch_all(
                f"SELECT DISTINCT lora_weight FROM validation_expected_conditions WHERE validation_run_id IN ({placeholders})",
                tuple(run_ids),
            )
        }
    for run in runs:
        preset = validation_preset_for_run(run)
        if preset:
            selected_weights.update(round(float(weight), 1) for weight in json_loads(preset["weights_json"], []))
    selected_weights = set(display_weights or []) or existing_weights or selected_weights or {0, 0.4, 0.6, 0.8, 1.0}

    def option(weight: float) -> str:
        checked = " checked" if round(weight, 1) in selected_weights else ""
        exists_label = " data-existing=\"1\"" if round(weight, 1) in existing_weights else ""
        return (
            f"<label class=\"matrix-weight-option\"{exists_label}>"
            f"<input type=\"checkbox\" name=\"selected_weights\" value=\"{weight:g}\"{checked} form=\"cross-matrix-weight-form\">"
            f"<input type=\"checkbox\" name=\"display_weights\" value=\"{weight:g}\"{checked} form=\"cross-matrix-display-form\" hidden>"
            f"<span>{weight:g}</span>"
            "</label>"
        )

    default_options = "".join(option(index / 10) for index in range(0, 11))
    extra_options = "".join(option(index / 10) for index in range(11, 21))
    run_label = ", ".join(f"#{run_id}" for run_id in run_ids)
    return (
        f"<form id=\"cross-matrix-weight-form\" method=\"post\" action=\"/jobs/{job_id}/validation-runs/epoch-matrix/weights\">{hidden_runs}</form>"
        f"<form id=\"cross-matrix-display-form\" method=\"get\" action=\"/jobs/{job_id}/validation-runs/epoch-matrix\">{display_hidden_runs}</form>"
        f"<form id=\"cross-matrix-missing-review-form\" method=\"post\" action=\"/jobs/{job_id}/validation-runs/epoch-matrix/missing-review\">{review_hidden_runs}</form>"
        "<div class=\"matrix-weight-panel\">"
        "<div class=\"matrix-weight-head\">"
        "<strong>weight選択</strong>"
        f"<span class=\"matrix-scale-note\">表示中の検証Run {html.escape(run_label)} で、チェックしたweightをMatrix操作の対象にします。追加生成は不足画像だけを作成し、既存画像はスキップします。生成後は不足Embedding / Machine Reviewまで自動で再計算します。</span>"
        "</div>"
        f"<div class=\"matrix-weight-options\">{default_options}</div>"
        "<div class=\"matrix-weight-extra hidden\" data-matrix-extra-weights>"
        f"<div class=\"matrix-weight-options\">{extra_options}</div>"
        "</div>"
        "<div class=\"matrix-weight-head\">"
        "<button class=\"matrix-weight-toggle\" type=\"button\" data-matrix-toggle-extra-weights>1.1〜2.0を表示</button>"
        "<button class=\"matrix-weight-toggle\" type=\"submit\" form=\"cross-matrix-display-form\">選択weightでMatrix表示を更新</button>"
        "<button class=\"matrix-weight-submit\" type=\"submit\" form=\"cross-matrix-weight-form\">選択weightを追加生成して不足レビューも再計算</button>"
        "</div>"
        "</div>"
    )


def cross_matrix_missing_review_controls(job_id: int, run_ids: list[int]) -> str:
    hidden = "".join(f"<input type=\"hidden\" name=\"run_ids\" value=\"{int(run_id)}\">" for run_id in run_ids)
    run_label = ", ".join(f"#{int(run_id)}" for run_id in run_ids)
    return (
        "<div class=\"matrix-review-panel\">"
        "<strong>機械補助レビュー</strong>"
        f"<span class=\"matrix-scale-note\">表示中の検証Run {html.escape(run_label)} の不足Embedding / Machine Reviewだけを順番に実行します。</span>"
        f"<form method=\"post\" action=\"/jobs/{job_id}/validation-runs/epoch-matrix/missing-review\">"
        f"{hidden}"
        "<button class=\"matrix-review-submit\" type=\"submit\">表示中Runの不足レビューだけ再計算</button>"
        "</form>"
        "</div>"
    )


def matrix_navigation(run_id: int) -> str:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    job_link = f"<a class=\"button\" href=\"/jobs/{int(run['job_id'])}\">Jobへ戻る</a>" if run else ""
    project_link = f"<a class=\"button\" href=\"/projects/{int(run['project_id'])}\">Projectへ戻る</a>" if run and run["project_id"] else ""
    profile_link = f"<a class=\"button\" href=\"/lora-library/{int(run['selected_lora_profile_id'])}/edit\">LoRAプロファイルへ戻る</a>" if run and run["selected_lora_profile_id"] else ""
    return (
        "<div class=\"matrix-actions\">"
        f"<a class=\"button\" href=\"/validation-runs/{run_id}\">検証Runへ戻る</a>"
        f"{job_link}"
        f"{project_link}"
        f"{profile_link}"
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

  function matrixRunIds() {
    const ids = new Set();
    const params = new URLSearchParams(window.location.search);
    ["run_ids", "run_id"].forEach((name) => {
      params.getAll(name).forEach((value) => {
        const id = Number(value);
        if (id > 0) ids.add(id);
      });
    });
    document.querySelectorAll('input[name="run_ids"]').forEach((input) => {
      const id = Number(input.value);
      if (id > 0) ids.add(id);
    });
    return Array.from(ids);
  }

  function syncMatrixRunInputsForSubmit(form) {
    if (!form || !form.id) return;
    const runIds = matrixRunIds().sort((a, b) => a - b);
    form.querySelectorAll('input[name="run_ids"]').forEach((input) => input.remove());
    runIds.forEach((runId) => {
      const input = document.createElement("input");
      input.type = "hidden";
      input.name = "run_ids";
      input.value = String(runId);
      form.appendChild(input);
    });
    const action = new URL(form.getAttribute("action") || window.location.pathname, window.location.href);
    action.search = "";
    form.action = action.pathname;
  }

  function matrixWeightStateKey() {
    const runPart = matrixRunIds().sort((a, b) => a - b).join(",");
    return `lora-studio:matrix-weights:${window.location.pathname}:${runPart}`;
  }

  function selectedMatrixWeights() {
    return Array.from(document.querySelectorAll('input[name="selected_weights"]:checked'))
      .map((input) => input.value)
      .sort((a, b) => Number(a) - Number(b));
  }

  function saveMatrixWeightState() {
    const inputs = document.querySelectorAll('input[name="selected_weights"]');
    if (!inputs.length) return;
    window.sessionStorage.setItem(matrixWeightStateKey(), JSON.stringify(selectedMatrixWeights()));
  }

  function restoreMatrixWeightState() {
    const inputs = Array.from(document.querySelectorAll('input[name="selected_weights"]'));
    if (!inputs.length) return;
    let values = null;
    try {
      values = JSON.parse(window.sessionStorage.getItem(matrixWeightStateKey()) || "null");
    } catch (error) {
      values = null;
    }
    if (!Array.isArray(values)) return;
    const selected = new Set(values.map(String));
    inputs.forEach((input) => {
      input.checked = selected.has(input.value);
    });
    syncDisplayWeightInputs();
  }

  function syncDisplayWeightInputs() {
    const checked = new Set(Array.from(document.querySelectorAll('input[name="selected_weights"]:checked')).map((input) => input.value));
    document.querySelectorAll('input[name="display_weights"]').forEach((input) => {
      input.checked = checked.has(input.value);
    });
  }

  function stripMatrixMessageAndReload() {
    const url = new URL(window.location.href);
    url.searchParams.delete("run_id");
    if (!url.searchParams.has("matrix_message")) {
      window.location.replace(url.toString());
      return;
    }
    url.searchParams.delete("matrix_message");
    window.location.replace(url.toString());
  }

  async function pollMatrixGeneration() {
    const runIds = matrixRunIds();
    if (!runIds.length) return;
    const results = await Promise.all(runIds.map(async (runId) => {
      try {
        const response = await fetch(`/validation-runs/${runId}/generation/status`, {
          headers: {"Accept": "application/json", "X-Requested-With": "fetch"},
          cache: "no-store"
        });
        if (!response.ok) return {run_id: runId, status: "unknown"};
        return response.json();
      } catch (error) {
        return {run_id: runId, status: "unknown"};
      }
    }));
    const active = results.filter((row) => ["running", "queued"].includes(row.status));
    if (active.length) {
      window.__matrixHadActiveGeneration = true;
      return;
    }
    const url = new URL(window.location.href);
    const hasStartMessage = (url.searchParams.get("matrix_message") || "").includes("開始");
    if (window.__matrixHadActiveGeneration || hasStartMessage) {
      stripMatrixMessageAndReload();
    }
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

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-matrix-toggle-extra-weights]");
    if (!button) return;
    const panel = document.querySelector("[data-matrix-extra-weights]");
    if (!panel) return;
    const hidden = panel.classList.toggle("hidden");
    button.textContent = hidden ? "1.1〜2.0を表示" : "1.1〜2.0を隠す";
  });

  document.addEventListener("change", (event) => {
    if (!event.target.closest('input[name="selected_weights"]')) return;
    syncDisplayWeightInputs();
    saveMatrixWeightState();
  });

  document.addEventListener("submit", (event) => {
    const form = event.target.closest("#cross-matrix-weight-form, #cross-matrix-display-form, #cross-matrix-missing-review-form");
    if (!form) return;
    syncMatrixRunInputsForSubmit(form);
  });

  syncDisplayWeightInputs();
  restoreMatrixWeightState();
  applyScale();
  applyHiresMode();
  pollMatrixGeneration();
  window.setInterval(pollMatrixGeneration, 10000);
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
