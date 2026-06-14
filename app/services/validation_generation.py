from __future__ import annotations

import html
import json
import os
import re
import subprocess
import threading
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
    supported = [row for row in conditions if not bool(row["hires_enabled"])]
    skipped_hires = [row for row in conditions if bool(row["hires_enabled"])]
    baseline = [row for row in supported if float(row["lora_weight"] or 0) == 0]
    lora_rows = [row for row in supported if float(row["lora_weight"] or 0) != 0]

    gen_dir = generation_dir(run_id)
    out_dir = generation_output_dir(run_id)
    gen_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_prompt_path = gen_dir / "validation_prompts_for_sd_scripts.txt"
    baseline_prompt_path = gen_dir / "validation_prompts_baseline_for_sd_scripts.txt"
    lora_prompt_path = gen_dir / "validation_prompts_lora_for_sd_scripts.txt"
    all_prompt_path.write_text("\n".join(prompt_line(run_id, row, include_am=float(row["lora_weight"] or 0) != 0) for row in supported) + ("\n" if supported else ""), encoding="utf-8")
    baseline_prompt_path.write_text("\n".join(prompt_line(run_id, row, include_am=False) for row in baseline) + ("\n" if baseline else ""), encoding="utf-8")
    lora_prompt_path.write_text("\n".join(prompt_line(run_id, row, include_am=True) for row in lora_rows) + ("\n" if lora_rows else ""), encoding="utf-8")

    commands = []
    base_args = common_gen_img_args(
        venv_python=venv_python,
        gen_img=gen_img,
        base_model_path=base_model_path,
        out_dir=out_dir,
        model_family=str(job["model_family"] or ""),
        mixed_precision=str(environment["mixed_precision"] or ""),
    )
    if baseline:
        commands.append(
            {
                "name": "baseline_weight_0_no_lora",
                "baseline_mode": "no_network_weights",
                "prompt_file": str(baseline_prompt_path),
                "condition_count": len(baseline),
                "argv": [*base_args, "--from_file", str(baseline_prompt_path)],
            }
        )
    if lora_rows:
        commands.append(
            {
                "name": "lora_weights",
                "baseline_mode": "",
                "prompt_file": str(lora_prompt_path),
                "condition_count": len(lora_rows),
                "argv": [
                    *base_args,
                    "--from_file",
                    str(lora_prompt_path),
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
        "skipped_hires_count": len(skipped_hires),
        "skipped_hires_message": "Hires generation not implemented yet" if skipped_hires else "",
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
        "skipped_hires_count": len(skipped_hires),
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
) -> list[str]:
    args = [
        str(venv_python),
        str(gen_img),
        "--ckpt",
        str(base_model_path),
        "--outdir",
        str(out_dir),
        "--W",
        "1024",
        "--H",
        "1024",
        "--scale",
        "7",
        "--steps",
        "28",
        "--sampler",
        "euler_a",
        "--no_preview",
    ]
    if model_family.upper() == "SDXL":
        args.append("--sdxl")
    if "bf16" in mixed_precision.lower():
        args.append("--bf16")
    else:
        args.append("--fp16")
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
        "# Hires generation is skipped in this version.",
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
        raise RuntimeError("生成対象の条件がありません。Hiresありのみの場合は初期版では生成できません。")
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


def write_validation_matrix(run_id: int) -> str:
    conditions = [dict(row) for row in fetch_all("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY prompt_key, hires_enabled, seed, lora_weight", (run_id,))]
    images = [dict(row) for row in fetch_all("SELECT * FROM validation_images WHERE validation_run_id = ? ORDER BY prompt_key, hires_enabled, seed, lora_weight, id", (run_id,))]
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
    lines = [
        "<!doctype html>",
        "<meta charset=\"utf-8\">",
        f"<title>Validation Matrix #{run_id}</title>",
        "<style>body{font-family:'Segoe UI','Yu Gothic UI',sans-serif;background:#f6f7f4;color:#20231f;margin:24px;overflow-x:auto}table{border-collapse:collapse;width:max-content;min-width:100%;margin:16px 0;background:white}th,td{border:1px solid #d8ddd4;padding:8px;vertical-align:top}th{background:#eef2eb}.cell{min-width:1040px}.missing{color:#8b4f39;font-weight:700}img{width:auto;max-width:none;border-radius:6px;display:block;margin-bottom:6px}.muted{color:#657064;font-size:12px}.matrix-actions{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 18px}.button{display:inline-flex;align-items:center;min-height:34px;padding:7px 12px;border-radius:6px;background:#2f7668;color:white;text-decoration:none;font-weight:700;border:0;cursor:pointer;font:inherit}.button.secondary{background:#dce4df;color:#20231f}.matrix-review{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr)) auto auto;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid #d8ddd4;align-items:end}.matrix-review label{display:grid;gap:3px;font-size:12px;color:#4a554f}.matrix-review input,.matrix-review select,.matrix-review textarea{box-sizing:border-box;width:100%;border:1px solid #cfd8d1;border-radius:5px;padding:5px;font:inherit}.matrix-review textarea{min-height:34px;resize:vertical}.matrix-review button{border:0;border-radius:6px;background:#2f7668;color:white;font-weight:700;padding:7px 9px;cursor:pointer}.matrix-review button:disabled{background:#cfd8d1;cursor:wait}.matrix-review .save-status{font-size:12px;color:#2f7668;font-weight:700;min-height:16px}.matrix-review.saved{outline:2px solid #b8ded0;border-radius:6px}</style>",
        f"<h1>Validation Matrix #{run_id}</h1>",
        matrix_navigation(run_id),
        "<p class=\"muted\">画像は原寸で表示します。横スクロールしながら細部を比較してください。Matrix上では総合点・強さ・採用判断・メモだけを素早く保存できます。</p>",
    ]
    for (prompt_key, hires), rows in sections.items():
        seeds = sorted({int(row["seed"]) for row in rows})
        weights = sorted({float(row["lora_weight"]) for row in rows})
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
                        lines.append(f"<img src=\"/validation-images/{int(image['id'])}\" alt=\"validation image\">")
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
    if grid_images:
        lines.append("<h2>Grid画像</h2><div>")
        for image in grid_images:
            lines.append(f"<figure><img src=\"/validation-images/{int(image['id'])}\" alt=\"grid image\"><figcaption>{html.escape(image['memo'] or '')}</figcaption></figure>")
        lines.append("</div>")
    lines.append(matrix_navigation(run_id))
    lines.append(matrix_review_script())
    matrix_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(matrix_path)


def matrix_navigation(run_id: int) -> str:
    return (
        "<div class=\"matrix-actions\">"
        f"<a class=\"button\" href=\"/validation-runs/{run_id}\">外部検証へ戻る</a>"
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
