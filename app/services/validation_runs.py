from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from app import settings
from app.db import connect, fetch_all, fetch_one, utc_now


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def validation_presets() -> list[Any]:
    return fetch_all("SELECT * FROM validation_presets WHERE is_active = 1 ORDER BY validation_level, name")


def preset_expected_count(preset: Any) -> int:
    prompts = json_loads(preset["prompts_json"], [])
    seeds = json_loads(preset["seeds_json"], [])
    weights = json_loads(preset["weights_json"], [])
    hires_modes = json_loads(preset["hires_modes_json"], [])
    return len(prompts) * len(seeds) * len(weights) * len(hires_modes)


def expand_preset_conditions(preset: Any, trigger_word: str, lora_filename: str, base_model: str) -> list[dict[str, Any]]:
    prompts = json_loads(preset["prompts_json"], [])
    seeds = json_loads(preset["seeds_json"], [])
    weights = json_loads(preset["weights_json"], [])
    hires_modes = json_loads(preset["hires_modes_json"], [])
    lora_name = Path(lora_filename or "selected_lora").stem
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_key = prompt.get("prompt_key") or prompt.get("name") or "prompt"
        prompt_text = (prompt.get("prompt") or "").replace("{trigger_word}", trigger_word or "")
        for seed in seeds:
            seed_value = int(seed) + int(prompt.get("seed_offset") or 0)
            for weight in weights:
                for hires_enabled in hires_modes:
                    condition = {
                        "validation_preset_id": preset["id"],
                        "preset_name": preset["name"],
                        "prompt_key": prompt_key,
                        "prompt_name": prompt.get("name") or prompt_key,
                        "prompt": prompt_text,
                        "webui_prompt": f"<lora:{lora_name}:{weight}>, {prompt_text}",
                        "negative_prompt": preset["negative_prompt"] or "",
                        "seed": seed_value,
                        "lora_weight": float(weight),
                        "width": int(preset["width"]),
                        "height": int(preset["height"]),
                        "hires_enabled": bool(hires_enabled),
                        "hires_scale": preset["hires_scale"],
                        "hires_denoising_strength": preset["hires_denoising_strength"],
                        "hires_upscaler": preset["hires_upscaler"] or "",
                        "sampler": preset["sampler"] or "",
                        "steps": preset["steps"],
                        "cfg_scale": preset["cfg_scale"],
                        "base_model": base_model or "",
                    }
                    condition["condition_hash"] = make_condition_hash(condition)
                    rows.append(condition)
    return rows


def make_condition_hash(data: dict[str, Any]) -> str:
    keys = [
        "validation_preset_id",
        "prompt_key",
        "seed",
        "lora_weight",
        "width",
        "height",
        "hires_enabled",
        "hires_scale",
        "hires_denoising_strength",
        "sampler",
        "steps",
        "cfg_scale",
        "negative_prompt",
        "base_model",
    ]
    payload = {key: data.get(key) for key in keys}
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def create_validation_run(
    job_id: int,
    validation_preset_id: str,
    base_model: str,
    trigger_word: str,
    memo: str,
    profile_id: int | None = None,
) -> int:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    selected_output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    if profile_id is None:
        profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE job_id = ? ORDER BY updated_at DESC, id DESC LIMIT 1", (job_id,))
    else:
        profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (profile_id,))
    preset = fetch_one("SELECT * FROM validation_presets WHERE id = ?", (validation_preset_id,))
    if preset is None:
        raise ValueError(f"Validation preset not found: {validation_preset_id}")
    if selected_output is None and profile is not None and profile["selected_output_id"]:
        selected_output = fetch_one("SELECT * FROM training_outputs WHERE id = ?", (profile["selected_output_id"],))
    lora_filename = Path(
        (selected_output["file_path"] if selected_output else "")
        or (profile["selected_model_path"] if profile else "")
        or "selected_lora.safetensors"
    ).name
    base_model_value = base_model.strip() or (profile["base_model"] if profile else "") or Path(job["base_model_path"]).stem
    trigger_value = trigger_word.strip() or (profile["trigger_word"] if profile else "") or job["trigger_word_at_creation"] or ""
    expected = preset_expected_count(preset)
    now = utc_now()
    name = f"Job #{job_id} {preset['name']}"
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO validation_runs(
                job_id, selected_output_id, selected_lora_profile_id,
                validation_preset_id, name, validation_level, base_model,
                trigger_word, lora_filename, recommended_weight_min,
                recommended_weight_max, expected_image_count, actual_image_count,
                status, created_at, updated_at, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'planned', ?, ?, ?)
            """,
            (
                job_id,
                selected_output["id"] if selected_output else profile["selected_output_id"] if profile else None,
                profile["id"] if profile else profile_id,
                preset["id"],
                name,
                preset["validation_level"],
                base_model_value,
                trigger_value,
                lora_filename,
                profile["recommended_weight_min"] if profile else None,
                profile["recommended_weight_max"] if profile else None,
                expected,
                now,
                now,
                memo.strip(),
            ),
        )
        run_id = int(cur.lastrowid)
        if profile:
            conn.execute(
                "UPDATE selected_lora_profiles SET last_validation_preset_id = ?, updated_at = ? WHERE id = ?",
                (preset["id"], now, profile["id"]),
            )
    write_validation_prompt_pack(run_id)
    return run_id


def validation_run_dir(run_id: int) -> Path:
    return settings.EXPORTS_DIR / "validation_runs" / f"validation_run_{run_id:06d}"


def validation_image_dir(run_id: int) -> Path:
    return validation_run_dir(run_id) / "images"


def write_validation_prompt_pack(run_id: int) -> dict[str, str]:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None:
        raise ValueError(f"Validation run not found: {run_id}")
    preset = fetch_one("SELECT * FROM validation_presets WHERE id = ?", (run["validation_preset_id"],))
    if preset is None:
        raise ValueError("Validation preset not found")
    output_dir = validation_run_dir(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = expand_preset_conditions(preset, run["trigger_word"] or "", run["lora_filename"] or "", run["base_model"] or "")
    md_lines = [
        f"# Validation Prompts Run #{run_id}",
        "",
        f"- Preset: {preset['name']}",
        f"- Level: {preset['validation_level']}",
        f"- Base model: {run['base_model'] or '-'}",
        f"- Trigger: {run['trigger_word'] or '-'}",
        f"- LoRA: {Path(run['lora_filename'] or 'selected_lora').stem}",
        "",
    ]
    for index, row in enumerate(conditions, start=1):
        md_lines.extend(
            [
                f"## {index}. {row['prompt_key']} / seed {row['seed']} / weight {row['lora_weight']:g} / Hires {row['hires_enabled']}",
                "",
                "Prompt:",
                "```text",
                row["webui_prompt"],
                "```",
                "Negative:",
                "```text",
                row["negative_prompt"],
                "```",
                f"Size: {row['width']}x{row['height']} / Sampler: {row['sampler']} / Steps: {row['steps']} / CFG: {row['cfg_scale']}",
                "",
            ]
        )
    (output_dir / "validation_prompts.md").write_text("\n".join(md_lines), encoding="utf-8")
    (output_dir / "validation_prompts.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "validation_conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=2), encoding="utf-8")
    grid_lines = [
        f"# Validation Grid Plan Run #{run_id}",
        "",
        "- X軸: lora_weight",
        "- Y軸: seed",
        "- prompt_keyごとに分ける",
        "- Hiresあり/なしは別画像として保存",
        "- Hiresあり画像は最終見栄え確認用で、Hiresなし基準と直接比較しない",
    ]
    (output_dir / "validation_grid_plan.md").write_text("\n".join(grid_lines) + "\n", encoding="utf-8")
    checklist = [
        "# Validation Checklist",
        "",
        "- weight 0.4で弱すぎないか",
        "- weight 0.6で自然に特徴が出るか",
        "- weight 0.8で安定するか",
        "- weight 1.0で強すぎないか",
        "- full_bodyで破綻しないか",
        "- expression_poseで固定化しないか",
        "- HiresありでLoRAの癖が増幅しすぎないか",
    ]
    (output_dir / "validation_checklist.md").write_text("\n".join(checklist) + "\n", encoding="utf-8")
    return {
        "dir": str(output_dir),
        "prompts_md": str(output_dir / "validation_prompts.md"),
        "prompts_json": str(output_dir / "validation_prompts.json"),
    }


def load_validation_run_bundle(run_id: int) -> dict[str, Any]:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None:
        raise ValueError(f"Validation run not found: {run_id}")
    preset = fetch_one("SELECT * FROM validation_presets WHERE id = ?", (run["validation_preset_id"],)) if run["validation_preset_id"] else None
    images = fetch_all("SELECT * FROM validation_images WHERE validation_run_id = ? ORDER BY image_role, prompt_key, seed, lora_weight, id", (run_id,))
    if preset:
        conditions = expand_preset_conditions(preset, run["trigger_word"] or "", run["lora_filename"] or "", run["base_model"] or "")
    else:
        conditions = []
    registered = {row["condition_hash"] for row in images if row["condition_hash"]}
    missing = [row for row in conditions if row["condition_hash"] not in registered]
    return {
        "run": run,
        "preset": preset,
        "images": images,
        "conditions": conditions,
        "missing": missing,
        "output_dir": str(validation_run_dir(run_id)),
    }


def copy_managed_validation_image(run_id: int, source_path: str) -> str:
    source = Path(source_path)
    if not source.exists() or not source.is_file():
        raise ValueError("Image file not found")
    target_dir = validation_image_dir(run_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    counter = 1
    while target.exists():
        target = target_dir / f"{source.stem}_{counter}{source.suffix}"
        counter += 1
    shutil.copy2(source, target)
    return str(target)


def update_validation_run_counts(run_id: int) -> None:
    row = fetch_one("SELECT COUNT(*) AS count FROM validation_images WHERE validation_run_id = ?", (run_id,))
    count = int(row["count"] if row else 0)
    status = "images_registered" if count else "planned"
    now = utc_now()
    with connect() as conn:
        conn.execute(
            "UPDATE validation_runs SET actual_image_count = ?, status = ?, updated_at = ? WHERE id = ?",
            (count, status, now, run_id),
        )

