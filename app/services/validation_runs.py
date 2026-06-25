from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from app import settings
from app.services.storage_paths import exports_root
from app.db import connect, fetch_all, fetch_one, utc_now
from app.services.image_store import unique_copy


BASELINE_MODE = "no_lora_tag"


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def validation_presets() -> list[Any]:
    return fetch_all(
        """
        SELECT *
        FROM validation_presets
        WHERE is_active = 1
        ORDER BY
            CASE validation_level
                WHEN 'quick' THEN 1
                WHEN 'standard' THEN 2
                WHEN 'extended' THEN 3
                ELSE 99
            END,
            name
        """
    )


def preset_expected_count(preset: Any) -> int:
    prompts = json_loads(preset["prompts_json"], [])
    seeds = json_loads(preset["seeds_json"], [])
    weights = json_loads(preset["weights_json"], [])
    hires_modes = json_loads(preset["hires_modes_json"], [])
    return len(prompts) * len(seeds) * len(weights) * len(hires_modes)


def expand_preset_conditions(
    preset: Any,
    trigger_word: str,
    lora_filename: str,
    base_model: str,
    weights_override: list[float] | None = None,
) -> list[dict[str, Any]]:
    prompts = json_loads(preset["prompts_json"], [])
    seeds = json_loads(preset["seeds_json"], [])
    weights = weights_override if weights_override is not None else json_loads(preset["weights_json"], [])
    hires_modes = json_loads(preset["hires_modes_json"], [])
    lora_name = Path(lora_filename or "selected_lora").stem
    rows: list[dict[str, Any]] = []
    order = 1
    for prompt in prompts:
        prompt_key = prompt.get("prompt_key") or prompt.get("name") or "prompt"
        prompt_text = (prompt.get("prompt") or "").replace("{trigger_word}", trigger_word or "")
        for seed in seeds:
            seed_value = int(seed) + int(prompt.get("seed_offset") or 0)
            for weight in weights:
                for hires_enabled in hires_modes:
                    weight_value = float(weight)
                    webui_prompt = prompt_text
                    if not (weight_value == 0 and BASELINE_MODE == "no_lora_tag"):
                        webui_prompt = f"<lora:{lora_name}:{weight_value:g}>, {prompt_text}"
                    condition = {
                        "validation_preset_id": preset["id"],
                        "preset_version": preset["version"],
                        "preset_name": preset["name"],
                        "prompt_key": prompt_key,
                        "prompt_name": prompt.get("name") or prompt_key,
                        "prompt": prompt_text,
                        "webui_prompt": webui_prompt,
                        "negative_prompt": preset["negative_prompt"] or "",
                        "seed": seed_value,
                        "lora_weight": weight_value,
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
                        "trigger_word": trigger_word or "",
                        "lora_filename": lora_filename or "",
                        "expected_order": order,
                    }
                    condition["condition_hash"] = make_condition_hash(condition)
                    rows.append(condition)
                    order += 1
    return rows


def normalize_validation_weights(values: list[Any]) -> list[float]:
    weights: list[float] = []
    for value in values:
        try:
            weight = round(float(value), 1)
        except (TypeError, ValueError):
            continue
        if 0 <= weight <= 2.0 and abs(weight * 10 - round(weight * 10)) < 0.000001:
            weights.append(weight)
    return sorted(set(weights))


def add_validation_run_weights(run_id: int, weights: list[Any]) -> dict[str, Any]:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None or not run["validation_preset_id"]:
        raise ValueError(f"Validation Run not found: {run_id}")
    preset = validation_preset_for_run(run)
    if preset is None:
        raise ValueError("Validation preset not found")
    selected_weights = normalize_validation_weights(weights)
    if not selected_weights:
        raise ValueError("追加生成するweightを選択してください。")

    existing_hashes = {
        row["condition_hash"]
        for row in fetch_all("SELECT condition_hash FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
    }
    existing_weights = {
        round(float(row["lora_weight"] or 0), 1)
        for row in fetch_all("SELECT DISTINCT lora_weight FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
    }
    max_order_row = fetch_one("SELECT MAX(expected_order) AS max_order FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
    next_order = int(max_order_row["max_order"] or 0) + 1 if max_order_row else 1
    candidates = expand_preset_conditions(
        preset,
        run["trigger_word"] or "",
        run["lora_filename"] or "",
        run["base_model"] or "",
        weights_override=selected_weights,
    )
    new_rows = [row for row in candidates if row["condition_hash"] not in existing_hashes]
    now = utc_now()
    with connect() as conn:
        for row in new_rows:
            conn.execute(
                """
                INSERT INTO validation_expected_conditions(
                    validation_run_id, validation_preset_id, prompt_key, seed,
                    lora_weight, hires_enabled, width, height, sampler, steps,
                    cfg_scale, condition_hash, expected_order, preset_version,
                    prompt, webui_prompt, negative_prompt, trigger_word,
                    lora_filename, base_model, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    row["validation_preset_id"],
                    row["prompt_key"],
                    row["seed"],
                    row["lora_weight"],
                    1 if row["hires_enabled"] else 0,
                    row["width"],
                    row["height"],
                    row["sampler"],
                    row["steps"],
                    row["cfg_scale"],
                    row["condition_hash"],
                    next_order,
                    row.get("preset_version"),
                    row.get("prompt"),
                    row.get("webui_prompt"),
                    row.get("negative_prompt"),
                    row.get("trigger_word"),
                    row.get("lora_filename"),
                    row.get("base_model"),
                    now,
                ),
            )
            next_order += 1
        expected_count = conn.execute(
            "SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ?",
            (run_id,),
        ).fetchone()["count"]
        conn.execute(
            "UPDATE validation_runs SET expected_image_count = ?, updated_at = ? WHERE id = ?",
            (expected_count, now, run_id),
        )
    relink_validation_images(run_id)
    added_weights = sorted(set(selected_weights) - existing_weights)
    return {
        "selected_weights": selected_weights,
        "added_conditions": len(new_rows),
        "added_weights": added_weights,
        "existing_weights": sorted(existing_weights),
    }


def make_condition_hash(data: dict[str, Any]) -> str:
    keys = [
        "validation_preset_id",
        "preset_version",
        "prompt_key",
        "prompt",
        "webui_prompt",
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
        "trigger_word",
        "lora_filename",
        "base_model",
    ]
    payload = {key: normalize_hash_value(data.get(key)) for key in keys}
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_hash_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def create_validation_run(
    job_id: int,
    validation_preset_id: str,
    base_model: str,
    trigger_word: str,
    memo: str,
    profile_id: int | None = None,
    selected_output_id: int | None = None,
) -> int:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    if selected_output_id is not None:
        selected_output = fetch_one(
            "SELECT * FROM training_outputs WHERE id = ? AND job_id = ? AND file_type = 'model'",
            (selected_output_id, job_id),
        )
        if selected_output is None:
            raise ValueError(f"LoRA output not found: {selected_output_id}")
    else:
        selected_output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    latest_profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE job_id = ? ORDER BY updated_at DESC, id DESC LIMIT 1", (job_id,))
    if profile_id is None and selected_output_id is not None:
        profile = fetch_one(
            "SELECT * FROM selected_lora_profiles WHERE job_id = ? AND selected_output_id = ? ORDER BY updated_at DESC, id DESC LIMIT 1",
            (job_id, selected_output_id),
        )
    elif profile_id is None:
        profile = latest_profile
    else:
        profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (profile_id,))
    profile_defaults = profile or latest_profile
    preset = fetch_one("SELECT * FROM validation_presets WHERE id = ?", (validation_preset_id,))
    if preset is None:
        raise ValueError(f"Validation preset not found: {validation_preset_id}")
    if selected_output is None and profile_defaults is not None and profile_defaults["selected_output_id"]:
        selected_output = fetch_one("SELECT * FROM training_outputs WHERE id = ?", (profile_defaults["selected_output_id"],))
    lora_filename = Path(
        (selected_output["file_path"] if selected_output else "")
        or (profile_defaults["selected_model_path"] if profile_defaults else "")
        or "selected_lora.safetensors"
    ).name
    base_model_value = base_model.strip() or (profile_defaults["base_model"] if profile_defaults else "") or Path(job["base_model_path"]).stem
    trigger_value = trigger_word.strip() or (profile_defaults["trigger_word"] if profile_defaults else "") or job["trigger_word_at_creation"] or ""
    reference_set_id = profile_defaults["reference_set_id"] if profile_defaults and "reference_set_id" in profile_defaults.keys() else None
    reference_set_version_id = profile_defaults["reference_set_version_id"] if profile_defaults and "reference_set_version_id" in profile_defaults.keys() else None
    if reference_set_id and not reference_set_version_id:
        ref = fetch_one("SELECT current_version_id FROM reference_sets WHERE id = ?", (reference_set_id,))
        reference_set_version_id = ref["current_version_id"] if ref else None
    expected = preset_expected_count(preset)
    preset_snapshot = json.dumps(dict(preset), ensure_ascii=False, sort_keys=True)
    now = utc_now()
    epoch_suffix = f" epoch {selected_output['epoch']}" if selected_output is not None and selected_output["epoch"] is not None else ""
    name = f"Job #{job_id} {preset['name']}{epoch_suffix}"
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO validation_runs(
                project_id, job_id, selected_output_id, selected_lora_profile_id,
                validation_run_kind, source_training_job_id, selected_epoch,
                pipeline_status, validation_preset_id, name, validation_level, base_model,
                trigger_word, lora_filename, recommended_weight_min,
                recommended_weight_max, expected_image_count, actual_image_count,
                status, preset_snapshot_json, reference_set_id, reference_set_version_id,
                created_at, updated_at, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'planned', ?, ?, ?, ?, ?, ?)
            """,
            (
                job["project_id"] if "project_id" in job.keys() else None,
                job_id,
                selected_output["id"] if selected_output else profile["selected_output_id"] if profile else None,
                profile["id"] if profile else profile_id,
                "weight_calibration",
                job_id,
                selected_output["epoch"] if selected_output is not None else profile["selected_epoch"] if profile and "selected_epoch" in profile.keys() else None,
                "planned",
                preset["id"],
                name,
                preset["validation_level"],
                base_model_value,
                trigger_value,
                lora_filename,
                profile_defaults["recommended_weight_min"] if profile_defaults else None,
                profile_defaults["recommended_weight_max"] if profile_defaults else None,
                expected,
                preset_snapshot,
                reference_set_id,
                reference_set_version_id,
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
    ensure_expected_conditions(run_id)
    write_validation_prompt_pack(run_id)
    return run_id


def validation_run_dir(run_id: int) -> Path:
    return exports_root() / "validation_runs" / f"validation_run_{run_id:06d}"


def validation_image_dir(run_id: int) -> Path:
    return validation_run_dir(run_id) / "images"


def validation_preset_for_run(run: Any) -> dict[str, Any] | Any | None:
    snapshot = run["preset_snapshot_json"] if "preset_snapshot_json" in run.keys() else None
    if snapshot:
        data = json_loads(snapshot, {})
        if data:
            return data
    if not run["validation_preset_id"]:
        return None
    return fetch_one("SELECT * FROM validation_presets WHERE id = ?", (run["validation_preset_id"],))


def snapshot_validation_preset(preset: Any) -> str:
    return json.dumps(dict(preset), ensure_ascii=False, sort_keys=True)


def backfill_validation_runs() -> None:
    runs = fetch_all("SELECT * FROM validation_runs WHERE validation_preset_id IS NOT NULL ORDER BY id")
    for run in runs:
        backfill_validation_run(run)


def backfill_validation_run(run: Any) -> None:
    preset = validation_preset_for_run(run)
    if preset is None:
        return
    current_preset = fetch_one("SELECT * FROM validation_presets WHERE id = ?", (run["validation_preset_id"],))
    now = utc_now()
    if not (run["preset_snapshot_json"] if "preset_snapshot_json" in run.keys() else None) and current_preset is not None:
        with connect() as conn:
            conn.execute(
                "UPDATE validation_runs SET preset_snapshot_json = ?, updated_at = ? WHERE id = ?",
                (snapshot_validation_preset(current_preset), now, run["id"]),
            )
        run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run["id"],))
        preset = validation_preset_for_run(run)
        if preset is None:
            return
    image_count = fetch_one("SELECT COUNT(*) AS count FROM validation_images WHERE validation_run_id = ?", (run["id"],))["count"]
    condition_count = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ?", (run["id"],))["count"]
    expected = preset_expected_count(preset)
    if int(image_count) == 0 and int(condition_count) not in {0, expected}:
        with connect() as conn:
            conn.execute("DELETE FROM validation_expected_conditions WHERE validation_run_id = ?", (run["id"],))
        ensure_expected_conditions(int(run["id"]))
        return
    if int(condition_count) == 0:
        ensure_expected_conditions(int(run["id"]))
        return
    expanded = {
        int(row["expected_order"]): row
        for row in expand_preset_conditions(preset, run["trigger_word"] or "", run["lora_filename"] or "", run["base_model"] or "")
    }
    rows = fetch_all("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order", (run["id"],))
    with connect() as conn:
        for row in rows:
            source = expanded.get(int(row["expected_order"]))
            if not source:
                continue
            conn.execute(
                """
                UPDATE validation_expected_conditions
                SET preset_version = COALESCE(preset_version, ?),
                    prompt = COALESCE(NULLIF(prompt, ''), ?),
                    webui_prompt = COALESCE(NULLIF(webui_prompt, ''), ?),
                    negative_prompt = COALESCE(negative_prompt, ?),
                    trigger_word = COALESCE(trigger_word, ?),
                    lora_filename = COALESCE(lora_filename, ?),
                    base_model = COALESCE(base_model, ?)
                WHERE id = ?
                """,
                (
                    source.get("preset_version"),
                    source.get("prompt"),
                    source.get("webui_prompt"),
                    source.get("negative_prompt"),
                    source.get("trigger_word"),
                    source.get("lora_filename"),
                    source.get("base_model"),
                    row["id"],
                ),
            )


def ensure_expected_conditions(run_id: int) -> list[Any]:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None or not run["validation_preset_id"]:
        return []
    preset = validation_preset_for_run(run)
    if preset is None:
        return []
    existing_rows = fetch_all("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order", (run_id,))
    condition_count = len(existing_rows)
    expected_count = int(run["expected_image_count"] or 0) or preset_expected_count(preset)
    if condition_count > 0 and condition_count == expected_count:
        return existing_rows
    image_count_row = fetch_one("SELECT COUNT(*) AS count FROM validation_images WHERE validation_run_id = ?", (run_id,))
    image_count = int(image_count_row["count"] if image_count_row else 0)
    if image_count > 0 and condition_count != expected_count:
        return existing_rows
    conditions = expand_preset_conditions(preset, run["trigger_word"] or "", run["lora_filename"] or "", run["base_model"] or "")
    now = utc_now()
    with connect() as conn:
        conn.execute("DELETE FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
        conn.executemany(
            """
            INSERT INTO validation_expected_conditions(
                validation_run_id, validation_preset_id, prompt_key, seed,
                lora_weight, hires_enabled, width, height, sampler, steps,
                cfg_scale, condition_hash, expected_order, preset_version,
                prompt, webui_prompt, negative_prompt, trigger_word,
                lora_filename, base_model, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    row["validation_preset_id"],
                    row["prompt_key"],
                    row["seed"],
                    row["lora_weight"],
                    1 if row["hires_enabled"] else 0,
                    row["width"],
                    row["height"],
                    row["sampler"],
                    row["steps"],
                    row["cfg_scale"],
                    row["condition_hash"],
                    row["expected_order"],
                    row.get("preset_version"),
                    row.get("prompt"),
                    row.get("webui_prompt"),
                    row.get("negative_prompt"),
                    row.get("trigger_word"),
                    row.get("lora_filename"),
                    row.get("base_model"),
                    now,
                )
                for row in conditions
            ],
        )
        conn.execute(
            "UPDATE validation_runs SET expected_image_count = ?, updated_at = ? WHERE id = ?",
            (len(conditions), now, run_id),
        )
    relink_validation_images(run_id)
    return fetch_all("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order", (run_id,))


def expected_condition_mismatch_warning(run_id: int) -> str:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None or not run["validation_preset_id"]:
        return ""
    preset = validation_preset_for_run(run)
    expected_count = int(run["expected_image_count"] or 0) or (preset_expected_count(preset) if preset else 0)
    condition_row = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
    image_row = fetch_one("SELECT COUNT(*) AS count FROM validation_images WHERE validation_run_id = ?", (run_id,))
    condition_count = int(condition_row["count"] if condition_row else 0)
    image_count = int(image_row["count"] if image_row else 0)
    if image_count > 0 and expected_count and condition_count != expected_count:
        return (
            "Expected Condition count mismatch: "
            f"expected {expected_count}, actual {condition_count}. "
            "既存画像が登録済みのため、condition_hashを維持して自動再生成は行いません。"
        )
    return ""


def write_validation_prompt_pack(run_id: int) -> dict[str, str]:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None:
        raise ValueError(f"Validation run not found: {run_id}")
    preset = validation_preset_for_run(run)
    if preset is None:
        raise ValueError("Validation preset not found")
    output_dir = validation_run_dir(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [dict(row) for row in ensure_expected_conditions(run_id)]
    if not conditions:
        conditions = expand_preset_conditions(preset, run["trigger_word"] or "", run["lora_filename"] or "", run["base_model"] or "")
    md_lines = [
        f"# Validation Prompts Run #{run_id}",
        "",
        f"- Preset: {preset['name']}",
        f"- Level: {preset['validation_level']}",
        f"- Base model: {run['base_model'] or '-'}",
        f"- Trigger: {run['trigger_word'] or '-'}",
        f"- LoRA: {Path(run['lora_filename'] or 'selected_lora').stem}",
        f"- Baseline mode: {BASELINE_MODE}",
        "",
        "weight 0 はベースモデル比較用です。標準ではLoRAタグを付けません。",
        "",
    ]
    for index, row in enumerate(conditions, start=1):
        md_lines.extend(
            [
                f"## {index}. {row['prompt_key']} / seed {row['seed']} / weight {float(row['lora_weight']):g} / Hires {bool(row['hires_enabled'])}",
                "",
                "Prompt:",
                "```text",
                row.get("webui_prompt") or row.get("prompt") or "",
                "```",
                "Negative:",
                "```text",
                row.get("negative_prompt") or "",
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
        "- Grid画像は便利ですが、条件ごとの個別レビューにはIndividual画像が望ましいです。",
        "- Hiresあり画像は最終見栄え確認用で、Hiresなし基準と直接比較しません。",
    ]
    (output_dir / "validation_grid_plan.md").write_text("\n".join(grid_lines) + "\n", encoding="utf-8")
    checklist = [
        "# Validation Checklist",
        "",
        "- weight 0はLoRAなしのベースモデル比較として見る",
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


def load_validation_run_bundle(run_id: int, *, include_images: bool = True) -> dict[str, Any]:
    ensure_expected_conditions(run_id)
    update_validation_run_counts(run_id)
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None:
        raise ValueError(f"Validation run not found: {run_id}")
    preset = validation_preset_for_run(run) if run["validation_preset_id"] else None
    conditions = fetch_all("SELECT * FROM validation_expected_conditions WHERE validation_run_id = ? ORDER BY expected_order", (run_id,))
    if include_images:
        images = fetch_all("SELECT * FROM validation_images WHERE validation_run_id = ? ORDER BY image_role, prompt_key, seed, lora_weight, id", (run_id,))
        coverage = build_coverage(run_id, conditions, images)
        review_rows = images
    else:
        images = []
        coverage = build_coverage_from_db(run_id, conditions)
        review_rows = fetch_validation_review_rows(run_id)
    weight_matrix = build_weight_review_matrix(review_rows)
    suggestion = calculate_suggested_weights(run, preset, review_rows)
    profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (run["selected_lora_profile_id"],)) if run["selected_lora_profile_id"] else None
    reference_set = None
    reference_images = []
    reference_set_id = run["reference_set_id"] if "reference_set_id" in run.keys() and run["reference_set_id"] else None
    reference_set_version_id = run["reference_set_version_id"] if "reference_set_version_id" in run.keys() and run["reference_set_version_id"] else None
    if not reference_set_id and profile and profile["reference_set_id"]:
        reference_set_id = profile["reference_set_id"]
        reference_set_version_id = profile["reference_set_version_id"] if "reference_set_version_id" in profile.keys() else None
    if reference_set_id:
        reference_set = fetch_one(
            """
            SELECT r.*, v.version_no AS current_version_no, v.completeness_label, v.completeness_message
            FROM reference_sets r
            LEFT JOIN reference_set_versions v ON v.id = COALESCE(?, r.current_version_id) AND v.reference_set_id = r.id
            WHERE r.id = ?
            """,
            (reference_set_version_id, reference_set_id),
        )
        if reference_set_version_id:
            reference_images = fetch_all("SELECT * FROM reference_images WHERE reference_set_version_id = ? ORDER BY sort_order, id", (reference_set_version_id,))
        else:
            reference_images = fetch_all("SELECT * FROM reference_images WHERE reference_set_id = ? ORDER BY sort_order, id", (reference_set_id,))
    return {
        "run": run,
        "preset": preset,
        "images": images,
        "conditions": conditions,
        "coverage": coverage,
        "missing": [row for row in coverage["rows"] if row["status"] == "missing"],
        "unmatched_images": [dict(row) for row in images if row["image_role"] == "individual" and not row["expected_condition_id"]],
        "grid_images": [dict(row) for row in images if row["image_role"] == "grid"],
        "weight_matrix": weight_matrix,
        "suggestion": suggestion,
        "profile": profile,
        "reference_set": reference_set,
        "reference_images": reference_images,
        "output_dir": str(validation_run_dir(run_id)),
        "condition_warning": expected_condition_mismatch_warning(run_id),
    }


def fetch_validation_review_rows(run_id: int) -> list[Any]:
    return fetch_all(
        """
        SELECT
            id, image_role, expected_condition_id, lora_weight, hires_enabled, ignored,
            rating_face, rating_costume, rating_style, rating_stability, rating_flexibility, rating_overall,
            strength_label, overfit_level, adoption_label, failure_tags_json
        FROM validation_images
        WHERE validation_run_id = ?
        ORDER BY image_role, lora_weight, id
        """,
        (run_id,),
    )


def build_coverage_from_db(run_id: int, conditions: list[Any]) -> dict[str, Any]:
    rows_by_condition = {
        int(row["expected_condition_id"]): row
        for row in fetch_all(
            """
            SELECT
                expected_condition_id,
                COUNT(*) AS image_count,
                SUM(CASE WHEN ignored THEN 1 ELSE 0 END) AS ignored_count,
                SUM(CASE
                    WHEN COALESCE(rating_face, 0) > 0
                      OR COALESCE(rating_costume, 0) > 0
                      OR COALESCE(rating_style, 0) > 0
                      OR COALESCE(rating_stability, 0) > 0
                      OR COALESCE(rating_flexibility, 0) > 0
                      OR COALESCE(rating_overall, 0) > 0
                      OR COALESCE(strength_label, '') != ''
                      OR COALESCE(overfit_level, '') != ''
                      OR COALESCE(adoption_label, '') != ''
                      OR COALESCE(failure_tags_json, '') NOT IN ('', '[]')
                    THEN 1 ELSE 0 END) AS reviewed_count
            FROM validation_images
            WHERE validation_run_id = ?
              AND image_role = 'individual'
              AND expected_condition_id IS NOT NULL
            GROUP BY expected_condition_id
            """,
            (run_id,),
        )
    }
    rows = []
    registered = 0
    reviewed = 0
    ignored = 0
    for condition in conditions:
        item = dict(condition)
        linked = rows_by_condition.get(int(condition["id"]))
        image_count = int(linked["image_count"] if linked else 0)
        ignored_count = int(linked["ignored_count"] if linked else 0)
        reviewed_count = int(linked["reviewed_count"] if linked else 0)
        if image_count <= 0:
            status = "missing"
        elif ignored_count >= image_count:
            status = "ignored"
            ignored += 1
        elif reviewed_count > 0:
            status = "reviewed"
            registered += 1
            reviewed += 1
        else:
            status = "image_registered"
            registered += 1
        item["status"] = status
        item["image_count"] = image_count
        rows.append(item)
    expected = len(conditions)
    missing = expected - registered - ignored
    return {
        "rows": rows,
        "expected_image_count": expected,
        "registered_condition_count": registered,
        "reviewed_condition_count": reviewed,
        "missing_condition_count": max(0, missing),
        "ignored_condition_count": ignored,
        "coverage_rate": registered / expected if expected else 0,
        "review_rate": reviewed / expected if expected else 0,
    }


def build_coverage(run_id: int, conditions: list[Any], images: list[Any]) -> dict[str, Any]:
    images_by_condition: dict[int, list[Any]] = defaultdict(list)
    for image in images:
        if image["image_role"] == "individual" and image["expected_condition_id"]:
            images_by_condition[int(image["expected_condition_id"])].append(image)
    rows = []
    registered = 0
    reviewed = 0
    ignored = 0
    for condition in conditions:
        linked = images_by_condition.get(int(condition["id"]), [])
        if not linked:
            status = "missing"
        elif all(image["ignored"] for image in linked):
            status = "ignored"
            ignored += 1
        elif any(image_is_reviewed(image) for image in linked):
            status = "reviewed"
            registered += 1
            reviewed += 1
        else:
            status = "image_registered"
            registered += 1
        item = dict(condition)
        item["status"] = status
        item["image_count"] = len(linked)
        rows.append(item)
    expected = len(conditions)
    missing = expected - registered - ignored
    return {
        "rows": rows,
        "expected_image_count": expected,
        "registered_condition_count": registered,
        "reviewed_condition_count": reviewed,
        "missing_condition_count": max(0, missing),
        "ignored_condition_count": ignored,
        "coverage_rate": registered / expected if expected else 0,
        "review_rate": reviewed / expected if expected else 0,
    }


def image_is_reviewed(image: Any) -> bool:
    rating_keys = ["rating_face", "rating_costume", "rating_style", "rating_stability", "rating_flexibility", "rating_overall"]
    if any(rating_is_positive(image[key]) for key in rating_keys if key in image.keys()):
        return True
    if any(str(image[key] or "").strip() for key in ("strength_label", "overfit_level", "adoption_label") if key in image.keys()):
        return True
    return has_failure_tags(image["failure_tags_json"] if "failure_tags_json" in image.keys() else None)


def has_failure_tags(value: str | None) -> bool:
    if not value:
        return False
    try:
        tags = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return False
    return bool(tags)


def rating_is_positive(value: Any) -> bool:
    try:
        return value is not None and int(value) > 0
    except (TypeError, ValueError):
        return False


def build_weight_review_matrix(images: list[Any]) -> dict[str, Any]:
    individual = [image for image in images if image["image_role"] == "individual" and image["lora_weight"] is not None and not image["ignored"]]
    rows = [image for image in individual if image_is_reviewed(image)]
    return {
        "by_weight": aggregate_by(rows, "lora_weight"),
        "by_prompt": aggregate_by(rows, "prompt_key"),
        "by_seed": aggregate_by(rows, "seed"),
        "by_hires": aggregate_by(rows, "hires_enabled"),
        "strength_distribution": dict(Counter(image["strength_label"] for image in rows if image["strength_label"])),
        "overfit_distribution": dict(Counter(image["overfit_level"] for image in rows if image["overfit_level"])),
        "adoption_distribution": dict(Counter(image["adoption_label"] for image in rows if image["adoption_label"])),
        "recommended_weights": sorted(
            {float(image["lora_weight"]) for image in rows if float(image["lora_weight"]) > 0 and image["strength_label"] in {"recommended", "strong_but_usable"}}
        ),
        "too_strong_weights": sorted({float(image["lora_weight"]) for image in rows if float(image["lora_weight"]) > 0 and image["strength_label"] in {"too_strong", "broken"}}),
        "too_weak_weights": sorted({float(image["lora_weight"]) for image in rows if float(image["lora_weight"]) > 0 and image["strength_label"] == "too_weak"}),
    }


def aggregate_by(rows: list[Any], key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    result = []
    for value, items in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        overall = [int(item["rating_overall"]) for item in items if item["rating_overall"] is not None and int(item["rating_overall"]) > 0]
        result.append(
            {
                "key": value,
                "count": len(items),
                "avg_overall": mean(overall) if overall else None,
                "strength_distribution": dict(Counter(item["strength_label"] for item in items if item["strength_label"])),
                "overfit_distribution": dict(Counter(item["overfit_level"] for item in items if item["overfit_level"])),
                "adoption_distribution": dict(Counter(item["adoption_label"] for item in items if item["adoption_label"])),
            }
        )
    return result


def calculate_suggested_weights(run: Any, preset: Any, images: list[Any]) -> dict[str, Any]:
    recommended = []
    fallback = []
    light_candidates = []
    strong_candidates = []
    for image in images:
        if image["image_role"] != "individual" or image["lora_weight"] is None or image["ignored"]:
            continue
        if image["hires_enabled"]:
            continue
        if preset and preset["validation_level"] not in {"quick", "standard"}:
            continue
        if not image_is_reviewed(image):
            continue
        strength = image["strength_label"] or ""
        adoption = image["adoption_label"] or ""
        overall = int(image["rating_overall"] or 0)
        weight = float(image["lora_weight"])
        if weight == 0:
            continue
        if strength in {"too_weak", "too_strong", "broken"}:
            continue
        if adoption == "reject":
            continue
        if strength == "weak_but_usable":
            light_candidates.append(weight)
        if strength == "strong_but_usable":
            strong_candidates.append(weight)
        if strength == "recommended" or adoption == "adopt":
            recommended.append((weight, strength, adoption, overall))
        elif adoption == "candidate" or overall >= 3 or strength in {"weak_but_usable", "strong_but_usable"}:
            fallback.append((weight, strength, adoption, overall))
    weights = sorted({row[0] for row in recommended})
    fallback_weights = sorted({row[0] for row in fallback})
    all_weights = sorted({float(image["lora_weight"]) for image in images if image["image_role"] == "individual" and image["lora_weight"] is not None})
    if preset is not None:
        expected_weights = [float(value) for value in json_loads(preset["weights_json"], [])]
        all_weights = sorted(set(all_weights) | set(expected_weights))
    too_strong = sorted({float(image["lora_weight"]) for image in images if image["strength_label"] in {"too_strong", "broken"} and image["lora_weight"] is not None})
    too_weak = sorted({float(image["lora_weight"]) for image in images if image["strength_label"] == "too_weak" and image["lora_weight"] is not None})
    if weights:
        min_weight, max_weight = min(weights), max(weights)
    elif fallback_weights:
        min_weight, max_weight = min(fallback_weights), max(fallback_weights)
    else:
        min_weight = run["recommended_weight_min"]
        max_weight = run["recommended_weight_max"]
    if weights:
        reason = "HiresなしのQuick/Standard評価で、recommended/adoptのweightを推奨範囲にしました。weak/strong候補は補助範囲に分けています。"
    elif fallback_weights:
        reason = "recommended/adoptが未入力のため、candidateまたはrating 3以上のweightを暫定推奨範囲にしました。"
    else:
        reason = "評価済み条件が少ないため、既存Profile/Runの推奨weightを維持します。"
    light = min([weight for weight in all_weights if weight > 0], default=min_weight)
    strong = max([weight for weight in all_weights if weight > 0], default=max_weight)
    if light_candidates:
        light = min(light_candidates)
    if strong_candidates:
        strong = max(strong_candidates)
    if too_weak:
        light = min(too_weak)
    if too_strong:
        strong = max(too_strong)
    return {
        "suggested_weight_min": min_weight,
        "suggested_weight_max": max_weight,
        "suggested_light_weight": light,
        "suggested_strong_weight": strong,
        "suggested_weight_reason": reason,
    }


def persist_suggestion(run_id: int) -> dict[str, Any]:
    bundle = load_validation_run_bundle(run_id)
    suggestion = bundle["suggestion"]
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE validation_runs
            SET suggested_weight_min = ?, suggested_weight_max = ?,
                suggested_light_weight = ?, suggested_strong_weight = ?,
                suggested_weight_reason = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                suggestion["suggested_weight_min"],
                suggestion["suggested_weight_max"],
                suggestion["suggested_light_weight"],
                suggestion["suggested_strong_weight"],
                suggestion["suggested_weight_reason"],
                now,
                run_id,
            ),
        )
    update_validation_run_counts(run_id)
    return suggestion


def apply_suggestion_to_profile(run_id: int) -> dict[str, Any]:
    suggestion = persist_suggestion(run_id)
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None or not run["selected_lora_profile_id"]:
        raise ValueError("Validation Run is not linked to a LoRA Profile.")
    profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (run["selected_lora_profile_id"],))
    if profile is None:
        raise ValueError("LoRA Profile not found.")
    before = dict(profile)
    policy = run["suggested_weight_reason"] or suggestion["suggested_weight_reason"]
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE selected_lora_profiles
            SET recommended_weight_min = ?, recommended_weight_max = ?,
                light_weight = ?, strong_weight = ?,
                last_validation_preset_id = ?, validation_policy_memo = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                suggestion["suggested_weight_min"],
                suggestion["suggested_weight_max"],
                suggestion["suggested_light_weight"],
                suggestion["suggested_strong_weight"],
                run["validation_preset_id"],
                policy,
                now,
                profile["id"],
            ),
        )
        conn.execute(
            "UPDATE validation_runs SET status = 'completed', profile_applied_at = ?, updated_at = ? WHERE id = ?",
            (now, now, run_id),
        )
    after = fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (profile["id"],))
    return {"before": before, "after": dict(after), "suggestion": suggestion}


def copy_managed_validation_image(run_id: int, source_path: str) -> str:
    target = unique_copy(Path(source_path), validation_image_dir(run_id))
    return str(target)


def relink_validation_images(run_id: int) -> None:
    expected = {row["condition_hash"]: row["id"] for row in fetch_all("SELECT id, condition_hash FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))}
    images = fetch_all("SELECT id, condition_hash FROM validation_images WHERE validation_run_id = ? AND image_role = 'individual'", (run_id,))
    with connect() as conn:
        for image in images:
            conn.execute(
                "UPDATE validation_images SET expected_condition_id = ? WHERE id = ?",
                (expected.get(image["condition_hash"]), image["id"]),
            )


def update_validation_run_counts(run_id: int) -> None:
    row = fetch_one("SELECT COUNT(*) AS count FROM validation_images WHERE validation_run_id = ?", (run_id,))
    count = int(row["count"] if row else 0)
    conditions = ensure_expected_conditions(run_id)
    images = fetch_all("SELECT * FROM validation_images WHERE validation_run_id = ?", (run_id,))
    coverage = build_coverage(run_id, conditions, images)
    if count == 0:
        status = "planned"
    elif coverage["reviewed_condition_count"] == 0:
        status = "images_registered"
    elif coverage["reviewed_condition_count"] < max(1, coverage["registered_condition_count"]):
        status = "partially_reviewed"
    elif coverage["missing_condition_count"] == 0 or coverage["reviewed_condition_count"] >= coverage["registered_condition_count"]:
        status = "reviewed"
    else:
        status = "partially_reviewed"
    current = fetch_one("SELECT status FROM validation_runs WHERE id = ?", (run_id,))
    if current and current["status"] in {"completed", "archived", "failed", "stopped"}:
        status = current["status"]
    now = utc_now()
    with connect() as conn:
        conn.execute(
            "UPDATE validation_runs SET actual_image_count = ?, status = ?, updated_at = ? WHERE id = ?",
            (count, status, now, run_id),
        )


def update_validation_run_status(run_id: int, status: str) -> None:
    allowed = {"planned", "images_registered", "partially_reviewed", "reviewed", "completed", "failed", "stopped", "archived"}
    if status not in allowed:
        raise ValueError(f"Invalid status: {status}")
    now = utc_now()
    with connect() as conn:
        conn.execute("UPDATE validation_runs SET status = ?, updated_at = ? WHERE id = ?", (status, now, run_id))


def write_validation_report(run_id: int) -> str:
    bundle = load_validation_run_bundle(run_id)
    run = bundle["run"]
    preset = bundle["preset"]
    coverage = bundle["coverage"]
    matrix = bundle["weight_matrix"]
    suggestion = persist_suggestion(run_id)
    report_path = validation_run_dir(run_id) / "validation_report.md"
    lines = [
        f"# Validation Report Run #{run_id}",
        "",
        "## Run",
        f"- name: {run['name']}",
        f"- status: {run['status']}",
        f"- preset: {preset['name'] if preset else 'legacy / preset unspecified'}",
        f"- level: {run['validation_level'] or '-'}",
        f"- base model: {run['base_model'] or '-'}",
        f"- trigger: {run['trigger_word'] or '-'}",
        "",
        "## Coverage",
        f"- expected: {coverage['expected_image_count']}",
        f"- registered: {coverage['registered_condition_count']}",
        f"- reviewed: {coverage['reviewed_condition_count']}",
        f"- missing: {coverage['missing_condition_count']}",
        f"- coverage_rate: {coverage['coverage_rate']:.1%}",
        f"- review_rate: {coverage['review_rate']:.1%}",
        "",
        "## Suggested Weight",
        f"- min: {suggestion['suggested_weight_min']}",
        f"- max: {suggestion['suggested_weight_max']}",
        f"- light: {suggestion['suggested_light_weight']}",
        f"- strong: {suggestion['suggested_strong_weight']}",
        f"- reason: {suggestion['suggested_weight_reason']}",
        "",
        "## Weight Review Summary",
    ]
    for row in matrix["by_weight"]:
        avg = f"{row['avg_overall']:.2f}" if row["avg_overall"] is not None else "-"
        lines.append(f"- weight {row['key']}: count={row['count']} avg_overall={avg} strength={row['strength_distribution']} adoption={row['adoption_distribution']}")
    lines.extend(
        [
            "",
            "## Grid Images",
            *[f"- #{image['id']}: {Path(image['image_path']).name} / {image['memo'] or '-'}" for image in bundle["grid_images"]],
            "",
            "## Individual Images",
            *[f"- #{image['id']}: {Path(image['image_path']).name} / {image['prompt_key'] or '-'} / weight {image['lora_weight']}" for image in bundle["images"] if image["image_role"] == "individual"],
            "",
            "## Notes",
            "- Hiresあり結果は最終見栄え確認として扱い、標準比較はHiresなしで行います。",
            "- weight 0 はLoRAなしのベースモデル比較です。",
            "- Grid画像は参考資料です。条件ごとのレビューにはIndividual画像が望ましいです。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)
