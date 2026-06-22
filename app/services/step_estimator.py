from __future__ import annotations

import math
from typing import Any


DEFAULT_TARGETS = {
    "target_steps_min": 2500,
    "target_steps_recommended": 5000,
    "target_steps_max": 8000,
    "target_checkpoint_count": 6,
    "step_target_note": "Global fallback target.",
    "step_target_source": "global",
    "recipe_target_steps_min": None,
    "recipe_target_steps_recommended": None,
    "recipe_target_steps_max": None,
    "optimizer_target_steps_min": None,
    "optimizer_target_steps_recommended": None,
    "optimizer_target_steps_max": None,
    "optimizer_name": "",
    "optimizer_lr_meaning": "",
    "optimizer_category": "",
    "recipe_name": "",
}


def int_or_default(value: Any, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def optional_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def target_config_from_preset(preset: Any | None) -> dict[str, Any]:
    if preset is None:
        return dict(DEFAULT_TARGETS)
    result = dict(DEFAULT_TARGETS)
    keys = preset.keys() if hasattr(preset, "keys") else preset
    for key in result:
        if key in keys:
            result[key] = preset[key]
    return result


def target_config_from_catalog(
    *,
    training_recipe: Any | None = None,
    optimizer_profile: Any | None = None,
    optimizer_definition: Any | None = None,
    preset: Any | None = None,
) -> dict[str, Any]:
    result = dict(DEFAULT_TARGETS)

    def copy_target(prefix: str, source: Any | None) -> None:
        if source is None:
            return
        keys = source.keys() if hasattr(source, "keys") else source
        for key in ("target_steps_min", "target_steps_recommended", "target_steps_max"):
            if key in keys:
                result[f"{prefix}_{key}"] = source[key]

    copy_target("recipe", training_recipe)
    copy_target("optimizer", optimizer_profile or optimizer_definition)

    sources = (
        ("recipe", training_recipe),
        ("optimizer_profile", optimizer_profile),
        ("optimizer_definition", optimizer_definition),
        ("preset_compat", preset),
        ("global", DEFAULT_TARGETS),
    )
    for source_name, source in sources:
        if source is None:
            continue
        keys = source.keys() if hasattr(source, "keys") else source
        min_steps = source["target_steps_min"] if "target_steps_min" in keys else None
        recommended_steps = source["target_steps_recommended"] if "target_steps_recommended" in keys else None
        max_steps = source["target_steps_max"] if "target_steps_max" in keys else None
        if recommended_steps not in (None, ""):
            result["target_steps_min"] = min_steps
            result["target_steps_recommended"] = recommended_steps
            result["target_steps_max"] = max_steps
            result["step_target_source"] = source_name
            if "target_checkpoint_count" in keys and source["target_checkpoint_count"] not in (None, ""):
                result["target_checkpoint_count"] = source["target_checkpoint_count"]
            if "step_target_note" in keys:
                result["step_target_note"] = source["step_target_note"]
            elif "note" in keys:
                result["step_target_note"] = source["note"]
            break

    if training_recipe is not None:
        keys = training_recipe.keys() if hasattr(training_recipe, "keys") else training_recipe
        if "name" in keys:
            result["recipe_name"] = training_recipe["name"]
    if optimizer_definition is not None:
        keys = optimizer_definition.keys() if hasattr(optimizer_definition, "keys") else optimizer_definition
        for source_key, result_key in (
            ("name", "optimizer_name"),
            ("lr_meaning", "optimizer_lr_meaning"),
            ("category", "optimizer_category"),
        ):
            if source_key in keys:
                result[result_key] = optimizer_definition[source_key]
    return result


def subset_weighted_image_count(subsets: list[dict[str, Any]] | None, image_count: int, repeats: int) -> int:
    if subsets:
        # TODO: support ratio-preserving per-subset repeats. The initial assistant applies one repeats value to every subset.
        total = 0
        for subset in subsets:
            count = int_or_default(subset.get("image_count"), 0)
            repeat = int_or_default(subset.get("num_repeats", subset.get("repeats")), repeats)
            total += max(0, count) * max(0, repeat)
        return total
    return max(0, image_count) * max(0, repeats)


def estimate_steps(
    *,
    image_count: int,
    params: dict[str, Any],
    target: dict[str, Any] | None = None,
    subsets: list[dict[str, Any]] | None = None,
    num_processes: int | None = None,
) -> dict[str, Any]:
    repeats = int_or_default(params.get("repeats"), 1)
    epochs = int_or_default(params.get("max_train_epochs"), 1)
    batch = int_or_default(params.get("train_batch_size"), 1)
    grad_accum = int_or_default(params.get("gradient_accumulation_steps"), 1)
    process_count = max(1, int_or_default(num_processes if num_processes is not None else params.get("num_processes"), 1))
    effective_batch = batch * grad_accum * process_count
    max_train_steps = optional_int(params.get("max_train_steps"))
    weighted_images = subset_weighted_image_count(subsets, image_count, repeats)

    warnings: list[str] = []
    if image_count <= 0 and not subsets:
        warnings.append("image_count=0です。データセット画像数を確認してください。")
    if effective_batch <= 0:
        warnings.append("effective_batch_sizeが0以下です。batch / gradient_accumulation / num_processesを確認してください。")
        steps_per_epoch = 0
    else:
        steps_per_epoch = math.ceil(weighted_images / effective_batch) if weighted_images else 0

    if max_train_steps is not None:
        total_steps = max_train_steps
        mode = "step_override"
        message = "max_train_stepsが指定されています。epoch基準レビューとは意味が変わる可能性があります。"
        warnings.append(message)
    else:
        total_steps = steps_per_epoch * max(0, epochs)
        mode = "epoch"
        message = ""

    target = {**DEFAULT_TARGETS, **(target or {})}
    min_steps = optional_int(target.get("target_steps_min"))
    recommended_steps = optional_int(target.get("target_steps_recommended"))
    max_steps = optional_int(target.get("target_steps_max"))
    checkpoint_count = max(1, int_or_default(target.get("target_checkpoint_count"), 6))

    status = "UNKNOWN"
    if effective_batch <= 0 or weighted_images <= 0:
        status = "ERROR"
    elif min_steps is not None and total_steps < min_steps:
        status = "LOW"
        message = "この設定は推奨stepより少なめです。学習が弱い可能性があります。"
    elif recommended_steps is not None and total_steps < recommended_steps:
        status = "LOW"
        message = "推奨stepには少し届いていません。弱めの学習になる可能性があります。"
    elif max_steps is not None and total_steps > max_steps * 1.25:
        status = "TOO_HIGH"
        message = "推奨上限を大きく超えています。過学習や時間増大に注意してください。"
    elif max_steps is not None and total_steps > max_steps:
        status = "HIGH"
        message = "推奨上限を超えています。固定化や学習時間に注意してください。"
    else:
        status = "OK"
        message = "Recipeの目標step範囲内です。"

    if repeats > 50:
        warnings.append("STRONG WARNING: repeatsが50を超えています。過学習や固定化に強く注意してください。")
    elif repeats > 30:
        warnings.append("repeatsが非常に大きいです。データ重複による固定化に注意してください。")
    save_every = int_or_default(params.get("save_every_n_epochs"), 1)
    sample_every = int_or_default(params.get("sample_every_n_epochs"), 1)
    proposed_interval = max(1, math.ceil(max(1, epochs) / checkpoint_count))
    estimated_checkpoints = math.ceil(max(1, epochs) / max(1, save_every)) if mode == "epoch" else None
    if mode == "epoch" and epochs > checkpoint_count and save_every <= 1:
        warnings.append("epoch数が多く、毎epoch保存です。出力LoRAが増えすぎるため保存間隔の調整を推奨します。")

    return {
        "mode": mode,
        "image_count": image_count,
        "weighted_image_count": weighted_images,
        "repeats": repeats,
        "max_train_epochs": epochs,
        "train_batch_size": batch,
        "gradient_accumulation_steps": grad_accum,
        "num_processes": process_count,
        "effective_batch_size": effective_batch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "target_steps_min": min_steps,
        "target_steps_recommended": recommended_steps,
        "target_steps_max": max_steps,
        "target_checkpoint_count": checkpoint_count,
        "target_source": target.get("step_target_source") or "global",
        "target_note": target.get("step_target_note") or "",
        "recipe_target_steps_min": optional_int(target.get("recipe_target_steps_min")),
        "recipe_target_steps_recommended": optional_int(target.get("recipe_target_steps_recommended")),
        "recipe_target_steps_max": optional_int(target.get("recipe_target_steps_max")),
        "optimizer_target_steps_min": optional_int(target.get("optimizer_target_steps_min")),
        "optimizer_target_steps_recommended": optional_int(target.get("optimizer_target_steps_recommended")),
        "optimizer_target_steps_max": optional_int(target.get("optimizer_target_steps_max")),
        "optimizer_name": target.get("optimizer_name") or "",
        "optimizer_lr_meaning": target.get("optimizer_lr_meaning") or "",
        "optimizer_category": target.get("optimizer_category") or "",
        "recipe_name": target.get("recipe_name") or "",
        "status": status,
        "message": message,
        "warnings": warnings,
        "save_every_n_epochs_proposal": proposed_interval,
        "sample_every_n_epochs_proposal": proposed_interval,
        "estimated_output_checkpoint_count": estimated_checkpoints,
    }


def calculate_required_repeats(
    *,
    image_count: int,
    params: dict[str, Any],
    target_steps: int,
    subsets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    epochs = max(1, int_or_default(params.get("max_train_epochs"), 1))
    batch = max(1, int_or_default(params.get("train_batch_size"), 1))
    grad = max(1, int_or_default(params.get("gradient_accumulation_steps"), 1))
    processes = max(1, int_or_default(params.get("num_processes"), 1))
    effective_batch = batch * grad * processes
    base_images = sum(max(0, int_or_default(subset.get("image_count"), 0)) for subset in subsets) if subsets else max(0, image_count)
    if base_images <= 0:
        return {
            "required_repeats": None,
            "error": "image_count=0のためrepeatsを自動計算できません。",
        }
    required = max(1, math.ceil(target_steps * effective_batch / (base_images * epochs)))

    def total_for(repeats: int) -> int:
        estimate = estimate_steps(
            image_count=image_count,
            params={**params, "repeats": repeats},
            target={"target_steps_recommended": target_steps},
            subsets=subsets,
        )
        return int(estimate["total_steps"] or 0)

    while required > 1 and total_for(required - 1) >= target_steps:
        required -= 1
    while total_for(required) < target_steps:
        required += 1

    estimate = estimate_steps(
        image_count=image_count,
        params={**params, "repeats": required},
        target={"target_steps_recommended": target_steps},
        subsets=subsets,
    )
    return {
        "required_repeats": required,
        "expected_total_steps": estimate["total_steps"],
        "steps_per_epoch": estimate["steps_per_epoch"],
        "effective_batch_size": effective_batch,
        "target_steps": target_steps,
        "error": "",
    }


def suggest_target_steps(
    *,
    image_count: int,
    params: dict[str, Any],
    target_steps: int,
    target: dict[str, Any] | None = None,
    strategy: str = "balanced",
    subsets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    base_repeats = int_or_default(params.get("repeats"), 1)
    base_epochs = int_or_default(params.get("max_train_epochs"), 1)
    batch = int_or_default(params.get("train_batch_size"), 1)
    grad = int_or_default(params.get("gradient_accumulation_steps"), 1)
    processes = int_or_default(params.get("num_processes"), 1)
    effective_batch = max(1, batch * grad * processes)
    checkpoint_count = max(1, int_or_default((target or {}).get("target_checkpoint_count"), 6))

    candidates: list[dict[str, Any]] = []

    def add(repeats: int, epochs: int, label: str) -> None:
        if repeats < 1 or epochs < 1:
            return
        estimate = estimate_steps(image_count=image_count, params={**params, "repeats": repeats, "max_train_epochs": epochs}, target=target, subsets=subsets)
        warning = ""
        if epochs > checkpoint_count and estimate["save_every_n_epochs_proposal"] > 1:
            warning = f"保存/サンプル間隔は {estimate['save_every_n_epochs_proposal']} epochごとを推奨します。"
        candidates.append(
            {
                "strategy": label,
                "repeats": repeats,
                "max_train_epochs": epochs,
                "expected_total_steps": estimate["total_steps"],
                "steps_per_epoch": estimate["steps_per_epoch"],
                "save_every_n_epochs_proposal": estimate["save_every_n_epochs_proposal"],
                "sample_every_n_epochs_proposal": estimate["sample_every_n_epochs_proposal"],
                "warning": warning,
                "delta": abs(int(estimate["total_steps"] or 0) - target_steps),
            }
        )

    weighted_base = subset_weighted_image_count(subsets, image_count, max(1, base_repeats))
    per_epoch_keep_repeats = math.ceil(weighted_base / effective_batch) if weighted_base else 0
    if strategy in {"keep_repeats_adjust_epochs", "balanced", "custom"} and per_epoch_keep_repeats:
        add(base_repeats, max(1, math.ceil(target_steps / per_epoch_keep_repeats)), "keep_repeats_adjust_epochs")

    if strategy in {"keep_epochs_adjust_repeats", "balanced", "custom"} and image_count > 0:
        repeats = max(1, math.ceil((target_steps * effective_batch) / (max(1, image_count) * max(1, base_epochs))))
        add(repeats, base_epochs, "keep_epochs_adjust_repeats")

    if strategy in {"balanced", "custom"}:
        for repeats in range(1, 41):
            weighted = subset_weighted_image_count(subsets, image_count, repeats)
            per_epoch = math.ceil(weighted / effective_batch) if weighted else 0
            if not per_epoch:
                continue
            epochs = max(1, math.ceil(target_steps / per_epoch))
            add(repeats, epochs, "balanced")

    unique: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate in candidates:
        unique.setdefault((candidate["repeats"], candidate["max_train_epochs"]), candidate)
    return sorted(unique.values(), key=lambda row: (row["delta"], row["max_train_epochs"], row["repeats"]))[:5]
