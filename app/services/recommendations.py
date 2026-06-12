from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.db import connect, create_job, fetch_all, fetch_one, utc_now


PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def regenerate_recommendations(job_id: int) -> list[dict[str, Any]]:
    context = build_context(job_id)
    now = utc_now()
    with connect() as conn:
        conn.execute(
            "UPDATE experiment_recommendations SET status = 'dismissed', updated_at = ? WHERE source_job_id = ? AND status != 'job_created'",
            (now, job_id),
        )
        created_rows = conn.execute(
            "SELECT recommendation_type, title FROM experiment_recommendations WHERE source_job_id = ? AND status = 'job_created'",
            (job_id,),
        ).fetchall()
    created_keys = {(row["recommendation_type"], row["title"]) for row in created_rows}
    recommendations = build_recommendations(context)
    with connect() as conn:
        for item in sorted(recommendations, key=lambda row: (PRIORITY_ORDER.get(row["priority"], 9), row["title"])):
            if (item["recommendation_type"], item["title"]) in created_keys:
                continue
            conn.execute(
                """
                INSERT INTO experiment_recommendations(
                    source_job_id, source_profile_id, recommendation_type, priority,
                    title, summary, reason, suggested_params_json, expected_effect,
                    risk_note, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?)
                """,
                (
                    job_id,
                    context["profile"]["id"] if context["profile"] else None,
                    item["recommendation_type"],
                    item["priority"],
                    item["title"],
                    item.get("summary") or "",
                    item.get("reason") or "",
                    json.dumps(item.get("suggested_params") or {}, ensure_ascii=False, indent=2),
                    item.get("expected_effect") or "",
                    item.get("risk_note") or "",
                    now,
                    now,
                ),
            )
    return list_recommendations(job_id)


def build_context(job_id: int) -> dict[str, Any]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE job_id = ? ORDER BY updated_at DESC, id DESC LIMIT 1", (job_id,))
    summary = fetch_one("SELECT * FROM training_metric_summaries WHERE job_id = ?", (job_id,))
    epochs = fetch_all("SELECT * FROM training_epoch_summaries WHERE job_id = ? ORDER BY epoch", (job_id,))
    samples = fetch_all("SELECT * FROM sample_images WHERE job_id = ?", (job_id,))
    weight_reviews = fetch_all(
        """
        SELECT w.*, p.validation_level
        FROM validation_weight_reviews w
        LEFT JOIN validation_presets p ON p.id = w.validation_preset_id
        WHERE w.job_id = ?
        ORDER BY
            CASE COALESCE(p.validation_level, '')
                WHEN 'standard' THEN 1
                WHEN 'quick' THEN 2
                WHEN 'extended' THEN 3
                ELSE 4
            END,
            w.hires_enabled,
            w.lora_weight,
            w.id
        """,
        (job_id,),
    )
    dataset_analysis = fetch_one("SELECT * FROM dataset_analysis WHERE dataset_id = ?", (job["dataset_id"],))
    params = json.loads(job["params_json"])
    return {
        "job": job,
        "profile": profile,
        "summary": summary,
        "epochs": epochs,
        "sample_ratings": sample_rating_summary(samples),
        "weight_reviews": weight_reviews,
        "dataset_analysis": dataset_analysis,
        "params": params,
    }


def build_recommendations(context: dict[str, Any]) -> list[dict[str, Any]]:
    params = dict(context["params"])
    job = context["job"]
    profile = context["profile"]
    summary = context["summary"]
    dataset_label = context["dataset_analysis"]["trigger_consistency_label"] if context["dataset_analysis"] else None
    selected_epoch = int(profile["selected_epoch"] or job["adopted_epoch"] or 0) if profile else int(job["adopted_epoch"] or 0)
    max_epochs = int(params.get("max_train_epochs") or 1)
    recommended_min = profile["recommended_weight_min"] if profile else None
    recommended_max = profile["recommended_weight_max"] if profile else None
    strong_weight = profile["strong_weight"] if profile else None
    validation_memo = profile["validation_memo"] if profile else ""
    weight_rubric = weight_review_rubric_summary(context["weight_reviews"])
    validation_scope = validation_condition_scope(context["weight_reviews"])
    epoch_label = summary["epoch_trend_label"] if summary else "UNKNOWN"
    health_label = summary["health_label"] if summary else "UNKNOWN"
    visual_good = selected_epoch_has_good_rating(context, selected_epoch)
    later_lower = later_epoch_rating_declines(context, selected_epoch)
    recs: list[dict[str, Any]] = []

    if dataset_label and dataset_label not in {"OK", "UNKNOWN"}:
        recs.append(
            recommendation(
                "dataset_fix",
                "high",
                "Dataset整備を優先",
                "trigger/captionの整合性を先に確認してください。",
                f"Dataset trigger consistency is {dataset_label}.",
                {},
                "学習パラメータを触る前に呼び出し安定性を改善できる可能性があります。",
                "Dataset修正後は既存Jobとの比較条件が変わります。",
            )
        )

    if selected_epoch and recommended_min is not None and recommended_max is not None and dataset_label in {"OK", None, "UNKNOWN"}:
        if health_label in {"OK", "WARNING"} and (visual_good or recommended_max <= 0.8 or weight_rubric["has_adopt"]):
            recs.append(
                recommendation(
                    "adopt_current",
                    "high",
                    "現在のLoRAは採用可能",
                    "追加学習なしで実用利用できる可能性が高いです。",
                    "\n".join(
                        [
                            f"採用epochは {selected_epoch} です。",
                            f"推奨weightは {recommended_min:g}〜{recommended_max:g} です。",
                            f"Dataset trigger consistencyは {dataset_label or 'UNKNOWN'} です。",
                            f"Epoch trendは {epoch_label} です。",
                            validation_scope["note"],
                        ]
                    ),
                    {},
                    "追加学習なしで実用利用可能。",
                    "1.0ではやや強い場合があるため、通常は推奨weight範囲を使ってください。",
                )
            )

    if selected_epoch and selected_epoch < max_epochs:
        reduced = dict(params)
        reduced["max_train_epochs"] = selected_epoch
        recs.append(
            recommendation(
                "reduce_epoch",
                "medium",
                f"Standard {selected_epoch} Epoch固定版を試す",
                "採用epochで止める短縮版です。",
                "\n".join(
                    [
                        f"採用epochが {selected_epoch} です。",
                        "後半epochの視覚評価が下がっている可能性があります。" if later_lower else "採用epoch以降の追加学習は必須ではありません。",
                    ]
                ),
                reduced,
                "不要な後半学習を避け、学習時間を短縮できる可能性があります。",
                f"既にepoch {selected_epoch} 出力があるため、再学習の優先度は高くありません。",
            )
        )

    if (strong_weight and strong_weight >= 1.0 and (weight_rubric["strong_warning"] or contains_strong_warning(validation_memo))) or health_label == "WARNING":
        lower_lr = dict(params)
        for key in ("learning_rate", "unet_lr"):
            if key in lower_lr and lower_lr[key]:
                lower_lr[key] = float(lower_lr[key]) * 0.5
        lower_lr["max_train_epochs"] = max_epochs
        lower_lr["network_dim"] = params.get("network_dim", 32)
        lower_lr["network_alpha"] = params.get("network_alpha", 16)
        recs.append(
            recommendation(
                "lower_lr",
                "medium",
                f"Lower LR {max_epochs} Epochを試す",
                "少し弱めに学習する比較案です。",
                "weight 1.0がやや強い、またはstep lossに揺れがあります。",
                lower_lr,
                "画風の押し出しや固定化を少し抑えられる可能性があります。",
                "顔特徴が弱くなる可能性があります。",
            )
        )

    higher_dim = dict(params)
    higher_dim["network_dim"] = 64
    higher_dim["network_alpha"] = 32
    recs.append(
        recommendation(
            "higher_dim",
            "low",
            "Dim64強化版は現時点では低優先",
            "強化版の比較案ですが、現状では優先度を下げます。",
            "現在のLoRAは0.6〜0.8で十分効いており、1.0では強すぎる可能性があります。",
            higher_dim,
            "顔再現は強くなる可能性があります。",
            "固定化・過学習・画風過多のリスクが高いです。",
        )
    )

    te_trial = dict(params)
    te_trial["network_train_unet_only"] = False
    te_trial["cache_text_encoder_outputs"] = False
    te_trial["text_encoder_lr1"] = te_trial.get("text_encoder_lr1") or 0.000005
    te_trial["text_encoder_lr2"] = te_trial.get("text_encoder_lr2") or 0.000005
    recs.append(
        recommendation(
            "text_encoder_trial",
            "low",
            "Text Encoder学習は現時点では保留",
            "trigger反応が不足した時の比較案です。",
            "triggerは機能しており、現在のLoRAはweightに反応しています。",
            te_trial,
            "triggerと顔特徴の結びつきが強くなる可能性があります。",
            "TE学習は過学習や汎用性低下のリスクがあります。",
        )
    )

    if recommended_min is not None and recommended_min >= 0.8 and not weight_rubric["strong_warning"] and not contains_strong_warning(validation_memo):
        stronger = dict(params)
        stronger["network_dim"] = max(64, int(stronger.get("network_dim") or 32))
        stronger["network_alpha"] = max(32, int(stronger.get("network_alpha") or 16))
        recs.append(
            recommendation("strengthen", "medium", "特徴を強める設定を試す", "LoRAの効きが弱い場合の案です。", "推奨weightが高めです。", stronger, "低いweightでも特徴が出やすくなる可能性があります。", "固定化が強まる可能性があります。")
        )
    if (recommended_min is not None and recommended_min <= 0.4) or weight_rubric["too_strong_at_low_weight"]:
        weaker = dict(params)
        for key in ("learning_rate", "unet_lr"):
            if key in weaker and weaker[key]:
                weaker[key] = float(weaker[key]) * 0.5
        weaker["network_dim"] = min(int(weaker.get("network_dim") or 32), 16)
        weaker["network_alpha"] = min(int(weaker.get("network_alpha") or 16), 8)
        recs.append(
            recommendation("generalize", "medium", "弱め・汎化寄り設定を試す", "効きが強すぎる場合の案です。", "低weightでも強く出る傾向があります。", weaker, "固定化を抑えられる可能性があります。", "顔特徴が弱くなる可能性があります。")
        )

    return dedupe_recommendations(recs)


def recommendation(
    recommendation_type: str,
    priority: str,
    title: str,
    summary: str,
    reason: str,
    suggested_params: dict[str, Any],
    expected_effect: str,
    risk_note: str,
) -> dict[str, Any]:
    return {
        "recommendation_type": recommendation_type,
        "priority": priority,
        "title": title,
        "summary": summary,
        "reason": reason,
        "suggested_params": suggested_params,
        "expected_effect": expected_effect,
        "risk_note": risk_note,
    }


def dedupe_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for row in rows:
        key = (row["recommendation_type"], row["title"])
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def sample_rating_summary(samples: list[Any]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[Any]] = {}
    for sample in samples:
        if sample["epoch"] is not None:
            grouped.setdefault(int(sample["epoch"]), []).append(sample)
    result: dict[int, dict[str, Any]] = {}
    for epoch, rows in grouped.items():
        values = []
        adoptions = []
        for row in rows:
            value = row["rating_overall"] if "rating_overall" in row.keys() else row["rating"] if "rating" in row.keys() else None
            if value is not None and int(value) > 0:
                values.append(int(value))
            if "adoption_label" in row.keys() and row["adoption_label"]:
                adoptions.append(row["adoption_label"])
        result[epoch] = {"avg_overall": sum(values) / len(values) if values else None, "count": len(values), "adoptions": adoptions}
    return result


def selected_epoch_has_good_rating(context: dict[str, Any], selected_epoch: int) -> bool:
    if not selected_epoch:
        return False
    row = context["sample_ratings"].get(selected_epoch)
    return bool(row and (("adopt" in row.get("adoptions", [])) or (row["avg_overall"] is not None and row["avg_overall"] >= 3)))


def later_epoch_rating_declines(context: dict[str, Any], selected_epoch: int) -> bool:
    selected = context["sample_ratings"].get(selected_epoch)
    if not selected or selected["avg_overall"] is None:
        return False
    later = [row["avg_overall"] for epoch, row in context["sample_ratings"].items() if epoch > selected_epoch and row["avg_overall"] is not None]
    return bool(later and max(later) < selected["avg_overall"])


def contains_strong_warning(text: str | None) -> bool:
    text = text or ""
    return "やや強い" in text or "強すぎ" in text or "strong" in text.lower()


def weight_review_rubric_summary(rows: list[Any]) -> dict[str, bool]:
    rows = preferred_weight_reviews(rows)
    strong_labels = {"too_strong", "broken", "strong_but_usable"}
    weak_labels = {"too_weak", "weak_but_usable"}
    strong_tags = {"LoRA効果強すぎ", "画風過多", "背景汚染", "構図固定", "衣装固定", "表情固定"}
    weak_tags = {"LoRA効果弱い", "顔が弱い", "衣装が弱い", "trigger反応弱い"}
    result = {
        "strong_warning": False,
        "too_strong_at_low_weight": False,
        "weak_warning": False,
        "has_adopt": False,
        "severe_overfit": False,
    }
    for row in rows:
        weight = float(row["lora_weight"] or 0)
        strength = row["strength_label"] if "strength_label" in row.keys() else None
        overfit = row["overfit_level"] if "overfit_level" in row.keys() else None
        adoption = row["adoption_label"] if "adoption_label" in row.keys() else None
        tags = parse_tags(row["failure_tags_json"] if "failure_tags_json" in row.keys() else "")
        if adoption == "adopt":
            result["has_adopt"] = True
        if overfit == "severe":
            result["severe_overfit"] = True
        if strength in strong_labels or any(tag in strong_tags for tag in tags):
            result["strong_warning"] = True
            if weight <= 0.6:
                result["too_strong_at_low_weight"] = True
        if strength in weak_labels or any(tag in weak_tags for tag in tags):
            result["weak_warning"] = True
    return result


def preferred_weight_reviews(rows: list[Any]) -> list[Any]:
    non_hires = [row for row in rows if "hires_enabled" not in row.keys() or not row["hires_enabled"]]
    standard_or_quick = [
        row for row in non_hires
        if "validation_level" not in row.keys() or row["validation_level"] in {"standard", "quick", None, ""}
    ]
    return standard_or_quick or non_hires or rows


def validation_condition_scope(rows: list[Any]) -> dict[str, Any]:
    if not rows:
        return {"mixed": False, "note": "Validation reviews are not available."}
    levels = {row["validation_level"] for row in rows if "validation_level" in row.keys() and row["validation_level"]}
    hires_values = {int(row["hires_enabled"] or 0) for row in rows if "hires_enabled" in row.keys()}
    mixed = len(levels) > 1 or len(hires_values) > 1
    note = "Validation basis: non-Hires Standard/Quick reviews are preferred."
    if mixed:
        note += " Validation条件が混在しているため注意してください。"
    return {"mixed": mixed, "note": note}


def parse_tags(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def list_recommendations(job_id: int) -> list[dict[str, Any]]:
    rows = fetch_all(
        "SELECT * FROM experiment_recommendations WHERE source_job_id = ? AND status != 'dismissed' ORDER BY CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, id",
        (job_id,),
    )
    return [dict(row) for row in rows]


def set_recommendation_status(recommendation_id: int, status: str) -> int:
    now = utc_now()
    with connect() as conn:
        row = conn.execute("SELECT * FROM experiment_recommendations WHERE id = ?", (recommendation_id,)).fetchone()
        if row is None:
            raise ValueError(f"Recommendation not found: {recommendation_id}")
        conn.execute("UPDATE experiment_recommendations SET status = ?, updated_at = ? WHERE id = ?", (status, now, recommendation_id))
        return int(row["source_job_id"])


def create_draft_job_from_recommendation(recommendation_id: int) -> int:
    recommendation = fetch_one("SELECT * FROM experiment_recommendations WHERE id = ?", (recommendation_id,))
    if recommendation is None:
        raise ValueError(f"Recommendation not found: {recommendation_id}")
    source = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (recommendation["source_job_id"],))
    if source is None:
        raise ValueError(f"Source job not found: {recommendation['source_job_id']}")
    params = json.loads(recommendation["suggested_params_json"] or "{}")
    if not params:
        params = json.loads(source["params_json"])
    job_id = create_job(
        {
            "name": f"{source['name']} - {recommendation['title']}",
            "dataset_id": source["dataset_id"],
            "preset_id": source["preset_id"],
            "base_model_path": source["base_model_path"],
            "vae_path": source["vae_path"] or "",
            "output_name": f"{source['output_name']}_rec_{recommendation_id}",
            "memo": f"Created from recommendation #{recommendation_id}: {recommendation['title']}",
            "sample_prompt_template_id": source["sample_prompt_template_id"] or "",
            "parent_job_id": source["id"],
            "params": params,
        }
    )
    now = utc_now()
    with connect() as conn:
        conn.execute(
            "UPDATE training_jobs SET dataset_version_id = ?, updated_at = ? WHERE id = ?",
            (source["dataset_version_id"], now, job_id),
        )
        conn.execute(
            "UPDATE experiment_recommendations SET created_job_id = ?, status = 'job_created', updated_at = ? WHERE id = ?",
            (job_id, now, recommendation_id),
        )
    return job_id


def write_recommendation_report(job_id: int) -> str:
    context = build_context(job_id)
    recommendations = list_recommendations(job_id)
    job = context["job"]
    profile = context["profile"]
    summary = context["summary"]
    report_dir = Path(job["run_dir"]) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"recommendations_job_{job_id:06d}.md"
    lines = [
        f"# Recommendation Report Job #{job_id}",
        "",
        "## Source Job",
        f"- name: {job['name']}",
        f"- status: {job['status']}",
        f"- selected LoRA: {job['adopted_model_path'] or '-'}",
        "",
        "## Profile",
    ]
    if profile:
        lines.extend(
            [
                f"- profile: {profile['profile_name']}",
                f"- trigger: {profile['trigger_word'] or '-'}",
                f"- selected epoch: {profile['selected_epoch'] or '-'}",
                f"- recommended weight: {profile['recommended_weight_min'] if profile['recommended_weight_min'] is not None else '-'} - {profile['recommended_weight_max'] if profile['recommended_weight_max'] is not None else '-'}",
                f"- light / strong: {profile['light_weight'] if profile['light_weight'] is not None else '-'} / {profile['strong_weight'] if profile['strong_weight'] is not None else '-'}",
                f"- validation memo: {profile['validation_memo'] or '-'}",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Loss Summary"])
    if summary:
        lines.extend(
            [
                f"- health: {summary['health_label']}",
                f"- raw: {summary['raw_loss_label']}",
                f"- smoothed: {summary['smoothed_loss_label']}",
                f"- epoch: {summary['epoch_trend_label']}",
                f"- message: {summary['health_message']}",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Recommendations"])
    for item in recommendations:
        params = json.loads(item["suggested_params_json"] or "{}")
        diff = param_diff(context["params"], params)
        lines.extend(
            [
                f"### #{item['id']} {item['title']}",
                f"- type: {item['recommendation_type']}",
                f"- priority: {item['priority']}",
                f"- status: {item['status']}",
                f"- summary: {item['summary'] or '-'}",
                f"- reason: {item['reason'] or '-'}",
                f"- expected: {item['expected_effect'] or '-'}",
                f"- risk: {item['risk_note'] or '-'}",
                "- suggested param diff:",
            ]
        )
        if diff:
            lines.extend(f"  - {key}: {old} -> {new}" for key, old, new in diff)
        else:
            lines.append("  - no param change")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "- Recommendations are rule-based suggestions only.",
            "- Draft Jobs are not run automatically.",
            "- ChatGPT API and AI image evaluation are future extensions.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def param_diff(base: dict[str, Any], suggested: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    rows = []
    for key in sorted(set(base) | set(suggested)):
        old = base.get(key)
        new = suggested.get(key)
        if old != new:
            rows.append((key, old, new))
    return rows
