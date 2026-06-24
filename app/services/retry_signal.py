from __future__ import annotations

import json
from typing import Any

from app.db import fetch_all, fetch_one
from app.services.review_sessions import review_machine_candidate_summary
from app.services.step_estimator import optional_int, target_config_from_catalog


ACCEPTABLE = "ACCEPTABLE"
UNDERTRAINED_STEP_SHORTAGE = "UNDERTRAINED_STEP_SHORTAGE"
UNDERTRAINED_STILL_IMPROVING = "UNDERTRAINED_STILL_IMPROVING"
OVERTRAINED = "OVERTRAINED"
PARAMETER_TOO_WEAK = "PARAMETER_TOO_WEAK"
PARAMETER_TOO_STRONG = "PARAMETER_TOO_STRONG"
DATASET_OR_CAPTION_ISSUE = "DATASET_OR_CAPTION_ISSUE"
NO_CLEAR_WINNER = "NO_CLEAR_WINNER"


def retry_signal_for_job(job_id: int) -> dict[str, Any]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        return empty_signal("Jobが見つかりません。")
    project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (job["project_id"],)) if job["project_id"] else None
    profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE job_id = ? ORDER BY updated_at DESC, id DESC LIMIT 1", (job_id,))
    latest_review = fetch_one("SELECT * FROM review_sessions WHERE job_id = ? ORDER BY id DESC LIMIT 1", (job_id,))
    latest_validation = fetch_one("SELECT * FROM validation_runs WHERE job_id = ? ORDER BY id DESC LIMIT 1", (job_id,))
    return build_retry_signal(job=job, project=project, profile=profile, review_session=latest_review, validation_run=latest_validation)


def retry_signal_for_project(project_id: int) -> dict[str, Any]:
    project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (project_id,))
    if project is None:
        return empty_signal("Projectが見つかりません。")
    job = None
    if project["selected_job_id"]:
        job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (project["selected_job_id"],))
    if job is None:
        job = fetch_one("SELECT * FROM training_jobs WHERE project_id = ? ORDER BY id DESC LIMIT 1", (project_id,))
    if job is None:
        return empty_signal("Projectに学習Jobがありません。")
    profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE project_id = ? ORDER BY updated_at DESC, id DESC LIMIT 1", (project_id,))
    review_session = fetch_one("SELECT * FROM review_sessions WHERE project_id = ? ORDER BY id DESC LIMIT 1", (project_id,))
    validation_run = fetch_one("SELECT * FROM validation_runs WHERE project_id = ? ORDER BY id DESC LIMIT 1", (project_id,))
    return build_retry_signal(job=job, project=project, profile=profile, review_session=review_session, validation_run=validation_run)


def retry_signal_for_review_session(session_id: int) -> dict[str, Any]:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
    if session is None:
        return empty_signal("Review Sessionが見つかりません。")
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (session["job_id"],)) if session["job_id"] else None
    project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (session["project_id"],)) if session["project_id"] else None
    profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE job_id = ? ORDER BY updated_at DESC, id DESC LIMIT 1", (job["id"],)) if job else None
    validation_run = fetch_one("SELECT * FROM validation_runs WHERE job_id = ? ORDER BY id DESC LIMIT 1", (job["id"],)) if job else None
    return build_retry_signal(job=job, project=project, profile=profile, review_session=session, validation_run=validation_run)


def retry_signal_for_profile(profile_id: int) -> dict[str, Any]:
    profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (profile_id,))
    if profile is None:
        return empty_signal("LoRA Profileが見つかりません。")
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (profile["job_id"],))
    project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (profile["project_id"],)) if profile["project_id"] else None
    review_session = fetch_one("SELECT * FROM review_sessions WHERE job_id = ? ORDER BY id DESC LIMIT 1", (profile["job_id"],))
    validation_run = fetch_one("SELECT * FROM validation_runs WHERE selected_lora_profile_id = ? OR job_id = ? ORDER BY id DESC LIMIT 1", (profile_id, profile["job_id"]))
    return build_retry_signal(job=job, project=project, profile=profile, review_session=review_session, validation_run=validation_run)


def build_retry_signal(
    *,
    job: Any | None,
    project: Any | None = None,
    profile: Any | None = None,
    review_session: Any | None = None,
    validation_run: Any | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    actions: list[str] = []
    confidence = "low"
    label = ACCEPTABLE
    evidence: dict[str, Any] = {}

    if job is None:
        return empty_signal("学習Jobの情報が見つかりません。")

    step_context = retry_step_context(job)
    step_status = step_context["step_status"]
    expected_steps = step_context["expected_steps"]
    target_steps = step_context["target_steps_recommended"]
    target_steps_min = step_context["target_steps_min"]
    evidence["expected_steps"] = expected_steps
    evidence["target_steps_recommended"] = target_steps
    evidence["target_steps_min"] = target_steps_min
    evidence["step_status"] = step_status
    if step_status == "TOO_LOW" or (target_steps_min is not None and expected_steps is not None and expected_steps < target_steps_min):
        label = UNDERTRAINED_STEP_SHORTAGE
        confidence = "medium"
        reasons.append(f"想定Stepが目標Step範囲に対して {step_status} です。")
        actions.append("まずrepeats / epochs / batchを見直し、target stepsに近づける案を検討してください。")

    dataset_issue = dataset_or_caption_issue(job, project)
    if dataset_issue:
        label = DATASET_OR_CAPTION_ISSUE
        confidence = "high"
        reasons.append(dataset_issue)
        actions.append("Retry条件を触る前にDataset再スキャン、trigger、caption整合性を確認してください。")

    loss_signal = loss_trend_signal(int(job["id"]))
    evidence["loss_signal"] = loss_signal
    if label == ACCEPTABLE and loss_signal["label"]:
        label = loss_signal["label"]
        confidence = "medium"
        reasons.extend(loss_signal["reasons"])
        actions.extend(loss_signal["actions"])

    candidate_signal = candidate_position_signal(job)
    evidence["candidate_position"] = candidate_signal
    if label == ACCEPTABLE and candidate_signal["label"]:
        label = candidate_signal["label"]
        confidence = "medium"
        reasons.extend(candidate_signal["reasons"])
        actions.extend(candidate_signal["actions"])

    review_signal = review_session_signal(review_session)
    evidence["review_signal"] = review_signal
    if review_signal["label"] == NO_CLEAR_WINNER and label == ACCEPTABLE:
        label = NO_CLEAR_WINNER
        confidence = "medium"
        reasons.extend(review_signal["reasons"])
        actions.extend(review_signal["actions"])

    weight_signal = weight_calibration_signal(int(job["id"]), profile, validation_run)
    evidence["weight_signal"] = weight_signal
    if weight_signal["label"] and label != DATASET_OR_CAPTION_ISSUE:
        label = weight_signal["label"]
        confidence = "medium" if confidence == "low" else confidence
        reasons.extend(weight_signal["reasons"])
        actions.extend(weight_signal["actions"])

    overfit_signal = overfit_risk_signal(int(job["id"]), validation_run)
    evidence["overfit_signal"] = overfit_signal
    if overfit_signal and label == ACCEPTABLE:
        label = OVERTRAINED
        confidence = "medium"
        reasons.append(overfit_signal)
        actions.append("高weightや後半epochを避け、採用epoch短縮または低LR案を検討してください。")

    failure = failure_tag_signal(int(job["id"]))
    if failure:
        label = DATASET_OR_CAPTION_ISSUE if label == ACCEPTABLE else label
        confidence = "medium"
        reasons.append(failure)
        actions.append("失敗タグの傾向を確認し、Dataset / prompt / caption修正を優先してください。")

    if label == ACCEPTABLE and profile and validation_run and validation_run["status"] in {"completed", "ready_for_review", "images_registered"}:
        min_weight = profile["recommended_weight_min"] if "recommended_weight_min" in profile.keys() else validation_run["recommended_weight_min"]
        max_weight = profile["recommended_weight_max"] if "recommended_weight_max" in profile.keys() else validation_run["recommended_weight_max"]
        if min_weight is not None and max_weight is not None:
            confidence = "high"
            reasons.append("採用LoRAとWeight検証結果があり、強いリトライシグナルはありません。")
            actions.append("現在のLoRAは採用可能です。推奨weightを確認し、必要ならexport / cleanupへ進んでください。")

    if not reasons:
        reasons.append("Step、loss、レビュー、Weight検証に強いリトライシグナルはありません。")
        actions.append("現状採用または追加の人間評価を優先し、自動Retryは行いません。")
        confidence = "medium"

    return {
        "retry_signal_label": label,
        "confidence": confidence,
        "reasons": unique(reasons),
        "recommended_next_actions": unique(actions),
        "evidence": evidence,
        "job_id": int(job["id"]),
        "project_id": int(project["id"]) if project and project["id"] else job["project_id"],
        "review_session_id": int(review_session["id"]) if review_session and review_session["id"] else None,
        "profile_id": int(profile["id"]) if profile and profile["id"] else None,
        "validation_run_id": int(validation_run["id"]) if validation_run and validation_run["id"] else None,
    }


def empty_signal(reason: str) -> dict[str, Any]:
    return {
        "retry_signal_label": "UNKNOWN",
        "confidence": "low",
        "reasons": [reason],
        "recommended_next_actions": ["必要な結果データが揃ってから再確認してください。"],
        "evidence": {},
    }


def retry_step_context(job: Any) -> dict[str, Any]:
    expected_steps = optional_int(job["expected_total_steps_at_creation"] if "expected_total_steps_at_creation" in job.keys() else None)
    if expected_steps is None and "expected_total_steps" in job.keys():
        expected_steps = optional_int(job["expected_total_steps"])
    target = target_context_for_job(job)
    target_min = optional_int(target.get("target_steps_min"))
    target_recommended = optional_int(target.get("target_steps_recommended"))
    target_max = optional_int(target.get("target_steps_max"))
    stored_status = str(job["step_status_at_creation"] or "")
    step_status = stored_status or "UNKNOWN"
    if expected_steps is not None:
        if target_min is not None and expected_steps < target_min:
            step_status = "LOW"
        elif target_max is not None and expected_steps > target_max * 1.25:
            step_status = "TOO_HIGH"
        elif target_max is not None and expected_steps > target_max:
            step_status = "HIGH"
        elif target_min is not None and target_recommended is not None and expected_steps >= target_min:
            step_status = "OK"
    return {
        "expected_steps": expected_steps,
        "target_steps_min": target_min,
        "target_steps_recommended": target_recommended,
        "target_steps_max": target_max,
        "step_status": step_status,
    }


def target_context_for_job(job: Any) -> dict[str, Any]:
    preset = fetch_one("SELECT * FROM presets WHERE id = ?", (job["preset_id"],)) if job["preset_id"] else None
    recipe = None
    profile = None
    definition = None
    recipe_v2_id = job["recipe_v2_id"] if "recipe_v2_id" in job.keys() else None
    if recipe_v2_id:
        recipe = fetch_one("SELECT * FROM training_recipes_v2 WHERE id = ?", (recipe_v2_id,))
        if recipe is not None:
            if recipe["optimizer_profile_id"]:
                profile = fetch_one("SELECT * FROM optimizer_profiles_v2 WHERE id = ?", (recipe["optimizer_profile_id"],))
            if recipe["optimizer_definition_id"]:
                definition = fetch_one("SELECT * FROM optimizer_definitions_v2 WHERE id = ?", (recipe["optimizer_definition_id"],))
            return target_config_from_catalog(
                training_recipe=recipe,
                optimizer_profile=profile,
                optimizer_definition=definition,
                preset=preset,
            )
    params = json_loads(job["params_json"], {})
    optimizer_type = str(params.get("optimizer_type") or "")
    if optimizer_type:
        definition = fetch_one("SELECT * FROM optimizer_definitions_v2 WHERE id = ?", (optimizer_type,))
    return target_config_from_catalog(
        training_recipe=None,
        optimizer_profile=None,
        optimizer_definition=definition,
        preset=preset,
    )


def dataset_or_caption_issue(job: Any, project: Any | None) -> str:
    labels = [job["trigger_consistency_label_at_creation"] if "trigger_consistency_label_at_creation" in job.keys() else None]
    if project and "trigger_word" in project.keys() and not str(project["trigger_word"] or "").strip():
        return "Projectのtrigger wordが未設定です。"
    analysis = fetch_one("SELECT trigger_consistency_label FROM dataset_analysis WHERE dataset_id = ?", (job["dataset_id"],))
    if analysis:
        labels.append(analysis["trigger_consistency_label"])
    bad = [label for label in labels if label and label not in {"OK", "UNKNOWN"}]
    return f"Dataset / caption整合性が {bad[0]} です。" if bad else ""


def loss_trend_signal(job_id: int) -> dict[str, Any]:
    summary = fetch_one("SELECT * FROM training_metric_summaries WHERE job_id = ?", (job_id,))
    if summary:
        trend = str(summary["epoch_trend_label"] or "")
        health = str(summary["health_label"] or "")
        if trend in {"STILL_IMPROVING", "IMPROVING"}:
            return signal(UNDERTRAINED_STILL_IMPROVING, [f"Loss傾向は {trend} です。"], ["少し長めのepochまたはtarget steps増加を検討してください。"])
        if trend in {"OVERTRAINING", "WORSENING"} or health == "DANGER":
            return signal(OVERTRAINED, [f"Loss健全性または傾向が {health or trend} を示しています。"], ["採用epoch短縮、低LR、または早めepochの比較を優先してください。"])
    rows = fetch_all("SELECT epoch, moving_avg_final_loss, avg_loss FROM training_epoch_summaries WHERE job_id = ? AND epoch IS NOT NULL ORDER BY epoch", (job_id,))
    values = [(int(row["epoch"]), row["moving_avg_final_loss"] if row["moving_avg_final_loss"] is not None else row["avg_loss"]) for row in rows]
    values = [(epoch, float(value)) for epoch, value in values if value is not None]
    if len(values) >= 3:
        last = values[-1][1]
        prev = values[-2][1]
        first = values[0][1]
        if last < prev < first:
            return signal(UNDERTRAINED_STILL_IMPROVING, ["最終epochでもLossがまだ下がっています。"], ["終盤epochを候補に含め、必要なら少し長めのRetryを検討してください。"])
        if last > prev:
            return signal(OVERTRAINED, ["最終epochでLossが悪化しています。"], ["後半epochの過学習を疑い、早めepochまたは低LR案を確認してください。"])
    return signal(None, [], [])


def candidate_position_signal(job: Any) -> dict[str, Any]:
    params = json_loads(job["params_json"], {})
    max_epoch = int(params.get("max_train_epochs") or 0)
    selected = int(job["adopted_epoch"] or 0)
    candidates = fetch_all("SELECT epoch, candidate_label FROM training_epoch_candidate_summaries WHERE job_id = ? ORDER BY candidate_rank", (job["id"],))
    primary = next((int(row["epoch"]) for row in candidates if row["candidate_label"] == "primary" and row["epoch"] is not None), 0)
    best = selected or primary
    if max_epoch and best == max_epoch:
        return signal(UNDERTRAINED_STILL_IMPROVING, [f"最良候補epochが最終epoch {best} です。"], ["最終epochが最良なら、少し長めのRetryや近隣epoch追加確認を検討してください。"])
    if max_epoch and best and best <= max(1, max_epoch // 3):
        return signal(OVERTRAINED, [f"最良候補epoch {best} が最大epoch {max_epoch} に対して早めです。"], ["短縮epoch、低LR、または過学習傾向の確認を優先してください。"])
    return signal(None, [], [])


def review_session_signal(review_session: Any | None) -> dict[str, Any]:
    if not review_session:
        return signal(None, [], [])
    status = str(review_session["status"] or "")
    if status in {"planned", "prepared"}:
        return signal(None, [], [], {"status": status, "review_ready": False})
    human = fetch_all(
        """
        SELECT epoch, AVG(rating_overall) AS avg_rating, COUNT(*) AS count
        FROM review_session_images
        WHERE review_session_id = ? AND rating_overall IS NOT NULL
        GROUP BY epoch
        ORDER BY avg_rating DESC, count DESC, epoch
        """,
        (review_session["id"],),
    )
    if human:
        top = human[0]
        return signal(
            None,
            [],
            ["人間評価が入力済みです。機械補助レビューより人間評価を優先して採用epochを確認してください。"],
            {"human_review_count": sum(int(row["count"] or 0) for row in human), "human_top_epoch": top["epoch"], "human_top_rating": top["avg_rating"]},
        )
    summary = review_machine_candidate_summary(int(review_session["id"]))
    if summary.get("confidence") == "no_clear_winner":
        return signal(NO_CLEAR_WINNER, ["Review Sessionの機械補助レビューでは明確な勝者が出ていません。"], ["Review Matrixで候補群を人間評価し、採用epochを決めてください。"], {"candidate_group": summary.get("candidate_group")})
    return signal(None, [], [], {"primary_candidate": summary.get("primary_candidate")})


def weight_calibration_signal(job_id: int, profile: Any | None, validation_run: Any | None) -> dict[str, Any]:
    images = fetch_all(
        """
        SELECT strength_label, adoption_label, rating_overall, lora_weight, overfit_level
        FROM validation_images
        WHERE job_id = ? AND COALESCE(ignored, 0) = 0 AND image_role = 'individual'
        """,
        (job_id,),
    )
    strengths = [str(row["strength_label"] or "") for row in images]
    if strengths and strengths.count("too_weak") >= max(1, len(strengths) // 2):
        return signal(PARAMETER_TOO_WEAK, ["Weight検証で too_weak 判定が多く出ています。"], ["LR、dim、epoch、または推奨weight上限を見直してください。"])
    if strengths and sum(1 for item in strengths if item in {"too_strong", "broken"}) >= max(1, len(strengths) // 2):
        return signal(PARAMETER_TOO_STRONG, ["Weight検証で too_strong / broken 判定が多く出ています。"], ["低LR、短縮epoch、または推奨weight下げを検討してください。"])
    min_weight = profile["recommended_weight_min"] if profile and "recommended_weight_min" in profile.keys() else validation_run["recommended_weight_min"] if validation_run else None
    max_weight = profile["recommended_weight_max"] if profile and "recommended_weight_max" in profile.keys() else validation_run["recommended_weight_max"] if validation_run else None
    if min_weight is not None and max_weight is not None:
        try:
            if float(max_weight) <= 0.4:
                return signal(PARAMETER_TOO_STRONG, [f"推奨weight範囲が低めです（{float(min_weight):g}-{float(max_weight):g}）。"], ["LoRAが強すぎる可能性があるため、低LR/短縮epochを検討してください。"])
            if float(min_weight) >= 0.9:
                return signal(PARAMETER_TOO_WEAK, [f"推奨weight範囲が高めです（{float(min_weight):g}-{float(max_weight):g}）。"], ["LoRAが弱い可能性があるため、学習量または表現力を見直してください。"])
        except (TypeError, ValueError):
            pass
    return signal(None, [], [])


def overfit_risk_signal(job_id: int, validation_run: Any | None) -> str:
    score = fetch_one(
        """
        SELECT overfit_risk_label, COUNT(*) AS count
        FROM machine_review_scores
        WHERE job_id = ? AND overfit_risk_label IN ('high', 'possible_overfit')
        GROUP BY overfit_risk_label
        ORDER BY count DESC LIMIT 1
        """,
        (job_id,),
    )
    if score:
        return f"機械補助レビューの過学習リスクに {score['overfit_risk_label']} が含まれます（{score['count']}件）。"
    images = fetch_one("SELECT COUNT(*) AS count FROM validation_images WHERE job_id = ? AND overfit_level IN ('moderate','severe')", (job_id,))
    if images and int(images["count"] or 0) > 0:
        return f"Human validation marked overfit on {images['count']} image(s)."
    return ""


def failure_tag_signal(job_id: int) -> str:
    rows = fetch_all("SELECT failure_tags_json FROM sample_images WHERE job_id = ?", (job_id,))
    tags: list[str] = []
    for row in rows:
        tags.extend(json_loads(row["failure_tags_json"], []))
    if tags:
        top = sorted(set(str(tag) for tag in tags))[:5]
        return "失敗タグがあります: " + ", ".join(top)
    return ""


def signal(label: str | None, reasons: list[str], actions: list[str], evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"label": label, "reasons": reasons, "actions": actions, "evidence": evidence or {}}


def json_loads(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def unique(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
