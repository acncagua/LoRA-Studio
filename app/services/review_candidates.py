from __future__ import annotations

import json
from typing import Any

from app.db import connect, fetch_all, fetch_one, utc_now


CANDIDATE_LABELS = {"primary", "secondary", "check", "low_priority", "avoid"}


def ensure_epoch_candidates(job_id: int) -> list[dict[str, Any]]:
    rows = fetch_all("SELECT * FROM training_epoch_candidate_summaries WHERE job_id = ? ORDER BY candidate_rank, epoch", (job_id,))
    if rows:
        return [candidate_dict(row) for row in rows]
    return regenerate_epoch_candidates(job_id)


def regenerate_epoch_candidates(job_id: int) -> list[dict[str, Any]]:
    epochs = fetch_all("SELECT * FROM training_epoch_summaries WHERE job_id = ? ORDER BY epoch", (job_id,))
    if not epochs:
        return []
    outputs = fetch_all("SELECT epoch, selected FROM training_outputs WHERE job_id = ? AND file_type = 'model' AND deleted_at IS NULL", (job_id,))
    samples = fetch_all("SELECT epoch FROM sample_images WHERE job_id = ? AND deleted_at IS NULL", (job_id,))
    job = fetch_one("SELECT adopted_epoch FROM training_jobs WHERE id = ?", (job_id,))
    output_epochs = {int(row["epoch"]) for row in outputs if row["epoch"] is not None}
    sample_epochs = {int(row["epoch"]) for row in samples if row["epoch"] is not None}
    selected_epochs = {int(row["epoch"]) for row in outputs if row["epoch"] is not None and row["selected"]}
    if job and job["adopted_epoch"] is not None:
        selected_epochs.add(int(job["adopted_epoch"]))

    scored = score_epochs(epochs, output_epochs, sample_epochs, selected_epochs)
    selected_ranked = pick_labels(scored)
    now = utc_now()
    with connect() as conn:
        conn.execute("DELETE FROM training_epoch_candidate_summaries WHERE job_id = ?", (job_id,))
        for rank, row in enumerate(selected_ranked, start=1):
            conn.execute(
                """
                INSERT INTO training_epoch_candidate_summaries(
                    job_id, epoch, candidate_rank, candidate_label, score, reason_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, row["epoch"], rank, row["candidate_label"], row["score"], json.dumps(row["reasons"], ensure_ascii=False), now, now),
            )
    update_sample_review_priority(job_id)
    return ensure_epoch_candidates(job_id)


def score_epochs(epochs: list[Any], output_epochs: set[int], sample_epochs: set[int], selected_epochs: set[int]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in epochs if row["epoch"] is not None]
    moving_values = [float(row["moving_avg_final_loss"]) for row in rows if row.get("moving_avg_final_loss") is not None]
    avg_values = [float(row["avg_loss"]) for row in rows if row.get("avg_loss") is not None]
    min_moving = min(moving_values) if moving_values else None
    min_avg = min(avg_values) if avg_values else None
    max_epoch = max(int(row["epoch"]) for row in rows)
    by_epoch = {int(row["epoch"]): row for row in rows}
    scored = []
    for row in rows:
        epoch = int(row["epoch"])
        score = 0.0
        reasons: list[str] = []
        moving = row.get("moving_avg_final_loss")
        avg = row.get("avg_loss")
        if moving is not None and min_moving is not None:
            score += max(0.0, 40.0 - (float(moving) - min_moving) * 1000.0)
            if float(moving) == min_moving:
                reasons.append("moving average final lossが最小")
        if avg is not None and min_avg is not None:
            score += max(0.0, 25.0 - (float(avg) - min_avg) * 800.0)
            if float(avg) == min_avg:
                reasons.append("epoch平均lossが低い")
        prev_avg = by_epoch.get(epoch - 1, {}).get("avg_loss")
        next_avg = by_epoch.get(epoch + 1, {}).get("avg_loss")
        if avg is not None and prev_avg is not None and next_avg is not None and float(avg) <= float(prev_avg) and float(avg) <= float(next_avg):
            score += 15.0
            reasons.append("前後epochよりlossが低い")
        if epoch in output_epochs:
            score += 12.0
            reasons.append("出力LoRAあり")
        if epoch in sample_epochs:
            score += 12.0
            reasons.append("sample画像あり")
        if epoch in selected_epochs:
            score += 60.0
            reasons.append("採用済みepoch")
        if epoch >= max_epoch - 1 and moving is not None and min_moving is not None and float(moving) > min_moving * 1.08:
            score -= 15.0
            reasons.append("後半でlossが悪化傾向")
        scored.append({"epoch": epoch, "score": round(score, 4), "reasons": reasons or ["loss summaryから参考候補として抽出"]})
    return sorted(scored, key=lambda row: (-row["score"], row["epoch"]))


def pick_labels(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not scored:
        return []
    picked_epochs: set[int] = set()
    result = []
    primary = scored[0]
    result.append({**primary, "candidate_label": "primary"})
    picked_epochs.add(primary["epoch"])
    by_epoch = {row["epoch"]: row for row in scored}
    if any("採用済みepoch" in reason for reason in primary["reasons"]):
        prev_row = by_epoch.get(primary["epoch"] - 1)
        next_row = by_epoch.get(primary["epoch"] + 1)
        if prev_row:
            result.append({**prev_row, "candidate_label": "secondary"})
            picked_epochs.add(prev_row["epoch"])
        if next_row:
            result.append({**next_row, "candidate_label": "check"})
            picked_epochs.add(next_row["epoch"])
    if not any(item["candidate_label"] == "secondary" for item in result):
        for row in scored[1:]:
            if row["epoch"] not in picked_epochs:
                result.append({**row, "candidate_label": "secondary"})
                picked_epochs.add(row["epoch"])
                break
    if not any(item["candidate_label"] == "check" for item in result):
        for row in scored:
            if row["epoch"] not in picked_epochs:
                result.append({**row, "candidate_label": "check"})
                picked_epochs.add(row["epoch"])
                break
    for row in scored:
        if row["epoch"] not in picked_epochs:
            result.append({**row, "candidate_label": "low_priority"})
    return sorted(result, key=lambda row: row["epoch"])


def update_sample_review_priority(job_id: int) -> None:
    candidates = {row["epoch"]: row for row in ensure_epoch_candidates_no_create(job_id)}
    prompts = {row["id"]: row for row in fetch_all("SELECT id, prompt_role FROM sample_prompts WHERE job_id = ?", (job_id,))}
    samples = fetch_all("SELECT id, epoch, prompt_id FROM sample_images WHERE job_id = ? AND deleted_at IS NULL", (job_id,))
    with connect() as conn:
        for sample in samples:
            label = candidates.get(sample["epoch"], {}).get("candidate_label") if sample["epoch"] is not None else None
            prompt = prompts.get(sample["prompt_id"]) if sample["prompt_id"] else None
            prompt_role = prompt["prompt_role"] if prompt and "prompt_role" in prompt.keys() else None
            priority = "low"
            auto_label = "low_priority"
            if label == "primary":
                priority = "high"
                auto_label = "candidate_epoch"
            elif label in {"secondary", "check"}:
                priority = "medium"
                auto_label = "candidate_epoch"
            reason = f"候補epoch={label or 'なし'}"
            if prompt_role == "full_body":
                reason += "。full_bodyは顔評価N/A推奨"
            conn.execute(
                "UPDATE sample_images SET review_priority = ?, auto_review_label = ?, auto_review_reason = ? WHERE id = ?",
                (priority, auto_label, reason, sample["id"]),
            )


def ensure_epoch_candidates_no_create(job_id: int) -> list[dict[str, Any]]:
    return [candidate_dict(row) for row in fetch_all("SELECT * FROM training_epoch_candidate_summaries WHERE job_id = ? ORDER BY candidate_rank, epoch", (job_id,))]


def candidate_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    try:
        data["reasons"] = json.loads(data.get("reason_json") or "[]")
    except json.JSONDecodeError:
        data["reasons"] = []
    data["reason_text"] = " / ".join(data["reasons"]) if data["reasons"] else "-"
    return data
