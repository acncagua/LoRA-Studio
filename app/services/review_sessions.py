from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.db import connect, fetch_all, fetch_one, utc_now
from app.services.review_candidates import ensure_epoch_candidates


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
    return {
        "session": session,
        "condition_count": int(condition_count["c"] if condition_count else 0),
        "image_count": int(image_count["c"] if image_count else 0),
        "candidate_epochs": candidate_epochs,
        "matrix_path": matrix_path,
        "can_open_matrix": bool(matrix_path and Path(matrix_path).exists()),
    }


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
