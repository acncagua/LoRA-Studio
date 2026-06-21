from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app import settings as app_settings
from app.db import connect, fetch_all, fetch_one, utc_now
from app.services.embedding_service import (
    EmbeddingSource,
    active_embedding_model,
    dataset_image_sources,
    embedding_coverage,
    embedding_status_for_source,
    latest_embedding_for,
    reference_image_sources,
    review_session_image_sources,
    sample_image_sources,
    validation_image_sources,
)
from app.services.training_runner import process_exists


DEFAULT_REASON = "機械補助判定は参考情報です。最終判断は人間評価を優先してください。"
STALE_RUNNING_GRACE_SECONDS = 60


def _iso_age_seconds(value: str | None) -> float:
    if not value:
        return STALE_RUNNING_GRACE_SECONDS + 1
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return STALE_RUNNING_GRACE_SECONDS + 1
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def load_machine_review_settings() -> dict[str, Any]:
    row = fetch_one("SELECT * FROM machine_review_settings ORDER BY id LIMIT 1")
    if row:
        return dict(row)
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO machine_review_settings(
                active_embedding_model_id, reference_similarity_method,
                overfit_nearest_threshold, overfit_margin_threshold,
                reference_low_threshold, low_confidence_when_mock_provider,
                minimum_reference_images_character, minimum_reference_images_style,
                include_dataset_nearest_check, created_at, updated_at
            )
            VALUES ('mock_image_512', 'avg_max_blend', 0.90, 0.05, 0.20, 1, 3, 6, 1, ?, ?)
            """,
            (now, now),
        )
    return dict(fetch_one("SELECT * FROM machine_review_settings ORDER BY id LIMIT 1"))


def update_machine_review_settings(data: dict[str, Any]) -> None:
    now = utc_now()
    with connect() as conn:
        row = conn.execute("SELECT id FROM machine_review_settings ORDER BY id LIMIT 1").fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO machine_review_settings(
                    active_embedding_model_id, reference_similarity_method,
                    overfit_nearest_threshold, overfit_margin_threshold,
                    reference_low_threshold, low_confidence_when_mock_provider,
                    minimum_reference_images_character, minimum_reference_images_style,
                    include_dataset_nearest_check, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["active_embedding_model_id"],
                    data["reference_similarity_method"],
                    data["overfit_nearest_threshold"],
                    data["overfit_margin_threshold"],
                    data["reference_low_threshold"],
                    data["low_confidence_when_mock_provider"],
                    data["minimum_reference_images_character"],
                    data["minimum_reference_images_style"],
                    data["include_dataset_nearest_check"],
                    now,
                    now,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE machine_review_settings
                SET active_embedding_model_id = ?, reference_similarity_method = ?,
                    overfit_nearest_threshold = ?, overfit_margin_threshold = ?,
                    reference_low_threshold = ?, low_confidence_when_mock_provider = ?,
                    minimum_reference_images_character = ?, minimum_reference_images_style = ?,
                    include_dataset_nearest_check = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    data["active_embedding_model_id"],
                    data["reference_similarity_method"],
                    data["overfit_nearest_threshold"],
                    data["overfit_margin_threshold"],
                    data["reference_low_threshold"],
                    data["low_confidence_when_mock_provider"],
                    data["minimum_reference_images_character"],
                    data["minimum_reference_images_style"],
                    data["include_dataset_nearest_check"],
                    now,
                    row["id"],
                ),
            )


def context_for_training_job(job_id: int) -> dict[str, Any]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Training job not found: {job_id}")
    reference_set = None
    if job["project_id"]:
        reference_set = fetch_one(
            """
            SELECT * FROM reference_sets
            WHERE project_id = ? AND current_version_id IS NOT NULL
            ORDER BY is_default DESC, id DESC LIMIT 1
            """,
            (job["project_id"],),
        )
    if reference_set is None:
        reference_set = fetch_one("SELECT * FROM reference_sets WHERE current_version_id IS NOT NULL ORDER BY is_default DESC, id DESC LIMIT 1")
    return {
        "project_id": job["project_id"],
        "job_id": job_id,
        "validation_run_id": None,
        "dataset_id": job["dataset_id"],
        "dataset_version_id": job["dataset_version_id"],
        "reference_set_id": reference_set["id"] if reference_set else None,
        "reference_set_version_id": reference_set["current_version_id"] if reference_set else None,
    }


def context_for_validation_run(run_id: int) -> dict[str, Any]:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None:
        raise ValueError(f"Validation run not found: {run_id}")
    job_ctx = context_for_training_job(run["job_id"])
    return {
        **job_ctx,
        "project_id": run["project_id"] or job_ctx["project_id"],
        "job_id": run["job_id"],
        "validation_run_id": run_id,
        "reference_set_id": run["reference_set_id"] or job_ctx["reference_set_id"],
        "reference_set_version_id": run["reference_set_version_id"] or job_ctx["reference_set_version_id"],
    }


def context_for_review_session(session_id: int) -> dict[str, Any]:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (session_id,))
    if session is None:
        raise ValueError(f"Review session not found: {session_id}")
    job_ctx = context_for_training_job(session["job_id"])
    return {
        **job_ctx,
        "project_id": session["project_id"] or job_ctx["project_id"],
        "job_id": session["job_id"],
        "validation_run_id": None,
        "dataset_id": session["dataset_id"] or job_ctx["dataset_id"],
        "dataset_version_id": session["dataset_version_id"] or job_ctx["dataset_version_id"],
        "reference_set_id": session["reference_set_id"] or job_ctx["reference_set_id"],
        "reference_set_version_id": session["reference_set_version_id"] or job_ctx["reference_set_version_id"],
    }


VectorCache = dict[tuple[str, int | None, str, str], tuple[np.ndarray, dict[str, Any]] | None]


def vector_for_source(source: EmbeddingSource, model_id: str) -> tuple[np.ndarray, dict[str, Any]] | None:
    if embedding_status_for_source(source, model_id) != "ready":
        return None
    embedding = latest_embedding_for(source, model_id)
    if not embedding or embedding["status"] != "ready" or not embedding["embedding_path"]:
        return None
    path = Path(embedding["embedding_path"])
    if not path.exists():
        return None
    vector = np.load(path).astype("float32")
    if vector.ndim != 1:
        vector = vector.reshape(-1)
    return vector, embedding


def cached_vector_for_source(source: EmbeddingSource, model_id: str, cache: VectorCache | None = None) -> tuple[np.ndarray, dict[str, Any]] | None:
    if cache is None:
        return vector_for_source(source, model_id)
    key = (source.source_type, source.source_id, source.source_path, model_id)
    if key not in cache:
        cache[key] = vector_for_source(source, model_id)
    return cache[key]


def similarity(a: np.ndarray, b: np.ndarray, normalized: bool) -> float:
    if normalized:
        return float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def scored_neighbors(
    source_vec: np.ndarray,
    sources: list[EmbeddingSource],
    model_id: str,
    normalized: bool,
    vector_cache: VectorCache | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in sources:
        loaded = cached_vector_for_source(target, model_id, vector_cache)
        if loaded is None:
            continue
        vec, embedding = loaded
        if vec.shape != source_vec.shape:
            continue
        rows.append({"source": target, "embedding": embedding, "similarity": similarity(source_vec, vec, normalized)})
    rows.sort(key=lambda item: item["similarity"], reverse=True)
    return rows


def reference_sources(reference_set_version_id: int | None) -> list[EmbeddingSource]:
    if not reference_set_version_id:
        return []
    rows = fetch_all(
        """
        SELECT ri.*, rs.project_id
        FROM reference_images ri
        LEFT JOIN reference_sets rs ON rs.id = ri.reference_set_id
        WHERE ri.reference_set_version_id = ?
          AND COALESCE(ri.include_in_machine_review, 1) = 1
        ORDER BY ri.sort_order, ri.id
        """,
        (reference_set_version_id,),
    )
    return [
        EmbeddingSource(
            source_type="reference_image",
            source_id=row["id"],
            source_path=row["image_path"],
            project_id=row["project_id"],
            dataset_id=row["dataset_id"],
            dataset_version_id=row["dataset_version_id"],
            reference_set_id=row["reference_set_id"],
            reference_set_version_id=row["reference_set_version_id"],
        )
        for row in rows
    ]


def source_metadata(source_type: str, source_id: int) -> dict[str, Any]:
    if source_type == "sample_image":
        row = fetch_one(
            """
            SELECT si.*, sp.name AS prompt_key, sp.prompt_role AS prompt_role
            FROM sample_images si
            LEFT JOIN sample_prompts sp ON sp.id = si.prompt_id
            WHERE si.id = ?
            """,
            (source_id,),
        )
    elif source_type == "validation_image":
        row = fetch_one("SELECT * FROM validation_images WHERE id = ?", (source_id,))
    elif source_type == "review_session_image":
        row = fetch_one(
            """
            SELECT rsi.*, rsc.prompt, rsc.negative_prompt
            FROM review_session_images rsi
            LEFT JOIN review_session_conditions rsc ON rsc.id = rsi.condition_id
            WHERE rsi.id = ?
            """,
            (source_id,),
        )
    else:
        row = None
    return dict(row) if row else {}


def confidence_for(provider: str, reference_count: int, settings: dict[str, Any], reference_label: str | None) -> tuple[str, list[str]]:
    reasons: list[str] = [DEFAULT_REASON]
    if provider == "mock" and settings.get("low_confidence_when_mock_provider"):
        reasons.append("mock providerのスコアは意味的な画像評価ではありません。機能経路テスト用です。")
        return "low", reasons
    minimum = int(settings.get("minimum_reference_images_character") or 3)
    if reference_count == 0:
        reasons.append("Reference画像がありません。")
        return "unavailable", reasons
    if reference_count < minimum:
        reasons.append(f"Reference Setが{reference_count}枚だけのため、判定が偏る可能性があります。")
        return "low", reasons
    if reference_label and reference_label.upper() in {"WARNING", "ERROR"}:
        reasons.append(f"Reference Set completeness が {reference_label} です。")
        return "low", reasons
    return "medium", reasons


def overfit_label(nearest: float | None, margin: float | None, provider: str, settings: dict[str, Any]) -> str:
    if provider == "mock":
        return "unknown"
    if nearest is None:
        return "unknown"
    high = float(settings.get("overfit_nearest_threshold") or 0.9)
    margin_high = float(settings.get("overfit_margin_threshold") or 0.05)
    if nearest >= high and (margin or 0) >= margin_high:
        return "high"
    if nearest >= high:
        return "medium"
    return "low"


def assist_label(ref_max: float | None, overfit: str, confidence: str, provider: str, settings: dict[str, Any]) -> tuple[str, float | None]:
    if confidence == "unavailable" or ref_max is None:
        return "unavailable", None
    if provider == "mock" or confidence == "low":
        return "low_confidence", ref_max
    score = ref_max
    if overfit == "high":
        return "possible_overfit", score - 0.2
    if ref_max >= 0.75:
        return "primary_candidate", score
    if ref_max >= float(settings.get("reference_low_threshold") or 0.2):
        return "secondary_candidate", score
    return "check_manually", score


def score_source(
    source: EmbeddingSource,
    context: dict[str, Any],
    model: dict[str, Any],
    settings: dict[str, Any],
    vector_cache: VectorCache | None = None,
) -> dict[str, Any]:
    model_id = model["id"]
    loaded = cached_vector_for_source(source, model_id, vector_cache)
    if loaded is None:
        raise ValueError(f"Embedding is not ready: {source.source_type}#{source.source_id}")
    source_vec, embedding = loaded
    normalized = bool(embedding["normalized"])
    provider = model.get("provider") or embedding.get("provider") or "mock"
    meta = source_metadata(source.source_type, int(source.source_id or 0))

    refs = reference_sources(context.get("reference_set_version_id"))
    ref_label = None
    if context.get("reference_set_version_id"):
        ver = fetch_one("SELECT completeness_label FROM reference_set_versions WHERE id = ?", (context["reference_set_version_id"],))
        ref_label = ver["completeness_label"] if ver else None
    ref_scores = scored_neighbors(source_vec, refs, model_id, normalized, vector_cache)
    ref_values = [row["similarity"] for row in ref_scores]
    ref_avg = sum(ref_values) / len(ref_values) if ref_values else None
    ref_max = ref_values[0] if ref_values else None
    nearest_ref_id = ref_scores[0]["source"].source_id if ref_scores else None

    dataset_scores: list[dict[str, Any]] = []
    if settings.get("include_dataset_nearest_check") and context.get("dataset_version_id"):
        dataset_scores = scored_neighbors(source_vec, dataset_image_sources(int(context["dataset_version_id"])), model_id, normalized, vector_cache)
    dataset_values = [row["similarity"] for row in dataset_scores]
    dataset_avg = sum(dataset_values) / len(dataset_values) if dataset_values else None
    nearest_dataset = dataset_scores[0] if dataset_scores else None
    nearest_dataset_similarity = dataset_values[0] if dataset_values else None
    second_dataset_similarity = dataset_values[1] if len(dataset_values) > 1 else None
    margin = (nearest_dataset_similarity - second_dataset_similarity) if nearest_dataset_similarity is not None and second_dataset_similarity is not None else None
    nearest_dataset_id = nearest_dataset["source"].source_id if nearest_dataset else None

    confidence, reasons = confidence_for(provider, len(refs), settings, ref_label)
    overfit = overfit_label(nearest_dataset_similarity, margin, provider, settings)
    label, assist_score = assist_label(ref_max, overfit, confidence, provider, settings)
    if context.get("reference_set_version_id") and context.get("dataset_version_id"):
        ref_version = fetch_one("SELECT dataset_version_id FROM reference_set_versions WHERE id = ?", (context["reference_set_version_id"],))
        if ref_version and ref_version["dataset_version_id"] and ref_version["dataset_version_id"] != context["dataset_version_id"]:
            reasons.append("Reference Setと対象学習ジョブのDataset Versionが異なります。")

    values = {
        "source_type": source.source_type,
        "source_id": source.source_id,
        "project_id": context.get("project_id"),
        "job_id": context.get("job_id"),
        "validation_run_id": context.get("validation_run_id"),
        "reference_set_id": context.get("reference_set_id"),
        "reference_set_version_id": context.get("reference_set_version_id"),
        "dataset_id": context.get("dataset_id"),
        "dataset_version_id": context.get("dataset_version_id"),
        "embedding_model_id": model_id,
        "provider": provider,
        "prompt_key": meta.get("prompt_key") or meta.get("image_role"),
        "prompt_role": meta.get("prompt_role") or meta.get("image_role"),
        "epoch": meta.get("epoch"),
        "lora_weight": meta.get("lora_weight"),
        "reference_similarity_avg": ref_avg,
        "reference_similarity_max": ref_max,
        "nearest_reference_image_id": nearest_ref_id,
        "nearest_reference_similarity": ref_max,
        "dataset_similarity_avg": dataset_avg,
        "nearest_dataset_image_id": nearest_dataset_id,
        "nearest_dataset_similarity": nearest_dataset_similarity,
        "dataset_top1_margin": margin,
        "overfit_risk_label": overfit,
        "assist_score": assist_score,
        "assist_label": label,
        "confidence_label": confidence,
        "reason_json": json.dumps({"reasons": reasons, "reference_count": len(refs), "provider": provider}, ensure_ascii=False),
    }
    upsert_machine_review_score(values)
    return values


def upsert_machine_review_score(values: dict[str, Any]) -> int:
    now = utc_now()
    values = {**values, "created_at": now, "updated_at": now}
    existing = fetch_one(
        """
        SELECT id FROM machine_review_scores
        WHERE source_type = ? AND source_id = ? AND embedding_model_id = ?
          AND COALESCE(reference_set_version_id, 0) = COALESCE(?, 0)
          AND COALESCE(dataset_version_id, 0) = COALESCE(?, 0)
        """,
        (
            values["source_type"],
            values["source_id"],
            values["embedding_model_id"],
            values.get("reference_set_version_id"),
            values.get("dataset_version_id"),
        ),
    )
    with connect() as conn:
        if existing:
            values["id"] = existing["id"]
            conn.execute(
                """
                UPDATE machine_review_scores SET
                    project_id = :project_id, job_id = :job_id, validation_run_id = :validation_run_id,
                    reference_set_id = :reference_set_id, reference_set_version_id = :reference_set_version_id,
                    dataset_id = :dataset_id, dataset_version_id = :dataset_version_id,
                    provider = :provider, prompt_key = :prompt_key, prompt_role = :prompt_role,
                    epoch = :epoch, lora_weight = :lora_weight,
                    reference_similarity_avg = :reference_similarity_avg,
                    reference_similarity_max = :reference_similarity_max,
                    nearest_reference_image_id = :nearest_reference_image_id,
                    nearest_reference_similarity = :nearest_reference_similarity,
                    dataset_similarity_avg = :dataset_similarity_avg,
                    nearest_dataset_image_id = :nearest_dataset_image_id,
                    nearest_dataset_similarity = :nearest_dataset_similarity,
                    dataset_top1_margin = :dataset_top1_margin,
                    overfit_risk_label = :overfit_risk_label,
                    assist_score = :assist_score, assist_label = :assist_label,
                    confidence_label = :confidence_label, reason_json = :reason_json,
                    updated_at = :updated_at
                WHERE id = :id
                """,
                values,
            )
            return int(existing["id"])
        cur = conn.execute(
            """
            INSERT INTO machine_review_scores(
                source_type, source_id, project_id, job_id, validation_run_id,
                reference_set_id, reference_set_version_id, dataset_id, dataset_version_id,
                embedding_model_id, provider, prompt_key, prompt_role, epoch, lora_weight,
                reference_similarity_avg, reference_similarity_max,
                nearest_reference_image_id, nearest_reference_similarity,
                dataset_similarity_avg, nearest_dataset_image_id, nearest_dataset_similarity,
                dataset_top1_margin, overfit_risk_label, assist_score, assist_label,
                confidence_label, reason_json, created_at, updated_at
            )
            VALUES (
                :source_type, :source_id, :project_id, :job_id, :validation_run_id,
                :reference_set_id, :reference_set_version_id, :dataset_id, :dataset_version_id,
                :embedding_model_id, :provider, :prompt_key, :prompt_role, :epoch, :lora_weight,
                :reference_similarity_avg, :reference_similarity_max,
                :nearest_reference_image_id, :nearest_reference_similarity,
                :dataset_similarity_avg, :nearest_dataset_image_id, :nearest_dataset_similarity,
                :dataset_top1_margin, :overfit_risk_label, :assist_score, :assist_label,
                :confidence_label, :reason_json, :created_at, :updated_at
            )
            """,
            values,
        )
        return int(cur.lastrowid)


def machine_review_context(target_type: str, target_id: int, reference_set_version_id: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = load_machine_review_settings()
    model = active_embedding_model()
    if settings.get("active_embedding_model_id"):
        row = fetch_one("SELECT * FROM embedding_models WHERE id = ?", (settings["active_embedding_model_id"],))
        if row:
            model = dict(row)
    if target_type == "training_job_samples":
        context = context_for_training_job(target_id)
    elif target_type in {"validation_run_images", "validation_run_images_missing"}:
        context = context_for_validation_run(target_id)
    elif target_type == "review_session_images":
        context = context_for_review_session(target_id)
    else:
        raise ValueError(f"Unsupported machine review target: {target_type}")
    if reference_set_version_id:
        ref = fetch_one("SELECT * FROM reference_set_versions WHERE id = ?", (reference_set_version_id,))
        if ref:
            context["reference_set_version_id"] = reference_set_version_id
            context["reference_set_id"] = ref["reference_set_id"]
    return context, model


def create_machine_review_job(target_type: str, target_id: int, context: dict[str, Any] | None = None, model_id: str | None = None, reference_set_version_id: int | None = None) -> int:
    if context is None or model_id is None:
        context, model = machine_review_context(target_type, target_id, reference_set_version_id=reference_set_version_id)
        model_id = model.get("id") or "mock_image_512"
    sources = sources_for_target(target_type, target_id, context=context, model_id=model_id)
    now = utc_now()
    log_dir = app_settings.LOGS_DIR / "machine_review"
    log_dir.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO machine_review_jobs(
                target_type, target_id, reference_set_id, reference_set_version_id,
                dataset_id, dataset_version_id, embedding_model_id, status, total_count,
                log_path, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?, ?)
            """,
            (
                target_type,
                target_id,
                context.get("reference_set_id"),
                context.get("reference_set_version_id"),
                context.get("dataset_id"),
                context.get("dataset_version_id"),
                model_id,
                len(sources),
                "",
                now,
                now,
            ),
        )
        job_id = int(cur.lastrowid)
        log_path = log_dir / f"machine_review_job_{job_id:06d}.log"
        conn.execute("UPDATE machine_review_jobs SET log_path = ? WHERE id = ?", (str(log_path), job_id))
        return job_id


def sources_for_target(target_type: str, target_id: int, context: dict[str, Any] | None = None, model_id: str | None = None) -> list[EmbeddingSource]:
    if target_type == "training_job_samples":
        return sample_image_sources(target_id)
    if target_type == "validation_run_images":
        return validation_image_sources(target_id)
    if target_type == "validation_run_images_missing":
        if context is None or model_id is None:
            context, model = machine_review_context("validation_run_images", target_id)
            model_id = model.get("id") or "mock_image_512"
        return [
            source
            for source in validation_image_sources(target_id)
            if not has_current_machine_review_score(source, context, model_id)
        ]
    if target_type == "review_session_images":
        return review_session_image_sources(target_id)
    return []


def has_current_machine_review_score(source: EmbeddingSource, context: dict[str, Any], model_id: str) -> bool:
    if source.source_id is None:
        return False
    row = fetch_one(
        """
        SELECT id FROM machine_review_scores
        WHERE source_type = ? AND source_id = ? AND embedding_model_id = ?
          AND COALESCE(reference_set_version_id, 0) = COALESCE(?, 0)
          AND COALESCE(dataset_version_id, 0) = COALESCE(?, 0)
        LIMIT 1
        """,
        (
            source.source_type,
            source.source_id,
            model_id,
            context.get("reference_set_version_id"),
            context.get("dataset_version_id"),
        ),
    )
    return row is not None


def run_machine_review_job(job_id: int) -> dict[str, Any]:
    job = fetch_one("SELECT * FROM machine_review_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Machine Review Job not found: {job_id}")
    target_type = job["target_type"]
    target_id = int(job["target_id"])
    context, model = machine_review_context(target_type, target_id, reference_set_version_id=job["reference_set_version_id"])
    model_id = job["embedding_model_id"] or model.get("id") or "mock_image_512"
    model_row = fetch_one("SELECT * FROM embedding_models WHERE id = ?", (model_id,))
    if model_row:
        model = dict(model_row)
    settings = load_machine_review_settings()
    sources = sources_for_target(target_type, target_id, context=context, model_id=model_id)
    vector_cache: VectorCache = {}
    now = utc_now()
    started = time.time()
    scored = skipped = failed = processed = 0
    error_message = ""
    with connect() as conn:
        conn.execute(
            "UPDATE machine_review_jobs SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ? WHERE id = ?",
            (now, now, job_id),
        )
    for source in sources:
        processed += 1
        try:
            score_source(source, context, model, settings, vector_cache)
            scored += 1
            print(f"[{processed}/{len(sources)}] scored {source.source_type}#{source.source_id}", flush=True)
        except Exception as exc:
            message = str(exc)
            if "Embedding is not ready" in message:
                skipped += 1
                error_message = message
                print(f"[{processed}/{len(sources)}] skipped {source.source_type}#{source.source_id}: {message}", flush=True)
            else:
                failed += 1
                error_message = message
                print(f"[{processed}/{len(sources)}] failed {source.source_type}#{source.source_id}: {message}", flush=True)
        with connect() as conn:
            conn.execute(
                """
                UPDATE machine_review_jobs
                SET processed_count = ?, scored_count = ?, skipped_count = ?,
                    failed_count = ?, error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (processed, scored, skipped, failed, error_message, utc_now(), job_id),
            )
    elapsed = int(time.time() - started)
    ended = utc_now()
    status = "completed" if failed == 0 else "failed"
    with connect() as conn:
        conn.execute(
            """
            UPDATE machine_review_jobs
            SET status = ?, processed_count = ?, scored_count = ?, skipped_count = ?,
                failed_count = ?, ended_at = ?, elapsed_seconds = ?, return_code = ?,
                error_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, processed, scored, skipped, failed, ended, elapsed, 0 if status == "completed" else 1, error_message, ended, job_id),
        )
    return {"job_id": job_id, "status": status, "processed": processed, "scored": scored, "failed": failed}


def run_machine_review(target_type: str, target_id: int, reference_set_version_id: int | None = None) -> dict[str, Any]:
    job_id = create_machine_review_job(target_type, target_id, reference_set_version_id=reference_set_version_id)
    return run_machine_review_job(job_id)


def start_machine_review_job(job_id: int) -> None:
    reconcile_stale_machine_review_jobs()
    running = fetch_one("SELECT id FROM machine_review_jobs WHERE status = 'running' AND id != ? LIMIT 1", (job_id,))
    if running:
        raise RuntimeError(f"Machine Review Job #{running['id']} が実行中です。")
    row = fetch_one("SELECT * FROM machine_review_jobs WHERE id = ?", (job_id,))
    if row is None:
        raise RuntimeError("Machine Review Jobが見つかりません。")
    if row["status"] not in {"planned", "failed", "stopped"}:
        raise RuntimeError(f"Machine Review Job #{job_id} は開始できない状態です: {row['status']}")
    argv = [sys.executable, "-m", "app.services.machine_review_worker", "--machine-review-job-id", str(job_id)]
    log_path = Path(row["log_path"] or app_settings.LOGS_DIR / "machine_review" / f"machine_review_job_{job_id:06d}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        argv,
        cwd=str(app_settings.ROOT_DIR),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_handle.close()
    now = utc_now()
    with connect() as conn:
        conn.execute(
            "UPDATE machine_review_jobs SET status = 'running', process_id = ?, started_at = ?, updated_at = ? WHERE id = ?",
            (proc.pid, now, now, job_id),
        )


def reconcile_stale_machine_review_jobs() -> int:
    rows = fetch_all("SELECT * FROM machine_review_jobs WHERE status = 'running' ORDER BY id")
    fixed = 0
    now = utc_now()
    for row in rows:
        pid = row["process_id"]
        if pid:
            try:
                if process_exists(int(pid)):
                    continue
            except (TypeError, ValueError):
                pass
        elif _iso_age_seconds(row["started_at"] or row["updated_at"] or row["created_at"]) <= STALE_RUNNING_GRACE_SECONDS:
            continue
        message = "Machine Review process was not found. Marked stopped by stale reconciliation."
        with connect() as conn:
            conn.execute(
                """
                UPDATE machine_review_jobs
                SET status = 'stopped', process_id = NULL,
                    ended_at = COALESCE(ended_at, ?), updated_at = ?,
                    return_code = COALESCE(return_code, -1),
                    error_message = COALESCE(NULLIF(error_message, ''), ?)
                WHERE id = ? AND status = 'running'
                """,
                (now, now, message, row["id"]),
            )
        fixed += 1
    return fixed


def stop_machine_review_job(job_id: int) -> None:
    row = fetch_one("SELECT * FROM machine_review_jobs WHERE id = ?", (job_id,))
    if row is None:
        return
    pid = row["process_id"]
    if pid:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, check=False)
        else:
            try:
                os.kill(int(pid), 15)
            except OSError:
                pass
    now = utc_now()
    with connect() as conn:
        conn.execute(
            "UPDATE machine_review_jobs SET status = 'stopped', ended_at = ?, updated_at = ?, return_code = COALESCE(return_code, -1) WHERE id = ?",
            (now, now, job_id),
        )


def latest_machine_review_jobs(limit: int = 20) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in fetch_all(
            "SELECT * FROM machine_review_jobs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
    ]


def machine_review_score_coverage(target_type: str, target_id: int, model_id: str) -> dict[str, Any]:
    if target_type == "training_job_samples":
        source_type = "sample_image"
        where = "job_id = ?"
    elif target_type == "validation_run_images":
        source_type = "validation_image"
        where = "validation_run_id = ?"
    elif target_type == "review_session_images":
        source_type = "review_session_image"
        where = "source_id IN (SELECT id FROM review_session_images WHERE review_session_id = ?)"
    else:
        source_type = ""
        where = "0"
    total = len(sources_for_target(target_type, target_id))
    row = fetch_one(
        f"""
        SELECT COUNT(DISTINCT source_id) AS count
        FROM machine_review_scores
        WHERE source_type = ? AND embedding_model_id = ? AND {where}
        """,
        (source_type, model_id, target_id),
    )
    ready = int(row["count"] or 0) if row else 0
    return {"total": total, "ready": ready, "missing": max(total - ready, 0), "ready_rate": (ready / total) if total else 0}


def readiness_label(coverage: dict[str, Any] | None) -> str:
    if not coverage or int(coverage.get("total") or 0) == 0:
        return "missing"
    if int(coverage.get("ready") or 0) >= int(coverage.get("total") or 0):
        return "ok"
    if int(coverage.get("ready") or 0) > 0:
        return "partial"
    return "missing"


def reference_role_distribution(reference_set_version_id: int | None) -> dict[str, Any]:
    expected_roles = ["face_front", "upper_body", "full_body", "expression"]
    if not reference_set_version_id:
        return {"roles": {}, "missing_roles": expected_roles, "dominant_role": None, "dominant_rate": 0}
    rows = fetch_all(
        """
        SELECT image_role, COUNT(*) AS count
        FROM reference_images
        WHERE reference_set_version_id = ?
          AND COALESCE(include_in_machine_review, 1) = 1
        GROUP BY image_role
        ORDER BY image_role
        """,
        (reference_set_version_id,),
    )
    roles = {row["image_role"] or "unknown": int(row["count"] or 0) for row in rows}
    total = sum(roles.values())
    dominant_role = None
    dominant_rate = 0.0
    if roles and total:
        dominant_role, dominant_count = sorted(roles.items(), key=lambda item: (-item[1], item[0]))[0]
        dominant_rate = dominant_count / total
    return {
        "roles": roles,
        "missing_roles": [role for role in expected_roles if roles.get(role, 0) == 0],
        "dominant_role": dominant_role,
        "dominant_rate": dominant_rate,
    }


def machine_review_readiness(target_type: str, target_id: int) -> dict[str, Any]:
    context, model = machine_review_context(target_type, target_id)
    model_id = model.get("id") or "mock_image_512"
    provider = model.get("provider") or "mock"
    reference_version_id = context.get("reference_set_version_id")
    dataset_version_id = context.get("dataset_version_id")
    reference_coverage = embedding_coverage("reference_set_version", int(reference_version_id)) if reference_version_id else None
    dataset_coverage = embedding_coverage("dataset_version", int(dataset_version_id)) if dataset_version_id else None
    if target_type == "training_job_samples":
        target_coverage = embedding_coverage("training_job_samples", target_id)
        target_label = "サンプル画像"
    else:
        target_coverage = embedding_coverage("validation_run", target_id)
        target_label = "検証画像"
    score_coverage = machine_review_score_coverage(target_type, target_id, model_id)
    reference_set = fetch_one("SELECT * FROM reference_sets WHERE id = ?", (context["reference_set_id"],)) if context.get("reference_set_id") else None
    reference_version = fetch_one("SELECT * FROM reference_set_versions WHERE id = ?", (reference_version_id,)) if reference_version_id else None
    reference_count = int(reference_coverage.get("total", 0) if reference_coverage else 0)
    role_distribution = reference_role_distribution(int(reference_version_id)) if reference_version_id else reference_role_distribution(None)

    warnings: list[str] = []
    actions: list[str] = []
    if provider == "mock":
        warnings.append("mock providerのスコアは意味的な画像評価ではありません。機能経路テスト用です。")
    if not reference_version_id:
        actions.append("Reference Setを作成し、基準画像を登録してください。")
    elif reference_count < int(load_machine_review_settings().get("minimum_reference_images_character") or 3):
        warnings.append("Reference Setの枚数が少ないため、機械補助レビューは低信頼になります。")
        actions.append("可能ならReference Setを増やしてください。")
    if role_distribution["missing_roles"]:
        warnings.append("Reference Setのroleが不足しています: " + ", ".join(role_distribution["missing_roles"]))
    if role_distribution["dominant_rate"] >= 0.75 and reference_count >= 3:
        warnings.append(f"Reference Setが {role_distribution['dominant_role']} に偏っています。顔・上半身・全身・表情を分散すると補助判定が安定します。")
    if readiness_label(reference_coverage) != "ok":
        actions.append("Reference画像Embeddingを計算してください。")
    if readiness_label(dataset_coverage) != "ok":
        actions.append("Dataset画像Embeddingを計算してください。")
    if readiness_label(target_coverage) != "ok":
        actions.append(f"{target_label}Embeddingを計算してください。")
    if score_coverage["ready"] < score_coverage["total"]:
        actions.append("機械補助レビューを実行してください。")
    if not actions:
        actions.append("Review Queueまたは検証画像で機械補助レビュー結果を確認してください。")

    return {
        "target_type": target_type,
        "target_id": target_id,
        "context": context,
        "model": model,
        "model_id": model_id,
        "provider": provider,
        "reference_set": dict(reference_set) if reference_set else None,
        "reference_version": dict(reference_version) if reference_version else None,
        "reference_count": reference_count,
        "reference_role_distribution": role_distribution,
        "reference_coverage": reference_coverage,
        "dataset_coverage": dataset_coverage,
        "target_coverage": target_coverage,
        "target_label": target_label,
        "score_coverage": score_coverage,
        "reference_label": readiness_label(reference_coverage),
        "dataset_label": readiness_label(dataset_coverage),
        "target_embedding_label": readiness_label(target_coverage),
        "score_label": "ok" if score_coverage["ready"] >= score_coverage["total"] and score_coverage["total"] else "missing",
        "warnings": warnings,
        "next_actions": actions,
    }


def scores_for_job(job_id: int) -> list[dict[str, Any]]:
    return [dict(row) for row in fetch_all("SELECT * FROM machine_review_scores WHERE job_id = ? AND source_type = 'sample_image' ORDER BY epoch, source_id", (job_id,))]


def scores_for_validation_run(run_id: int) -> list[dict[str, Any]]:
    return [dict(row) for row in fetch_all("SELECT * FROM machine_review_scores WHERE validation_run_id = ? AND source_type = 'validation_image' ORDER BY prompt_key, lora_weight, source_id", (run_id,))]


def score_map_for_samples(job_id: int) -> dict[int, dict[str, Any]]:
    return {int(row["source_id"]): dict(row) for row in fetch_all("SELECT * FROM machine_review_scores WHERE job_id = ? AND source_type = 'sample_image'", (job_id,))}


def score_map_for_validation(run_id: int) -> dict[int, dict[str, Any]]:
    return {int(row["source_id"]): dict(row) for row in fetch_all("SELECT * FROM machine_review_scores WHERE validation_run_id = ? AND source_type = 'validation_image'", (run_id,))}


def epoch_machine_summary(job_id: int) -> dict[int, dict[str, Any]]:
    rows = scores_for_job(job_id)
    by_epoch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("epoch") is None:
            continue
        by_epoch.setdefault(int(row["epoch"]), []).append(row)
    result: dict[int, dict[str, Any]] = {}
    for epoch, items in by_epoch.items():
        ref_values = [r["reference_similarity_max"] for r in items if r["reference_similarity_max"] is not None]
        ds_values = [r["nearest_dataset_similarity"] for r in items if r["nearest_dataset_similarity"] is not None]
        margins = [r["dataset_top1_margin"] for r in items if r.get("dataset_top1_margin") is not None]
        labels = [r["assist_label"] for r in items if r.get("assist_label")]
        confidences = [r["confidence_label"] for r in items if r.get("confidence_label")]
        nearest_ids = [r["nearest_dataset_image_id"] for r in items if r.get("nearest_dataset_image_id") is not None]
        result[epoch] = {
            "epoch": epoch,
            "count": len(items),
            "reference_similarity_max": max(ref_values) if ref_values else None,
            "reference_similarity_avg": sum(ref_values) / len(ref_values) if ref_values else None,
            "nearest_dataset_similarity": max(ds_values) if ds_values else None,
            "nearest_dataset_similarity_avg": sum(ds_values) / len(ds_values) if ds_values else None,
            "dataset_top1_margin_avg": sum(margins) / len(margins) if margins else None,
            "nearest_dataset_top_id": most_common([str(value) for value in nearest_ids]),
            "nearest_dataset_top_count": max([nearest_ids.count(value) for value in set(nearest_ids)], default=0),
            "assist_label": most_common(labels) or "unavailable",
            "confidence_label": most_common(confidences) or "unavailable",
            "overfit_risk_label": most_common([r["overfit_risk_label"] for r in items if r.get("overfit_risk_label")]) or "unknown",
        }
    return result


def validation_weight_summary(run_id: int) -> list[dict[str, Any]]:
    rows = scores_for_validation_run(run_id)
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row.get("lora_weight"), row.get("prompt_key")), []).append(row)
    summary = []
    for (weight, prompt_key), items in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        ref = [r["reference_similarity_max"] for r in items if r["reference_similarity_max"] is not None]
        ds = [r["nearest_dataset_similarity"] for r in items if r["nearest_dataset_similarity"] is not None]
        summary.append(
            {
                "lora_weight": weight,
                "prompt_key": prompt_key,
                "count": len(items),
                "reference_similarity_avg": sum(ref) / len(ref) if ref else None,
                "nearest_dataset_similarity": max(ds) if ds else None,
                "assist_label": most_common([r["assist_label"] for r in items if r.get("assist_label")]) or "unavailable",
                "confidence_label": most_common([r["confidence_label"] for r in items if r.get("confidence_label")]) or "unavailable",
                "overfit_risk_label": most_common([r["overfit_risk_label"] for r in items if r.get("overfit_risk_label")]) or "unknown",
            }
        )
    return summary


def reference_set_readiness(reference_set: Any, coverage: dict[str, Any] | None) -> dict[str, Any]:
    image_count = int(reference_set["image_count"] or 0) if reference_set and "image_count" in reference_set.keys() else int(coverage.get("total", 0) if coverage else 0)
    completeness = (reference_set["completeness_label"] or "UNKNOWN") if reference_set and "completeness_label" in reference_set.keys() else "UNKNOWN"
    ready = int(coverage.get("ready", 0) if coverage else 0)
    total = int(coverage.get("total", 0) if coverage else 0)
    if image_count == 0 or total == 0 or ready == 0:
        label = "ERROR"
    elif completeness == "OK" and ready == total:
        label = "OK"
    else:
        label = "WARNING"
    return {"label": label, "image_count": image_count, "ready": ready, "total": total, "completeness_label": completeness}


def most_common(values: list[str]) -> str | None:
    if not values:
        return None
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
