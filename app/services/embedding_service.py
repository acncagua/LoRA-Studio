from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from app import settings
from app.db import connect, fetch_all, fetch_one, utc_now

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
JOB_TYPE_LABELS = {
    "dataset_version": "Dataset Version",
    "reference_set_version": "Reference Set Version",
    "training_job_samples": "Sample Images",
    "validation_run": "Validation Run",
    "review_session": "Review Session",
    "selected_sources": "Selected Sources",
}

STALE_RUNNING_GRACE_SECONDS = 10 * 60


@dataclass
class EmbeddingSource:
    source_type: str
    source_id: int | None
    source_path: str
    project_id: int | None = None
    dataset_id: int | None = None
    dataset_version_id: int | None = None
    reference_set_id: int | None = None
    reference_set_version_id: int | None = None
    job_id: int | None = None
    validation_run_id: int | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _iso_age_seconds(value: str | None) -> float:
    if not value:
        return STALE_RUNNING_GRACE_SECONDS + 1
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return STALE_RUNNING_GRACE_SECONDS + 1
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed).total_seconds()


def image_metadata(path: Path, max_image_size: int | None = None) -> dict[str, Any]:
    stat = path.stat()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        if max_image_size and max(width, height) > max_image_size:
            image.thumbnail((max_image_size, max_image_size))
            width, height = image.size
    return {
        "sha256": file_sha256(path),
        "file_size": stat.st_size,
        "mtime": stat.st_mtime,
        "width": width,
        "height": height,
    }


def load_embedding_settings() -> dict[str, Any]:
    row = fetch_one("SELECT * FROM embedding_settings ORDER BY id LIMIT 1")
    if row is None:
        init_default_embedding_settings()
        row = fetch_one("SELECT * FROM embedding_settings ORDER BY id LIMIT 1")
    return dict(row) if row else {}


def active_embedding_model() -> dict[str, Any]:
    settings_row = load_embedding_settings()
    model_id = settings_row.get("active_embedding_model_id") or "mock_image_512"
    row = fetch_one("SELECT * FROM embedding_models WHERE id = ?", (model_id,))
    if row is None:
        row = fetch_one("SELECT * FROM embedding_models WHERE id = 'mock_image_512'")
    return dict(row) if row else {}


def init_default_embedding_settings() -> None:
    now = utc_now()
    with connect() as conn:
        existing = conn.execute("SELECT id FROM embedding_settings ORDER BY id LIMIT 1").fetchone()
        if existing:
            return
        conn.execute(
            """
            INSERT INTO embedding_settings(
                active_embedding_model_id, python_path, device, dtype, batch_size,
                cache_root, allow_model_download, max_image_size, num_workers,
                created_at, updated_at
            )
            VALUES ('mock_image_512', '', 'auto', 'fp32', 8, ?, 0, 1024, 1, ?, ?)
            """,
            (str(settings.EMBEDDINGS_DIR), now, now),
        )


def update_embedding_settings(values: dict[str, Any]) -> None:
    now = utc_now()
    with connect() as conn:
        row = conn.execute("SELECT id FROM embedding_settings ORDER BY id LIMIT 1").fetchone()
        params = {
            "active_embedding_model_id": values.get("active_embedding_model_id") or "mock_image_512",
            "python_path": values.get("python_path") or "",
            "device": values.get("device") or "auto",
            "dtype": values.get("dtype") or "fp32",
            "batch_size": int(values.get("batch_size") or 8),
            "cache_root": values.get("cache_root") or str(settings.EMBEDDINGS_DIR),
            "allow_model_download": 1 if values.get("allow_model_download") else 0,
            "max_image_size": int(values.get("max_image_size") or 1024),
            "num_workers": int(values.get("num_workers") or 1),
            "updated_at": now,
        }
        if row:
            params["id"] = row["id"]
            conn.execute(
                """
                UPDATE embedding_settings SET
                    active_embedding_model_id = :active_embedding_model_id,
                    python_path = :python_path,
                    device = :device,
                    dtype = :dtype,
                    batch_size = :batch_size,
                    cache_root = :cache_root,
                    allow_model_download = :allow_model_download,
                    max_image_size = :max_image_size,
                    num_workers = :num_workers,
                    updated_at = :updated_at
                WHERE id = :id
                """,
                params,
            )
        else:
            params["created_at"] = now
            conn.execute(
                """
                INSERT INTO embedding_settings(
                    active_embedding_model_id, python_path, device, dtype, batch_size,
                    cache_root, allow_model_download, max_image_size, num_workers,
                    created_at, updated_at
                )
                VALUES (
                    :active_embedding_model_id, :python_path, :device, :dtype,
                    :batch_size, :cache_root, :allow_model_download,
                    :max_image_size, :num_workers, :created_at, :updated_at
                )
                """,
                params,
            )


def provider_preflight(model_id: str | None = None, deep: bool = True) -> dict[str, Any]:
    model = dict(fetch_one("SELECT * FROM embedding_models WHERE id = ?", (model_id,)) or active_embedding_model())
    settings_row = load_embedding_settings()
    provider = model.get("provider") or "mock"
    result: dict[str, Any] = {
        "provider": provider,
        "model_id": model.get("id"),
        "model_name": model.get("model_name"),
        "status": "OK",
        "checks": [],
    }

    python_path = settings_row.get("python_path") or sys.executable
    try:
        uses_worker_python = Path(python_path).resolve() != Path(sys.executable).resolve()
    except OSError:
        uses_worker_python = bool(settings_row.get("python_path"))
    result["checks"].append({"name": "python_path", "status": "OK" if Path(python_path).exists() else "WARNING", "message": python_path})
    result["download_allowed"] = bool(settings_row.get("allow_model_download"))

    if provider == "mock":
        result["checks"].append({"name": "mock provider", "status": "OK", "message": "外部モデル不要で利用できます。"})
        result["vector_dim"] = int(model.get("vector_dim") or 512)
    elif provider == "transformers_clip":
        transformers_ok = importlib.util.find_spec("transformers") is not None
        torch_ok = importlib.util.find_spec("torch") is not None
        if transformers_ok:
            result["checks"].append({"name": "transformers import", "status": "OK", "message": "transformers is importable."})
        elif uses_worker_python:
            result["checks"].append(
                {
                    "name": "transformers import",
                    "status": "INFO",
                    "message": "アプリ側Pythonでは未検出です。指定されたworker Pythonで確認します。",
                }
            )
        else:
            result["status"] = "WARNING"
            result["checks"].append({"name": "transformers import", "status": "WARNING", "message": "transformers is not installed."})
        if torch_ok:
            requested_device = str(settings_row.get("device") or "auto")
            requested_dtype = str(settings_row.get("dtype") or "fp32")
            resolved_device = requested_device
            resolved_dtype = "fp32" if requested_device == "cpu" else requested_dtype
            result["checks"].append(
                {
                    "name": "torch/device",
                    "status": "OK",
                    "message": f"torch is importable. requested_device={requested_device}; requested_dtype={requested_dtype}; cpu実行時はfp32にfallbackします。",
                }
            )
            result["device"] = resolved_device
            result["dtype"] = resolved_dtype
        elif uses_worker_python:
            result["checks"].append(
                {
                    "name": "torch import",
                    "status": "INFO",
                    "message": "アプリ側Pythonでは未検出です。指定されたworker Pythonで確認します。",
                }
            )
        else:
            result["status"] = "WARNING"
            result["checks"].append({"name": "torch import", "status": "WARNING", "message": "torch is not installed."})
        if not settings_row.get("allow_model_download"):
            result["checks"].append(
                {
                    "name": "model download",
                    "status": "WARNING" if transformers_ok and not uses_worker_python else "INFO",
                    "message": "downloadは無効です。openai/clip-vit-base-patch32 がローカルcacheに無い場合、実行時に失敗します。",
                }
            )
            if transformers_ok and not uses_worker_python:
                result["status"] = "WARNING"
        else:
            result["checks"].append({"name": "model download", "status": "OK", "message": "明示的にdownloadが許可されています。"})
        result["vector_dim"] = int(model.get("vector_dim") or 512)
        if deep:
            try:
                completed = subprocess.run(
                    [python_path, "-m", "app.services.embedding_worker", "--preflight-model-id", str(model.get("id"))],
                    cwd=str(settings.ROOT_DIR),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=900,
                    check=False,
                )
                stdout = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "{}"
                worker_result = json.loads(stdout)
                result["worker_return_code"] = completed.returncode
                result["checks"].extend(worker_result.get("checks", []))
                for key in ("vector_dim", "norm", "device", "dtype"):
                    if key in worker_result:
                        result[key] = worker_result[key]
                if completed.returncode != 0 or worker_result.get("status") == "ERROR":
                    result["status"] = "ERROR"
                    if completed.stderr.strip():
                        result["checks"].append({"name": "worker stderr", "status": "ERROR", "message": completed.stderr.strip()[-1000:]})
                elif worker_result.get("status") == "OK":
                    result["status"] = "OK"
                elif worker_result.get("status") == "WARNING" and result["status"] != "ERROR":
                    result["status"] = "WARNING"
            except Exception as exc:
                result["status"] = "ERROR"
                result["checks"].append({"name": "worker preflight", "status": "ERROR", "message": str(exc)})
    elif provider == "open_clip":
        open_clip_ok = importlib.util.find_spec("open_clip") is not None
        torch_ok = importlib.util.find_spec("torch") is not None
        if open_clip_ok:
            result["checks"].append({"name": "open_clip import", "status": "OK", "message": "open_clip is importable."})
        elif uses_worker_python:
            result["checks"].append(
                {
                    "name": "open_clip import",
                    "status": "INFO",
                    "message": "アプリ側Pythonでは未検出です。指定されたworker Pythonで確認します。",
                }
            )
        else:
            result["status"] = "WARNING"
            result["checks"].append({"name": "open_clip import", "status": "WARNING", "message": "open_clip_torch is not installed."})
        if torch_ok:
            requested_device = str(settings_row.get("device") or "auto")
            requested_dtype = str(settings_row.get("dtype") or "fp16")
            resolved_dtype = "fp32" if requested_device == "cpu" else requested_dtype
            result["checks"].append(
                {
                    "name": "torch/device",
                    "status": "OK",
                    "message": f"torch is importable. requested_device={requested_device}; requested_dtype={requested_dtype}; cpu実行時はfp32にfallbackします。",
                }
            )
            result["device"] = requested_device
            result["dtype"] = resolved_dtype
        elif uses_worker_python:
            result["checks"].append(
                {
                    "name": "torch import",
                    "status": "INFO",
                    "message": "アプリ側Pythonでは未検出です。指定されたworker Pythonで確認します。",
                }
            )
        else:
            result["status"] = "WARNING"
            result["checks"].append({"name": "torch import", "status": "WARNING", "message": "torch is not installed."})
        if not settings_row.get("allow_model_download"):
            result["checks"].append(
                {
                    "name": "model download",
                    "status": "WARNING" if open_clip_ok and not uses_worker_python else "INFO",
                    "message": "downloadは無効です。ViT-B-32 / laion2b_s34b_b79k がローカルcacheに無い場合、実行時に失敗します。",
                }
            )
            if open_clip_ok and not uses_worker_python:
                result["status"] = "WARNING"
        else:
            result["checks"].append({"name": "model download", "status": "OK", "message": "明示的にdownloadが許可されています。"})
        result["vector_dim"] = int(model.get("vector_dim") or 512)
        if deep:
            try:
                completed = subprocess.run(
                    [python_path, "-m", "app.services.embedding_worker", "--preflight-model-id", str(model.get("id"))],
                    cwd=str(settings.ROOT_DIR),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=900,
                    check=False,
                )
                stdout = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "{}"
                worker_result = json.loads(stdout)
                result["worker_return_code"] = completed.returncode
                result["checks"].extend(worker_result.get("checks", []))
                for key in ("vector_dim", "norm", "device", "dtype"):
                    if key in worker_result:
                        result[key] = worker_result[key]
                if completed.returncode != 0 or worker_result.get("status") == "ERROR":
                    result["status"] = "ERROR"
                    if completed.stderr.strip():
                        result["checks"].append({"name": "worker stderr", "status": "ERROR", "message": completed.stderr.strip()[-1000:]})
                elif worker_result.get("status") == "OK":
                    result["status"] = "OK"
                elif worker_result.get("status") == "WARNING" and result["status"] != "ERROR":
                    result["status"] = "WARNING"
            except Exception as exc:
                result["status"] = "ERROR"
                result["checks"].append({"name": "worker preflight", "status": "ERROR", "message": str(exc)})
    else:
        result["status"] = "ERROR"
        result["checks"].append({"name": "provider", "status": "ERROR", "message": f"Unknown provider: {provider}"})

    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE embedding_settings
            SET last_preflight_status = ?, last_preflight_message = ?, last_preflight_json = ?, updated_at = ?
            WHERE id = (SELECT id FROM embedding_settings ORDER BY id LIMIT 1)
            """,
            (
                result["status"],
                "; ".join(f"{c['name']}={c['status']}" for c in result["checks"]),
                json.dumps(result, ensure_ascii=False, indent=2),
                now,
            ),
        )
    return result


def provider_requires_gpu_exclusivity(provider: str | None) -> bool:
    return provider in {"transformers_clip", "open_clip"}


def assert_embedding_can_start(model: dict[str, Any]) -> None:
    provider = model.get("provider") or "mock"
    if not provider_requires_gpu_exclusivity(provider):
        return
    running_training = fetch_one("SELECT id FROM training_jobs WHERE status = 'running' LIMIT 1")
    running_generation = fetch_one("SELECT id FROM validation_generation_runs WHERE status = 'running' LIMIT 1")
    if running_training:
        raise RuntimeError(f"学習ジョブ #{running_training['id']} が実行中です。GPUを使うEmbedding providerは学習完了後に開始してください。")
    if running_generation:
        raise RuntimeError(f"検証画像生成 #{running_generation['id']} が実行中です。GPUを使うEmbedding providerは生成完了後に開始してください。")


def dataset_image_sources(dataset_version_id: int) -> list[EmbeddingSource]:
    version = fetch_one("SELECT * FROM dataset_versions WHERE id = ?", (dataset_version_id,))
    if version is None:
        return []
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (version["dataset_id"],))
    if dataset is None:
        return []
    root = Path(dataset["path"])
    sources: list[EmbeddingSource] = []
    if not root.exists():
        return sources
    for index, path in enumerate(sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), start=1):
        sources.append(
            EmbeddingSource(
                source_type="dataset_image",
                source_id=index,
                source_path=str(path),
                dataset_id=dataset["id"],
                dataset_version_id=dataset_version_id,
            )
        )
    return sources


def reference_image_sources(reference_set_version_id: int) -> list[EmbeddingSource]:
    rows = fetch_all(
        """
        SELECT ri.*, rs.project_id
        FROM reference_images ri
        LEFT JOIN reference_sets rs ON rs.id = ri.reference_set_id
        WHERE ri.reference_set_version_id = ?
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


def sample_image_sources(job_id: int) -> list[EmbeddingSource]:
    job = fetch_one("SELECT project_id, dataset_id, dataset_version_id FROM training_jobs WHERE id = ?", (job_id,))
    rows = fetch_all("SELECT * FROM sample_images WHERE job_id = ? AND deleted_at IS NULL ORDER BY id", (job_id,))
    return [
        EmbeddingSource(
            source_type="sample_image",
            source_id=row["id"],
            source_path=row["image_path"],
            project_id=job["project_id"] if job else None,
            dataset_id=job["dataset_id"] if job else None,
            dataset_version_id=job["dataset_version_id"] if job else None,
            job_id=job_id,
        )
        for row in rows
    ]


def validation_image_sources(validation_run_id: int) -> list[EmbeddingSource]:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (validation_run_id,))
    rows = fetch_all("SELECT * FROM validation_images WHERE validation_run_id = ? AND ignored = 0 ORDER BY id", (validation_run_id,))
    return [
        EmbeddingSource(
            source_type="validation_image",
            source_id=row["id"],
            source_path=row["image_path"],
            project_id=run["project_id"] if run else None,
            job_id=row["job_id"],
            validation_run_id=validation_run_id,
            reference_set_id=run["reference_set_id"] if run and "reference_set_id" in run.keys() else None,
            reference_set_version_id=run["reference_set_version_id"] if run and "reference_set_version_id" in run.keys() else None,
        )
        for row in rows
    ]


def review_session_image_sources(review_session_id: int) -> list[EmbeddingSource]:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ?", (review_session_id,))
    rows = fetch_all("SELECT * FROM review_session_images WHERE review_session_id = ? AND deleted_at IS NULL ORDER BY id", (review_session_id,))
    return [
        EmbeddingSource(
            source_type="review_session_image",
            source_id=row["id"],
            source_path=row["image_path"],
            project_id=session["project_id"] if session else None,
            dataset_id=session["dataset_id"] if session else None,
            dataset_version_id=session["dataset_version_id"] if session else None,
            reference_set_id=session["reference_set_id"] if session else None,
            reference_set_version_id=session["reference_set_version_id"] if session else None,
            job_id=row["job_id"],
        )
        for row in rows
    ]


def sources_for_job_type(job_type: str, target_id: int) -> list[EmbeddingSource]:
    if job_type == "dataset_version":
        return dataset_image_sources(target_id)
    if job_type == "reference_set_version":
        return reference_image_sources(target_id)
    if job_type == "training_job_samples":
        return sample_image_sources(target_id)
    if job_type == "validation_run":
        return validation_image_sources(target_id)
    if job_type == "review_session":
        return review_session_image_sources(target_id)
    return []


def latest_embedding_for(source: EmbeddingSource, model_id: str) -> dict[str, Any] | None:
    params: tuple[Any, ...]
    if source.source_id is not None:
        query = """
            SELECT * FROM image_embeddings
            WHERE source_type = ? AND source_id = ? AND embedding_model_id = ?
              AND COALESCE(dataset_version_id, 0) = COALESCE(?, 0)
              AND COALESCE(source_path, '') = COALESCE(?, '')
            ORDER BY updated_at DESC, id DESC LIMIT 1
        """
        params = (source.source_type, source.source_id, model_id, source.dataset_version_id, source.source_path)
    else:
        query = """
            SELECT * FROM image_embeddings
            WHERE source_type = ? AND source_path = ? AND embedding_model_id = ?
              AND COALESCE(dataset_version_id, 0) = COALESCE(?, 0)
            ORDER BY updated_at DESC, id DESC LIMIT 1
        """
        params = (source.source_type, source.source_path, model_id, source.dataset_version_id)
    row = fetch_one(query, params)
    return dict(row) if row else None


def embedding_status_for_source(source: EmbeddingSource, model_id: str) -> str:
    path = Path(source.source_path)
    embedding = latest_embedding_for(source, model_id)
    if not path.exists():
        return "missing_source"
    if embedding is None:
        return "not_computed"
    if embedding["status"] in {"failed", "missing_source"}:
        return embedding["status"]
    try:
        stat = path.stat()
    except OSError:
        return "missing_source"
    if embedding["image_file_size"] != stat.st_size or abs(float(embedding["image_mtime"] or 0) - stat.st_mtime) > 0.0001:
        return "stale"
    return "ready"


def embedding_coverage(job_type: str, target_id: int, model_id: str | None = None) -> dict[str, Any]:
    model = active_embedding_model() if model_id is None else dict(fetch_one("SELECT * FROM embedding_models WHERE id = ?", (model_id,)) or {})
    model_id = model.get("id") or "mock_image_512"
    sources = sources_for_job_type(job_type, target_id)
    counts = {"ready": 0, "stale": 0, "failed": 0, "missing_source": 0, "not_computed": 0}
    for source in sources:
        status = embedding_status_for_source(source, model_id)
        counts[status] = counts.get(status, 0) + 1
    total = len(sources)
    return {
        "job_type": job_type,
        "target_id": target_id,
        "model_id": model_id,
        "model_name": model.get("name") or model_id,
        "total": total,
        "ready": counts["ready"],
        "stale": counts["stale"],
        "failed": counts["failed"],
        "missing": counts["missing_source"],
        "not_computed": counts["not_computed"],
        "ready_rate": (counts["ready"] / total) if total else 0,
    }


def create_embedding_job(job_type: str, target_id: int, recompute: str = "missing") -> int:
    model = active_embedding_model()
    model_id = model.get("id") or "mock_image_512"
    provider = model.get("provider") or "mock"
    sources = sources_for_job_type(job_type, target_id)
    if recompute == "stale":
        sources = [s for s in sources if embedding_status_for_source(s, model_id) in {"stale", "failed", "missing_source"}]
    elif recompute == "missing":
        sources = [s for s in sources if embedding_status_for_source(s, model_id) in {"not_computed", "stale", "failed", "missing_source"}]

    now = utc_now()
    log_dir = settings.LOGS_DIR / "embeddings"
    log_dir.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO embedding_jobs(
                job_type, target_id, embedding_model_id, provider, status,
                total_count, log_path, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, 'planned', ?, ?, ?, ?)
            """,
            (job_type, target_id, model_id, provider, len(sources), "", now, now),
        )
        job_id = int(cur.lastrowid)
        log_path = log_dir / f"embedding_job_{job_id:06d}.log"
        conn.execute("UPDATE embedding_jobs SET log_path = ? WHERE id = ?", (str(log_path), job_id))
        for source in sources:
            conn.execute(
                """
                INSERT INTO embedding_job_items(
                    embedding_job_id, source_type, source_id, source_path,
                    status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'pending', ?, ?)
                """,
                (job_id, source.source_type, source.source_id, source.source_path, now, now),
            )
    return job_id


def start_embedding_job(job_id: int) -> None:
    reconcile_stale_embedding_jobs()
    running = fetch_one("SELECT id FROM embedding_jobs WHERE status = 'running' AND id != ? LIMIT 1", (job_id,))
    if running:
        raise RuntimeError(f"Embedding Job #{running['id']} が実行中です。")
    row = fetch_one("SELECT * FROM embedding_jobs WHERE id = ?", (job_id,))
    if row is None:
        raise RuntimeError("Embedding Jobが見つかりません。")
    if row["status"] not in {"planned", "failed", "stopped"}:
        raise RuntimeError(f"Embedding Job #{job_id} は開始できない状態です: {row['status']}")
    model = fetch_one("SELECT * FROM embedding_models WHERE id = ?", (row["embedding_model_id"],))
    assert_embedding_can_start(dict(model) if model else {"provider": row["provider"]})
    python_path = load_embedding_settings().get("python_path") or sys.executable
    argv = [python_path, "-m", "app.services.embedding_worker", "--embedding-job-id", str(job_id)]
    log_path = Path(row["log_path"] or settings.LOGS_DIR / "embeddings" / f"embedding_job_{job_id:06d}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        argv,
        cwd=str(settings.ROOT_DIR),
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
            "UPDATE embedding_jobs SET status = 'running', process_id = ?, started_at = ?, updated_at = ? WHERE id = ?",
            (proc.pid, now, now, job_id),
        )


def reconcile_stale_embedding_jobs() -> int:
    rows = fetch_all("SELECT * FROM embedding_jobs WHERE status = 'running' ORDER BY id")
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
        message = "Embedding process was not found. Marked stopped by stale reconciliation."
        with connect() as conn:
            conn.execute(
                """
                UPDATE embedding_jobs
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


def stop_embedding_job(job_id: int) -> None:
    row = fetch_one("SELECT * FROM embedding_jobs WHERE id = ?", (job_id,))
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
            "UPDATE embedding_jobs SET status = 'stopped', ended_at = ?, updated_at = ?, return_code = COALESCE(return_code, -1) WHERE id = ?",
            (now, now, job_id),
        )


def upsert_image_embedding(source: EmbeddingSource, model: dict[str, Any], metadata: dict[str, Any], embedding_path: Path, status: str, error: str = "") -> int:
    now = utc_now()
    with connect() as conn:
        existing = None
        if source.source_id is not None:
            existing = conn.execute(
                """
                SELECT id FROM image_embeddings
                WHERE source_type = ? AND source_id = ? AND embedding_model_id = ?
                  AND COALESCE(dataset_version_id, 0) = COALESCE(?, 0)
                  AND COALESCE(source_path, '') = COALESCE(?, '')
                ORDER BY id DESC LIMIT 1
                """,
                (source.source_type, source.source_id, model["id"], source.dataset_version_id, source.source_path),
            ).fetchone()
        if existing is None:
            existing = conn.execute(
                """
                SELECT id FROM image_embeddings
                WHERE source_type = ? AND source_path = ? AND embedding_model_id = ?
                  AND COALESCE(dataset_version_id, 0) = COALESCE(?, 0)
                ORDER BY id DESC LIMIT 1
                """,
                (source.source_type, source.source_path, model["id"], source.dataset_version_id),
            ).fetchone()
        values = {
            "source_type": source.source_type,
            "source_id": source.source_id,
            "source_path": source.source_path,
            "project_id": source.project_id,
            "dataset_id": source.dataset_id,
            "dataset_version_id": source.dataset_version_id,
            "reference_set_id": source.reference_set_id,
            "reference_set_version_id": source.reference_set_version_id,
            "job_id": source.job_id,
            "validation_run_id": source.validation_run_id,
            "embedding_model_id": model["id"],
            "provider": model.get("provider"),
            "model_name": model.get("model_name") or model.get("name"),
            "embedding_type": model.get("embedding_type") or "image",
            "vector_dim": metadata.get("vector_dim") or model.get("vector_dim"),
            "normalized": int(model.get("normalize") or 0),
            "embedding_path": str(embedding_path) if embedding_path else "",
            "image_sha256": metadata.get("sha256"),
            "image_file_size": metadata.get("file_size"),
            "image_mtime": metadata.get("mtime"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "status": status,
            "error_message": error,
            "updated_at": now,
        }
        if existing:
            values["id"] = existing["id"]
            conn.execute(
                """
                UPDATE image_embeddings SET
                    source_path = :source_path, project_id = :project_id,
                    dataset_id = :dataset_id, dataset_version_id = :dataset_version_id,
                    reference_set_id = :reference_set_id,
                    reference_set_version_id = :reference_set_version_id,
                    job_id = :job_id, validation_run_id = :validation_run_id,
                    provider = :provider, model_name = :model_name,
                    embedding_type = :embedding_type, vector_dim = :vector_dim,
                    normalized = :normalized, embedding_path = :embedding_path,
                    image_sha256 = :image_sha256, image_file_size = :image_file_size,
                    image_mtime = :image_mtime, width = :width, height = :height,
                    status = :status, error_message = :error_message,
                    updated_at = :updated_at
                WHERE id = :id
                """,
                values,
            )
            return int(existing["id"])
        values["created_at"] = now
        cur = conn.execute(
            """
            INSERT INTO image_embeddings(
                source_type, source_id, source_path, project_id, dataset_id,
                dataset_version_id, reference_set_id, reference_set_version_id,
                job_id, validation_run_id, embedding_model_id, provider, model_name,
                embedding_type, vector_dim, normalized, embedding_path,
                image_sha256, image_file_size, image_mtime, width, height,
                status, error_message, created_at, updated_at
            )
            VALUES (
                :source_type, :source_id, :source_path, :project_id, :dataset_id,
                :dataset_version_id, :reference_set_id, :reference_set_version_id,
                :job_id, :validation_run_id, :embedding_model_id, :provider,
                :model_name, :embedding_type, :vector_dim, :normalized,
                :embedding_path, :image_sha256, :image_file_size, :image_mtime,
                :width, :height, :status, :error_message, :created_at, :updated_at
            )
            """,
            values,
        )
        return int(cur.lastrowid)


def embedding_cache_path(model_id: str, source_type: str, source_id: int | None, source_path: str) -> Path:
    if source_id is not None:
        path_hash = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:10]
        safe_id = f"{source_id:06d}_{path_hash}"
    else:
        safe_id = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]
    return settings.EMBEDDINGS_DIR / f"model_{model_id}" / source_type / f"{source_type}_{safe_id}_{model_id}.npy"


def embedding_cache_size() -> dict[str, Any]:
    root = settings.EMBEDDINGS_DIR
    totals: dict[str, int] = {"total": 0}
    by_model: dict[str, int] = {}
    by_source_type: dict[str, int] = {}
    if root.exists():
        for path in root.rglob("*.npy"):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            totals["total"] += size
            model_key = path.parts[-3] if len(path.parts) >= 3 else "unknown"
            source_key = path.parts[-2] if len(path.parts) >= 2 else "unknown"
            by_model[model_key] = by_model.get(model_key, 0) + size
            by_source_type[source_key] = by_source_type.get(source_key, 0) + size
    return {"total": totals["total"], "by_model": by_model, "by_source_type": by_source_type}


def format_bytes(size: int | None) -> str:
    value = float(size or 0)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def cleanup_preview(status: str) -> dict[str, Any]:
    if status == "stale":
        candidates = fetch_all("SELECT * FROM image_embeddings WHERE status IN ('ready', 'stale')")
        rows = []
        for row in candidates:
            path = Path(row["source_path"] or "")
            if not path.exists():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if row["status"] == "stale" or row["image_file_size"] != stat.st_size or abs(float(row["image_mtime"] or 0) - stat.st_mtime) > 0.0001:
                rows.append(row)
    else:
        rows = fetch_all("SELECT * FROM image_embeddings WHERE status = ?", (status,))
    total = 0
    for row in rows:
        path = Path(row["embedding_path"] or "")
        if path.exists():
            total += path.stat().st_size
    return {"status": status, "count": len(rows), "bytes": total, "label": format_bytes(total), "rows": rows}


def latest_embedding_jobs(limit: int = 20) -> list[dict[str, Any]]:
    rows = fetch_all("SELECT * FROM embedding_jobs ORDER BY id DESC LIMIT ?", (limit,))
    return [dict(row) for row in rows]


def read_embedding_log(job_id: int, lines: int = 40) -> str:
    row = fetch_one("SELECT log_path FROM embedding_jobs WHERE id = ?", (job_id,))
    if not row or not row["log_path"]:
        return ""
    path = Path(row["log_path"])
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])
