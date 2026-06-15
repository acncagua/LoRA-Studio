from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageOps

from app.db import connect, fetch_all, fetch_one, utc_now
from app.services.embedding_service import (
    EmbeddingSource,
    embedding_cache_path,
    image_metadata,
    sources_for_job_type,
    upsert_image_embedding,
)


def mock_embedding(image_path: Path, vector_dim: int = 512, normalize: bool = True) -> np.ndarray:
    digest = hashlib.sha256(image_path.read_bytes()).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(vector_dim).astype(np.float32)
    if normalize:
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
    return vector


def load_image_for_validation(path: Path, max_image_size: int) -> None:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if max_image_size and max(image.size) > max_image_size:
            image.thumbnail((max_image_size, max_image_size))


def source_from_item(item: Any, source_map: dict[tuple[str, int | None, str], EmbeddingSource]) -> EmbeddingSource:
    key = (item["source_type"], item["source_id"], item["source_path"])
    source = source_map.get(key)
    if source:
        return source
    return EmbeddingSource(item["source_type"], item["source_id"], item["source_path"])


def update_job_counts(conn: Any, job_id: int) -> None:
    counts = {
        row["status"]: row["count"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS count FROM embedding_job_items WHERE embedding_job_id = ? GROUP BY status",
            (job_id,),
        ).fetchall()
    }
    processed = sum(counts.get(status, 0) for status in ("ready", "failed", "skipped", "missing_source"))
    conn.execute(
        """
        UPDATE embedding_jobs
        SET processed_count = ?, ready_count = ?, failed_count = ?,
            skipped_count = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            processed,
            counts.get("ready", 0),
            counts.get("failed", 0) + counts.get("missing_source", 0),
            counts.get("skipped", 0),
            utc_now(),
            job_id,
        ),
    )


def run_embedding_job(job_id: int) -> int:
    job = fetch_one("SELECT * FROM embedding_jobs WHERE id = ?", (job_id,))
    if job is None:
        print(f"Embedding job #{job_id} not found", flush=True)
        return 2
    model = fetch_one("SELECT * FROM embedding_models WHERE id = ?", (job["embedding_model_id"],))
    settings = fetch_one("SELECT * FROM embedding_settings ORDER BY id LIMIT 1")
    if model is None or settings is None:
        print("Embedding model or settings not found", flush=True)
        return 2

    started = time.time()
    now = utc_now()
    with connect() as conn:
        conn.execute("UPDATE embedding_jobs SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ? WHERE id = ?", (now, now, job_id))

    provider = model["provider"]
    vector_dim = int(model["vector_dim"] or 512)
    normalize = bool(model["normalize"])
    max_image_size = int(settings["max_image_size"] or 1024)
    source_map = {
        (source.source_type, source.source_id, source.source_path): source
        for source in sources_for_job_type(job["job_type"], job["target_id"])
    }
    items = fetch_all("SELECT * FROM embedding_job_items WHERE embedding_job_id = ? ORDER BY id", (job_id,))
    print(f"Embedding job #{job_id} started: provider={provider}, items={len(items)}", flush=True)

    for index, item in enumerate(items, start=1):
        item_id = int(item["id"])
        path = Path(item["source_path"])
        status = "ready"
        error = ""
        embedding_id = None
        try:
            if not path.exists() or not path.is_file():
                raise FileNotFoundError("source image is missing")
            load_image_for_validation(path, max_image_size)
            metadata = image_metadata(path, max_image_size=max_image_size)
            source = source_from_item(item, source_map)
            if provider != "mock":
                raise RuntimeError(f"provider {provider} is configured but not implemented for execution in Phase 11.2")
            vector = mock_embedding(path, vector_dim=vector_dim, normalize=normalize)
            metadata["vector_dim"] = int(vector.shape[0])
            target = embedding_cache_path(model["id"], item["source_type"], item["source_id"], item["source_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, vector)
            embedding_id = upsert_image_embedding(source, dict(model), metadata, target, "ready")
            print(f"[{index}/{len(items)}] ready {item['source_type']}#{item['source_id'] or '-'} {path.name}", flush=True)
        except FileNotFoundError as exc:
            status = "missing_source"
            error = str(exc)
            metadata = {}
            source = source_from_item(item, source_map)
            embedding_id = upsert_image_embedding(source, dict(model), metadata, Path(""), "missing_source", error)
            print(f"[{index}/{len(items)}] missing {path}: {error}", flush=True)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            try:
                source = source_from_item(item, source_map)
                embedding_id = upsert_image_embedding(source, dict(model), {}, Path(""), "failed", error)
            except Exception:
                embedding_id = None
            print(f"[{index}/{len(items)}] failed {path}: {error}", flush=True)

        with connect() as conn:
            conn.execute(
                """
                UPDATE embedding_job_items
                SET status = ?, embedding_id = ?, error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, embedding_id, error, utc_now(), item_id),
            )
            update_job_counts(conn, job_id)

    finished = utc_now()
    elapsed = int(time.time() - started)
    with connect() as conn:
        counts = conn.execute("SELECT failed_count FROM embedding_jobs WHERE id = ?", (job_id,)).fetchone()
        failed = int(counts["failed_count"] or 0) if counts else 0
        final_status = "failed" if failed and failed == len(items) else "completed"
        conn.execute(
            """
            UPDATE embedding_jobs
            SET status = ?, ended_at = ?, elapsed_seconds = ?, return_code = ?,
                updated_at = ?, error_message = ?
            WHERE id = ?
            """,
            (final_status, finished, elapsed, 0 if final_status == "completed" else 1, finished, "" if final_status == "completed" else "all items failed", job_id),
        )
    print(f"Embedding job #{job_id} finished: status={final_status}, elapsed={elapsed}s", flush=True)
    return 0 if final_status == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-job-id", type=int, required=True)
    args = parser.parse_args()
    return run_embedding_job(args.embedding_job_id)


if __name__ == "__main__":
    raise SystemExit(main())

