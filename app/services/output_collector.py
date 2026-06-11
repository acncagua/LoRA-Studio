from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from app.db import connect, fetch_all, fetch_one, utc_now

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_epoch_step(path: Path) -> tuple[int | None, int | None]:
    name = path.stem.lower()
    epoch = parse_number(name, r"epoch[-_]?(\d+)")
    step = parse_number(name, r"step[-_]?(\d+)")
    if epoch is None:
        epoch = parse_number(name, r"e[-_]?(\d+)")
    if step is None:
        step = parse_number(name, r"s[-_]?(\d+)")
    return epoch, step


def parse_prompt_index(path: Path) -> int | None:
    name = path.stem.lower()
    prompt_index = parse_number(name, r"prompt[-_]?(\d+)")
    if prompt_index is None:
        prompt_index = parse_number(name, r"p[-_]?(\d+)")
    return prompt_index


def parse_number(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.width, image.height
    except Exception:
        return None, None


def collect_job_results(job_id: int) -> dict[str, int]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")

    run_dir = Path(job["run_dir"])
    model_count = collect_models(job_id, run_dir / "models")
    sample_count = collect_samples(job_id, run_dir / "samples")
    return {"models": model_count, "samples": sample_count}


def collect_models(job_id: int, models_dir: Path) -> int:
    if not models_dir.exists():
        return 0
    now = utc_now()
    inserted = 0
    with connect() as conn:
        for path in sorted(models_dir.rglob("*.safetensors")):
            file_path = str(path)
            existing = conn.execute(
                "SELECT id FROM training_outputs WHERE job_id = ? AND file_path = ?",
                (job_id, file_path),
            ).fetchone()
            if existing:
                continue
            epoch, step = parse_epoch_step(path)
            conn.execute(
                """
                INSERT INTO training_outputs(
                    job_id, epoch, step, file_path, file_type, file_size,
                    sha256, selected, created_at
                )
                VALUES (?, ?, ?, ?, 'model', ?, ?, 0, ?)
                """,
                (job_id, epoch, step, file_path, path.stat().st_size, sha256_file(path), now),
            )
            inserted += 1
    return inserted


def collect_samples(job_id: int, samples_dir: Path) -> int:
    if not samples_dir.exists():
        return 0
    prompt_rows = fetch_all(
        "SELECT * FROM sample_prompts WHERE job_id = ? ORDER BY sort_order, id",
        (job_id,),
    )
    prompts_by_order = {int(row["sort_order"]): row for row in prompt_rows}
    now = utc_now()
    inserted = 0
    with connect() as conn:
        for path in sorted(p for p in samples_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
            file_path = str(path)
            existing = conn.execute(
                "SELECT id FROM sample_images WHERE job_id = ? AND image_path = ?",
                (job_id, file_path),
            ).fetchone()
            if existing:
                continue
            epoch, step = parse_epoch_step(path)
            prompt_index = parse_prompt_index(path)
            prompt = prompts_by_order.get(prompt_index or 0)
            width, height = image_size(path)
            conn.execute(
                """
                INSERT INTO sample_images(
                    job_id, prompt_id, epoch, step, image_path, prompt,
                    negative_prompt, seed, width, height, cfg_scale, steps, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    prompt["id"] if prompt else None,
                    epoch,
                    step,
                    file_path,
                    prompt["prompt"] if prompt else None,
                    prompt["negative_prompt"] if prompt else None,
                    prompt["seed"] if prompt else None,
                    width or (prompt["width"] if prompt else None),
                    height or (prompt["height"] if prompt else None),
                    prompt["cfg_scale"] if prompt else None,
                    prompt["steps"] if prompt else None,
                    now,
                ),
            )
            inserted += 1
    return inserted
