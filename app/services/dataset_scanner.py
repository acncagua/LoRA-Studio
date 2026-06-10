from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def scan_dataset(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_dir():
        return {"status": "missing", "image_count": 0, "caption_count": 0, "missing_caption_count": 0, "resolution_summary": {}, "tag_summary": {}}

    images = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    caption_count = 0
    missing_caption_count = 0
    tag_counter: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()

    try:
        from PIL import Image
    except Exception:
        Image = None

    for image_path in images:
        caption_path = image_path.with_suffix(".txt")
        if caption_path.exists():
            caption_count += 1
            try:
                text = caption_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = caption_path.read_text(encoding="utf-8", errors="replace")
            for tag in [part.strip() for part in text.replace("\n", ",").split(",")]:
                if tag:
                    tag_counter[tag] += 1
        else:
            missing_caption_count += 1

        if Image is not None:
            try:
                with Image.open(image_path) as img:
                    resolutions[f"{img.width}x{img.height}"] += 1
            except Exception:
                resolutions["unreadable"] += 1

    return {
        "status": "ok" if images else "empty",
        "image_count": len(images),
        "caption_count": caption_count,
        "missing_caption_count": missing_caption_count,
        "resolution_summary": dict(resolutions.most_common(20)),
        "tag_summary": dict(tag_counter.most_common(50)),
    }
