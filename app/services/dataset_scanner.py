from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CAPTION_EXTENSION = ".txt"
GENERIC_TRIGGER_TAGS = {
    "1girl",
    "1boy",
    "solo",
    "looking at viewer",
    "smile",
    "upper body",
    "full body",
    "simple background",
    "outdoors",
    "anime style",
}


def scan_dataset(path: Path, trigger_word: str = "") -> dict[str, Any]:
    if not path.exists() or not path.is_dir():
        return empty_scan("missing")

    images = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    captions = sorted(p for p in path.rglob(f"*{CAPTION_EXTENSION}") if p.is_file())
    supported_files = set(images) | set(captions)
    unsupported_files = sorted(p for p in path.rglob("*") if p.is_file() and p not in supported_files)

    caption_count = 0
    missing_caption_count = 0
    empty_caption_count = 0
    unreadable_caption_count = 0
    trigger_count = 0
    tag_counter: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    broken_images: list[str] = []
    missing_caption_images: list[str] = []
    caption_without_images: list[str] = []
    caption_status: list[dict[str, str]] = []
    trigger = trigger_word.strip()

    try:
        from PIL import Image
    except Exception:
        Image = None

    image_stems = {image_path.with_suffix("") for image_path in images}
    for caption_path in captions:
        if caption_path.with_suffix("") not in image_stems:
            caption_without_images.append(str(caption_path))

    for image_path in images:
        caption_path = image_path.with_suffix(CAPTION_EXTENSION)
        if caption_path.exists():
            caption_count += 1
            try:
                text = caption_path.read_text(encoding="utf-8")
                caption_status.append({"path": str(caption_path), "status": "utf-8"})
            except UnicodeDecodeError:
                text = caption_path.read_text(encoding="utf-8", errors="replace")
                unreadable_caption_count += 1
                caption_status.append({"path": str(caption_path), "status": "warning: decode errors replaced"})
            except OSError as exc:
                text = ""
                unreadable_caption_count += 1
                caption_status.append({"path": str(caption_path), "status": f"warning: {exc}"})
            if not text.strip():
                empty_caption_count += 1
            if trigger and trigger in text:
                trigger_count += 1
            for tag in [part.strip() for part in text.replace("\n", ",").split(",")]:
                if tag:
                    tag_counter[tag] += 1
        else:
            missing_caption_count += 1
            missing_caption_images.append(str(image_path))

        if Image is not None:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    img.verify()
                    resolutions[f"{width}x{height}"] += 1
                    widths.append(int(width))
                    heights.append(int(height))
            except Exception:
                resolutions["unreadable"] += 1
                broken_images.append(str(image_path))

    return {
        "status": "ok" if images else "empty",
        "image_count": len(images),
        "caption_count": caption_count,
        "missing_caption_count": missing_caption_count,
        "supported_image_count": len(images),
        "unsupported_file_count": len(unsupported_files),
        "broken_image_count": len(broken_images),
        "empty_caption_count": empty_caption_count,
        "unreadable_caption_count": unreadable_caption_count,
        "caption_encoding_summary": caption_encoding_summary(caption_status),
        "trigger_word": trigger,
        "trigger_word_count": trigger_count,
        "trigger_word_rate": trigger_count / caption_count if caption_count else None,
        "trigger_consistency": trigger_consistency(trigger, caption_count, trigger_count),
        "trigger_candidates": trigger_candidates(tag_counter),
        "resolution_summary": dict(resolutions.most_common(20)),
        "image_size_summary": image_size_summary(widths, heights, resolutions),
        "tag_summary": dict(tag_counter.most_common(50)),
        "missing_caption_images": missing_caption_images,
        "caption_without_images": caption_without_images,
        "broken_images": broken_images,
        "unsupported_files": [str(item) for item in unsupported_files[:200]],
    }


def empty_scan(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "image_count": 0,
        "caption_count": 0,
        "missing_caption_count": 0,
        "supported_image_count": 0,
        "unsupported_file_count": 0,
        "broken_image_count": 0,
        "empty_caption_count": 0,
        "unreadable_caption_count": 0,
        "caption_encoding_summary": {},
        "trigger_word": "",
        "trigger_word_count": 0,
        "trigger_word_rate": None,
        "trigger_consistency": trigger_consistency("", 0, 0),
        "trigger_candidates": [],
        "resolution_summary": {},
        "image_size_summary": {},
        "tag_summary": {},
        "missing_caption_images": [],
        "caption_without_images": [],
        "broken_images": [],
        "unsupported_files": [],
    }


def image_size_summary(widths: list[int], heights: list[int], resolutions: Counter[str]) -> dict[str, Any]:
    if not widths or not heights:
        return {
            "min_width": None,
            "min_height": None,
            "max_width": None,
            "max_height": None,
            "average_width": None,
            "average_height": None,
            "resolutions": dict(resolutions.most_common(20)),
        }
    return {
        "min_width": min(widths),
        "min_height": min(heights),
        "max_width": max(widths),
        "max_height": max(heights),
        "average_width": round(mean(widths), 2),
        "average_height": round(mean(heights), 2),
        "resolutions": dict(resolutions.most_common(20)),
    }


def caption_encoding_summary(rows: list[dict[str, str]]) -> dict[str, int]:
    counter: Counter[str] = Counter(row["status"] for row in rows)
    return dict(counter)


def trigger_consistency(trigger_word: str, caption_count: int, trigger_count: int) -> dict[str, Any]:
    if not trigger_word or caption_count <= 0:
        return {
            "label": "UNKNOWN",
            "message": "trigger_word is not set or captions are unavailable.",
        }
    rate = trigger_count / caption_count
    if rate >= 0.8:
        return {
            "label": "OK",
            "message": f"trigger_word appears in {trigger_count}/{caption_count} captions.",
        }
    if trigger_count > 0:
        return {
            "label": "WARNING",
            "message": f"trigger_word appears in only {trigger_count}/{caption_count} captions.",
        }
    return {
        "label": "ERROR",
        "message": f"trigger_word does not appear in any captions ({trigger_count}/{caption_count}).",
    }


def trigger_candidates(tag_counter: Counter[str]) -> list[dict[str, Any]]:
    candidates = []
    for tag, count in tag_counter.most_common(50):
        if tag.lower() in GENERIC_TRIGGER_TAGS:
            continue
        candidates.append({"tag": tag, "count": count})
        if len(candidates) >= 10:
            break
    return candidates
