from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app import settings

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def normalize_user_path(value: str | Path) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def verify_image_file(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError("Unsupported image extension. Use png, jpg, jpeg, or webp.")
    if not path.exists() or not path.is_file():
        raise ValueError("Image file not found")
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Image file could not be verified") from exc


def unique_copy(source: Path, target_dir: Path) -> Path:
    source = Path(normalize_user_path(source))
    verify_image_file(source)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    counter = 1
    while target.exists():
        target = target_dir / f"{source.stem}_{counter}{source.suffix}"
        counter += 1
    shutil.copy2(source, target)
    verify_image_file(target)
    return target


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validation_images_root() -> Path:
    return settings.EXPORTS_DIR / "validation_runs"


def reference_images_root() -> Path:
    return settings.EXPORTS_DIR / "reference_sets"


def ensure_allowed_file(path_text: str, root: Path, label: str) -> Path:
    path = Path(normalize_user_path(path_text)).resolve()
    if not is_relative_to(path, root):
        raise PermissionError(f"{label} path is outside the allowed managed directory")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} file not found")
    return path


def managed_reference_image_dir(reference_set_id: int) -> Path:
    return reference_images_root() / f"reference_set_{reference_set_id:06d}" / "images"


def copy_managed_reference_image(reference_set_id: int, source_path: str) -> str:
    target = unique_copy(Path(normalize_user_path(source_path)), managed_reference_image_dir(reference_set_id))
    return str(target)
