from __future__ import annotations

import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from app import settings
from app.db import connect, fetch_all, fetch_one, utc_now
from app.services.dataset_scanner import IMAGE_EXTENSIONS
from app.services.image_store import normalize_user_path, reference_images_root, unique_copy

REFERENCE_TYPES = {"character", "style", "mixed", "other"}
ROLE_LABELS = {
    "face_front": "顔正面",
    "face_angle": "顔角度違い",
    "expression": "表情",
    "upper_body": "上半身",
    "full_body": "全身",
    "close_up": "寄り",
    "background_scene": "背景・場面",
    "color_palette": "色味",
    "linework": "線",
    "rendering": "塗り",
    "lighting": "光",
    "style": "style",
    "other": "その他",
}
REFERENCE_TYPE_LABELS = {
    "character": "キャラクター",
    "style": "スタイル",
    "mixed": "混合",
    "other": "その他",
}
CHARACTER_REQUIRED = {"face_front", "upper_body", "full_body"}
CHARACTER_RECOMMENDED = CHARACTER_REQUIRED | {"face_angle", "expression"}
STYLE_RECOMMENDED = {"close_up", "upper_body", "full_body", "background_scene", "color_palette", "linework", "rendering", "lighting"}


def valid_role(role: str) -> str:
    role = (role or "other").strip()
    aliases = {
        "face": "face_front",
    }
    role = aliases.get(role, role)
    return role if role in ROLE_LABELS else "other"


def valid_reference_type(value: str) -> str:
    value = (value or "character").strip()
    return value if value in REFERENCE_TYPES else "other"


def latest_dataset_version_id(dataset_id: int | None) -> int | None:
    if not dataset_id:
        return None
    row = fetch_one("SELECT id FROM dataset_versions WHERE dataset_id = ? ORDER BY version_no DESC LIMIT 1", (dataset_id,))
    return int(row["id"]) if row else None


def create_reference_set(
    *,
    name: str,
    reference_type: str,
    dataset_id: int | None = None,
    dataset_version_id: int | None = None,
    project_id: int | None = None,
    trigger_word: str = "",
    description: str = "",
    selection_mode: str = "manual",
    memo: str = "",
) -> int:
    now = utc_now()
    reference_type = valid_reference_type(reference_type)
    if dataset_id and dataset_version_id is None:
        dataset_version_id = latest_dataset_version_id(dataset_id)
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO reference_sets(
                name, project_id, dataset_id, dataset_version_id, current_dataset_version_id,
                reference_type, selection_mode, trigger_word, description,
                is_default, is_archived, created_at, updated_at, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
            """,
            (
                name.strip(),
                project_id,
                dataset_id,
                dataset_version_id,
                dataset_version_id,
                reference_type,
                selection_mode.strip() or "manual",
                trigger_word.strip(),
                description.strip(),
                now,
                now,
                memo.strip(),
            ),
        )
        set_id = int(cur.lastrowid)
        version_id = create_reference_version(conn, set_id, memo="初期バージョン")
        conn.execute("UPDATE reference_sets SET current_version_id = ? WHERE id = ?", (version_id, set_id))
        if project_id:
            conn.execute(
                "UPDATE lora_projects SET default_reference_set_id = COALESCE(default_reference_set_id, ?), default_reference_set_version_id = COALESCE(default_reference_set_version_id, ?), updated_at = ? WHERE id = ?",
                (set_id, version_id, now, project_id),
            )
    return set_id


def create_reference_version(conn: Any, reference_set_id: int, memo: str = "") -> int:
    row = conn.execute("SELECT * FROM reference_sets WHERE id = ?", (reference_set_id,)).fetchone()
    if row is None:
        raise ValueError("Reference Setが見つかりません。")
    version_no = int(conn.execute("SELECT COALESCE(MAX(version_no), 0) + 1 AS next_no FROM reference_set_versions WHERE reference_set_id = ?", (reference_set_id,)).fetchone()["next_no"])
    now = utc_now()
    cur = conn.execute(
        """
        INSERT INTO reference_set_versions(
            reference_set_id, version_no, dataset_id, dataset_version_id, trigger_word,
            reference_type, image_count, roles_json, completeness_label,
            completeness_message, locked_at, memo, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, 0, '{}', 'ERROR', 'リファレンス画像がありません。', ?, ?, ?, ?)
        """,
        (
            reference_set_id,
            version_no,
            row["dataset_id"],
            row["current_dataset_version_id"] or row["dataset_version_id"],
            row["trigger_word"] or "",
            valid_reference_type(row["reference_type"]),
            now,
            memo,
            now,
            now,
        ),
    )
    return int(cur.lastrowid)


def reference_image_dir(reference_set_id: int, version_id: int | None) -> Path:
    if version_id:
        return reference_images_root() / f"reference_set_{reference_set_id:06d}" / f"version_{version_id:06d}" / "images"
    return reference_images_root() / f"reference_set_{reference_set_id:06d}" / "images"


def image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    payload = path.read_bytes()
    return {
        "width": int(width),
        "height": int(height),
        "file_size": path.stat().st_size,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def caption_for_image(image_path: Path) -> str:
    caption_path = image_path.with_suffix(".txt")
    if not caption_path.exists():
        return ""
    return caption_path.read_text(encoding="utf-8", errors="replace").strip()


def tags_from_caption(caption: str) -> list[str]:
    return [part.strip() for part in caption.replace("\n", ",").split(",") if part.strip()]


def add_reference_image(
    *,
    reference_set_id: int,
    image_path: str,
    image_role: str,
    prompt_role_hint: str = "",
    caption: str = "",
    source_type: str = "manual",
    include_in_machine_review: bool = True,
    exclude_reason: str = "",
    sort_order: int = 0,
    memo: str = "",
) -> int:
    reference_set = fetch_one("SELECT * FROM reference_sets WHERE id = ?", (reference_set_id,))
    if reference_set is None:
        raise ValueError("Reference Setが見つかりません。")
    version_id = reference_set["current_version_id"]
    if not version_id:
        with connect() as conn:
            version_id = create_reference_version(conn, reference_set_id, memo="自動作成")
            conn.execute("UPDATE reference_sets SET current_version_id = ? WHERE id = ?", (version_id, reference_set_id))
    source = Path(normalize_user_path(image_path))
    managed = unique_copy(source, reference_image_dir(reference_set_id, int(version_id)))
    source_caption = caption.strip() or caption_for_image(source)
    metadata = image_metadata(managed)
    now = utc_now()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO reference_images(
                reference_set_id, reference_set_version_id, dataset_id, dataset_version_id,
                image_path, source_type, source_image_path, image_role, prompt_role_hint,
                caption, caption_snapshot, tags_json, width, height, file_size, sha256,
                include_in_machine_review, exclude_reason, sort_order, created_at, updated_at, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reference_set_id,
                version_id,
                reference_set["dataset_id"],
                reference_set["current_dataset_version_id"] or reference_set["dataset_version_id"],
                str(managed),
                source_type,
                str(source),
                valid_role(image_role),
                prompt_role_hint.strip(),
                source_caption,
                source_caption,
                json.dumps(tags_from_caption(source_caption), ensure_ascii=False),
                metadata["width"],
                metadata["height"],
                metadata["file_size"],
                metadata["sha256"],
                1 if include_in_machine_review else 0,
                exclude_reason.strip(),
                sort_order,
                now,
                now,
                memo.strip(),
            ),
        )
        image_id = int(cur.lastrowid)
        refresh_reference_version_summary(conn, int(version_id))
        conn.execute("UPDATE reference_sets SET updated_at = ? WHERE id = ?", (now, reference_set_id))
    return image_id


def update_reference_image(
    image_id: int,
    *,
    image_role: str,
    prompt_role_hint: str = "",
    include_in_machine_review: bool = True,
    exclude_reason: str = "",
    memo: str = "",
) -> None:
    now = utc_now()
    with connect() as conn:
        row = conn.execute("SELECT * FROM reference_images WHERE id = ?", (image_id,)).fetchone()
        if row is None:
            raise ValueError("Reference画像が見つかりません。")
        conn.execute(
            """
            UPDATE reference_images
            SET image_role = ?, prompt_role_hint = ?, include_in_machine_review = ?,
                exclude_reason = ?, memo = ?, updated_at = ?
            WHERE id = ?
            """,
            (valid_role(image_role), prompt_role_hint.strip(), 1 if include_in_machine_review else 0, exclude_reason.strip(), memo.strip(), now, image_id),
        )
        if row["reference_set_version_id"]:
            refresh_reference_version_summary(conn, int(row["reference_set_version_id"]))


def refresh_reference_version_summary(conn: Any, version_id: int) -> None:
    version = conn.execute("SELECT * FROM reference_set_versions WHERE id = ?", (version_id,)).fetchone()
    if version is None:
        return
    image_count = int(conn.execute("SELECT COUNT(*) AS count FROM reference_images WHERE reference_set_version_id = ?", (version_id,)).fetchone()["count"] or 0)
    roles = {
        str(row["image_role"] or "other"): int(row["count"] or 0)
        for row in conn.execute(
            "SELECT image_role, COUNT(*) AS count FROM reference_images WHERE reference_set_version_id = ? GROUP BY image_role",
            (version_id,),
        ).fetchall()
    }
    label, message = completeness_for(valid_reference_type(version["reference_type"]), image_count, roles)
    now = utc_now()
    conn.execute(
        """
        UPDATE reference_set_versions
        SET image_count = ?, roles_json = ?, completeness_label = ?,
            completeness_message = ?, updated_at = ?
        WHERE id = ?
        """,
        (image_count, json.dumps(roles, ensure_ascii=False, sort_keys=True), label, message, now, version_id),
    )


def completeness_for(reference_type: str, image_count: int, roles: dict[str, int]) -> tuple[str, str]:
    if image_count <= 0:
        return "ERROR", "リファレンス画像がありません。"
    if reference_type == "character":
        coverage = sum(1 for key in CHARACTER_REQUIRED if roles.get(key, 0) > 0)
        if image_count >= 3 and coverage >= 2:
            return "OK", "顔・上半身・全身のうち複数の役割が揃っています。"
        missing = [ROLE_LABELS[key] for key in CHARACTER_REQUIRED if roles.get(key, 0) <= 0]
        return "WARNING", f"character確認として不足があります: {', '.join(missing)}。"
    if reference_type == "style":
        coverage = sum(1 for key in ("close_up", "upper_body", "full_body", "background_scene") if roles.get(key, 0) > 0)
        if image_count >= 6 and coverage >= 3:
            return "OK", "style確認として画像数と役割が概ね揃っています。"
        if image_count <= 2:
            return "ERROR", "style確認には画像が少なすぎます。"
        return "WARNING", "style確認として画像数または役割に偏りがあります。"
    if reference_type == "mixed":
        if image_count >= 4:
            return "OK", "mixed用途として最低限の画像数があります。"
        return "WARNING", "mixed用途としては画像数が少なめです。"
    return "UNKNOWN", "分類が未設定のため、人間が確認してください。"


def dataset_image_candidates(dataset_id: int, limit: int = 60) -> list[dict[str, Any]]:
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        return []
    root = Path(dataset["path"])
    if not root.exists():
        return []
    rows = []
    for image_path in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
        caption = caption_for_image(image_path)
        rows.append(
            {
                "path": str(image_path),
                "name": image_path.name,
                "caption": caption,
                "suggested_role": infer_role_from_caption(caption, image_path.name),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def infer_role_from_caption(caption: str, filename: str = "") -> str:
    text = f"{caption}, {filename}".lower()
    if "full body" in text or "全身" in text:
        return "full_body"
    if "upper body" in text or "上半身" in text or "cowboy shot" in text:
        return "upper_body"
    if "close-up" in text or "close up" in text or "portrait" in text or "face" in text or "顔" in text:
        return "face_front"
    if "smile" in text or "expression" in text or "表情" in text:
        return "expression"
    if "background" in text or "outdoors" in text or "scene" in text or "背景" in text:
        return "background_scene"
    return "other"


def reference_detail(set_id: int) -> dict[str, Any]:
    reference_set = fetch_one(
        """
        SELECT r.*, d.name AS dataset_name, dv.version_no AS dataset_version_no,
               p.name AS project_name, v.version_no AS current_version_no,
               v.image_count AS current_image_count, v.roles_json AS current_roles_json,
               v.completeness_label, v.completeness_message
        FROM reference_sets r
        LEFT JOIN datasets d ON d.id = r.dataset_id
        LEFT JOIN dataset_versions dv ON dv.id = r.current_dataset_version_id
        LEFT JOIN lora_projects p ON p.id = r.project_id
        LEFT JOIN reference_set_versions v ON v.id = r.current_version_id
        WHERE r.id = ?
        """,
        (set_id,),
    )
    if reference_set is None:
        raise ValueError("Reference Setが見つかりません。")
    versions = fetch_all("SELECT * FROM reference_set_versions WHERE reference_set_id = ? ORDER BY version_no DESC", (set_id,))
    images = fetch_all("SELECT * FROM reference_images WHERE reference_set_id = ? ORDER BY sort_order, id", (set_id,))
    candidates = dataset_image_candidates(int(reference_set["dataset_id"]), 40) if reference_set["dataset_id"] else []
    try:
        current_roles = json.loads(reference_set["current_roles_json"] or "{}")
    except json.JSONDecodeError:
        current_roles = {}
    return {
        "reference_set": reference_set,
        "versions": versions,
        "images": images,
        "candidates": candidates,
        "current_roles": current_roles,
        "roles": ROLE_LABELS,
        "reference_type_labels": REFERENCE_TYPE_LABELS,
    }


def reference_set_rows() -> list[Any]:
    return fetch_all(
        """
        SELECT r.*, d.name AS dataset_name, dv.version_no AS dataset_version_no,
               p.name AS project_name, v.version_no AS current_version_no,
               v.image_count AS version_image_count, v.completeness_label,
               v.completeness_message
        FROM reference_sets r
        LEFT JOIN datasets d ON d.id = r.dataset_id
        LEFT JOIN dataset_versions dv ON dv.id = r.current_dataset_version_id
        LEFT JOIN lora_projects p ON p.id = r.project_id
        LEFT JOIN reference_set_versions v ON v.id = r.current_version_id
        ORDER BY r.is_archived, r.updated_at DESC, r.id DESC
        """
    )


def set_project_default(reference_set_id: int) -> None:
    row = fetch_one("SELECT * FROM reference_sets WHERE id = ?", (reference_set_id,))
    if row is None:
        raise ValueError("Reference Setが見つかりません。")
    if not row["project_id"]:
        raise ValueError("Projectに紐づいていないReference Setです。")
    now = utc_now()
    with connect() as conn:
        conn.execute(
            "UPDATE reference_sets SET is_default = 0 WHERE project_id = ?",
            (row["project_id"],),
        )
        conn.execute(
            "UPDATE reference_sets SET is_default = 1, updated_at = ? WHERE id = ?",
            (now, reference_set_id),
        )
        conn.execute(
            "UPDATE lora_projects SET default_reference_set_id = ?, default_reference_set_version_id = ?, updated_at = ? WHERE id = ?",
            (reference_set_id, row["current_version_id"], now, row["project_id"]),
        )


def archive_reference_set(reference_set_id: int, archived: bool) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE reference_sets SET is_archived = ?, updated_at = ? WHERE id = ?",
            (1 if archived else 0, utc_now(), reference_set_id),
        )


def export_reference_artifacts(reference_set_id: int) -> dict[str, str]:
    detail = reference_detail(reference_set_id)
    reference_set = detail["reference_set"]
    images = detail["images"]
    out_dir = settings.EXPORTS_DIR / "reference_sets" / f"reference_set_{reference_set_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "contact_sheet.html"
    md_path = out_dir / "reference_report.md"
    title = f"Reference Set #{reference_set_id} {reference_set['name']}"
    figures = []
    for image in images:
        rel = Path(image["image_path"])
        figures.append(
            f"<figure><img src=\"{html.escape(str(rel))}\" alt=\"reference {image['id']}\"><figcaption>#{image['id']} {html.escape(image['image_role'] or 'other')}<br>{html.escape(image['caption_snapshot'] or image['caption'] or '')}</figcaption></figure>"
        )
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset=\"utf-8\"><title>" + html.escape(title) + "</title>",
                "<style>body{font-family:system-ui,sans-serif;margin:24px} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px} img{max-width:100%;height:auto;border:1px solid #ddd} figcaption{font-size:12px}</style></head><body>",
                f"<h1>{html.escape(title)}</h1>",
                f"<p>Type: {html.escape(reference_set['reference_type'] or '-')} / completeness: {html.escape(reference_set['completeness_label'] or '-')}</p>",
                "<div class=\"grid\">",
                *figures,
                "</div></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        f"# {title}",
        "",
        f"- Type: {reference_set['reference_type'] or '-'}",
        f"- Dataset: {reference_set['dataset_name'] or '-'}",
        f"- Completeness: {reference_set['completeness_label'] or '-'}",
        f"- Message: {reference_set['completeness_message'] or '-'}",
        "",
        "## Images",
        "",
    ]
    for image in images:
        lines.append(f"- #{image['id']} `{image['image_role'] or 'other'}` {image['image_path']}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"html": str(html_path), "markdown": str(md_path)}
