from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import settings
from app.db import connect, fetch_all, init_db
from app.services.image_store import (
    is_relative_to,
    managed_reference_image_dir,
    reference_images_root,
    unique_copy,
    verify_image_file,
    validation_images_root,
)
from app.services.validation_runs import validation_image_dir


def migrate_validation_images(dry_run: bool) -> list[dict[str, str]]:
    results = []
    rows = fetch_all("SELECT * FROM validation_images ORDER BY id")
    for row in rows:
        source = Path(row["image_path"])
        if is_relative_to(source, validation_images_root()):
            results.append({"table": "validation_images", "id": str(row["id"]), "status": "already_managed", "path": str(source)})
            continue
        target_dir = validation_target_dir(row)
        results.append(migrate_row("validation_images", row["id"], "image_path", source, target_dir, dry_run))
    return results


def validation_target_dir(row) -> Path:
    if row["validation_run_id"]:
        return validation_image_dir(int(row["validation_run_id"]))
    return validation_images_root() / f"legacy_job_{int(row['job_id']):06d}" / "images"


def migrate_reference_images(dry_run: bool) -> list[dict[str, str]]:
    results = []
    rows = fetch_all("SELECT * FROM reference_images ORDER BY id")
    for row in rows:
        source = Path(row["image_path"])
        if is_relative_to(source, reference_images_root()):
            results.append({"table": "reference_images", "id": str(row["id"]), "status": "already_managed", "path": str(source)})
            continue
        target_dir = managed_reference_image_dir(int(row["reference_set_id"]))
        results.append(migrate_row("reference_images", row["id"], "image_path", source, target_dir, dry_run))
    return results


def migrate_row(table: str, row_id: int, column: str, source: Path, target_dir: Path, dry_run: bool) -> dict[str, str]:
    if not source.exists() or not source.is_file():
        return {"table": table, "id": str(row_id), "status": "skipped_missing", "path": str(source)}
    try:
        if dry_run:
            verify_image_file(source)
            return {"table": table, "id": str(row_id), "status": "would_copy", "path": str(source), "target_dir": str(target_dir)}
        target = unique_copy(source, target_dir)
    except ValueError as exc:
        return {"table": table, "id": str(row_id), "status": "skipped_invalid", "path": str(source), "error": str(exc)}
    with connect() as conn:
        conn.execute(f"UPDATE {table} SET {column} = ? WHERE id = ?", (str(target), row_id))
    return {"table": table, "id": str(row_id), "status": "copied", "path": str(source), "target": str(target)}


def write_report(results: list[dict[str, str]], dry_run: bool) -> Path:
    report_dir = settings.EXPORTS_DIR / "diagnostics"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / ("managed_image_migration_dry_run.json" if dry_run else "managed_image_migration.json")
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy legacy validation/reference images into managed directories.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report actions without updating the database.")
    args = parser.parse_args()
    init_db()
    results = migrate_validation_images(args.dry_run) + migrate_reference_images(args.dry_run)
    report_path = write_report(results, args.dry_run)
    copied = sum(1 for row in results if row["status"] in {"copied", "would_copy"})
    skipped = sum(1 for row in results if row["status"].startswith("skipped"))
    print(f"Managed image migration {'dry-run ' if args.dry_run else ''}complete. copied={copied} skipped={skipped} report={report_path}")


if __name__ == "__main__":
    main()
