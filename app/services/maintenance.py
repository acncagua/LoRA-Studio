from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app import settings
from app.app_version import app_version_info
from app.db import fetch_all, fetch_one

HEAVY_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".zip"}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_app_backup() -> Path:
    backup_dir = settings.ROOT_DIR / "backups" / "app_backups" / f"backup_{timestamp()}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    if settings.DB_PATH.exists():
        shutil.copy2(settings.DB_PATH, backup_dir / "app.db")
    write_app_settings_snapshot(backup_dir / "app_settings.json")
    copy_tree_light(settings.EXPORTS_DIR, backup_dir / "exports")
    copy_run_reports(backup_dir / "runs" / "reports")
    (backup_dir / "README.txt").write_text(
        "\n".join(
            [
                "LoRA-Studio backup",
                "このバックアップは運用ベータ前の軽量バックアップです。",
                "DB、app settings、exports内の軽量ファイル、runs配下のreportsを中心に保存します。",
                "大型のモデル、画像、動画、zipは初期版では除外します。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return backup_dir


def write_app_settings_snapshot(path: Path) -> None:
    rows = fetch_all("SELECT key, value, updated_at FROM app_settings ORDER BY key")
    path.write_text(json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_tree_light(source: Path, destination: Path) -> None:
    if not source.exists():
        destination.mkdir(parents=True, exist_ok=True)
        return
    shutil.copytree(source, destination, ignore=ignore_heavy_files)


def ignore_heavy_files(_dir: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if Path(name).suffix.lower() in HEAVY_SUFFIXES:
            ignored.add(name)
    return ignored


def copy_run_reports(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if not settings.RUNS_DIR.exists():
        return
    for report_dir in settings.RUNS_DIR.glob("job_*/reports"):
        target = destination / report_dir.parent.name
        shutil.copytree(report_dir, target, dirs_exist_ok=True, ignore=ignore_heavy_files)
    comparisons = settings.RUNS_DIR / "comparisons"
    if comparisons.exists():
        shutil.copytree(comparisons, destination / "comparisons", dirs_exist_ok=True, ignore=ignore_heavy_files)


def export_diagnostics() -> Path:
    output_dir = settings.EXPORTS_DIR / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"diagnostics_{timestamp()}.md"
    output_path.write_text(build_diagnostics_markdown(), encoding="utf-8")
    return output_path


def build_diagnostics_markdown() -> str:
    version = app_version_info(get_setting("db_schema_version"))
    env = fetch_one("SELECT * FROM environments ORDER BY id DESC LIMIT 1")
    counts = diagnostic_counts()
    status_counts = job_status_counts()
    latest_errors = collect_latest_errors()
    warnings = known_warnings(counts)
    lines = [
        "# LoRA-Studio Diagnostics",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- App version: {version.app_version}",
        f"- Git commit: {version.git_commit}",
        f"- DB schema version: {version.db_schema_version}",
        "",
        "## Environment",
        f"- Python: {env['python_version'] if env else '-'}",
        f"- Torch: {env['torch_version'] if env else '-'}",
        f"- Torch CUDA: {env['torch_cuda_version'] if env else '-'}",
        f"- CUDA available: {bool(env['cuda_available']) if env else '-'}",
        f"- GPU: {env['gpu_name'] if env else '-'}",
        f"- sd-scripts path: {env['sd_scripts_path'] if env else '-'}",
        f"- sd-scripts commit: {env['sd_scripts_commit_hash'] if env else '-'}",
        "",
        "## Counts",
        f"- Datasets: {counts['datasets']}",
        f"- Jobs: {counts['jobs']}",
        f"- Jobs completed: {status_counts.get('completed', 0)}",
        f"- Jobs failed: {status_counts.get('failed', 0)}",
        f"- Jobs running: {status_counts.get('running', 0)}",
        f"- LoRA Library profiles: {counts['profiles']}",
        f"- Validation Runs: {counts['validation_runs']}",
        "",
        "## Latest Errors",
        latest_errors or "- No recent error lines found.",
        "",
        "## Known Warnings",
        *[f"- {warning}" for warning in warnings],
        "",
    ]
    return "\n".join(lines)


def get_setting(key: str) -> str | None:
    row = fetch_one("SELECT value FROM app_settings WHERE key = ?", (key,))
    return row["value"] if row else None


def diagnostic_counts() -> dict[str, int]:
    return {
        "datasets": int(fetch_one("SELECT COUNT(*) AS count FROM datasets")["count"]),
        "jobs": int(fetch_one("SELECT COUNT(*) AS count FROM training_jobs")["count"]),
        "profiles": int(fetch_one("SELECT COUNT(*) AS count FROM selected_lora_profiles")["count"]),
        "validation_runs": int(fetch_one("SELECT COUNT(*) AS count FROM validation_runs")["count"]),
    }


def job_status_counts() -> dict[str, int]:
    rows = fetch_all("SELECT status, COUNT(*) AS count FROM training_jobs GROUP BY status")
    return {row["status"]: int(row["count"]) for row in rows}


def collect_latest_errors() -> str:
    patterns = ("error", "exception", "traceback", "failed")
    lines: list[str] = []
    for log_path in sorted(settings.LOGS_DIR.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)[:8]:
        try:
            text_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
        except OSError:
            continue
        for line in text_lines:
            lowered = line.lower()
            if any(pattern in lowered for pattern in patterns):
                lines.append(f"- `{log_path.name}`: {line[:500]}")
    return "\n".join(lines[-30:])


def known_warnings(counts: dict[str, int]) -> list[str]:
    warnings = [
        "Codex内ブラウザはWindows sandbox権限で CreateProcessAsUserW failed: 5 になる場合があります。HTTP画面確認で代替できます。",
        "HiresありValidationは標準比較ではなく、最終見栄え確認用です。",
    ]
    incomplete = fetch_one(
        """
        SELECT COUNT(*) AS count
        FROM validation_runs
        WHERE COALESCE(actual_image_count, 0) < COALESCE(expected_image_count, 0)
        """
    )
    if incomplete and int(incomplete["count"]) > 0:
        warnings.append("Validation Runでregistered数がexpected未満のものがあります。Recommendationの信頼度に注意してください。")
    mismatch = fetch_one(
        """
        SELECT COUNT(*) AS count
        FROM validation_runs r
        LEFT JOIN (
            SELECT validation_run_id, COUNT(*) AS condition_count
            FROM validation_expected_conditions
            GROUP BY validation_run_id
        ) c ON c.validation_run_id = r.id
        LEFT JOIN (
            SELECT validation_run_id, COUNT(*) AS image_count
            FROM validation_images
            WHERE validation_run_id IS NOT NULL
            GROUP BY validation_run_id
        ) i ON i.validation_run_id = r.id
        WHERE r.validation_preset_id IS NOT NULL
          AND COALESCE(i.image_count, 0) > 0
          AND COALESCE(c.condition_count, 0) != COALESCE(r.expected_image_count, 0)
        """
    )
    if mismatch and int(mismatch["count"]) > 0:
        warnings.append("Expected Condition count mismatch: 登録済み画像を保護するため、既存condition_hashを維持して自動再生成をスキップしています。")
    if counts["validation_runs"] == 0:
        warnings.append("Validation Runがまだありません。採用LoRAの外部Validationは未確認です。")
    return warnings


def maintenance_summary() -> dict[str, Any]:
    version = app_version_info(get_setting("db_schema_version"))
    env = fetch_one("SELECT * FROM environments ORDER BY id DESC LIMIT 1")
    return {
        "version": version,
        "environment": env,
        "counts": diagnostic_counts(),
        "status_counts": job_status_counts(),
        "warnings": known_warnings(diagnostic_counts()),
    }
