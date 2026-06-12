from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app import settings
from app.app_version import app_version_info
from app.db import connect, create_dataset_version, create_job, fetch_all, fetch_one, import_latest_environment, init_db, insert_dataset, upsert_dataset_analysis
from app.services.command_builder import prepare_job_files
from app.services.exports import export_selected_lora, write_compare_contact_sheet, write_job_contact_sheet, write_validation_pack
from app.services.image_store import (
    copy_managed_reference_image,
    ensure_allowed_file,
    normalize_user_path,
    reference_images_root,
    unique_copy,
    validation_images_root,
    verify_image_file,
)
from app.services.maintenance import create_app_backup, export_diagnostics, maintenance_summary
from app.services.output_collector import collect_job_results
from app.services.recommendations import create_draft_job_from_recommendation, list_recommendations, regenerate_recommendations, set_recommendation_status, write_recommendation_report
from app.services.training_runner import read_log_tail, start_job, stop_job, validate_job_ready
from app.services.validation_runs import (
    apply_suggestion_to_profile,
    calculate_suggested_weights,
    copy_managed_validation_image,
    create_validation_run,
    expand_preset_conditions,
    persist_suggestion,
    load_validation_run_bundle,
    make_condition_hash,
    update_validation_run_status,
    update_validation_run_counts,
    validation_presets,
    validation_run_dir,
    write_validation_report,
    write_validation_prompt_pack,
)

app = FastAPI(title=settings.APP_NAME)
app.mount("/static", StaticFiles(directory=settings.ROOT_DIR / "app" / "static"), name="static")

RUBRIC_VERSION = "1.0"
STRENGTH_LABELS = [
    ("", "未評価"),
    ("too_weak", "弱すぎ"),
    ("weak_but_usable", "弱いが使用可"),
    ("recommended", "推奨"),
    ("strong_but_usable", "強いが使用可"),
    ("too_strong", "強すぎ"),
    ("broken", "破綻"),
]
OVERFIT_LEVELS = [("", "未評価"), ("none", "なし"), ("slight", "軽微"), ("moderate", "中程度"), ("severe", "重度")]
ADOPTION_LABELS = [("", "未評価"), ("reject", "不採用"), ("candidate", "候補"), ("adopt", "採用")]
FAILURE_TAGS = [
    "顔が弱い",
    "顔が変わる",
    "衣装が弱い",
    "衣装固定",
    "背景汚染",
    "構図固定",
    "表情固定",
    "手足破綻",
    "画風過多",
    "LoRA効果弱い",
    "LoRA効果強すぎ",
    "trigger反応弱い",
    "triggerなし暴発",
]

templates = Environment(
    loader=FileSystemLoader(settings.ROOT_DIR / "app" / "templates"),
    autoescape=select_autoescape(["html", "xml"]),
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def render(request: Request, template: str, **context: Any) -> HTMLResponse:
    tpl = templates.get_template(template)
    context.setdefault("app_name", settings.APP_NAME)
    context.setdefault("request", request)
    context.setdefault("sd_scripts_release_tag", settings.SD_SCRIPTS_RELEASE_TAG)
    context.setdefault("sd_scripts_release_commit", settings.SD_SCRIPTS_RELEASE_COMMIT)
    context.setdefault("app_meta", current_app_meta())
    return HTMLResponse(tpl.render(**context))


def current_app_meta() -> dict[str, Any]:
    schema_row = fetch_one("SELECT value FROM app_settings WHERE key = ?", ("db_schema_version",))
    env = fetch_one("SELECT sd_scripts_commit_hash FROM environments ORDER BY id DESC LIMIT 1")
    version = app_version_info(schema_row["value"] if schema_row else None)
    return {
        "app_version": version.app_version,
        "git_commit": version.git_commit,
        "db_schema_version": version.db_schema_version,
        "sd_scripts_commit_hash": (env["sd_scripts_commit_hash"] if env and env["sd_scripts_commit_hash"] else settings.SD_SCRIPTS_RELEASE_COMMIT),
    }


@app.get("/api/browse-directory")
def api_browse_directory(title: str = "フォルダを選択", initial_path: str = "") -> dict[str, str]:
    return {"path": open_windows_directory_dialog(title, initial_path)}


@app.get("/api/browse-file")
def api_browse_file(title: str = "ファイルを選択", kind: str = "file", initial_path: str = "") -> dict[str, str]:
    return {"path": open_windows_file_dialog(title, kind, initial_path)}


def open_windows_directory_dialog(title: str, initial_path: str = "") -> str:
    initial_directory = resolve_dialog_initial_path(initial_path)
    selected_path_command = (
        f"$dialog.SelectedPath = {ps_quote(initial_directory)}; "
        if initial_directory
        else ""
    )
    command = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
        f"$dialog.Description = {ps_quote(title)}; "
        "$dialog.ShowNewFolderButton = $false; "
        f"{selected_path_command}"
        "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
        "  Write-Output $dialog.SelectedPath "
        "}"
    )
    return run_dialog_command(command)


def open_windows_file_dialog(title: str, kind: str, initial_path: str = "") -> str:
    if kind == "model":
        filter_text = "Stable Diffusion model (*.safetensors;*.ckpt)|*.safetensors;*.ckpt|All files (*.*)|*.*"
    elif kind == "image":
        filter_text = "Image files (*.png;*.jpg;*.jpeg;*.webp)|*.png;*.jpg;*.jpeg;*.webp|All files (*.*)|*.*"
    else:
        filter_text = "All files (*.*)|*.*"
    initial_directory, file_name = resolve_file_dialog_initial_values(initial_path)
    initial_directory_command = (
        f"$dialog.InitialDirectory = {ps_quote(initial_directory)}; "
        if initial_directory
        else ""
    )
    file_name_command = f"$dialog.FileName = {ps_quote(file_name)}; " if file_name else ""
    command = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        "$dialog = New-Object System.Windows.Forms.OpenFileDialog; "
        f"$dialog.Title = {ps_quote(title)}; "
        f"$dialog.Filter = {ps_quote(filter_text)}; "
        f"{initial_directory_command}"
        f"{file_name_command}"
        "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
        "  Write-Output $dialog.FileName "
        "}"
    )
    return run_dialog_command(command)


def resolve_dialog_initial_path(initial_path: str) -> str:
    if not initial_path:
        return ""
    path = Path(initial_path)
    if path.is_file():
        return str(path.parent)
    if path.is_dir():
        return str(path)
    if path.parent.is_dir():
        return str(path.parent)
    return ""


def resolve_file_dialog_initial_values(initial_path: str) -> tuple[str, str]:
    if not initial_path:
        return "", ""
    path = Path(initial_path)
    if path.is_file():
        return str(path.parent), path.name
    if path.is_dir():
        return str(path), ""
    if path.parent.is_dir():
        return str(path.parent), path.name
    return "", ""


def run_dialog_command(command: str) -> str:
    if not is_windows():
        raise HTTPException(status_code=400, detail="Windows dialog is only available on Windows.")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Sta", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail=(result.stderr or "Dialog was not completed.").strip())
    return result.stdout.strip()


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def is_windows() -> bool:
    import sys

    return sys.platform == "win32"


STATUS_LABELS = {
    "draft": "下書き",
    "prepared": "準備済み",
    "prepared_dirty": "設定変更あり",
    "running": "実行中",
    "completed": "完了",
    "failed": "失敗",
    "stopped": "停止",
}

EDITABLE_JOB_STATUSES = {"draft", "prepared", "prepared_dirty", "failed", "stopped"}


def job_action_state(job: Any, selected_output: Any | None = None) -> dict[str, Any]:
    status = job["status"]
    dirty = bool(job["config_dirty"] if "config_dirty" in job.keys() else 0)
    return {
        "edit": status in EDITABLE_JOB_STATUSES,
        "prepare": status in {"draft", "prepared", "prepared_dirty", "failed", "stopped"},
        "preflight": status == "prepared" and not dirty,
        "run": status in {"prepared", "failed", "stopped"} and not dirty,
        "stop": status == "running",
        "reimport": status in {"completed", "failed", "stopped"},
        "select": status == "completed",
        "clone": status != "running",
        "revised": status in {"running", "completed"},
        "compare": status == "completed",
        "export": status == "completed" and selected_output is not None,
        "contact_sheet": status == "completed",
        "dirty": dirty,
    }


def recommended_next_action(job: Any, selected_output: Any | None = None) -> str:
    status = job["status"]
    dirty = bool(job["config_dirty"] if "config_dirty" in job.keys() else 0)
    if dirty or status == "prepared_dirty":
        return "次にやること: 設定が変更されています。実行前にファイル準備を再実行してください。"
    if status == "draft":
        return "次にやること: 編集内容を確認し、ファイル準備を実行してください。"
    if status == "prepared":
        return "次にやること: 事前確認後、実行できます。"
    if status == "running":
        return "次にやること: 学習中です。train.logを確認できます。必要なら停止してください。"
    if status == "completed" and selected_output is not None:
        return "次にやること: 採用LoRA出力またはValidation Runを作成してください。"
    if status == "completed":
        return "次にやること: sample画像を評価し、採用LoRAを選択してください。"
    if status == "failed":
        return "次にやること: train.log末尾を確認し、設定を修正して再度ファイル準備を実行してください。"
    if status == "stopped":
        return "次にやること: 必要なら設定を見直し、ファイル準備後に実行してください。"
    return "次にやること: ジョブ詳細の操作パネルから次の操作を選んでください。"


def job_latest_action(job: Any) -> str:
    status = job["status"]
    dirty = bool(job["config_dirty"] if "config_dirty" in job.keys() else 0)
    if dirty or status == "prepared_dirty":
        return "再準備が必要"
    if status == "draft":
        return "ファイル準備"
    if status == "prepared":
        return "事前確認 / 実行"
    if status == "running":
        return "ログ確認 / 停止"
    if status == "completed":
        return "評価 / Export / 比較"
    if status == "failed":
        return "ログ確認 / 修正"
    if status == "stopped":
        return "再準備 / 実行"
    return "詳細確認"


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    stats = {
        "presets": fetch_one("SELECT COUNT(*) AS count FROM presets")["count"],
        "datasets": fetch_one("SELECT COUNT(*) AS count FROM datasets")["count"],
        "jobs": fetch_one("SELECT COUNT(*) AS count FROM training_jobs")["count"],
        "running": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'running'")["count"],
        "completed": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'completed'")["count"],
        "failed": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'failed'")["count"],
        "stopped": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'stopped'")["count"],
    }
    jobs = fetch_all("SELECT * FROM training_jobs ORDER BY id DESC LIMIT 8")
    attention_jobs = fetch_all(
        """
        SELECT * FROM training_jobs
        WHERE status IN ('draft', 'prepared', 'prepared_dirty', 'failed', 'running')
           OR COALESCE(config_dirty, 0) = 1
        ORDER BY
            CASE status
                WHEN 'running' THEN 1
                WHEN 'failed' THEN 2
                WHEN 'prepared_dirty' THEN 3
                WHEN 'prepared' THEN 4
                WHEN 'draft' THEN 5
                ELSE 9
            END,
            id DESC
        LIMIT 10
        """
    )
    validation_profiles = lora_library_profiles(limit=6)
    return render(
        request,
        "dashboard.html",
        stats=stats,
        jobs=jobs,
        attention_jobs=attention_jobs,
        validation_profiles=validation_profiles,
        status_labels=STATUS_LABELS,
    )


@app.get("/maintenance", response_class=HTMLResponse)
def maintenance(request: Request, backup_path: str = "", diagnostics_path: str = "") -> HTMLResponse:
    return render(
        request,
        "maintenance.html",
        summary=maintenance_summary(),
        backup_path=backup_path,
        diagnostics_path=diagnostics_path,
    )


@app.post("/maintenance/backup")
def maintenance_backup() -> RedirectResponse:
    backup_path = create_app_backup()
    return RedirectResponse(f"/maintenance?backup_path={quote(str(backup_path))}", status_code=303)


@app.post("/maintenance/diagnostics")
def maintenance_diagnostics() -> RedirectResponse:
    diagnostics_path = export_diagnostics()
    return RedirectResponse(f"/maintenance?diagnostics_path={quote(str(diagnostics_path))}", status_code=303)


@app.get("/workflow", response_class=HTMLResponse)
def recommended_workflow(request: Request) -> HTMLResponse:
    return render(request, "workflow.html")


@app.get("/environment", response_class=HTMLResponse)
def environment(request: Request) -> HTMLResponse:
    import_latest_environment()
    settings_rows = fetch_all("SELECT * FROM app_settings ORDER BY key")
    environments = fetch_all("SELECT * FROM environments ORDER BY id DESC")
    return render(request, "environment.html", settings_rows=settings_rows, environments=environments, settings=settings)


@app.get("/presets", response_class=HTMLResponse)
def presets(request: Request) -> HTMLResponse:
    rows = fetch_all("SELECT * FROM presets ORDER BY model_family DESC, name")
    return render(request, "presets.html", presets=rows)


@app.get("/presets/{preset_id}", response_class=HTMLResponse)
def preset_detail(request: Request, preset_id: str) -> HTMLResponse:
    preset = fetch_one("SELECT * FROM presets WHERE id = ?", (preset_id,))
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    return render(request, "preset_detail.html", preset=preset)


@app.get("/datasets", response_class=HTMLResponse)
def datasets(request: Request) -> HTMLResponse:
    rows = fetch_all("SELECT * FROM datasets ORDER BY id DESC")
    return render(request, "datasets.html", datasets=rows, default_dataset_path=str(settings.ROOT_DIR / "datasets"))


@app.post("/datasets")
def datasets_create(name: str = Form(...), path: str = Form(...), model_family: str = Form("SDXL"), trigger_word: str = Form(""), class_token: str = Form("person"), memo: str = Form("")) -> RedirectResponse:
    insert_dataset(name, path, model_family, trigger_word, class_token, memo)
    return RedirectResponse("/datasets", status_code=303)


@app.get("/datasets/{dataset_id}", response_class=HTMLResponse)
def dataset_detail(request: Request, dataset_id: int) -> HTMLResponse:
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    analysis = fetch_one("SELECT * FROM dataset_analysis WHERE dataset_id = ?", (dataset_id,))
    if analysis is None:
        rescan_dataset(dataset_id)
        dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        analysis = fetch_one("SELECT * FROM dataset_analysis WHERE dataset_id = ?", (dataset_id,))
    history = fetch_all("SELECT * FROM caption_edit_history WHERE dataset_id = ? ORDER BY id DESC LIMIT 10", (dataset_id,))
    versions = fetch_all("SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY version_no DESC", (dataset_id,))
    return render(
        request,
        "dataset_detail.html",
        dataset=dataset,
        analysis=decode_analysis(analysis),
        history=history,
        versions=versions,
        missing_trigger_captions=missing_trigger_caption_rows(dict(dataset))[:100],
        caption_preview=None,
        restore_preview=None,
    )


@app.post("/datasets/{dataset_id}/rescan")
def dataset_rescan(dataset_id: int) -> RedirectResponse:
    rescan_dataset(dataset_id, memo="Manual rescan")
    return RedirectResponse(f"/datasets/{dataset_id}", status_code=303)


@app.post("/datasets/{dataset_id}/update-trigger")
def dataset_update_trigger(dataset_id: int, trigger_word: str = Form(...)) -> RedirectResponse:
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    now = settings_now()
    with connect() as conn:
        conn.execute(
            "UPDATE datasets SET trigger_word = ?, updated_at = ? WHERE id = ?",
            (trigger_word.strip(), now, dataset_id),
        )
    rescan_dataset(dataset_id, memo=f"Updated trigger_word to {trigger_word.strip()}")
    return RedirectResponse(f"/datasets/{dataset_id}", status_code=303)


@app.post("/datasets/{dataset_id}/caption-prepend-preview", response_class=HTMLResponse)
def dataset_caption_prepend_preview(request: Request, dataset_id: int, trigger_word: str = Form("")) -> HTMLResponse:
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    analysis = fetch_one("SELECT * FROM dataset_analysis WHERE dataset_id = ?", (dataset_id,))
    preview = preview_caption_prepend(dict(dataset), trigger_word.strip() or dataset["trigger_word"] or "")
    history = fetch_all("SELECT * FROM caption_edit_history WHERE dataset_id = ? ORDER BY id DESC LIMIT 10", (dataset_id,))
    versions = fetch_all("SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY version_no DESC", (dataset_id,))
    return render(
        request,
        "dataset_detail.html",
        dataset=dataset,
        analysis=decode_analysis(analysis),
        history=history,
        versions=versions,
        missing_trigger_captions=missing_trigger_caption_rows(dict(dataset))[:100],
        caption_preview=preview,
        restore_preview=None,
    )


@app.post("/datasets/{dataset_id}/caption-prepend-confirm")
def dataset_caption_prepend_confirm(
    dataset_id: int,
    trigger_word: str = Form(...),
    confirm: str = Form(""),
) -> RedirectResponse:
    if confirm != "yes":
        raise HTTPException(status_code=400, detail="Confirm checkbox is required.")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    result = apply_caption_prepend(dict(dataset), trigger_word.strip())
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO caption_edit_history(
                dataset_id, action, trigger_word, changed_count, skipped_count,
                backup_path, created_at, memo
            )
            VALUES (?, 'prepend_trigger', ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                trigger_word.strip(),
                result["changed_count"],
                result["skipped_count"],
                result["backup_path"],
                now,
                result["memo"],
            ),
        )
    rescan_dataset(dataset_id, memo=f"After caption prepend: {trigger_word.strip()}")
    return RedirectResponse(f"/datasets/{dataset_id}", status_code=303)


@app.post("/datasets/{dataset_id}/restore-preview", response_class=HTMLResponse)
def dataset_restore_preview(request: Request, dataset_id: int, history_id: int = Form(...)) -> HTMLResponse:
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    history_row = fetch_one("SELECT * FROM caption_edit_history WHERE id = ? AND dataset_id = ?", (history_id, dataset_id))
    if dataset is None or history_row is None:
        raise HTTPException(status_code=404, detail="Dataset or history not found")
    analysis = fetch_one("SELECT * FROM dataset_analysis WHERE dataset_id = ?", (dataset_id,))
    history = fetch_all("SELECT * FROM caption_edit_history WHERE dataset_id = ? ORDER BY id DESC LIMIT 10", (dataset_id,))
    versions = fetch_all("SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY version_no DESC", (dataset_id,))
    preview = preview_restore(dict(dataset), dict(history_row))
    return render(
        request,
        "dataset_detail.html",
        dataset=dataset,
        analysis=decode_analysis(analysis),
        history=history,
        versions=versions,
        missing_trigger_captions=missing_trigger_caption_rows(dict(dataset))[:100],
        caption_preview=None,
        restore_preview=preview,
    )


@app.post("/datasets/{dataset_id}/restore-confirm")
def dataset_restore_confirm(dataset_id: int, history_id: int = Form(...), confirm: str = Form("")) -> RedirectResponse:
    if confirm != "yes":
        raise HTTPException(status_code=400, detail="Confirm checkbox is required.")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    history_row = fetch_one("SELECT * FROM caption_edit_history WHERE id = ? AND dataset_id = ?", (history_id, dataset_id))
    if dataset is None or history_row is None:
        raise HTTPException(status_code=404, detail="Dataset or history not found")
    result = apply_restore(dict(dataset), dict(history_row))
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO caption_edit_history(
                dataset_id, action, trigger_word, changed_count, skipped_count,
                backup_path, created_at, memo
            )
            VALUES (?, 'restore', ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                dataset["trigger_word"],
                result["changed_count"],
                result["skipped_count"],
                history_row["backup_path"],
                now,
                f"Restored captions from history #{history_id}",
            ),
        )
    rescan_dataset(dataset_id, memo=f"After restore from history #{history_id}")
    return RedirectResponse(f"/datasets/{dataset_id}", status_code=303)


@app.get("/sample-prompt-templates", response_class=HTMLResponse)
def sample_prompt_templates(request: Request) -> HTMLResponse:
    templates_rows = fetch_all("SELECT * FROM sample_prompt_templates ORDER BY is_builtin DESC, name")
    return render(request, "sample_prompt_templates.html", templates=templates_rows)


def rescan_dataset(dataset_id: int, memo: str = "Rescan") -> None:
    from app.services.dataset_scanner import scan_dataset

    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    scan = scan_dataset(Path(dataset["path"]), dataset["trigger_word"] or "")
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE datasets
            SET image_count = ?, caption_count = ?, missing_caption_count = ?,
                resolution_summary_json = ?, tag_summary_json = ?,
                scan_status = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                scan["image_count"],
                scan["caption_count"],
                scan["missing_caption_count"],
                json.dumps(scan.get("resolution_summary") or {}, ensure_ascii=False),
                json.dumps(scan.get("tag_summary") or {}, ensure_ascii=False),
                scan["status"],
                now,
                dataset_id,
            ),
        )
        upsert_dataset_analysis(conn, dataset_id, scan)
        create_dataset_version(conn, dataset_id, scan, memo)


def preview_caption_prepend(dataset: dict[str, Any], trigger_word: str) -> dict[str, Any]:
    rows = caption_prepend_rows(dataset, trigger_word)
    changed = [row for row in rows if row["status"] == "change"]
    skipped = [row for row in rows if row["status"] != "change"]
    backup_path = caption_backup_path(int(dataset["id"]))
    return {
        "trigger_word": trigger_word,
        "changed_count": len(changed),
        "skipped_count": len(skipped),
        "backup_path": str(backup_path),
        "samples": changed[:5],
        "warnings": [row for row in rows if row["status"] == "warning"][:20],
    }


def apply_caption_prepend(dataset: dict[str, Any], trigger_word: str) -> dict[str, Any]:
    rows = caption_prepend_rows(dataset, trigger_word)
    changed = [row for row in rows if row["status"] == "change"]
    skipped_count = len(rows) - len(changed)
    backup_dir = caption_backup_path(int(dataset["id"]))
    backup_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset["path"]).resolve()
    for row in changed:
        caption_path = Path(row["path"])
        relative = caption_path.resolve().relative_to(dataset_path)
        backup_file = backup_dir / relative
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(caption_path, backup_file)
        caption_path.write_text(row["after"], encoding="utf-8")
    return {
        "changed_count": len(changed),
        "skipped_count": skipped_count,
        "backup_path": str(backup_dir),
        "memo": "Prepended trigger_word to captions that did not already contain it.",
    }


def caption_prepend_rows(dataset: dict[str, Any], trigger_word: str) -> list[dict[str, str]]:
    if not trigger_word:
        return []
    dataset_path = Path(dataset["path"])
    rows = []
    for caption_path in sorted(dataset_path.rglob("*.txt")):
        try:
            before = caption_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            rows.append({"path": str(caption_path), "status": "warning", "before": "", "after": "Unreadable as UTF-8; skipped."})
            continue
        except OSError as exc:
            rows.append({"path": str(caption_path), "status": "warning", "before": "", "after": f"{exc}; skipped."})
            continue
        if trigger_word in before:
            rows.append({"path": str(caption_path), "status": "skip", "before": before, "after": before})
            continue
        stripped = before.strip()
        after = f"{trigger_word}, {stripped}\n" if stripped else f"{trigger_word}\n"
        rows.append({"path": str(caption_path), "status": "change", "before": before, "after": after})
    return rows


def caption_backup_path(dataset_id: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return settings.ROOT_DIR / "backups" / "datasets" / f"dataset_{dataset_id:06d}" / f"captions_{stamp}"


def missing_trigger_caption_rows(dataset: dict[str, Any]) -> list[dict[str, str]]:
    trigger_word = (dataset.get("trigger_word") or "").strip()
    if not trigger_word:
        return []
    rows = []
    dataset_path = Path(dataset["path"])
    for caption_path in sorted(dataset_path.rglob("*.txt")):
        try:
            text = caption_path.read_text(encoding="utf-8")
        except Exception:
            continue
        if trigger_word in text:
            continue
        image_path = matching_image_path(caption_path)
        rows.append(
            {
                "caption_path": str(caption_path),
                "image_filename": image_path.name if image_path else caption_path.with_suffix("").name,
                "caption_filename": caption_path.name,
                "preview": text.strip()[:240],
            }
        )
    return rows


def matching_image_path(caption_path: Path) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = caption_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def preview_restore(dataset: dict[str, Any], history_row: dict[str, Any]) -> dict[str, Any]:
    backup_path = Path(history_row["backup_path"] or "")
    dataset_path = Path(dataset["path"]).resolve()
    rows = []
    if not backup_path.exists():
        return {"history_id": history_row["id"], "backup_path": str(backup_path), "changed_count": 0, "samples": [], "missing": True}
    for backup_file in sorted(backup_path.rglob("*.txt")):
        relative = backup_file.relative_to(backup_path)
        target = dataset_path / relative
        try:
            before = target.read_text(encoding="utf-8") if target.exists() else ""
            after = backup_file.read_text(encoding="utf-8")
        except OSError:
            continue
        rows.append({"path": str(target), "before": before, "after": after})
    return {
        "history_id": history_row["id"],
        "backup_path": str(backup_path),
        "changed_count": len(rows),
        "samples": rows[:5],
        "missing": False,
    }


def apply_restore(dataset: dict[str, Any], history_row: dict[str, Any]) -> dict[str, Any]:
    preview = preview_restore(dataset, history_row)
    if preview.get("missing"):
        raise HTTPException(status_code=400, detail="Backup path does not exist.")
    changed = 0
    backup_path = Path(history_row["backup_path"])
    dataset_path = Path(dataset["path"]).resolve()
    for backup_file in sorted(backup_path.rglob("*.txt")):
        relative = backup_file.relative_to(backup_path)
        target = dataset_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(backup_file.read_text(encoding="utf-8"), encoding="utf-8")
        changed += 1
    return {"changed_count": changed, "skipped_count": 0}


def settings_now() -> str:
    from app.db import utc_now

    return utc_now()


def decode_analysis(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    decoded = dict(row)
    for key in (
        "caption_encoding_summary_json",
        "image_size_summary_json",
        "tag_summary_json",
        "missing_caption_images_json",
        "caption_without_images_json",
        "broken_images_json",
        "unsupported_files_json",
        "analysis_json",
        "trigger_candidates_json",
    ):
        value = decoded.get(key)
        decoded[key.removesuffix("_json")] = json.loads(value) if value else {} if key.endswith("summary_json") or key == "analysis_json" else []
    return decoded


@app.get("/jobs", response_class=HTMLResponse)
def jobs_list(request: Request) -> HTMLResponse:
    rows = fetch_all(
        """
        SELECT
            j.*,
            d.name AS dataset_name,
            p.name AS preset_name,
            p.model_family AS preset_family,
            s.health_label AS summary_health_label
        FROM training_jobs j
        LEFT JOIN datasets d ON d.id = j.dataset_id
        LEFT JOIN presets p ON p.id = j.preset_id
        LEFT JOIN training_metric_summaries s ON s.job_id = j.id
        ORDER BY j.id DESC
        """
    )
    jobs = []
    for row in rows:
        item = dict(row)
        item["latest_action"] = job_latest_action(row)
        item["actions"] = job_action_state(row, None)
        jobs.append(item)
    return render(request, "jobs.html", jobs=jobs, status_labels=STATUS_LABELS)


@app.get("/jobs/new", response_class=HTMLResponse)
def job_new(request: Request) -> HTMLResponse:
    datasets = fetch_all("SELECT * FROM datasets ORDER BY id DESC")
    presets = fetch_all("SELECT * FROM presets ORDER BY model_family DESC, name")
    sample_prompt_templates = fetch_all("SELECT * FROM sample_prompt_templates ORDER BY is_builtin DESC, name")
    trigger_infos = {row["dataset_id"]: row for row in fetch_all("SELECT * FROM dataset_analysis")}
    return render(
        request,
        "job_create.html",
        datasets=datasets,
        presets=presets,
        sample_prompt_templates=sample_prompt_templates,
        trigger_infos=trigger_infos,
        available_models=list_available_models(),
        default_model_path=str(settings.ROOT_DIR / "models"),
        default_project_path=str(settings.ROOT_DIR),
    )


@app.post("/jobs")
def job_create(name: str = Form(...), dataset_id: int = Form(...), preset_id: str = Form(...), base_model_path: str = Form(...), vae_path: str = Form(""), output_name: str = Form(""), memo: str = Form(""), sample_prompt_template_id: str = Form("")) -> RedirectResponse:
    job_id = create_job({"name": name, "dataset_id": dataset_id, "preset_id": preset_id, "base_model_path": base_model_path, "vae_path": vae_path, "output_name": output_name, "memo": memo, "sample_prompt_template_id": sample_prompt_template_id})
    return RedirectResponse(f"/jobs/{job_id}?created=1", status_code=303)


def list_available_models() -> list[dict[str, str]]:
    models_dir = settings.ROOT_DIR / "models"
    if not models_dir.exists():
        return []
    extensions = {".safetensors", ".ckpt"}
    rows = []
    for path in sorted(p for p in models_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions):
        rows.append({"name": path.name, "path": str(path)})
    return rows


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(
    request: Request,
    job_id: int,
    exported: str | None = None,
    created: str | None = None,
    preflight: str | None = None,
) -> HTMLResponse:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
    outputs = fetch_all("SELECT * FROM training_outputs WHERE job_id = ? ORDER BY epoch, step, id", (job_id,))
    samples = fetch_all("SELECT * FROM sample_images WHERE job_id = ? ORDER BY epoch, id", (job_id,))
    sample_prompts = fetch_all("SELECT * FROM sample_prompts WHERE job_id = ? ORDER BY sort_order, id", (job_id,))
    metrics = fetch_all("SELECT * FROM training_metrics WHERE job_id = ? ORDER BY step, id", (job_id,))
    metric_summary = fetch_one("SELECT * FROM training_metric_summaries WHERE job_id = ?", (job_id,))
    epoch_summaries = fetch_all("SELECT * FROM training_epoch_summaries WHERE job_id = ? ORDER BY epoch", (job_id,))
    decorated_epochs = decorate_epoch_summaries(epoch_summaries, outputs, samples)
    log_tail = read_log_tail(dict(job))
    selected_output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    validation_results = fetch_all("SELECT * FROM validation_results WHERE job_id = ? ORDER BY created_at DESC, id DESC", (job_id,))
    validation_images = fetch_all("SELECT * FROM validation_images WHERE job_id = ? ORDER BY created_at DESC, id DESC", (job_id,))
    validation_weight_reviews = fetch_all("SELECT * FROM validation_weight_reviews WHERE job_id = ? ORDER BY lora_weight, id", (job_id,))
    validation_runs = fetch_all("SELECT * FROM validation_runs WHERE job_id = ? ORDER BY id DESC", (job_id,))
    validation_summary = build_validation_summary(validation_results)
    selected_lora_profile = ensure_selected_lora_profile(job_id) if selected_output else None
    recommendations = list_recommendations(job_id)
    dataset_version = fetch_one("SELECT * FROM dataset_versions WHERE id = ?", (job["dataset_version_id"],)) if job["dataset_version_id"] else None
    params = json.loads(job["params_json"])
    return render(
        request,
        "job_detail.html",
        job=job,
        dataset=dataset,
        outputs=outputs,
        samples=samples,
        sample_prompts=sample_prompts,
        sample_groups=group_samples(sample_prompts, samples),
        metrics=metrics,
        metric_summary=metric_summary,
        epoch_summaries=decorated_epochs,
        epoch_visual_summaries=build_epoch_visual_summaries(decorated_epochs, samples),
        health_details=health_details(metric_summary, len(metrics)),
        trigger_status=job_trigger_status(job, dataset, sample_prompts),
        loss_chart=build_loss_chart(metrics),
        log_tail=log_tail,
        selected_output=selected_output,
        validation_results=validation_results,
        validation_images=validation_images,
        validation_weight_reviews=validation_weight_reviews,
        validation_runs=validation_runs,
        validation_presets=validation_presets(),
        validation_summary=validation_summary,
        selected_lora_profile=selected_lora_profile,
        recommendations=recommendations,
        rubric_options=rubric_options(),
        validation_pack_path=validation_pack_path(job_id),
        default_project_path=str(settings.ROOT_DIR),
        dataset_version=dataset_version,
        no_metadata_enabled=bool(params.get("no_metadata")),
        action_state=job_action_state(job, selected_output),
        next_action=recommended_next_action(job, selected_output),
        status_labels=STATUS_LABELS,
        created=created,
        preflight=preflight,
        exported=exported,
    )


@app.get("/jobs/{job_id}/edit", response_class=HTMLResponse)
def job_edit(request: Request, job_id: int) -> HTMLResponse:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    datasets = fetch_all("SELECT * FROM datasets ORDER BY id DESC")
    presets = fetch_all("SELECT * FROM presets ORDER BY model_family DESC, name")
    sample_prompt_templates = fetch_all("SELECT * FROM sample_prompt_templates ORDER BY is_builtin DESC, name")
    dataset_versions = fetch_all("SELECT * FROM dataset_versions ORDER BY dataset_id, version_no DESC")
    editable = job["status"] in EDITABLE_JOB_STATUSES
    return render(
        request,
        "job_edit.html",
        job=job,
        params=json.loads(job["params_json"]),
        params_json_pretty=json.dumps(json.loads(job["params_json"]), ensure_ascii=False, indent=2),
        datasets=datasets,
        presets=presets,
        sample_prompt_templates=sample_prompt_templates,
        dataset_versions=dataset_versions,
        available_models=list_available_models(),
        default_model_path=str(settings.ROOT_DIR / "models"),
        editable=editable,
        status_labels=STATUS_LABELS,
    )


@app.post("/jobs/{job_id}/edit")
def job_edit_save(
    job_id: int,
    name: str = Form(...),
    dataset_id: int = Form(...),
    dataset_version_id: str = Form(""),
    preset_id: str = Form(...),
    base_model_path: str = Form(...),
    vae_path: str = Form(""),
    output_name: str = Form(...),
    sample_prompt_template_id: str = Form(""),
    memo: str = Form(""),
    max_train_epochs: str = Form(""),
    repeats: str = Form(""),
    train_batch_size: str = Form(""),
    learning_rate: str = Form(""),
    unet_lr: str = Form(""),
    text_encoder_lr: str = Form(""),
    network_dim: str = Form(""),
    network_alpha: str = Form(""),
    resolution: str = Form(""),
    save_every_n_epochs: str = Form(""),
    sample_every_n_epochs: str = Form(""),
    optimizer_type: str = Form(""),
    lr_scheduler: str = Form(""),
    no_metadata: str = Form(""),
    params_json: str = Form("{}"),
) -> RedirectResponse:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in EDITABLE_JOB_STATUSES:
        raise HTTPException(status_code=400, detail="このジョブは実行済みまたは実行中のため、直接編集できません。派生ドラフトを作成してください。")
    preset = fetch_one("SELECT * FROM presets WHERE id = ?", (preset_id,))
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if preset is None or dataset is None:
        raise HTTPException(status_code=400, detail="プリセットまたはデータセットが見つかりません。")
    try:
        params = json.loads(params_json or "{}")
        if not isinstance(params, dict):
            raise ValueError("params_json must be an object")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Advanced JSONを読み込めません: {exc}") from exc
    try:
        update_params_from_form(
            params,
            {
                "max_train_epochs": max_train_epochs,
                "repeats": repeats,
                "train_batch_size": train_batch_size,
                "learning_rate": learning_rate,
                "unet_lr": unet_lr,
                "text_encoder_lr": text_encoder_lr,
                "network_dim": network_dim,
                "network_alpha": network_alpha,
                "resolution": resolution,
                "save_every_n_epochs": save_every_n_epochs,
                "sample_every_n_epochs": sample_every_n_epochs,
                "optimizer_type": optimizer_type,
                "lr_scheduler": lr_scheduler,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"主要パラメータの値が不正です: {exc}") from exc
    params["no_metadata"] = bool(no_metadata)
    selected_version_id = optional_int(dataset_version_id)
    if selected_version_id is None:
        version = fetch_one("SELECT id FROM dataset_versions WHERE dataset_id = ? ORDER BY version_no DESC LIMIT 1", (dataset_id,))
        selected_version_id = version["id"] if version else None
    analysis = fetch_one("SELECT * FROM dataset_analysis WHERE dataset_id = ?", (dataset_id,))
    next_status = "prepared_dirty" if job["status"] in {"prepared", "prepared_dirty"} else "draft"
    dirty = 1 if job["status"] != "draft" else 0
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET name = ?, dataset_id = ?, dataset_version_id = ?, preset_id = ?,
                model_family = ?, training_script = ?, base_model_path = ?, vae_path = ?,
                output_name = ?, sample_prompt_template_id = ?, memo = ?,
                params_json = ?, status = ?, config_dirty = ?, command_line = NULL,
                trigger_word_at_creation = ?, trigger_occurrence_count_at_creation = ?,
                trigger_occurrence_rate_at_creation = ?, trigger_consistency_label_at_creation = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                name.strip(),
                dataset_id,
                selected_version_id,
                preset_id,
                preset["model_family"],
                preset["training_script"],
                base_model_path.strip(),
                vae_path.strip() or None,
                output_name.strip() or name.strip().replace(" ", "_"),
                sample_prompt_template_id.strip() or None,
                memo.strip(),
                json.dumps(params, ensure_ascii=False, indent=2),
                next_status,
                dirty,
                dataset["trigger_word"] or "",
                analysis["trigger_word_count"] if analysis else None,
                analysis["trigger_word_rate"] if analysis else None,
                analysis["trigger_consistency_label"] if analysis else None,
                now,
                job_id,
            ),
        )
    return RedirectResponse(f"/jobs/{job_id}?preflight=edited", status_code=303)


@app.post("/jobs/{job_id}/preflight")
def job_preflight(job_id: int, acknowledge_trigger_mismatch: str = Form("")) -> RedirectResponse:
    try:
        validate_job_ready(job_id, acknowledge_trigger_mismatch=acknowledge_trigger_mismatch == "yes")
    except Exception as exc:
        return RedirectResponse(f"/jobs/{job_id}?preflight={quote('NG: ' + str(exc))}", status_code=303)
    return RedirectResponse(f"/jobs/{job_id}?preflight={quote('OK: 実行できます。')}", status_code=303)


@app.post("/jobs/{job_id}/revised-draft")
def job_create_revised_draft(job_id: int) -> RedirectResponse:
    new_id = create_revised_draft(job_id)
    return RedirectResponse(f"/jobs/{new_id}/edit", status_code=303)


@app.post("/jobs/{job_id}/prepare")
def job_prepare(job_id: int) -> RedirectResponse:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in {"draft", "prepared", "prepared_dirty", "failed", "stopped"}:
        raise HTTPException(status_code=400, detail="この状態のジョブではファイル準備を実行できません。")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
    if dataset is None:
        raise HTTPException(status_code=400, detail="Dataset not found")
    files = prepare_job_files(dict(job), dict(dataset))
    with connect() as conn:
        conn.execute("UPDATE training_jobs SET command_line = ?, status = 'prepared', config_dirty = 0, updated_at = datetime('now') WHERE id = ?", (files["command"], job_id))
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/run")
def job_run(job_id: int, acknowledge_trigger_mismatch: str = Form("")) -> RedirectResponse:
    try:
        start_job(job_id, acknowledge_trigger_mismatch=acknowledge_trigger_mismatch == "yes")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/stop")
def job_stop(job_id: int) -> RedirectResponse:
    try:
        stop_job(job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/reimport")
def job_reimport(job_id: int) -> RedirectResponse:
    try:
        collect_job_results(job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/clone")
def job_clone(job_id: int, name: str = Form("")) -> RedirectResponse:
    source = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if source is None:
        raise HTTPException(status_code=404, detail="Job not found")
    clone_name = name.strip() or f"{source['name']}_clone"
    new_id = create_job(
        {
            "name": clone_name,
            "dataset_id": source["dataset_id"],
            "preset_id": source["preset_id"],
            "base_model_path": source["base_model_path"],
            "vae_path": source["vae_path"] or "",
            "output_name": f"{source['output_name']}_clone",
            "memo": f"Cloned from Job #{job_id}",
            "params": json.loads(source["params_json"]),
            "parent_job_id": job_id,
            "sample_prompt_template_id": source["sample_prompt_template_id"] or "",
        }
    )
    copy_sample_prompts(job_id, new_id)
    return RedirectResponse(f"/jobs/{new_id}", status_code=303)


@app.post("/jobs/{job_id}/variant")
def job_variant(job_id: int, variant: str = Form(...)) -> RedirectResponse:
    source = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if source is None:
        raise HTTPException(status_code=404, detail="Job not found")
    params = json.loads(source["params_json"])
    label = apply_variant(params, variant)
    new_id = create_job(
        {
            "name": f"{source['name']}_{variant}",
            "dataset_id": source["dataset_id"],
            "preset_id": source["preset_id"],
            "base_model_path": source["base_model_path"],
            "vae_path": source["vae_path"] or "",
            "output_name": f"{source['output_name']}_{variant}",
            "memo": f"Quick Variant from Job #{job_id}: {label}",
            "params": params,
            "parent_job_id": job_id,
            "sample_prompt_template_id": source["sample_prompt_template_id"] or "",
        }
    )
    copy_sample_prompts(job_id, new_id)
    return RedirectResponse(f"/jobs/{new_id}", status_code=303)


@app.post("/jobs/{job_id}/outputs/{output_id}/select")
def job_select_output(job_id: int, output_id: int) -> RedirectResponse:
    output = fetch_one(
        "SELECT * FROM training_outputs WHERE id = ? AND job_id = ? AND file_type = 'model'",
        (output_id, job_id),
    )
    if output is None:
        raise HTTPException(status_code=404, detail="Output not found")
    with connect() as conn:
        conn.execute("UPDATE training_outputs SET selected = 0 WHERE job_id = ?", (job_id,))
        conn.execute("UPDATE training_outputs SET selected = 1 WHERE id = ?", (output_id,))
        conn.execute(
            """
            UPDATE training_jobs
            SET adopted_epoch = ?, adopted_model_path = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (output["epoch"], output["file_path"], job_id),
        )
    ensure_selected_lora_profile(job_id)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/select-epoch")
def job_select_epoch(job_id: int, epoch: int = Form(...)) -> RedirectResponse:
    output = fetch_one(
        """
        SELECT * FROM training_outputs
        WHERE job_id = ? AND file_type = 'model' AND epoch = ?
        ORDER BY step DESC, id DESC
        LIMIT 1
        """,
        (job_id, epoch),
    )
    if output is None:
        raise HTTPException(status_code=404, detail=f"Output not found for epoch {epoch}")
    with connect() as conn:
        conn.execute("UPDATE training_outputs SET selected = 0 WHERE job_id = ?", (job_id,))
        conn.execute("UPDATE training_outputs SET selected = 1 WHERE id = ?", (output["id"],))
        conn.execute(
            """
            UPDATE training_jobs
            SET adopted_epoch = ?, adopted_model_path = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (output["epoch"], output["file_path"], job_id),
        )
    ensure_selected_lora_profile(job_id)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/samples/{image_id}/review")
def job_review_sample(
    job_id: int,
    image_id: int,
    rating_face: int = Form(0),
    rating_costume: int = Form(0),
    rating_style: int = Form(0),
    rating_stability: int = Form(0),
    rating_overall: int = Form(0),
    rating: int = Form(0),
    strength_label: str = Form(""),
    overfit_level: str = Form(""),
    adoption_label: str = Form(""),
    failure_tags: list[str] = Form([]),
    memo: str = Form(""),
) -> RedirectResponse:
    sample = fetch_one("SELECT * FROM sample_images WHERE id = ? AND job_id = ?", (image_id, job_id))
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample image not found")
    rating_face = clamp_rating(rating_face)
    rating_costume = clamp_rating(rating_costume)
    rating_style = clamp_rating(rating_style)
    rating_stability = clamp_rating(rating_stability)
    rating_overall = clamp_rating(rating_overall if rating_overall is not None else rating)
    with connect() as conn:
        conn.execute(
            """
            UPDATE sample_images
            SET rating = ?, rating_face = ?, rating_costume = ?, rating_style = ?,
                rating_stability = ?, rating_overall = ?, strength_label = ?,
                overfit_level = ?, adoption_label = ?, failure_tags_json = ?,
                rubric_version = ?, memo = ?
            WHERE id = ? AND job_id = ?
            """,
            (
                rating_overall,
                rating_face,
                rating_costume,
                rating_style,
                rating_stability,
                rating_overall,
                clean_choice(strength_label, {key for key, _ in STRENGTH_LABELS}),
                clean_choice(overfit_level, {key for key, _ in OVERFIT_LEVELS}),
                clean_choice(adoption_label, {key for key, _ in ADOPTION_LABELS}),
                json.dumps(clean_failure_tags(failure_tags), ensure_ascii=False),
                RUBRIC_VERSION,
                memo,
                image_id,
                job_id,
            ),
        )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/export-contact-sheet")
def job_export_contact_sheet(job_id: int) -> RedirectResponse:
    path = write_job_contact_sheet(job_id)
    return RedirectResponse(f"/jobs/{job_id}?exported={path}", status_code=303)


@app.post("/jobs/{job_id}/export-selected-lora")
def job_export_selected_lora(job_id: int) -> RedirectResponse:
    ensure_selected_lora_profile(job_id)
    result = export_selected_lora(job_id)
    ensure_selected_lora_profile(job_id)
    return RedirectResponse(f"/jobs/{job_id}?exported={result['directory']}", status_code=303)


@app.post("/jobs/{job_id}/export-validation-pack")
def job_export_validation_pack(job_id: int) -> RedirectResponse:
    try:
        ensure_selected_lora_profile(job_id)
        result = write_validation_pack(job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ensure_selected_lora_profile(job_id)
    return RedirectResponse(f"/jobs/{job_id}?exported={result['directory']}", status_code=303)


@app.post("/jobs/{job_id}/recommendations/regenerate")
def job_regenerate_recommendations(job_id: int) -> RedirectResponse:
    try:
        regenerate_recommendations(job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/recommendations/export")
def job_export_recommendation_report(job_id: int) -> RedirectResponse:
    try:
        path = write_recommendation_report(job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}?exported={path}", status_code=303)


@app.post("/recommendations/{recommendation_id}/create-draft")
def recommendation_create_draft(recommendation_id: int) -> RedirectResponse:
    try:
        job_id = create_draft_job_from_recommendation(recommendation_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/recommendations/{recommendation_id}/dismiss")
def recommendation_dismiss(recommendation_id: int) -> RedirectResponse:
    try:
        job_id = set_recommendation_status(recommendation_id, "dismissed")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/recommendations/{recommendation_id}/accept")
def recommendation_accept(recommendation_id: int) -> RedirectResponse:
    try:
        job_id = set_recommendation_status(recommendation_id, "accepted")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/validation-results")
def job_add_validation_result(
    job_id: int,
    prompt_type: str = Form(...),
    lora_weight: float = Form(...),
    face_score: int = Form(0),
    costume_score: int = Form(0),
    stability_score: int = Form(0),
    flexibility_score: int = Form(0),
    overall_score: int = Form(0),
    memo: str = Form(""),
    image_path: str = Form(""),
) -> RedirectResponse:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    selected_output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    managed_image_path = ""
    image_path_value = normalize_user_path(image_path)
    if image_path_value:
        try:
            managed_image_path = str(unique_copy(Path(image_path_value), validation_images_root() / f"legacy_job_{job_id:06d}" / "images"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO validation_results(
                job_id, selected_output_id, prompt_type, lora_weight,
                face_score, costume_score, stability_score, flexibility_score,
                overall_score, memo, image_path, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                selected_output["id"] if selected_output else None,
                prompt_type.strip(),
                lora_weight,
                clamp_rating(face_score),
                clamp_rating(costume_score),
                clamp_rating(stability_score),
                clamp_rating(flexibility_score),
                clamp_rating(overall_score),
                memo.strip(),
                managed_image_path,
                now,
                now,
            ),
        )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/external-validation/images")
def job_add_validation_image(
    job_id: int,
    image_path: str = Form(...),
    validation_type: str = Form("external"),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    base_model: str = Form(""),
    sampler: str = Form(""),
    steps: str = Form(""),
    cfg_scale: str = Form(""),
    width: str = Form(""),
    height: str = Form(""),
    hires_enabled: str = Form(""),
    hires_scale: str = Form(""),
    lora_weights: str = Form(""),
    seeds: str = Form(""),
    rating_face: int = Form(0),
    rating_costume: int = Form(0),
    rating_style: int = Form(0),
    rating_stability: int = Form(0),
    rating_overall: int = Form(0),
    strength_label: str = Form(""),
    overfit_level: str = Form(""),
    adoption_label: str = Form(""),
    failure_tags: list[str] = Form([]),
    recommended_weight_min: str = Form(""),
    recommended_weight_max: str = Form(""),
    memo: str = Form(""),
) -> RedirectResponse:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    selected_output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        managed_image_path = unique_copy(Path(normalize_user_path(image_path)), validation_images_root() / f"legacy_job_{job_id:06d}" / "images")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO validation_images(
                job_id, selected_output_id, image_path, validation_type, prompt,
                negative_prompt, base_model, sampler, steps, cfg_scale, width, height,
                hires_enabled, hires_scale, lora_weights, seeds,
                rating_face, rating_costume, rating_style, rating_stability, rating_overall,
                strength_label, overfit_level, adoption_label, failure_tags_json, rubric_version,
                recommended_weight_min, recommended_weight_max, memo, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                selected_output["id"] if selected_output else None,
                str(managed_image_path),
                validation_type.strip() or "external",
                prompt.strip(),
                negative_prompt.strip(),
                base_model.strip(),
                sampler.strip(),
                optional_int(steps),
                optional_float(cfg_scale),
                optional_int(width),
                optional_int(height),
                1 if hires_enabled else 0,
                optional_float(hires_scale),
                lora_weights.strip(),
                seeds.strip(),
                clamp_rating(rating_face),
                clamp_rating(rating_costume),
                clamp_rating(rating_style),
                clamp_rating(rating_stability),
                clamp_rating(rating_overall),
                clean_choice(strength_label, {key for key, _ in STRENGTH_LABELS}),
                clean_choice(overfit_level, {key for key, _ in OVERFIT_LEVELS}),
                clean_choice(adoption_label, {key for key, _ in ADOPTION_LABELS}),
                json.dumps(clean_failure_tags(failure_tags), ensure_ascii=False),
                RUBRIC_VERSION,
                optional_float(recommended_weight_min),
                optional_float(recommended_weight_max),
                memo.strip(),
                now,
                now,
            ),
        )
    if selected_output:
        sync_profile_from_validation(job_id)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/external-validation/weight-review")
def job_add_validation_weight_review(
    job_id: int,
    lora_weight: float = Form(...),
    validation_type: str = Form("external"),
    rating_face: int = Form(0),
    rating_costume: int = Form(0),
    rating_style: int = Form(0),
    rating_stability: int = Form(0),
    rating_overall: int = Form(0),
    strength_label: str = Form(""),
    overfit_level: str = Form(""),
    adoption_label: str = Form(""),
    failure_tags: list[str] = Form([]),
    recommended_weight_min: str = Form(""),
    recommended_weight_max: str = Form(""),
    memo: str = Form(""),
) -> RedirectResponse:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    selected_output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO validation_weight_reviews(
                job_id, selected_output_id, lora_weight, validation_type,
                rating_face, rating_costume, rating_style, rating_stability, rating_overall,
                strength_label, overfit_level, adoption_label, failure_tags_json, rubric_version,
                recommended_weight_min, recommended_weight_max, memo, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                selected_output["id"] if selected_output else None,
                lora_weight,
                validation_type.strip() or "external",
                clamp_rating(rating_face),
                clamp_rating(rating_costume),
                clamp_rating(rating_style),
                clamp_rating(rating_stability),
                clamp_rating(rating_overall),
                clean_choice(strength_label, {key for key, _ in STRENGTH_LABELS}),
                clean_choice(overfit_level, {key for key, _ in OVERFIT_LEVELS}),
                clean_choice(adoption_label, {key for key, _ in ADOPTION_LABELS}),
                json.dumps(clean_failure_tags(failure_tags), ensure_ascii=False),
                RUBRIC_VERSION,
                optional_float(recommended_weight_min),
                optional_float(recommended_weight_max),
                memo.strip(),
                now,
                now,
            ),
        )
    if selected_output:
        sync_profile_from_validation(job_id)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.get("/validation-images/{image_id}")
def validation_image_file(image_id: int) -> FileResponse:
    image = fetch_one("SELECT * FROM validation_images WHERE id = ?", (image_id,))
    if image is None:
        raise HTTPException(status_code=404, detail="Validation image not found")
    try:
        path = ensure_allowed_file(image["image_path"], validation_images_root(), "Validation image")
        verify_image_file(path)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Validation image file not found")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Validation image file not found or invalid") from exc
    return FileResponse(path)


@app.get("/validation-presets", response_class=HTMLResponse)
def validation_preset_list(request: Request) -> HTMLResponse:
    return render(request, "validation_presets.html", presets=validation_presets())


@app.post("/jobs/{job_id}/validation-runs")
def job_create_validation_run(
    job_id: int,
    validation_preset_id: str = Form(...),
    base_model: str = Form(""),
    trigger_word: str = Form(""),
    memo: str = Form(""),
) -> RedirectResponse:
    try:
        run_id = create_validation_run(job_id, validation_preset_id, base_model, trigger_word, memo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.get("/validation-runs/{run_id}", response_class=HTMLResponse)
def validation_run_detail(request: Request, run_id: int) -> HTMLResponse:
    try:
        bundle = load_validation_run_bundle(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return render(
        request,
        "validation_run_detail.html",
        **bundle,
        default_project_path=str(settings.ROOT_DIR),
        rubric_options=rubric_options(),
        apply_result=None,
        report_path=None,
    )


@app.post("/validation-runs/{run_id}/export")
def validation_run_export(run_id: int) -> RedirectResponse:
    try:
        write_validation_prompt_pack(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.post("/validation-runs/{run_id}/export-report", response_class=HTMLResponse)
def validation_run_export_report(request: Request, run_id: int) -> HTMLResponse:
    try:
        report_path = write_validation_report(run_id)
        bundle = load_validation_run_bundle(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return render(
        request,
        "validation_run_detail.html",
        **bundle,
        default_project_path=str(settings.ROOT_DIR),
        rubric_options=rubric_options(),
        apply_result=None,
        report_path=report_path,
    )


@app.post("/validation-runs/{run_id}/suggest")
def validation_run_suggest(run_id: int) -> RedirectResponse:
    try:
        persist_suggestion(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.post("/validation-runs/{run_id}/apply", response_class=HTMLResponse)
def validation_run_apply_to_profile(request: Request, run_id: int) -> HTMLResponse:
    try:
        apply_result = apply_suggestion_to_profile(run_id)
        bundle = load_validation_run_bundle(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return render(
        request,
        "validation_run_detail.html",
        **bundle,
        default_project_path=str(settings.ROOT_DIR),
        rubric_options=rubric_options(),
        apply_result=apply_result,
        report_path=None,
    )


@app.post("/validation-runs/{run_id}/status")
def validation_run_update_status(run_id: int, status: str = Form(...)) -> RedirectResponse:
    try:
        update_validation_run_status(run_id, status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.post("/validation-runs/{run_id}/images/individual")
def validation_run_add_individual_image(
    run_id: int,
    image_path: str = Form(...),
    prompt_key: str = Form(...),
    seed: int = Form(...),
    lora_weight: float = Form(...),
    hires_enabled: str = Form(""),
    memo: str = Form(""),
) -> RedirectResponse:
    register_validation_run_image(
        run_id=run_id,
        source_path=image_path,
        image_role="individual",
        prompt_key=prompt_key,
        seed=seed,
        lora_weight=lora_weight,
        hires_enabled=bool(hires_enabled),
        memo=memo,
    )
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.post("/validation-runs/{run_id}/images/grid")
def validation_run_add_grid_image(
    run_id: int,
    image_path: str = Form(...),
    prompt_key: str = Form(""),
    hires_enabled: str = Form(""),
    grid_axis_x: str = Form("weight"),
    grid_axis_y: str = Form("seed"),
    memo: str = Form(""),
) -> RedirectResponse:
    grid_memo = "\n".join(part for part in [memo.strip(), f"grid_axis_x={grid_axis_x}", f"grid_axis_y={grid_axis_y}"] if part)
    register_validation_run_image(
        run_id=run_id,
        source_path=image_path,
        image_role="grid",
        prompt_key=prompt_key,
        seed=None,
        lora_weight=None,
        hires_enabled=bool(hires_enabled),
        memo=grid_memo,
    )
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.post("/validation-runs/{run_id}/images/{image_id}/review")
def validation_run_review_image(
    run_id: int,
    image_id: int,
    rating_face: int = Form(0),
    rating_costume: int = Form(0),
    rating_style: int = Form(0),
    rating_stability: int = Form(0),
    rating_flexibility: int = Form(0),
    rating_overall: int = Form(0),
    strength_label: str = Form(""),
    overfit_level: str = Form(""),
    adoption_label: str = Form(""),
    failure_tags: list[str] = Form([]),
    ignored: str = Form(""),
    memo: str = Form(""),
) -> RedirectResponse:
    image = fetch_one("SELECT * FROM validation_images WHERE id = ? AND validation_run_id = ?", (image_id, run_id))
    if image is None:
        raise HTTPException(status_code=404, detail="Validation image not found")
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE validation_images
            SET rating_face = ?, rating_costume = ?, rating_style = ?,
                rating_stability = ?, rating_flexibility = ?, rating_overall = ?,
                strength_label = ?, overfit_level = ?, adoption_label = ?,
                failure_tags_json = ?, rubric_version = ?, ignored = ?,
                memo = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                clamp_rating(rating_face),
                clamp_rating(rating_costume),
                clamp_rating(rating_style),
                clamp_rating(rating_stability),
                clamp_rating(rating_flexibility),
                clamp_rating(rating_overall),
                clean_choice(strength_label, {key for key, _ in STRENGTH_LABELS}),
                clean_choice(overfit_level, {key for key, _ in OVERFIT_LEVELS}),
                clean_choice(adoption_label, {key for key, _ in ADOPTION_LABELS}),
                json.dumps(clean_failure_tags(failure_tags), ensure_ascii=False),
                RUBRIC_VERSION,
                1 if str(ignored).lower() in {"1", "true", "on", "yes"} else 0,
                memo.strip(),
                now,
                image_id,
            ),
        )
    update_validation_run_counts(run_id)
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


def register_validation_run_image(
    run_id: int,
    source_path: str,
    image_role: str,
    prompt_key: str,
    seed: int | None,
    lora_weight: float | None,
    hires_enabled: bool,
    memo: str,
) -> None:
    bundle = load_validation_run_bundle(run_id)
    run = bundle["run"]
    preset = bundle["preset"]
    if preset is None:
        raise HTTPException(status_code=400, detail="Validation preset is required for run image registration.")
    try:
        managed_path = copy_managed_validation_image(run_id, source_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    condition = None
    if image_role == "individual":
        for row in bundle["conditions"]:
            if (
                row["prompt_key"] == prompt_key
                and int(row["seed"]) == int(seed or 0)
                and float(row["lora_weight"]) == float(lora_weight or 0)
                and bool(row["hires_enabled"]) == hires_enabled
            ):
                condition = dict(row)
                break
    if condition is None:
        condition = {
            "validation_preset_id": preset["id"],
            "prompt_key": prompt_key.strip() or None,
            "seed": seed,
            "lora_weight": lora_weight,
            "width": preset["width"],
            "height": preset["height"],
            "hires_enabled": hires_enabled,
            "hires_scale": preset["hires_scale"],
            "hires_denoising_strength": preset["hires_denoising_strength"],
            "sampler": preset["sampler"],
            "steps": preset["steps"],
            "cfg_scale": preset["cfg_scale"],
            "negative_prompt": preset["negative_prompt"],
            "base_model": run["base_model"] or "",
            "prompt": "",
        }
        if image_role == "individual":
            condition["condition_hash"] = make_condition_hash(condition)
        else:
            condition["condition_hash"] = None
    now = settings_now()
    expected_condition_id = None
    if condition.get("condition_hash"):
        expected_row = fetch_one(
            "SELECT id FROM validation_expected_conditions WHERE validation_run_id = ? AND condition_hash = ?",
            (run_id, condition["condition_hash"]),
        )
        expected_condition_id = expected_row["id"] if expected_row else None
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO validation_images(
                job_id, selected_output_id, expected_condition_id,
                validation_run_id, validation_preset_id,
                prompt_key, seed, lora_weight, image_path, validation_type,
                prompt, negative_prompt, base_model, sampler, steps, cfg_scale,
                width, height, hires_enabled, hires_scale, lora_weights, seeds,
                grid_image_flag, image_role, condition_hash, memo, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'external_run', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run["job_id"],
                run["selected_output_id"],
                expected_condition_id,
                run_id,
                preset["id"],
                condition.get("prompt_key"),
                condition.get("seed"),
                condition.get("lora_weight"),
                managed_path,
                condition.get("prompt") or "",
                condition.get("negative_prompt") or preset["negative_prompt"] or "",
                run["base_model"] or "",
                condition.get("sampler") or preset["sampler"],
                condition.get("steps") or preset["steps"],
                condition.get("cfg_scale") or preset["cfg_scale"],
                condition.get("width") or preset["width"],
                condition.get("height") or preset["height"],
                1 if hires_enabled else 0,
                condition.get("hires_scale") or preset["hires_scale"],
                "" if lora_weight is None else str(lora_weight),
                "" if seed is None else str(seed),
                1 if image_role == "grid" else 0,
                image_role,
                condition.get("condition_hash"),
                memo.strip(),
                now,
                now,
            ),
        )
    update_validation_run_counts(run_id)


@app.get("/lora-library", response_class=HTMLResponse)
def lora_library(request: Request) -> HTMLResponse:
    return render(request, "lora_library.html", profiles=lora_library_profiles())


def lora_library_profiles(limit: int | None = None) -> list[dict[str, Any]]:
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    rows = fetch_all(
        f"""
        SELECT p.*, j.name AS job_name, j.status AS job_status, o.file_size, o.sha256
        FROM selected_lora_profiles p
        LEFT JOIN training_jobs j ON j.id = p.job_id
        LEFT JOIN training_outputs o ON o.id = p.selected_output_id
        ORDER BY p.updated_at DESC, p.id DESC
        {limit_clause}
        """
    )
    profiles = []
    for row in rows:
        profile = dict(row)
        last_run = fetch_one("SELECT * FROM validation_runs WHERE selected_lora_profile_id = ? OR job_id = ? ORDER BY id DESC LIMIT 1", (profile["id"], profile["job_id"]))
        profile["last_validation_run_id"] = last_run["id"] if last_run else None
        profile["last_validation_status"] = last_run["status"] if last_run else None
        if last_run:
            try:
                coverage = load_validation_run_bundle(int(last_run["id"]))["coverage"]
                profile["validation_coverage_rate"] = coverage["coverage_rate"]
                profile["validation_expected_count"] = coverage["expected_image_count"]
                profile["validation_registered_count"] = coverage["registered_condition_count"]
                profile["validation_reviewed_count"] = coverage["reviewed_condition_count"]
                profile["validation_warning"] = validation_profile_warning(profile, last_run, coverage)
            except Exception:
                profile["validation_coverage_rate"] = None
                profile["validation_expected_count"] = None
                profile["validation_registered_count"] = None
                profile["validation_reviewed_count"] = None
                profile["validation_warning"] = "validation status unavailable"
        else:
            profile["validation_coverage_rate"] = None
            profile["validation_expected_count"] = None
            profile["validation_registered_count"] = None
            profile["validation_reviewed_count"] = None
            profile["validation_warning"] = "preset unspecified" if not profile["default_validation_preset_id"] else "validation incomplete"
        profiles.append(profile)
    return profiles


@app.get("/lora-library/{profile_id}/edit", response_class=HTMLResponse)
def lora_profile_edit(request: Request, profile_id: int) -> HTMLResponse:
    profile = fetch_one(
        """
        SELECT p.*, j.name AS job_name, j.status AS job_status
        FROM selected_lora_profiles p
        LEFT JOIN training_jobs j ON j.id = p.job_id
        WHERE p.id = ?
        """,
        (profile_id,),
    )
    if profile is None:
        raise HTTPException(status_code=404, detail="LoRA profile not found")
    weight_reviews = fetch_all("SELECT * FROM validation_weight_reviews WHERE job_id = ? ORDER BY lora_weight, id", (profile["job_id"],))
    validation_images = fetch_all("SELECT * FROM validation_images WHERE job_id = ? ORDER BY created_at DESC, id DESC", (profile["job_id"],))
    validation_runs = fetch_all("SELECT * FROM validation_runs WHERE selected_lora_profile_id = ? OR job_id = ? ORDER BY id DESC", (profile_id, profile["job_id"]))
    reference_sets_rows = fetch_all("SELECT * FROM reference_sets ORDER BY updated_at DESC, id DESC")
    recommendations = list_recommendations(int(profile["job_id"]))
    return render(
        request,
        "lora_profile_edit.html",
        profile=profile,
        weight_reviews=weight_reviews,
        validation_images=validation_images,
        validation_runs=validation_runs,
        validation_presets=validation_presets(),
        reference_sets=reference_sets_rows,
        recommendations=recommendations,
        rubric_options=rubric_options(),
    )


@app.post("/lora-library/{profile_id}/edit")
def lora_profile_update(
    profile_id: int,
    profile_name: str = Form(...),
    trigger_word: str = Form(""),
    base_model: str = Form(""),
    recommended_weight_min: str = Form(""),
    recommended_weight_max: str = Form(""),
    light_weight: str = Form(""),
    strong_weight: str = Form(""),
    default_validation_preset_id: str = Form(""),
    reference_set_id: str = Form(""),
    validation_policy_memo: str = Form(""),
    validation_memo: str = Form(""),
    library_memo: str = Form(""),
) -> RedirectResponse:
    now = settings_now()
    with connect() as conn:
        cur = conn.execute(
            """
            UPDATE selected_lora_profiles
            SET profile_name = ?, trigger_word = ?, base_model = ?,
                recommended_weight_min = ?, recommended_weight_max = ?,
                light_weight = ?, strong_weight = ?,
                default_validation_preset_id = ?, reference_set_id = ?,
                validation_policy_memo = ?,
                validation_memo = ?, library_memo = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                profile_name.strip(),
                trigger_word.strip(),
                base_model.strip(),
                optional_float(recommended_weight_min),
                optional_float(recommended_weight_max),
                optional_float(light_weight),
                optional_float(strong_weight),
                default_validation_preset_id.strip() or None,
                int(reference_set_id) if reference_set_id else None,
                validation_policy_memo.strip(),
                validation_memo.strip(),
                library_memo.strip(),
                now,
                profile_id,
            ),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="LoRA profile not found")
    return RedirectResponse(f"/lora-library/{profile_id}/edit", status_code=303)


@app.post("/lora-library/{profile_id}/validation-runs")
def profile_create_validation_run(
    profile_id: int,
    validation_preset_id: str = Form(...),
    base_model: str = Form(""),
    trigger_word: str = Form(""),
    memo: str = Form(""),
) -> RedirectResponse:
    profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (profile_id,))
    if profile is None:
        raise HTTPException(status_code=404, detail="LoRA profile not found")
    try:
        run_id = create_validation_run(int(profile["job_id"]), validation_preset_id, base_model, trigger_word, memo, profile_id=profile_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.get("/reference-sets", response_class=HTMLResponse)
def reference_sets(request: Request) -> HTMLResponse:
    rows = fetch_all(
        """
        SELECT r.*, d.name AS dataset_name
        FROM reference_sets r
        LEFT JOIN datasets d ON d.id = r.dataset_id
        ORDER BY r.updated_at DESC, r.id DESC
        """
    )
    datasets = fetch_all("SELECT id, name, trigger_word FROM datasets ORDER BY id DESC")
    return render(request, "reference_sets.html", reference_sets=rows, datasets=datasets)


@app.post("/reference-sets")
def reference_set_create(
    name: str = Form(...),
    dataset_id: str = Form(""),
    trigger_word: str = Form(""),
    description: str = Form(""),
    memo: str = Form(""),
) -> RedirectResponse:
    dataset_version_id = None
    if dataset_id:
        version = fetch_one("SELECT id FROM dataset_versions WHERE dataset_id = ? ORDER BY version_no DESC LIMIT 1", (int(dataset_id),))
        dataset_version_id = version["id"] if version else None
    now = settings_now()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO reference_sets(
                name, dataset_id, dataset_version_id, trigger_word,
                description, created_at, updated_at, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name.strip(),
                int(dataset_id) if dataset_id else None,
                dataset_version_id,
                trigger_word.strip(),
                description.strip(),
                now,
                now,
                memo.strip(),
            ),
        )
        set_id = int(cur.lastrowid)
    return RedirectResponse(f"/reference-sets/{set_id}", status_code=303)


@app.get("/reference-sets/{set_id}", response_class=HTMLResponse)
def reference_set_detail(request: Request, set_id: int) -> HTMLResponse:
    row = fetch_one(
        """
        SELECT r.*, d.name AS dataset_name
        FROM reference_sets r
        LEFT JOIN datasets d ON d.id = r.dataset_id
        WHERE r.id = ?
        """,
        (set_id,),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Reference set not found")
    images = fetch_all("SELECT * FROM reference_images WHERE reference_set_id = ? ORDER BY sort_order, id", (set_id,))
    return render(request, "reference_set_detail.html", reference_set=row, images=images, default_project_path=str(settings.ROOT_DIR))


@app.post("/reference-sets/{set_id}/images")
def reference_image_add(
    set_id: int,
    image_path: str = Form(...),
    image_role: str = Form("other"),
    caption: str = Form(""),
    sort_order: int = Form(0),
) -> RedirectResponse:
    row = fetch_one("SELECT * FROM reference_sets WHERE id = ?", (set_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Reference set not found")
    try:
        managed_path = copy_managed_reference_image(set_id, image_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    valid_roles = {"face", "upper_body", "full_body", "expression", "style", "other"}
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO reference_images(reference_set_id, image_path, image_role, caption, sort_order, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (set_id, managed_path, image_role if image_role in valid_roles else "other", caption.strip(), sort_order, now),
        )
        conn.execute("UPDATE reference_sets SET updated_at = ? WHERE id = ?", (now, set_id))
    return RedirectResponse(f"/reference-sets/{set_id}", status_code=303)


@app.get("/reference-images/{image_id}")
def reference_image_file(image_id: int) -> FileResponse:
    image = fetch_one("SELECT * FROM reference_images WHERE id = ?", (image_id,))
    if image is None:
        raise HTTPException(status_code=404, detail="Reference image not found")
    try:
        path = ensure_allowed_file(image["image_path"], reference_images_root(), "Reference image")
        verify_image_file(path)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Reference image file not found")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Reference image file not found or invalid") from exc
    return FileResponse(path)


def validation_profile_warning(profile: dict[str, Any], run: Any, coverage: dict[str, Any]) -> str:
    warnings = []
    if not profile.get("default_validation_preset_id"):
        warnings.append("preset unspecified")
    if run["status"] not in {"reviewed", "completed"} or coverage["missing_condition_count"] > 0:
        warnings.append("validation incomplete")
    if run["validation_level"] == "extended":
        warnings.append("extended validation is supplemental")
    if coverage["registered_condition_count"] == 0:
        warnings.append("no individual validation")
    return ", ".join(warnings) if warnings else "OK"


@app.get("/jobs/{job_id}/samples/{image_id}")
def job_sample_image(job_id: int, image_id: int) -> FileResponse:
    image = fetch_one("SELECT * FROM sample_images WHERE id = ? AND job_id = ?", (image_id, job_id))
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if image is None or job is None:
        raise HTTPException(status_code=404, detail="Sample image not found")
    image_path = Path(image["image_path"]).resolve()
    samples_dir = (Path(job["run_dir"]) / "samples").resolve()
    if samples_dir not in image_path.parents and image_path != samples_dir:
        raise HTTPException(status_code=403, detail="Sample image path is not allowed")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Sample image file not found")
    return FileResponse(image_path)


def copy_sample_prompts(source_job_id: int, target_job_id: int) -> None:
    rows = fetch_all("SELECT * FROM sample_prompts WHERE job_id = ? ORDER BY sort_order, id", (source_job_id,))
    if not rows:
        return
    now = settings_now()
    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO sample_prompts(
                job_id, name, prompt, negative_prompt, width, height,
                seed, cfg_scale, steps, sort_order, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    target_job_id,
                    row["name"],
                    row["prompt"],
                    row["negative_prompt"],
                    row["width"],
                    row["height"],
                    row["seed"],
                    row["cfg_scale"],
                    row["steps"],
                    row["sort_order"],
                    now,
                )
                for row in rows
            ],
        )


def create_revised_draft(source_job_id: int) -> int:
    source = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (source_job_id,))
    if source is None:
        raise HTTPException(status_code=404, detail="Job not found")
    new_id = create_job(
        {
            "name": f"{source['name']} revised",
            "dataset_id": source["dataset_id"],
            "preset_id": source["preset_id"],
            "base_model_path": source["base_model_path"],
            "vae_path": source["vae_path"] or "",
            "output_name": f"{source['output_name']}_revised",
            "memo": f"派生ドラフト from ジョブ #{source_job_id}",
            "params": json.loads(source["params_json"]),
            "parent_job_id": source_job_id,
            "sample_prompt_template_id": source["sample_prompt_template_id"] or "",
        }
    )
    copy_sample_prompts(source_job_id, new_id)
    with connect() as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET dataset_version_id = ?, config_dirty = 0, updated_at = ?
            WHERE id = ?
            """,
            (source["dataset_version_id"], settings_now(), new_id),
        )
    return new_id


def update_params_from_form(params: dict[str, Any], values: dict[str, str]) -> None:
    integer_keys = {
        "max_train_epochs",
        "repeats",
        "train_batch_size",
        "network_dim",
        "network_alpha",
        "save_every_n_epochs",
        "sample_every_n_epochs",
    }
    float_keys = {"learning_rate", "unet_lr"}
    for key, raw_value in values.items():
        value = raw_value.strip() if isinstance(raw_value, str) else raw_value
        if value in ("", None):
            continue
        if key in integer_keys:
            params[key] = int(value)
        elif key in float_keys:
            params[key] = float(value)
        else:
            params[key] = value


def job_trigger_status(job: Any, dataset: Any, sample_prompts: list[Any]) -> dict[str, Any]:
    if dataset is None:
        return {
            "label": "UNKNOWN",
            "message": "Dataset is unavailable.",
            "snapshot_message": "snapshot unavailable",
            "sample_prompt_uses_trigger": None,
            "sample_prompt_message": "Sample prompts are unavailable.",
        }
    analysis = fetch_one("SELECT * FROM dataset_analysis WHERE dataset_id = ?", (dataset["id"],))
    trigger_word = dataset["trigger_word"] or ""
    current_label = analysis["trigger_consistency_label"] if analysis else "UNKNOWN"
    current_count = analysis["trigger_word_count"] if analysis else None
    current_rate = analysis["trigger_word_rate"] if analysis else None
    current_message = analysis["trigger_consistency_message"] if analysis else "No current dataset analysis."
    snapshot_label = job["trigger_consistency_label_at_creation"] if "trigger_consistency_label_at_creation" in job.keys() else None
    snapshot_word = job["trigger_word_at_creation"] if "trigger_word_at_creation" in job.keys() else None
    snapshot_count = job["trigger_occurrence_count_at_creation"] if "trigger_occurrence_count_at_creation" in job.keys() else None
    snapshot_rate = job["trigger_occurrence_rate_at_creation"] if "trigger_occurrence_rate_at_creation" in job.keys() else None
    if snapshot_label:
        snapshot_message = (
            f"created with trigger '{snapshot_word or '-'}': "
            f"{snapshot_label} ({snapshot_count if snapshot_count is not None else '-'})"
        )
    else:
        snapshot_message = f"snapshot unavailable; current dataset trigger consistency is {current_label}"
    prompts = [row["prompt"] for row in sample_prompts]
    if not prompts:
        sample_prompt_uses_trigger = None
        sample_prompt_message = "No sample prompts have been prepared yet."
    else:
        sample_prompt_uses_trigger = bool(trigger_word and any(trigger_word in prompt for prompt in prompts))
        sample_prompt_message = f"sample prompt uses trigger_word: {'yes' if sample_prompt_uses_trigger else 'no'}"
        if sample_prompt_uses_trigger and current_label == "ERROR":
            sample_prompt_message += "; sample prompt uses trigger_word, but captions do not contain it. Evaluation may be invalid."
        elif not sample_prompt_uses_trigger and trigger_word:
            sample_prompt_message += "; sample prompts do not use the current dataset trigger_word."
    return {
        "label": snapshot_label or current_label,
        "message": current_message,
        "trigger_word": trigger_word,
        "current_count": current_count,
        "current_rate": current_rate,
        "current_label": current_label,
        "snapshot_message": snapshot_message,
        "sample_prompt_uses_trigger": sample_prompt_uses_trigger,
        "sample_prompt_message": sample_prompt_message,
    }


def apply_variant(params: dict[str, Any], variant: str) -> str:
    if variant == "lower_lr":
        params["learning_rate"] = halve_float(params.get("learning_rate"))
        params["unet_lr"] = halve_float(params.get("unet_lr"))
        return "Lower LR"
    if variant == "higher_lr":
        params["learning_rate"] = min(0.0002, multiply_float(params.get("learning_rate"), 1.5))
        params["unet_lr"] = min(0.0002, multiply_float(params.get("unet_lr"), 1.5))
        return "Higher LR"
    if variant == "lower_dim":
        params["network_dim"] = max(1, int(params.get("network_dim") or 1) // 2)
        params["network_alpha"] = max(1, int(params.get("network_alpha") or 1) // 2)
        return "Lower Dim"
    if variant == "higher_dim":
        params["network_dim"] = int(params.get("network_dim") or 1) * 2
        params["network_alpha"] = int(params.get("network_alpha") or 1) * 2
        return "Higher Dim"
    if variant == "more_epoch":
        params["max_train_epochs"] = int(params.get("max_train_epochs") or 1) + 2
        return "More Epoch"
    if variant == "fewer_epoch":
        params["max_train_epochs"] = max(1, int(params.get("max_train_epochs") or 1) - 1)
        return "Fewer Epoch"
    raise HTTPException(status_code=400, detail=f"Unknown variant: {variant}")


def halve_float(value: Any) -> float:
    return multiply_float(value, 0.5)


def multiply_float(value: Any, factor: float) -> float:
    return float(value or 0) * factor


COMPARE_PARAM_KEYS = [
    "optimizer_type",
    "lr_scheduler",
    "learning_rate",
    "unet_lr",
    "text_encoder_lr",
    "text_encoder_lr1",
    "text_encoder_lr2",
    "network_dim",
    "network_alpha",
    "train_batch_size",
    "repeats",
    "max_train_epochs",
    "resolution",
    "save_every_n_epochs",
    "sample_every_n_epochs",
    "save_every_n_steps",
    "sample_every_n_steps",
]

COMPARE_METRIC_KEYS = [
    "expected_total_steps",
    "actual_max_step",
    "initial_loss",
    "final_loss",
    "min_loss",
    "loss_drop_rate",
    "loss_volatility",
    "spike_count",
    "late_stage_slope",
    "health_label",
    "health_message",
    "step_consistency_label",
]


@app.get("/compare", response_class=HTMLResponse)
def compare_epochs(
    request: Request,
    job_a: int | None = None,
    job_b: int | None = None,
    job_ids: list[int] | None = Query(None),
    exported: str | None = None,
) -> HTMLResponse:
    if job_ids and len(job_ids) >= 2:
        job_a, job_b = job_ids[0], job_ids[1]
    elif job_a and job_ids:
        job_b = next((candidate for candidate in job_ids if candidate != job_a), None)
    jobs = fetch_all("SELECT id, name, status, adopted_epoch FROM training_jobs ORDER BY id DESC")
    if not job_a or not job_b:
        return render(request, "compare_epochs.html", jobs=jobs, comparison=None, exported=exported, selected_job_a=job_a)
    comparison = build_job_comparison(job_a, job_b)
    return render(request, "compare_epochs.html", jobs=jobs, comparison=comparison, exported=exported, selected_job_a=job_a)


@app.post("/compare/export")
def export_comparison(job_a: int = Form(...), job_b: int = Form(...)) -> RedirectResponse:
    comparison = build_job_comparison(job_a, job_b)
    path = write_comparison_markdown(comparison)
    return RedirectResponse(f"/compare?job_a={job_a}&job_b={job_b}&exported={path}", status_code=303)


@app.post("/compare/export-contact-sheet")
def export_compare_contact_sheet(job_a: int = Form(...), job_b: int = Form(...)) -> RedirectResponse:
    comparison = build_job_comparison(job_a, job_b)
    path = write_compare_contact_sheet(comparison)
    return RedirectResponse(f"/compare?job_a={job_a}&job_b={job_b}&exported={path}", status_code=303)


def build_job_comparison(job_a: int, job_b: int) -> dict[str, Any]:
    left = load_compare_job(job_a)
    right = load_compare_job(job_b)
    return {
        "left": left,
        "right": right,
        "warnings": compare_warnings(left, right),
        "param_rows": build_param_rows(left["params"], right["params"]),
        "metric_rows": build_metric_rows(left, right),
        "epoch_rows": build_epoch_compare_rows(left, right),
        "sample_groups": build_compare_sample_groups(left, right),
    }


def load_compare_job(job_id: int) -> dict[str, Any]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
    dataset_version = fetch_one("SELECT * FROM dataset_versions WHERE id = ?", (job["dataset_version_id"],)) if job["dataset_version_id"] else None
    preset = fetch_one("SELECT * FROM presets WHERE id = ?", (job["preset_id"],))
    summary = fetch_one("SELECT * FROM training_metric_summaries WHERE job_id = ?", (job_id,))
    epoch_summaries = fetch_all("SELECT * FROM training_epoch_summaries WHERE job_id = ? ORDER BY epoch", (job_id,))
    metrics = fetch_all("SELECT * FROM training_metrics WHERE job_id = ? ORDER BY step, id", (job_id,))
    outputs = fetch_all("SELECT * FROM training_outputs WHERE job_id = ? ORDER BY epoch, step, id", (job_id,))
    samples = fetch_all("SELECT * FROM sample_images WHERE job_id = ? ORDER BY epoch, step, id", (job_id,))
    sample_prompts = fetch_all("SELECT * FROM sample_prompts WHERE job_id = ? ORDER BY sort_order, id", (job_id,))
    selected_output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    validation_image = fetch_one(
        """
        SELECT * FROM validation_images
        WHERE job_id = ? AND image_role = 'individual'
        ORDER BY validation_run_id DESC, updated_at DESC, id DESC LIMIT 1
        """,
        (job_id,),
    )
    params = json.loads(job["params_json"])
    decorated_epochs = decorate_epoch_summaries(epoch_summaries, outputs, samples)
    return {
        "job": job,
        "dataset": dataset,
        "dataset_version": dataset_version,
        "preset": preset,
        "summary": summary,
        "epoch_summaries": decorated_epochs,
        "epoch_visual_summaries": build_epoch_visual_summaries(decorated_epochs, samples),
        "metrics": metrics,
        "outputs": outputs,
        "samples": samples,
        "sample_prompts": sample_prompts,
        "selected_output": selected_output,
        "validation_image": validation_image,
        "params": params,
        "loss_chart": build_loss_chart(metrics),
        "health_details": health_details(summary, len(metrics)),
    }


def decorate_epoch_summaries(epoch_rows: list[Any], outputs: list[Any], samples: list[Any]) -> list[dict[str, Any]]:
    output_by_epoch = outputs_by_epoch(outputs)
    sample_counts: dict[int, int] = {}
    for sample in samples:
        if sample["epoch"] is not None:
            sample_counts[int(sample["epoch"])] = sample_counts.get(int(sample["epoch"]), 0) + 1
    decorated = []
    for row in epoch_rows:
        item = dict(row)
        output = output_by_epoch.get(row["epoch"])
        item["sample_count"] = sample_counts.get(row["epoch"], 0)
        item["output_file"] = Path(output["file_path"]).name if output else "-"
        item["output_selected"] = bool(output["selected"]) if output else False
        decorated.append(item)
    return decorated


def outputs_by_epoch(outputs: list[Any]) -> dict[int, Any]:
    output_by_epoch: dict[int, Any] = {}
    for output in outputs:
        if output["epoch"] is None:
            continue
        epoch = int(output["epoch"])
        existing = output_by_epoch.get(epoch)
        if existing is None or output["selected"] or (
            not existing["selected"] and Path(output["file_path"]).stem.lower().endswith("final")
        ):
            output_by_epoch[epoch] = output
    return output_by_epoch


def clamp_rating(value: Any) -> int:
    try:
        return max(0, min(5, int(value or 0)))
    except (TypeError, ValueError):
        return 0


def rubric_options() -> dict[str, Any]:
    return {
        "version": RUBRIC_VERSION,
        "strength_labels": STRENGTH_LABELS,
        "overfit_levels": OVERFIT_LEVELS,
        "adoption_labels": ADOPTION_LABELS,
        "failure_tags": FAILURE_TAGS,
    }


def clean_choice(value: str, allowed: set[str]) -> str:
    value = (value or "").strip()
    return value if value in allowed else ""


def clean_failure_tags(values: list[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    allowed = set(FAILURE_TAGS)
    result = []
    for value in values:
        if value in allowed and value not in result:
            result.append(value)
    return result


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def rating_value(sample: Any, key: str) -> int | None:
    value = sample[key] if key in sample.keys() else None
    if value is None and key == "rating_overall":
        value = sample["rating"] if "rating" in sample.keys() else None
    return int(value) if value is not None else None


def average_rating(samples: list[Any], key: str) -> float | None:
    values = [rating_value(sample, key) for sample in samples]
    values = [value for value in values if value is not None and value > 0]
    if not values:
        return None
    return sum(values) / len(values)


def validation_pack_path(job_id: int) -> str | None:
    path = settings.EXPORTS_DIR / "validation_packs" / f"job_{job_id:06d}"
    return str(path) if path.exists() else None


def build_validation_summary(results: list[Any]) -> dict[str, Any]:
    rows = [dict(row) for row in results]
    return {
        "best_weight_by_overall": best_group_value(rows, "lora_weight", "overall_score"),
        "best_weight_by_stability": best_group_value(rows, "lora_weight", "stability_score"),
        "by_weight": average_score_rows(rows, "lora_weight"),
        "by_prompt_type": average_score_rows(rows, "prompt_type"),
    }


def best_group_value(rows: list[dict[str, Any]], group_key: str, score_key: str) -> Any:
    averages = average_score_rows(rows, group_key)
    scored = [row for row in averages if row.get(score_key) is not None]
    if not scored:
        return None
    best = max(scored, key=lambda row: row[score_key])
    return best["key"]


def average_score_rows(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        key = row.get(group_key)
        if key is None or key == "":
            continue
        grouped.setdefault(key, []).append(row)
    result = []
    for key in sorted(grouped, key=lambda item: str(item)):
        items = grouped[key]
        result.append(
            {
                "key": key,
                "count": len(items),
                "face_score": average_int_field(items, "face_score"),
                "costume_score": average_int_field(items, "costume_score"),
                "stability_score": average_int_field(items, "stability_score"),
                "flexibility_score": average_int_field(items, "flexibility_score"),
                "overall_score": average_int_field(items, "overall_score"),
            }
        )
    return result


def average_int_field(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [int(row[key]) for row in rows if row.get(key) is not None and int(row[key]) > 0]
    if not values:
        return None
    return sum(values) / len(values)


def ensure_selected_lora_profile(job_id: int) -> Any:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    if job is None or output is None:
        return None
    existing = fetch_one(
        "SELECT * FROM selected_lora_profiles WHERE job_id = ? AND selected_output_id = ?",
        (job_id, output["id"]),
    )
    if existing:
        sync_profile_selected_fields(job, output, int(existing["id"]))
        return fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (existing["id"],))
    now = settings_now()
    profile_name = f"Job #{job_id} {job['name']} epoch {output['epoch'] or job['adopted_epoch'] or '-'}"
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO selected_lora_profiles(
                job_id, selected_output_id, profile_name, trigger_word, selected_epoch,
                selected_model_path, exported_model_path, base_model,
                recommended_weight_min, recommended_weight_max, light_weight, strong_weight,
                validation_memo, library_memo, default_validation_preset_id,
                validation_policy_memo, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, '', '', ?, ?, ?, ?)
            """,
            (
                job_id,
                output["id"],
                profile_name,
                job["trigger_word_at_creation"] or "",
                output["epoch"] if output["epoch"] is not None else job["adopted_epoch"],
                output["file_path"],
                exported_model_path(job_id),
                base_model_label(job["base_model_path"]),
                "standard_validation_v1",
                "通常比較はHiresなしのStandard Validationを基準にする。HiresありはExtended Validationで最終見栄え確認として扱う。",
                now,
                now,
            ),
        )
        profile_id = int(cur.lastrowid)
    sync_profile_from_validation(job_id)
    return fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (profile_id,))


def sync_profile_selected_fields(job: Any, output: Any, profile_id: int) -> None:
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE selected_lora_profiles
            SET selected_epoch = ?, selected_model_path = ?, exported_model_path = ?,
                trigger_word = COALESCE(NULLIF(trigger_word, ''), ?),
                base_model = COALESCE(NULLIF(base_model, ''), ?),
                updated_at = ?
            WHERE id = ?
            """,
            (
                output["epoch"] if output["epoch"] is not None else job["adopted_epoch"],
                output["file_path"],
                exported_model_path(int(job["id"])),
                job["trigger_word_at_creation"] or "",
                base_model_label(job["base_model_path"]),
                now,
                profile_id,
            ),
        )


def sync_profile_from_validation(job_id: int) -> None:
    profile = ensure_selected_lora_profile_without_sync(job_id)
    if profile is None:
        return
    range_row = fetch_one(
        """
        SELECT recommended_weight_min, recommended_weight_max
        FROM validation_weight_reviews
        WHERE job_id = ? AND recommended_weight_min IS NOT NULL AND recommended_weight_max IS NOT NULL
        ORDER BY updated_at DESC, id DESC LIMIT 1
        """,
        (job_id,),
    ) or fetch_one(
        """
        SELECT recommended_weight_min, recommended_weight_max
        FROM validation_images
        WHERE job_id = ? AND recommended_weight_min IS NOT NULL AND recommended_weight_max IS NOT NULL
        ORDER BY updated_at DESC, id DESC LIMIT 1
        """,
        (job_id,),
    )
    low_review = fetch_one("SELECT lora_weight FROM validation_weight_reviews WHERE job_id = ? ORDER BY lora_weight ASC LIMIT 1", (job_id,))
    high_review = fetch_one("SELECT lora_weight FROM validation_weight_reviews WHERE job_id = ? ORDER BY lora_weight DESC LIMIT 1", (job_id,))
    memo_row = fetch_one(
        """
        SELECT memo FROM validation_weight_reviews
        WHERE job_id = ? AND memo IS NOT NULL AND memo != ''
        ORDER BY updated_at DESC, id DESC LIMIT 1
        """,
        (job_id,),
    ) or fetch_one(
        """
        SELECT memo FROM validation_images
        WHERE job_id = ? AND memo IS NOT NULL AND memo != ''
        ORDER BY updated_at DESC, id DESC LIMIT 1
        """,
        (job_id,),
    )
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE selected_lora_profiles
            SET recommended_weight_min = COALESCE(?, recommended_weight_min),
                recommended_weight_max = COALESCE(?, recommended_weight_max),
                light_weight = COALESCE(light_weight, ?),
                strong_weight = COALESCE(strong_weight, ?),
                validation_memo = COALESCE(NULLIF(validation_memo, ''), ?),
                updated_at = ?
            WHERE id = ?
            """,
            (
                range_row["recommended_weight_min"] if range_row else None,
                range_row["recommended_weight_max"] if range_row else None,
                low_review["lora_weight"] if low_review else None,
                high_review["lora_weight"] if high_review else None,
                memo_row["memo"] if memo_row else "",
                now,
                profile["id"],
            ),
        )


def ensure_selected_lora_profile_without_sync(job_id: int) -> Any:
    output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    if output is None:
        return None
    profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE job_id = ? AND selected_output_id = ?", (job_id, output["id"]))
    if profile:
        return profile
    return ensure_selected_lora_profile(job_id)


def exported_model_path(job_id: int) -> str:
    export_dir = settings.EXPORTS_DIR / "selected_loras" / f"job_{job_id:06d}"
    if not export_dir.exists():
        return ""
    files = sorted(export_dir.glob("*.safetensors"))
    return str(files[0]) if files else ""


def base_model_label(path_value: str) -> str:
    if not path_value:
        return ""
    return Path(path_value).stem


def build_epoch_visual_summaries(epoch_rows: list[dict[str, Any]], samples: list[Any]) -> list[dict[str, Any]]:
    samples_by_epoch: dict[int, list[Any]] = {}
    for sample in samples:
        if sample["epoch"] is not None:
            samples_by_epoch.setdefault(int(sample["epoch"]), []).append(sample)
    rows = []
    for epoch in sorted(set(samples_by_epoch) | {int(row["epoch"]) for row in epoch_rows if row.get("epoch") is not None}):
        loss_row = next((row for row in epoch_rows if int(row["epoch"]) == epoch), {})
        epoch_samples = samples_by_epoch.get(epoch, [])
        has_rating = any(sample_has_rating(sample) for sample in epoch_samples)
        rows.append(
            {
                "epoch": epoch,
                "avg_loss": loss_row.get("avg_loss"),
                "moving_avg_final_loss": loss_row.get("moving_avg_final_loss"),
                "sample_count": len(epoch_samples),
                "avg_face": average_rating(epoch_samples, "rating_face"),
                "avg_costume": average_rating(epoch_samples, "rating_costume"),
                "avg_style": average_rating(epoch_samples, "rating_style"),
                "avg_stability": average_rating(epoch_samples, "rating_stability"),
                "avg_overall": average_rating(epoch_samples, "rating_overall"),
                "memo_count": sum(1 for sample in epoch_samples if sample["memo"]),
                "has_rating": has_rating,
                "output_file": loss_row.get("output_file") or "-",
                "output_selected": bool(loss_row.get("output_selected")),
            }
        )
    return rows


def sample_has_rating(sample: Any) -> bool:
    keys = ["rating", "rating_face", "rating_costume", "rating_style", "rating_stability", "rating_overall"]
    return any((rating_value(sample, key) or 0) > 0 for key in keys)


def compare_warnings(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    warnings = []
    left_version = left["job"]["dataset_version_id"]
    right_version = right["job"]["dataset_version_id"]
    if left_version != right_version:
        warnings.append("WARNING: These jobs were trained with different or unavailable dataset versions.")
    if left["job"]["trigger_word_at_creation"] != right["job"]["trigger_word_at_creation"]:
        warnings.append("WARNING: These jobs used different trigger_word values at creation.")
    validation_keys = [
        ("validation_preset_id", "validation presets"),
        ("base_model", "base models"),
        ("width", "image widths"),
        ("height", "image heights"),
        ("hires_enabled", "Hires settings"),
        ("seed", "seeds"),
        ("prompt_key", "prompt keys"),
        ("sampler", "samplers"),
        ("steps", "step counts"),
        ("cfg_scale", "CFG scales"),
    ]
    left_image = left.get("validation_image")
    right_image = right.get("validation_image")
    if left_image and right_image:
        for key, label in validation_keys:
            if left_image[key] != right_image[key]:
                warnings.append(f"WARNING: Latest validation images use different {label}.")
        if left_image["hires_enabled"] != right_image["hires_enabled"]:
            warnings.append("WARNING: Hires images should not be directly compared with non-Hires baseline images.")
    return warnings


def build_param_rows(left_params: dict[str, Any], right_params: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    emphasized = {"learning_rate", "unet_lr", "text_encoder_lr", "text_encoder_lr1", "text_encoder_lr2", "network_dim", "network_alpha", "max_train_epochs", "repeats"}
    for key in COMPARE_PARAM_KEYS:
        left_value = left_params.get(key)
        right_value = right_params.get(key)
        rows.append({"key": key, "left": render_value(left_value), "right": render_value(right_value), "changed": left_value != right_value, "emphasized": key in emphasized})
    return rows


def build_epoch_compare_rows(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_rows = {row["epoch"]: row for row in left["epoch_summaries"]}
    right_rows = {row["epoch"]: row for row in right["epoch_summaries"]}
    rows = []
    for epoch in sorted(set(left_rows) | set(right_rows)):
        left_row = left_rows.get(epoch)
        right_row = right_rows.get(epoch)
        rows.append(
            {
                "epoch": epoch,
                "left_avg_loss": render_value(left_row["avg_loss"] if left_row else None),
                "right_avg_loss": render_value(right_row["avg_loss"] if right_row else None),
                "left_ma_final": render_value(left_row["moving_avg_final_loss"] if left_row else None),
                "right_ma_final": render_value(right_row["moving_avg_final_loss"] if right_row else None),
                "left_samples": left_row["sample_count"] if left_row else 0,
                "right_samples": right_row["sample_count"] if right_row else 0,
            }
        )
    return rows


def build_metric_rows(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in COMPARE_METRIC_KEYS:
        left_value = metric_value(left, key)
        right_value = metric_value(right, key)
        rows.append({"key": key, "left": render_value(left_value), "right": render_value(right_value), "changed": left_value != right_value})
    return rows


def metric_value(bundle: dict[str, Any], key: str) -> Any:
    if key in {"expected_total_steps", "actual_max_step", "step_consistency_label"}:
        return bundle["job"][key]
    summary = bundle["summary"]
    return summary[key] if summary and key in summary.keys() else None


def render_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def health_details(summary: Any, metric_count: int) -> dict[str, Any]:
    threshold = max(2, metric_count // 3) if metric_count else None
    summary_data = {}
    if summary and summary["summary_json"]:
        try:
            summary_data = json.loads(summary["summary_json"])
        except json.JSONDecodeError:
            summary_data = {}
    raw_label = summary["raw_loss_label"] if summary else None
    smoothed_label = summary["smoothed_loss_label"] if summary else None
    epoch_label = summary["epoch_trend_label"] if summary else None
    supplemental_note = ""
    if (raw_label == "WARNING" or smoothed_label == "WARNING") and epoch_label == "OK":
        supplemental_note = "step単位では揺れがありますが、epoch単位では破綻していません。画像評価が良ければ採用候補です。"
    return {
        "spike_threshold": threshold,
        "spike_rule": "raw: current > previous * 1.5 / adjusted: rawかつdelta > 0.02かつrolling medianから乖離",
        "quality_note": "Loss health is a training-log health check, not an image quality score.",
        "adoption_note": "WARNING can still be usable when sample images look better.",
        "spike_count": summary["spike_count"] if summary else None,
        "raw_spike_count": summary_data.get("raw_spike_count", summary["spike_count"] if summary else None),
        "adjusted_spike_count": summary_data.get("adjusted_spike_count", summary["spike_count"] if summary else None),
        "spike_abs_delta_threshold": summary_data.get("spike_abs_delta_threshold", 0.02),
        "spike_median_ratio_threshold": summary_data.get("spike_median_ratio_threshold", 1.35),
        "spike_median_delta_threshold": summary_data.get("spike_median_delta_threshold", 0.02),
        "loss_volatility": summary["loss_volatility"] if summary else None,
        "late_stage_slope": summary["late_stage_slope"] if summary else None,
        "min_loss_step": summary["min_loss_step"] if summary else None,
        "final_loss": summary["final_loss"] if summary else None,
        "health_message": summary["health_message"] if summary else None,
        "supplemental_note": supplemental_note,
    }


def build_compare_sample_groups(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_prompts = {row["sort_order"]: row for row in left["sample_prompts"]}
    right_prompts = {row["sort_order"]: row for row in right["sample_prompts"]}
    groups = []
    for order in sorted(set(left_prompts) | set(right_prompts)):
        left_prompt = left_prompts.get(order)
        right_prompt = right_prompts.get(order)
        prompt = left_prompt or right_prompt
        left_samples = samples_for_prompt(left["samples"], left_prompt)
        right_samples = samples_for_prompt(right["samples"], right_prompt)
        positions = sorted(set(left_samples) | set(right_samples))
        groups.append(
            {
                "title": prompt["name"] if prompt else f"Prompt {order}",
                "prompt": prompt["prompt"] if prompt else "",
                "rows": [
                    {
                        "label": f"epoch {position}" if isinstance(position, int) else str(position),
                        "left": left_samples.get(position),
                        "right": right_samples.get(position),
                    }
                    for position in positions
                ],
            }
        )
    return groups


def samples_for_prompt(samples: list[Any], prompt: Any) -> dict[Any, dict[str, Any]]:
    if prompt is None:
        return {}
    rows = [dict(sample) for sample in samples if sample["prompt_id"] == prompt["id"]]
    result = {}
    for index, sample in enumerate(rows, start=1):
        sample["filename"] = Path(sample["image_path"]).name
        key = sample["epoch"] if sample["epoch"] is not None else sample["step"] if sample["step"] is not None else index
        result[key] = sample
    return result


def write_comparison_markdown(comparison: dict[str, Any]) -> str:
    left = comparison["left"]
    right = comparison["right"]
    left_id = int(left["job"]["id"])
    right_id = int(right["job"]["id"])
    output_dir = settings.RUNS_DIR / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"compare_job_{left_id:06d}_job_{right_id:06d}.md"
    lines = [
        f"# Compare Job #{left_id} vs Job #{right_id}",
        "",
        "## Jobs",
        f"- Job #{left_id}: {left['job']['name']} / {left['preset']['name'] if left['preset'] else '-'}",
        f"  - Parent Job: #{left['job']['parent_job_id']}" if left["job"]["parent_job_id"] else "  - Parent Job: -",
        f"  - Dataset Version: {left['job']['dataset_version_id'] or 'snapshot unavailable'}",
        f"  - Trigger at creation: {left['job']['trigger_word_at_creation'] or 'snapshot unavailable'}",
        f"- Job #{right_id}: {right['job']['name']} / {right['preset']['name'] if right['preset'] else '-'}",
        f"  - Parent Job: #{right['job']['parent_job_id']}" if right["job"]["parent_job_id"] else "  - Parent Job: -",
        f"  - Dataset Version: {right['job']['dataset_version_id'] or 'snapshot unavailable'}",
        f"  - Trigger at creation: {right['job']['trigger_word_at_creation'] or 'snapshot unavailable'}",
        "",
        "## Warnings",
    ]
    if comparison["warnings"]:
        lines.extend(f"- {warning}" for warning in comparison["warnings"])
    else:
        lines.append("- No dataset version or trigger mismatch warnings.")
    lines.extend([
        "",
        "## Parameter Differences",
    ])
    for row in comparison["param_rows"]:
        marker = "changed" if row["changed"] else "same"
        lines.append(f"- {row['key']}: {row['left']} | {row['right']} ({marker})")
    lines.extend(["", "## Metrics"])
    for row in comparison["metric_rows"]:
        marker = "changed" if row["changed"] else "same"
        lines.append(f"- {row['key']}: {row['left']} | {row['right']} ({marker})")
    lines.extend(["", "## Epoch Loss Comparison"])
    for row in comparison["epoch_rows"]:
        lines.append(
            f"- epoch {row['epoch']}: avg_loss {row['left_avg_loss']} | {row['right_avg_loss']}; "
            f"moving_avg_final {row['left_ma_final']} | {row['right_ma_final']}; "
            f"samples {row['left_samples']} | {row['right_samples']}"
        )
    lines.extend(
        [
            "",
            "## Selected LoRA",
            f"- Job #{left_id}: {left['job']['adopted_model_path'] or '-'}",
            f"- Job #{right_id}: {right['job']['adopted_model_path'] or '-'}",
            "",
            "## Human Notes",
        ]
    )
    for bundle in (left, right):
        lines.append(f"### Job #{bundle['job']['id']}")
        for sample in bundle["samples"]:
            if sample["rating"] is not None or sample["memo"]:
                lines.append(f"- {Path(sample['image_path']).name}: rating={sample['rating'] or 0}, memo={sample['memo'] or ''}")
    lines.extend(
        [
            "",
            "## Health Note",
            "Loss health is a training-log health check, not an image quality score. WARNING can still be usable when sample images look better.",
            "",
            "## Sample Files",
        ]
    )
    for group in comparison["sample_groups"]:
        lines.append(f"### {group['title']}")
        for row in group["rows"]:
            left_name = row["left"]["filename"] if row["left"] else "-"
            right_name = row["right"]["filename"] if row["right"] else "-"
            lines.append(f"- {row['label']}: {left_name} | {right_name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def parse_json_list(value: Any) -> list[str]:
    if not value:
        return []
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def group_samples(sample_prompts: list[Any], samples: list[Any]) -> list[dict[str, Any]]:
    prompt_map = {row["id"]: row for row in sample_prompts}
    groups: dict[int, dict[str, Any]] = {}
    fallback_key = 0
    for sample in samples:
        sample_item = dict(sample)
        sample_item["filename"] = Path(sample["image_path"]).name
        sample_item["failure_tags"] = parse_json_list(sample_item.get("failure_tags_json"))
        key = sample["prompt_id"] or fallback_key
        prompt = prompt_map.get(sample["prompt_id"])
        groups.setdefault(
            key,
            {
                "prompt": prompt,
                "title": prompt["name"] if prompt else "Unmatched prompt",
                "samples": [],
            },
        )
        groups[key]["samples"].append(sample_item)
    for prompt in sample_prompts:
        groups.setdefault(prompt["id"], {"prompt": prompt, "title": prompt["name"], "samples": []})
    for group in groups.values():
        group["samples"].sort(key=lambda item: (
            item["epoch"] if item["epoch"] is not None else 999999,
            item["step"] if item["step"] is not None else 999999,
            item["created_at"],
            item["id"],
        ))
    return sorted(groups.values(), key=lambda group: group["prompt"]["sort_order"] if group["prompt"] else 999999)


def build_loss_chart(metrics: list[Any]) -> dict[str, Any] | None:
    loss_rows = [row for row in metrics if row["loss"] is not None and row["step"] is not None]
    if len(loss_rows) < 2:
        return None
    width = 720
    height = 220
    pad = 28
    steps = [int(row["step"]) for row in loss_rows]
    losses = [float(row["loss"]) for row in loss_rows]
    min_step, max_step = min(steps), max(steps)
    min_loss, max_loss = min(losses), max(losses)
    if min_loss == max_loss:
        min_loss -= 0.001
        max_loss += 0.001

    def point(step: int, value: float) -> tuple[float, float]:
        x = pad + (step - min_step) / max(1, max_step - min_step) * (width - pad * 2)
        y = height - pad - (value - min_loss) / max(0.000001, max_loss - min_loss) * (height - pad * 2)
        return round(x, 2), round(y, 2)

    raw_points = " ".join(f"{x},{y}" for x, y in (point(step, value) for step, value in zip(steps, losses)))
    averages = moving_average(losses, 10)
    ma_points = " ".join(f"{x},{y}" for x, y in (point(step, value) for step, value in zip(steps, averages)))
    return {
        "width": width,
        "height": height,
        "raw_points": raw_points,
        "ma_points": ma_points,
        "min_loss": min_loss,
        "max_loss": max_loss,
        "min_step": min_step,
        "max_step": max_step,
    }


def moving_average(values: list[float], window: int) -> list[float]:
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(sum(values[start:index + 1]) / (index - start + 1))
    return result
