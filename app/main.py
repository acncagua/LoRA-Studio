from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app import settings
from app.app_version import app_version_info
from app.db import connect, create_dataset_version, create_job, fetch_all, fetch_one, import_latest_environment, init_db, insert_dataset, upsert_dataset_analysis
from app.services.command_builder import prepare_job_files
from app.services.embedding_service import (
    active_embedding_model,
    cleanup_preview as embedding_cleanup_preview,
    create_embedding_job,
    embedding_cache_size,
    embedding_coverage,
    format_bytes as embedding_format_bytes,
    latest_embedding_jobs,
    load_embedding_settings,
    provider_preflight,
    read_embedding_log,
    reconcile_stale_embedding_jobs,
    start_embedding_job,
    stop_embedding_job,
    update_embedding_settings,
)
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
from app.services.machine_review import (
    create_machine_review_job,
    epoch_machine_summary,
    latest_machine_review_jobs,
    load_machine_review_settings,
    machine_review_readiness,
    reference_set_readiness,
    run_machine_review,
    score_map_for_samples,
    score_map_for_validation,
    scores_for_job,
    scores_for_validation_run,
    start_machine_review_job,
    stop_machine_review_job,
    update_machine_review_settings,
    validation_weight_summary,
)
from app.services.output_collector import collect_job_results
from app.services.operation_monitor import (
    embedding_monitor,
    machine_review_monitor,
    review_session_monitor,
    running_training_monitor,
    training_progress_from_log,
    validation_generation_monitor,
)
from app.services.recommendations import create_draft_job_from_recommendation, list_recommendations, regenerate_recommendations, set_recommendation_status, write_recommendation_report
from app.services.review_candidates import ensure_epoch_candidates, regenerate_epoch_candidates
from app.services.review_sessions import (
    ensure_candidate_review_plan,
    prepare_review_generation,
    reconcile_stale_review_sessions,
    review_session_summary,
    start_review_preparation,
    stop_review_preparation,
    write_review_matrix,
)
from app.services.reference_sets import (
    ROLE_LABELS,
    REFERENCE_TYPE_LABELS,
    add_reference_image,
    archive_reference_set,
    create_reference_set,
    delete_reference_image,
    export_reference_artifacts,
    dataset_image_candidates,
    reference_detail,
    reference_set_rows,
    set_project_default,
    update_reference_image,
)
from app.services.storage_cleanup import (
    cleanup_outputs,
    cleanup_project_outputs,
    cleanup_samples,
    exported_selected_preview,
    failed_outputs_preview,
    format_bytes as storage_format_bytes,
    project_cleanup_preview,
    project_storage_summary,
    purge_trash,
    sample_cleanup_preview,
    storage_usage,
    unselected_model_preview,
)
from app.services.training_runner import read_log_tail, reconcile_stale_running_jobs, start_job, stop_job, validate_job_ready
from app.services.validation_generation import (
    build_epoch_cross_matrix_html,
    count_generated_images,
    generation_view_state,
    import_generated_images,
    prepare_validation_generation,
    reconcile_stale_validation_generations,
    start_validation_assist_sequence,
    start_validation_generation,
    start_validation_generation_sequence,
    stop_validation_generation,
    validation_assist_log_state,
    validation_run_dir as generation_validation_run_dir,
    validation_generation_log_tail,
    write_validation_matrix,
)
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
DEFAULT_JOB_PRESET_ID = "sdxl_2d_face_adamw8bit_standard"
INTEGRATION_SMOKE_PRESET_ID = "integration_smoke_sdxl"
SMOKE_STEP_LIMIT_KEYS = {"max_train_steps", "save_every_n_steps", "sample_every_n_steps"}
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

PRESET_DISPLAY_NAMES = {
    "integration_smoke_sdxl": "結合確認 - SDXL",
    "sdxl_2d_face_pilot_3epoch": "SDXL 2D顔 - 軽量確認 3 Epoch",
    "sdxl_2d_face_pilot_generalize_3epoch": "SDXL 2D顔 - 汎化寄り軽量確認 3 Epoch",
    "sdxl_2d_face_standard_6epoch": "SDXL 2D顔 - 標準 6 Epoch",
}

VALIDATION_PRESET_DISPLAY_NAMES = {
    "quick_validation_v1": "クイック検証 v1",
    "standard_validation_v1": "標準検証 v1",
    "extended_validation_v1": "拡張検証 v1",
}

VALIDATION_LEVEL_LABELS = {
    "quick": "クイック",
    "standard": "標準",
    "extended": "拡張",
}

STATUS_TEXT_REPLACEMENTS = {
    "Pilot": "軽量確認",
    "Standard": "標準",
    "Validation": "検証",
}

VALIDATION_RUN_STATUS_LABELS = {
    "planned": "予定",
    "images_registered": "画像登録済み",
    "partially_reviewed": "一部レビュー済み",
    "reviewed": "レビュー済み",
    "completed": "完了",
    "archived": "アーカイブ",
}

MACHINE_LABELS = {
    "primary_candidate": "第一候補",
    "secondary_candidate": "候補",
    "check_manually": "要確認",
    "possible_overfit": "過学習注意",
    "low_confidence": "低信頼",
    "unavailable": "利用不可",
    "high": "高",
    "medium": "中",
    "low": "低",
    "unknown": "不明",
}

templates = Environment(
    loader=FileSystemLoader(settings.ROOT_DIR / "app" / "templates"),
    autoescape=select_autoescape(["html", "xml"]),
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    reconcile_stale_running_jobs()
    reconcile_stale_validation_generations()
    reconcile_stale_embedding_jobs()
    reconcile_stale_review_sessions()


def render(request: Request, template: str, **context: Any) -> HTMLResponse:
    tpl = templates.get_template(template)
    context.setdefault("app_name", settings.APP_NAME)
    context.setdefault("request", request)
    context.setdefault("sd_scripts_release_tag", settings.SD_SCRIPTS_RELEASE_TAG)
    context.setdefault("sd_scripts_release_commit", settings.SD_SCRIPTS_RELEASE_COMMIT)
    context.setdefault("app_meta", current_app_meta())
    context.setdefault("display_preset_name", display_preset_name)
    context.setdefault("display_validation_preset_name", display_validation_preset_name)
    context.setdefault("display_validation_level", display_validation_level)
    context.setdefault("display_validation_status", display_validation_status)
    context.setdefault("validation_run_status_options", list(VALIDATION_RUN_STATUS_LABELS.items()))
    context.setdefault("display_status_text", display_status_text)
    context.setdefault("display_machine_label", display_machine_label)
    context.setdefault("static_asset_version", static_asset_version())
    return HTMLResponse(tpl.render(**context))


def static_asset_version() -> str:
    paths = [
        settings.ROOT_DIR / "app" / "static" / "css" / "app.css",
        settings.ROOT_DIR / "app" / "static" / "js" / "app.js",
    ]
    mtimes = []
    for path in paths:
        try:
            mtimes.append(str(int(path.stat().st_mtime)))
        except OSError:
            mtimes.append("0")
    return "-".join(mtimes)


def add_query_param(url: str, **params: Any) -> str:
    target = url or "/settings/embeddings"
    parts = urlsplit(target)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    for key, value in params.items():
        if value is not None:
            query[key] = str(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def safe_local_redirect(target: str, fallback: str) -> str:
    if not target:
        return fallback
    parts = urlsplit(target)
    if parts.scheme or parts.netloc:
        return fallback
    if not parts.path.startswith("/") or parts.path.startswith("//"):
        return fallback
    return urlunsplit(("", "", parts.path, parts.query, parts.fragment))


def display_preset_name(preset: Any) -> str:
    preset_id = ""
    name = ""
    if isinstance(preset, str):
        preset_id = preset
    elif preset:
        preset_id = preset["id"] if "id" in preset.keys() else ""
        name = preset["name"] if "name" in preset.keys() else ""
    if preset_id in PRESET_DISPLAY_NAMES:
        return PRESET_DISPLAY_NAMES[preset_id]
    text = name or preset_id or "-"
    replacements = [
        ("Integration Smoke", "結合確認"),
        ("Pilot Generalize", "汎化寄り軽量確認"),
        ("Pilot", "軽量確認"),
        ("Standard", "標準"),
        ("Generalize", "汎化寄り"),
        ("Strong", "強め"),
        ("Soft", "弱め"),
        ("Medium Dataset", "中規模Dataset"),
        ("Small Dataset", "少数Dataset"),
        ("2D Face", "2D顔"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def wants_json_response(request: Request) -> bool:
    return request.headers.get("x-requested-with") == "fetch" or "application/json" in request.headers.get("accept", "")


def display_validation_preset_name(preset: Any) -> str:
    preset_id = ""
    name = ""
    if isinstance(preset, str):
        preset_id = preset
    elif preset:
        preset_id = preset["id"] if "id" in preset.keys() else ""
        name = preset["name"] if "name" in preset.keys() else ""
    return VALIDATION_PRESET_DISPLAY_NAMES.get(preset_id, name or preset_id or "-")


def display_validation_level(level: str | None) -> str:
    return VALIDATION_LEVEL_LABELS.get(level or "", level or "-")


def display_validation_status(status: str | None) -> str:
    return VALIDATION_RUN_STATUS_LABELS.get(status or "", status or "-")


def display_status_text(text: str | None) -> str:
    value = text or "-"
    for old, new in STATUS_TEXT_REPLACEMENTS.items():
        value = value.replace(old, new)
    return value


def display_machine_label(value: str | None) -> str:
    return MACHINE_LABELS.get(value or "", value or "-")


def running_embedding_job(job_type: str, target_id: int) -> dict[str, Any] | None:
    row = fetch_one(
        """
        SELECT * FROM embedding_jobs
        WHERE status = 'running' AND job_type = ? AND target_id = ?
        ORDER BY id DESC LIMIT 1
        """,
        (job_type, target_id),
    )
    return dict(row) if row else None


def running_machine_review_job(target_type: str, target_id: int) -> dict[str, Any] | None:
    row = fetch_one(
        """
        SELECT * FROM machine_review_jobs
        WHERE status = 'running' AND target_type = ? AND target_id = ?
        ORDER BY id DESC LIMIT 1
        """,
        (target_type, target_id),
    )
    return dict(row) if row else None


def job_operation_monitor(job: Any, review_preparation: dict[str, Any]) -> dict[str, Any] | None:
    training = running_training_monitor(dict(job))
    if training:
        return training
    session = review_preparation.get("session") if review_preparation else None
    if session:
        monitor = review_session_monitor(dict(session))
        if monitor:
            return monitor
    embedding = running_embedding_job("training_job_samples", int(job["id"]))
    if embedding:
        return embedding_monitor(embedding)
    machine = running_machine_review_job("training_job_samples", int(job["id"]))
    if machine:
        return machine_review_monitor(machine)
    return None


def review_session_operation_monitor(session: Any) -> dict[str, Any] | None:
    monitor = review_session_monitor(dict(session))
    if monitor:
        return monitor
    session_id = int(session["id"])
    embedding = running_embedding_job("review_session", session_id)
    if embedding:
        return embedding_monitor(embedding)
    machine = running_machine_review_job("review_session_images", session_id)
    if machine:
        return machine_review_monitor(machine)
    return None


def validation_run_operation_monitor(run_id: int, generation: Any | None) -> dict[str, Any] | None:
    generation_dict = dict(generation) if generation else None
    monitor = validation_generation_monitor(generation_dict, run_id)
    if monitor:
        return monitor
    embedding = running_embedding_job("validation_run", run_id)
    if embedding:
        return embedding_monitor(embedding)
    machine = running_machine_review_job("validation_run_images", run_id)
    if machine:
        return machine_review_monitor(machine)
    return None


def job_filter_where(view: str) -> str:
    if view == "all":
        return "1 = 1"
    if view == "archived":
        return "j.archived_at IS NOT NULL AND j.deleted_at IS NULL"
    if view == "deleted":
        return "j.deleted_at IS NOT NULL"
    if view in {"draft", "running", "completed", "failed"}:
        return f"j.status = '{view}' AND j.archived_at IS NULL AND j.deleted_at IS NULL"
    if view == "prepared":
        return "j.status IN ('prepared', 'prepared_dirty') AND j.archived_at IS NULL AND j.deleted_at IS NULL"
    return "j.archived_at IS NULL AND j.deleted_at IS NULL"


def project_filter_where(view: str) -> str:
    if view == "all":
        return "1 = 1"
    if view == "archived":
        return "p.archived_at IS NOT NULL AND p.deleted_at IS NULL"
    if view == "deleted":
        return "p.deleted_at IS NOT NULL"
    return "p.archived_at IS NULL AND p.deleted_at IS NULL"


def validate_sample_prompt_template_id(template_id: str) -> str:
    value = template_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", value):
        raise HTTPException(status_code=400, detail="IDは英数字、ハイフン、アンダースコアで3〜80文字にしてください。")
    return value


def validate_sample_prompts_json(prompts_json: str) -> str:
    try:
        prompts = json.loads(prompts_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"prompts_jsonがJSONとして読めません: {exc}") from exc
    if not isinstance(prompts, list) or not prompts:
        raise HTTPException(status_code=400, detail="prompts_jsonは1件以上の配列にしてください。")
    required = {"name", "prompt"}
    for index, prompt in enumerate(prompts, start=1):
        if not isinstance(prompt, dict):
            raise HTTPException(status_code=400, detail=f"{index}件目がオブジェクトではありません。")
        missing = [key for key in required if not str(prompt.get(key, "")).strip()]
        if missing:
            raise HTTPException(status_code=400, detail=f"{index}件目に必須項目がありません: {', '.join(missing)}")
    return json.dumps(prompts, ensure_ascii=False, indent=2)


def default_sample_prompts_json() -> str:
    prompts = [
        {
            "name": "basic_face",
            "prompt": "{trigger_word}, 1girl, upper body, looking at viewer, simple background",
            "negative_prompt": "low quality, worst quality, bad anatomy, bad hands",
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "steps": 28,
            "cfg_scale": 7,
        },
        {
            "name": "full_body",
            "prompt": "{trigger_word}, 1girl, full body, standing, outdoors",
            "negative_prompt": "low quality, worst quality, bad anatomy, bad hands",
            "width": 1024,
            "height": 1024,
            "seed": 43,
            "steps": 28,
            "cfg_scale": 7,
        },
    ]
    return json.dumps(prompts, ensure_ascii=False, indent=2)


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


def row_value(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key] if key in row.keys() else default
    except AttributeError:
        return getattr(row, key, default)


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
DELETABLE_JOB_STATUSES = {"draft", "prepared", "prepared_dirty"}
JOB_FILTERS = [
    ("active", "有効"),
    ("draft", "下書き"),
    ("prepared", "準備済み"),
    ("running", "実行中"),
    ("completed", "完了"),
    ("failed", "失敗"),
    ("archived", "アーカイブ"),
    ("deleted", "削除済み"),
    ("all", "すべて"),
]


def job_action_state(job: Any, selected_output: Any | None = None) -> dict[str, Any]:
    status = job["status"]
    dirty = bool(job["config_dirty"] if "config_dirty" in job.keys() else 0)
    archived = bool(row_value(job, "archived_at"))
    deleted = bool(row_value(job, "deleted_at"))
    return {
        "edit": status in EDITABLE_JOB_STATUSES and not archived and not deleted,
        "prepare": status in {"draft", "prepared", "prepared_dirty", "failed", "stopped"} and not archived and not deleted,
        "preflight": status == "prepared" and not dirty,
        "run": status in {"prepared", "failed", "stopped"} and not dirty and not archived and not deleted,
        "stop": status == "running",
        "reimport": status in {"completed", "failed", "stopped"},
        "select": status == "completed",
        "clone": status != "running" and not deleted,
        "revised": status in {"running", "completed"} and not deleted,
        "compare": status == "completed",
        "export": status == "completed" and selected_output is not None,
        "contact_sheet": status == "completed",
        "archive": status != "running" and not archived and not deleted,
        "restore": archived and not deleted,
        "delete": status in DELETABLE_JOB_STATUSES and not deleted,
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
        return "次にやること: 採用LoRA出力または検証Runを作成してください。"
    if status == "completed":
        return "次にやること: sample画像を評価し、採用LoRAを選択してください。"
    if status == "failed":
        return "次にやること: train.log末尾を確認し、設定を修正して再度ファイル準備を実行してください。"
    if status == "stopped":
        return "次にやること: 必要なら設定を見直し、ファイル準備後に実行してください。"
    return "次にやること: ジョブ詳細の操作パネルから次の操作を選んでください。"


def preset_training_length_label(params: dict[str, Any]) -> str:
    steps = params.get("max_train_steps")
    if steps not in (None, ""):
        return f"{steps} step"
    epochs = params.get("max_train_epochs")
    if epochs not in (None, ""):
        return f"{epochs} epoch"
    return "epoch/step未指定"


def preset_option_rows(rows: list[Any]) -> list[dict[str, Any]]:
    options = []
    for row in rows:
        item = dict(row)
        try:
            params = json.loads(item.get("params_json") or "{}")
        except json.JSONDecodeError:
            params = {}
        item["training_length_label"] = preset_training_length_label(params)
        item["option_label"] = f"{item['model_family']} / {item['name']} / {item['training_length_label']}"
        options.append(item)
    return options


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


PROJECT_STATUS_LABELS = {
    "draft": "下書き",
    "training": "学習中",
    "reviewing": "評価中",
    "selected": "採用済み",
    "validating": "検証中",
    "completed": "完了",
    "archived": "アーカイブ",
}


PILOT_PRESET_ID = "sdxl_2d_face_pilot_3epoch"
STANDARD_PRESET_ID = "sdxl_2d_face_standard_6epoch"


def create_lora_project(data: dict[str, Any]) -> int:
    now = settings_now()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO lora_projects(
                name, description, dataset_id, current_dataset_version_id, trigger_word,
                base_model_path, status, created_at, updated_at, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["name"],
                data.get("description") or "",
                data.get("dataset_id"),
                data.get("dataset_version_id"),
                data.get("trigger_word") or "",
                data.get("base_model_path") or "",
                data.get("status") or "draft",
                now,
                now,
                data.get("memo") or "",
            ),
        )
        return int(cur.lastrowid)


def recent_projects(limit: int = 8, view: str = "active") -> list[dict[str, Any]]:
    where = project_filter_where(view)
    rows = fetch_all(
        f"""
        SELECT p.*, d.name AS dataset_name, sj.name AS selected_job_name,
               so.epoch AS selected_epoch, vr.status AS latest_validation_status
        FROM lora_projects p
        LEFT JOIN datasets d ON d.id = p.dataset_id
        LEFT JOIN training_jobs sj ON sj.id = p.selected_job_id
        LEFT JOIN training_outputs so ON so.id = p.selected_output_id
        LEFT JOIN validation_runs vr ON vr.id = (
            SELECT id FROM validation_runs WHERE project_id = p.id ORDER BY id DESC LIMIT 1
        )
        WHERE {where}
        ORDER BY p.updated_at DESC, p.id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [dict(row) for row in rows]


def project_workflow_status(project: Any, jobs: list[dict[str, Any]], validation_runs: list[Any], recommendations: list[Any]) -> list[dict[str, str]]:
    has_dataset = bool(project["dataset_id"])
    has_pilot = any("pilot" in (job.get("preset_name") or job.get("name") or "").lower() for job in jobs)
    has_standard = any("standard" in (job.get("preset_name") or job.get("name") or "").lower() for job in jobs)
    has_completed = any(job.get("status") == "completed" for job in jobs)
    has_selected = bool(project["selected_job_id"] or project["selected_output_id"])
    has_export = bool(project["selected_lora_profile_id"])
    has_validation = bool(validation_runs)
    has_recommendation = bool(recommendations)
    return [
        {"name": "データセット整備", "status": "OK" if has_dataset else "未設定"},
        {"name": "軽量確認ジョブ（任意安全確認）", "status": "OK" if has_pilot else "任意"},
        {"name": "標準ジョブ", "status": "OK" if has_standard else "未実施"},
        {"name": "Sample評価", "status": "OK" if has_completed else "未実施"},
        {"name": "採用LoRA", "status": "OK" if has_selected else "未選択"},
        {"name": "Export", "status": "OK" if has_export else "未出力"},
        {"name": "検証Run", "status": "OK" if has_validation else "未実施"},
        {"name": "Recommendation", "status": "OK" if has_recommendation else "未生成"},
    ]


def project_next_action(project: Any, jobs: list[Any], review_sessions: list[Any], validation_runs: list[Any], recommendations: list[Any]) -> dict[str, str]:
    """Return the highest-priority next action for the project workspace."""
    running_job = next((job for job in jobs if job["status"] == "running"), None)
    if running_job:
        return {
            "label": "学習ジョブが実行中です",
            "description": f"#{running_job['id']} {running_job['name']} の進捗とログを確認してください。",
            "href": f"/jobs/{running_job['id']}#technical-log",
            "button": "実行中ジョブを開く",
        }
    running_session = next((session for session in review_sessions if session["status"] == "running"), None)
    if running_session:
        return {
            "label": "Review Sessionが実行中です",
            "description": f"#{running_session['id']} の画像生成、Embedding、Machine Reviewの完了を待ってください。",
            "href": f"/review-sessions/{running_session['id']}",
            "button": "Review Sessionを開く",
        }
    ready_session = next((session for session in review_sessions if session["matrix_path"] and not project["selected_output_id"]), None)
    if ready_session:
        return {
            "label": "候補epochを選んでください",
            "description": f"Review Matrix #{ready_session['id']} は準備済みです。画像を比較して採用epochを選択してください。",
            "href": f"/review-sessions/{ready_session['id']}",
            "button": "Review Sessionを開く",
        }
    incomplete_run = next(
        (
            run
            for run in validation_runs
            if (run["expected_image_count"] or 0) > (run["actual_image_count"] or 0)
            or (run["actual_image_count"] or 0) > (run["reviewed_count"] or 0)
        ),
        None,
    )
    if incomplete_run:
        return {
            "label": "検証Runが未完了です",
            "description": f"検証Run #{incomplete_run['id']} の画像登録またはレビューが残っています。",
            "href": f"/validation-runs/{incomplete_run['id']}",
            "button": "検証Runを開く",
        }
    draft_job = next((job for job in jobs if job["status"] in {"draft", "prepared", "prepared_dirty"}), None)
    if draft_job:
        return {
            "label": "未実行の学習ジョブがあります",
            "description": f"#{draft_job['id']} {draft_job['name']} を準備または実行してください。",
            "href": f"/jobs/{draft_job['id']}",
            "button": "学習ジョブを開く",
        }
    unresolved_rec = next((rec for rec in recommendations if rec["status"] not in {"accepted", "dismissed", "job_created"}), None)
    if unresolved_rec:
        return {
            "label": "次回実験提案を確認してください",
            "description": unresolved_rec["title"] or "未処理の提案があります。",
            "href": f"/jobs/{unresolved_rec['source_job_id']}#recommendations" if unresolved_rec["source_job_id"] else f"/projects/{project['id']}#recommendations",
            "button": "提案を見る",
        }
    if project["selected_lora_profile_id"]:
        return {
            "label": "採用LoRAプロファイルを確認できます",
            "description": "採用LoRA、推奨weight、外部検証結果を確認してください。",
            "href": f"/lora-library/{project['selected_lora_profile_id']}/edit",
            "button": "LoRAプロファイルを開く",
        }
    if project["selected_job_id"]:
        return {
            "label": "採用学習ジョブを確認できます",
            "description": "採用済みの学習ジョブからExportや検証Run作成へ進めます。",
            "href": f"/jobs/{project['selected_job_id']}",
            "button": "採用ジョブを開く",
        }
    if jobs:
        return {
            "label": "最新の学習ジョブを確認してください",
            "description": f"#{jobs[0]['id']} {jobs[0]['name']} の状態を確認してください。",
            "href": f"/jobs/{jobs[0]['id']}",
            "button": "最新ジョブを開く",
        }
    return {
        "label": "学習ジョブを作成してください",
        "description": "このProjectに最初の学習ジョブを追加します。",
        "href": f"/jobs/new?project_id={project['id']}",
        "button": "学習ジョブを追加",
    }


def decorate_validation_runs_for_job(validation_runs: list[Any]) -> list[dict[str, Any]]:
    decorated: list[dict[str, Any]] = []
    for row in validation_runs:
        item = dict(row)
        run_id = int(item["id"])
        output_dir = Path(item["output_dir"]) if item.get("output_dir") else generation_validation_run_dir(run_id) / "generation" / "images"
        log_path = Path(item["log_path"]) if item.get("log_path") else generation_validation_run_dir(run_id) / "generation" / "generation.log"
        item["generation_file_count"] = count_generated_images(output_dir)
        item["generation_log_preview"] = validation_generation_log_tail(run_id, max_lines=5)
        item["generation_log_updated_at"] = datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if log_path.exists() else ""
        item["generation_log_size"] = log_path.stat().st_size if log_path.exists() else 0
        assist_log = validation_assist_log_state(run_id, max_lines=6)
        item["assist_log_exists"] = assist_log["exists"]
        item["assist_log_preview"] = assist_log["log_preview"]
        item["assist_log_updated_at"] = assist_log["log_updated_at"]
        item["assist_log_size"] = assist_log["log_size"]
        decorated.append(item)
    return decorated


def current_running_validation_generation() -> dict[str, Any] | None:
    row = fetch_one(
        """
        SELECT vg.*, vr.job_id, vr.name AS validation_run_name, o.epoch AS selected_epoch
        FROM validation_generation_runs vg
        LEFT JOIN validation_runs vr ON vr.id = vg.validation_run_id
        LEFT JOIN training_outputs o ON o.id = vr.selected_output_id
        WHERE vg.status = 'running'
        ORDER BY vg.id DESC
        LIMIT 1
        """
    )
    return dict(row) if row else None


def pilot_recommendation(project: Any) -> dict[str, Any]:
    dataset_version_id = project["current_dataset_version_id"] if project else None
    base_model_path = project["base_model_path"] if project else ""
    model_family_row = fetch_one("SELECT model_family FROM datasets WHERE id = ?", (project["dataset_id"],)) if project and project["dataset_id"] else None
    model_family = model_family_row["model_family"] if model_family_row else ""
    reasons: list[str] = []

    real_completed = fetch_one(
        """
        SELECT COUNT(*) AS count FROM training_jobs
        WHERE status = 'completed'
          AND COALESCE(return_code, 0) = 0
          AND COALESCE(preset_id, '') != 'integration_smoke_sdxl'
          AND COALESCE(output_model_count, 0) > 0
        """
    )["count"]
    if int(real_completed or 0) == 0:
        return {
            "label": "REQUIRED",
            "reason": "sd-scripts環境で実モデル学習の完走実績がまだありません。まず軽量確認で動作確認してください。",
            "button_prefix": "推奨: ",
        }

    standard_completed = fetch_one(
        """
        SELECT COUNT(*) AS count FROM training_jobs
        WHERE status = 'completed'
          AND COALESCE(return_code, 0) = 0
          AND preset_id = ?
          AND dataset_version_id IS ?
          AND base_model_path = ?
          AND model_family = ?
        """,
        (STANDARD_PRESET_ID, dataset_version_id, base_model_path, model_family),
    )["count"]
    if int(standard_completed or 0) > 0:
        return {
            "label": "SKIPPABLE",
            "reason": "同じdataset version / base model / model familyで標準ジョブの完走実績があります。事前確認がOKなら標準学習へ直行して構いません。",
            "button_prefix": "",
        }

    project_pilot_completed = fetch_one(
        """
        SELECT COUNT(*) AS count FROM training_jobs
        WHERE project_id = ? AND status = 'completed' AND COALESCE(return_code, 0) = 0
          AND preset_id IN (?, ?)
        """,
        (project["id"], PILOT_PRESET_ID, "sdxl_2d_face_pilot_generalize_3epoch"),
    )["count"] if project else 0
    if int(project_pilot_completed or 0) > 0:
        return {
            "label": "OPTIONAL",
            "reason": "同じProject内で軽量確認が完走済みです。必要なら追加確認として軽量確認を挟めますが、標準学習へ進んでもよい状態です。",
            "button_prefix": "",
        }

    base_completed = fetch_one(
        "SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'completed' AND COALESCE(return_code, 0) = 0 AND base_model_path = ?",
        (base_model_path,),
    )["count"]
    if int(base_completed or 0) == 0:
        reasons.append("このbase modelで完了済み学習ジョブがありません。")

    version_completed = fetch_one(
        "SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'completed' AND COALESCE(return_code, 0) = 0 AND dataset_version_id IS ?",
        (dataset_version_id,),
    )["count"]
    if dataset_version_id and int(version_completed or 0) == 0:
        reasons.append("このdataset versionで完了済み学習ジョブがありません。")

    preset_completed = fetch_one(
        "SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'completed' AND COALESCE(return_code, 0) = 0 AND preset_id = ?",
        (STANDARD_PRESET_ID,),
    )["count"]
    if int(preset_completed or 0) == 0:
        reasons.append("標準6Epoch系統の完了済み学習ジョブがありません。")

    if reasons:
        return {
            "label": "RECOMMENDED",
            "reason": " ".join(reasons) + " 軽量確認で軽く確認してから標準学習へ進むことを推奨します。",
            "button_prefix": "推奨: ",
        }
    return {
        "label": "OPTIONAL",
        "reason": "この環境では実モデル学習が完走済みです。Dataset/triggerを確認し、事前確認がOKなら軽量確認は任意です。",
        "button_prefix": "",
    }


def should_reset_params_to_selected_preset(current_preset_id: str | None, selected_preset_id: str, params: dict[str, Any]) -> bool:
    if current_preset_id != selected_preset_id:
        return True
    if selected_preset_id != INTEGRATION_SMOKE_PRESET_ID and any(key in params for key in SMOKE_STEP_LIMIT_KEYS):
        return True
    return False


def params_for_selected_preset(current_preset_id: str | None, selected_preset: Any, submitted_params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    selected_preset_id = selected_preset["id"]
    if should_reset_params_to_selected_preset(current_preset_id, selected_preset_id, submitted_params):
        return json.loads(selected_preset["params_json"]), True
    return submitted_params, False


def create_project_preset_job(project_id: int, preset_id: str, skip_reason: str = "", source_job_id: int | None = None) -> int:
    project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (project_id,))
    preset = fetch_one("SELECT * FROM presets WHERE id = ?", (preset_id,))
    if project is None or preset is None:
        raise HTTPException(status_code=404, detail="Project or preset not found")
    source = None
    if source_job_id:
        source = fetch_one("SELECT * FROM training_jobs WHERE id = ? AND project_id = ?", (source_job_id, project_id))
        if source is None:
            raise HTTPException(status_code=400, detail="source_job_id is not in this Project.")
    if source is None:
        source = fetch_one("SELECT * FROM training_jobs WHERE project_id = ? ORDER BY id DESC LIMIT 1", (project_id,))
    name_suffix = "軽量確認3Epoch" if preset_id == PILOT_PRESET_ID else "標準6Epoch"
    job_id = create_job(
        {
            "project_id": project_id,
            "name": f"{project['name']} {name_suffix}",
            "dataset_id": project["dataset_id"],
            "preset_id": preset_id,
            "base_model_path": project["base_model_path"],
            "vae_path": source["vae_path"] if source and source["vae_path"] else "",
            "output_name": f"{project['name']}_{preset_id}".replace(" ", "_"),
            "memo": skip_reason.strip() if skip_reason.strip() else f"Project #{project_id}から作成",
            "parent_job_id": source["id"] if source else None,
            "sample_prompt_template_id": source["sample_prompt_template_id"] if source and source["sample_prompt_template_id"] else "",
        }
    )
    now = settings_now()
    with connect() as conn:
        conn.execute(
            "UPDATE training_jobs SET dataset_version_id = ?, updated_at = ? WHERE id = ?",
            (project["current_dataset_version_id"], now, job_id),
        )
        if skip_reason.strip():
            conn.execute(
                """
                UPDATE lora_projects
                SET memo = TRIM(COALESCE(memo, '') || CASE WHEN COALESCE(memo, '') = '' THEN '' ELSE char(10) END || ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (f"軽量確認スキップ: {skip_reason.strip()}", now, project_id),
            )
    return job_id


def project_training_jobs(project_id: int) -> list[dict[str, Any]]:
    rows = fetch_all(
        """
        SELECT j.*, p.name AS preset_name, s.health_label,
               o.id AS selected_output_id, o.epoch AS selected_epoch
        FROM training_jobs j
        LEFT JOIN presets p ON p.id = j.preset_id
        LEFT JOIN training_metric_summaries s ON s.job_id = j.id
        LEFT JOIN training_outputs o ON o.job_id = j.id AND o.selected = 1
        WHERE j.project_id = ? AND j.deleted_at IS NULL
        ORDER BY j.id
        """,
        (project_id,),
    )
    jobs = []
    for row in rows:
        item = dict(row)
        try:
            params = json.loads(item.get("params_json") or "{}")
        except json.JSONDecodeError:
            params = {}
        item["max_train_epochs"] = params.get("max_train_epochs") or params.get("max_train_steps") or "-"
        jobs.append(item)
    return jobs


def folder_size(path_value: str | None) -> int:
    if not path_value:
        return 0
    root = Path(path_value)
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def job_delete_preview(job_id: int) -> dict[str, Any]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    counts = {
        "outputs": fetch_one("SELECT COUNT(*) AS count FROM training_outputs WHERE job_id = ?", (job_id,))["count"],
        "samples": fetch_one("SELECT COUNT(*) AS count FROM sample_images WHERE job_id = ?", (job_id,))["count"],
        "metrics": fetch_one("SELECT COUNT(*) AS count FROM training_metrics WHERE job_id = ?", (job_id,))["count"],
        "validation_images": fetch_one("SELECT COUNT(*) AS count FROM validation_images WHERE job_id = ?", (job_id,))["count"],
        "validation_runs": fetch_one("SELECT COUNT(*) AS count FROM validation_runs WHERE job_id = ?", (job_id,))["count"],
        "recommendations": fetch_one("SELECT COUNT(*) AS count FROM experiment_recommendations WHERE source_job_id = ? OR created_job_id = ?", (job_id, job_id))["count"],
        "profiles": fetch_one("SELECT COUNT(*) AS count FROM selected_lora_profiles WHERE job_id = ?", (job_id,))["count"],
        "exports": 0,
    }
    export_dirs = [
        settings.EXPORTS_DIR / "selected_loras" / f"job_{job_id:06d}",
        settings.EXPORTS_DIR / "contact_sheets" / f"job_{job_id:06d}",
        settings.EXPORTS_DIR / "validation_packs" / f"job_{job_id:06d}",
        settings.EXPORTS_DIR / "recommendations" / f"job_{job_id:06d}",
    ]
    counts["exports"] = sum(1 for directory in export_dirs if directory.exists())
    selected_links = (
        int(counts["profiles"] or 0) > 0
        or bool(job["adopted_model_path"])
        or bool(fetch_one("SELECT 1 FROM training_outputs WHERE job_id = ? AND selected = 1 LIMIT 1", (job_id,)))
        or bool(fetch_one("SELECT 1 FROM lora_projects WHERE selected_job_id = ? OR selected_output_id IN (SELECT id FROM training_outputs WHERE job_id = ?) LIMIT 1", (job_id, job_id)))
    )
    status = job["status"]
    can_delete_db = status in DELETABLE_JOB_STATUSES and not selected_links and not row_value(job, "deleted_at")
    warnings = []
    if status == "completed":
        warnings.append("完了済み学習ジョブは再現性維持のため、削除ではなくアーカイブ推奨です。")
    if selected_links:
        warnings.append("採用LoRAまたはLoRAライブラリに紐づいているため、完全削除は拒否します。")
    if counts["outputs"] or counts["samples"] or counts["metrics"]:
        warnings.append("成果物・sample・metricsがあります。削除すると後から比較しづらくなります。")
    size = folder_size(job["run_dir"])
    return {
        "job": job,
        "counts": counts,
        "run_dir": job["run_dir"],
        "folder_size": size,
        "folder_size_label": format_bytes(size),
        "selected_links": selected_links,
        "can_delete_db": can_delete_db,
        "warnings": warnings,
    }


def project_delete_preview(project_id: int) -> dict[str, Any]:
    project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (project_id,))
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    counts = {
        "jobs": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE project_id = ? AND deleted_at IS NULL", (project_id,))["count"],
        "completed_jobs": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE project_id = ? AND status = 'completed' AND deleted_at IS NULL", (project_id,))["count"],
        "draft_jobs": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE project_id = ? AND status IN ('draft','prepared','prepared_dirty') AND deleted_at IS NULL", (project_id,))["count"],
        "profiles": fetch_one("SELECT COUNT(*) AS count FROM selected_lora_profiles WHERE project_id = ?", (project_id,))["count"],
        "validation_runs": fetch_one("SELECT COUNT(*) AS count FROM validation_runs WHERE project_id = ?", (project_id,))["count"],
        "recommendations": fetch_one("SELECT COUNT(*) AS count FROM experiment_recommendations WHERE project_id = ?", (project_id,))["count"],
    }
    job_dirs = fetch_all("SELECT run_dir FROM training_jobs WHERE project_id = ?", (project_id,))
    runs_size = sum(folder_size(row["run_dir"]) for row in job_dirs)
    exports_size = folder_size(str(settings.EXPORTS_DIR / "selected_loras"))
    selected_links = bool(project["selected_job_id"] or project["selected_output_id"] or project["selected_lora_profile_id"] or counts["profiles"])
    return {
        "project": project,
        "counts": counts,
        "runs_size": runs_size,
        "runs_size_label": format_bytes(runs_size),
        "exports_size": exports_size,
        "exports_size_label": format_bytes(exports_size),
        "selected_links": selected_links,
    }


def cleanup_job_candidates(limit: int = 30) -> list[dict[str, Any]]:
    rows = fetch_all(
        """
        SELECT j.*, p.name AS preset_name,
               (SELECT COUNT(*) FROM training_outputs o WHERE o.job_id = j.id AND COALESCE(o.deleted_at, '') = '') AS output_count,
               EXISTS(SELECT 1 FROM training_outputs o WHERE o.job_id = j.id AND o.selected = 1 AND COALESCE(o.deleted_at, '') = '') AS has_selected_output
        FROM training_jobs j
        LEFT JOIN presets p ON p.id = j.preset_id
        WHERE j.archived_at IS NULL
          AND j.deleted_at IS NULL
          AND (
              j.status IN ('draft', 'prepared', 'prepared_dirty')
              OR (j.status IN ('failed', 'stopped') AND COALESCE(j.output_model_count, 0) = 0)
              OR (j.status IN ('failed', 'stopped') AND COALESCE(j.output_model_count, 0) > 0
                  AND NOT EXISTS(SELECT 1 FROM training_outputs o WHERE o.job_id = j.id AND o.selected = 1 AND COALESCE(o.deleted_at, '') = ''))
              OR j.preset_id = ?
              OR (LOWER(COALESCE(j.name, '') || ' ' || COALESCE(j.preset_id, '')) LIKE '%pilot%' AND j.adopted_model_path IS NULL)
              OR LOWER(COALESCE(j.name, '')) LIKE '%test%'
          )
        ORDER BY j.updated_at ASC, j.id ASC
        LIMIT ?
        """,
        (INTEGRATION_SMOKE_PRESET_ID, limit),
    )
    candidates = []
    for row in rows:
        reason = "整理候補"
        if row["status"] in {"draft", "prepared", "prepared_dirty"}:
            reason = "未実行の下書き/準備済みです。不要なら削除できます。"
        elif row["status"] in {"failed", "stopped"} and int(row["output_count"] or 0) == 0:
            reason = "成果物がない失敗/停止ジョブです。"
        elif row["status"] in {"failed", "stopped"} and not row["has_selected_output"]:
            reason = "採用LoRAがない失敗/停止ジョブです。出力モデルやサンプルの整理候補です。"
        elif row["preset_id"] == INTEGRATION_SMOKE_PRESET_ID:
            reason = "結合確認ジョブです。完了後はアーカイブ候補です。"
        elif "pilot" in ((row["preset_id"] or "") + " " + (row["name"] or "")).lower():
            reason = "未採用の軽量確認ジョブです。"
        candidates.append({**dict(row), "cleanup_reason": reason, "actions": job_action_state(row, None)})
    return candidates


def update_project_selected_from_job(project_id: int, job_id: int) -> None:
    project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (project_id,))
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    if project is None or job is None or output is None:
        raise HTTPException(status_code=400, detail="Project採用に反映できる選択済みLoRAがありません。")
    profile = ensure_selected_lora_profile(job_id)
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE lora_projects
            SET selected_job_id = ?, selected_output_id = ?, selected_lora_profile_id = ?,
                recommended_weight_min = COALESCE(?, recommended_weight_min),
                recommended_weight_max = COALESCE(?, recommended_weight_max),
                status = CASE WHEN status IN ('draft', 'training', 'reviewing') THEN 'selected' ELSE status END,
                updated_at = ?
            WHERE id = ?
            """,
            (
                job_id,
                output["id"],
                profile["id"] if profile else None,
                profile["recommended_weight_min"] if profile else None,
                profile["recommended_weight_max"] if profile else None,
                now,
                project_id,
            ),
        )
        if profile:
            conn.execute("UPDATE selected_lora_profiles SET project_id = ?, updated_at = ? WHERE id = ?", (project_id, now, profile["id"]))


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    reconcile_stale_running_jobs()
    stats = {
        "presets": fetch_one("SELECT COUNT(*) AS count FROM presets")["count"],
        "datasets": fetch_one("SELECT COUNT(*) AS count FROM datasets")["count"],
        "jobs": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE archived_at IS NULL AND deleted_at IS NULL")["count"],
        "running": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'running' AND archived_at IS NULL AND deleted_at IS NULL")["count"],
        "completed": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'completed' AND archived_at IS NULL AND deleted_at IS NULL")["count"],
        "failed": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'failed' AND archived_at IS NULL AND deleted_at IS NULL")["count"],
        "stopped": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'stopped' AND archived_at IS NULL AND deleted_at IS NULL")["count"],
        "active_projects": fetch_one("SELECT COUNT(*) AS count FROM lora_projects WHERE archived_at IS NULL AND deleted_at IS NULL")["count"],
        "archived_projects": fetch_one("SELECT COUNT(*) AS count FROM lora_projects WHERE archived_at IS NOT NULL AND deleted_at IS NULL")["count"],
        "archived_jobs": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE archived_at IS NOT NULL AND deleted_at IS NULL")["count"],
        "draft_jobs": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'draft' AND archived_at IS NULL AND deleted_at IS NULL")["count"],
    }
    jobs = fetch_all("SELECT * FROM training_jobs WHERE archived_at IS NULL AND deleted_at IS NULL ORDER BY id DESC LIMIT 8")
    attention_jobs = fetch_all(
        """
        SELECT * FROM training_jobs
        WHERE archived_at IS NULL
          AND deleted_at IS NULL
          AND (status IN ('draft', 'prepared', 'prepared_dirty', 'failed', 'running')
           OR COALESCE(config_dirty, 0) = 1
          )
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
    projects = recent_projects(limit=6)
    cleanup_candidates = cleanup_job_candidates(limit=8)
    return render(
        request,
        "dashboard.html",
        stats=stats,
        jobs=jobs,
        attention_jobs=attention_jobs,
        validation_profiles=validation_profiles,
        projects=projects,
        cleanup_candidates=cleanup_candidates,
        status_labels=STATUS_LABELS,
        project_status_labels=PROJECT_STATUS_LABELS,
    )


@app.get("/projects", response_class=HTMLResponse)
def projects_list(request: Request, view: str = Query("active")) -> HTMLResponse:
    if view not in {"active", "archived", "deleted", "all"}:
        view = "active"
    return render(
        request,
        "projects.html",
        projects=recent_projects(limit=100, view=view),
        view=view,
        project_status_labels=PROJECT_STATUS_LABELS,
    )


@app.get("/review-sessions", response_class=HTMLResponse)
def review_sessions_list(request: Request) -> HTMLResponse:
    sessions = fetch_all(
        """
        SELECT rs.*, p.name AS project_name, tj.name AS job_name,
               (
                   SELECT COUNT(*)
                   FROM review_session_conditions rsc
                   WHERE rsc.review_session_id = rs.id
               ) AS condition_count,
               (
                   SELECT COUNT(*)
                   FROM review_session_images rsi
                   WHERE rsi.review_session_id = rs.id
                     AND rsi.deleted_at IS NULL
               ) AS image_count
        FROM review_sessions rs
        LEFT JOIN lora_projects p ON p.id = rs.project_id
        LEFT JOIN training_jobs tj ON tj.id = rs.job_id
        ORDER BY rs.id DESC
        LIMIT 100
        """
    )
    return render(request, "review_sessions.html", sessions=sessions)


@app.get("/review-sessions/{session_id}", response_class=HTMLResponse)
def review_session_detail(request: Request, session_id: int) -> HTMLResponse:
    session = fetch_one(
        """
        SELECT rs.*, p.name AS project_name, tj.name AS job_name, tj.adopted_epoch,
               so.epoch AS selected_epoch
        FROM review_sessions rs
        LEFT JOIN lora_projects p ON p.id = rs.project_id
        LEFT JOIN training_jobs tj ON tj.id = rs.job_id
        LEFT JOIN training_outputs so ON so.id = (
            SELECT id FROM training_outputs
            WHERE job_id = rs.job_id AND selected = 1
            ORDER BY id DESC LIMIT 1
        )
        WHERE rs.id = ?
        """,
        (session_id,),
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Review Session not found")
    conditions = fetch_all(
        """
        SELECT epoch, COUNT(*) AS count
        FROM review_session_conditions
        WHERE review_session_id = ?
        GROUP BY epoch
        ORDER BY epoch
        """,
        (session_id,),
    )
    images = fetch_one(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN deleted_at IS NULL THEN 1 ELSE 0 END) AS active
        FROM review_session_images
        WHERE review_session_id = ?
        """,
        (session_id,),
    )
    scores = fetch_one(
        """
        SELECT COUNT(*) AS total
        FROM machine_review_scores mrs
        JOIN review_session_images rsi ON rsi.id = mrs.source_id
        WHERE mrs.source_type = 'review_session_image'
          AND rsi.review_session_id = ?
        """,
        (session_id,),
    )
    candidate_epochs: list[int] = []
    if session["candidate_epochs_json"]:
        try:
            candidate_epochs = [int(value) for value in json.loads(session["candidate_epochs_json"])]
        except Exception:
            candidate_epochs = []
    outputs = []
    if session["job_id"] and candidate_epochs:
        placeholders = ",".join("?" for _ in candidate_epochs)
        outputs = fetch_all(
            f"""
            SELECT id, epoch, file_path, selected
            FROM training_outputs
            WHERE job_id = ? AND epoch IN ({placeholders})
            ORDER BY epoch, id
            """,
            (session["job_id"], *candidate_epochs),
        )
    output_by_epoch = {int(row["epoch"]): row for row in outputs if row["epoch"] is not None}
    status = session["status"] or ""
    matrix_path = str(session["matrix_path"] or "")
    matrix_ready = bool(matrix_path and Path(matrix_path).exists())
    can_select_epoch = status == "completed" and matrix_ready
    if status in {"planned", "prepared"}:
        primary_action = "start"
        primary_label = "このプランで候補レビューを生成"
    elif status == "running":
        primary_action = "progress"
        primary_label = "進捗を確認"
    elif status == "completed" and matrix_ready:
        primary_action = "open_matrix"
        primary_label = "レビューMatrixを開く"
    elif status == "completed":
        primary_action = "build_matrix"
        primary_label = "レビューMatrixを作成"
    elif status in {"failed", "stopped"}:
        primary_action = "check_log"
        primary_label = "ログ確認"
    else:
        primary_action = "start"
        primary_label = "このプランで候補レビューを生成"
    selected_epoch = session["selected_epoch"] if session["selected_epoch"] is not None else session["adopted_epoch"]
    selected_epoch_in_session = selected_epoch is not None and int(selected_epoch) in set(candidate_epochs)
    return render(
        request,
        "review_session_detail.html",
        session=session,
        conditions=conditions,
        images=images,
        scores=scores,
        candidate_epochs=candidate_epochs,
        outputs=outputs,
        output_by_epoch=output_by_epoch,
        primary_action=primary_action,
        primary_label=primary_label,
        matrix_ready=matrix_ready,
        can_select_epoch=can_select_epoch,
        selected_epoch=selected_epoch,
        selected_epoch_in_session=selected_epoch_in_session,
        operation_monitor=review_session_operation_monitor(session),
    )


@app.get("/validation-runs", response_class=HTMLResponse)
def validation_runs_list(request: Request) -> HTMLResponse:
    runs = fetch_all(
        """
        SELECT vr.*, p.name AS project_name, tj.name AS job_name, vp.name AS preset_name,
               (
                   SELECT COUNT(*)
                   FROM validation_images vi
                   WHERE vi.validation_run_id = vr.id
                     AND COALESCE(vi.ignored, 0) = 0
               ) AS image_count,
               (
                   SELECT COUNT(*)
                   FROM validation_images vi
                   WHERE vi.validation_run_id = vr.id
                     AND COALESCE(vi.ignored, 0) = 0
                     AND (
                         COALESCE(vi.rating_overall, 0) > 0
                         OR COALESCE(vi.rating_face, 0) > 0
                         OR COALESCE(vi.rating_costume, 0) > 0
                         OR COALESCE(vi.rating_style, 0) > 0
                         OR COALESCE(vi.rating_stability, 0) > 0
                         OR COALESCE(vi.rating_flexibility, 0) > 0
                     )
               ) AS reviewed_count
        FROM validation_runs vr
        LEFT JOIN lora_projects p ON p.id = vr.project_id
        LEFT JOIN training_jobs tj ON tj.id = vr.job_id
        LEFT JOIN validation_presets vp ON vp.id = vr.validation_preset_id
        ORDER BY vr.id DESC
        LIMIT 100
        """
    )
    return render(request, "validation_runs.html", runs=runs)


@app.get("/projects/{project_id}", response_class=HTMLResponse)
def project_detail(request: Request, project_id: int) -> HTMLResponse:
    project = fetch_one(
        """
        SELECT p.*, d.name AS dataset_name, dv.version_no AS dataset_version_no,
               sj.name AS selected_job_name, so.file_path AS selected_model_path,
               so.epoch AS selected_epoch, sp.profile_name AS selected_profile_name,
               sp.validation_policy_memo,
               rs.name AS default_reference_set_name,
               rsv.version_no AS default_reference_version_no,
               rsv.completeness_label AS default_reference_completeness_label
        FROM lora_projects p
        LEFT JOIN datasets d ON d.id = p.dataset_id
        LEFT JOIN dataset_versions dv ON dv.id = p.current_dataset_version_id
        LEFT JOIN training_jobs sj ON sj.id = p.selected_job_id
        LEFT JOIN training_outputs so ON so.id = p.selected_output_id
        LEFT JOIN selected_lora_profiles sp ON sp.id = p.selected_lora_profile_id
        LEFT JOIN reference_sets rs ON rs.id = p.default_reference_set_id
        LEFT JOIN reference_set_versions rsv ON rsv.id = p.default_reference_set_version_id
        WHERE p.id = ?
        """,
        (project_id,),
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    jobs = project_training_jobs(project_id)
    validation_runs = fetch_all(
        """
        SELECT vr.*, vp.name AS preset_name,
               (
                   SELECT COUNT(*) FROM validation_images vi
                   WHERE vi.validation_run_id = vr.id
                     AND COALESCE(vi.ignored, 0) = 0
                     AND (
                         COALESCE(vi.rating_overall, 0) > 0
                         OR COALESCE(vi.rating_face, 0) > 0
                         OR COALESCE(vi.rating_costume, 0) > 0
                         OR COALESCE(vi.rating_style, 0) > 0
                         OR COALESCE(vi.rating_stability, 0) > 0
                         OR COALESCE(vi.rating_flexibility, 0) > 0
                     )
               ) AS reviewed_count
        FROM validation_runs vr
        LEFT JOIN validation_presets vp ON vp.id = vr.validation_preset_id
        WHERE vr.project_id = ?
        ORDER BY vr.id DESC
        """,
        (project_id,),
    )
    review_sessions = fetch_all(
        """
        SELECT rs.*,
               tj.name AS job_name,
               (
                   SELECT COUNT(*)
                   FROM review_session_conditions rsc
                   WHERE rsc.review_session_id = rs.id
               ) AS condition_count,
               (
                   SELECT COUNT(*)
                   FROM review_session_images rsi
                   WHERE rsi.review_session_id = rs.id
                     AND rsi.deleted_at IS NULL
               ) AS image_count
        FROM review_sessions rs
        LEFT JOIN training_jobs tj ON tj.id = rs.job_id
        WHERE rs.project_id = ?
        ORDER BY rs.id DESC
        """,
        (project_id,),
    )
    reference_sets_rows = fetch_all(
        """
        SELECT r.*, v.version_no AS current_version_no, v.image_count,
               v.completeness_label, v.completeness_message
        FROM reference_sets r
        LEFT JOIN reference_set_versions v ON v.id = r.current_version_id
        WHERE r.project_id = ?
        ORDER BY r.is_default DESC, r.updated_at DESC, r.id DESC
        """,
        (project_id,),
    )
    recommendations = fetch_all(
        """
        SELECT * FROM experiment_recommendations
        WHERE project_id = ? AND status != 'dismissed'
        ORDER BY CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, id
        """,
        (project_id,),
    )
    return render(
        request,
        "project_detail.html",
        project=project,
        jobs=jobs,
        validation_runs=validation_runs,
        review_sessions=review_sessions,
        reference_sets=reference_sets_rows,
        recommendations=recommendations,
        workflow_status=project_workflow_status(project, jobs, validation_runs, recommendations),
        project_next_action=project_next_action(project, jobs, review_sessions, validation_runs, recommendations),
        pilot_guidance=pilot_recommendation(project),
        storage_summary=project_storage_summary(project_id),
        project_status_labels=PROJECT_STATUS_LABELS,
        status_labels=STATUS_LABELS,
    )


@app.post("/projects/{project_id}/create-preset-job")
def project_create_preset_job(
    project_id: int,
    preset_id: str = Form(...),
    skip_reason: str = Form(""),
    source_job_id: str = Form(""),
) -> RedirectResponse:
    job_id = create_project_preset_job(project_id, preset_id, skip_reason, int(source_job_id) if source_job_id.strip() else None)
    return RedirectResponse(f"/jobs/{job_id}/edit", status_code=303)


@app.post("/projects/{project_id}/select-job/{job_id}")
def project_select_job(project_id: int, job_id: int) -> RedirectResponse:
    update_project_selected_from_job(project_id, job_id)
    return RedirectResponse(f"/projects/{project_id}", status_code=303)


@app.post("/projects/{project_id}/archive")
def project_archive(
    project_id: int,
    archive_reason: str = Form(""),
    archive_jobs: str = Form(""),
) -> RedirectResponse:
    project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (project_id,))
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE lora_projects
            SET archived_at = COALESCE(archived_at, ?), archive_reason = ?, updated_at = ?
            WHERE id = ? AND deleted_at IS NULL
            """,
            (now, archive_reason.strip(), now, project_id),
        )
        if archive_jobs == "1":
            conn.execute(
                """
                UPDATE training_jobs
                SET archived_at = COALESCE(archived_at, ?), archived_reason = ?, updated_at = ?
                WHERE project_id = ? AND deleted_at IS NULL AND status != 'running'
                """,
                (now, f"Project #{project_id} のアーカイブに合わせて整理", now, project_id),
            )
    return RedirectResponse("/projects?view=archived", status_code=303)


@app.post("/projects/{project_id}/restore")
def project_restore(project_id: int) -> RedirectResponse:
    with connect() as conn:
        conn.execute(
            "UPDATE lora_projects SET archived_at = NULL, archive_reason = NULL, updated_at = ? WHERE id = ? AND deleted_at IS NULL",
            (settings_now(), project_id),
        )
    return RedirectResponse(f"/projects/{project_id}", status_code=303)


@app.get("/projects/{project_id}/delete-preview", response_class=HTMLResponse)
def project_delete_preview_page(request: Request, project_id: int) -> HTMLResponse:
    return render(request, "project_delete_preview.html", preview=project_delete_preview(project_id))


@app.post("/projects/{project_id}/delete")
def project_delete(project_id: int, delete_reason: str = Form("")) -> RedirectResponse:
    preview = project_delete_preview(project_id)
    if preview["selected_links"]:
        raise HTTPException(status_code=400, detail="採用LoRAに紐づくProjectは削除できません。先に採用状態を見直すか、アーカイブしてください。")
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE lora_projects
            SET deleted_at = COALESCE(deleted_at, ?), delete_reason = ?, updated_at = ?
            WHERE id = ?
            """,
            (now, delete_reason.strip(), now, project_id),
        )
    return RedirectResponse("/projects?view=deleted", status_code=303)


@app.post("/jobs/{job_id}/sync-project-selection")
def job_sync_project_selection(request: Request, job_id: int):
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None or not job["project_id"]:
        raise HTTPException(status_code=400, detail="この学習ジョブはProjectに紐づいていません。")
    update_project_selected_from_job(int(job["project_id"]), job_id)
    if wants_json_response(request):
        return JSONResponse({"ok": True, "job_id": job_id, "project_id": int(job["project_id"]), "message": "Project採用に反映しました。"})
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.get("/maintenance", response_class=HTMLResponse)
def maintenance(request: Request, backup_path: str = "", diagnostics_path: str = "") -> HTMLResponse:
    usage = storage_usage()
    usage["embedding_cache"] = embedding_cache_size()
    return render(
        request,
        "maintenance.html",
        summary=maintenance_summary(),
        storage=usage,
        embedding_jobs=latest_embedding_jobs(10),
        failed_embedding_cleanup=embedding_cleanup_preview("failed"),
        stale_embedding_cleanup=embedding_cleanup_preview("stale"),
        format_bytes=embedding_format_bytes,
        cleanup_candidates=cleanup_job_candidates(limit=20),
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


@app.get("/storage", response_class=HTMLResponse)
def storage_page(request: Request) -> HTMLResponse:
    storage = storage_usage()
    storage["embedding_cache"] = embedding_cache_size()
    return render(request, "storage_usage.html", storage=storage, status_labels=STATUS_LABELS, format_bytes=embedding_format_bytes)


@app.post("/storage/trash/purge")
def storage_purge_trash() -> RedirectResponse:
    purged = purge_trash()
    return RedirectResponse(f"/storage?purged={quote(storage_format_bytes(purged))}", status_code=303)


@app.get("/workflow", response_class=HTMLResponse)
def recommended_workflow(request: Request) -> HTMLResponse:
    return render(request, "workflow.html")


@app.get("/environment", response_class=HTMLResponse)
def environment(request: Request) -> HTMLResponse:
    settings_rows = fetch_all("SELECT * FROM app_settings ORDER BY key")
    environments = fetch_all("SELECT * FROM environments ORDER BY id DESC")
    return render(request, "environment.html", settings_rows=settings_rows, environments=environments, settings=settings)


@app.post("/environment/refresh")
def environment_refresh() -> RedirectResponse:
    import_latest_environment()
    return RedirectResponse("/environment?refreshed=1", status_code=303)


@app.get("/settings/embeddings", response_class=HTMLResponse)
def embedding_settings_page(request: Request, preflight_model_id: str | None = None) -> HTMLResponse:
    models = fetch_all("SELECT * FROM embedding_models ORDER BY provider, name")
    settings_row = load_embedding_settings()
    preflight = None
    if preflight_model_id:
        preflight = provider_preflight(preflight_model_id)
    return render(
        request,
        "embedding_settings.html",
        models=models,
        embedding_settings=settings_row,
        active_model=active_embedding_model(),
        machine_review_settings=load_machine_review_settings(),
        preflight=preflight,
        embedding_jobs=latest_embedding_jobs(),
        machine_review_jobs=latest_machine_review_jobs(),
        cache_size=embedding_cache_size(),
        format_bytes=embedding_format_bytes,
        failed_cleanup=embedding_cleanup_preview("failed"),
        stale_cleanup=embedding_cleanup_preview("stale"),
    )


@app.post("/settings/embeddings")
def embedding_settings_save(
    active_embedding_model_id: str = Form("mock_image_512"),
    python_path: str = Form(""),
    device: str = Form("auto"),
    dtype: str = Form("fp32"),
    batch_size: int = Form(8),
    cache_root: str = Form(""),
    allow_model_download: str = Form(""),
    max_image_size: int = Form(1024),
    num_workers: int = Form(1),
) -> RedirectResponse:
    update_embedding_settings(
        {
            "active_embedding_model_id": active_embedding_model_id,
            "python_path": python_path,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "cache_root": cache_root,
            "allow_model_download": allow_model_download == "1",
            "max_image_size": max_image_size,
            "num_workers": num_workers,
        }
    )
    return RedirectResponse("/settings/embeddings", status_code=303)


@app.post("/settings/machine-review")
def machine_review_settings_save(
    active_embedding_model_id: str = Form("mock_image_512"),
    reference_similarity_method: str = Form("avg_max_blend"),
    overfit_nearest_threshold: float = Form(0.90),
    overfit_margin_threshold: float = Form(0.05),
    reference_low_threshold: float = Form(0.20),
    low_confidence_when_mock_provider: str = Form(""),
    minimum_reference_images_character: int = Form(3),
    minimum_reference_images_style: int = Form(6),
    include_dataset_nearest_check: str = Form(""),
) -> RedirectResponse:
    update_machine_review_settings(
        {
            "active_embedding_model_id": active_embedding_model_id,
            "reference_similarity_method": reference_similarity_method,
            "overfit_nearest_threshold": overfit_nearest_threshold,
            "overfit_margin_threshold": overfit_margin_threshold,
            "reference_low_threshold": reference_low_threshold,
            "low_confidence_when_mock_provider": low_confidence_when_mock_provider == "1",
            "minimum_reference_images_character": minimum_reference_images_character,
            "minimum_reference_images_style": minimum_reference_images_style,
            "include_dataset_nearest_check": include_dataset_nearest_check == "1",
        }
    )
    return RedirectResponse("/settings/embeddings#machine-review-settings", status_code=303)


@app.post("/settings/embeddings/preflight")
def embedding_preflight(active_embedding_model_id: str = Form("mock_image_512")) -> RedirectResponse:
    provider_preflight(active_embedding_model_id)
    return RedirectResponse(f"/settings/embeddings?preflight_model_id={quote(active_embedding_model_id)}", status_code=303)


@app.post("/embeddings/jobs")
def embedding_job_create(request: Request, job_type: str = Form(...), target_id: int = Form(...), recompute: str = Form("missing"), return_to: str = Form("")):
    destination = return_to or "/settings/embeddings"
    running_embedding = fetch_one("SELECT id FROM embedding_jobs WHERE status = 'running' LIMIT 1")
    if running_embedding:
        message = f"Embedding Job #{running_embedding['id']} が実行中です。完了後に再実行してください。"
        if wants_json_response(request):
            return JSONResponse({"ok": False, "message": message, "running_job_id": running_embedding["id"]}, status_code=409)
        return RedirectResponse(
            add_query_param(destination, embedding_error=message),
            status_code=303,
        )
    try:
        embedding_model = active_embedding_model()
        if embedding_model.get("provider") != "mock":
            running_training = fetch_one("SELECT id FROM training_jobs WHERE status = 'running' LIMIT 1")
            running_generation = fetch_one("SELECT id FROM validation_generation_runs WHERE status = 'running' LIMIT 1")
            if running_training or running_generation:
                raise RuntimeError("学習または検証画像生成が実行中です。GPUを使うEmbedding providerは、実行中の処理が終わってから開始してください。")
        embedding_job_id = create_embedding_job(job_type, target_id, recompute=recompute)
        start_embedding_job(embedding_job_id)
    except RuntimeError as exc:
        if wants_json_response(request):
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
        return RedirectResponse(add_query_param(destination, embedding_error=str(exc)), status_code=303)
    redirect_url = add_query_param(destination, embedding_job_id=embedding_job_id, embedding_message=f"Embedding Job #{embedding_job_id} を開始しました。")
    if wants_json_response(request):
        return JSONResponse({"ok": True, "embedding_job_id": embedding_job_id, "message": f"Embedding Job #{embedding_job_id} を開始しました。", "redirect_url": redirect_url})
    return RedirectResponse(
        redirect_url,
        status_code=303,
    )


@app.get("/embeddings/jobs/{embedding_job_id}/status")
def embedding_job_status(embedding_job_id: int) -> JSONResponse:
    row = fetch_one("SELECT * FROM embedding_jobs WHERE id = ?", (embedding_job_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Embedding Job not found")
    log_path = Path(row["log_path"]) if row["log_path"] else None
    log_size = log_path.stat().st_size if log_path and log_path.exists() else 0
    log_updated_at = datetime.fromtimestamp(log_path.stat().st_mtime).isoformat(timespec="seconds") if log_path and log_path.exists() else ""
    return JSONResponse(
        {
            "id": row["id"],
            "status": row["status"],
            "total_count": row["total_count"],
            "processed_count": row["processed_count"],
            "ready_count": row["ready_count"],
            "failed_count": row["failed_count"],
            "skipped_count": row["skipped_count"],
            "error_message": row["error_message"],
            "ended_at": row["ended_at"],
            "process_id": row["process_id"],
            "return_code": row["return_code"],
            "log_tail": _tail_file(row["log_path"], 20),
            "log_size": log_size,
            "log_updated_at": log_updated_at,
        }
    )


@app.get("/jobs/{job_id}/machine-review-readiness/status")
def job_machine_review_readiness_status(job_id: int) -> JSONResponse:
    job = fetch_one("SELECT id FROM training_jobs WHERE id = ? AND deleted_at IS NULL", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(machine_review_readiness("training_job_samples", job_id))


@app.post("/embeddings/jobs/{embedding_job_id}/stop")
def embedding_job_stop(embedding_job_id: int, return_to: str = Form("")) -> RedirectResponse:
    stop_embedding_job(embedding_job_id)
    return RedirectResponse(return_to or "/settings/embeddings", status_code=303)


@app.post("/machine-review/jobs")
def machine_review_run(target_type: str = Form(...), target_id: int = Form(...), reference_set_version_id: int | None = Form(None), return_to: str = Form("")) -> RedirectResponse:
    destination = return_to or "/settings/embeddings"
    running = fetch_one("SELECT id FROM machine_review_jobs WHERE status = 'running' LIMIT 1")
    if running:
        return RedirectResponse(
            add_query_param(destination, machine_review_error=f"機械補助レビューJob #{running['id']} が実行中です。完了後に再実行してください。"),
            status_code=303,
        )
    try:
        machine_review_job_id = create_machine_review_job(target_type, target_id, reference_set_version_id=reference_set_version_id)
        start_machine_review_job(machine_review_job_id)
    except (RuntimeError, ValueError) as exc:
        return RedirectResponse(add_query_param(destination, machine_review_error=str(exc)), status_code=303)
    return RedirectResponse(
        add_query_param(
            destination,
            machine_review_job_id=machine_review_job_id,
            machine_review_message=f"機械補助レビューJob #{machine_review_job_id} を開始しました。",
        ),
        status_code=303,
    )


@app.get("/machine-review/jobs/{machine_review_job_id}/status")
def machine_review_job_status(machine_review_job_id: int) -> JSONResponse:
    row = fetch_one("SELECT * FROM machine_review_jobs WHERE id = ?", (machine_review_job_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Machine Review Job not found")
    log_path = Path(row["log_path"]) if row["log_path"] else None
    log_size = log_path.stat().st_size if log_path and log_path.exists() else 0
    log_updated_at = datetime.fromtimestamp(log_path.stat().st_mtime).isoformat(timespec="seconds") if log_path and log_path.exists() else ""
    return JSONResponse(
        {
            "id": row["id"],
            "status": row["status"],
            "total_count": row["total_count"],
            "processed_count": row["processed_count"],
            "scored_count": row["scored_count"],
            "skipped_count": row["skipped_count"],
            "failed_count": row["failed_count"],
            "error_message": row["error_message"],
            "ended_at": row["ended_at"],
            "process_id": row["process_id"],
            "return_code": row["return_code"],
            "log_tail": _tail_file(row["log_path"], 20),
            "log_size": log_size,
            "log_updated_at": log_updated_at,
        }
    )


@app.post("/machine-review/jobs/{machine_review_job_id}/stop")
def machine_review_job_stop(machine_review_job_id: int, return_to: str = Form("")) -> RedirectResponse:
    try:
        stop_machine_review_job(machine_review_job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(return_to or "/settings/embeddings", status_code=303)


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
    latest_version = versions[0] if versions else None
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
        embedding_coverage=embedding_coverage("dataset_version", latest_version["id"]) if latest_version else None,
        embedding_target_id=latest_version["id"] if latest_version else None,
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
    latest_version = versions[0] if versions else None
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
        embedding_coverage=embedding_coverage("dataset_version", latest_version["id"]) if latest_version else None,
        embedding_target_id=latest_version["id"] if latest_version else None,
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
    latest_version = versions[0] if versions else None
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
        embedding_coverage=embedding_coverage("dataset_version", latest_version["id"]) if latest_version else None,
        embedding_target_id=latest_version["id"] if latest_version else None,
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


@app.get("/sample-prompt-templates/new", response_class=HTMLResponse)
def sample_prompt_template_new(request: Request) -> HTMLResponse:
    return render(
        request,
        "sample_prompt_template_form.html",
        template_row=None,
        prompts_json=default_sample_prompts_json(),
        mode="new",
    )


@app.post("/sample-prompt-templates")
def sample_prompt_template_create(
    template_id: str = Form(...),
    name: str = Form(...),
    purpose: str = Form(""),
    prompts_json: str = Form(...),
) -> RedirectResponse:
    clean_id = validate_sample_prompt_template_id(template_id)
    if fetch_one("SELECT id FROM sample_prompt_templates WHERE id = ?", (clean_id,)):
        raise HTTPException(status_code=400, detail="同じIDのテンプレートが既にあります。")
    now = settings_now()
    clean_json = validate_sample_prompts_json(prompts_json)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO sample_prompt_templates(id, name, purpose, prompts_json, is_builtin, created_at, updated_at)
            VALUES (?, ?, ?, ?, 0, ?, ?)
            """,
            (clean_id, name.strip(), purpose.strip(), clean_json, now, now),
        )
    return RedirectResponse("/sample-prompt-templates", status_code=303)


@app.get("/sample-prompt-templates/{template_id}/edit", response_class=HTMLResponse)
def sample_prompt_template_edit(request: Request, template_id: str) -> HTMLResponse:
    template_row = fetch_one("SELECT * FROM sample_prompt_templates WHERE id = ?", (template_id,))
    if template_row is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return render(
        request,
        "sample_prompt_template_form.html",
        template_row=template_row,
        prompts_json=json.dumps(json.loads(template_row["prompts_json"]), ensure_ascii=False, indent=2),
        mode="edit",
    )


@app.post("/sample-prompt-templates/{template_id}/edit")
def sample_prompt_template_update(
    template_id: str,
    name: str = Form(...),
    purpose: str = Form(""),
    prompts_json: str = Form(...),
) -> RedirectResponse:
    template_row = fetch_one("SELECT * FROM sample_prompt_templates WHERE id = ?", (template_id,))
    if template_row is None:
        raise HTTPException(status_code=404, detail="Template not found")
    now = settings_now()
    clean_json = validate_sample_prompts_json(prompts_json)
    with connect() as conn:
        conn.execute(
            """
            UPDATE sample_prompt_templates
            SET name = ?, purpose = ?, prompts_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (name.strip(), purpose.strip(), clean_json, now, template_id),
        )
    return RedirectResponse("/sample-prompt-templates", status_code=303)


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
def jobs_list(request: Request, view: str = Query("active")) -> HTMLResponse:
    reconcile_stale_running_jobs()
    if view not in {key for key, _ in JOB_FILTERS}:
        view = "active"
    where = job_filter_where(view)
    rows = fetch_all(
        f"""
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
        WHERE {where}
        ORDER BY j.id DESC
        """
    )
    jobs = []
    for row in rows:
        item = dict(row)
        item["latest_action"] = job_latest_action(row)
        item["actions"] = job_action_state(row, None)
        jobs.append(item)
    return render(
        request,
        "jobs.html",
        jobs=jobs,
        view=view,
        job_filters=JOB_FILTERS,
        cleanup_candidates=cleanup_job_candidates(limit=12),
        status_labels=STATUS_LABELS,
    )


@app.post("/jobs/{job_id}/archive")
def job_archive(job_id: int, archived_reason: str = Form("")) -> RedirectResponse:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "running":
        raise HTTPException(status_code=400, detail="実行中の学習ジョブはアーカイブできません。先に停止してください。")
    with connect() as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET archived_at = COALESCE(archived_at, ?), archived_reason = ?, updated_at = ?
            WHERE id = ? AND deleted_at IS NULL
            """,
            (settings_now(), archived_reason.strip(), settings_now(), job_id),
        )
    return RedirectResponse("/jobs?view=archived", status_code=303)


@app.post("/jobs/{job_id}/restore")
def job_restore(job_id: int) -> RedirectResponse:
    with connect() as conn:
        conn.execute(
            "UPDATE training_jobs SET archived_at = NULL, archived_reason = NULL, updated_at = ? WHERE id = ? AND deleted_at IS NULL",
            (settings_now(), job_id),
        )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}/delete-preview", response_class=HTMLResponse)
def job_delete_preview_page(request: Request, job_id: int) -> HTMLResponse:
    return render(request, "job_delete_preview.html", preview=job_delete_preview(job_id), status_labels=STATUS_LABELS)


@app.post("/jobs/{job_id}/delete")
def job_delete(job_id: int, delete_reason: str = Form(""), delete_mode: str = Form("db_only")) -> RedirectResponse:
    preview = job_delete_preview(job_id)
    if delete_mode != "db_only":
        raise HTTPException(status_code=400, detail="ファイル削除は初期版では未実装です。DBのみ削除を選んでください。")
    if not preview["can_delete_db"]:
        raise HTTPException(status_code=400, detail="この学習ジョブは削除できません。アーカイブを使ってください。")
    with connect() as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET deleted_at = COALESCE(deleted_at, ?), delete_reason = ?, updated_at = ?
            WHERE id = ?
            """,
            (settings_now(), delete_reason.strip(), settings_now(), job_id),
        )
    return RedirectResponse("/jobs?view=deleted", status_code=303)


@app.get("/jobs/new", response_class=HTMLResponse)
def job_new(request: Request, project_id: str = Query("")) -> HTMLResponse:
    datasets = fetch_all("SELECT * FROM datasets ORDER BY id DESC")
    presets = preset_option_rows(fetch_all("SELECT * FROM presets ORDER BY model_family DESC, name"))
    projects = fetch_all("SELECT * FROM lora_projects WHERE status != 'archived' ORDER BY updated_at DESC, id DESC")
    dataset_versions = fetch_all("SELECT * FROM dataset_versions ORDER BY dataset_id, version_no DESC")
    sample_prompt_templates = fetch_all("SELECT * FROM sample_prompt_templates ORDER BY is_builtin DESC, name")
    trigger_infos = {row["dataset_id"]: row for row in fetch_all("SELECT * FROM dataset_analysis")}
    selected_project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (int(project_id),)) if project_id.strip() else None
    return render(
        request,
        "job_create.html",
        datasets=datasets,
        presets=presets,
        projects=projects,
        dataset_versions=dataset_versions,
        default_preset_id=DEFAULT_JOB_PRESET_ID,
        sample_prompt_templates=sample_prompt_templates,
        trigger_infos=trigger_infos,
        selected_project=selected_project,
        selected_project_id=int(project_id) if project_id.strip() else None,
        available_models=list_available_models(),
        default_model_path=str(settings.ROOT_DIR / "models"),
        default_project_path=str(settings.ROOT_DIR),
    )


@app.post("/jobs")
def job_create(
    name: str = Form(...),
    dataset_id: int = Form(...),
    preset_id: str = Form(...),
    base_model_path: str = Form(...),
    vae_path: str = Form(""),
    output_name: str = Form(""),
    memo: str = Form(""),
    sample_prompt_template_id: str = Form(""),
    project_mode: str = Form("new"),
    project_id: str = Form(""),
    project_name: str = Form(""),
    project_description: str = Form(""),
    project_trigger_word: str = Form(""),
    dataset_version_id: str = Form(""),
) -> RedirectResponse:
    selected_project_id: int | None = None
    if project_mode == "existing" and project_id.strip():
        project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (int(project_id),))
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        selected_project_id = int(project["id"])
        dataset_id = int(project["dataset_id"] or dataset_id)
        base_model_path = project["base_model_path"] or base_model_path
        dataset_version_id = str(project["current_dataset_version_id"] or dataset_version_id or "")
    else:
        dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        latest_version = fetch_one("SELECT id FROM dataset_versions WHERE dataset_id = ? ORDER BY version_no DESC LIMIT 1", (dataset_id,))
        selected_project_id = create_lora_project(
            {
                "name": project_name.strip() or name,
                "description": project_description.strip(),
                "dataset_id": dataset_id,
                "dataset_version_id": int(dataset_version_id) if dataset_version_id.strip() else (latest_version["id"] if latest_version else None),
                "trigger_word": project_trigger_word.strip() or (dataset["trigger_word"] if dataset else ""),
                "base_model_path": base_model_path,
                "status": "draft",
                "memo": memo,
            }
        )
    job_id = create_job({
        "project_id": selected_project_id,
        "name": name,
        "dataset_id": dataset_id,
        "preset_id": preset_id,
        "base_model_path": base_model_path,
        "vae_path": vae_path,
        "output_name": output_name,
        "memo": memo,
        "sample_prompt_template_id": sample_prompt_template_id,
    })
    if dataset_version_id.strip():
        with connect() as conn:
            conn.execute("UPDATE training_jobs SET dataset_version_id = ?, updated_at = ? WHERE id = ?", (int(dataset_version_id), settings_now(), job_id))
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
    generation_error: str | None = None,
    generation_message: str | None = None,
    review_prepare: str | None = None,
    review_prepare_error: str | None = None,
    review_filter: str = Query("candidates"),
    review_session_id: int | None = Query(None),
) -> HTMLResponse:
    reconcile_stale_running_jobs()
    reconcile_stale_validation_generations()
    reconcile_stale_review_sessions()
    reconcile_stale_embedding_jobs()
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
    outputs = fetch_all("SELECT * FROM training_outputs WHERE job_id = ? ORDER BY epoch, step, id", (job_id,))
    samples = fetch_all("SELECT * FROM sample_images WHERE job_id = ? ORDER BY epoch, id", (job_id,))
    sample_prompts = fetch_all("SELECT * FROM sample_prompts WHERE job_id = ? ORDER BY sort_order, id", (job_id,))
    metrics = fetch_all("SELECT * FROM training_metrics WHERE job_id = ? ORDER BY step, id", (job_id,))
    metric_table = build_metric_table(metrics)
    metric_summary = fetch_one("SELECT * FROM training_metric_summaries WHERE job_id = ?", (job_id,))
    epoch_summaries = fetch_all("SELECT * FROM training_epoch_summaries WHERE job_id = ? ORDER BY epoch", (job_id,))
    decorated_epochs = decorate_epoch_summaries(epoch_summaries, outputs, samples)
    log_tail = read_log_tail(dict(job))
    selected_output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    review_candidates = ensure_epoch_candidates(job_id)
    candidate_map = {int(row["epoch"]): row for row in review_candidates}
    candidate_epochs = {int(row["epoch"]) for row in review_candidates if row["candidate_label"] in {"primary", "secondary", "check"}}
    validation_results = fetch_all("SELECT * FROM validation_results WHERE job_id = ? ORDER BY created_at DESC, id DESC", (job_id,))
    validation_images = fetch_all("SELECT * FROM validation_images WHERE job_id = ? ORDER BY created_at DESC, id DESC", (job_id,))
    validation_weight_reviews = fetch_all("SELECT * FROM validation_weight_reviews WHERE job_id = ? ORDER BY lora_weight, id", (job_id,))
    validation_runs = fetch_all(
        """
        SELECT vr.*, o.epoch AS selected_epoch,
               vg.status AS generation_status,
               vg.process_id AS generation_process_id,
               vg.return_code AS generation_return_code
        FROM validation_runs vr
        LEFT JOIN training_outputs o ON o.id = vr.selected_output_id
        LEFT JOIN validation_generation_runs vg ON vg.id = (
            SELECT id FROM validation_generation_runs
            WHERE validation_run_id = vr.id
            ORDER BY id DESC LIMIT 1
        )
        WHERE vr.job_id = ?
        ORDER BY vr.id DESC
        """,
        (job_id,),
    )
    validation_runs = decorate_validation_runs_for_job(validation_runs)
    running_generation = current_running_validation_generation()
    validation_summary = build_validation_summary(validation_results)
    selected_lora_profile = selected_lora_profile_for_display(job_id, selected_output)
    recommendations = list_recommendations(job_id)
    review_preparation = review_session_summary(job_id, current_session_id=review_session_id)
    operation_monitor = job_operation_monitor(job, review_preparation)
    machine_score_map = score_map_for_samples(job_id)
    machine_epoch_summary = epoch_machine_summary(job_id)
    dataset_version = fetch_one("SELECT * FROM dataset_versions WHERE id = ?", (job["dataset_version_id"],)) if job["dataset_version_id"] else None
    params = json.loads(job["params_json"])
    project = fetch_one("SELECT * FROM lora_projects WHERE id = ?", (job["project_id"],)) if "project_id" in job.keys() and job["project_id"] else None
    project_jobs = fetch_all("SELECT id, name, status FROM training_jobs WHERE project_id = ? ORDER BY id", (project["id"],)) if project else []
    project_selected_job = fetch_one("SELECT id, name FROM training_jobs WHERE id = ?", (project["selected_job_id"],)) if project and project["selected_job_id"] else None
    pilot_guidance = pilot_recommendation(project) if project else None
    return render(
        request,
        "job_detail.html",
        job=job,
        dataset=dataset,
        outputs=outputs,
        samples=samples,
        sample_prompts=sample_prompts,
        sample_groups=group_samples(sample_prompts, samples, candidate_map, review_filter, machine_score_map),
        metrics=metrics,
        metric_rows=metric_table["rows"],
        metric_table=metric_table,
        metric_summary=metric_summary,
        epoch_summaries=decorated_epochs,
        epoch_visual_summaries=build_epoch_visual_summaries(decorated_epochs, samples, candidate_map, machine_epoch_summary),
        review_candidates=review_candidates,
        machine_review_readiness=machine_review_readiness("training_job_samples", job_id),
        machine_review_scores=scores_for_job(job_id),
        machine_epoch_summary=machine_epoch_summary,
        candidate_map=candidate_map,
        candidate_epochs=candidate_epochs,
        review_filter=review_filter if review_filter in {"candidates", "all", "unrated"} else "candidates",
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
        review_preparation=review_preparation,
        operation_monitor=operation_monitor,
        rubric_options=rubric_options(),
        validation_pack_path=validation_pack_path(job_id),
        default_project_path=str(settings.ROOT_DIR),
        dataset_version=dataset_version,
        no_metadata_enabled=bool(params.get("no_metadata")),
        project=project,
        project_jobs=project_jobs,
        project_selected_job=project_selected_job,
        pilot_guidance=pilot_guidance,
        pilot_preset_id=PILOT_PRESET_ID,
        standard_preset_id=STANDARD_PRESET_ID,
        action_state=job_action_state(job, selected_output),
        next_action=recommended_next_action(job, selected_output),
        status_labels=STATUS_LABELS,
        created=created,
        preflight=preflight,
        exported=exported,
        generation_error=generation_error,
        generation_message=generation_message,
        review_prepare=review_prepare,
        review_prepare_error=review_prepare_error,
        running_generation=running_generation,
        sample_embedding_coverage=embedding_coverage("training_job_samples", job_id),
    )


@app.get("/jobs/{job_id}/log-tail/status")
def job_log_tail_status(job_id: int) -> JSONResponse:
    reconcile_stale_running_jobs()
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    log_path = Path(job["run_dir"]) / "logs" / "train.log"
    log_size = 0
    log_updated_at = ""
    if log_path.exists():
        try:
            stat = log_path.stat()
            log_size = stat.st_size
            log_updated_at = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        except OSError:
            log_size = 0
            log_updated_at = ""
    log_tail = read_log_tail(dict(job))
    progress_current, progress_total, progress_label = training_progress_from_log(log_tail)
    return JSONResponse(
        {
            "job_id": job_id,
            "status": job["status"],
            "process_id": job["process_id"],
            "return_code": job["return_code"],
            "log_tail": log_tail,
            "current": progress_current,
            "total": progress_total,
            "progress_label": progress_label,
            "log_size": log_size,
            "log_updated_at": log_updated_at,
        }
    )


@app.post("/jobs/{job_id}/review-preparation/plan")
def job_review_preparation_plan(job_id: int) -> RedirectResponse:
    try:
        session = ensure_candidate_review_plan(job_id, force=True)
    except Exception as exc:
        return RedirectResponse(f"/jobs/{job_id}?review_prepare_error={quote(str(exc))}#review-preparation", status_code=303)
    if session is None:
        return RedirectResponse(f"/jobs/{job_id}?review_prepare_error={quote('候補epochまたは出力LoRAが見つかりません。')}#review-preparation", status_code=303)
    try:
        prepare_review_generation(int(session["id"]))
    except Exception as exc:
        return RedirectResponse(f"/jobs/{job_id}?review_prepare_error={quote(str(exc))}#review-preparation", status_code=303)
    return RedirectResponse(f"/jobs/{job_id}?review_prepare=1#review-preparation", status_code=303)


def _wants_json(request: Request) -> bool:
    return request.headers.get("x-requested-with") == "fetch" or "application/json" in request.headers.get("accept", "")


def _tail_file(path_value: str | None, max_lines: int = 20) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return ""
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:])
    except OSError:
        return ""


@app.post("/jobs/{job_id}/review-preparation/run")
def job_review_preparation_run(request: Request, job_id: int):
    try:
        session = ensure_candidate_review_plan(job_id, force=True)
        if session is None:
            if _wants_json(request):
                return JSONResponse({"ok": False, "message": "候補epochまたは出力LoRAが見つかりません。"}, status_code=400)
            return RedirectResponse(f"/jobs/{job_id}?review_prepare_error={quote('候補epochまたは出力LoRAが見つかりません。')}#review-preparation", status_code=303)
        pid = start_review_preparation(int(session["id"]))
    except Exception as exc:
        if _wants_json(request):
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
        return RedirectResponse(f"/jobs/{job_id}?review_prepare_error={quote(str(exc))}#review-preparation", status_code=303)
    if _wants_json(request):
        return JSONResponse(
            {
                "ok": True,
                "review_session_id": int(session["id"]),
                "message": f"Review Preparationを開始しました。PID: {pid}",
                "redirect_url": f"/jobs/{job_id}#review-preparation",
            }
        )
    return RedirectResponse(f"/jobs/{job_id}?review_prepare={quote(f'Review Preparationを開始しました。PID: {pid}')}#review-preparation", status_code=303)


@app.post("/jobs/{job_id}/review-sessions/{session_id}/run")
def job_review_session_run(request: Request, job_id: int, session_id: int):
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ? AND job_id = ?", (session_id, job_id))
    if session is None:
        if _wants_json(request):
            return JSONResponse({"ok": False, "message": "レビューセッションが見つかりません。"}, status_code=404)
        raise HTTPException(status_code=404, detail="レビューセッションが見つかりません")
    if session["status"] == "running":
        if _wants_json(request):
            return JSONResponse({"ok": False, "message": "レビュー準備は既に実行中です。"}, status_code=400)
        return RedirectResponse(f"/jobs/{job_id}?review_prepare_error={quote('レビュー準備は既に実行中です。')}#review-preparation", status_code=303)
    try:
        pid = start_review_preparation(session_id)
    except Exception as exc:
        if _wants_json(request):
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
        return RedirectResponse(f"/jobs/{job_id}?review_prepare_error={quote(str(exc))}#review-preparation", status_code=303)
    if _wants_json(request):
        return JSONResponse(
            {
                "ok": True,
                "review_session_id": session_id,
                "message": f"レビュー準備を開始しました。PID: {pid}",
                "redirect_url": f"/jobs/{job_id}?review_session_id={session_id}#review-preparation",
            }
        )
    return RedirectResponse(f"/jobs/{job_id}?review_session_id={session_id}&review_prepare={quote(f'レビュー準備を開始しました。PID: {pid}')}#review-preparation", status_code=303)


@app.post("/jobs/{job_id}/review-sessions/{session_id}/stop")
def job_review_session_stop(job_id: int, session_id: int) -> RedirectResponse:
    session = fetch_one("SELECT id FROM review_sessions WHERE id = ? AND job_id = ?", (session_id, job_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Review Session not found")
    try:
        stop_review_preparation(session_id)
    except Exception as exc:
        return RedirectResponse(f"/jobs/{job_id}?review_prepare_error={quote(str(exc))}#review-preparation", status_code=303)
    return RedirectResponse(f"/jobs/{job_id}?review_prepare={quote('Review Preparationを停止しました。')}#review-preparation", status_code=303)


@app.post("/jobs/{job_id}/review-sessions/{session_id}/matrix/build")
def job_review_session_matrix_build(job_id: int, session_id: int) -> RedirectResponse:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ? AND job_id = ?", (session_id, job_id))
    if session is None:
        raise HTTPException(status_code=404, detail="レビューセッションが見つかりません")
    if session["status"] == "running":
        return RedirectResponse(f"/jobs/{job_id}?review_prepare_error={quote('レビュー準備が実行中です。完了後にMatrixを作成してください。')}#review-preparation", status_code=303)
    try:
        matrix_path = write_review_matrix(session_id)
        with connect() as conn:
            conn.execute(
                "UPDATE review_sessions SET matrix_path = ?, updated_at = ? WHERE id = ?",
                (matrix_path, datetime.utcnow().isoformat(), session_id),
            )
    except Exception as exc:
        return RedirectResponse(f"/jobs/{job_id}?review_prepare_error={quote(str(exc))}#review-preparation", status_code=303)
    return RedirectResponse(f"/jobs/{job_id}?review_session_id={session_id}&review_prepare={quote('レビューMatrixを作成しました。')}#review-preparation", status_code=303)


@app.get("/jobs/{job_id}/review-sessions/{session_id}/status")
def job_review_session_status(job_id: int, session_id: int) -> JSONResponse:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ? AND job_id = ?", (session_id, job_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Review Session not found")
    generated = 0
    if session["output_dir"]:
        output_dir = Path(session["output_dir"])
        if output_dir.exists():
            generated = sum(1 for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    imported_count = int(fetch_one("SELECT COUNT(*) AS count FROM review_session_images WHERE review_session_id = ?", (session_id,))["count"] or 0)
    log_path = Path(session["log_path"]) if session["log_path"] else None
    log_size = log_path.stat().st_size if log_path and log_path.exists() else 0
    return JSONResponse(
        {
            "id": session_id,
            "job_id": job_id,
            "status": session["status"],
            "expected_image_count": session["expected_image_count"] or 0,
            "generated_image_count": session["generated_image_count"] or generated,
            "live_generated_image_count": generated,
            "imported_image_count": imported_count,
            "scored_image_count": session["scored_image_count"] or 0,
            "generation_process_id": session["generation_process_id"],
            "return_code": session["return_code"],
            "log_size": log_size,
            "log_tail": _tail_file(session["log_path"], 20),
            "matrix_ready": bool(session["matrix_path"] and Path(session["matrix_path"]).exists()),
            "matrix_url": f"/jobs/{job_id}/review-sessions/{session_id}/matrix" if session["matrix_path"] else "",
            "error_message": session["error_message"] or "",
        }
    )


@app.get("/jobs/{job_id}/review-sessions/{session_id}/matrix", response_class=HTMLResponse)
def job_review_session_matrix(job_id: int, session_id: int) -> HTMLResponse:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ? AND job_id = ?", (session_id, job_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Review Session not found")
    matrix_path = session["matrix_path"]
    if not matrix_path or not Path(str(matrix_path)).exists():
        raise HTTPException(status_code=404, detail="Review Matrix not generated yet")
    matrix_file = Path(str(matrix_path))
    body = matrix_file.read_text(encoding="utf-8", errors="replace")
    project_id = session["project_id"] if "project_id" in session.keys() else None
    nav_parts = [
        '<div class="matrix-actions">',
        f'<a class="button" href="/review-sessions/{session_id}">Review Sessionへ戻る</a>',
        f'<a class="button" href="/jobs/{job_id}#review-preparation">Jobへ戻る</a>',
    ]
    if project_id:
        nav_parts.append(f'<a class="button" href="/projects/{project_id}">Projectへ戻る</a>')
    nav_parts.append('<button type="button" onclick="window.close()">閉じる</button>')
    nav_parts.append("</div>")
    nav = "".join(nav_parts)
    replaced = re.sub(r'<div class="matrix-actions">.*?</div>', nav, body, flags=re.S)
    if replaced == body:
        replaced = body.replace("<body>", f"<body>\n{nav}", 1)
    return HTMLResponse(replaced)


@app.get("/jobs/{job_id}/review-sessions/{session_id}/images/{filename:path}")
def job_review_session_image(job_id: int, session_id: int, filename: str) -> FileResponse:
    session = fetch_one("SELECT * FROM review_sessions WHERE id = ? AND job_id = ?", (session_id, job_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Review Session not found")
    if not session["output_dir"]:
        raise HTTPException(status_code=404, detail="Review Session image directory not found")
    output_dir = Path(session["output_dir"]).resolve()
    try:
        image_path = ensure_allowed_file(str(output_dir / filename), output_dir, "Review Session image")
        verify_image_file(image_path)
    except (PermissionError, FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Review Session image not found")
    return FileResponse(image_path)


@app.get("/jobs/{job_id}/edit", response_class=HTMLResponse)
def job_edit(request: Request, job_id: int) -> HTMLResponse:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    datasets = fetch_all("SELECT * FROM datasets ORDER BY id DESC")
    presets = preset_option_rows(fetch_all("SELECT * FROM presets ORDER BY model_family DESC, name"))
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
    params, reset_to_preset = params_for_selected_preset(job["preset_id"], preset, params)
    if not reset_to_preset:
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
            "project_id": source["project_id"] if "project_id" in source.keys() else None,
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
            "project_id": source["project_id"] if "project_id" in source.keys() else None,
            "sample_prompt_template_id": source["sample_prompt_template_id"] or "",
        }
    )
    copy_sample_prompts(job_id, new_id)
    return RedirectResponse(f"/jobs/{new_id}", status_code=303)


@app.post("/jobs/{job_id}/outputs/{output_id}/select")
def job_select_output(request: Request, job_id: int, output_id: int):
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
    regenerate_epoch_candidates(job_id)
    if wants_json_response(request):
        return JSONResponse(
            {
                "ok": True,
                "job_id": job_id,
                "output_id": output_id,
                "epoch": output["epoch"],
                "file_path": output["file_path"],
                "message": "採用LoRAを更新しました。",
            }
        )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/select-epoch")
def job_select_epoch(request: Request, job_id: int, epoch: int = Form(...), return_to: str = Form("")):
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
    regenerate_epoch_candidates(job_id)
    if wants_json_response(request):
        return JSONResponse(
            {
                "ok": True,
                "job_id": job_id,
                "output_id": output["id"],
                "epoch": output["epoch"],
                "file_path": output["file_path"],
                "message": "採用LoRAを更新しました。",
            }
        )
    return RedirectResponse(safe_local_redirect(return_to, f"/jobs/{job_id}"), status_code=303)


@app.post("/jobs/{job_id}/samples/{image_id}/review")
def job_review_sample(
    request: Request,
    job_id: int,
    image_id: int,
    rating_face: str = Form(""),
    rating_costume: str = Form(""),
    rating_style: str = Form(""),
    rating_stability: str = Form(""),
    rating_flexibility: str = Form(""),
    rating_overall: str = Form(""),
    rating: str = Form(""),
    strength_label: str = Form(""),
    overfit_level: str = Form(""),
    adoption_label: str = Form(""),
    failure_tags: list[str] = Form([]),
    memo: str = Form(""),
):
    sample = fetch_one("SELECT * FROM sample_images WHERE id = ? AND job_id = ?", (image_id, job_id))
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample image not found")
    values = {
        "rating_face": nullable_rating(rating_face),
        "rating_costume": nullable_rating(rating_costume),
        "rating_style": nullable_rating(rating_style),
        "rating_stability": nullable_rating(rating_stability),
        "rating_flexibility": nullable_rating(rating_flexibility),
        "rating_overall": nullable_rating(rating_overall if rating_overall != "" else rating),
    }
    with connect() as conn:
        conn.execute(
            """
            UPDATE sample_images
            SET rating = ?, rating_face = ?, rating_costume = ?, rating_style = ?,
                rating_stability = ?, rating_flexibility = ?, rating_overall = ?, strength_label = ?,
                overfit_level = ?, adoption_label = ?, failure_tags_json = ?,
                rubric_version = ?, memo = ?
            WHERE id = ? AND job_id = ?
            """,
            (
                values["rating_overall"],
                values["rating_face"],
                values["rating_costume"],
                values["rating_style"],
                values["rating_stability"],
                values["rating_flexibility"],
                values["rating_overall"],
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
    if request.headers.get("x-requested-with") == "fetch" or "application/json" in request.headers.get("accept", ""):
        return JSONResponse({"ok": True, "image_id": image_id, "ratings": values})
    return RedirectResponse(f"/jobs/{job_id}#sample-{image_id}", status_code=303)


@app.post("/jobs/{job_id}/samples/review-bulk")
async def job_review_samples_bulk(request: Request, job_id: int) -> JSONResponse:
    payload = await request.json()
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="items is required")
    updated = 0
    with connect() as conn:
        for item in items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            image_id = int(item["id"])
            sample = conn.execute("SELECT id FROM sample_images WHERE id = ? AND job_id = ?", (image_id, job_id)).fetchone()
            if sample is None:
                continue
            values = {
                "rating_face": nullable_rating(item.get("rating_face")),
                "rating_costume": nullable_rating(item.get("rating_costume")),
                "rating_style": nullable_rating(item.get("rating_style")),
                "rating_stability": nullable_rating(item.get("rating_stability")),
                "rating_flexibility": nullable_rating(item.get("rating_flexibility")),
                "rating_overall": nullable_rating(item.get("rating_overall")),
            }
            conn.execute(
                """
                UPDATE sample_images
                SET rating = ?, rating_face = ?, rating_costume = ?, rating_style = ?,
                    rating_stability = ?, rating_flexibility = ?, rating_overall = ?,
                    strength_label = ?, overfit_level = ?, adoption_label = ?,
                    failure_tags_json = ?, rubric_version = ?, memo = ?
                WHERE id = ? AND job_id = ?
                """,
                (
                    values["rating_overall"],
                    values["rating_face"],
                    values["rating_costume"],
                    values["rating_style"],
                    values["rating_stability"],
                    values["rating_flexibility"],
                    values["rating_overall"],
                    clean_choice(str(item.get("strength_label") or ""), {key for key, _ in STRENGTH_LABELS}),
                    clean_choice(str(item.get("overfit_level") or ""), {key for key, _ in OVERFIT_LEVELS}),
                    clean_choice(str(item.get("adoption_label") or ""), {key for key, _ in ADOPTION_LABELS}),
                    json.dumps(clean_failure_tags(item.get("failure_tags")), ensure_ascii=False),
                    RUBRIC_VERSION,
                    item.get("memo") or "",
                    image_id,
                    job_id,
                ),
            )
            updated += 1
    return JSONResponse({"ok": True, "updated": updated})


@app.post("/jobs/{job_id}/review-candidates/regenerate")
def job_regenerate_review_candidates(job_id: int) -> RedirectResponse:
    regenerate_epoch_candidates(job_id)
    return RedirectResponse(f"/jobs/{job_id}?review_filter=candidates#review-queue", status_code=303)


@app.post("/jobs/{job_id}/export-contact-sheet")
def job_export_contact_sheet(job_id: int) -> RedirectResponse:
    path = write_job_contact_sheet(job_id)
    return RedirectResponse(f"/jobs/{job_id}?exported={path}", status_code=303)


@app.post("/jobs/{job_id}/export-selected-lora")
def job_export_selected_lora(job_id: int) -> RedirectResponse:
    ensure_selected_lora_profile(job_id)
    result = export_selected_lora(job_id)
    ensure_selected_lora_profile(job_id)
    suffix = " / hash一致" if result.get("hash_matched") else " / hash未確認"
    return RedirectResponse(f"/jobs/{job_id}?exported={quote(str(result['directory']) + suffix)}", status_code=303)


@app.get("/jobs/{job_id}/cleanup/unselected-models", response_class=HTMLResponse)
def job_cleanup_unselected_preview(request: Request, job_id: int) -> HTMLResponse:
    return render(request, "job_cleanup_preview.html", preview=unselected_model_preview(job_id))


@app.post("/jobs/{job_id}/cleanup/unselected-models")
def job_cleanup_unselected(job_id: int) -> RedirectResponse:
    result = cleanup_outputs(job_id, "unselected_models")
    return RedirectResponse(f"/jobs/{job_id}?exported={quote('未採用LoRAをTrashへ移動: ' + str(result['moved']) + '件 / ' + result['bytes_label'])}", status_code=303)


@app.get("/jobs/{job_id}/cleanup/exported-selected", response_class=HTMLResponse)
def job_cleanup_exported_selected_preview(request: Request, job_id: int) -> HTMLResponse:
    return render(request, "job_cleanup_preview.html", preview=exported_selected_preview(job_id))


@app.post("/jobs/{job_id}/cleanup/exported-selected")
def job_cleanup_exported_selected(job_id: int) -> RedirectResponse:
    try:
        result = cleanup_outputs(job_id, "exported_selected")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}?exported={quote('Export済み採用LoRAのruns側コピーをTrashへ移動: ' + str(result['moved']) + '件')}", status_code=303)


@app.get("/jobs/{job_id}/cleanup/failed-outputs", response_class=HTMLResponse)
def job_cleanup_failed_outputs_preview(request: Request, job_id: int) -> HTMLResponse:
    return render(request, "job_cleanup_preview.html", preview=failed_outputs_preview(job_id))


@app.post("/jobs/{job_id}/cleanup/failed-outputs")
def job_cleanup_failed_outputs(job_id: int) -> RedirectResponse:
    result = cleanup_outputs(job_id, "failed_outputs")
    return RedirectResponse(f"/jobs/{job_id}?exported={quote('失敗/停止ジョブの出力をTrashへ移動: ' + str(result['moved']) + '件 / ' + result['bytes_label'])}", status_code=303)


@app.get("/jobs/{job_id}/cleanup/samples", response_class=HTMLResponse)
def job_cleanup_samples_preview(request: Request, job_id: int, action: str = Query("delete_individual")) -> HTMLResponse:
    return render(request, "job_cleanup_preview.html", preview=sample_cleanup_preview(job_id, action))


@app.post("/jobs/{job_id}/cleanup/samples")
def job_cleanup_samples(job_id: int, action: str = Form("delete_individual")) -> RedirectResponse:
    try:
        result = cleanup_samples(job_id, action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}?exported={quote('サンプル画像をTrashへ移動: ' + str(result['moved']) + '件 / ' + result['bytes_label'])}", status_code=303)


@app.get("/projects/{project_id}/cleanup", response_class=HTMLResponse)
def project_cleanup_preview_page(request: Request, project_id: int, mode: str = Query("unselected_models")) -> HTMLResponse:
    try:
        preview = project_cleanup_preview(project_id, mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return render(request, "project_cleanup_preview.html", preview=preview)


@app.post("/projects/{project_id}/cleanup")
def project_cleanup(project_id: int, mode: str = Form(...)) -> RedirectResponse:
    try:
        result = cleanup_project_outputs(project_id, mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/projects/{project_id}?cleanup={quote(str(result['moved']) + '件 / ' + result['bytes_label'])}", status_code=303)


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


@app.post("/jobs/{job_id}/validation-runs/bulk")
def job_create_validation_runs_for_outputs(
    job_id: int,
    output_ids: list[int] = Form([]),
    validation_preset_id: str = Form(...),
    base_model: str = Form(""),
    trigger_word: str = Form(""),
    memo: str = Form(""),
) -> RedirectResponse:
    if not output_ids:
        raise HTTPException(status_code=400, detail="検証するEpochを1つ以上選択してください。")
    created: list[int] = []
    for output_id in output_ids:
        try:
            run_id = create_validation_run(
                job_id,
                validation_preset_id,
                base_model,
                trigger_word,
                memo,
                selected_output_id=output_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        created.append(run_id)
    if len(created) == 1:
        return_to = quote(f"/jobs/{job_id}#validation-runs", safe="")
        return RedirectResponse(f"/validation-runs/{created[0]}?return_to={return_to}", status_code=303)
    return RedirectResponse(f"/jobs/{job_id}#validation-runs", status_code=303)


@app.get("/validation-runs/{run_id}", response_class=HTMLResponse)
def validation_run_detail(request: Request, run_id: int, generation_error: str | None = None) -> HTMLResponse:
    reconcile_stale_validation_generations()
    try:
        bundle = load_validation_run_bundle(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    requested_job_return = request.query_params.get("return_to", "") if request is not None else ""
    job_return_to = requested_job_return if requested_job_return.startswith("/jobs/") else f"/jobs/{bundle['run']['job_id']}"
    generation_state = generation_view_state(run_id)
    return render(
        request,
        "validation_run_detail.html",
        **bundle,
        job_return_to=job_return_to,
        **generation_state,
        operation_monitor=validation_run_operation_monitor(run_id, generation_state["generation"]),
        default_project_path=str(settings.ROOT_DIR),
        rubric_options=rubric_options(),
        apply_result=None,
        report_path=None,
        generation_error=generation_error,
        running_generation=current_running_validation_generation(),
        validation_embedding_coverage=embedding_coverage("validation_run", run_id),
        machine_review_readiness=machine_review_readiness("validation_run_images", run_id),
        machine_review_scores=scores_for_validation_run(run_id),
        machine_score_map=score_map_for_validation(run_id),
        machine_weight_summary=validation_weight_summary(run_id),
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
    generation_state = generation_view_state(run_id)
    return render(
        request,
        "validation_run_detail.html",
        **bundle,
        **generation_state,
        operation_monitor=validation_run_operation_monitor(run_id, generation_state["generation"]),
        default_project_path=str(settings.ROOT_DIR),
        rubric_options=rubric_options(),
        apply_result=None,
        report_path=report_path,
        validation_embedding_coverage=embedding_coverage("validation_run", run_id),
        machine_review_readiness=machine_review_readiness("validation_run_images", run_id),
        machine_review_scores=scores_for_validation_run(run_id),
        machine_score_map=score_map_for_validation(run_id),
        machine_weight_summary=validation_weight_summary(run_id),
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
    generation_state = generation_view_state(run_id)
    return render(
        request,
        "validation_run_detail.html",
        **bundle,
        **generation_state,
        operation_monitor=validation_run_operation_monitor(run_id, generation_state["generation"]),
        default_project_path=str(settings.ROOT_DIR),
        rubric_options=rubric_options(),
        apply_result=apply_result,
        report_path=None,
        validation_embedding_coverage=embedding_coverage("validation_run", run_id),
        machine_review_readiness=machine_review_readiness("validation_run_images", run_id),
        machine_review_scores=scores_for_validation_run(run_id),
        machine_score_map=score_map_for_validation(run_id),
        machine_weight_summary=validation_weight_summary(run_id),
    )


@app.post("/validation-runs/{run_id}/status")
def validation_run_update_status(run_id: int, status: str = Form(...)) -> RedirectResponse:
    try:
        update_validation_run_status(run_id, status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.post("/validation-runs/{run_id}/generation/prepare")
def validation_generation_prepare(run_id: int) -> RedirectResponse:
    try:
        prepare_validation_generation(run_id)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.post("/validation-runs/{run_id}/generation/run")
def validation_generation_run(run_id: int) -> RedirectResponse:
    try:
        start_validation_generation(run_id)
    except (ValueError, RuntimeError) as exc:
        return RedirectResponse(f"/validation-runs/{run_id}?generation_error={quote(str(exc))}", status_code=303)
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.post("/jobs/{job_id}/validation-runs/{run_id}/generation/run")
def job_validation_generation_run(job_id: int, run_id: int) -> RedirectResponse:
    run = fetch_one("SELECT id FROM validation_runs WHERE id = ? AND job_id = ?", (run_id, job_id))
    if run is None:
        raise HTTPException(status_code=404, detail="Validation Run not found")
    try:
        start_validation_generation(run_id)
    except (ValueError, RuntimeError) as exc:
        return RedirectResponse(f"/jobs/{job_id}?generation_error={quote(str(exc))}#validation-runs", status_code=303)
    return RedirectResponse(f"/jobs/{job_id}#validation-runs", status_code=303)


@app.post("/jobs/{job_id}/validation-runs/generation/run-selected")
def job_validation_generation_run_selected(job_id: int, run_ids: list[int] = Form(default=[])) -> RedirectResponse:
    if not run_ids:
        return RedirectResponse(f"/jobs/{job_id}?generation_error={quote('画像生成する検証Runを選択してください。')}#validation-runs", status_code=303)
    placeholders = ",".join("?" for _ in run_ids)
    rows = fetch_all(
        f"""
        SELECT id FROM validation_runs
        WHERE job_id = ? AND id IN ({placeholders})
        ORDER BY id DESC
        """,
        (job_id, *run_ids),
    )
    valid_run_ids = [int(row["id"]) for row in rows]
    if not valid_run_ids:
        return RedirectResponse(f"/jobs/{job_id}?generation_error={quote('選択した検証Runが見つかりません。')}#validation-runs", status_code=303)
    try:
        count = start_validation_generation_sequence(valid_run_ids)
    except (ValueError, RuntimeError) as exc:
        return RedirectResponse(f"/jobs/{job_id}?generation_error={quote(str(exc))}#validation-runs", status_code=303)
    message = f"選択した検証Run {count} 件の画像生成を順番に開始しました。"
    return RedirectResponse(f"/jobs/{job_id}?generation_message={quote(message)}#validation-runs", status_code=303)


@app.post("/jobs/{job_id}/validation-runs/assist/run-selected")
def job_validation_assist_run_selected(request: Request, job_id: int, run_ids: list[int] = Form(default=[])):
    if not run_ids:
        message = "Embedding計算する検証Runを選択してください。"
        if wants_json_response(request):
            return JSONResponse({"ok": False, "message": message}, status_code=400)
        return RedirectResponse(f"/jobs/{job_id}?generation_error={quote(message)}#validation-runs", status_code=303)
    placeholders = ",".join("?" for _ in run_ids)
    rows = fetch_all(
        f"""
        SELECT id FROM validation_runs
        WHERE job_id = ? AND id IN ({placeholders})
        ORDER BY id DESC
        """,
        (job_id, *run_ids),
    )
    valid_run_ids = [int(row["id"]) for row in rows]
    if not valid_run_ids:
        message = "選択した検証Runが見つかりません。"
        if wants_json_response(request):
            return JSONResponse({"ok": False, "message": message}, status_code=404)
        return RedirectResponse(f"/jobs/{job_id}?generation_error={quote(message)}#validation-runs", status_code=303)
    try:
        count = start_validation_assist_sequence(valid_run_ids)
    except (ValueError, RuntimeError) as exc:
        if wants_json_response(request):
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=409)
        return RedirectResponse(f"/jobs/{job_id}?generation_error={quote(str(exc))}#validation-runs", status_code=303)
    message = f"選択した検証Run {count} 件のEmbedding / 機械補助レビューを順番に開始しました。"
    if wants_json_response(request):
        return JSONResponse({"ok": True, "message": message, "count": count})
    return RedirectResponse(f"/jobs/{job_id}?generation_message={quote(message)}#validation-runs", status_code=303)


@app.post("/validation-runs/{run_id}/generation/stop")
def validation_generation_stop(run_id: int) -> RedirectResponse:
    stop_validation_generation(run_id)
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.post("/jobs/{job_id}/validation-runs/{run_id}/generation/stop")
def job_validation_generation_stop(job_id: int, run_id: int) -> RedirectResponse:
    run = fetch_one("SELECT id FROM validation_runs WHERE id = ? AND job_id = ?", (run_id, job_id))
    if run is None:
        raise HTTPException(status_code=404, detail="Validation Run not found")
    stop_validation_generation(run_id)
    return RedirectResponse(f"/jobs/{job_id}#validation-runs", status_code=303)


@app.get("/validation-runs/{run_id}/generation/status")
def validation_generation_status(run_id: int) -> JSONResponse:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None:
        raise HTTPException(status_code=404, detail="Validation Run not found")
    state = generation_view_state(run_id)
    generation = state["generation"]
    status = generation["status"] if generation else ""
    process_id = generation["process_id"] if generation else None
    actual_row = fetch_one("SELECT actual_image_count, expected_image_count FROM validation_runs WHERE id = ?", (run_id,))
    return JSONResponse(
        {
            "run_id": run_id,
            "status": status,
            "status_label": status or "-",
            "process_id": process_id,
            "process_alive": state["generation_process_alive"],
            "generated_image_count": generation["generated_image_count"] if generation else 0,
            "imported_image_count": generation["imported_image_count"] if generation else 0,
            "actual_image_count": actual_row["actual_image_count"] if actual_row else 0,
            "expected_image_count": actual_row["expected_image_count"] if actual_row else run["expected_image_count"],
            "file_count": state["generation_output_image_count"],
            "log_preview": validation_generation_log_tail(run_id, max_lines=5),
            "log_size": state["generation_log_size"],
            "log_updated_at": state["generation_log_updated_at"],
            "return_code": generation["return_code"] if generation else None,
            "done": status in {"completed", "failed", "stopped"},
        }
    )


@app.get("/validation-runs/{run_id}/assist/status")
def validation_assist_status(run_id: int) -> JSONResponse:
    run = fetch_one("SELECT * FROM validation_runs WHERE id = ?", (run_id,))
    if run is None:
        raise HTTPException(status_code=404, detail="Validation Run not found")
    state = validation_assist_log_state(run_id, max_lines=8)
    embedding = fetch_one(
        """
        SELECT id, status, total_count, processed_count, ready_count, failed_count
        FROM embedding_jobs
        WHERE job_type = 'validation_run' AND target_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (run_id,),
    )
    machine_review = fetch_one(
        """
        SELECT id, status, total_count, processed_count, scored_count, skipped_count, failed_count
        FROM machine_review_jobs
        WHERE target_type = 'validation_run_images' AND target_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (run_id,),
    )
    return JSONResponse(
        {
            "run_id": run_id,
            "exists": state["exists"],
            "log_preview": state["log_preview"],
            "log_size": state["log_size"],
            "log_updated_at": state["log_updated_at"],
            "embedding": dict(embedding) if embedding else None,
            "machine_review": dict(machine_review) if machine_review else None,
        }
    )


@app.post("/validation-runs/{run_id}/generation/import")
def validation_generation_import(run_id: int) -> RedirectResponse:
    try:
        import_generated_images(run_id)
        write_validation_matrix(run_id)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.get("/validation-runs/{run_id}/matrix")
def validation_generation_matrix(run_id: int) -> FileResponse:
    path = generation_validation_run_dir(run_id) / "validation_matrix.html"
    try:
        write_validation_matrix(run_id)
    except Exception as exc:
        if not path.exists():
            raise HTTPException(status_code=404, detail="validation_matrix.html not found") from exc
    return FileResponse(path)


@app.get("/jobs/{job_id}/validation-runs/epoch-matrix", response_class=HTMLResponse)
def validation_epoch_cross_matrix(job_id: int, run_ids: list[int] = Query(default=[])) -> HTMLResponse:
    try:
        html_text = build_epoch_cross_matrix_html(job_id, run_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return HTMLResponse(html_text)


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
    request: Request,
    run_id: int,
    image_id: int,
    rating_face: str = Form(""),
    rating_costume: str = Form(""),
    rating_style: str = Form(""),
    rating_stability: str = Form(""),
    rating_flexibility: str = Form(""),
    rating_overall: str = Form(""),
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
                nullable_rating(rating_face),
                nullable_rating(rating_costume),
                nullable_rating(rating_style),
                nullable_rating(rating_stability),
                nullable_rating(rating_flexibility),
                nullable_rating(rating_overall),
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
    if wants_json_response(request):
        return JSONResponse(
            {
                "ok": True,
                "image_id": image_id,
                "message": "レビューを保存しました。",
            }
        )
    return RedirectResponse(f"/validation-runs/{run_id}", status_code=303)


@app.post("/validation-runs/{run_id}/images/{image_id}/matrix-review")
def validation_run_matrix_review_image(
    run_id: int,
    image_id: int,
    rating_overall: str = Form(""),
    strength_label: str = Form(""),
    adoption_label: str = Form(""),
    memo: str = Form(""),
) -> JSONResponse:
    image = fetch_one("SELECT * FROM validation_images WHERE id = ? AND validation_run_id = ?", (image_id, run_id))
    if image is None:
        raise HTTPException(status_code=404, detail="Validation image not found")
    now = settings_now()
    rating_value = nullable_rating(rating_overall)
    strength_value = clean_choice(strength_label, {key for key, _ in STRENGTH_LABELS})
    adoption_value = clean_choice(adoption_label, {key for key, _ in ADOPTION_LABELS})
    with connect() as conn:
        conn.execute(
            """
            UPDATE validation_images
            SET rating_overall = ?, strength_label = ?, adoption_label = ?,
                rubric_version = ?, memo = ?, updated_at = ?
            WHERE id = ? AND validation_run_id = ?
            """,
            (
                rating_value,
                strength_value,
                adoption_value,
                RUBRIC_VERSION,
                memo.strip(),
                now,
                image_id,
                run_id,
            ),
        )
    update_validation_run_counts(run_id)
    return JSONResponse(
        {
            "ok": True,
            "image_id": image_id,
            "rating_overall": rating_value,
            "strength_label": strength_value,
            "adoption_label": adoption_value,
            "message": "Matrix評価を保存しました。",
        }
    )


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
        SELECT p.*, j.name AS job_name, j.status AS job_status, lp.name AS project_name,
               o.file_size, o.sha256
        FROM selected_lora_profiles p
        LEFT JOIN training_jobs j ON j.id = p.job_id
        LEFT JOIN lora_projects lp ON lp.id = p.project_id
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
        SELECT p.*, j.name AS job_name, j.status AS job_status, lp.name AS project_name
        FROM selected_lora_profiles p
        LEFT JOIN training_jobs j ON j.id = p.job_id
        LEFT JOIN lora_projects lp ON lp.id = p.project_id
        WHERE p.id = ?
        """,
        (profile_id,),
    )
    if profile is None:
        raise HTTPException(status_code=404, detail="LoRA profile not found")
    weight_reviews = fetch_all("SELECT * FROM validation_weight_reviews WHERE job_id = ? ORDER BY lora_weight, id", (profile["job_id"],))
    validation_images = fetch_all("SELECT * FROM validation_images WHERE job_id = ? ORDER BY created_at DESC, id DESC", (profile["job_id"],))
    validation_runs = fetch_all("SELECT * FROM validation_runs WHERE selected_lora_profile_id = ? OR job_id = ? ORDER BY id DESC", (profile_id, profile["job_id"]))
    reference_sets_rows = fetch_all(
        """
        SELECT r.*, v.version_no AS current_version_no, v.completeness_label
        FROM reference_sets r
        LEFT JOIN reference_set_versions v ON v.id = r.current_version_id
        WHERE COALESCE(r.is_archived, 0) = 0
        ORDER BY r.is_default DESC, r.updated_at DESC, r.id DESC
        """
    )
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
    reference_set_version_id: str = Form(""),
    validation_policy_memo: str = Form(""),
    validation_memo: str = Form(""),
    library_memo: str = Form(""),
) -> RedirectResponse:
    now = settings_now()
    selected_reference_version_id = int(reference_set_version_id) if reference_set_version_id else None
    if reference_set_id and selected_reference_version_id is None:
        ref = fetch_one("SELECT current_version_id FROM reference_sets WHERE id = ?", (int(reference_set_id),))
        selected_reference_version_id = int(ref["current_version_id"]) if ref and ref["current_version_id"] else None
    with connect() as conn:
        cur = conn.execute(
            """
            UPDATE selected_lora_profiles
            SET profile_name = ?, trigger_word = ?, base_model = ?,
                recommended_weight_min = ?, recommended_weight_max = ?,
                light_weight = ?, strong_weight = ?,
                default_validation_preset_id = ?, reference_set_id = ?, reference_set_version_id = ?,
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
                selected_reference_version_id,
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
    rows = reference_set_rows()
    datasets = fetch_all("SELECT id, name, trigger_word FROM datasets ORDER BY id DESC")
    projects = fetch_all("SELECT id, name, trigger_word, dataset_id, current_dataset_version_id FROM lora_projects WHERE deleted_at IS NULL ORDER BY updated_at DESC, id DESC")
    return render(
        request,
        "reference_sets.html",
        reference_sets=rows,
        datasets=datasets,
        projects=projects,
        reference_type_labels=REFERENCE_TYPE_LABELS,
    )


@app.post("/reference-sets")
def reference_set_create(
    name: str = Form(...),
    reference_type: str = Form("character"),
    dataset_id: str = Form(""),
    dataset_version_id: str = Form(""),
    project_id: str = Form(""),
    trigger_word: str = Form(""),
    description: str = Form(""),
    selection_mode: str = Form("manual"),
    memo: str = Form(""),
) -> RedirectResponse:
    set_id = create_reference_set(
        name=name,
        reference_type=reference_type,
        dataset_id=int(dataset_id) if dataset_id else None,
        dataset_version_id=int(dataset_version_id) if dataset_version_id else None,
        project_id=int(project_id) if project_id else None,
        trigger_word=trigger_word,
        description=description,
        selection_mode=selection_mode,
        memo=memo,
    )
    return RedirectResponse(f"/reference-sets/{set_id}", status_code=303)


@app.get("/reference-sets/{set_id}", response_class=HTMLResponse)
def reference_set_detail(request: Request, set_id: int) -> HTMLResponse:
    try:
        detail = reference_detail(set_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Reference set not found")
    dataset_versions = fetch_all(
        "SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY version_no DESC",
        (detail["reference_set"]["dataset_id"],),
    ) if detail["reference_set"]["dataset_id"] else []
    ref_coverage = embedding_coverage("reference_set_version", detail["reference_set"]["current_version_id"]) if detail["reference_set"]["current_version_id"] else None
    return render(
        request,
        "reference_set_detail.html",
        **detail,
        dataset_versions=dataset_versions,
        default_project_path=str(settings.ROOT_DIR),
        reference_embedding_coverage=ref_coverage,
        machine_review_readiness=reference_set_readiness(detail["reference_set"], ref_coverage),
    )


@app.post("/reference-sets/{set_id}/images")
def reference_image_add(
    set_id: int,
    image_path: str = Form(...),
    image_role: str = Form("other"),
    prompt_role_hint: str = Form(""),
    caption: str = Form(""),
    source_type: str = Form("manual"),
    include_in_machine_review: str = Form("1"),
    exclude_reason: str = Form(""),
    sort_order: int = Form(0),
    memo: str = Form(""),
) -> RedirectResponse:
    if not isinstance(caption, str):
        sort_order = int(caption)
        caption = ""
    prompt_role_hint = prompt_role_hint if isinstance(prompt_role_hint, str) else ""
    caption = caption if isinstance(caption, str) else ""
    source_type = source_type if isinstance(source_type, str) else "manual"
    exclude_reason = exclude_reason if isinstance(exclude_reason, str) else ""
    memo = memo if isinstance(memo, str) else ""
    include_flag = include_in_machine_review if isinstance(include_in_machine_review, str) else "1"
    try:
        add_reference_image(
            reference_set_id=set_id,
            image_path=image_path,
            image_role=image_role,
            prompt_role_hint=prompt_role_hint,
            caption=caption,
            source_type=source_type,
            include_in_machine_review=include_flag == "1",
            exclude_reason=exclude_reason,
            sort_order=sort_order,
            memo=memo,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/reference-sets/{set_id}", status_code=303)


@app.post("/reference-sets/{set_id}/dataset-images")
def reference_dataset_image_add(
    request: Request,
    set_id: int,
    image_path: str = Form(...),
    image_role: str = Form("other"),
):
    try:
        image_id = add_reference_image(reference_set_id=set_id, image_path=image_path, image_role=image_role, source_type="dataset")
    except ValueError as exc:
        if request.headers.get("x-requested-with") == "fetch":
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if request.headers.get("x-requested-with") == "fetch":
        image = fetch_one("SELECT * FROM reference_images WHERE id = ?", (image_id,))
        return JSONResponse(
            {
                "ok": True,
                "image_id": image_id,
                "image_url": f"/reference-images/{image_id}",
                "image_role": image["image_role"] if image else image_role,
                "width": image["width"] if image else None,
                "height": image["height"] if image else None,
                "file_size": image["file_size"] if image else None,
                "caption": (image["caption_snapshot"] or image["caption"] or "") if image else "",
                "completeness": reference_completeness_payload(set_id),
                "message": "追加済み",
            }
        )
    return RedirectResponse(f"/reference-sets/{set_id}", status_code=303)


@app.get("/reference-sets/{set_id}/dataset-candidates/{candidate_id}")
def reference_dataset_candidate_image(set_id: int, candidate_id: int) -> FileResponse:
    reference_set = fetch_one("SELECT * FROM reference_sets WHERE id = ?", (set_id,))
    if reference_set is None or not reference_set["dataset_id"]:
        raise HTTPException(status_code=404, detail="Reference Set dataset not found")
    candidates = dataset_image_candidates(int(reference_set["dataset_id"]), 80)
    candidate = next((item for item in candidates if int(item["candidate_id"]) == candidate_id), None)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Dataset candidate image not found")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (reference_set["dataset_id"],))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        path = ensure_allowed_file(candidate["path"], Path(dataset["path"]), "Dataset candidate image")
        verify_image_file(path)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dataset candidate image file not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Dataset candidate image file not found or invalid") from exc
    return FileResponse(path)


@app.post("/reference-images/{image_id}/edit")
def reference_image_update(
    image_id: int,
    reference_set_id: int = Form(...),
    image_role: str = Form("other"),
    prompt_role_hint: str = Form(""),
    include_in_machine_review: str = Form(""),
    exclude_reason: str = Form(""),
    memo: str = Form(""),
) -> RedirectResponse:
    try:
        update_reference_image(
            image_id,
            image_role=image_role,
            prompt_role_hint=prompt_role_hint,
            include_in_machine_review=include_in_machine_review == "1",
            exclude_reason=exclude_reason,
            memo=memo,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RedirectResponse(f"/reference-sets/{reference_set_id}", status_code=303)


@app.post("/reference-images/{image_id}/delete")
def reference_image_delete(
    request: Request,
    image_id: int,
    reference_set_id: int = Form(...),
):
    try:
        result = delete_reference_image(image_id)
    except ValueError as exc:
        if request.headers.get("x-requested-with") == "fetch":
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=404)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    target_set_id = result.get("reference_set_id") or reference_set_id
    if request.headers.get("x-requested-with") == "fetch":
        return JSONResponse(
            {
                "ok": True,
                "image_id": image_id,
                "completeness": reference_completeness_payload(int(target_set_id)),
                "message": "取り消しました",
            }
        )
    return RedirectResponse(f"/reference-sets/{target_set_id}", status_code=303)


@app.post("/reference-sets/{set_id}/set-default")
def reference_set_default(set_id: int) -> RedirectResponse:
    try:
        set_project_default(set_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/reference-sets/{set_id}", status_code=303)


@app.post("/reference-sets/{set_id}/archive")
def reference_set_archive(set_id: int) -> RedirectResponse:
    archive_reference_set(set_id, True)
    return RedirectResponse("/reference-sets", status_code=303)


def reference_completeness_payload(set_id: int) -> dict[str, Any]:
    detail = reference_detail(set_id)
    reference_set = detail["reference_set"]
    roles = detail["current_roles"]
    return {
        "label": reference_set["completeness_label"] or "UNKNOWN",
        "message": reference_set["completeness_message"] or "-",
        "roles": [
            {"role": role, "label": ROLE_LABELS.get(role, role), "count": count}
            for role, count in roles.items()
        ],
    }


@app.post("/reference-sets/{set_id}/restore")
def reference_set_restore(set_id: int) -> RedirectResponse:
    archive_reference_set(set_id, False)
    return RedirectResponse(f"/reference-sets/{set_id}", status_code=303)


@app.post("/reference-sets/{set_id}/export", response_class=HTMLResponse)
def reference_set_export(request: Request, set_id: int) -> HTMLResponse:
    try:
        paths = export_reference_artifacts(set_id)
        detail = reference_detail(set_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    ref_coverage = embedding_coverage("reference_set_version", detail["reference_set"]["current_version_id"]) if detail["reference_set"]["current_version_id"] else None
    return render(
        request,
        "reference_set_detail.html",
        **detail,
        dataset_versions=[],
        default_project_path=str(settings.ROOT_DIR),
        export_paths=paths,
        reference_embedding_coverage=ref_coverage,
        machine_review_readiness=reference_set_readiness(detail["reference_set"], ref_coverage),
    )


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
            "project_id": source["project_id"] if "project_id" in source.keys() else None,
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


def nullable_rating(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return clamp_rating(value)


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
    project_defaults = None
    if "project_id" in job.keys() and job["project_id"]:
        project_defaults = fetch_one(
            "SELECT default_reference_set_id, default_reference_set_version_id FROM lora_projects WHERE id = ?",
            (job["project_id"],),
        )
    now = settings_now()
    profile_name = f"Job #{job_id} {job['name']} epoch {output['epoch'] or job['adopted_epoch'] or '-'}"
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO selected_lora_profiles(
                project_id, job_id, selected_output_id, profile_name, trigger_word, selected_epoch,
                selected_model_path, exported_model_path, base_model,
                recommended_weight_min, recommended_weight_max, light_weight, strong_weight,
                validation_memo, library_memo, default_validation_preset_id,
                validation_policy_memo, reference_set_id, reference_set_version_id,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, '', '', ?, ?, ?, ?, ?, ?)
            """,
            (
                job["project_id"] if "project_id" in job.keys() else None,
                job_id,
                output["id"],
                profile_name,
                job["trigger_word_at_creation"] or "",
                output["epoch"] if output["epoch"] is not None else job["adopted_epoch"],
                output["file_path"],
                exported_model_path(job_id),
                base_model_label(job["base_model_path"]),
                "standard_validation_v1",
                "通常比較はHiresなしの標準検証を基準にする。Hiresありは拡張検証で最終見栄え確認として扱う。",
                project_defaults["default_reference_set_id"] if project_defaults else None,
                project_defaults["default_reference_set_version_id"] if project_defaults else None,
                now,
                now,
            ),
        )
        profile_id = int(cur.lastrowid)
    sync_profile_from_validation(job_id)
    return fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (profile_id,))


def selected_lora_profile_for_display(job_id: int, selected_output: Any | None) -> Any:
    if selected_output is None:
        return None
    return fetch_one(
        "SELECT * FROM selected_lora_profiles WHERE job_id = ? AND selected_output_id = ?",
        (job_id, selected_output["id"]),
    )


def sync_profile_selected_fields(job: Any, output: Any, profile_id: int) -> None:
    now = settings_now()
    project_defaults = None
    if "project_id" in job.keys() and job["project_id"]:
        project_defaults = fetch_one(
            "SELECT default_reference_set_id, default_reference_set_version_id FROM lora_projects WHERE id = ?",
            (job["project_id"],),
        )
    with connect() as conn:
        conn.execute(
            """
            UPDATE selected_lora_profiles
            SET selected_epoch = ?, selected_model_path = ?, exported_model_path = ?,
                trigger_word = COALESCE(NULLIF(trigger_word, ''), ?),
                base_model = COALESCE(NULLIF(base_model, ''), ?),
                project_id = COALESCE(project_id, ?),
                reference_set_id = COALESCE(reference_set_id, ?),
                reference_set_version_id = COALESCE(reference_set_version_id, ?),
                updated_at = ?
            WHERE id = ?
            """,
            (
                output["epoch"] if output["epoch"] is not None else job["adopted_epoch"],
                output["file_path"],
                exported_model_path(int(job["id"])),
                job["trigger_word_at_creation"] or "",
                base_model_label(job["base_model_path"]),
                job["project_id"] if "project_id" in job.keys() else None,
                project_defaults["default_reference_set_id"] if project_defaults else None,
                project_defaults["default_reference_set_version_id"] if project_defaults else None,
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


def build_epoch_visual_summaries(epoch_rows: list[dict[str, Any]], samples: list[Any], candidate_map: dict[int, dict[str, Any]] | None = None, machine_epoch_summary: dict[int, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    candidate_map = candidate_map or {}
    machine_epoch_summary = machine_epoch_summary or {}
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
                "candidate_label": candidate_map.get(epoch, {}).get("candidate_label"),
                "candidate_rank": candidate_map.get(epoch, {}).get("candidate_rank"),
                "candidate_reason": candidate_map.get(epoch, {}).get("reason_text"),
                "machine": machine_epoch_summary.get(epoch),
                "avg_loss": loss_row.get("avg_loss"),
                "moving_avg_final_loss": loss_row.get("moving_avg_final_loss"),
                "sample_count": len(epoch_samples),
                "avg_face": average_rating(epoch_samples, "rating_face"),
                "avg_costume": average_rating(epoch_samples, "rating_costume"),
                "avg_style": average_rating(epoch_samples, "rating_style"),
                "avg_stability": average_rating(epoch_samples, "rating_stability"),
                "avg_flexibility": average_rating(epoch_samples, "rating_flexibility"),
                "avg_overall": average_rating(epoch_samples, "rating_overall"),
                "rating_count": sum(1 for sample in epoch_samples if sample_has_rating(sample)),
                "na_count": count_na_ratings(epoch_samples),
                "memo_count": sum(1 for sample in epoch_samples if sample["memo"]),
                "has_rating": has_rating,
                "output_file": loss_row.get("output_file") or "-",
                "output_selected": bool(loss_row.get("output_selected")),
            }
        )
    return rows


def sample_has_rating(sample: Any) -> bool:
    keys = ["rating", "rating_face", "rating_costume", "rating_style", "rating_stability", "rating_flexibility", "rating_overall"]
    return any((rating_value(sample, key) or 0) > 0 for key in keys)


def count_na_ratings(samples: list[Any]) -> int:
    keys = ["rating_face", "rating_costume", "rating_style", "rating_stability", "rating_flexibility", "rating_overall"]
    total = 0
    for sample in samples:
        for key in keys:
            if key in sample.keys() and sample[key] is None:
                total += 1
    return total


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


def group_samples(sample_prompts: list[Any], samples: list[Any], candidate_map: dict[int, dict[str, Any]] | None = None, review_filter: str = "all", machine_score_map: dict[int, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    candidate_map = candidate_map or {}
    machine_score_map = machine_score_map or {}
    candidate_epochs = {epoch for epoch, row in candidate_map.items() if row.get("candidate_label") in {"primary", "secondary", "check"}}
    prompt_map = {row["id"]: row for row in sample_prompts}
    groups: dict[int, dict[str, Any]] = {}
    fallback_key = 0
    for sample in samples:
        sample_item = dict(sample)
        sample_item["machine_score"] = machine_score_map.get(int(sample["id"]))
        epoch = sample_item["epoch"]
        candidate = candidate_map.get(int(epoch)) if epoch is not None else None
        sample_item["candidate_label"] = candidate.get("candidate_label") if candidate else "low_priority"
        sample_item["candidate_rank"] = candidate.get("candidate_rank") if candidate else None
        sample_item["is_candidate_epoch"] = bool(epoch is not None and int(epoch) in candidate_epochs)
        sample_item["is_unrated"] = not sample_has_rating(sample)
        sample_item["prompt_role"] = prompt_role_value(prompt_map.get(sample["prompt_id"]))
        if review_filter == "candidates" and not sample_item["is_candidate_epoch"]:
            continue
        if review_filter == "unrated" and not sample_item["is_unrated"]:
            continue
        sample_item["filename"] = Path(sample["image_path"]).name
        sample_item["failure_tags"] = parse_json_list(sample_item.get("failure_tags_json"))
        key = sample["prompt_id"] or fallback_key
        prompt = prompt_map.get(sample["prompt_id"])
        groups.setdefault(
            key,
            {
                "prompt": prompt,
                "title": prompt["name"] if prompt else "Unmatched prompt",
                "prompt_role": prompt_role_value(prompt),
                "rubric": prompt_role_rubric(prompt_role_value(prompt)),
                "samples": [],
            },
        )
        groups[key]["samples"].append(sample_item)
    for prompt in sample_prompts:
        groups.setdefault(prompt["id"], {"prompt": prompt, "title": prompt["name"], "prompt_role": prompt_role_value(prompt), "rubric": prompt_role_rubric(prompt_role_value(prompt)), "samples": []})
    for group in groups.values():
        group["samples"].sort(key=lambda item: (
            item["epoch"] if item["epoch"] is not None else 999999,
            item["step"] if item["step"] is not None else 999999,
            item["created_at"],
            item["id"],
        ))
    return sorted(groups.values(), key=lambda group: group["prompt"]["sort_order"] if group["prompt"] else 999999)


def prompt_role_value(prompt: Any) -> str:
    if prompt is None:
        return "other"
    return prompt["prompt_role"] if "prompt_role" in prompt.keys() and prompt["prompt_role"] else "other"


def prompt_role_rubric(role: str | None) -> dict[str, str]:
    role = role or "other"
    rubrics = {
        "face": {"face": "required", "costume": "optional", "style": "required", "stability": "required", "flexibility": "optional"},
        "full_body": {"face": "optional / N/A可", "costume": "required", "style": "optional", "stability": "required", "flexibility": "required"},
        "expression_pose": {"face": "required", "costume": "optional", "style": "required", "stability": "required", "flexibility": "required"},
        "clothes": {"face": "optional", "costume": "required", "style": "required", "stability": "required", "flexibility": "optional"},
        "background": {"face": "optional", "costume": "optional", "style": "required", "stability": "required", "flexibility": "required"},
    }
    return rubrics.get(role, {"face": "optional", "costume": "optional", "style": "optional", "stability": "required", "flexibility": "optional"})


def build_loss_chart(metrics: list[Any]) -> dict[str, Any] | None:
    loss_rows = [row for row in metrics if row["loss"] is not None and row["step"] is not None]
    if len(loss_rows) < 2:
        return None
    source_count = len(loss_rows)
    loss_rows = downsample_rows(loss_rows, 1200)
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
        "source_count": source_count,
        "point_count": len(loss_rows),
    }


def moving_average(values: list[float], window: int) -> list[float]:
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(sum(values[start:index + 1]) / (index - start + 1))
    return result


def downsample_rows(rows: list[Any], max_points: int) -> list[Any]:
    if len(rows) <= max_points:
        return rows
    last_index = len(rows) - 1
    indexes = {round(index * last_index / (max_points - 1)) for index in range(max_points)}
    return [rows[index] for index in sorted(indexes)]


def build_metric_table(metrics: list[Any], head: int = 3, tail: int = 40) -> dict[str, Any]:
    total = len(metrics)
    if total <= head + tail:
        return {
            "rows": metrics,
            "total": total,
            "shown": total,
            "head": min(head, total),
            "tail": max(0, total - head),
            "omitted": 0,
            "limited": False,
        }
    rows = metrics[:head] + metrics[-tail:]
    return {
        "rows": rows,
        "total": total,
        "shown": len(rows),
        "head": head,
        "tail": tail,
        "omitted": total - len(rows),
        "limited": True,
    }
