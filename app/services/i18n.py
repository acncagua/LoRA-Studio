from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import Request


DEFAULT_LOCALE = "ja"
FALLBACK_LOCALE = "en"
SUPPORTED_LOCALES = {"ja", "en"}
I18N_DIR = Path(__file__).resolve().parents[1] / "i18n"


ACTION_TEXT_KEYS = {'Projectを開く': 'primary.open_project',
 'Projectへ戻る': 'common.back',
 'Project内の学習ジョブ一覧': 'primary.open_project_jobs',
 '採用中の学習ジョブを開く': 'primary.open_selected_job',
 '現在の採用学習ジョブを開く': 'primary.open_selected_job',
 'このProjectに学習ジョブを追加': 'primary.add_job_to_project',
 'LoRAプロファイル': 'primary.open_profile_short',
 'LoRAプロファイルへ': 'primary.open_profile',
 '学習ジョブを編集': 'primary.edit_job',
 'ファイル準備': 'primary.prepare_files',
 'ファイル準備を再実行': 'primary.prepare_files_again',
 '事前確認': 'primary.preflight',
 '実行': 'primary.run',
 '停止': 'primary.stop',
 '結果を再取り込み': 'primary.reimport',
 'LoRAを選択': 'primary.select_lora',
 'この学習ジョブの選択LoRAをProject採用に反映': 'primary.sync_project_selection',
 '学習ジョブを複製': 'primary.clone_job',
 '派生ドラフトを作成': 'primary.create_derived_draft',
 '比較': 'primary.compare',
 '採用LoRA出力': 'primary.export_selected_lora',
 'コンタクトシート出力': 'primary.export_contact_sheet',
 'アーカイブ': 'primary.archive',
 'アーカイブから復元': 'primary.restore',
 '復元': 'primary.restore',
 '削除': 'primary.delete',
 '標準候補比較の進捗を確認': 'primary.check_standard_progress',
 '標準候補比較Matrixを開く': 'primary.open_standard_matrix',
 'レビュー準備の進捗を確認': 'primary.check_review_progress',
 '候補レビューを開始': 'primary.start_review',
 '候補レビューを作成': 'primary.create_candidate_review',
 'このプランで候補レビューを生成': 'primary.generate_review_plan',
 '進捗を確認': 'primary.progress',
 '進捗確認': 'primary.progress',
 'レビューMatrixを開く': 'primary.open_review_matrix',
 'Review Matrixを開く': 'primary.open_review_matrix',
 'レビューMatrixを作成': 'primary.create_review_matrix',
 'Review Matrixを作成': 'primary.create_review_matrix',
 'レビュー準備をリトライ': 'primary.retry_review_preparation',
 'ログ確認': 'primary.check_logs',
 'ログを確認': 'primary.check_logs',
 'weight検証Runを作成': 'primary.create_weight_validation',
 'Weight検証を開始': 'primary.start_weight_validation',
 'Weight検証を準備': 'primary.prepare_weight_validation',
 'Weight Review Matrixを開く': 'primary.open_weight_matrix',
 'Weight Review Matrixを作成': 'primary.create_weight_matrix',
 'Profileへ反映': 'primary.apply_profile',
 '再試行': 'primary.retry',
 '画像生成＋不足レビュー計算': 'primary.generate_images_and_review',
 '画像生成を停止': 'primary.stop_generation',
 'Reimport': 'primary.reimport',
 '検証レポート出力': 'primary.export_report',
 'Matrix再作成': 'primary.rebuild_matrix',
 '詳細': 'primary.open_details',
 'ジョブへ': 'primary.open_job_short',
 'Jobを開く': 'primary.open_job',
 '検証Runを開く': 'primary.open_validation',
 'Matrixを開く': 'primary.open_matrix',
 '概要を確認': 'primary.open_details',
 '再準備が必要': 'primary.prepare_files_again',
 '事前確認 / 実行': 'primary.run',
 'ログ確認 / 停止': 'primary.check_logs',
 '評価 / Export / 比較': 'primary.compare',
 'ログ確認 / 修正': 'primary.check_logs',
 '再準備 / 実行': 'primary.prepare_files_again',
 '詳細確認': 'primary.open_details'}


@lru_cache(maxsize=1)
def load_translations() -> dict[str, dict[str, str]]:
    translations: dict[str, dict[str, str]] = {}
    for locale in SUPPORTED_LOCALES:
        path = I18N_DIR / f"{locale}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        translations[locale] = {str(key): str(value) for key, value in data.items()}
    return translations


def reload_i18n() -> None:
    load_translations.cache_clear()


def normalize_locale(value: str | None) -> str:
    text = (value or "").split(",", 1)[0].split("-", 1)[0].strip().lower()
    return text if text in SUPPORTED_LOCALES else DEFAULT_LOCALE


def request_locale(request: Request | None) -> str:
    if request is None:
        return DEFAULT_LOCALE
    query_lang = request.query_params.get("lang")
    if query_lang:
        return normalize_locale(query_lang)
    cookie_lang = request.cookies.get("locale")
    if cookie_lang:
        return normalize_locale(cookie_lang)
    return DEFAULT_LOCALE


def translate(key: str, locale: str = DEFAULT_LOCALE, default: str | None = None) -> str:
    normalized = normalize_locale(locale)
    translations = load_translations()
    return (
        translations.get(normalized, {}).get(key)
        or translations.get(FALLBACK_LOCALE, {}).get(key)
        or default
        or key
    )


def translate_action_text(text: str | None, locale: str = DEFAULT_LOCALE) -> str:
    value = text or ""
    key = ACTION_TEXT_KEYS.get(value)
    if not key:
        return value
    return translate(key, locale, value)


def language_url(request: Request | None, locale: str) -> str:
    if request is None:
        return f"?lang={normalize_locale(locale)}"
    parts = urlsplit(str(request.url))
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["lang"] = normalize_locale(locale)
    return urlunsplit(("", "", parts.path, urlencode(query), parts.fragment))


def localized_json_value(value: Any, locale: str, fallback: str | None = None) -> str:
    if not value:
        return fallback or ""
    data = value
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return fallback or value
    if not isinstance(data, dict):
        return fallback or str(value)
    normalized = normalize_locale(locale)
    return str(data.get(normalized) or data.get(FALLBACK_LOCALE) or data.get(DEFAULT_LOCALE) or fallback or "")
