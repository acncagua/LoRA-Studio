import json
from pathlib import Path

from starlette.requests import Request

from app.main import job_new
from app.services.i18n import I18N_DIR, language_url, load_translations, localized_json_value, reload_i18n, translate


def make_request(query_string: bytes = b"", cookie: str = "") -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if cookie:
        headers.append((b"cookie", cookie.encode("utf-8")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/jobs/new",
            "root_path": "",
            "scheme": "http",
            "query_string": query_string,
            "headers": headers,
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
    )


def test_translate_defaults_to_japanese_and_falls_back_to_english() -> None:
    assert translate("wizard.mode.purpose", "ja") == "用途から選ぶ"
    assert translate("wizard.mode.purpose", "en") == "Choose by Purpose"
    assert translate("missing.key", "ja", "fallback") == "fallback"
    assert translate("missing.key", "en") == "missing.key"


def test_translation_json_files_load_without_bom() -> None:
    reload_i18n()
    translations = load_translations()

    assert translations["ja"]["wizard.mode.purpose"] == "用途から選ぶ"
    assert translations["en"]["wizard.mode.purpose"] == "Choose by Purpose"
    for locale in ("ja", "en"):
        path = Path(I18N_DIR) / f"{locale}.json"
        assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)


def test_language_url_handles_template_render_without_request() -> None:
    assert language_url(None, "en") == "?lang=en"


def test_localized_json_value_uses_locale_then_english_fallback() -> None:
    value = {"ja": "顔キャラ・標準", "en": "Character Face Balanced"}
    assert localized_json_value(value, "ja") == "顔キャラ・標準"
    assert localized_json_value(value, "en") == "Character Face Balanced"
    assert localized_json_value({"en": "Fallback"}, "ja") == "Fallback"


def test_jobs_new_english_query_sets_locale_cookie() -> None:
    response = job_new(make_request(b"lang=en"), project_id="", mode="")
    body = response.body.decode("utf-8")

    assert response.status_code == 200
    assert "Create Training Job" in body
    assert "Creation Method" in body
    assert "locale=en" in response.headers["set-cookie"]


def test_jobs_new_uses_locale_cookie() -> None:
    response = job_new(make_request(cookie="locale=en"), project_id="", mode="")
    body = response.body.decode("utf-8")

    assert response.status_code == 200
    assert "Create Training Job" in body
    assert "Creation Method" in body


def test_jobs_new_japanese_query_overrides_cookie() -> None:
    response = job_new(make_request(b"lang=ja", cookie="locale=en"), project_id="", mode="")
    body = response.body.decode("utf-8")

    assert response.status_code == 200
    assert "学習ジョブ作成" in body
    assert "作成方法" in body
    assert "locale=ja" in response.headers["set-cookie"]
