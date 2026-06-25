from app.services.i18n import language_url, localized_json_value, translate


def test_translate_defaults_to_japanese_and_falls_back_to_english() -> None:
    assert translate("wizard.mode.purpose", "ja") == "用途から選ぶ"
    assert translate("wizard.mode.purpose", "en") == "Choose by Purpose"
    assert translate("missing.key", "ja", "fallback") == "fallback"


def test_language_url_handles_template_render_without_request() -> None:
    assert language_url(None, "en") == "?lang=en"


def test_localized_json_value_uses_locale_then_english_fallback() -> None:
    value = {"ja": "顔キャラ・標準", "en": "Character Face Balanced"}
    assert localized_json_value(value, "ja") == "顔キャラ・標準"
    assert localized_json_value(value, "en") == "Character Face Balanced"
    assert localized_json_value({"en": "Fallback"}, "ja") == "Fallback"
