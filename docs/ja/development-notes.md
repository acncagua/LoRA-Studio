# Development Notes

## テスト

基本チェック:

```powershell
python -m compileall app start_lora_studio.py
node --check app/static/js/app.js
python -m pytest -q -p no:cacheprovider
git diff --check
```

## i18n

UI翻訳辞書は `app/i18n/ja.json` と `app/i18n/en.json` に分離しています。新しい画面文言はテンプレートへ直書きせず翻訳キーを追加してください。

Action stateは `label_key` / `description_key` を優先します。`ACTION_TEXT_KEYS` は既存日本語文言互換のfallbackです。新規actionでは主導線に使わないでください。

sd-scripts log、生成command、raw args、tracebackは翻訳対象外です。

## DB / migration

既存DBを壊さないmigrationを優先します。新規カラムはnullableまたはdefault付きにし、legacy Job / legacy preset / existing Review Sessionが開ける状態を維持してください。

## Git運用

ユーザーが明示した場合のみtag / release / pushを行います。作業中に未追跡生成物が出た場合は、正式保存するか `.gitignore` へ追加するかを判断し、不要な生成物はコミットに混ぜないでください。
