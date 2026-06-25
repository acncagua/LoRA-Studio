# Development Notes

## Demo DB / Screenshot Workflow

README、README_ja、docs用スクリーンショットには、実運用DBではなくDemo DBを使います。

```powershell
python scripts/create_demo_db.py --output demo/demo.sqlite
python start_lora_studio.py --db demo/demo.sqlite --demo --no-browser
```

英語表示で撮影する場合は `http://127.0.0.1:8768/?lang=en` を開きます。`?lang=en` はlocale cookieへ保存されるため、左ナビで遷移しても英語表示を維持できます。

Demo DBには以下だけを入れます。

- 架空のProject / Dataset / Job / Review Session / Validation Run / LoRA Profile
- `demo/fixtures/images` の合成PNG
- `demo/fixtures/reports` のサニタイズ済みHTML
- Optimizer / Recipeの公開向け状態例

実モデル、実LoRA、実パス、個人メモ、実ログは入れません。Demo modeでは学習、画像生成、削除などの書き込み操作をブロックします。

Playwrightが使える場合は、主要画面の英語スクリーンショットを以下で撮影できます。

```powershell
python scripts/capture_demo_screenshots.py --base-url http://127.0.0.1:8768 --db demo/demo.sqlite
```

生成される `demo/demo.sqlite`、`demo/runtime/`、`demo/screenshots/generated/` はGit管理外です。

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
