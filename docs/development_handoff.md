# LoRA-Studio 開発引き継ぎメモ

作成日時: 2026-06-21  
対象リポジトリ: `D:\OnlineStrage\OneDrive\ドキュメント\Codex\LoRA-Studio`  
実作業パス: `D:\OnlineStrage\OneDrive\ドキュメント\Codex\LoRA-Studio`  
注意: `D:\OnlineStrage\OneDrive\ドキュメント\LoRA-Studio` は別フォルダです。通常の開発・改修作業は必ず `ドキュメント\Codex\LoRA-Studio` 側で行ってください。  
現在のHEAD: `cfb768f` (`v0.4.1`, `phase11.6.16.2`)  
現在のアプリ表示バージョン: `v0.4.1`  
DB schema version: `2026.06.12-beta`

この文書は、新しいChatへLoRA-Studioの開発・改修を引き継ぐためのまとめです。  
実装詳細は必ず現在のコードを正として確認してください。

## 1. プロダクト概要

LoRA-Studioは、Stable Diffusion系LoRA、主にSDXL / SD1.5の2DキャラクターLoRA制作をローカルで管理するFastAPI + Jinja2 + SQLiteアプリです。

目的はLoRA学習そのものの自動最適化ではなく、以下を一元管理することです。

- Dataset登録、検査、caption整備
- Dataset Version管理
- 学習ジョブ作成、実行、停止、ログ保存
- loss / metrics / step整合性確認
- 出力LoRAとsample画像の取り込み
- 採用前epoch比較用Review Session
- 採用後weight検証用Validation Run
- Reference Set / Reference Version管理
- Embedding Cache
- Machine Review Assist
- LoRA Library / 採用LoRA Profile管理
- Recommendation / 次回実験提案
- Storage cleanup / Archive / Delete

現在は運用ベータ段階です。実運用に使われていますが、画面導線や内部APIはまだ変更されます。

## 2. 現在の重要な概念整理

### Project

1つのLoRA作成単位です。  
Dataset、Dataset Version、Reference Set、Training Job、Review Session、Validation Run、採用LoRA Profileを束ねます。

関連テーブル:

- `lora_projects`
- `reference_sets`
- `training_jobs`
- `review_sessions`
- `validation_runs`
- `selected_lora_profiles`
- `experiment_recommendations`

### Training Job

1回の学習実行です。  
学習設定、sd-scripts実行、ログ、metrics、出力LoRA、sample画像を保持します。

関連テーブル:

- `training_jobs`
- `training_outputs`
- `sample_prompts`
- `sample_images`
- `training_metrics`
- `training_metric_summaries`
- `training_epoch_summaries`
- `training_epoch_candidate_summaries`
- `environment_snapshots`

### Review Session

採用前に候補epochを比較するための作業単位です。  
loss候補epochとその前後epochから、少数prompt / seed / weightで画像生成し、Review Matrixで横並び比較します。

Validation Runとは役割が違います。

- Review Session: 採用epochを決める前の候補比較
- Validation Run: 採用epoch決定後のweight検証

関連テーブル:

- `review_sessions`
- `review_session_conditions`
- `review_session_images`
- `machine_review_scores`
- `machine_review_jobs`

### Validation Run

採用LoRAに対して、weight別・prompt別・seed別の見え方を確認する検証単位です。  
Quick / Standard / Extended のValidation Presetから作成します。

関連テーブル:

- `validation_runs`
- `validation_expected_conditions`
- `validation_images`
- `validation_generation_runs`
- `validation_weight_reviews`
- `validation_results`
- `validation_presets`

### Reference Set

Machine Review AssistでReference Similarityを計算するための基準画像セットです。  
キャラLoRAでは3〜5枚以上、画風LoRAでは6〜12枚程度を推奨する思想です。

関連テーブル:

- `reference_sets`
- `reference_set_versions`
- `reference_images`

### Embedding / Machine Review

画像embeddingを計算し、Reference類似度やDataset近傍類似度を補助情報として表示します。  
人間評価を置き換えません。

Provider:

- `mock_image_512`: CI / 機能経路テスト用
- `transformers_clip`: `openai/clip-vit-base-patch32`
- `open_clip`: `ViT-B-32` / `laion2b_s34b_b79k`

関連テーブル:

- `embedding_models`
- `embedding_settings`
- `image_embeddings`
- `embedding_jobs`
- `embedding_job_items`
- `machine_review_settings`
- `machine_review_scores`
- `machine_review_jobs`

## 3. 主要ディレクトリ

```text
app/
  main.py                         FastAPIルートの中心
  db.py                           SQLite migration / seed / DB helper
  settings.py                     パス・設定
  app_version.py                  app version / git hash / DB schema
  services/                       ドメイン処理
  templates/                      Jinja2テンプレート
  templates/partials/             共通部分テンプレート
  static/css/app.css              UI CSS
  static/js/app.js                AJAX / polling / UI補助

data/
  app.db                          ローカルDB、Git対象外
  embeddings/                     embedding npy cache、Git対象外

datasets/                         実データセット、Git対象外
models/                           base model、Git対象外
external/sd-scripts/              kohya-ss sd-scripts、Git対象外
runs/                             学習実行成果物、Git対象外
exports/                          export / validation / reports、Git対象外
logs/                             setup / worker logs、Git対象外
trash/                            app trash、Git対象外
backups/                          backup出力、Git対象外
scripts/                          setup / oneoff / maintenance scripts
tests/                            pytest / unittest tests
docs/                             補助ドキュメント、スクリーンショット
```

## 4. Git対象外の重要ディレクトリ

`.gitignore` で以下は除外されています。

- `data/*.db`
- `data/embeddings/`
- `models/`
- `datasets/*`
- `logs/*.log`
- `logs/embeddings/`
- `logs/machine_review/`
- `server*.log`
- `runs/job_*/`
- `runs/comparisons/`
- `external/sd-scripts/`
- `exports/*`
- `backups/`
- `trash/`

注意:

- 実モデル、実Dataset、生成画像、学習成果物はGitに入れない。
- release zipにも `__pycache__`、`data/embeddings`、`exports`、`trash` は含めない方針。

## 5. 主要サービスファイル

### app/main.py

FastAPIルートの中心です。  
画面表示、POST操作、画像配信、ジョブ起動、Validation Run、Review Session、Embedding、Machine Reviewなど多くを抱えています。

肥大化しているため、今後はサービス層へ移す余地があります。

### app/db.py

SQLite DBの初期化、migration、seed処理を持ちます。

重要:

- 既存DB互換を壊さないこと。
- `ALTER TABLE ADD COLUMN` 型の簡易migrationが多い。
- 特定ユーザーの実データID依存処理は本体に入れない方針。oneoff scriptへ分離すること。

### app/services/command_builder.py

sd-scripts学習コマンド生成。  
`command.txt` は表示用、`command_argv.json` が実行用です。

注意:

- 実行はshell文字列ではなくargv配列。
- `venv_python_path -m accelerate.commands.launch ...` を優先。
- `cwd` は sd-scripts ディレクトリ。
- SDXLは `sdxl_train_network.py`、SD1.5は `train_network.py`。

### app/services/training_runner.py

学習ジョブ実行 / 停止 / background処理。  
Windowsでは停止に `taskkill /PID <pid> /T /F` を使います。

### app/services/metrics_collector.py

TensorBoard / train.log からloss metricsを取り込み、summaryを作成します。

### app/services/output_collector.py

`runs/job_xxxxxx/models` の `.safetensors`、`runs/job_xxxxxx/samples` の画像をDBへ取り込みます。

### app/services/validation_runs.py

Validation Run作成、Expected Condition、画像登録、weight reviewなど。

### app/services/validation_generation.py

sd-scripts `gen_img.py` を使ったValidation画像生成。  
Quick / Standard / Extended の検証画像を生成します。

現在のValidation Preset思想:

- Quick: Hiresなし、3 prompts × 2 seeds × 3 weights = 18枚
- Standard: Hiresなし、3 prompts × 3 seeds × 5 weights = 45枚
- Extended: Hiresなし / Hiresあり、3 prompts × 3 seeds × 5 weights × 2 = 90枚

sd-scripts側のsampler表記は `euler_a` などに正規化が必要です。

### app/services/review_sessions.py

採用前epoch比較のReview Sessionを扱います。

役割:

- candidate epoch条件生成
- Review Session画像生成
- 画像DB登録
- embedding / machine review連携
- Review Matrix HTML出力

### app/services/review_candidates.py

lossやサンプル評価から候補epochを抽出するロジック。

### app/services/embedding_service.py

Embedding provider管理、preflight、embedding保存、coverage、stale判定。

### app/services/embedding_worker.py

Embedding Job worker。  
CLI:

```powershell
python -m app.services.embedding_worker --embedding-job-id <id>
```

### app/services/machine_review.py

Reference Similarity / Dataset Nearest Similarity / assist label / overfit riskの計算。

### app/services/machine_review_worker.py

Machine Review Job worker。  
CLI:

```powershell
python -m app.services.machine_review_worker --machine-review-job-id <id>
```

### app/services/reference_sets.py

Reference Set / Version / completeness / contact sheet / report。

### app/services/operation_monitor.py

Active Operation Monitor用のログtail、mtime、状態表示補助。

### app/services/storage_cleanup.py

runs配下のモデル・sample cleanup、Trash、storage usage。

## 6. 主要テンプレート

### base.html

全画面共通レイアウト、左メニュー、version表示。  
`request=None` でも落ちないよう注意が必要です。

### dashboard.html

Project / Job / status summary / cleanup link。

### project_detail.html

Projectを作業中心にする方向で整理中。  
Current Status、Next Action、Training Jobs、Review Sessions、Validation Runs、Selected LoRA、Storageなど。

### job_detail.html

Training Job詳細。  
現在はタブ構成:

- 概要
- 結果
- レビュー
- 検証
- その他

過去の肥大化を抑えるため、Job全体の主要操作と各タブ内操作を分ける方針です。

重要:

- Job全体のPrimary ActionとReview Session内のPrimary Actionを混同しない。
- Review Preparation表示は1つのCurrent Review Sessionに限定する。
- planned / completed / matrixあり / matrixなし の状態別に文言を変える。
- stale query messageが残らないよう注意。

### review_session_detail.html

Review Session詳細。  
候補epoch、画像生成状態、Embedding / Machine Review状態、Review Matrix、epoch採用導線。

planned状態ではepoch採用を表示しない、または無効化する方針。

### validation_run_detail.html

Validation Run詳細。  
Coverage、画像確認、Weight Review、Machine Assist、Apply to Profileなど。

### reference_set_detail.html

Reference Set詳細。  
候補画像表示、Reference画像一覧、completeness、embedding coverage、Machine Review readiness。

### embedding_settings.html

Embedding Provider設定、preflight、job一覧、cache状況。

### partials/active_operation_monitor.html

長時間処理の短いログtail表示。  
Training / Review Preparation / Validation Generation / Embedding / Machine Reviewで共通化するためのpartialです。

## 7. フロントエンドJS

`app/static/js/app.js` にAJAX / polling / UI補助が集約されています。

主な役割:

- Review Preparation開始後のpolling
- Validation Run画像レビューのAJAX保存
- LoRA採用ボタンのAJAX化
- Reference画像追加/削除のAJAX
- Embedding / Machine Review開始後のメッセージ表示
- Matrix画像倍率切り替えと100%ポップアップ
- 一時メッセージ用クエリパラメータ削除

最近直した重要点:

- Review Preparationが完了済みなのに「開始中...」が残る問題を修正。
- `review_prepare` / `review_prepare_error` を一時メッセージ削除対象に追加。
- 完了後に「レビューMatrixを開く」へ切り替える処理を追加。

## 8. 起動と環境

通常起動:

```bat
start_lora_studio.bat
```

手動起動:

```powershell
.\.venv\Scripts\python.exe .\start_lora_studio.py
```

開発時にsd-scripts setupを避ける場合:

```powershell
.\.venv\Scripts\python.exe .\start_lora_studio.py --skip-sd-scripts-setup
```

ポート:

- デフォルト: `8768`
- 通常起動では既存プロセスをkillしない。
- 強制解放する場合のみ `--force-release-port`。
- 対象はLoRA-Studio指定portのみ。

sd-scripts:

- 確認済み: `kohya-ss/sd-scripts v0.10.5`
- 対応commit: `a1b48df`
- 配置: `external/sd-scripts`
- venv: `external/sd-scripts/venv`

Python検出:

1. `data/python_cmd.txt`
2. `LORA_STUDIO_PYTHON_EXE` / `LORA_STUDIO_PYTHON`
3. `%LOCALAPPDATA%\Programs\Python\...`
4. `%ProgramFiles%\Python...`
5. `py -3.10`
6. `py -3.12`
7. PATH上の `python`

特定ユーザー名を含む絶対パスはハードコードしない。

## 9. テスト・確認コマンド

基本:

```powershell
python -m compileall app start_lora_studio.py
python -m pytest -q -p no:cacheprovider
git diff --check
```

補足:

- `pytest -q` ではなく `python -m pytest -q` を推奨。
- OpenCLIP / transformers / GPU依存は環境差があるため、通常テストでは `-p no:cacheprovider` を使うことが多い。
- FastAPI `on_event` deprecation warning は既知。

現在直近確認:

- `python -m compileall app start_lora_studio.py` 成功
- `python -m pytest -q -p no:cacheprovider` 成功: `73 passed`
- `git diff --check` 成功

## 10. DB主要テーブル対応表

### Core

- `app_settings`
- `environments`
- `environment_snapshots`

### Dataset

- `datasets`
- `dataset_analysis`
- `dataset_versions`
- `caption_edit_history`

### Training

- `presets`
- `training_jobs`
- `training_outputs`
- `training_metrics`
- `training_metric_summaries`
- `training_epoch_summaries`
- `training_epoch_candidate_summaries`
- `sample_prompts`
- `sample_images`
- `sample_prompt_templates`

### Project / Profile

- `lora_projects`
- `selected_lora_profiles`

### Review Session

- `review_sessions`
- `review_session_conditions`
- `review_session_images`

### Validation

- `validation_presets`
- `validation_runs`
- `validation_expected_conditions`
- `validation_generation_runs`
- `validation_images`
- `validation_weight_reviews`
- `validation_results`

### Reference

- `reference_sets`
- `reference_set_versions`
- `reference_images`

### Embedding / Machine Review

- `embedding_models`
- `embedding_settings`
- `image_embeddings`
- `embedding_jobs`
- `embedding_job_items`
- `machine_review_settings`
- `machine_review_jobs`
- `machine_review_scores`

### Recommendation

- `recommendation_rules`
- `experiment_recommendations`
- `evaluation_rubrics`

### Storage / Cleanup

- `file_cleanup_history`

## 11. 開発・改修履歴

### v0.1-beta / Phase 10系

基礎ワークフローを構築。

- DB初期化
- built-in preset
- Dataset登録
- Job draft作成
- Prepare Files
- sd-scripts setup
- Job Run / Stop / log保存
- training_outputs / sample_images取り込み
- loss / metrics取り込み
- Job比較
- sample rating / memo
- selected LoRA
- Validation Preset / Validation Run
- LoRA Library
- Recommendation Engine
- Storage cleanup
- Maintenance / Backup / Diagnostics

### Phase 10.7.x

運用ベータ前の安定化。

- 画像配信安全化
- Validation条件固定
- Legacy Validation整理
- oneoff script分離
- sqlite connection close整理
- validation image / reference image管理root化
- Archive / Delete / Cleanup導線

### v0.2.0-beta / Phase 10.11.1

sd-scripts `gen_img.py` によるValidation画像生成を安定化。

- sampler正規化
- 生成中ログ表示
- 生成完了後のボタン復帰
- Validation Matrix改善
- READMEのOSS向け整理

### Phase 11.1

Reference Set / Reference Versionを追加。

- Reference画像のrole管理
- completeness表示
- Project / Profile / Validation RunへのReference紐づけ
- Reference Contact Sheet / Markdown Report

### Phase 11.2

Embedding Cache基盤。

- `embedding_models`
- `embedding_settings`
- `image_embeddings`
- `embedding_jobs`
- `embedding_job_items`
- mock provider
- Dataset / Reference / Sample / Validation coverage
- stale / missing_source

### Phase 11.3

Machine Review Assist初期版。

- `machine_review_scores`
- `machine_review_jobs`
- `machine_review_settings`
- Reference Similarity
- Dataset Nearest Similarity
- overfit risk
- assist label / confidence label
- Job / Validation Run / Reference Setへの表示

### Phase 11.3.3

Machine Reviewをbackground job化し、Readiness導線を追加。

- request=None耐性
- dataset image embedding衝突対策
- Machine Review worker化
- Readiness Panel
- Machine Review Jobs一覧

### Phase 11.4

`transformers_clip` provider追加。

- `openai/clip-vit-base-patch32`
- cuda優先、cpu fallback
- cpu fp32
- allow_model_download制御
- Training / Validation Generation中の実provider開始拒否

### Phase 11.5 / v0.3.0-beta

OpenCLIP provider追加。

- provider: `open_clip`
- model: `ViT-B-32`
- pretrained: `laion2b_s34b_b79k`
- active provider切り替え

### Phase 11.6

Training Job完了後のReview Preparation Pipeline。

- `review_sessions`
- `review_session_conditions`
- `review_session_images`
- loss候補epochからReview Plan作成
- Candidate Review画像生成
- Embedding
- Machine Review
- Review Matrix HTML生成
- Job詳細Reviewタブ
- Review Session詳細
- Project / Job / Review Session導線整理

### Phase 11.6.15〜11.6.16

情報設計と導線整理。

- Projectを作業中心へ
- Job詳細をタブ化
- Review Sessionを採用前epoch評価の中心へ
- Validation Runを採用後weight検証へ
- Review Matrixの戻り導線
- Review Session状態別ボタン文言
- Primary Actionの意味整理

### Phase 11.6.16.2 / v0.4.1

Matrix表示とValidation Run一括処理改善。

- Matrix画像倍率 25 / 50 / 75 / 100
- 画像クリック時100%ポップアップ
- 複数Validation Runの一括画像生成
- 複数Validation RunのEmbedding / Machine Review一括処理
- 簡易ログ折りたたみ

## 12. 現在の重要な既知課題

### 非同期処理完了後のUI更新残り

症状:

- 処理は完了しているのにボタンが「開始中...」のまま残る。
- 件数やステータスが更新されず、手動リロードが必要に見える。
- URLクエリに一時メッセージが残る。

直近修正:

- Review Preparation完了後、開始中ボタンを戻す。
- Matrix作成済みなら「レビューMatrixを開く」へ切り替える。
- `review_prepare` / `review_prepare_error` を一時メッセージ削除対象へ追加。
- 完了済みなのに古い「レビュー準備を開始しました。PID...」を表示しないようテンプレート側で保護。

未確認:

- Validation Run一括処理
- Embedding一括処理
- Machine Review一括処理
- Dataset / Reference Set embedding
- Active Operation Monitor

### Review Session情報混在

過去に、planned sessionとcompleted sessionの数値が混ざって表示される問題があった。  
方針:

- Review Preparationメイン表示は必ず1つのCurrent Review Sessionに限定。
- completed / matrix-ready / runningを優先。
- planned sessionは別枠表示。
- candidate_epochs、condition_count、registered_image_count、machine_review_count、matrix_pathは同一 `review_session_id` から取る。

### Triggerの取り扱い

新規Project作成 / 既存Project追加 / dataset trigger / project trigger / job triggerの混同が起きやすい。

注意:

- 既存Projectに追加する場合、Project triggerを正とする。
- 新規Projectの場合、既存Projectコンボは非表示または無効にする方が自然。
- Job作成時にDataset Versionが別Datasetのものを選べてしまうと不整合になるため、Dataset選択に応じてVersion候補を絞るべき。

### Validation PresetとHires

ExtendedはHiresなし + Hiresありで90枚という設計。  
Hiresあり画像はStandard比較ではなく、最終見栄え確認の意味合い。

sd-scripts `gen_img.py` のHires互換はWebUI完全一致ではないため、「近い目安」として扱う。

### Matrix画像表示

現在:

- 倍率切り替えあり
- 画像クリックで100%ポップアップ
- 横スクロール前提

要注意:

- Hiresあり/なしが混ざると画像サイズ差で見づらくなるため、Matrix側に通常 / Hires表示切り替えが必要。

## 13. 実装時の注意

- 日本語ファイルはUTF-8で保存。
- 実データ、モデル、runs、exports、trash、embeddingsはGit対象外。
- 既存DBを壊さない。migrationは後方互換重視。
- 特定ユーザーの実データIDやtrigger名を本体seedに入れない。
- sd-scripts本体は変更しない。
- Pythonパスに特定ユーザー名をハードコードしない。
- UI文言は基本日本語。用語揺れ注意。
  - Jobは「学習ジョブ」
  - Projectは「Project（LoRA作成単位）」または文脈上「Project」
  - Review Sessionは「レビューセッション」
  - Validation Runは「検証Run」
  - Embedding / Machine Reviewは必要に応じてカタカナや併記
- Machine Reviewは自動判定ではなく補助情報。
- 人間評価を最優先する。
- mock providerのスコアは意味評価ではない。

## 14. 今後の改修で優先度が高いもの

1. 非同期処理の完了状態UIを横断的に点検する。
2. Active Operation MonitorをTraining / Review / Validation / Embedding / Machine Reviewで統一する。
3. Dataset選択に応じたDataset Version候補の絞り込み。
4. 新規Project作成時は既存Projectコンボを非表示または無効化。
5. draft Project / draft Jobの安全な削除導線。
6. Review Session / Validation Run一括処理の進捗とログ表示の統一。
7. HiresありMatrixの表示切り替え。
8. README詳細をdocsへ分割。
9. FastAPI `on_event` をlifespanへ移行。

## 15. 現在の作業ツリー注意

このメモ作成時点で、作業ツリーには既存の未コミット変更が多数あります。

確認された変更ファイル例:

- `README_ja.md`
- `app/db.py`
- `app/main.py`
- `app/services/command_builder.py`
- `app/services/training_runner.py`
- `app/services/validation_generation.py`
- `app/services/validation_runs.py`
- `app/static/js/app.js`
- `app/templates/dataset_detail.html`
- `app/templates/job_create.html`
- `app/templates/job_detail.html`
- `app/templates/jobs.html`
- `app/templates/project_detail.html`
- `app/templates/projects.html`
- `app/templates/reference_set_detail.html`
- `app/templates/validation_run_detail.html`
- `start_lora_studio.py`

新しいChatでは、必ず `git status --short` と `git diff` を確認してから作業を始めてください。  
ユーザー作業や前回作業の途中変更を勝手に戻さないでください。

## 16. 直近の調査メモ

直近の表示残り問題については以下も参照してください。

- `docs/current_investigation_status.md`

概要:

- Job #31 / Review Session #14 で、Review Preparation完了済みなのに「開始中...」が残った。
- 実体はcompletedで、Matrixも作成済み。
- UI側の一時状態とURLメッセージが残る問題として修正。
- `/jobs/31` HTTP 200、古い開始メッセージなし、Matrixリンクありを確認済み。
