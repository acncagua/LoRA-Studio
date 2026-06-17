# Changelog

## v0.3.0-beta - 2026-06-17

Release Version: v0.3.0-beta
Internal Milestone: Phase 11.5

LoRA Studio v0.3.0-beta は、Phase 11.5 時点の運用ベータリリースです。

### Phase 11.6

- Training Job完了後のReview Preparation Pipelineを追加しました。
- loss候補epochとその前後epochから `review_sessions` / `review_session_conditions` を作成し、候補epochレビュー用の少数条件を固定できるようにしました。
- Candidate Epoch Review presetとして、`basic_face` / `full_body` / `expression_pose`、seed `111111`、weight `0.6 / 0.8`、Hiresなしの条件を生成します。
- sd-scripts `gen_img.py` による候補画像生成、生成画像DB登録、Embedding計算、Machine Review Assist、Review Matrix HTML生成をbackground処理としてつなぎました。
- Job詳細にReview Preparationパネルを追加し、状態、候補epoch、画像数、Embedding coverage、Machine Review coverage、ログ末尾、停止ボタン、Review Matrixリンクを表示します。
- 採用前のReview Preparationと、採用後のWeight Calibration / Validation RunをUI上で分けて表示するようにしました。
- Review Sessionのplan作成とMatrix出力に対する最小テストを追加しました。

### Phase 11.5

- OpenCLIP provider `open_clip` を試験導入しました。
- 標準モデルを `ViT-B-32` / `laion2b_s34b_b79k` とし、512次元の画像embeddingを保存できるようにしました。
- `allow_model_download=true` の場合のみOpenCLIPのpretrained weight downloadを許可します。
- deviceはcuda優先、fallbackでcpu、cpuではdtypeをfp32に固定します。
- `transformers_clip` と同様、学習中または検証画像生成中はOpenCLIP Embedding Jobを開始しないようにしました。
- mock providerはCI/テスト用として継続し、OpenCLIPとMachine Review Assistでprovider/modelを分けて保存します。

### Phase 11.4.1

- transformers_clip Machine Review Calibrationとして、Reference role分布、epoch別Dataset近傍傾向、CLIP ViT-B/32の読み方に関する注意文言を追加しました。
- Job #12向けCalibration Reportを `runs/job_000012/reports/machine_review_calibration_job_000012.md` に出力しました。
- Machine Assistは人間評価を置き換えず、顔専用判定ではないことをREADMEとUIで明確化しました。

### Phase 11.4

- 実embedding providerの初回実装として `transformers_clip` を追加しました。
- 標準モデルを `openai/clip-vit-base-patch32` とし、512次元の画像embeddingを保存できるようにしました。
- `allow_model_download=true` の場合のみ `from_pretrained` によるモデルdownloadを許可します。
- deviceはcuda優先、fallbackでcpu、cpuではdtypeをfp32に固定します。
- GPU競合を避けるため、学習中または検証画像生成中は実providerのEmbedding Jobを開始しないようにしました。
- mock providerはCI/テスト用として継続します。

### Phase 11.3

- Reference Similarity Assist / Machine Review Assist初期版を追加しました。
- `machine_review_scores`、`machine_review_jobs`、`machine_review_settings` を追加しました。
- Sample画像とValidation画像について、Reference Setとの平均/最大類似度、nearest reference、Dataset nearest similarity、top1 margin、overfit risk、assist label、confidence labelを保存できるようにしました。
- mock provider利用時は、スコアを意味評価として扱わないようlow confidence表示にしました。
- Job詳細、Validation Run詳細、Reference Set詳細、Embedding設定にMachine Review Assist関連の表示と実行導線を追加しました。
- READMEにMachine Review Assist、Reference Similarity、Dataset Nearest Similarity、mock providerの注意、人間評価優先の説明を追加しました。

### Phase 11.2

- Embedding Cache / Feature Extraction Foundationを追加しました。
- `embedding_models`、`embedding_settings`、`image_embeddings`、`embedding_jobs`、`embedding_job_items` を追加しました。
- テスト用の `mock_image_512` providerをseed登録し、外部モデルなしでDataset / Reference / Sample / Validation画像のembeddingを計算できるようにしました。
- embedding workerを `python -m app.services.embedding_worker --embedding-job-id <id>` で実行できるようにし、embedding本体は `.npy` として `data/embeddings/` に保存します。
- Dataset詳細、Reference Set詳細、学習ジョブ詳細、Validation Run詳細にEmbedding Coverageを表示しました。
- Embedding設定画面でactive model、provider preflight、cache size、job一覧を確認できるようにしました。
- Maintenance / StorageにEmbedding cacheの容量表示とcleanup previewを追加しました。
- Phase11.2ではreference similarity、prompt alignment、aesthetic score、overfit risk、Machine Review Assist採点、ChatGPT API連携、AI画像評価はまだ実装していません。

### Phase 11.1

- Reference Set / Reference Version管理を追加しました。
- 既存Reference Setをv1へ移行し、既存Reference画像をversionへ紐づけるmigrationを追加しました。
- Reference画像にsource、caption snapshot、tags、画像サイズ、file size、sha256、Machine Review対象フラグ、除外理由、メモを保存できるようにしました。
- Character / Style / Mixed向けの役割とCompleteness表示を追加しました。
- Reference Set一覧・詳細・画像追加・役割編集・Project標準設定・Archive/Restore導線を整備しました。
- Project、LoRA Profile、Validation RunへReference Set / Versionの紐づけを追加しました。
- Reference Contact Sheet HTMLとMarkdown Report出力を追加しました。
- Phase11.1ではembedding、CLIP/OpenCLIP/DINO/SigLIP、ChatGPT API、AI画像評価、WebUI API自動生成は未実装です。

## v0.2.0-beta - 2026-06-15

Release Version: v0.2.0-beta
Internal Milestone: Phase 10.11.1

LoRA Studio v0.2.0-beta は、Phase 10.11.1 時点の運用ベータリリースです。

### 主な変更

- sd-scripts `gen_img.py` を使ったValidation画像生成を安定化しました。
- Validation生成時のsampler表記をsd-scripts互換値へ正規化しました。
- 生成中ログの簡易表示とAJAX更新を改善しました。
- 生成完了後に画像生成ボタンが復帰しないUI問題を修正しました。
- Validation Run / Matrix / レビューまわりの操作導線を改善しました。
- READMEのOSS向けスクリーンショットと説明を整理しました。
- one-offのローカル移行処理をアプリ本体から分離しました。

### 注意

- WebUI / reForge APIによる自動生成は未実装です。
- sd-scripts `gen_img.py` による標準Validation画像生成には対応しています。
- ChatGPT API連携、AI画像評価、FLUX、LyCORIS/LoConは未実装です。

## v0.1-beta - 2026-06-12

LoRA-Studioを運用ベータ前の安定化状態として記録します。目的は、SDXL/SD1.5の2DキャラクターLoRA学習を、Dataset整備から学習、sample確認、採用LoRA管理、外部Validation、次回実験提案までローカルで一貫して追跡できるようにすることです。

### 主要機能

- Dataset登録、Rescan、caption検査、trigger consistency確認
- Dataset version管理とcaption編集履歴
- built-in学習プリセット、Pilot 3 Epoch、Standard 6 Epoch
- sd-scripts v0.10.5環境セットアップと環境情報表示
- Job作成、Prepare Files、Run、Stop、ログ保存、成果物取り込み
- loss/metrics取り込み、step整合性、epoch summary
- sample画像一覧、prompt/epoch比較、rating/memo保存
- Job比較UI、比較Markdown/Contact Sheet出力
- selected LoRA export、LoRA Library Profile管理
- Validation Preset、Reference Set、Validation Run、Coverage Matrix
- Rubricベースの人間評価、Weight Review Matrix、suggested weight反映
- Recommendation Engineと提案からdraft Job作成
- Maintenance画面、軽量Backup、Diagnostics出力
- Dashboard/Footerのversion/git/DB/sd-scripts情報表示

### 未実装機能

- WebUI APIによる自動txt2img生成
- ChatGPT API連携
- AI画像評価、顔類似度判定
- 自動リトライ
- 複数Job自動探索、複数Job同時実行
- FLUX対応
- LyCORIS/LoCon対応

### Known Issues

- Codex内ブラウザはWindows sandbox権限で `CreateProcessAsUserW failed: 5` になる場合があります。HTTP画面確認で代替可能です。
- HiresありValidationは標準比較ではなく、最終見栄え確認用です。
- Validation Runでregistered数がexpected未満の場合は、Recommendationの信頼度に注意してください。
- Backup初期版は大型モデル、画像、動画、zipを除外し、DB、設定、exports内の軽量ファイル、reportsを中心に保存します。

### Phase10.7.1 Stabilization

- 起動時のポート解放をアプリ指定ポートに限定し、`7865` を既定でkillしないように整理しました。
- Validation / Reference画像配信を管理ディレクトリ配下に制限し、Reference画像は管理rootへコピー保存する運用へ寄せました。
- Validation Presetのsnapshot保存とExpected Condition固定化を追加し、旧Validation Pack系UIをLegacy扱いに整理しました。

### Phase10.7.2 Emergency Stabilization

- Validation Run画像登録時の `sqlite3.Row.get` 問題を修正し、Expected Condition一致時に `expected_condition_id` が入るようにしました。
- Legacy External Validation画像とValidation Result画像の管理rootコピーを統一し、Validation Resultは画像なし登録を維持しました。
- 既存Validation RunのExpected Condition backfillと、既存画像パスを管理rootへ移行するoneoff scriptを追加しました。

### Phase10.7.3 Stabilization

- SQLite接続のclose漏れを減らし、`fetch_all()` / `fetch_one()` を明示的にcloseする形へ整理しました。
- 画像登録済みValidation RunではExpected Condition不一致があっても自動DELETE/再生成せず、既存condition_hashを保護するようにしました。
- Validation / Reference画像配信時にPIL verifyを行い、壊れた画像や非画像を返さないようにしました。
- ユーザー入力パスの前後空白・引用符を正規化し、Windowsダイアログ由来の引用符付きパスを受けられるようにしました。
