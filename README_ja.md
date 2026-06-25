# LoRA-Studio

[English README](README.md)

## 概要

LoRA-Studioは、Stable Diffusion系LoRA、特にSDXL / SD1.5のキャラクター・イラスト向けLoRA制作をローカルで管理するためのワークフロー支援ツールです。

自動で創作判断を置き換えるのではなく、Dataset、学習Job、Validation、候補レビュー、Recipe / Optimizer、採用LoRA選定を再現しやすい形でまとめて扱うことを目的にしています。

## スクリーンショット

スクリーンショットはOSS応募向けにサニタイズした英語デモ画面です。ローカル開発環境の表示やサンプルデータとは一部異なる場合があります。

### Dashboard

![Dashboard](docs/screenshots/dashboard.png)

### Recommended Workflow

![Recommended Workflow](docs/screenshots/recommended-workflow.png)

### Create Training Job

![Create Training Job](docs/screenshots/create-training-job.png)

### Training Job Management

![Training Job Management](docs/screenshots/training-job-management.png)

### Training Result Management

![Training Result Management](docs/screenshots/training-result-management.png)

## 主な機能

- Project単位のLoRA実験管理
- Dataset登録、再スキャン、trigger確認、Dataset Version snapshot
- 学習Jobの作成、準備、実行、停止、複製、アーカイブ
- Recipe v2 / Optimizer Master、Step Estimator、Compatibility Check
- sd-scripts `networks.lora` の `conv_dim` / `conv_alpha` を使うLoRA-C3Lier Recipe
- Post-training Review AutomationとCandidate Standard Comparison
- Review Matrixと人間評価欄による候補epoch選定
- 採用LoRA向けValidation Run / Weight Calibration Pipeline
- OpenCLIP / Machine Review AssistとReference Set
- Retry Signal SummaryとRecommendation Engineの分離
- 大容量生成物向けRuntime Storage設定とCleanup支援
- 日本語 / 英語スクリーンショット向けの段階的i18n基盤

## 推奨ワークフロー

1. 1つのLoRA制作単位としてProjectを作成します。
2. Datasetを登録または再スキャンし、caption / trigger整合性を確認します。
3. 学習前にDataset Versionを作成します。
4. Recipe Wizardまたはlegacy presetからTraining Jobを作成します。
5. ファイル準備とPreflight確認後、sd-scriptsで学習を実行します。
6. Review Session / Candidate Reviewで候補epochを比較します。
7. 採用LoRAを選び、Weight Calibration / Validationを実行します。
8. 推奨weight範囲をLoRA Profileへ反映します。
9. Export、Archive、不要な大容量出力のCleanupを行います。

## Quick Start

アプリ依存関係をセットアップします。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_app.ps1
```

LoRA-Studioを起動します。

```bat
start_lora_studio.bat
```

ブラウザで開きます。

```text
http://127.0.0.1:8768
```

必要に応じて検証済みsd-scripts環境をセットアップします。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_sd_scripts.ps1 -ReleaseTag v0.10.5 -CudaProfile cu128 -MixedPrecision bf16
```

## Demo DB / Screenshot Workflow

README、docs、OSS応募用スクリーンショット向けに、サニタイズ済みDemo DBを生成できます。

```powershell
python scripts/create_demo_db.py --output demo/demo.sqlite
```

Demo DBを読み取り専用Demo modeで起動します。

```powershell
python start_lora_studio.py --db demo/demo.sqlite --demo --no-browser
```

英語UIでスクリーンショットを撮る場合は以下を開きます。

```text
http://127.0.0.1:8768/?lang=en
```

Demo modeには架空のProject名、Dataset名、画像、レポート、パスだけが入ります。実学習、画像生成、削除などの書き込み操作はブロックされます。Demo runtimeや生成スクリーンショットはGit管理外です。Playwrightが使える環境では、以下で主要画面を撮影できます。

```powershell
python scripts/capture_demo_screenshots.py --base-url http://127.0.0.1:8768 --db demo/demo.sqlite
```

## ドキュメント

- [日本語ドキュメント目次](docs/ja/index.md)
- [初回セットアップ](docs/ja/getting-started.md)
- [基本ワークフロー](docs/ja/workflow.md)
- [Recipe / Optimizer / Step Estimator](docs/ja/recipes-optimizers.md)
- [Troubleshooting](docs/ja/troubleshooting.md)
- 分割前のREADME_ja全文は説明ロスト防止のため [archive](docs/ja/_archive_readme_ja_before_split.md) に保存しています。

## 現在の状態

現在のリリース: v0.5.4-beta
開発フェーズ: Phase 12.4.5

中核ワークフローはローカルLoRA制作で実運用できる状態ですが、ベータ期間中のためAPI、画面導線、Recipe catalogは変更される可能性があります。

## 必要環境

- Windowsローカル運用
- `scripts/setup_app.ps1` で作成するPython仮想環境
- SQLiteアプリDB
- kohya-ss/sd-scripts連携。ベータ運用では `v0.10.5` を検証基準にしています。
- SDXL / SD1.5 LoRA学習に適したNVIDIA GPU環境

## 注意事項

- Machine Review Assistは補助情報です。identity、衣装細部、画風、採用判断は人間の目視評価を優先します。
- Smoke Test / Mini Pilot OKは起動確認または短時間実学習の確認であり、最終品質を保証しません。
- LoRA-C3Lierはsd-scripts標準LoRAを3x3 Conv2d層へ拡張する方式として扱います。`networks.lora` と `conv_dim` / `conv_alpha` を使い、LyCORIS LoConとは別実装です。
- 大容量model、`runs`、`exports`、logs、embedding cacheは可能ならOneDriveなどの同期フォルダ外へ置いてください。
- sd-scripts log、生成command、raw args、tracebackはi18n翻訳対象外です。

## ライセンス

[LICENSE](LICENSE) を参照してください。
