# LoRA-Studio

Stable Diffusion系LoRA、特にSDXL/SD1.5の2Dキャラクター顔LoRA学習をローカルで管理するための支援ツールです。MVPでは自動最適化ではなく、学習実験の作成、実行準備、ログ、loss健全性、サンプル画像比較、採用epoch判断を一元管理することを目的にします。

## MVP範囲

- FastAPI + Jinja2 + SQLite のローカルWebアプリ
- SQLite DB初期化
- built-inプリセットseed登録
- データセット登録と簡易スキャン
- 学習ジョブdraft作成
- `dataset_config.toml`、`sample_prompts.txt`、`command.txt` 生成
- `sd-scripts v0.10.5` 固定の環境構築スクリプト

## 前提環境

- Windows 11
- Python 3.10系推奨
- Git
- NVIDIA GPU環境ではRTX 50系を想定し、初期CUDA profileは `cu128`

## アプリセットアップ

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_app.ps1
```

## 起動

```powershell
.\.venv\Scripts\python.exe .\start_lora_helper.py
```

ブラウザで `http://127.0.0.1:8768` を開きます。

起動時に指定ポートが既に使用されている場合、WindowsではそのポートをLISTENしている既存プロセスを終了してから起動します。

## sd-scripts環境構築

MVPでは `kohya-ss/sd-scripts` の最新リリースとして確認した `v0.10.5` を使用します。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_sd_scripts.ps1 -ReleaseTag v0.10.5 -CudaProfile cu128 -MixedPrecision bf16
```

このスクリプトは `external/sd-scripts` にcloneまたはfetchし、tag `v0.10.5` をcheckoutします。リリースページでは `v0.10.5` が Latest と表示され、対応commitは `a1b48df` です。

## 初回学習の流れ

1. Environment画面で `sd-scripts` の設定方針を確認する。
2. Datasets画面で画像フォルダを登録する。
3. Presets画面で初期プリセットを確認する。
4. New Job画面でデータセット、プリセット、base model pathを指定してdraftを作る。
5. Job詳細画面で `Prepare Files` を押し、設定ファイルと実行コマンドを生成する。
6. 生成された `runs/job_000001/config/command.txt` を確認して学習実行に進む。

## 既知の制限

- 学習プロセスの実行、停止、結果取り込みは次フェーズで実装する。
- loss解析とepoch比較UIは枠だけ用意している。
- AIによる画像自動評価やパラメータ自動最適化はMVP対象外。
