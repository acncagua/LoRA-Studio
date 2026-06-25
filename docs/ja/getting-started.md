# 初回セットアップ

## 前提

LoRA-StudioはWindowsローカル運用を前提にしています。アプリ本体はFastAPI / Jinja2 / SQLiteで動き、学習や画像生成はkohya-ss/sd-scriptsを呼び出します。

ベータ運用ではsd-scripts `v0.10.5` を検証基準にしています。GPU、CUDA、PyTorch、sd-scriptsの組み合わせはローカル環境の影響を強く受けます。

## アプリ依存関係

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_app.ps1
```

通常起動:

```bat
start_lora_studio.bat
```

手動起動:

```powershell
.\.venv\Scripts\python.exe .\start_lora_studio.py
```

ブラウザで `http://127.0.0.1:8768` を開きます。

既存プロセスがポートを掴んでいる場合だけ、明示的にポート解放付きで起動できます。

```powershell
.\.venv\Scripts\python.exe .\start_lora_studio.py --force-release-port
```

## sd-scripts環境

検証済みsd-scripts環境を作る場合:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_sd_scripts.ps1 -ReleaseTag v0.10.5 -CudaProfile cu128 -MixedPrecision bf16
```

LoRA-Studio管理下のsd-scripts環境では、DAdapt / Prodigy / Lion向けのoptional optimizer dependenciesも標準導入対象です。既存外部sd-scripts環境は勝手に変更せず、Environment画面のInstall操作を押した場合のみsd-scripts venvへ導入します。

## 初回に確認すること

- Dashboardが開くこと
- `/workflow` で推奨フローが見えること
- `/storage` でruntime rootやOneDrive warningを確認できること
- `/optimizers` と `/training-recipes` が開くこと
- Dataset登録先、base model、sample prompt templateの場所が分かること
