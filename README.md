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

起動時に `external/sd-scripts` または `external/sd-scripts/venv` が未作成の場合は、`sd-scripts v0.10.5` のセットアップを自動実行します。

venv作成に使うPythonは、`data/python_cmd.txt`、`LORA_STUDIO_PYTHON_EXE` / `LORA_STUDIO_PYTHON`、環境変数から生成したPython 3.10候補、`py -3.10`、環境変数から生成したPython 3.12候補、`py -3.12`、スキャンで見つけたその他Python、PATH上の `python` の順に探します。Codex通常サンドボックスでは `py` がユーザーインストールを見つけられない場合があるため、`py` だけには依存しません。

## sd-scripts環境構築

MVPでは `kohya-ss/sd-scripts` の最新リリースとして確認した `v0.10.5` を使用します。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_sd_scripts.ps1 -ReleaseTag v0.10.5 -CudaProfile cu128 -MixedPrecision bf16
```

このスクリプトは `external/sd-scripts` にcloneまたはfetchし、tag `v0.10.5` をcheckoutします。リリースページでは `v0.10.5` が Latest と表示され、対応commitは `a1b48df` です。

アプリだけを起動して `sd-scripts` セットアップをスキップしたい場合は、検証用に以下を使えます。

```powershell
.\.venv\Scripts\python.exe .\start_lora_helper.py --skip-sd-scripts-setup
```

Pythonを手動指定したい場合は、以下のいずれかを使えます。

```powershell
# 1回だけ指定する場合
powershell -ExecutionPolicy Bypass -File .\scripts\setup_sd_scripts.ps1 -PythonCmd "C:\path\to\python.exe"

# アプリ設定として保存する場合
Set-Content -Encoding UTF8 .\data\python_cmd.txt "C:\path\to\python.exe"
```

## 初回学習の流れ

1. Environment画面で `sd-scripts` の設定方針を確認する。
2. Datasets画面で画像フォルダを登録する。
3. Dataset詳細画面で画像、caption、trigger word、タグ傾向を検査する。
4. Presets画面で初期プリセットを確認する。
5. New Job画面でデータセット、プリセット、base model path、必要ならSample Prompt Templateを指定してdraftを作る。
6. Job詳細画面で `Prepare Files` を押し、設定ファイルと実行コマンドを生成する。
7. `Run` を押して学習を開始する。必要なら `Stop` で停止する。
8. 完了後、Job詳細画面でログ、出力LoRA、サンプル画像を確認する。手動で再取り込みしたい場合は `Reimport Results` を押す。

`Prepare Files` では表示用の `command.txt` に加えて、実行用の `command_argv.json` を生成します。実行時はshell文字列ではなくargv配列を `subprocess.Popen` に渡すため、Windowsのスペース入りパスでも壊れにくくしています。`dataset_config.toml` の `batch_size` はプリセットの `train_batch_size` から生成し、コマンドライン側では `--train_batch_size` を重複指定しません。学習時のbatch sizeは `dataset_config.toml` を正とします。

## Dataset Inspector

Datasets画面のIDリンクからDataset詳細を開くと、登録済みデータセットを再検査できます。`Rescan` は既存データを消さずに、画像数、caption数、欠損caption、壊れた画像、未対応ファイル、caption文字コード、画像サイズ、タグ集計、trigger word出現率を更新します。

`Top Caption Tags` はcaption内のカンマ区切りタグを集計したものです。キャラクター名や衣装、構図タグが想定通り多いかを確認します。`Trigger Count` は登録したtrigger wordがcaptionに何回出ているかを示します。0%の場合でも学習自体は可能ですが、trigger wordで呼び出すLoRAを作るならcaptionまたはsample prompt設計を見直してください。

未対応ファイルにはメタデータやcacheファイルも含まれます。画像とcaptionが揃っており、broken imageが0であれば、未対応ファイルが存在してもただちに問題とは限りません。

## Job派生

Job詳細画面では既存Jobから `Clone Job` と `Quick Variant` を作成できます。

- `Clone Job` はデータセット、base model、プリセット、params、sample prompt templateを引き継いだdraftを作ります。成果物、metrics、sample画像はコピーしません。
- `Quick Variant` は元Jobを親として、LR、network_dim/network_alpha、epoch数など一部パラメータだけを変えたdraftを作ります。

派生Jobには `parent_job_id` が保存され、Dashboard、Job詳細、Compare画面から親Jobを確認できます。まずPilotを走らせ、Compareで差を見て、CloneまたはQuick Variantで小さく条件を変える流れを推奨します。

## Sample Prompt Template

`Prompt Templates` 画面では組み込みのsample prompt templateを確認できます。New Jobでtemplateを選ぶと、`Prepare Files` 時にtemplate内の `{trigger_word}` がDatasetのtrigger wordへ置換され、`sample_prompts.txt` と `sample_prompts` テーブルへ保存されます。

初期テンプレート `SDXL Face Basic 3 Prompts` は、顔アップ、全身、表情とポーズの3枚をepochごとに比較するためのものです。デフォルト生成promptではなく固定テンプレートを使うと、Job間比較で同じprompt同士を見比べやすくなります。

## Metrics / Loss の見方

Job詳細画面の `Metrics / Loss` では、`Reimport Results` または学習完了時の自動取り込みで集計されたlossとstep整合性を確認できます。

- `Expected Steps` は `dataset_config.toml` のsubsetごとの画像数とrepeat、プリセットのepochまたは `max_train_steps` から計算した概算です。
- `Actual Step` はTensorBoard metricsまたは `train.log` から読み取った最大stepです。
- `Step Check` はexpectedとactualが概ね一致すれば `OK`、取得不能や差が大きい場合は `WARNING`、completedなのに極端にstepが少ない場合は `ERROR` になります。
- `Health` はLoRA品質や絵の良し悪しではなく、学習ログ上のloss推移の健全性ラベルです。
- TensorBoard eventがある場合はそれを優先し、無い場合は `train.log` の `avr_loss` をfallbackとして取り込みます。

Integration Smokeのようにstep数が極端に少ないジョブでは、loss healthは `UNKNOWN` や `DANGER` になりえます。これは品質評価ではなく、短すぎるログから見た注意表示です。

サンプル画像はJob詳細の `Samples By Prompt` にprompt別、epoch/step順で表示されます。各画像には人間確認用の `rating` と `memo` を保存できます。これはAI評価ではなく、目視メモ用途です。

`Health` が `WARNING` の場合は、`spike_count`、spike判定閾値、`loss_volatility`、`late_stage_slope`、`min_loss_step`、`final_loss`、`health_message` を確認します。`WARNING` は即不採用ではなく、lossログ上の注意表示です。サンプル画像の見た目が良い場合は採用候補になり得ます。最終的な `selected LoRA` は、loss健全性、epoch別サンプル、rating/memoを見て人間が選びます。

## 学習確認の推奨順

1. `Integration Smoke - SDXL`
   結合確認専用です。実モデルでsd-scripts起動、LoRA出力、サンプル生成、DB取り込み、画面表示を短時間で確認するため、`max_train_steps=2` の上限を使います。LoRA品質評価には使いません。

2. `SDXL 2D Face - Pilot 3 Epoch`
   実用前の短時間確認用です。50枚前後のデータセットで、おおよそ `50 images * repeats 2 * epochs 3 / batch 2 = 150 steps` を走らせ、loss推移、epochごとの差、sample比較、出力LoRA選択の導線を確認します。

3. `SDXL 2D Face - Pilot Generalize 3 Epoch`
   Standard寄りのPilotより弱めの短時間確認用です。同じデータセットとbase modelで `Pilot 3 Epoch` と比較し、固定化を避けた設定のsample差、loss推移、採用候補を確認します。

4. `SDXL 2D Face - AdamW8bit Standard`
   Pilotで評価導線とおおまかな挙動を確認してから、本学習候補として使います。

`expected_total_steps` は設定とデータセットから計算した概算、`actual_max_step` はTensorBoardまたは `train.log` から読めた実stepです。差が大きい場合は、dataset_config、batch size、repeat、epoch、`max_train_steps` の指定を確認してください。

## Job比較

DashboardのRecent Jobsで2件にチェックを入れて `Compare Selected` を押すか、`/compare?job_a=4&job_b=5` のようにURLを指定すると比較画面を開けます。

比較画面では、基本情報、親Job、プリセット、base model、採用LoRA、主要パラメータ差分、metrics差分、lossグラフ、prompt別sample画像を横並びで確認できます。Standard PilotとGeneralize Pilotは、同じデータセット、同じbase modelで作成し、両方の `expected_total_steps` と `actual_max_step` が概ね一致していることを確認してからsampleを比較します。

比較結果は `Export Markdown` で `runs/comparisons/compare_job_000004_job_000005.md` のようなMarkdownへ出力できます。MarkdownにはJob ID、プリセット名、パラメータ差分、metrics差分、selected LoRA、人間メモ、health注意、sampleファイル名を記録します。

## 実用前チェックの推奨ワークフロー

1. Datasetsでデータセットを登録する。
2. Dataset Inspectorで画像、caption、trigger word、タグ傾向を確認する。
3. `Integration Smoke - SDXL` で結合確認をする。
4. `Pilot 3 Epoch` と必要なら `Pilot Generalize 3 Epoch` を実行する。
5. Compare画面でloss health、step整合性、epoch別sample、rating/memo、selected LoRAを比較する。
6. 良さそうなJobをCloneし、Quick VariantでLRやdimを小さく変えて再確認する。
7. Standard本学習または長めの実用設定へ進む。

## 既知の制限

- 同時実行できる学習ジョブは1件のみ。
- loss解析はTensorBoardまたはtrain.logから取得できる範囲の簡易集計です。
- epoch比較UIはJob詳細内のprompt別サンプル比較が中心です。
- AIによる画像自動評価やパラメータ自動最適化はMVP対象外。
