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

## Trigger Consistency

`trigger_word` はLoRAを呼び出すための固有タグです。caption内のtriggerとsample prompt内のtriggerが揃っていないと、学習時に覚えたタグと評価時に呼び出すタグがずれ、sample比較が無効になりやすくなります。

Dataset詳細の `Trigger Consistency` は以下の基準で表示します。

- `OK`: captionの80%以上にtrigger_wordが出現する。
- `WARNING`: captionの1-79%にtrigger_wordが出現する。
- `ERROR`: caption内のtrigger_wordが0件。
- `UNKNOWN`: trigger_word未設定、またはcaptionを解析できない。

例えば `trigger_word=testchar` が `0/50` の場合、そのtriggerで呼び出すLoRAとしては強い不整合です。この場合は、captionに既に多く含まれている固有タグをtriggerにするか、captionの先頭に独自triggerを追加します。

Dataset詳細の `Top Tag Candidates` は、`1girl`、`solo`、`looking at viewer` などの一般タグを除外し、caption内で多く出る固有タグ候補を表示します。候補の `Use this as trigger_word` を押すとDatasetのtrigger_wordを変更し、Rescanします。

独自triggerをcaptionへ追加する場合は `Prepend Trigger To Captions` を使います。必ず `Preview` で変更件数、skip件数、変更前後サンプル、backup pathを確認してから `Confirm` します。Confirm時は `backups/datasets/dataset_000004/captions_YYYYMMDD_HHMMSS/` のような場所へ元captionをコピーし、UTF-8でcaptionを書き戻します。

既存Jobは作成済みのparamsやsample promptsを自動変更しません。新規Jobでは作成時のtrigger consistency snapshotを保存します。snapshotが無い既存Jobでは、現在のDataset分析とsample prompt内triggerの使用状況を参考表示します。

`Captions Missing Trigger` では、現在のtrigger_wordを含まないcaptionを一覧できます。画像ファイル名、captionファイル名、caption previewを確認して、prepend対象が妥当か見ます。

`Caption Edit History` の `Restore Preview` では、backup_pathからcaptionを復元する前に、現在値と復元後のサンプルを確認できます。`Confirm Restore` を実行すると、backup時点のcaptionへ戻し、Rescanと新しいDataset version作成を行います。

## Dataset Version

DatasetをRescanした時やcaption編集後には `dataset_versions` に状態スナップショットを残します。versionにはtrigger_word、画像数、caption数、trigger出現数、consistency label、画像manifest hash、caption manifest hash、統計JSONを保存します。

Job作成時には、その時点の最新 `dataset_version_id` を `training_jobs` に保存します。これにより、caption整備前のJobと整備後のJobを比較するときに、どのDataset状態で学習したかを区別できます。古いJobで `dataset_version_id` が無い場合は `snapshot unavailable` と表示します。

Dataset詳細の `Dataset Versions` では、各versionのtrigger出現率とmemoを確認できます。caption編集前後を比較するときは、編集前versionと編集後versionのtrigger consistencyがどう変わったかを見てください。

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

4. `SDXL 2D Face - Standard 6 Epoch`
   Dataset整備後の本番寄り短時間テストです。50枚前後のDatasetでは、おおよそ `50 images * repeats 2 * epochs 6 / batch 2 = 300 steps` を走らせ、epoch 4以降の過学習傾向、sample変化、採用候補epochを確認します。保存時メタデータ計算のメモリ負荷を避けるため `no_metadata` を使います。

5. `SDXL 2D Face - AdamW8bit Standard`
   PilotとStandard 6 Epochで評価導線とおおまかな挙動を確認してから、本学習候補として使います。

`expected_total_steps` は設定とデータセットから計算した概算、`actual_max_step` はTensorBoardまたは `train.log` から読めた実stepです。差が大きい場合は、dataset_config、batch size、repeat、epoch、`max_train_steps` の指定を確認してください。

lossはraw値だけでなく、10点moving averageとepoch summaryを一緒に見ます。raw lossはsample生成や保存前後でspikeしやすいため、`raw=WARNING` でも `smoothed=OK`、`epoch=OK` なら、lossログとしては即不採用ではありません。`health_label` はLoRA品質評価ではなく学習ログの健全性評価です。画像評価とloss健全性は別軸なので、WARNINGでもsample画像が良ければ採用候補になり得ます。

Job詳細の `Epoch Summary` では、epochごとのavg/min/max/final loss、moving average final、spike count、sample count、出力LoRAを確認できます。Compare画面ではJob同士のepoch別avg lossとmoving average finalも比較できます。Dataset versionが同じJob同士を比較すると、caption条件差を避けやすくなります。

## Job比較

DashboardのRecent Jobsで2件にチェックを入れて `Compare Selected` を押すか、`/compare?job_a=4&job_b=5` のようにURLを指定すると比較画面を開けます。

比較画面では、基本情報、親Job、Dataset version、trigger at creation、プリセット、base model、採用LoRA、主要パラメータ差分、metrics差分、lossグラフ、prompt別sample画像を横並びで確認できます。Standard PilotとGeneralize Pilotは、同じデータセット、同じbase modelで作成し、両方の `expected_total_steps` と `actual_max_step` が概ね一致していることを確認してからsampleを比較します。

比較対象Jobの `dataset_version_id` が異なる、または古いJobでsnapshotが無い場合は警告を表示します。caption整備前の旧Jobと整備後の新Jobは、純粋な品質比較ではなく、Dataset条件差を含む参考比較として扱ってください。`trigger_word_at_creation` が異なる場合も同様に注意が必要です。

比較結果は `Export Markdown` で `runs/comparisons/compare_job_000004_job_000005.md` のようなMarkdownへ出力できます。MarkdownにはJob ID、プリセット名、パラメータ差分、metrics差分、selected LoRA、人間メモ、health注意、sampleファイル名を記録します。

## 実用前チェックの推奨ワークフロー

1. Datasetsでデータセットを登録する。
2. Rescanする。
3. Dataset Inspectorで画像、caption、trigger word、タグ傾向、Trigger Consistencyを確認する。
4. `Captions Missing Trigger` でtrigger欠落captionを確認する。
5. triggerが0件または不足しているなら、既存タグをtriggerにするか、Preview/Confirm付きでcaptionへtriggerを追加する。
6. backup_pathとDataset versionが作られたことを確認する。
7. 必要ならRestore Preview/Confirmでテスト復元する。
8. 再度Rescanし、Trigger ConsistencyとDataset versionを確認する。
9. `Integration Smoke - SDXL` で結合確認をする。
10. `Pilot 3 Epoch` と必要なら `Pilot Generalize 3 Epoch` を実行する。
11. `Standard 6 Epoch` で本番寄り短時間テストを行う。
12. Compare画面でDataset version差分、raw/smoothed/epoch loss、step整合性、epoch別sample、rating/memo、selected LoRAを比較する。
13. 良さそうなJobをCloneし、Quick VariantでLRやdimを小さく変えて再確認する。
14. Standard本学習または長めの実用設定へ進む。

## 既知の制限

- 同時実行できる学習ジョブは1件のみ。
- loss解析はTensorBoardまたはtrain.logから取得できる範囲の簡易集計です。
- epoch比較UIはJob詳細内のprompt別サンプル比較が中心です。
- AIによる画像自動評価やパラメータ自動最適化はMVP対象外。
