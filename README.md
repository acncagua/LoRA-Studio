# LoRA-Studio

Stable Diffusion系LoRA、特にSDXL/SD1.5の2Dキャラクター顔LoRA学習をローカルで管理するための支援ツールです。MVPでは自動最適化ではなく、学習実験の作成、実行準備、ログ、loss健全性、サンプル画像比較、採用epoch判断を一元管理することを目的にします。

現在の運用ベータ記録は `CHANGELOG.md` の `v0.1-beta` を参照してください。

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
2. Datasets画面で画像フォルダを登録する。Windowsでは `参照` ボタンからフォルダ選択ダイアログを開けます。
3. Dataset詳細画面で画像、caption、trigger word、タグ傾向を検査する。
4. Presets画面で初期プリセットを確認する。
5. New Job画面でデータセット、プリセット、使用モデル、必要ならSample Prompt Templateを指定してdraftを作る。
6. Job詳細画面で `Prepare Files` を押し、設定ファイルと実行コマンドを生成する。
7. `Run` を押して学習を開始する。必要なら `Stop` で停止する。
8. 完了後、Job詳細画面でログ、出力LoRA、サンプル画像を確認する。手動で再取り込みしたい場合は `Reimport Results` を押す。

`Prepare Files` では表示用の `command.txt` に加えて、実行用の `command_argv.json` を生成します。実行時はshell文字列ではなくargv配列を `subprocess.Popen` に渡すため、Windowsのスペース入りパスでも壊れにくくしています。`dataset_config.toml` の `batch_size` はプリセットの `train_batch_size` から生成し、コマンドライン側では `--train_batch_size` を重複指定しません。学習時のbatch sizeは `dataset_config.toml` を正とします。

## Dataset Inspector

Datasets画面のIDリンクからDataset詳細を開くと、登録済みデータセットを再検査できます。`Rescan` は既存データを消さずに、画像数、caption数、欠損caption、壊れた画像、未対応ファイル、caption文字コード、画像サイズ、タグ集計、trigger word出現率を更新します。

Datasets画面の登録フォームでは、`参照` ボタンからWindowsのフォルダ選択ダイアログを開き、選んだフォルダの絶対パスを入力欄へセットできます。

使用する学習モデルはNew Job画面の `使用モデル` で指定します。プロジェクト直下の `models` フォルダに置いた `.safetensors` / `.ckpt` は候補として表示されます。別の場所のモデルを使う場合は、`参照` ボタンからWindowsのファイル選択ダイアログで選択できます。

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

## 視覚評価ワークフロー

Job詳細の `Samples By Prompt` では、各sample画像に以下の人間評価を保存できます。

- `Face`: 顔・髪型・表情などキャラクター性の入り具合。
- `Costume`: 衣装や装飾の再現。
- `Style`: 絵柄や塗りの安定。
- `Stability`: 崩れ、破綻、固定化の少なさ。
- `Overall`: 採用判断用の総合評価。既存の `rating` は `rating_overall` と互換扱いです。
- `Memo`: 目視メモ。良い点、崩れ、採用理由などを書きます。

`Epoch Visual Summary` は、sample ratingをepoch単位で集計し、`training_epoch_summaries` のloss情報と並べて表示します。`avg_loss`、10点moving average、sample数、各rating平均、memo数、対応するLoRA出力、selected状態を同じ表で確認できます。lossが `WARNING` でも、epoch別sampleとratingが良ければ採用候補になり得ます。

採用LoRAは、Outputs一覧の `Select` でも選べますが、`Set selected by epoch` からepoch単位でも選択できます。選択すると `training_outputs.selected`、`training_jobs.adopted_epoch`、`training_jobs.adopted_model_path` が更新されます。

推奨運用は以下です。

1. Job完了後に `Reimport Results` を実行する。
2. `Samples By Prompt` でepoch 3/4/5付近を目視する。
3. sample ratingとmemoを入力する。
4. `Epoch Visual Summary` で候補epochを確認する。
5. `Set selected by epoch` またはOutputs一覧でselected LoRAを決定する。
6. `Export Contact Sheet` で静的HTMLレポートを出力する。
7. `Export Selected LoRA` で採用LoRAを `exports/selected_loras/` に保存する。

## Contact Sheetと採用LoRA export

Job詳細の `Export Contact Sheet` は、`runs/job_xxxxxx/reports/contact_sheet_job_xxxxxx.html` に静的HTMLを出力します。Job基本情報、preset、Dataset version、trigger、selected LoRA、health、epoch別loss summary、prompt別・epoch別sample画像、rating/memo、画像ファイル名を含みます。画像は相対パスで参照するため、ローカルでHTMLを開いて確認できます。

Compare画面の `Export Compare Contact Sheet` は、`runs/comparisons/contact_sheet_compare_job_000010_job_000012.html` のようなHTMLを出力します。比較対象Jobの基本情報、パラメータ差分、Dataset version警告、epoch別loss比較、同prompt・同epochのsample画像、rating/memo、selected LoRAを記録します。

Job詳細の `Export Selected LoRA` は、選択済みLoRAを以下のように保存します。

```text
exports/selected_loras/
  job_000012/
    selected_model.safetensors
    selected_lora_info.json
    selected_lora_notes.md
```

`selected_lora_info.json` にはJob ID、Job名、Dataset ID/version、trigger、preset、params、selected epoch、元ファイルパス、export先、file size、sha256、health、export時刻、人間評価メモを保存します。`selected_lora_notes.md` は人間が読むための概要、trigger、Dataset version、選択epoch、loss summary、メモ、注意点です。

## Validation Packと実生成環境での手動検証

Job詳細の `Validation Pack出力` は、採用済みLoRAをreForge / WebUIで検証するためのファイル一式を `exports/validation_packs/job_xxxxxx/` に出力します。Packには以下が含まれます。

- `validation_prompts.md`: LoRA weight別、prompt type別の手動検証prompt。
- `validation_prompts.json`: 将来のWebUI API連携でも使いやすい構造化prompt。
- `validation_checklist.md`: 目視確認用チェックリスト。
- `lora_usage_example.txt`: reForge / WebUIでの短い使い方。
- `validation_result_template.md`: 手動検証結果の記録テンプレート。

まず `Export Selected LoRA` または `Validation Pack出力` を実行し、採用LoRAを `exports/selected_loras/job_xxxxxx/` に出力します。Validation Pack出力時にも採用LoRA exportは更新され、`selected_lora_info.json` には `validation_pack_path` が追記されます。

reForge / WebUIでは、LoRAファイルを `models/Lora` にコピーし、LoRA一覧を更新してからprompt内で `<lora:filename:weight>` を使います。promptに入れるLoRA名は `.safetensors` 拡張子を除いたファイル名です。最初は学習時と同じbase modelで確認してください。違うbase modelで試すのは、同一base modelで崩れや暴発がないことを見てからにします。

Validation Packは `0.4 / 0.6 / 0.8 / 1.0` のweightを出力します。目安は以下です。

- `0.4`: 弱すぎず、特徴が少し出るか。
- `0.6`: 自然に特徴が出るか。
- `0.8`: 顔特徴が安定するか。最初の採用候補になりやすいweightです。
- `1.0`: 崩れ、固定化、背景汚染が強くならないか。

Job詳細の `Validation Results` では、手動生成した画像の評価を記録できます。`prompt_type`、`weight`、顔、衣装、安定性、柔軟性、総合、メモ、任意の `image_path` を保存します。保存した結果から、weight別・prompt_type別の平均スコア、`best_weight_by_overall`、`best_weight_by_stability` を表示します。best weightは自動採用ではなく、人間が画像とメモを見て判断するための補助値です。

## External ValidationとLoRAライブラリ

Job詳細の `External Validation` では、reForge / WebUIなど外部生成環境で作成したValidation画像を、採用LoRAに紐づけて保存できます。保存できる主な項目は、画像パス、validation type、prompt、negative prompt、base model、sampler、steps、CFG、画像サイズ、Hires設定、LoRA weight、seed、顔/衣装/スタイル/安定性/総合rating、推奨weight範囲、メモです。

評価入力は `evaluation_rubrics` の active schemaに沿って、rating 1〜5に加えて、強さ、過学習、採用判断、failure tagsを定型項目として保存します。memoは任意補足です。これにより、自由文の揺れに左右されずRecommendation Engineが評価を使えます。将来ChatGPT画像評価を追加する場合も、このrubric schemaに従ったJSONを保存する想定です。現時点ではChatGPT API連携は行いません。

同じセクションの `weight別評価` では、画像1枚単位ではなくLoRA weight単位の評価を記録できます。`0.4` をlight、`1.0` をstrongのように扱い、`recommended_weight_min / recommended_weight_max` で実用時の推奨範囲を保存します。

採用LoRAを選択すると `selected_lora_profiles` にLoRA Profileが作成され、`LoRAライブラリ` 画面からJob横断で一覧できます。Profile編集画面ではtrigger、base model、推奨weight、light/strong weight、validation memo、library memoを調整できます。Job単位の結果確認から、実際に使うLoRAのライブラリ管理へ移すための画面です。

`Export Contact Sheet` と `selected_lora_notes.md` にはExternal ValidationのProfile、weight評価、Validation画像情報も追記されます。外部生成環境での目視結果を、学習時のlossやsample ratingと一緒に残せます。

## Validation PresetとReference Set

Validation Presetは、reForge / WebUIなど外部生成環境で人間がValidation画像を作る時の標準条件です。LoRA-Studioは画像生成を自動実行せず、prompt、seed、weight、Hires条件を固定したPrompt Packを出力し、生成後の画像をValidation Runへ登録します。

built-in presetは以下の3段階です。

- `Quick Validation v1`: 3 prompts × 1 seed × 3 weights × Hiresなし = 9枚。採用LoRAが使えそうかを短時間で確認します。
- `Standard Validation v1`: 3 prompts × 3 seeds × 5 weights × Hiresなし = 45枚。採用候補LoRAの標準比較です。推奨weightは原則としてこのHiresなし結果を基準にします。
- `Extended Validation v1`: 3 prompts × 2 seeds × 2 weights × Hiresあり/なし = 24枚。採用候補の最終見栄え確認です。

標準比較はHiresなしで行います。Hiresありは最終見栄え確認であり、Hiresなしの素のLoRA比較とは直接混ぜて判断しません。seedは固定で構いません。weightを毎回細かく刻みすぎると生成時間が増えるため、まずはQuick、採用候補ならStandard、最後にExtendedの順で確認します。

Job詳細またはLoRA Profile編集画面の `Create Validation Run` からPresetを選ぶと、`exports/validation_runs/validation_run_xxxxxx/` に以下を出力します。

- `validation_prompts.md`: WebUI / reForgeでそのまま使えるprompt一覧。
- `validation_prompts.json`: 将来のWebUI API連携や自動処理でも使いやすい構造化条件。
- `validation_conditions.json`: condition hash付きの全条件。
- `validation_grid_plan.md`: weight × seedなどのGrid作成方針。
- `validation_checklist.md`: 目視確認用チェックリスト。

Validation Run詳細では、個別画像またはGrid画像を登録できます。個別画像はprompt_key、seed、weight、Hires条件から `condition_hash` を作り、同一条件比較に使います。Grid画像は資料として登録でき、初期版では自動分割しません。条件が違うValidation画像は直接比較せず、画面上の警告を確認してください。

Validation Run詳細の `Coverage Matrix` は、Presetが期待する全条件に対して、画像登録済み、Rubricレビュー済み、未登録、ignoredを一覧します。`weight 0` はLoRAタグを付けないbaseline条件です。baselineは比較基準としてレビューできますが、推奨weight計算の候補には入れません。

登録画像のレビューは、rating 1〜5、強さ、過学習、採用判断、failure tagsを中心に入力します。memoは補足です。`Weight Review Matrix` ではweight、prompt、seed、Hires別に評価分布を確認できます。個別画像は期待条件に紐づくとCoverageに反映され、Grid画像は比較資料として保持されます。

`Suggested Weight` はHiresなしのQuick/Standard結果を優先し、`too_weak`、`too_strong`、`broken`、`reject` を除外して計算します。評価済み条件が少ない場合は既存Profile/Runの推奨weightを維持します。`Profileへ反映` は人間が押した時だけ `selected_lora_profiles` に保存し、Validation Runを完了扱いにします。

`Validation Report出力` は `exports/validation_runs/validation_run_xxxxxx/validation_report.md` に、Coverage、推奨weight、Weight Review Matrix、登録画像、注意事項をMarkdownで保存します。LoRAライブラリ画面では最新Validation Runのstatus、coverage、レビュー不足警告を確認できます。

Reference Setは、人間評価時に見る参考画像の固定セットです。DatasetやLoRA Profileに紐づけて、顔、上半身、全身、表情、スタイルなどの参照画像を登録できます。現時点ではAI評価には使いませんが、将来のWebUI API / ChatGPT API評価でもValidation PresetとReference Setを使う予定です。

## Recommendation Engineと次回実験提案

Job詳細の `提案を再生成` は、loss summary、epoch summary、sample rating、External Validation、推奨weight、Dataset trigger consistencyを見て、次回試すべき実験案をルールベースで作成します。提案は `experiment_recommendations` に保存され、Job詳細とLoRAライブラリのProfile編集画面で確認できます。

Recommendation Engineは自動実行を行いません。`Draft Job作成` を押すと、提案の `suggested_params_json` を使った下書きJobだけを作成します。元Jobのdataset、dataset version、base model、sample prompt template、trigger情報を引き継ぎ、`parent_job_id` とmemoに推薦元を残します。Runは必ず人間がJob詳細で押してください。

推奨weightは次回パラメータ提案の重要な材料です。0.6〜0.8でよく効き、1.0が強い場合は、現状採用またはLower LR/短縮epochを優先します。0.8以上でも弱い場合は、higher dimやstrengthen寄りの提案が出ます。0.4〜0.6で強すぎる場合は、lower LR、lower dim、fewer epochなどを検討します。

health WARNINGは即不採用ではありません。step単位の揺れがあっても、epoch trendがOKで画像評価や外部Validationが良ければ採用候補として扱います。逆にhealth OKでも画像評価が低い場合は、パラメータよりDataset、caption、sample promptの見直しを優先します。

`提案レポート出力` は `runs/job_xxxxxx/reports/recommendations_job_xxxxxx.md` にMarkdownを保存します。source job、selected LoRA、Profile、Validation summary、loss summary、recommendations一覧、suggested params差分、注意事項を含みます。自動リトライ、ChatGPT API評価、AI画像評価は将来拡張です。

Environment画面には将来のWebUI API連携用に `webui_api_enabled=false` と `webui_api_url=http://127.0.0.1:7865` を表示します。現時点ではWebUI APIによる自動txt2imgは実装しておらず、API呼び出しも行いません。

## MaintenanceとDiagnostics

サイドバーの `メンテナンス` では、軽量BackupとDiagnostics出力を実行できます。Backupは `backups/app_backups/backup_YYYYMMDD_HHMMSS/` にDB、app settings、exports内の軽量ファイル、runs配下のreportsを保存します。大型のモデル、画像、動画、zipは初期版では除外します。

Diagnosticsは `exports/diagnostics/diagnostics_YYYYMMDD_HHMMSS.md` に、app version、git commit、DB schema version、Python/Torch/CUDA/GPU、sd-scripts、Dataset/Job/LoRA Library/Validation Run件数、latest errors、known warningsを出力します。API tokenなどの秘密情報は出力しません。

サイドバーの `推奨ワークフロー` には、Dataset登録からValidation Run Review、Recommendation確認、Draft Job作成までの標準運用手順をアプリ内で表示します。

## Known Issues

- Codex内ブラウザはWindows sandbox権限で `CreateProcessAsUserW failed: 5` になる場合があります。
- Codex内ブラウザが使えない場合でも、HTTP画面確認で代替可能です。
- HiresありValidationは標準比較ではなく、最終見栄え確認用です。
- Validation Runでregistered数がexpected未満の場合は、Recommendationの信頼度に注意してください。

## no_metadataとsafetensors取り込み

`no_metadata` は、sd-scriptsが `.safetensors` 保存時にmetadata関連処理で大きなメモリを使う場合の回避策です。Job #11ではsafetensors metadata/hash関連で `MemoryError` が発生しましたが、Standard 6 Epoch presetに `no_metadata: true` を追加したJob #12は完走しました。

メタデータなしでもLoRA本体としては利用できます。ただし、後から学習条件を確認するため、LoRA-Studio側のDB、`job_config.json`、`selected_lora_info.json` を一緒に保存することが重要です。

LoRA-Studio側のsha256計算はストリーミング処理で行い、ファイル全体を一度に読み込みません。metadata読み取りは必須にせず、metadata/hash周辺でエラーがあっても、可能な限り成果物取り込み全体を失敗させない方針です。エラーがある場合はOutputs一覧の `Metadata Error` に表示します。

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
