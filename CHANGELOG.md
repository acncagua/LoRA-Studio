# Changelog

## phase12.4 / v0.5.3-beta - 2026-06-25

### 主な変更

- 段階的i18n基盤を追加しました。標準表示は日本語、`?lang=en` または左ナビの言語切替で英語表示にできます。
- 初期翻訳対象として、左ナビ、ページ見出し、主要ボタン、状態バッジ、Primary Action、Job作成Wizardの主要表示を扱います。
- Built-in Recipe v2 / Optimizer masterに `labels_json`、`descriptions_json`、`risk_notes_json` を追加し、多言語表示を保存できるようにしました。
- RecipeカードとOptimizer/Recipe Libraryで、built-in masterの表示名・説明・risk noteをlocaleに応じて表示する基盤を追加しました。

### 注意点

- sd-scripts log、生成command、raw args、tracebackは翻訳しません。
- Custom Recipeやユーザー入力名は翻訳せず、入力された名前をそのまま表示します。
- 初期版のため、詳細フォーム内の全ラベル翻訳は段階的に追加します。

## phase12.3.1 / v0.5.3-beta - 2026-06-25

Phase 12.3.1 adds Practical Mini Pilot checks for Optimizer Profiles after Smoke OK.

### 主な変更

- Candidate Standard Comparisonでweight 0 baseline共有の基盤を追加しました。論理条件数、物理生成数、共有baseline数、削減枚数を比較グループに保存し、Matrixでは共有baselineにbadgeを表示します。
- Candidate Standard ComparisonのEmbeddingをValidation Run単位ではなくgroup単位で実行できるようにしました。共有baselineや同一パス画像はユニーク化され、Embedding Jobのcache hit / elapsedを比較グループ側で確認できます。
- Runtime Storage Settingsを追加しました。`runtime_root` を設定すると、新規Job runs、Validation exports、Embedding cache、logsをOneDrive外へ配置できます。既存DBに保存済みのパスは移行せずそのまま使います。
- Storage画面にruntime root、runs root、exports root、embedding cache root、logs root、OneDrive warning、書き込みテスト導線を追加しました。
- Candidate Standard ComparisonのPerformance Summaryとして、generation / embedding / machine review / matrixの秒数と秒/枚を比較グループ一覧に表示します。
- `optimizer_mini_pilot_runs` / `optimizer_mini_pilot_items` を追加し、Optimizer Profileごとの短時間Mini Pilot結果を保存できるようにしました。
- `optimizer_profiles_v2` に `mini_pilot_status` / `last_mini_pilot_at` / `last_mini_pilot_result_id` を追加しました。
- `/optimizer-mini-pilots` を追加し、対象Profile選択、実行前見積もり、順次実行、Run履歴、Item結果、Report出力を確認できるようにしました。
- Optimizer詳細画面のMini Pilot導線を、スキップ記録ではなくMini Pilot Run作成へ接続しました。
- Mini Pilot用Jobでは、短時間step、sample出力、safetensors出力を使い、loss summary、artifact check、sample画像数を保存します。
- Mini Pilot Reportを `reports/optimizer_mini_pilots/*.md` と `logs/*.json` に出力できるようにしました。

### 注意点

- Mini Pilot OKは品質保証ではありません。Smoke OKより一段進んだ軽量実用確認です。
- Machine Review / Mini Validationは初期版では結果枠を用意し、重い自動評価は明示実行の対象として扱います。
- DAdaptLionなどExperimental ProfileはMini Pilot OKでも実験扱いを維持します。
- Runtime root設定は新規生成物から適用します。既存Job / 既存Validation Run / 既存Matrixの移行は今回行いません。
- weight 0 baseline共有はCandidate Standard Comparison専用です。Weight Calibrationや単独Validation Runには適用しません。

## phase12.3 / v0.5.2-beta - 2026-06-24

Phase 12.3 adds an Optimizer Profile Validation / Smoke Test foundation so Recipe v2 entries can show whether their optimizer profile has been prepared or smoke-tested against sd-scripts.

### 主な変更

- `optimizer_profile_test_results` を追加し、Prepare Test / 2-step Smoke Test / Mini Pilotの結果、command/log path、return code、環境情報を保存できるようにしました。
- `optimizer_definitions_v2` / `optimizer_profiles_v2` にsd-scripts optimizer type、必須/推奨params、command params、smoke params、依存メモ、検証状態を保存できるようにしました。
- AdamW8bit Balanced、PagedAdamW8bit Balanced、Prodigy Soft、Adafactor Auto / Fixed、Lion Soft / Balanced Experimental、DAdaptAdam Auto、DAdaptLion Autoの初期Profileを登録しました。
- SDXL Character Face向けに、各Optimizer Profileへ対応するBuilt-in Recipeを追加しました。
- Command BuilderでAdafactor Autoの `learning_rate=null` を許容し、nullの場合は `--learning_rate` / `--unet_lr` を出さないようにしました。
- `optimizer_args` を `--optimizer_args decouple=True weight_decay=0.01 ...` の配列形式で出力するようにしました。
- Optimizer詳細画面に `Prepare Test Job`、`Run 2-step Smoke Test`、`Run Mini Pilot`、最終結果表示の導線を追加しました。
- `/optimizer-master-checks` を追加し、Built-in Optimizer ProfileのPrepare / 2-step Smoke / LoRA artifact / Image Smoke結果をMatrixとして保存・表示できるようにしました。
- `optimizer_master_check_runs` / `optimizer_master_check_items` を追加し、profileごとのJob、LoRA artifact hash、safetensors確認、image smoke結果、failure categoryを追跡できるようにしました。
- Master Check Reportを `reports/optimizer_master_checks/*.md` と `logs/*.json` に出力できるようにしました。
- Failure classificationとして、LoRA-Studioロジック、マスタパラメータ、依存不足、sd-scripts未対応、環境問題を初期推定するようにしました。
- `optional_optimizer_dependencies` を追加し、`dadaptation` / `prodigyopt` / `lion-pytorch` をsd-scripts venv上で確認・installできるようにしました。
- Environment画面とOptimizer Master Check画面にOptimizer optional dependency状態とInstall導線を追加しました。
- LoRA-Studio管理sd-scripts環境のセットアップでは、`install_optional_optimizer_deps=true` を初期値としてoptional optimizer dependenciesを標準導入します。既存外部sd-scripts環境は明示操作時だけinstallします。
- DAdapt系の不足依存 `dadaptation` をsd-scripts venvへ導入し、DAdaptAdam Auto / DAdaptLion Autoの再Smokeで `image_smoke_ok` を確認しました。最終サマリとしてBuilt-in Optimizer Profile 9件すべてが `prepare_ok` / `smoke_ok` / `image_smoke_ok` になりました。
- Optimizer Master Check最終サマリを `docs/optimizer_master_check_final_summary_phase12.3.md` に保存しました。
- Smoke Test用Jobは `max_train_steps=2`、低dim、最小サンプル設定で作成し、品質評価ではなく起動確認として扱います。
- Recipeカード / Recipe Library / Job作成Wizardに `Untested` / `Prepare OK` / `Smoke OK` / `Failed` badgeを表示するようにしました。
- Job作成Wizardで未検証・失敗済みOptimizer Profileを選んだ場合、Compatibility WARNINGを表示します。

### 注意点

- Smoke OKは実用品質を保証しません。sd-scriptsで起動できることを確認するための最小検証です。
- AdamW8bit以外のProdigy / Adafactor / Lion / DAdapt系はAdvanced / Experimentalとして扱い、Smoke結果とログを確認してから使う想定です。
- Prodigy / DAdapt系の `learning_rate=1.0` は通常LRではなくAuto-LR倍率です。Adafactor Autoはrelative_step運用、Adafactor Fixedは固定LR fallbackです。
- Mini PilotはPhase 12.3では基盤のみで、明示的な長時間Runは自動実行しません。

## phase12.2.1 / v0.5.1-beta - 2026-06-24

Phase 12.2.1 expands the built-in Recipe v2 master data and makes the Job creation Wizard put Recipe selection at the center of the screen, with different candidate views for purpose-first and optimizer-first workflows.

### 主な変更

- SDXL Character Face / Style / Costume と SD1.5向けのBuilt-in Recipeを追加しました。
- 用途から選ぶ入口では、目的に合うRecipe候補をOptimizerカテゴリ別に表示するようにしました。
- Optimizerから選ぶ入口では、選択Optimizerで使えるRecipe候補をPurpose別に表示するようにしました。
- mode選択後は作成入口カードを折りたたみ、Recipe候補と選択中Recipeパネルを主役にしました。
- フィルタoptionに候補件数を表示し、0件候補は選べないようにしました。
- Recipeカードにtarget steps、key params、expected behavior、risk、選択ボタンを表示しました。
- Recipe表示名に `short_label` / `full_label` / `direct_select_label` を追加し、Model Family選択後はカードタイトルからSDXL/SD1.5の繰り返しを省くようにしました。
- 直接Recipe selectは折りたたみ内へ移動し、検索・確認用としてModel Familyを含むラベルを表示するようにしました。
- Optimizer入口にLR意味、default LR、target steps、risk、compatibility notesを確認できる説明パネルを追加しました。
- 条件に合うRecipeがない場合のEmpty stateを追加しました。

## phase12.2 / v0.5.0-beta - 2026-06-23

LoRA Studio Phase 12.2 improves the Recipe v2 Job creation experience with a wizard-style flow, Recipe cards, Parameter Editor v2, structured override diffs, and richer Recipe / Optimizer browsing.

### 主な変更

- `/jobs/new` を、作成方法、Project / Dataset、Recipe、Parameter Editor、Step Estimate / Compatibility、作成サマリの順で確認するウィザード型画面に整理しました。
- Job作成モードとして、用途から選ぶ、Optimizerから選ぶ、既存Jobから派生、完全カスタムのカード導線を追加しました。
- Recipeカードを追加し、用途、Optimizer、Network Type、target steps、risk、Recipe詳細 / Optimizer詳細へのリンクを見ながら選択できるようにしました。
- Parameter Editor v2として、Basic Params、Advanced Params、Raw Args、Resolved Params、User Override Diffを分離して表示しました。
- Job作成時のuser overridesを `{key: {from, to, reason}}` 形式で保存し、Job詳細で差分理由を確認できるようにしました。
- Compatibility CheckをERROR / WARNING / NOTEに分け、ERRORがある場合は画面上でも下書きJob作成を止めるようにしました。
- Recipe Library `/training-recipes` をカード表示と追加フィルタに更新し、Recipe詳細ページを追加しました。
- Optimizer Master `/optimizers` にLR semantics説明を追加し、Optimizer詳細ページでprofile、allowed scheduler、関連Recipeを確認できるようにしました。
- Job詳細から現在paramsをCustom Recipeとして保存できる最小導線を追加しました。
- legacy presetは折りたたみ表示として残し、既存Job互換を維持しています。

### 確認

- Phase 12.2 acceptance用に、用途、Optimizer、派生、Customの下書きJob作成を実DBで確認しました。
- 作成した下書きJobのうち1件でPrepare Filesを実行し、`command_argv` / dataset config / sample prompts生成経路が壊れていないことを確認しました。

## phase12.1 / v0.4.9-beta - 2026-06-23

LoRA Studio Phase 12.1 adds the Training Recipe / Optimizer Master v2 foundation while keeping legacy presets and existing Jobs compatible.

### 主な変更

- `optimizer_definitions_v2` / `optimizer_profiles_v2` / `network_type_definitions` / `training_purposes` / `training_recipes_v2` / `training_recipe_versions` を追加しました。
- AdamW8bit、PagedAdamW8bit、Adafactor、Lion、Lion8bit、PagedLion8bit、DAdaptAdam、DAdaptLion、Prodigy、CustomをOptimizer Master v2へ登録しました。
- SDXL character face向けのSmoke、Pilot、Standard 6 Epoch、Standard 10 Epoch、Generalize、Lion / Adafactor / DAdaptAdam / Prodigy experimental recipesをTraining Recipe v2として登録しました。
- Job作成画面に「用途から選ぶ」「Optimizerから選ぶ」「既存Jobから派生」「完全カスタム」の入口とTraining Recipe v2選択を追加しました。
- Recipe Library `/training-recipes` と Optimizer Master `/optimizers` を追加しました。
- Job作成時にRecipe v2、Optimizer、Network Type、Training Purpose、resolved params、user overrides、Step Estimateをsnapshot保存できるようにしました。
- Compatibility Checkの基礎として、TE学習とcache_text_encoder_outputs / network_train_unet_onlyの矛盾、DAdapt / Prodigy scheduler、Adafactor relative_step、未対応Network Typeを検出します。
- 既存presetはlegacyとして残し、既存Jobの表示と作成経路を維持しています。

### 注意点

- Phase 12.1ではLoCon / LoHa / LoKr / LyCORISの本格実行には対応していません。Network Type metadata上はplanned / unsupportedとして表示します。
- Lion / Adafactor / DAdapt / Prodigyの実学習評価は今回の対象外です。Recipeは比較用・Advanced用の土台です。
- Raw ArgsはAdvanced扱いで、互換性チェックをすり抜ける可能性があります。

## phase11.9.2 / v0.4.7-beta - 2026-06-23

LoRA Studio phase11.9.2 / v0.4.7-beta is the pre-Phase-12.1 consolidation release. It aligns release metadata and collects the stabilization work added after phase11.9.1.

### 主な変更

- Candidate Standard Comparisonで、loss候補epochごとのStandard Validation v1作成、comparison group一括実行、Epoch横断Matrix確認までの導線を整理しました。
- Retry Signal SummaryをProject / Job / Review Session / LoRA Profileに表示し、Draft Job作成や自動Runを行わない読み取り専用の現状診断として整理しました。
- Step Estimator / Target Step AssistantでOptimizer / Recipe由来のtarget stepsを使い、通常運用ではrepeats / epochs / batchで学習量を調整する流れを整理しました。
- Weight Calibration導線を、採用済みLoRAのweight検証Pipelineとして整理し、開始前確認、Matrix表示、Profile反映への導線を補強しました。
- Performance SummaryでReview Preparation / Weight Calibrationのstage timing、gen_img.py起動回数、Embedding / Machine Review / import内訳、OneDrive配下警告を確認できるようにしました。
- Validation generation follow-upを統合し、画像生成後に不足Embedding / Machine Reviewを自動で続ける導線へ整理しました。
- 実行中処理モニターに、全体経過、段階経過、推定合計、推定残り、推定完了時刻、処理速度を表示するようにしました。
- 学習プリセットのdefault `train_batch_size` を1へ寄せ、SDXL学習時のメモリ安全性を優先しました。
- SDXL UNet-only学習で `--text_encoder_lr 0 0` を渡さないようにし、Windows native crash時の診断ログを追加しました。
- README / README_ja / app_version.pyのrelease / phase表記を `v0.4.7-beta` / `phase11.9.2` に揃えました。

### 既知の注意点

- 実行中処理モニターの推定時間は、sd-scriptsのtqdmログまたは現在までの平均速度から算出する目安です。モデル保存、sample生成、OS同期、GPU負荷変動により実時間とはずれる場合があります。
- OneDriveなど同期フォルダ配下の `runs` / `exports` / model / embedding cache は、ファイルロックや同期負荷で生成・import・Embeddingが遅くなる場合があります。
- Standard Candidate ComparisonとWeight Calibrationは画像数が多くなるため、学習中や他GPU処理中は同時実行せず、必要に応じてplanned状態で待機させる運用を推奨します。
- Retry Signal Summaryは現状診断であり、自動リトライJob作成や自動Runはまだ行いません。

## phase11.9.1 / v0.4.6-beta - 2026-06-23

LoRA Studio phase11.9.1 / v0.4.6-beta consolidates Candidate Standard Comparison, Retry Signal Summary, performance profiling, and validation image generation follow-up review automation into the current beta release.

### 主な変更

- Candidate Standard Comparisonでloss候補epochごとにStandard Validation v1を作成し、comparison groupとして一括実行・横断Matrix確認できるようにしました。
- Validation画像生成の通常操作は、生成完了後に不足Embedding / Machine Reviewまで自動で再計算する導線へ統合しました。
- 横断Matrixのweight追加生成も、追加生成後に不足レビュー再計算まで自動で続くようにしました。
- Job詳細のValidation Run一覧で、画像生成とレビュー計算の一括操作を分かりやすく整理しました。
- Retry Signal Summary、Performance Summary、Post-training Review Automationの表示・文言・導線を調整しました。

## Phase 11.7.1 Candidate Standard Comparison

Phase 11.7.1 adds a Standard Candidate Comparison route for running Standard Validation v1 across loss candidate epochs as one comparison group.

### 主な変更

- Review Preparationに `Quick Candidate Review` / `Standard Candidate Comparison` / `Manual` の方式選択を追加しました。
- loss候補epochの `primary` / `secondary` / `check` を対象に、Standard Validation v1のValidation Runを候補epochごとに作成できるようにしました。
- `candidate_comparison_groups` を追加し、候補epoch、Validation Run群、想定画像数、実行状態、横断Matrixパスをまとめて追跡します。
- Standard Candidate Comparison groupから画像生成、import、Embedding、Machine Review、各Run Matrix、Epoch横断Matrixまで一括実行できます。
- 開始前に候補epoch数、45枚 x epoch数、推定時間、推定容量を表示します。
- Quick Candidate Reviewは採用epochを決める軽量レビュー、Standard Candidate Comparisonは候補epochの標準45枚比較、Weight Calibrationは採用済みLoRAの推奨weight決定としてUI文言を整理しました。
- SQLite接続にbusy timeout / WAL設定を追加し、background worker更新には軽いretry/backoffを入れました。

## Unreleased - Phase 11.9.0

LoRA Studio Phase 11.9.0 adds Retry Signal Summary as a read-only decision aid before any automatic retry workflow.

### 主な変更

- Training / Review / Weight Calibrationの結果から `retry_signal_label`、`confidence`、理由、推奨next actionを算出するようにしました。
- Project詳細、Job詳細、Review Session詳細、LoRA Profile詳細にRetry Signal Summaryを表示します。
- 分類は `ACCEPTABLE` / `UNDERTRAINED_STEP_SHORTAGE` / `UNDERTRAINED_STILL_IMPROVING` / `OVERTRAINED` / `PARAMETER_TOO_WEAK` / `PARAMETER_TOO_STRONG` / `DATASET_OR_CAPTION_ISSUE` / `NO_CLEAR_WINNER` です。
- expected steps vs target、loss trend、best candidate epoch位置、Machine Assist、human rating、Weight Calibration推奨weight、overfit risk、failure tagsを判断材料として扱います。
- Draft Job作成や自動Runは実行せず、人間が次に何を確認するべきかを示すサマリに留めています。

## Unreleased - Phase 11.8.x Performance Optimization

Phase 11.8.x optimizes and profiles the Review Preparation / Weight Calibration pipelines based on real quick_auto and Weight Calibration timing results.

### 主な変更

- Review Preparation / Weight Calibration PipelineにPerformance Summaryを追加し、stage timing、gen_img.py起動回数、画像mtime由来のfirst/last image、OneDrive配下警告を確認できるようにしました。
- Machine Reviewをtarget/reference/dataset embeddingの一括ロードとnumpy matrix multiplicationへ寄せ、load/similarity/DB writeのperformance logを保存するようにしました。
- Machine Review score保存を1件ずつのDB接続から一括トランザクションへ変更し、18件の実データでDB writeが約29秒から約2秒前後に短縮されることを確認しました。
- Embedding Jobのprovider/device/dtype/batch_sizeをログとPerformance Summaryに表示し、ジョブ単位でモデルを初期化していることを確認しやすくしました。
- Review / Validation import stageのfile scan、condition match、duplicate check、image dimension、sha256、DB writeの詳細計測を追加しました。
- `runs` / `exports` / `data/embeddings` / model pathがOneDrive配下に見える場合のWARNINGを強化し、README / README_jaにも同期フォルダ外の利用推奨を追記しました。

## Unreleased - Phase 11.8.1

LoRA Studio Phase 11.8.1 validates the Post-training Review Automation `quick_auto` path with real data.

### 主な変更

- Job #45 / Review Session #25で、Quick Candidate Reviewの実画像生成、DB登録、Embedding、Machine Review、Review Matrix作成までのE2Eを確認しました。
- `quick_auto` / `standard_auto` のcompletedまたはrunning中Review Sessionに対してTraining completed hookが再実行されても、既存Sessionを再起動しないようにしました。
- `quick_auto`完了後、Job詳細 / Project詳細 / Review Session詳細からReview Matrixへ辿れることを確認しました。

## Unreleased - Phase 11.8

LoRA Studio Phase 11.8 adds Post-training Review Automation for candidate epoch review planning after training completion.

### 主な変更

- Training Job完了後にloss候補epochを抽出し、採用前Review Planを自動作成できるようにしました。
- `manual` / `plan_only` / `quick_auto` / `standard_auto` のPost-training Review Automation modeを追加しました。defaultは安全な `plan_only` です。
- Quick Candidate Reviewは候補epoch最大3件、3 prompts、seed 111111、weight 0.6 / 0.8、Hiresなしの最大18枚で作成します。
- `standard_auto` は候補epochごとに標準45条件相当でReview Planを作成し、`max_auto_images` 超過時は自動実行せず確認待ちにします。
- Review Session詳細に「近隣Epochを追加検証」を追加し、中心epochの±1/±2を追加Review Sessionとして作成できるようにしました。
- Machine Assistが僅差の場合は `no_clear_winner` として候補群を表示し、人間評価優先で判断する導線を追加しました。
- Training / Validation / Embedding / Machine Review実行中やmax_auto_images超過時は自動実行せず、plannedで待機またはユーザー確認待ちにします。

## phase11.7.1 / v0.4.4-beta - 2026-06-22

LoRA Studio phase11.7.1 / v0.4.4-beta adds the Weight Calibration Pipeline for adopted LoRA / selected output validation.

### 主な変更

- Step Estimator / Target Step Assistantを追加し、image count、repeats、epochs、batch、gradient accumulationからexpected total stepsを表示できるようにしました。
- Optimizer Definition / Optimizer Profile / Training Recipeマスタからtarget step min/recommended/maxとtarget checkpoint countを解決し、Job作成・編集・詳細・Preflightでstep量の過不足を確認できるようにしました。
- AdamW8bit、PagedAdamW8bit、Adafactor、Lion、DAdaptAdam、DAdaptLion、Prodigyの基本step目安とLR意味を登録しました。
- 目標stepに合わせてrepeatsを自動計算し、Job snapshotにrepeats自動計算フラグとtarget steps sourceを保存できるようにしました。
- 高epoch時のsave/sample interval提案と、`max_train_steps`直接指定時のAdvanced警告を追加しました。
- Validation Runを採用後weight検証の主導線として整理し、`validation_run_kind`、`pipeline_status`、`matrix_path`、採用epoch/source情報を保存できるようにしました。
- Weight Calibration Preflightを追加し、selected output、LoRA/base model、trigger、preset、expected conditions、sd-scripts環境、Embedding provider、Reference/Dataset coverage、GPU競合を確認します。
- Validation Run詳細に `Prepare Weight Calibration`、`Weight検証を開始`、`Stop`、`Retry`、`Reimport`、`Weight Review Matrixを開く` の導線を追加しました。
- Pipelineはbackgroundで、Expected Conditions確認、sd-scripts画像生成、画像DB登録、Embedding計算、Machine Review Assist、Weight Review Matrix生成、suggested weight算出を順番に実行します。
- Weight Review Matrixの出力パスをValidation Runに保存し、Project / Job / LoRA Libraryから最新検証状態を辿れるようにしました。
- Candidate Reviewは採用前epoch比較、Weight Calibrationは採用済みLoRAのweight調整としてREADMEに明記しました。
- Standard Validation 45枚、weight 0 baseline、Hiresなし基準、Extendedは最終見栄え確認、人間評価優先、Profile Applyは手動、という運用方針をREADMEに追記しました。

## v0.4.1 - 2026-06-20

Release Version: v0.4.1
Internal Milestone: Phase 11.6.16.2

LoRA Studio v0.4.1 は、Phase 11.6.16.2 時点の運用ベータリリースです。

### 主な変更

- 検証Matrix / Epoch横断Matrix / Review Matrixに 25% / 50% / 75% / 100% の画像表示倍率切り替えを追加しました。
- Matrix画像クリック時に、元画像100%表示のポップアップを開けるようにしました。
- Matrixの初期表示倍率を25%にし、横断比較時の画面占有を抑えました。
- 複数Validation Runの画像生成、Embedding、機械補助レビューをJob詳細から順番に実行する導線を改善しました。
- 検証Run一覧の簡易ログを折りたたみ表示にし、完了済みログが画面を占有し続けないようにしました。

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
