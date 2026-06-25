# Training Job

Training Jobはsd-scriptsで実行する1回の学習設定です。Draft、prepared、running、completed、failed、stopped、archived、deletedなどの状態を持ちます。

## Job作成

Phase 12以降はRecipe Wizardを主導線にしています。

- 用途から選ぶ: 顔キャラ、style、costumeなど目的からRecipeを選びます。
- Optimizerから選ぶ: AdamW8bit、PagedAdamW8bit、Prodigy、Adafactor、Lion、DAdapt系などを先に選びます。
- 既存Jobから派生: 過去Jobのsnapshotを元に差分を作ります。
- 完全カスタム: Basic / Advanced / Raw Argsを手動で調整します。

## Prepare Files

Prepare Filesでは、sd-scriptsへ渡すファイルを作ります。

- `command_argv.json`
- `dataset_config.toml`
- `sample_prompts.txt`

設定保存後に内容を変えた場合は、実行前に再Prepareしてください。

## 実行中の確認

Job詳細では、PID、開始時刻、経過時間、step進捗、ログ末尾を確認できます。推定時間は過去のsec/stepやvalidation sec/imageを元に補助表示しますが、GPU負荷やI/Oで変動します。

## 結果確認

completed後は、出力LoRA、sample画像、loss、step consistency、候補epochを確認します。採用前にReview Sessionで画像比較を行い、採用LoRAを明示的に選びます。

## 失敗時

CUDA OOM、依存不足、sd-scripts引数不整合、SQLite lock、OneDrive I/O遅延などが原因になります。ログを確認し、必要ならbatch、repeats、optimizer、runtime rootを調整してください。
