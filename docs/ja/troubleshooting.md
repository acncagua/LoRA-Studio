# Troubleshooting

## 画面が500になる

サーバログと該当画面の直近変更を確認します。DB migration後やテンプレート変更後は、サーバ再起動で反映が必要な場合があります。

## 学習が失敗する

よくある原因:

- CUDA OOM
- sd-scripts依存不足
- optimizer package不足
- `optimizer_args` 不整合
- Dataset path / caption不備
- base model path不備
- OneDrive配下I/O遅延

OOMが疑わしい場合は `train_batch_size=1`、resolution、network dim、gradient checkpointing、cache設定を確認してください。

## SQLite lock

長時間import、monitor、background workerが重なるとlockが出る場合があります。busy timeout、短いtransaction、retry/backoffを使う方針です。発生時は実行中processを確認し、重い処理が終わってから再試行してください。

## Review / Validationの表示が更新されない

画像生成、import、embedding、Machine Review、matrix作成は別stageです。画面上は完了に見えても、後続stageが残っている場合があります。Active Operation Monitor、Performance Summary、log pathを確認してください。

## Optional optimizer dependency

DAdaptAdam / DAdaptLionには `dadaptation`、Prodigyには `prodigyopt`、Lionには `lion-pytorch` が必要です。導入先はLoRA-Studio本体venvではなくsd-scripts venvです。

## 旧README_jaの参照

分割前の長い説明は [_archive_readme_ja_before_split.md](_archive_readme_ja_before_split.md) に残しています。移行中に説明が見つからない場合はまずarchiveを検索してください。
