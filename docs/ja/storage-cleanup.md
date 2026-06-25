# Storage / Cleanup

LoRA学習は `.safetensors`、sample画像、validation画像、embedding cache、logsを大量に生成します。LoRA-Studioは削除やarchiveの導線を提供しますが、cleanupは自動では行いません。

## Runtime Storage Settings

runtime rootを設定すると、新規Jobのruns、Validation exports、logs、Embedding cacheをOneDrive外へ配置できます。既存DBに保存済みのartifact pathは移行せず、そのまま参照します。

## OneDrive注意

OneDrive配下に大容量model、runs、exports、embedding cacheを置くと、同期、ロック、sha256、画像importで遅くなる可能性があります。可能なら高速なローカルSSD上の同期対象外ディレクトリを使ってください。

## Cleanup

Storage画面では、容量の大きいruns、archive候補、削除済みJob、未使用出力などを確認します。削除操作はProjectやJobの意味を理解してから実行してください。

## Archive

採用済みでない大量出力はarchive対象にできます。採用LoRA、Validationに使った出力、LoRA Profileに紐づくファイルは誤削除しないよう注意してください。
