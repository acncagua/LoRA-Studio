# Project / Dataset

## Project

Projectは1つのLoRA制作単位です。採用Job、採用LoRA、Review Session、Validation Run、LoRA Profileをまとめます。

Project詳細では、現在の状態に応じたNext Action、最新Job、Review Plan、Weight Calibration、Retry Signal Summaryを確認できます。

## Dataset

Datasetは画像フォルダとcaption群を登録します。再スキャンにより画像数、caption有無、trigger consistency、dataset healthを確認します。

Datasetを修正したらDataset Versionを作り、どの学習JobがどのDataset状態から作られたかを残します。

## Trigger Consistency

trigger wordがcaption内に入っているか、sample promptとcaptionが噛み合っているかを確認します。triggerがcaptionにない、sample promptだけにある、またはcaptionが欠落している場合は、学習前に直すことを推奨します。

## Dataset Inspector

Dataset Inspectorは、画像とcaptionの確認、欠落検出、trigger確認、Dataset Version作成前の最終チェックに使います。

## Reference Set

Reference SetはMachine Review Assistや類似度確認に使う基準画像です。キャラクターLoRAでは `face_front`、`upper_body`、`full_body`、`expression` などの役割を持つ参照を揃えると判断しやすくなります。
