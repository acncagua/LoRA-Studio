# 基本ワークフロー

LoRA-Studioは、1本のLoRAを作る作業をProjectとして扱います。Datasetの状態、学習Job、候補レビュー、Validation、採用LoRA、次回実験提案を同じ文脈で追跡します。

## 推奨順

1. Projectを作成します。
2. Datasetを登録し、captionとtrigger consistencyを確認します。
3. Dataset Versionを作成します。
4. Recipe Wizardまたはlegacy presetでTraining Jobを作ります。
5. Prepare Filesで `command_argv.json`、`dataset_config.toml`、`sample_prompts.txt` を生成します。
6. PreflightとStep Estimateを確認します。
7. 学習を実行します。
8. 学習結果、loss、sample画像、出力LoRAを確認します。
9. Review Sessionで候補epochを比較します。
10. 採用epoch / 採用LoRAを選びます。
11. Weight Calibration / Validation Runで推奨weightを確認します。
12. LoRA Profileへ反映し、ExportやCleanupへ進みます。

## 役割の分離

- Candidate Review: 採用前に候補epochを比較する軽量または標準レビューです。
- Standard Candidate Comparison: loss候補epochごとにStandard Validation v1をまとめて実行する重い比較です。
- Weight Calibration: 採用済みLoRAの推奨weight範囲を決める採用後検証です。
- Retry Signal Summary: 読み取り専用の現状診断です。Draft Job作成や自動Runは行いません。
- Recommendation Engine: 次回実験案を作る機能です。ユーザーが明示操作した場合だけDraft Job作成へ進みます。

## 自動化との付き合い方

Post-training Review Automationは学習完了後の候補Review Plan作成を補助します。`plan_only` は計画だけ作り、`quick_auto` は安全条件を満たすと軽量Reviewを開始します。`standard_auto` は重いため、画像数や時間の安全制御を確認してから使います。
