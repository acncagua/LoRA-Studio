# LoRA-Studio 日本語ドキュメント

このディレクトリは日本語の詳細マニュアルです。GitHubトップは英語の [README.md](../../README.md)、日本語トップは [README_ja.md](../../README_ja.md) に置き、運用手順や詳細説明はこの配下へ分割しています。

分割前のREADME_ja全文は説明ロスト防止のため [_archive_readme_ja_before_split.md](_archive_readme_ja_before_split.md) に保存しています。

## 目次

- [初回セットアップ](getting-started.md)
- [基本ワークフロー](workflow.md)
- [Project / Dataset](projects-datasets.md)
- [Training Job](training-jobs.md)
- [Review Session / Candidate Review](review-sessions.md)
- [Validation / Weight Calibration](validation-weight-calibration.md)
- [Recipe / Optimizer / Step Estimator](recipes-optimizers.md)
- [Machine Review](machine-review.md)
- [Storage / Cleanup](storage-cleanup.md)
- [Troubleshooting](troubleshooting.md)
- [Development Notes](development-notes.md)

## 読む順番

初めて使う場合は、[初回セットアップ](getting-started.md)、[基本ワークフロー](workflow.md)、[Training Job](training-jobs.md) の順に読むと全体像を掴みやすいです。

Recipe v2やOptimizer MasterからJobを作る場合は、[Recipe / Optimizer / Step Estimator](recipes-optimizers.md) を先に確認してください。

候補epochの見比べや採用後のweight確認は、[Review Session / Candidate Review](review-sessions.md) と [Validation / Weight Calibration](validation-weight-calibration.md) に分けています。
