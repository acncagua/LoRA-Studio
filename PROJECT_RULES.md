# Project Rules

## Language

- ユーザー向け説明、README、画面表示、コメントの原則は日本語で記述する。
- コード上の識別子、関数名、クラス名は英語でよい。

## Encoding

- 日本語を含む全ファイルはUTF-8で保存する。
- Pythonでファイルを読み書きする場合は `encoding="utf-8"` を明示する。
- PowerShell、Markdown、JSON、TOML、HTMLテンプレートもUTF-8で保存する。

## Safety

- `sd-scripts` の内部ファイルは原則として改造しない。
- アプリ本体venvと `sd-scripts` venvは分離する。
- 実行コマンド、実行時params_json、環境snapshotを保存する。
- 学習結果の評価は画像評価とloss健全性を分ける。

## sd-scripts

- MVPでは `kohya-ss/sd-scripts` の `v0.10.5` を使用する。
- 対応commitの目安は `a1b48df` とする。
- 更新時はtagとcommit hashを記録する。

## Web App

- 初期起動hostは `127.0.0.1` とする。
- `0.0.0.0` 公開は明示設定がある場合のみ許可する。
