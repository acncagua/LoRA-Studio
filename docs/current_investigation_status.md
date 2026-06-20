# 現在の調査結果

作成日時: 2026-06-21 06:48:13 +09:00  
対象アプリ: LoRA-Studio  
確認コミット: `cfb768f`

## 基本情報詳細

- 対象画面: 学習ジョブ詳細 / Review Preparation / Review Session 周辺
- 直近で確認した事象: Review Preparation が完了済みなのに、画面上部の主要操作が「開始中...」のまま残る
- 代表例: Job #31 / Review Session #14
- 実際の状態:
  - Review Session は `completed`
  - 登録画像は `42 / 42`
  - Machine Review は `42 / 42`
  - Review Matrix は作成済み
- HTTP確認:
  - `/jobs/31?...#review-preparation` は `200`
  - 古い「レビュー準備を開始しました。PID...」表示は消えることを確認
  - 「開始中...」ボタンが残らないことを確認
  - 「レビューMatrixを開く」が表示されることを確認
- 実行確認:
  - `python -m compileall app start_lora_studio.py` 成功
  - `python -m pytest -q -p no:cacheprovider` 成功: `73 passed`
  - `git diff --check` 成功
- サーバ:
  - 修正後に `127.0.0.1:8768` で再起動済み

## 既知の不具合

- 長時間処理完了後、画面表示だけが古い「実行中」状態のまま残ることがある
- URLクエリ由来の一時メッセージが残り、実際のDB状態と矛盾した表示になることがある
- 非同期処理の完了後に、ボタン状態や件数表示が即時更新されない画面が複数存在する可能性がある
- Review Preparation周辺では、完了済みセッションがあるのに「開始中...」や「レビュー準備を開始しました」が残るケースを確認済み

### 原因候補

- POST後に付与される一時メッセージ用クエリパラメータが、処理完了後も表示条件に残っていた
- AJAX開始時に無効化したボタンを、完了時に復帰または適切なリンクへ差し替える処理が不足していた
- `Review Preparation` の状態表示がDBの最新状態ではなく、直前操作のUI状態に引きずられていた
- 複数の非同期処理で、共通の「開始 / 実行中 / 完了」表示更新ルールがまだ完全には統一されていない

### 調査済み

- Job #31 のReview Preparation表示
  - 完了済みなのに開始中表示が残る問題を確認
  - 完了時に「レビューMatrixを開く」へ切り替わるよう修正済み
- URLクエリメッセージ
  - `review_prepare`
  - `review_prepare_error`
  - 上記を一時メッセージ削除対象へ追加済み
- テンプレート側の保護
  - 完了済みセッションに対して、古い「レビュー準備を開始しました」系メッセージを表示しないよう修正済み
- 実行中プロセス状態
  - DB上の running 系操作を確認
  - 調査時点では training / validation generation / review session / embedding / machine review に実行中レコードなし
- 表示確認
  - `/jobs/31` は正常表示
  - 古い開始メッセージなし
  - 開始中ボタンなし
  - Review Matrixリンクあり

### 未確認

- 同種の表示残りが、以下の画面で完全に解消されているか
  - Validation Run 一括画像生成
  - Validation Run Embedding / Machine Review 一括実行
  - Dataset Embedding
  - Reference Set Embedding
  - Machine Review Jobs一覧
  - Active Operation Monitor表示
- 非同期処理完了後に、件数カードやボタン状態が全画面で自動更新されるか
- 複数処理を連続実行した場合に、古い通知や古いログ表示が残らないか
- ブラウザの戻る / リロード / タブ切り替え後に、同じ一時メッセージが再表示されないか
- completed / failed / stopped それぞれの終了状態で、主要操作ボタンが正しく切り替わるか
- Review Session以外の画面にも、同様のサーバ側ガードが必要かどうか

