# Changelog

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
