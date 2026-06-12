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
