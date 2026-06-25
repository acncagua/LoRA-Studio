# Machine Review

Machine Review Assistは画像の機械的な比較補助です。採用判断を自動化するものではありません。

## Provider

主な想定providerはOpenCLIP / transformers CLIPです。mock providerはテストやCI用に残しています。

## できること

- reference similarityの計算
- dataset nearestの確認
- 過学習やdataset寄りの兆候の補助表示
- candidate group / no_clear_winnerの補助判定
- Weight Calibration Matrixでの参考score表示

## できないこと

- 顔同一性の厳密判定
- 衣装や小物の細部評価
- 人間評価の置き換え
- 自動採用

## Reference Set

キャラクターLoRAでは、正面顔、上半身、全身、表情差分などをReference Setに入れると比較しやすくなります。style LoRAでは、より広い作例をReferenceにする方が安定します。

## Embedding Cache

Embeddingは画像単位でcacheされます。Validation RunやCandidate Comparisonでは、重複画像や共有baselineをまとめて処理することで時間短縮できます。
