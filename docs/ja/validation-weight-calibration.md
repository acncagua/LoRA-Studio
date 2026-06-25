# Validation / Weight Calibration

Validation RunとWeight Calibrationは、採用済みLoRAのweight範囲や安定性を確認するための採用後工程です。

## Standard Validation

標準構成では45枚を想定します。

- 3 prompts
- 3 seeds
- 5 weights
- Hiresなしを基本比較軸にする

weight `0` はLoRAなしbaselineです。

## Weight Calibration Pipeline

Weight Calibrationは次の工程を順に実行します。

1. expected conditions作成
2. sd-scripts画像生成
3. import
4. embedding
5. Machine Review
6. Matrix作成
7. Profile反映

推奨weightは自動では適用されません。明示的なApply操作でLoRA Profileへ反映します。

## Matrixの見方

Weight Review Matrixでは、weightごとの絵柄の強さ、破綻、dataset近傍、reference similarity、人間評価を確認します。`strong_but_usable` や `weak_but_usable` は採用判断の補助であり、最終的には用途ごとの見た目を優先します。

## Performance Summary

生成、import、embedding、Machine Review、matrix作成の時間を記録します。次回見積もりや遅いstageの切り分けに使います。
