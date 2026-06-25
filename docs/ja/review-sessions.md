# Review Session / Candidate Review

Review Sessionは採用前に候補epochを横並びで比較する画面です。採用済みLoRAのweight検証とは分けて扱います。

## Quick Candidate Review

軽量な候補比較です。loss候補epoch最大3件、prompt 3種、seed `111111`、weight `0.6` / `0.8`、Hiresなし、最大18枚程度を想定します。

## Standard Candidate Comparison

候補epochごとにStandard Validation v1相当の条件を作り、45枚 x 候補数をまとめて比較します。Candidate Comparison groupとして扱い、最後にEpoch横断Matrixで比較します。

weight 0 baselineはgroup内で共有できるため、論理上の比較条件を保ったまま物理生成枚数を減らせる場合があります。

## Review Matrix

Review Matrixでは、同一prompt / seed / weightごとに候補epochを横並びで見ます。Machine Assistは参考情報であり、最終判断は人間評価と画像比較を優先します。

Machine Assistの差が小さい場合は `no_clear_winner` として候補群を表示します。この場合は近隣Epoch追加検証や人間評価を優先してください。

## Expanded Neighbor Review

Review Session詳細から、中心epochの±1または±2を追加検証できます。候補が僅差の場合に、採用前の局所比較として使います。
