# Recipe / Optimizer / Step Estimator

## Recipe v2

Recipe v2は、目的、Optimizer、Network Type、target steps、主要学習paramsをまとめた学習設定マスタです。Recipeカードでは短い表示名、optimizer、purpose、target steps、risk、expected behaviorを確認できます。

## Optimizer Master

Optimizer MasterはOptimizerの意味、learning rate semantics、推奨target steps、依存関係、Smoke / Mini Pilot結果を管理します。

AdamW8bit / PagedAdamW8bit / Lionは通常LRです。Prodigy / DAdaptAdam / DAdaptLionの `learning_rate=1.0` はAuto-LR倍率であり、AdamW系の `1e-4` とは意味が違います。Adafactor Autoはrelative step運用です。

## Step Estimator

epoch数だけでは学習量を判断できないため、LoRA-Studioはexpected stepsを計算します。

```text
effective_batch_size = train_batch_size * gradient_accumulation_steps * num_processes
steps_per_epoch = ceil(total_training_images_with_repeats / effective_batch_size)
total_steps = steps_per_epoch * max_train_epochs
```

Target Step Assistantは、recipe / optimizerのrecommended target stepsに近づくようにrepeats、epochs、batchの候補を出します。通常運用では `max_train_steps` を直接指定せず、repeats / epochs / batchで調整します。

## Compatibility Check

cache text encoderとTE学習の矛盾、UNet onlyとTE LRの矛盾、DAdapt / Prodigy scheduler、Adafactor relative step、未対応Network TypeなどをERROR / WARNING / NOTEで表示します。

## Smoke / Mini Pilot

Smoke Testはsd-scriptsで起動できるかを見る最小確認です。Mini Pilotは100〜300step程度の短い実学習で、Optimizer Profileを実用候補として残せるかを見る補助です。どちらも品質保証ではありません。
