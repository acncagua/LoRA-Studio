# Phase 12.5.1 LoRA-C3Lier Comparison Plan

This note defines the next practical comparison after the Phase 12.5 LoRA-C3Lier smoke checks.

## Scope

- Standard LoRA vs LoRA-C3Lier only.
- SDXL Character Face first, with Costume and Style as follow-up targets.
- LyCORIS LoCon, LoHa, LoKr, and generic LyCORIS modules are out of scope.
- The goal is operational comparison, not automatic winner selection.

## Fixed Inputs

- Dataset: `dataset_sdxl_zho`
- Trigger: `zho`
- Base model: `silenceMix_v70.safetensors`
- Resolution: `1024x1024`
- Optimizer: `AdamW8bit`
- Text Encoder training: off
- Training batch size: `1`
- Scheduler: `constant`

## Candidate Recipes

### Standard LoRA

- Network module: `networks.lora`
- Network args: none
- Suggested network size: `network_dim=32`, `network_alpha=16`

### LoRA-C3Lier

- Network module: `networks.lora`
- Network args: `conv_dim=8`, `conv_alpha=4`
- Suggested network size: `network_dim=32`, `network_alpha=16`
- This is sd-scripts LoRA-C3Lier, not LyCORIS LoCon.

## Training Plan

Run both recipes with the same dataset, prompt template, base model, and target step count.

Recommended first pass:

- `max_train_steps=1000`
- `save_every_n_steps=250`
- `sample_every_n_steps=250`
- `no_metadata=true`
- `save_model_as=safetensors`

Recommended practical pass:

- `target_steps=5000`
- checkpoint/save cadence every `500` or `1000` steps
- same validation preset for both networks

## Evaluation Plan

Use the existing Review / Validation pipeline instead of judging training samples alone.

1. Confirm training completed with return code `0`.
2. Confirm LoRA artifacts:
   - file exists
   - safetensors readable
   - tensor count > 0
   - no NaN / Inf tensors
3. Confirm sample images:
   - PNG readable
   - not flat black/white
   - no obvious full-noise failure
4. Generate Standard Candidate Comparison or a reduced validation set for matched checkpoints.
5. Compare:
   - human rating
   - Machine Review similarity
   - nearest dataset warning
   - overfit / fixed composition tendency
   - recommended weight range

## Acceptance Notes

- LoRA-C3Lier can be considered practically available when at least one real Mini Pilot and one validation run complete without command, artifact, or image-smoke failure.
- A small visual difference at very short steps is not a failure.
- Quality preference between Standard LoRA and LoRA-C3Lier must remain a human review decision.
