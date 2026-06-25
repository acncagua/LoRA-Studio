# Phase 12.5.1 LoRA-C3Lier Practical Mini Pilot Report

Date: 2026-06-26

## Summary

Phase 12.5.1 validated LoRA-C3Lier beyond the initial Character Face smoke test.

- Costume LoRA-C3Lier: true 2-step smoke completed.
- Style LoRA-C3Lier: true 2-step smoke completed.
- Character Face LoRA-C3Lier: 1000-step Mini Pilot completed.
- Character Face final artifact passed safetensors checks.
- Character Face weight 0 / weight 1 image smoke completed.

No failure category was assigned.

## Environment

- sd-scripts path: `external/sd-scripts`
- sd-scripts commit: `a1b48df`
- Python: `3.10.11`
- Torch: `2.8.0+cu128`
- CUDA: `12.8`
- GPU: `NVIDIA GeForce RTX 5090`
- Mixed precision: `bf16`

## Dataset / Model

- Dataset ID: `9`
- Dataset: `dataset_sdxl_zho`
- Trigger: `zho`
- Images: `37`
- Captions: `37`
- Base model: `models/silenceMix_v70.safetensors`
- Reference set observed for the same project/dataset: `#6 zho`

## Network Configuration

LoRA-C3Lier uses the sd-scripts standard LoRA module with convolution parameters:

- `network_module=networks.lora`
- `network_dim=32`
- `network_alpha=16`
- `network_args=conv_dim=8 conv_alpha=4`

For 2-step smoke jobs, `network_dim=4` and `network_alpha=2` were intentionally used as smoke overrides. `conv_dim=8` and `conv_alpha=4` remained present.

## Costume True 2-Step Smoke

- Job ID: `101`
- Recipe: `sdxl_costume_lora_c3lier_adamw8bit_balanced`
- Status: `completed`
- Return code: `0`
- Elapsed: `119 sec`
- Output LoRA count: `3`
- Sample image count: `9`
- Loss values: step 1 `0.00894`, step 2 `0.135`
- Command checks:
  - `--network_module networks.lora`
  - `--network_args conv_dim=8 conv_alpha=4`
  - `--max_train_steps 2`
  - `--max_train_epochs` absent
  - `--text_encoder_lr` absent
- Safetensors artifact check: `ok`
- Tensor count: `2364`
- Sample PNG check: `ok`

## Style True 2-Step Smoke

- Job ID: `102`
- Recipe: `sdxl_style_lora_c3lier_adamw8bit_soft`
- Status: `completed`
- Return code: `0`
- Elapsed: `107 sec`
- Output LoRA count: `3`
- Sample image count: `9`
- Loss values: step 1 `0.173`, step 2 `0.104`
- Command checks:
  - `--network_module networks.lora`
  - `--network_args conv_dim=8 conv_alpha=4`
  - `--max_train_steps 2`
  - `--max_train_epochs` absent
  - `--text_encoder_lr` absent
- Safetensors artifact check: `ok`
- Tensor count: `2364`
- Sample PNG check: `ok`

## Character Face 1000-Step Mini Pilot

- Job ID: `103`
- Recipe: `sdxl_character_face_lora_c3lier_adamw8bit_balanced`
- Status: `completed`
- Return code: `0`
- Elapsed: `1131 sec`
- Steps: `1000`
- Approx seconds per step: `1.131`
- Output LoRA count: `5`
- Sample image count: `12`
- Final LoRA: `runs/job_000103/models/phase1251_lora_c3lier_character_face_1000step.safetensors`
- Final LoRA SHA256: `d27a90d86f80b6f9cd29d8c842a0ecf257d62208084bfcfb10c647bfd118b495`
- Final LoRA file size: `180006392 bytes`
- Safetensors artifact check: `ok`
- Tensor count: `2364`

Command checks:

- `--network_module networks.lora`
- `--network_dim 32`
- `--network_alpha 16`
- `--network_args conv_dim=8 conv_alpha=4`
- `--max_train_steps 1000`
- `--save_every_n_steps 250`
- `--sample_every_n_steps 250`
- `--max_train_epochs` absent
- `--text_encoder_lr` absent
- `--network_train_unet_only` present
- `--cache_text_encoder_outputs` present

Loss summary:

- Metric count: `1000`
- Initial loss: `0.0077`
- Final loss: `0.0993`
- Min loss: `0.00585`
- Max loss: `0.136`
- Moving average final loss: `0.09892`
- NaN / Inf: none observed
- Loss status: `ok`

Saved checkpoints:

- step `250`
- step `500`
- step `750`
- step `1000`
- final

## Image Smoke

Character Face final LoRA was used for a 512px weight 0 / weight 1 image smoke.

- Status: `image_smoke_ok`
- Elapsed: `41 sec`
- Difference score: `1.172689`
- Weight 0 image: PNG readable, `512x512`, variance `8462.124454509025`
- Weight 1 image: PNG readable, `512x512`, variance `5362.751967379663`

The generated smoke images and log are runtime artifacts and are intentionally not tracked by Git.

## Review / Matrix Notes

The 1000-step Mini Pilot uses step checkpoints rather than epoch checkpoints. LoRA-Studio generated a single epoch summary for the metrics, but no candidate Review Session was created automatically for this job.

This is acceptable for Phase 12.5.1 because the purpose was practical Mini Pilot validation of LoRA-C3Lier execution and artifacts. Full candidate comparison should use the follow-up comparison plan with matched Standard LoRA and LoRA-C3Lier validation runs.

## Recommendation

LoRA-C3Lier should remain enabled as a practical candidate network type for SDXL Character Face, Costume, and Style recipes.

Recommended next phase:

- Phase 12.5.2 or later: matched Standard LoRA vs LoRA-C3Lier practical comparison.
- Run equal-step Standard LoRA and LoRA-C3Lier jobs.
- Use Candidate Standard Comparison or Weight Calibration style validation to compare human review and machine assist signals.
