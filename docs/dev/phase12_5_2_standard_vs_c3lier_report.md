# Phase 12.5.2 Standard LoRA vs LoRA-C3Lier Lightweight Comparison

Date: 2026-06-26

Branch: `phase12.5.2-standard-vs-c3lier`

Start commit: `f2838b4 Document phase12.5.1 C3Lier mini pilot results`

Reference release tags:

- `phase12.5`
- `v0.5.5-beta`

## Purpose

This report records a lightweight practical comparison between Standard LoRA and LoRA-C3Lier for SDXL Character Face training.

The question is whether LoRA-C3Lier provides a practical advantage over Standard LoRA for Character Face use when both are trained under the same operational LoRA-Studio settings.

This is not a parameter-count-matched experiment. LoRA-C3Lier adds convolutional LoRA parameters through `conv_dim` / `conv_alpha`, so it can be larger and slightly more expensive. The comparison should be read as:

> Does the extra LoRA-C3Lier cost produce enough practical quality benefit to justify using it?

## Parity Gate

Phase 12.5.1 produced Job #103:

- Job: `#103 phase12.5.1_lora_c3lier_character_face_1000step_mini_pilot`
- Network type: `lora_c3lier`
- Return code: `0`
- Recorded elapsed: `1131s`
- Final recorded artifact: `runs/job_000103/models/phase1251_lora_c3lier_character_face_1000step.safetensors`

Job #103 was not reused for the formal Phase 12.5.2 comparison.

Reason:

- Job #103 was archived during cleanup.
- Its `training_outputs` rows are marked deleted.
- The final safetensors artifact and samples were missing from disk.
- The original run remains useful as a Phase 12.5.1 smoke/mini-pilot result, but it no longer satisfies the artifact availability requirement for a parity comparison.

Decision:

- Re-run both candidates in the current environment.
- Treat Job #103 as historical reference only.

## Fixed Inputs

| Field | Value |
|---|---|
| Project | `#11 zho_test` |
| Dataset | `#9 dataset_sdxl_zho` |
| Dataset version | `#13` |
| Dataset manifest hash | `d1fdff576789baf5db7841ee3800451f084683636f852ece6dcba8a8fce62b5c` |
| Caption manifest hash | `0664b74fd405ef6e4f2865741df70c4aef3815fd8283d3fe5dc5513a1f9c6ca8` |
| Reference set | Project default `#4` |
| Trigger | `zho` |
| Image count | `37` |
| Base model | `models/silenceMix_v70.safetensors` |
| Base model size | `6938041780` bytes |
| Base model SHA-256 | `b508eb54bf04f6fcc2166884f4d980ace77f47a77e6f704b67ec985246a9056b` |
| sd-scripts commit | `a1b48df` |
| Python | sd-scripts venv Python |
| Torch | `2.8.0+cu128` |
| CUDA | `12.8` |
| GPU | `NVIDIA GeForce RTX 5090` |

No explicit training seed was found in the job snapshots. The comparison still used the same dataset, base model, optimizer, resolution, batch, cache, precision, and step settings for both candidates.

## Candidate Jobs

| Field | Standard LoRA | LoRA-C3Lier |
|---|---:|---:|
| Training Job | `#104` | `#105` |
| Recipe | `sdxl_character_face_adamw8bit_balanced` | `sdxl_character_face_lora_c3lier_adamw8bit_balanced` |
| Network module | `networks.lora` | `networks.lora` |
| Network type | `standard_lora` | `lora_c3lier` |
| Network dim / alpha | `32 / 16` | `32 / 16` |
| Conv dim / alpha | none | `8 / 4` |
| Optimizer | `AdamW8bit` | `AdamW8bit` |
| Learning rate / UNet LR | `0.0001 / 0.0001` | `0.0001 / 0.0001` |
| Text Encoder LR | `0` | `0` |
| Scheduler | `constant` | `constant` |
| Batch size | `1` | `1` |
| Repeats | `10` | `10` |
| Resolution | `1024x1024` | `1024x1024` |
| Max train steps | `1000` | `1000` |
| Max train epochs | not emitted | not emitted |
| Cache latents | true | true |
| Cache text encoder outputs | true | true |
| Network train UNet only | true | true |
| Mixed / save precision | `bf16 / bf16` | `bf16 / bf16` |

## Command Parity

The final command comparison had no non-permitted differences.

Allowed differences:

- Job IDs
- output directories
- output names
- C3Lier-only network args

Standard LoRA command confirmation:

- `--network_module networks.lora`
- `--network_dim 32`
- `--network_alpha 16`
- `--max_train_steps 1000`
- no `--max_train_epochs`
- no `conv_dim`
- no `conv_alpha`
- no `lycoris.kohya`
- no `algo=locon`

LoRA-C3Lier command confirmation:

- `--network_module networks.lora`
- `--network_dim 32`
- `--network_alpha 16`
- `--network_args conv_dim=8 conv_alpha=4`
- `--max_train_steps 1000`
- no `--max_train_epochs`

An initial non-parity difference was found before execution: the Standard LoRA recipe emitted `min_snr_gamma=5` while the C3Lier comparison candidate did not. This was removed from Job #104 before training so that the comparison remained network-type focused.

## Training Results

| Metric | Standard LoRA | LoRA-C3Lier | Difference |
|---|---:|---:|---:|
| Job ID | `#104` | `#105` | - |
| Return code | `0` | `0` | - |
| Elapsed | `2325s` | `2273s` | C3Lier `52s` faster |
| sec/step | `2.325` | `2.273` | C3Lier `0.052s/step` faster |
| Initial loss | `0.0463` | `0.0272` | - |
| Final loss | `0.0995` | `0.1020` | C3Lier `+0.0025` |
| Min loss | `0.0202` | `0.0272` | C3Lier `+0.0070` |
| Max loss | `0.1250` | `0.1680` | C3Lier `+0.0430` |
| Moving avg final loss | `0.09754` | `0.10200` | C3Lier `+0.00446` |
| NaN / Inf | none | none | - |
| LoRA outputs | `5` | `5` | same |
| Sample outputs | `12` | `12` | same |

Loss is recorded as an operational stability metric only. It is not used as the quality winner.

## Artifact Comparison

| Metric | Standard LoRA | LoRA-C3Lier |
|---|---:|---:|
| Final artifact | `runs/job_000104/models/phase1252_standard_lora_character_face_1000step.safetensors` | `runs/job_000105/models/phase1252_lora_c3lier_character_face_1000step.safetensors` |
| SHA-256 | `b5a22f5f6291cddfbed4bb6b86833aab59a690050fa0a2b6177a92faea2d7b11` | `757997d583ad3afd443e0bb7218319a5654804fe83ea252227dc91d0b86d96ca` |
| File size | `170540188` bytes | `180006392` bytes |
| Size ratio | `1.000x` | `1.055x` |
| Tensor count | `2166` | `2364` |
| Total elements | `85115602` | `89836308` |
| Linear-related tensors | `2166` | `2358` |
| Conv-related tensors | `0` | `6` |
| safetensors readable | OK | OK |
| NaN / Inf tensors | none | none |

LoRA-C3Lier produced a slightly larger artifact, as expected from the additional convolutional LoRA tensors.

## Lightweight Validation

Runtime validation preset:

- ID: `phase1252_lightweight_comparison`
- Name: `Phase 12.5.2 Character Face Lightweight Comparison`

Conditions:

- Candidates: Standard LoRA, LoRA-C3Lier
- Prompt groups: `basic_face`, `expression_pose`
- Seeds: `111111`, `222222`, `333333`
- Weights: `0`, `0.6`, `0.8`, `1.0`
- Hires: off
- Sampler: Euler a
- Steps: `28`
- CFG: `7.0`
- Resolution: `1024x1024`

Counts:

| Metric | Count |
|---|---:|
| Logical expected conditions | `48` |
| Physical generated images | `48` |
| Imported images | `48` |
| Embedded images | `48` |
| Machine-reviewed images | `48` |
| Baseline sharing | no |

Baseline sharing was not implemented for this ad-hoc comparison because the existing sharing path is tied to candidate comparison groups. The comparison remained valid with duplicated baseline images.

Validation runs:

| Candidate | Validation Run | Expected | Generated / Imported | Generation elapsed |
|---|---:|---:|---:|---:|
| Standard LoRA | `#52` | `24` | `24 / 24` | `475s` |
| LoRA-C3Lier | `#53` | `24` | `24 / 24` | `476s` |

## Machine Review

Embedding provider:

- Provider: OpenCLIP
- Model key: `open_clip_vit_b32_laion2b`
- Model: `ViT-B-32 / laion2b_s34b_b79k`
- Device / dtype: CUDA / fp16

Jobs:

- Standard LoRA: embedding job `#286`, machine review job `#125`
- LoRA-C3Lier: embedding job `#287`, machine review job `#126`

Aggregate scores:

| Metric | Standard LoRA | LoRA-C3Lier |
|---|---:|---:|
| Score count | `24` | `24` |
| Avg reference similarity | `0.674025` | `0.674884` |
| Avg nearest reference similarity | `0.761548` | `0.763735` |
| Avg dataset similarity | `0.685495` | `0.688142` |
| Avg nearest dataset similarity | `0.798393` | `0.798015` |
| Avg assist score | `0.761548` | `0.763735` |

Weight-level reference / dataset averages:

| Weight | Standard ref | C3Lier ref | Standard dataset | C3Lier dataset |
|---:|---:|---:|---:|---:|
| `0.0` | `0.649728` | `0.649728` | `0.656388` | `0.656388` |
| `0.6` | `0.676341` | `0.678237` | `0.686665` | `0.690323` |
| `0.8` | `0.692209` | `0.691517` | `0.706878` | `0.709421` |
| `1.0` | `0.677824` | `0.680053` | `0.692050` | `0.696437` |

Machine Review differences were small. They do not establish a clear winner.

## Blind Review Assets

Runtime export directory:

`exports/phase12_5_2_standard_vs_c3lier`

Files:

- `contact_sheet_basic_face.png`
- `contact_sheet_expression_pose.png`
- `blind_candidate_map.json`
- `summary.json`

Blind mapping:

```json
{
  "candidate_a": "standard_lora",
  "candidate_b": "lora_c3lier"
}
```

The contact sheets use Candidate A / Candidate B labels and do not display network type names.

Review Session:

- ID: `#53`
- Name: `Phase 12.5.2 Standard LoRA vs LoRA-C3Lier Blind Comparison`
- Status: `completed`
- Automation status: `technical_complete`
- Expected / generated / imported / scored: `48 / 48 / 48 / 48`

## Human Review

Human Review has not been entered.

Decision status:

`human_review_pending`

Codex did not assign human scores, subjective winner labels, or final adoption decisions.

## Technical Acceptance

Technical comparison status:

`PASS`

Completed:

- Job #103 parity gate
- new Standard LoRA 1000-step training
- new LoRA-C3Lier 1000-step training
- command parity check
- artifact checks
- lightweight matched validation generation
- OpenCLIP embedding
- Machine Review
- blind contact sheets
- runtime Review Session
- performance and artifact summary

## Decision Guidance

Because Human Review is pending and Machine Review differences are small, Standard LoRA remains the default Character Face recommendation for now.

LoRA-C3Lier should move to a deeper Phase 12.5.3 test only if human review of the blind contact sheets shows a consistent visual advantage, especially for face identity, expression/pose stability, or structure quality at the same weights.

Suggested next phase options:

- If C3Lier looks better: run a 3000-5000 step comparison or a full 45-image Standard Validation comparison.
- If no clear visual winner: keep Standard LoRA as default and keep LoRA-C3Lier as an optional network type.
- If 1000 steps is inconclusive: repeat at 3000 steps with the same candidate structure.
