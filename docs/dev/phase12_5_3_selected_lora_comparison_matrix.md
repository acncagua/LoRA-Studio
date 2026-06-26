# Phase 12.5.3 Selected LoRA Comparison Matrix

## Purpose

Phase 12.5.3 adds a comparison layer for selected LoRA profiles registered in the LoRA Library.
The existing Candidate Standard Comparison remains focused on multiple epochs from the same Training Job.
The new comparison session can group final selected LoRA artifacts across different Jobs while preserving the exact candidates, artifacts, Validation Runs, parity result, Matrix, and final decision.

LoRA-C3Lier is read as "セリア" in Japanese explanations. The formal display name remains `LoRA-C3Lier`, and the internal ID remains `lora_c3lier`.

## Scope

- Compare 2 to 6 selected LoRA profiles from the same Project.
- Support Controlled and Practical comparison modes.
- Support Network Type, Optimizer Profile, Training Recipe, and Selected Artifact axes.
- Reuse existing Validation Runs when artifact and generation conditions match.
- Generate a named cross-run Matrix that always shows real candidate names.
- Save human comparison decisions without automatically applying them to Project, Recipe, Optimizer, or Profile settings.

Blind Review and Candidate A/B/C labels are intentionally not implemented in this phase. Fairness is handled by Parity Gate checks and fixed generation conditions.

## Database

New tables:

- `lora_comparison_sessions`
- `lora_comparison_candidates`

New `validation_runs` snapshot columns:

- `artifact_path_snapshot`
- `artifact_source_kind`
- `artifact_sha256_snapshot`
- `artifact_file_size_snapshot`

These columns allow existing and newly created Validation Runs to be tied back to the exact LoRA artifact used for generation.

## Artifact Resolver

`app/services/lora_artifacts.py` resolves LoRA artifacts in this order:

1. `training_outputs.file_path`
2. `training_outputs.external_copy_path`
3. `selected_lora_profiles.exported_model_path`
4. `selected_lora_profiles.selected_model_path`
5. `training_jobs.adopted_model_path`

If an expected SHA-256 exists, mismatched files are rejected and the resolver tries the next candidate path.
If no expected SHA-256 is recorded, the resolver calculates the current SHA-256 and records a warning in the candidate snapshot.

## Cleanup Protection

Storage cleanup now checks whether an output is referenced by a non-archived LoRA comparison session.
Cleanup is blocked if deleting the file would leave the comparison with no verified artifact copy.
If an exported copy is verified by SHA-256, deleting the runs-side selected artifact is still allowed.

## Parity Gate

All comparison modes require:

- 2 to 6 candidates
- Same Project
- Same model family
- Verified artifacts
- Same generation base model for the comparison
- Usable Validation Preset

Controlled Network Type comparison permits differences in `network_type_id` and network-specific args only.
For Standard LoRA vs LoRA-C3Lier, the expected difference is that LoRA-C3Lier uses `conv_dim` / `conv_alpha`, while Standard LoRA does not.

If the training seed is not recorded, the session can be created only as a warning-level comparison.

## Validation Run Reuse

Existing Validation Runs are reused when:

- The selected profile or selected output matches.
- The Validation Preset matches.
- The artifact SHA-256 matches or can be safely backfilled.
- The generation condition fingerprint matches across candidates.

The generation fingerprint includes prompt key, prompt, negative prompt, trigger, seed, weight, Hires flag, image size, sampler, steps, CFG, and base model.
Candidate-specific LoRA filenames and Validation Run IDs are excluded.

## Matrix

`build_run_cross_matrix_html()` is a new generic cross-run Matrix builder.
The existing `build_epoch_cross_matrix_html()` remains available for same-Job epoch comparisons.

LoRA comparison Matrix columns use the saved candidate display labels, for example:

- Standard LoRA
- LoRA-C3Lier（セリア）

The Matrix reuses existing Validation image review forms and Machine Review score display.

## Decisions

Supported decision states:

- `human_review_pending`
- `candidate_preferred`
- `no_clear_winner`
- `retest_required`

Saving `candidate_preferred` requires a candidate from the same session and no missing generated images.
The decision does not automatically update Project adoption, Recipe defaults, Optimizer profiles, or recommended weights.

## Phase 12.5.2 Reuse Acceptance

When runtime data is available, Job #104 and Job #105 can be registered as a Controlled / Network Type comparison.
The expected result is:

- Standard LoRA and LoRA-C3Lier（セリア） are shown by real name.
- Existing Validation Runs are reused.
- No new image generation is started when all images already exist.
- Decision remains `human_review_pending` until the user performs Human Review.

## Limitations

- Cross-Project comparison is not supported in this MVP.
- Job-crossing weight 0 baseline sharing is not implemented.
- External unregistered LoRA files are not supported.
- Blind review is intentionally out of scope.
