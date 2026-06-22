# LoRA-Studio

LoRA-Studio is a workflow-oriented training management platform
for Stable Diffusion LoRA development.

It helps creators manage datasets,
training jobs, dataset versions,
validation runs, experiment tracking,
and final LoRA selection through a
unified workflow.

Built around real-world SDXL / SD1.5
character LoRA iteration workflows,
where reproducibility, comparison,
and validation become more difficult
than training itself.

[日本語 READMEはこちら](README_ja.md)

---

## Status

Current release: v0.4.4-beta
Development phase: Phase 11.7

The core workflow is operational and
actively used for local LoRA production,
but APIs and workflows may still change.

## Why LoRA-Studio?

Training a LoRA is easy.

Managing dozens of experiments,
dataset revisions,
validation runs,
sample reviews,
and final model selection is not.

LoRA-Studio provides a unified workflow
for the entire LoRA lifecycle.

## Key Features

- Dataset Management
- Dataset Version Tracking
- Training Job Management
- Validation Runs
- Weight Calibration Pipeline
- Step Estimator / Target Step Assistant
- Reference Sets
- Experiment Comparison
- LoRA Selection Workflow
- Machine Review Assist
- Storage Cleanup Support

## Phase 11.7: Weight Calibration Pipeline

Candidate Review and Weight Calibration are separate workflows.
Candidate Review decides which epoch should be adopted.
Weight Calibration runs after a LoRA / selected output has been adopted,
and decides which LoRA weight range should be used.

Standard Validation uses 45 images:
3 prompts x 3 seeds x 5 weights.
Weight `0` is the baseline and is generated without LoRA network weights.
Standard comparison is based on non-Hires images.
Extended Validation can include Hires on/off comparisons,
but Hires results are treated as final appearance checks rather than the main weight recommendation basis.

The pipeline can be started from Project, Job, Review Session, LoRA Profile,
or Validation Run context. It prepares expected conditions, runs sd-scripts image generation,
imports images, computes Embeddings, runs Machine Review Assist, and writes a Weight Review Matrix.
Machine Assist is supporting information; human review fields take priority.
Suggested weights are never applied automatically. Use the explicit Apply to Profile action
to update the selected LoRA profile.

## Step Estimator / Target Step Assistant

Epoch count alone is not enough to judge training volume.
LoRA-Studio estimates expected training steps with:

`effective_batch_size = train_batch_size x gradient_accumulation_steps x num_processes`

`steps_per_epoch = ceil(sum(subset.image_count x subset.repeats) / effective_batch_size)`

`total_steps = steps_per_epoch x max_train_epochs`

Job creation, job editing, job detail, and preflight show the estimate against optimizer and recipe target steps.
Target steps are resolved from `training_recipes`, then `optimizer_profiles`,
then `optimizer_definitions`, with a global fallback if no catalog entry applies.
The assistant uses the recipe recommended target as its initial value and auto-calculates repeats,
and proposes save/sample intervals when epoch count would create too many checkpoints.
Increasing repeats does not increase the number of output LoRA checkpoints, but it increases steps per epoch.
Very high repeats can overfit or make the LoRA too fixed to the dataset.

Direct `max_train_steps` is treated as Advanced. It overrides epoch-based total steps and can make
epoch-based review less clear, so the normal workflow recommends adjusting repeats / epochs / batch first.

## Screenshots

### Dashboard

![Dashboard](docs/screenshots/dashboard.png)

The dashboard gives you a high-level view of recent projects, training job status, failed or draft jobs, and cleanup entry points.

### Recommended Workflow

![Recommended Workflow](docs/screenshots/recommended-workflow.png)

The workflow page shows the intended path from dataset preparation to training, review, validation, and the next experiment.

### Create Training Job

![Create Training Job](docs/screenshots/create-training-job.png)

Create a training job by selecting a project, dataset, dataset version, preset, base model, and sample prompt template in one place.

### Training Job Management

![Training Job Management](docs/screenshots/training-job-management.png)

Manage draft, prepared, running, completed, failed, archived, and deleted training jobs with clear actions for prepare, run, clone, compare, and archive.

### Training Result Management

![Training Result Management](docs/screenshots/training-result-management.png)

Inspect loss, step consistency, output LoRA files, sample images, selected epochs, and review notes from the training job detail screen.

Screenshots are captured from sanitized English demo views for OSS submission.
Labels and sample data may differ slightly from the local development UI.

---

## What It Does

LoRA-Studio is designed for local LoRA experimentation, especially SDXL / SD1.5 character and illustration LoRA workflows.

It does not try to fully automate creative judgment. Instead, it keeps the repetitive and error-prone parts organized:

- dataset registration and rescanning
- caption and trigger consistency checks
- dataset version snapshots
- project-based LoRA experiment tracking
- training job creation, preparation, execution, stop, and result import
- sd-scripts command generation with argv-based execution
- loss / metric import and step consistency checks
- epoch-by-epoch sample comparison
- selected LoRA management
- validation runs and validation image review
- reference set management
- embedding cache and Machine Review Assist
- storage cleanup for large `runs/` outputs

The goal is to make LoRA iteration easier to reproduce, compare, and clean up.

## Core Concepts

### Project

A Project represents one LoRA creation effort.
It groups datasets, training jobs, selected outputs, validation runs, reference sets, and review notes.
In normal use, start from the Project detail page. It acts as the workspace for the whole LoRA creation flow.

### Training Job

A Training Job represents one actual training run.
Jobs can be prepared, run, stopped, reviewed, cloned, archived, or used as the source for a new variant.
The Training Job detail page is for inspecting one run: setup, command generation, logs, metrics, outputs, and cleanup.
It is not intended to replace the Project page as the overall workspace.
The detail page is organized into tabs so setup, results, review, validation, recommendations, files, and technical data stay separated.

### Dataset Version

Dataset Versions capture the state of a dataset after rescans or caption edits.
This helps compare runs made before and after dataset cleanup.

### Validation Run

A Validation Run stores fixed validation conditions such as prompts, seeds, LoRA weights, and generated or imported validation images.
Validation Runs are mainly for post-selection weight calibration after a candidate epoch has been chosen.

### Reference Set

A Reference Set contains human-selected reference images used for visual review and Machine Review Assist.

### Review Session

A Review Session is the pre-selection comparison workspace for candidate epochs.
It stores candidate epoch conditions, generated review images, Machine Review Assist results, and the cross-epoch Review Matrix.
Use Review Sessions before selecting a final epoch. Use Validation Runs after selecting an epoch.
Open Review Sessions from the Project or Training Job page when you want to compare candidate epochs and select the final LoRA.

### Machine Review Assist

Machine Review Assist compares generated images with reference and dataset images using cached embeddings.
It is advisory only. Human review always takes priority.

Current real providers are:

- `transformers_clip`
- `open_clip`

Default models:

- transformers CLIP: `openai/clip-vit-base-patch32`
- OpenCLIP: `ViT-B-32` / `laion2b_s34b_b79k`

The mock provider remains available for tests and CI.
Model downloads are allowed only when the embedding setting explicitly enables model download.

### Candidate Review Preparation

After a training job completes, LoRA-Studio can prepare a small pre-selection review matrix before you choose the final epoch.

The Review Preparation pipeline uses loss candidate epochs and their neighboring epochs, generates a compact set of images with sd-scripts, registers the images, computes embeddings, runs Machine Review Assist, and writes a cross-epoch matrix HTML file.

This is separate from Validation Runs:

- Review Preparation is for choosing a candidate epoch before adoption.
- Weight Calibration / Validation Runs are for checking LoRA weights after an epoch has been selected.
- Standard Validation 45-image runs should usually happen after candidate selection, not before every epoch comparison.

## Current Scope

LoRA-Studio currently focuses on:

- local Windows workflow
- SQLite-backed project data
- FastAPI + Jinja2 web UI
- sd-scripts integration
- SDXL / SD1.5 LoRA training management
- validation and review support
- experiment traceability

The project currently uses `kohya-ss/sd-scripts` `v0.10.5` as the verified sd-scripts version for the beta workflow.

## Not In Scope Yet

The following are not implemented as full automatic workflows yet:

- WebUI / reForge API automatic generation
- ChatGPT API image evaluation
- full AI-based visual scoring
- face identity verification
- automatic parameter optimization
- FLUX support
- LyCORIS / LoCon support

## Quick Start

### 1. Install App Dependencies

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_app.ps1
```

### 2. Start LoRA-Studio

```bat
start_lora_studio.bat
```

Or run it manually:

```powershell
.\.venv\Scripts\python.exe .\start_lora_studio.py
```

Open:

```text
http://127.0.0.1:8768
```

Normal startup does not kill existing processes.
If you explicitly want LoRA-Studio to release its configured port first, use:

```powershell
.\.venv\Scripts\python.exe .\start_lora_studio.py --force-release-port
```

Only LoRA-Studio's configured port is targeted.

### 3. Set Up sd-scripts

LoRA-Studio can set up the verified sd-scripts version:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_sd_scripts.ps1 -ReleaseTag v0.10.5 -CudaProfile cu128 -MixedPrecision bf16
```

If `external/sd-scripts` or its venv is missing, startup can run the setup path automatically.

## Typical Workflow

1. Create or open a Project.
2. Register a dataset folder.
3. Rescan and check captions, trigger consistency, and dataset health.
4. Create a Dataset Version after cleanup.
5. Create a Training Job with a preset and base model.
6. Prepare files and run preflight checks.
7. Run training through sd-scripts.
8. Review logs, loss, outputs, and sample images.
9. Select a candidate LoRA epoch.
10. Run validation images and review them.
11. Export the selected LoRA.
12. Archive or clean up large unselected outputs.

Pilot training is optional.
It is useful for first-time setup, a new dataset, a new base model, a new preset, or after sd-scripts changes.
If preflight is OK and similar conditions already completed successfully, you can go directly to a standard training job.

## Machine Review Notes

Machine Review Assist is not an automatic judge.

`transformers_clip` and `open_clip` can help compare overall image meaning, composition, and atmosphere, but they are not face identity models.
For small facial details, costume details, and character-specific parts, human review remains the source of truth.

Reference Sets should include varied roles such as:

- `face_front`
- `upper_body`
- `full_body`
- `expression`

For character LoRA review, 3-5 reference images are a practical minimum.
For style LoRA review, a broader set is recommended.

## Storage Notes

Training can create many `.safetensors` files and sample images under `runs/`.
LoRA-Studio provides archive and cleanup tools, but cleanup is never automatic.

If the project is inside OneDrive, deleting or moving large files may also affect cloud sync.

## Tests

```powershell
python -m compileall app start_lora_studio.py
python -m pytest -q
git diff --check
```

## Documentation

The Japanese README currently contains the fuller operational manual:

- [README_ja.md](README_ja.md)

More focused English documentation may be split into `docs/` over time.

## License

See [LICENSE](LICENSE).
