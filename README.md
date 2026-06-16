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

Current release: v0.2.0-beta
Development phase: Phase 11.x

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
- Reference Sets
- Experiment Comparison
- LoRA Selection Workflow
- Machine Review Assist
- Storage Cleanup Support

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

### Training Job

A Training Job represents one actual training run.
Jobs can be prepared, run, stopped, reviewed, cloned, archived, or used as the source for a new variant.

### Dataset Version

Dataset Versions capture the state of a dataset after rescans or caption edits.
This helps compare runs made before and after dataset cleanup.

### Validation Run

A Validation Run stores fixed validation conditions such as prompts, seeds, LoRA weights, and generated or imported validation images.

### Reference Set

A Reference Set contains human-selected reference images used for visual review and Machine Review Assist.

### Machine Review Assist

Machine Review Assist compares generated images with reference and dataset images using cached embeddings.
It is advisory only. Human review always takes priority.

The current real provider is:

- `transformers_clip`
- default model: `openai/clip-vit-base-patch32`

The mock provider remains available for tests and CI.

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

`transformers_clip` can help compare overall image meaning, composition, and atmosphere, but it is not a face identity model.
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
