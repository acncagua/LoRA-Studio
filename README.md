# LoRA-Studio

[日本語 README](README_ja.md)

## Overview

LoRA-Studio is a local workflow manager for Stable Diffusion LoRA development,
with a focus on SDXL / SD1.5 character and illustration LoRA iteration.

It does not try to replace creative judgment. Instead, it keeps datasets,
training jobs, validation runs, candidate reviews, optimizer recipes, and final
LoRA selection in one reproducible workflow.

## Screenshots

Screenshots are captured from sanitized English demo views for OSS submission.
Labels and sample data may differ slightly from a local development workspace.

### Dashboard

![Dashboard](docs/screenshots/dashboard.png)

### Recommended Workflow

![Recommended Workflow](docs/screenshots/recommended-workflow.png)

### Create Training Job

![Create Training Job](docs/screenshots/create-training-job.png)

### Training Job Management

![Training Job Management](docs/screenshots/training-job-management.png)

### Training Result Management

![Training Result Management](docs/screenshots/training-result-management.png)

## Features

- Project-based LoRA experiment tracking
- Dataset registration, rescanning, trigger checks, and version snapshots
- Training Job creation, preparation, execution, stop, clone, and archive
- Recipe v2 / Optimizer Master with Step Estimator and Compatibility Check
- LoRA-C3Lier recipes for sd-scripts `networks.lora` with `conv_dim` / `conv_alpha`
- Post-training Review Automation and Candidate Standard Comparison
- Review Matrix and human review fields for candidate epoch selection
- Validation Run and Weight Calibration Pipeline for adopted LoRAs
- OpenCLIP / Machine Review Assist and Reference Sets
- Retry Signal Summary and Recommendation Engine separation
- Runtime storage settings and cleanup support for large generated artifacts
- Gradual Japanese / English i18n for screenshots and OSS-facing UI

## Recommended Workflow

1. Create a Project for one LoRA creation effort.
2. Register or rescan the dataset and check captions / trigger consistency.
3. Create a Dataset Version before training.
4. Create a Training Job using Recipe Wizard or a legacy preset.
5. Prepare files, review preflight, then run training through sd-scripts.
6. Use Review Session / Candidate Review to choose the candidate epoch.
7. Adopt the LoRA output and run Weight Calibration / Validation.
8. Apply the recommended weight range to the LoRA Profile.
9. Export, archive, or clean up large unused outputs.

## Quick Start

Install app dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_app.ps1
```

Start LoRA-Studio:

```bat
start_lora_studio.bat
```

Open:

```text
http://127.0.0.1:8768
```

Set up the verified sd-scripts environment when needed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_sd_scripts.ps1 -ReleaseTag v0.10.5 -CudaProfile cu128 -MixedPrecision bf16
```

## Demo DB / Screenshot Workflow

Create a sanitized demo database for README, documentation, and OSS submission
screenshots:

```powershell
python scripts/create_demo_db.py --output demo/demo.sqlite
```

Start LoRA-Studio against the demo database in read-only demo mode:

```powershell
python start_lora_studio.py --db demo/demo.sqlite --demo --no-browser
```

Open the English UI for screenshots:

```text
http://127.0.0.1:8768/?lang=en
```

Demo mode uses only synthetic project names, datasets, images, reports, and
paths. Training, generation, deletion, and other write actions are blocked.
Generated demo runtime data and screenshots are ignored by Git. If Playwright is
available, screenshots can be captured with:

```powershell
python scripts/capture_demo_screenshots.py --base-url http://127.0.0.1:8768 --db demo/demo.sqlite
```

## Documentation

- [Japanese documentation index](docs/ja/index.md)
- [Japanese README](README_ja.md)
- English detailed documentation will be split under `docs/` over time.

## Current Status

Current release: v0.5.4-beta
Development phase: Phase 12.4.5

The core workflow is operational and used for local LoRA production, but APIs,
screen flows, and recipe catalogs may still change during the beta period.

## Requirements

- Windows local workflow
- Python virtual environment created by `scripts/setup_app.ps1`
- SQLite application database
- kohya-ss/sd-scripts integration, verified against `v0.10.5` for the beta workflow
- NVIDIA GPU environment appropriate for SDXL / SD1.5 LoRA training

## Notes

- Machine Review Assist is advisory. Human visual review remains the final
  decision source for identity, costume details, style, and adoption.
- Smoke Test and Mini Pilot statuses confirm startup or short practical runs;
  they do not guarantee final LoRA quality.
- LoRA-C3Lier is treated as the sd-scripts standard LoRA extension for 3x3
  Conv2d layers via `networks.lora` and `conv_dim` / `conv_alpha`. LyCORIS
  LoCon is a separate future network type.
- Keep large model files, `runs`, `exports`, logs, and embedding caches outside
  OneDrive or other synchronized folders when possible.
- sd-scripts logs, generated commands, raw args, and tracebacks are intentionally
  not translated by the i18n layer.

## License

See [LICENSE](LICENSE).
