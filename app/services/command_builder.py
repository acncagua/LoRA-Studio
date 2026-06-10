from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from app import settings
from app.services.sample_prompt_writer import build_sample_prompts, write_sample_prompts


def write_dataset_config(path: Path, dataset_path: str, repeats: int, class_token: str, resolution: list[int]) -> None:
    width, height = resolution
    normalized_path = dataset_path.replace("\\", "/")
    content = f'''[general]
shuffle_caption = true
caption_extension = ".txt"
keep_tokens = 1

[[datasets]]
resolution = [{width}, {height}]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "{normalized_path}"
  num_repeats = {repeats}
  class_tokens = "{class_token}"
'''
    path.write_text(content, encoding="utf-8")


def prepare_job_files(job: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Path | str]:
    run_dir = Path(job["run_dir"])
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    params = json.loads(job["params_json"])
    resolution = params.get("resolution", [1024, 1024])
    dataset_config = config_dir / "dataset_config.toml"
    sample_prompts = config_dir / "sample_prompts.txt"
    job_config = config_dir / "job_config.json"
    command_txt = config_dir / "command.txt"

    write_dataset_config(dataset_config, dataset["path"], int(params.get("repeats", 10)), dataset.get("class_token") or "person", resolution)
    prompts = build_sample_prompts(dataset.get("trigger_word") or "trigger_word", int(resolution[0]), int(resolution[1]))
    write_sample_prompts(sample_prompts, prompts)

    command = build_command(job, dataset_config, sample_prompts)
    job_config.write_text(json.dumps({"job": job, "dataset": dataset, "params": params}, ensure_ascii=False, indent=2), encoding="utf-8")
    command_txt.write_text(command + "\n", encoding="utf-8")
    return {"dataset_config": dataset_config, "sample_prompts": sample_prompts, "job_config": job_config, "command": command}


def build_command(job: dict[str, Any], dataset_config: Path, sample_prompts: Path) -> str:
    params = json.loads(job["params_json"])
    script = settings.SD_SCRIPTS_DIR / job["training_script"]
    args: list[str] = [
        "accelerate", "launch", str(script),
        "--pretrained_model_name_or_path", job["base_model_path"],
        "--dataset_config", str(dataset_config),
        "--output_dir", job["output_dir"],
        "--output_name", job["output_name"],
        "--sample_prompts", str(sample_prompts),
    ]
    if job.get("vae_path"):
        args.extend(["--vae", job["vae_path"]])

    skip_keys = {"resolution", "repeats", "optimizer_args"}
    for key, value in params.items():
        if key in skip_keys:
            continue
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif isinstance(value, list):
            args.extend([f"--{key}", ",".join(str(part) for part in value)])
        elif value is not None:
            args.extend([f"--{key}", str(value)])

    for key, value in (params.get("optimizer_args") or {}).items():
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        args.extend(["--optimizer_args", f"{key}={rendered}"])

    return " ".join(shlex.quote(str(part)) for part in args)
