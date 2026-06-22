from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from app.db import fetch_one, latest_environment, replace_sample_prompts
from app.services.sample_prompt_writer import build_sample_prompts_from_template, write_sample_prompts

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def write_dataset_config(
    path: Path,
    dataset_path: str,
    repeats: int,
    class_token: str,
    resolution: list[int],
    batch_size: int,
    shuffle_caption: bool,
) -> None:
    width, height = resolution
    subsets = []
    for image_dir in image_directories(Path(dataset_path)):
        normalized_path = str(image_dir).replace("\\", "/")
        subsets.append(
            f'''
  [[datasets.subsets]]
  image_dir = "{normalized_path}"
  num_repeats = {repeats}
  class_tokens = "{class_token}"
'''
        )
    if not subsets:
        normalized_path = dataset_path.replace("\\", "/")
        subsets.append(
            f'''
  [[datasets.subsets]]
  image_dir = "{normalized_path}"
  num_repeats = {repeats}
  class_tokens = "{class_token}"
'''
        )
    content = f'''[general]
shuffle_caption = {str(shuffle_caption).lower()}
caption_extension = ".txt"
keep_tokens = 1

[[datasets]]
resolution = [{width}, {height}]
batch_size = {batch_size}
{''.join(subsets)}'''
    path.write_text(content, encoding="utf-8")


def normalize_resolution(value: Any) -> list[int]:
    if isinstance(value, str):
        cleaned = value.strip().strip("[]()")
        parts = [part.strip() for part in cleaned.replace("x", ",").split(",") if part.strip()]
        if len(parts) == 2:
            return [int(parts[0]), int(parts[1])]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [int(value[0]), int(value[1])]
    return [1024, 1024]


def image_directories(dataset_path: Path) -> list[Path]:
    if not dataset_path.exists():
        return []
    directories = {
        image.parent
        for image in dataset_path.rglob("*")
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    }
    return sorted(directories)


def prepare_job_files(job: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Path | str]:
    run_dir = Path(job["run_dir"])
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    params = json.loads(job["params_json"])
    resolution = normalize_resolution(params.get("resolution", [1024, 1024]))
    dataset_config = config_dir / "dataset_config.toml"
    sample_prompts = config_dir / "sample_prompts.txt"
    job_config = config_dir / "job_config.json"
    command_txt = config_dir / "command.txt"
    command_argv_json = config_dir / "command_argv.json"

    batch_size = int(params.get("train_batch_size", 1))
    write_dataset_config(
        dataset_config,
        dataset["path"],
        int(params.get("repeats", 10)),
        dataset.get("class_token") or "person",
        resolution,
        batch_size,
        not bool(params.get("cache_text_encoder_outputs")),
    )
    template = None
    if job.get("sample_prompt_template_id"):
        template = fetch_one("SELECT * FROM sample_prompt_templates WHERE id = ?", (job["sample_prompt_template_id"],))
    trigger_word = job.get("trigger_word_at_creation") or dataset.get("trigger_word") or "trigger_word"
    prompts = build_sample_prompts_from_template(template, trigger_word, int(resolution[0]), int(resolution[1]))
    write_sample_prompts(sample_prompts, prompts)
    replace_sample_prompts(int(job["id"]), prompts)

    command_argv = build_command_argv(job, dataset_config, sample_prompts)
    command = " ".join(shlex.quote(str(part)) for part in command_argv)
    job_config.write_text(json.dumps({"job": job, "dataset": dataset, "params": params}, ensure_ascii=False, indent=2), encoding="utf-8")
    command_txt.write_text(command + "\n", encoding="utf-8")
    command_argv_json.write_text(json.dumps(command_argv, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dataset_config": dataset_config,
        "sample_prompts": sample_prompts,
        "job_config": job_config,
        "command_argv": command_argv_json,
        "command": command,
    }


def build_command_argv(job: dict[str, Any], dataset_config: Path, sample_prompts: Path) -> list[str]:
    params = json.loads(job["params_json"])
    environment = latest_environment()
    if environment is None:
        raise RuntimeError("sd-scripts environment is not registered.")
    script = Path(environment["sd_scripts_path"]) / job["training_script"]
    args: list[str] = [
        environment["venv_python_path"],
        "-m",
        "accelerate.commands.launch",
        str(script),
        "--pretrained_model_name_or_path", job["base_model_path"],
        "--dataset_config", str(dataset_config),
        "--output_dir", job["output_dir"],
        "--output_name", job["output_name"],
        "--sample_prompts", str(sample_prompts),
        "--logging_dir", str(Path(job["run_dir"]) / "metrics"),
        "--log_with", "tensorboard",
    ]
    if job.get("vae_path"):
        args.extend(["--vae", job["vae_path"]])

    skip_keys = {"resolution", "repeats", "optimizer_args", "train_batch_size", "text_encoder_lr1", "text_encoder_lr2"}
    text_encoder_lr_values = [params.get("text_encoder_lr1"), params.get("text_encoder_lr2")]
    active_text_encoder_lrs = [
        value
        for value in text_encoder_lr_values
        if value not in (None, "", 0, 0.0, "0", "0.0")
    ]
    if active_text_encoder_lrs:
        args.extend(
            [
                "--text_encoder_lr",
                str(params.get("text_encoder_lr1", 0)),
                str(params.get("text_encoder_lr2", 0)),
            ]
        )
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

    return [str(part) for part in args]
