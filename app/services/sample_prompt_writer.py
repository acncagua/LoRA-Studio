from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_NEGATIVE = "low quality, worst quality, blurry, extra fingers, bad anatomy"


def build_sample_prompts(trigger_word: str, width: int, height: int) -> list[dict[str, object]]:
    trigger = trigger_word.strip() or "trigger_word"
    return [
        {"name": "basic face", "prompt": f"{trigger}, 1girl, portrait, looking at viewer, simple background", "negative_prompt": DEFAULT_NEGATIVE, "width": width, "height": height, "seed": 10101, "cfg_scale": 7.0, "steps": 28},
        {"name": "full body", "prompt": f"{trigger}, 1girl, full body, standing, casual clothes, outdoors", "negative_prompt": DEFAULT_NEGATIVE, "width": width, "height": height, "seed": 20202, "cfg_scale": 7.0, "steps": 28},
        {"name": "expression and pose", "prompt": f"{trigger}, 1girl, smile, dynamic pose, upper body, detailed eyes", "negative_prompt": DEFAULT_NEGATIVE, "width": width, "height": height, "seed": 30303, "cfg_scale": 7.0, "steps": 28},
    ]


def build_sample_prompts_from_template(template: Any, trigger_word: str, width: int, height: int) -> list[dict[str, object]]:
    if template is None:
        return build_sample_prompts(trigger_word, width, height)
    trigger = trigger_word.strip() or "trigger_word"
    prompts = []
    for item in json.loads(template["prompts_json"]):
        prompts.append(
            {
                "name": item["name"],
                "prompt": item["prompt"].replace("{trigger_word}", trigger),
                "negative_prompt": item.get("negative_prompt") or DEFAULT_NEGATIVE,
                "width": int(item.get("width") or width),
                "height": int(item.get("height") or height),
                "seed": int(item.get("seed") or 1),
                "cfg_scale": float(item.get("cfg_scale") or 7.0),
                "steps": int(item.get("steps") or 28),
            }
        )
    return prompts


def write_sample_prompts(path: Path, prompts: list[dict[str, object]]) -> None:
    lines = []
    for item in prompts:
        lines.append(
            f"{item['prompt']} --n {item['negative_prompt']} --w {item['width']} --h {item['height']} "
            f"--d {item['seed']} --l {item['cfg_scale']} --s {item['steps']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
