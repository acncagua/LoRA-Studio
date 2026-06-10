from __future__ import annotations

from pathlib import Path

DEFAULT_NEGATIVE = "low quality, worst quality, blurry, extra fingers, bad anatomy"


def build_sample_prompts(trigger_word: str, width: int, height: int) -> list[dict[str, object]]:
    trigger = trigger_word.strip() or "trigger_word"
    return [
        {"name": "basic face", "prompt": f"{trigger}, 1girl, portrait, looking at viewer, simple background", "negative_prompt": DEFAULT_NEGATIVE, "width": width, "height": height, "seed": 10101, "cfg_scale": 7.0, "steps": 28},
        {"name": "full body", "prompt": f"{trigger}, 1girl, full body, standing, casual clothes, outdoors", "negative_prompt": DEFAULT_NEGATIVE, "width": width, "height": height, "seed": 20202, "cfg_scale": 7.0, "steps": 28},
        {"name": "expression and pose", "prompt": f"{trigger}, 1girl, smile, dynamic pose, upper body, detailed eyes", "negative_prompt": DEFAULT_NEGATIVE, "width": width, "height": height, "seed": 30303, "cfg_scale": 7.0, "steps": 28},
    ]


def write_sample_prompts(path: Path, prompts: list[dict[str, object]]) -> None:
    lines = []
    for item in prompts:
        lines.append(
            f"{item['prompt']} --n {item['negative_prompt']} --w {item['width']} --h {item['height']} "
            f"--d {item['seed']} --l {item['cfg_scale']} --s {item['steps']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
