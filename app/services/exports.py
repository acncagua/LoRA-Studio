from __future__ import annotations

import html
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import settings
from app.db import fetch_all, fetch_one
from app.services.output_collector import sha256_file


def export_selected_lora(job_id: int) -> dict[str, str]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    if output is None:
        raise ValueError(f"Selected LoRA not found for job {job_id}")

    preset = fetch_one("SELECT * FROM presets WHERE id = ?", (job["preset_id"],))
    summary = fetch_one("SELECT * FROM training_metric_summaries WHERE job_id = ?", (job_id,))
    samples = fetch_all("SELECT * FROM sample_images WHERE job_id = ? ORDER BY epoch, prompt_id, id", (job_id,))
    source = Path(output["file_path"])
    if not source.exists():
        raise ValueError(f"Selected LoRA file not found: {source}")

    export_dir = settings.EXPORTS_DIR / "selected_loras" / f"job_{job_id:06d}"
    export_dir.mkdir(parents=True, exist_ok=True)
    exported_model = export_dir / source.name
    shutil.copy2(source, exported_model)
    sha256 = sha256_file(exported_model)
    selected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    info = {
        "job_id": job_id,
        "job_name": job["name"],
        "dataset_id": job["dataset_id"],
        "dataset_version_id": job["dataset_version_id"],
        "trigger_word_at_creation": job["trigger_word_at_creation"],
        "preset_id": job["preset_id"],
        "preset_name": preset["name"] if preset else None,
        "params_json": json.loads(job["params_json"]),
        "selected_epoch": job["adopted_epoch"],
        "selected_model_path_original": str(source),
        "selected_model_path_exported": str(exported_model),
        "file_size": exported_model.stat().st_size,
        "sha256": sha256,
        "health_label": summary["health_label"] if summary else None,
        "health_message": summary["health_message"] if summary else None,
        "selected_at": selected_at,
        "memo": human_memo(samples),
    }
    (export_dir / "selected_lora_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (export_dir / "selected_lora_notes.md").write_text(selected_lora_notes(job, preset, summary, info), encoding="utf-8")
    return {"directory": str(export_dir), "model": str(exported_model), "info": str(export_dir / "selected_lora_info.json")}


def write_job_contact_sheet(job_id: int) -> str:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
    preset = fetch_one("SELECT * FROM presets WHERE id = ?", (job["preset_id"],))
    summary = fetch_one("SELECT * FROM training_metric_summaries WHERE job_id = ?", (job_id,))
    epochs = fetch_all("SELECT * FROM training_epoch_summaries WHERE job_id = ? ORDER BY epoch", (job_id,))
    outputs = fetch_all("SELECT * FROM training_outputs WHERE job_id = ? ORDER BY epoch, step, id", (job_id,))
    prompts = fetch_all("SELECT * FROM sample_prompts WHERE job_id = ? ORDER BY sort_order, id", (job_id,))
    samples = fetch_all("SELECT * FROM sample_images WHERE job_id = ? ORDER BY prompt_id, epoch, id", (job_id,))

    reports_dir = Path(job["run_dir"]) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"contact_sheet_job_{job_id:06d}.html"
    output_by_epoch = {row["epoch"]: row for row in outputs if row["epoch"] is not None}
    lines = html_document_start(f"Contact Sheet Job #{job_id}")
    lines.extend(
        [
            f"<h1>Job #{job_id} {e(job['name'])}</h1>",
            "<section><h2>Summary</h2><dl>",
            f"<dt>Preset</dt><dd>{e(preset['name'] if preset else '-')}</dd>",
            f"<dt>Dataset</dt><dd>#{job['dataset_id']} {e(dataset['name'] if dataset else '-')}</dd>",
            f"<dt>Dataset Version</dt><dd>{job['dataset_version_id'] or '-'}</dd>",
            f"<dt>Trigger</dt><dd>{e(job['trigger_word_at_creation'] or '-')}</dd>",
            f"<dt>Selected LoRA</dt><dd>{e(job['adopted_model_path'] or '-')}</dd>",
            f"<dt>Health</dt><dd>{e(summary['health_label'] if summary else '-')} / {e(summary['health_message'] if summary else '-')}</dd>",
            "</dl></section>",
            "<section><h2>Epoch Loss Summary</h2><table><thead><tr><th>Epoch</th><th>Avg Loss</th><th>Moving Avg</th><th>Output</th><th>Selected</th></tr></thead><tbody>",
        ]
    )
    for epoch in epochs:
        output = output_by_epoch.get(epoch["epoch"])
        lines.append(
            "<tr>"
            f"<td>{epoch['epoch']}</td><td>{fmt(epoch['avg_loss'])}</td><td>{fmt(epoch['moving_avg_final_loss'])}</td>"
            f"<td>{e(Path(output['file_path']).name if output else '-')}</td><td>{'yes' if output and output['selected'] else ''}</td>"
            "</tr>"
        )
    lines.append("</tbody></table></section>")
    lines.extend(sample_sections(samples, prompts, path.parent))
    lines.extend(html_document_end())
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_compare_contact_sheet(comparison: dict[str, Any]) -> str:
    left = comparison["left"]
    right = comparison["right"]
    left_id = int(left["job"]["id"])
    right_id = int(right["job"]["id"])
    output_dir = settings.RUNS_DIR / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"contact_sheet_compare_job_{left_id:06d}_job_{right_id:06d}.html"
    lines = html_document_start(f"Compare Contact Sheet Job #{left_id} vs #{right_id}")
    lines.extend(
        [
            f"<h1>Compare Job #{left_id} vs Job #{right_id}</h1>",
            "<section><h2>Jobs</h2><div class=\"columns\">",
            job_card(left),
            job_card(right),
            "</div></section>",
            "<section><h2>Parameter Differences</h2><table><thead><tr><th>Param</th><th>Left</th><th>Right</th></tr></thead><tbody>",
        ]
    )
    for row in comparison["param_rows"]:
        css = " class=\"changed\"" if row["changed"] else ""
        lines.append(f"<tr{css}><td>{e(row['key'])}</td><td>{e(row['left'])}</td><td>{e(row['right'])}</td></tr>")
    lines.append("</tbody></table></section>")
    if comparison["warnings"]:
        lines.append("<section><h2>Warnings</h2>")
        lines.extend(f"<p class=\"notice\">{e(warning)}</p>" for warning in comparison["warnings"])
        lines.append("</section>")
    lines.append("<section><h2>Epoch Loss Comparison</h2><table><thead><tr><th>Epoch</th><th>Left Avg</th><th>Right Avg</th><th>Left MA</th><th>Right MA</th><th>Left Samples</th><th>Right Samples</th></tr></thead><tbody>")
    for row in comparison["epoch_rows"]:
        lines.append(f"<tr><td>{row['epoch']}</td><td>{e(row['left_avg_loss'])}</td><td>{e(row['right_avg_loss'])}</td><td>{e(row['left_ma_final'])}</td><td>{e(row['right_ma_final'])}</td><td>{row['left_samples']}</td><td>{row['right_samples']}</td></tr>")
    lines.append("</tbody></table></section>")
    lines.append("<section><h2>Samples</h2>")
    for group in comparison["sample_groups"]:
        lines.append(f"<h3>{e(group['title'])}</h3><table><thead><tr><th>Epoch/Step</th><th>Left</th><th>Right</th></tr></thead><tbody>")
        for row in group["rows"]:
            lines.append("<tr>")
            lines.append(f"<td>{e(row['label'])}</td>")
            lines.append(sample_cell(row["left"], path.parent))
            lines.append(sample_cell(row["right"], path.parent))
            lines.append("</tr>")
        lines.append("</tbody></table>")
    lines.append("</section>")
    lines.extend(html_document_end())
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def sample_sections(samples: list[Any], prompts: list[Any], base_dir: Path) -> list[str]:
    prompt_map = {row["id"]: row for row in prompts}
    grouped: dict[int, list[Any]] = {}
    for sample in samples:
        grouped.setdefault(sample["prompt_id"] or 0, []).append(sample)
    lines = ["<section><h2>Samples By Prompt</h2>"]
    for prompt_id, rows in grouped.items():
        prompt = prompt_map.get(prompt_id)
        title = prompt["name"] if prompt else "Unmatched prompt"
        lines.append(f"<h3>{e(title)}</h3><div class=\"grid\">")
        for sample in sorted(rows, key=lambda item: (item["epoch"] or 999999, item["step"] or 999999, item["id"])):
            lines.append(sample_figure(sample, base_dir))
        lines.append("</div>")
    lines.append("</section>")
    return lines


def sample_cell(sample: Any, base_dir: Path) -> str:
    if not sample:
        return "<td class=\"muted\">No image</td>"
    return f"<td>{sample_figure(sample, base_dir)}</td>"


def sample_figure(sample: Any, base_dir: Path) -> str:
    image_path = Path(sample["image_path"])
    src = os.path.relpath(image_path, base_dir).replace("\\", "/")
    return (
        "<figure>"
        f"<img src=\"{e(src)}\" alt=\"sample {sample['id']}\">"
        f"<figcaption>{e(image_path.name)}<br>epoch {sample['epoch'] or '-'} / step {sample['step'] or '-'}<br>"
        f"face {sample_value(sample, 'rating_face')} costume {sample_value(sample, 'rating_costume')} "
        f"style {sample_value(sample, 'rating_style')} stability {sample_value(sample, 'rating_stability')} "
        f"overall {sample_value(sample, 'rating_overall')}<br>{e(sample['memo'] or '')}</figcaption>"
        "</figure>"
    )


def job_card(bundle: dict[str, Any]) -> str:
    job = bundle["job"]
    preset = bundle["preset"]
    return (
        "<div class=\"card\"><dl>"
        f"<dt>Job</dt><dd>#{job['id']} {e(job['name'])}</dd>"
        f"<dt>Preset</dt><dd>{e(preset['name'] if preset else '-')}</dd>"
        f"<dt>Dataset Version</dt><dd>{job['dataset_version_id'] or '-'}</dd>"
        f"<dt>Status</dt><dd>{e(job['status'])}</dd>"
        f"<dt>Selected LoRA</dt><dd>{e(job['adopted_model_path'] or '-')}</dd>"
        "</dl></div>"
    )


def selected_lora_notes(job: Any, preset: Any, summary: Any, info: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# Selected LoRA Job #{info['job_id']}",
            "",
            "## 概要",
            f"- Job: {job['name']}",
            f"- Preset: {preset['name'] if preset else '-'}",
            f"- Exported: {info['selected_model_path_exported']}",
            "",
            "## 推奨trigger word",
            f"- {job['trigger_word_at_creation'] or '-'}",
            "",
            "## 使用Dataset version",
            f"- Dataset #{job['dataset_id']} / version {job['dataset_version_id'] or '-'}",
            "",
            "## 選択epoch",
            f"- {job['adopted_epoch'] or '-'}",
            "",
            "## loss summary",
            f"- health: {summary['health_label'] if summary else '-'}",
            f"- message: {summary['health_message'] if summary else '-'}",
            f"- final_loss: {summary['final_loss'] if summary else '-'}",
            f"- moving_avg_final_loss: {summary['moving_avg_final_loss'] if summary else '-'}",
            "",
            "## 人間評価メモ",
            info["memo"] or "-",
            "",
            "## 注意点",
            "- no_metadataを使ったLoRAはLoRA本体として利用可能です。",
            "- 学習条件の確認にはLoRA-StudioのDBとselected_lora_info.jsonを併用してください。",
            "",
        ]
    )


def human_memo(samples: list[Any]) -> str:
    lines = []
    for sample in samples:
        if sample["memo"]:
            lines.append(f"- {Path(sample['image_path']).name}: {sample['memo']}")
    return "\n".join(lines)


def sample_value(sample: Any, key: str) -> int:
    value = sample[key] if key in sample.keys() else None
    if value is None and key == "rating_overall":
        value = sample["rating"] if "rating" in sample.keys() else None
    return int(value or 0)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def html_document_start(title: str) -> list[str]:
    return [
        "<!doctype html>",
        "<html lang=\"ja\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{e(title)}</title>",
        "<style>",
        "body{font-family:Segoe UI,Yu Gothic UI,sans-serif;margin:24px;color:#20231f;background:#f6f7f4}",
        "section{margin:24px 0}table{width:100%;border-collapse:collapse;background:#fff}th,td{border:1px solid #d8ddd4;padding:8px;vertical-align:top}th{background:#eef2eb}",
        "dl{display:grid;grid-template-columns:160px 1fr;gap:6px 12px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{background:#fff;border:1px solid #d8ddd4;padding:12px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}figure{margin:0;background:#fff;border:1px solid #d8ddd4;padding:8px}img{max-width:100%;height:auto;display:block}figcaption{font-size:12px;color:#657064;word-break:break-word}.changed td{background:#fff8e8}.notice{background:#fff8e8;padding:8px}",
        "</style></head><body>",
    ]


def html_document_end() -> list[str]:
    return ["</body></html>"]
