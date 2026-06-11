from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app import settings
from app.db import connect, create_job, fetch_all, fetch_one, import_latest_environment, init_db, insert_dataset, upsert_dataset_analysis
from app.services.command_builder import prepare_job_files
from app.services.output_collector import collect_job_results
from app.services.training_runner import read_log_tail, start_job, stop_job

app = FastAPI(title=settings.APP_NAME)
app.mount("/static", StaticFiles(directory=settings.ROOT_DIR / "app" / "static"), name="static")

templates = Environment(
    loader=FileSystemLoader(settings.ROOT_DIR / "app" / "templates"),
    autoescape=select_autoescape(["html", "xml"]),
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def render(request: Request, template: str, **context: Any) -> HTMLResponse:
    tpl = templates.get_template(template)
    context.setdefault("app_name", settings.APP_NAME)
    context.setdefault("request", request)
    context.setdefault("sd_scripts_release_tag", settings.SD_SCRIPTS_RELEASE_TAG)
    context.setdefault("sd_scripts_release_commit", settings.SD_SCRIPTS_RELEASE_COMMIT)
    return HTMLResponse(tpl.render(**context))


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    stats = {
        "presets": fetch_one("SELECT COUNT(*) AS count FROM presets")["count"],
        "datasets": fetch_one("SELECT COUNT(*) AS count FROM datasets")["count"],
        "jobs": fetch_one("SELECT COUNT(*) AS count FROM training_jobs")["count"],
        "running": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'running'")["count"],
        "completed": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'completed'")["count"],
        "failed": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'failed'")["count"],
        "stopped": fetch_one("SELECT COUNT(*) AS count FROM training_jobs WHERE status = 'stopped'")["count"],
    }
    jobs = fetch_all("SELECT * FROM training_jobs ORDER BY id DESC LIMIT 8")
    return render(request, "dashboard.html", stats=stats, jobs=jobs)


@app.get("/environment", response_class=HTMLResponse)
def environment(request: Request) -> HTMLResponse:
    import_latest_environment()
    settings_rows = fetch_all("SELECT * FROM app_settings ORDER BY key")
    environments = fetch_all("SELECT * FROM environments ORDER BY id DESC")
    return render(request, "environment.html", settings_rows=settings_rows, environments=environments, settings=settings)


@app.get("/presets", response_class=HTMLResponse)
def presets(request: Request) -> HTMLResponse:
    rows = fetch_all("SELECT * FROM presets ORDER BY model_family DESC, name")
    return render(request, "presets.html", presets=rows)


@app.get("/presets/{preset_id}", response_class=HTMLResponse)
def preset_detail(request: Request, preset_id: str) -> HTMLResponse:
    preset = fetch_one("SELECT * FROM presets WHERE id = ?", (preset_id,))
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    return render(request, "preset_detail.html", preset=preset)


@app.get("/datasets", response_class=HTMLResponse)
def datasets(request: Request) -> HTMLResponse:
    rows = fetch_all("SELECT * FROM datasets ORDER BY id DESC")
    return render(request, "datasets.html", datasets=rows)


@app.post("/datasets")
def datasets_create(name: str = Form(...), path: str = Form(...), model_family: str = Form("SDXL"), trigger_word: str = Form(""), class_token: str = Form("person"), memo: str = Form("")) -> RedirectResponse:
    insert_dataset(name, path, model_family, trigger_word, class_token, memo)
    return RedirectResponse("/datasets", status_code=303)


@app.get("/datasets/{dataset_id}", response_class=HTMLResponse)
def dataset_detail(request: Request, dataset_id: int) -> HTMLResponse:
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    analysis = fetch_one("SELECT * FROM dataset_analysis WHERE dataset_id = ?", (dataset_id,))
    if analysis is None:
        rescan_dataset(dataset_id)
        dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        analysis = fetch_one("SELECT * FROM dataset_analysis WHERE dataset_id = ?", (dataset_id,))
    return render(request, "dataset_detail.html", dataset=dataset, analysis=decode_analysis(analysis))


@app.post("/datasets/{dataset_id}/rescan")
def dataset_rescan(dataset_id: int) -> RedirectResponse:
    rescan_dataset(dataset_id)
    return RedirectResponse(f"/datasets/{dataset_id}", status_code=303)


@app.get("/sample-prompt-templates", response_class=HTMLResponse)
def sample_prompt_templates(request: Request) -> HTMLResponse:
    templates_rows = fetch_all("SELECT * FROM sample_prompt_templates ORDER BY is_builtin DESC, name")
    return render(request, "sample_prompt_templates.html", templates=templates_rows)


def rescan_dataset(dataset_id: int) -> None:
    from app.services.dataset_scanner import scan_dataset

    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    scan = scan_dataset(Path(dataset["path"]), dataset["trigger_word"] or "")
    now = settings_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE datasets
            SET image_count = ?, caption_count = ?, missing_caption_count = ?,
                resolution_summary_json = ?, tag_summary_json = ?,
                scan_status = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                scan["image_count"],
                scan["caption_count"],
                scan["missing_caption_count"],
                json.dumps(scan.get("resolution_summary") or {}, ensure_ascii=False),
                json.dumps(scan.get("tag_summary") or {}, ensure_ascii=False),
                scan["status"],
                now,
                dataset_id,
            ),
        )
        upsert_dataset_analysis(conn, dataset_id, scan)


def settings_now() -> str:
    from app.db import utc_now

    return utc_now()


def decode_analysis(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    decoded = dict(row)
    for key in (
        "caption_encoding_summary_json",
        "image_size_summary_json",
        "tag_summary_json",
        "missing_caption_images_json",
        "caption_without_images_json",
        "broken_images_json",
        "unsupported_files_json",
        "analysis_json",
    ):
        value = decoded.get(key)
        decoded[key.removesuffix("_json")] = json.loads(value) if value else {} if key.endswith("summary_json") or key == "analysis_json" else []
    return decoded


@app.get("/jobs/new", response_class=HTMLResponse)
def job_new(request: Request) -> HTMLResponse:
    datasets = fetch_all("SELECT * FROM datasets ORDER BY id DESC")
    presets = fetch_all("SELECT * FROM presets ORDER BY model_family DESC, name")
    sample_prompt_templates = fetch_all("SELECT * FROM sample_prompt_templates ORDER BY is_builtin DESC, name")
    return render(request, "job_create.html", datasets=datasets, presets=presets, sample_prompt_templates=sample_prompt_templates)


@app.post("/jobs")
def job_create(name: str = Form(...), dataset_id: int = Form(...), preset_id: str = Form(...), base_model_path: str = Form(...), vae_path: str = Form(""), output_name: str = Form(""), memo: str = Form(""), sample_prompt_template_id: str = Form("")) -> RedirectResponse:
    job_id = create_job({"name": name, "dataset_id": dataset_id, "preset_id": preset_id, "base_model_path": base_model_path, "vae_path": vae_path, "output_name": output_name, "memo": memo, "sample_prompt_template_id": sample_prompt_template_id})
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: int) -> HTMLResponse:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
    outputs = fetch_all("SELECT * FROM training_outputs WHERE job_id = ? ORDER BY epoch, step, id", (job_id,))
    samples = fetch_all("SELECT * FROM sample_images WHERE job_id = ? ORDER BY epoch, id", (job_id,))
    sample_prompts = fetch_all("SELECT * FROM sample_prompts WHERE job_id = ? ORDER BY sort_order, id", (job_id,))
    metrics = fetch_all("SELECT * FROM training_metrics WHERE job_id = ? ORDER BY step, id", (job_id,))
    metric_summary = fetch_one("SELECT * FROM training_metric_summaries WHERE job_id = ?", (job_id,))
    log_tail = read_log_tail(dict(job))
    selected_output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    return render(
        request,
        "job_detail.html",
        job=job,
        dataset=dataset,
        outputs=outputs,
        samples=samples,
        sample_prompts=sample_prompts,
        sample_groups=group_samples(sample_prompts, samples),
        metrics=metrics,
        metric_summary=metric_summary,
        health_details=health_details(metric_summary, len(metrics)),
        loss_chart=build_loss_chart(metrics),
        log_tail=log_tail,
        selected_output=selected_output,
    )


@app.post("/jobs/{job_id}/prepare")
def job_prepare(job_id: int) -> RedirectResponse:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
    if dataset is None:
        raise HTTPException(status_code=400, detail="Dataset not found")
    files = prepare_job_files(dict(job), dict(dataset))
    with connect() as conn:
        conn.execute("UPDATE training_jobs SET command_line = ?, status = 'prepared', updated_at = datetime('now') WHERE id = ?", (files["command"], job_id))
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/run")
def job_run(job_id: int) -> RedirectResponse:
    try:
        start_job(job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/stop")
def job_stop(job_id: int) -> RedirectResponse:
    try:
        stop_job(job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/reimport")
def job_reimport(job_id: int) -> RedirectResponse:
    try:
        collect_job_results(job_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/clone")
def job_clone(job_id: int, name: str = Form("")) -> RedirectResponse:
    source = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if source is None:
        raise HTTPException(status_code=404, detail="Job not found")
    clone_name = name.strip() or f"{source['name']}_clone"
    new_id = create_job(
        {
            "name": clone_name,
            "dataset_id": source["dataset_id"],
            "preset_id": source["preset_id"],
            "base_model_path": source["base_model_path"],
            "vae_path": source["vae_path"] or "",
            "output_name": f"{source['output_name']}_clone",
            "memo": f"Cloned from Job #{job_id}",
            "params": json.loads(source["params_json"]),
            "parent_job_id": job_id,
            "sample_prompt_template_id": source["sample_prompt_template_id"] or "",
        }
    )
    copy_sample_prompts(job_id, new_id)
    return RedirectResponse(f"/jobs/{new_id}", status_code=303)


@app.post("/jobs/{job_id}/variant")
def job_variant(job_id: int, variant: str = Form(...)) -> RedirectResponse:
    source = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if source is None:
        raise HTTPException(status_code=404, detail="Job not found")
    params = json.loads(source["params_json"])
    label = apply_variant(params, variant)
    new_id = create_job(
        {
            "name": f"{source['name']}_{variant}",
            "dataset_id": source["dataset_id"],
            "preset_id": source["preset_id"],
            "base_model_path": source["base_model_path"],
            "vae_path": source["vae_path"] or "",
            "output_name": f"{source['output_name']}_{variant}",
            "memo": f"Quick Variant from Job #{job_id}: {label}",
            "params": params,
            "parent_job_id": job_id,
            "sample_prompt_template_id": source["sample_prompt_template_id"] or "",
        }
    )
    copy_sample_prompts(job_id, new_id)
    return RedirectResponse(f"/jobs/{new_id}", status_code=303)


@app.post("/jobs/{job_id}/outputs/{output_id}/select")
def job_select_output(job_id: int, output_id: int) -> RedirectResponse:
    output = fetch_one(
        "SELECT * FROM training_outputs WHERE id = ? AND job_id = ? AND file_type = 'model'",
        (output_id, job_id),
    )
    if output is None:
        raise HTTPException(status_code=404, detail="Output not found")
    with connect() as conn:
        conn.execute("UPDATE training_outputs SET selected = 0 WHERE job_id = ?", (job_id,))
        conn.execute("UPDATE training_outputs SET selected = 1 WHERE id = ?", (output_id,))
        conn.execute(
            """
            UPDATE training_jobs
            SET adopted_epoch = ?, adopted_model_path = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (output["epoch"], output["file_path"], job_id),
        )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/samples/{image_id}/review")
def job_review_sample(job_id: int, image_id: int, rating: int = Form(0), memo: str = Form("")) -> RedirectResponse:
    sample = fetch_one("SELECT * FROM sample_images WHERE id = ? AND job_id = ?", (image_id, job_id))
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample image not found")
    rating = max(0, min(5, int(rating)))
    with connect() as conn:
        conn.execute(
            "UPDATE sample_images SET rating = ?, memo = ? WHERE id = ? AND job_id = ?",
            (rating, memo, image_id, job_id),
        )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}/samples/{image_id}")
def job_sample_image(job_id: int, image_id: int) -> FileResponse:
    image = fetch_one("SELECT * FROM sample_images WHERE id = ? AND job_id = ?", (image_id, job_id))
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if image is None or job is None:
        raise HTTPException(status_code=404, detail="Sample image not found")
    image_path = Path(image["image_path"]).resolve()
    samples_dir = (Path(job["run_dir"]) / "samples").resolve()
    if samples_dir not in image_path.parents and image_path != samples_dir:
        raise HTTPException(status_code=403, detail="Sample image path is not allowed")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Sample image file not found")
    return FileResponse(image_path)


def copy_sample_prompts(source_job_id: int, target_job_id: int) -> None:
    rows = fetch_all("SELECT * FROM sample_prompts WHERE job_id = ? ORDER BY sort_order, id", (source_job_id,))
    if not rows:
        return
    now = settings_now()
    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO sample_prompts(
                job_id, name, prompt, negative_prompt, width, height,
                seed, cfg_scale, steps, sort_order, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    target_job_id,
                    row["name"],
                    row["prompt"],
                    row["negative_prompt"],
                    row["width"],
                    row["height"],
                    row["seed"],
                    row["cfg_scale"],
                    row["steps"],
                    row["sort_order"],
                    now,
                )
                for row in rows
            ],
        )


def apply_variant(params: dict[str, Any], variant: str) -> str:
    if variant == "lower_lr":
        params["learning_rate"] = halve_float(params.get("learning_rate"))
        params["unet_lr"] = halve_float(params.get("unet_lr"))
        return "Lower LR"
    if variant == "higher_lr":
        params["learning_rate"] = min(0.0002, multiply_float(params.get("learning_rate"), 1.5))
        params["unet_lr"] = min(0.0002, multiply_float(params.get("unet_lr"), 1.5))
        return "Higher LR"
    if variant == "lower_dim":
        params["network_dim"] = max(1, int(params.get("network_dim") or 1) // 2)
        params["network_alpha"] = max(1, int(params.get("network_alpha") or 1) // 2)
        return "Lower Dim"
    if variant == "higher_dim":
        params["network_dim"] = int(params.get("network_dim") or 1) * 2
        params["network_alpha"] = int(params.get("network_alpha") or 1) * 2
        return "Higher Dim"
    if variant == "more_epoch":
        params["max_train_epochs"] = int(params.get("max_train_epochs") or 1) + 2
        return "More Epoch"
    if variant == "fewer_epoch":
        params["max_train_epochs"] = max(1, int(params.get("max_train_epochs") or 1) - 1)
        return "Fewer Epoch"
    raise HTTPException(status_code=400, detail=f"Unknown variant: {variant}")


def halve_float(value: Any) -> float:
    return multiply_float(value, 0.5)


def multiply_float(value: Any, factor: float) -> float:
    return float(value or 0) * factor


COMPARE_PARAM_KEYS = [
    "optimizer_type",
    "lr_scheduler",
    "learning_rate",
    "unet_lr",
    "text_encoder_lr",
    "text_encoder_lr1",
    "text_encoder_lr2",
    "network_dim",
    "network_alpha",
    "train_batch_size",
    "repeats",
    "max_train_epochs",
    "resolution",
    "save_every_n_epochs",
    "sample_every_n_epochs",
    "save_every_n_steps",
    "sample_every_n_steps",
]

COMPARE_METRIC_KEYS = [
    "expected_total_steps",
    "actual_max_step",
    "initial_loss",
    "final_loss",
    "min_loss",
    "loss_drop_rate",
    "loss_volatility",
    "spike_count",
    "late_stage_slope",
    "health_label",
    "health_message",
    "step_consistency_label",
]


@app.get("/compare", response_class=HTMLResponse)
def compare_epochs(
    request: Request,
    job_a: int | None = None,
    job_b: int | None = None,
    job_ids: list[int] | None = Query(None),
    exported: str | None = None,
) -> HTMLResponse:
    if job_ids and len(job_ids) >= 2:
        job_a, job_b = job_ids[0], job_ids[1]
    jobs = fetch_all("SELECT id, name, status, adopted_epoch FROM training_jobs ORDER BY id DESC")
    if not job_a or not job_b:
        return render(request, "compare_epochs.html", jobs=jobs, comparison=None, exported=exported)
    comparison = build_job_comparison(job_a, job_b)
    return render(request, "compare_epochs.html", jobs=jobs, comparison=comparison, exported=exported)


@app.post("/compare/export")
def export_comparison(job_a: int = Form(...), job_b: int = Form(...)) -> RedirectResponse:
    comparison = build_job_comparison(job_a, job_b)
    path = write_comparison_markdown(comparison)
    return RedirectResponse(f"/compare?job_a={job_a}&job_b={job_b}&exported={path}", status_code=303)


def build_job_comparison(job_a: int, job_b: int) -> dict[str, Any]:
    left = load_compare_job(job_a)
    right = load_compare_job(job_b)
    return {
        "left": left,
        "right": right,
        "param_rows": build_param_rows(left["params"], right["params"]),
        "metric_rows": build_metric_rows(left, right),
        "sample_groups": build_compare_sample_groups(left, right),
    }


def load_compare_job(job_id: int) -> dict[str, Any]:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
    preset = fetch_one("SELECT * FROM presets WHERE id = ?", (job["preset_id"],))
    summary = fetch_one("SELECT * FROM training_metric_summaries WHERE job_id = ?", (job_id,))
    metrics = fetch_all("SELECT * FROM training_metrics WHERE job_id = ? ORDER BY step, id", (job_id,))
    outputs = fetch_all("SELECT * FROM training_outputs WHERE job_id = ? ORDER BY epoch, step, id", (job_id,))
    samples = fetch_all("SELECT * FROM sample_images WHERE job_id = ? ORDER BY epoch, step, id", (job_id,))
    sample_prompts = fetch_all("SELECT * FROM sample_prompts WHERE job_id = ? ORDER BY sort_order, id", (job_id,))
    selected_output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1", (job_id,))
    params = json.loads(job["params_json"])
    return {
        "job": job,
        "dataset": dataset,
        "preset": preset,
        "summary": summary,
        "metrics": metrics,
        "outputs": outputs,
        "samples": samples,
        "sample_prompts": sample_prompts,
        "selected_output": selected_output,
        "params": params,
        "loss_chart": build_loss_chart(metrics),
        "health_details": health_details(summary, len(metrics)),
    }


def build_param_rows(left_params: dict[str, Any], right_params: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    emphasized = {"learning_rate", "unet_lr", "text_encoder_lr", "text_encoder_lr1", "text_encoder_lr2", "network_dim", "network_alpha", "max_train_epochs", "repeats"}
    for key in COMPARE_PARAM_KEYS:
        left_value = left_params.get(key)
        right_value = right_params.get(key)
        rows.append({"key": key, "left": render_value(left_value), "right": render_value(right_value), "changed": left_value != right_value, "emphasized": key in emphasized})
    return rows


def build_metric_rows(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in COMPARE_METRIC_KEYS:
        left_value = metric_value(left, key)
        right_value = metric_value(right, key)
        rows.append({"key": key, "left": render_value(left_value), "right": render_value(right_value), "changed": left_value != right_value})
    return rows


def metric_value(bundle: dict[str, Any], key: str) -> Any:
    if key in {"expected_total_steps", "actual_max_step", "step_consistency_label"}:
        return bundle["job"][key]
    summary = bundle["summary"]
    return summary[key] if summary and key in summary.keys() else None


def render_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def health_details(summary: Any, metric_count: int) -> dict[str, Any]:
    threshold = max(2, metric_count // 3) if metric_count else None
    return {
        "spike_threshold": threshold,
        "spike_rule": "current loss > previous loss * 1.5",
        "quality_note": "Loss health is a training-log health check, not an image quality score.",
        "adoption_note": "WARNING can still be usable when sample images look better.",
        "spike_count": summary["spike_count"] if summary else None,
        "loss_volatility": summary["loss_volatility"] if summary else None,
        "late_stage_slope": summary["late_stage_slope"] if summary else None,
        "min_loss_step": summary["min_loss_step"] if summary else None,
        "final_loss": summary["final_loss"] if summary else None,
        "health_message": summary["health_message"] if summary else None,
    }


def build_compare_sample_groups(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_prompts = {row["sort_order"]: row for row in left["sample_prompts"]}
    right_prompts = {row["sort_order"]: row for row in right["sample_prompts"]}
    groups = []
    for order in sorted(set(left_prompts) | set(right_prompts)):
        left_prompt = left_prompts.get(order)
        right_prompt = right_prompts.get(order)
        prompt = left_prompt or right_prompt
        left_samples = samples_for_prompt(left["samples"], left_prompt)
        right_samples = samples_for_prompt(right["samples"], right_prompt)
        positions = sorted(set(left_samples) | set(right_samples))
        groups.append(
            {
                "title": prompt["name"] if prompt else f"Prompt {order}",
                "prompt": prompt["prompt"] if prompt else "",
                "rows": [
                    {
                        "label": f"epoch {position}" if isinstance(position, int) else str(position),
                        "left": left_samples.get(position),
                        "right": right_samples.get(position),
                    }
                    for position in positions
                ],
            }
        )
    return groups


def samples_for_prompt(samples: list[Any], prompt: Any) -> dict[Any, dict[str, Any]]:
    if prompt is None:
        return {}
    rows = [dict(sample) for sample in samples if sample["prompt_id"] == prompt["id"]]
    result = {}
    for index, sample in enumerate(rows, start=1):
        sample["filename"] = Path(sample["image_path"]).name
        key = sample["epoch"] if sample["epoch"] is not None else sample["step"] if sample["step"] is not None else index
        result[key] = sample
    return result


def write_comparison_markdown(comparison: dict[str, Any]) -> str:
    left = comparison["left"]
    right = comparison["right"]
    left_id = int(left["job"]["id"])
    right_id = int(right["job"]["id"])
    output_dir = settings.RUNS_DIR / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"compare_job_{left_id:06d}_job_{right_id:06d}.md"
    lines = [
        f"# Compare Job #{left_id} vs Job #{right_id}",
        "",
        "## Jobs",
        f"- Job #{left_id}: {left['job']['name']} / {left['preset']['name'] if left['preset'] else '-'}",
        f"  - Parent Job: #{left['job']['parent_job_id']}" if left["job"]["parent_job_id"] else "  - Parent Job: -",
        f"- Job #{right_id}: {right['job']['name']} / {right['preset']['name'] if right['preset'] else '-'}",
        f"  - Parent Job: #{right['job']['parent_job_id']}" if right["job"]["parent_job_id"] else "  - Parent Job: -",
        "",
        "## Parameter Differences",
    ]
    for row in comparison["param_rows"]:
        marker = "changed" if row["changed"] else "same"
        lines.append(f"- {row['key']}: {row['left']} | {row['right']} ({marker})")
    lines.extend(["", "## Metrics"])
    for row in comparison["metric_rows"]:
        marker = "changed" if row["changed"] else "same"
        lines.append(f"- {row['key']}: {row['left']} | {row['right']} ({marker})")
    lines.extend(
        [
            "",
            "## Selected LoRA",
            f"- Job #{left_id}: {left['job']['adopted_model_path'] or '-'}",
            f"- Job #{right_id}: {right['job']['adopted_model_path'] or '-'}",
            "",
            "## Human Notes",
        ]
    )
    for bundle in (left, right):
        lines.append(f"### Job #{bundle['job']['id']}")
        for sample in bundle["samples"]:
            if sample["rating"] is not None or sample["memo"]:
                lines.append(f"- {Path(sample['image_path']).name}: rating={sample['rating'] or 0}, memo={sample['memo'] or ''}")
    lines.extend(
        [
            "",
            "## Health Note",
            "Loss health is a training-log health check, not an image quality score. WARNING can still be usable when sample images look better.",
            "",
            "## Sample Files",
        ]
    )
    for group in comparison["sample_groups"]:
        lines.append(f"### {group['title']}")
        for row in group["rows"]:
            left_name = row["left"]["filename"] if row["left"] else "-"
            right_name = row["right"]["filename"] if row["right"] else "-"
            lines.append(f"- {row['label']}: {left_name} | {right_name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def group_samples(sample_prompts: list[Any], samples: list[Any]) -> list[dict[str, Any]]:
    prompt_map = {row["id"]: row for row in sample_prompts}
    groups: dict[int, dict[str, Any]] = {}
    fallback_key = 0
    for sample in samples:
        sample_item = dict(sample)
        sample_item["filename"] = Path(sample["image_path"]).name
        key = sample["prompt_id"] or fallback_key
        prompt = prompt_map.get(sample["prompt_id"])
        groups.setdefault(
            key,
            {
                "prompt": prompt,
                "title": prompt["name"] if prompt else "Unmatched prompt",
                "samples": [],
            },
        )
        groups[key]["samples"].append(sample_item)
    for prompt in sample_prompts:
        groups.setdefault(prompt["id"], {"prompt": prompt, "title": prompt["name"], "samples": []})
    for group in groups.values():
        group["samples"].sort(key=lambda item: (
            item["epoch"] if item["epoch"] is not None else 999999,
            item["step"] if item["step"] is not None else 999999,
            item["created_at"],
            item["id"],
        ))
    return sorted(groups.values(), key=lambda group: group["prompt"]["sort_order"] if group["prompt"] else 999999)


def build_loss_chart(metrics: list[Any]) -> dict[str, Any] | None:
    loss_rows = [row for row in metrics if row["loss"] is not None and row["step"] is not None]
    if len(loss_rows) < 2:
        return None
    width = 720
    height = 220
    pad = 28
    steps = [int(row["step"]) for row in loss_rows]
    losses = [float(row["loss"]) for row in loss_rows]
    min_step, max_step = min(steps), max(steps)
    min_loss, max_loss = min(losses), max(losses)
    if min_loss == max_loss:
        min_loss -= 0.001
        max_loss += 0.001

    def point(step: int, value: float) -> tuple[float, float]:
        x = pad + (step - min_step) / max(1, max_step - min_step) * (width - pad * 2)
        y = height - pad - (value - min_loss) / max(0.000001, max_loss - min_loss) * (height - pad * 2)
        return round(x, 2), round(y, 2)

    raw_points = " ".join(f"{x},{y}" for x, y in (point(step, value) for step, value in zip(steps, losses)))
    averages = moving_average(losses, 3)
    ma_points = " ".join(f"{x},{y}" for x, y in (point(step, value) for step, value in zip(steps, averages)))
    return {
        "width": width,
        "height": height,
        "raw_points": raw_points,
        "ma_points": ma_points,
        "min_loss": min_loss,
        "max_loss": max_loss,
        "min_step": min_step,
        "max_step": max_step,
    }


def moving_average(values: list[float], window: int) -> list[float]:
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(sum(values[start:index + 1]) / (index - start + 1))
    return result
