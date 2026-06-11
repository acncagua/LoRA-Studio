from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app import settings
from app.db import connect, create_job, fetch_all, fetch_one, import_latest_environment, init_db, insert_dataset
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


@app.get("/jobs/new", response_class=HTMLResponse)
def job_new(request: Request) -> HTMLResponse:
    datasets = fetch_all("SELECT * FROM datasets ORDER BY id DESC")
    presets = fetch_all("SELECT * FROM presets ORDER BY model_family DESC, name")
    return render(request, "job_create.html", datasets=datasets, presets=presets)


@app.post("/jobs")
def job_create(name: str = Form(...), dataset_id: int = Form(...), preset_id: str = Form(...), base_model_path: str = Form(...), vae_path: str = Form(""), output_name: str = Form(""), memo: str = Form("")) -> RedirectResponse:
    job_id = create_job({"name": name, "dataset_id": dataset_id, "preset_id": preset_id, "base_model_path": base_model_path, "vae_path": vae_path, "output_name": output_name, "memo": memo})
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


@app.get("/compare", response_class=HTMLResponse)
def compare_epochs(request: Request) -> HTMLResponse:
    jobs = fetch_all("SELECT id, name, status, adopted_epoch FROM training_jobs ORDER BY id DESC")
    return render(request, "compare_epochs.html", jobs=jobs)


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
