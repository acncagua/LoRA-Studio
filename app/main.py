from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app import settings
from app.db import connect, create_job, fetch_all, fetch_one, init_db, insert_dataset
from app.services.command_builder import prepare_job_files

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
    }
    jobs = fetch_all("SELECT * FROM training_jobs ORDER BY id DESC LIMIT 8")
    return render(request, "dashboard.html", stats=stats, jobs=jobs)


@app.get("/environment", response_class=HTMLResponse)
def environment(request: Request) -> HTMLResponse:
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
    return render(request, "job_detail.html", job=job, dataset=dataset, outputs=outputs, samples=samples)


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


@app.get("/compare", response_class=HTMLResponse)
def compare_epochs(request: Request) -> HTMLResponse:
    jobs = fetch_all("SELECT id, name, status, adopted_epoch FROM training_jobs ORDER BY id DESC")
    return render(request, "compare_epochs.html", jobs=jobs)
