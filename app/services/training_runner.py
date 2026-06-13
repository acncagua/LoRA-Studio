from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.db import connect, fetch_one, latest_environment, utc_now
from app.services.command_builder import prepare_job_files
from app.services.output_collector import collect_job_results

RUNNABLE_STATUSES = {"draft", "prepared", "failed", "stopped"}


def reconcile_stale_running_jobs() -> list[int]:
    rows = []
    with connect() as conn:
        rows = conn.execute("SELECT * FROM training_jobs WHERE status = 'running'").fetchall()
    fixed: list[int] = []
    for row in rows:
        job = dict(row)
        pid = job.get("process_id")
        if pid and process_exists(int(pid)):
            continue
        marker = Path(job.get("run_dir") or "").name
        if marker.startswith("job_") and job_marker_process_exists(marker):
            continue
        mark_running_job_lost(job)
        fixed.append(int(job["id"]))
    return fixed


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def job_marker_process_exists(marker: str) -> bool:
    if sys.platform != "win32":
        return False
    script = (
        "$marker = " + json.dumps(marker) + "; "
        "$p = Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
        "Where-Object { $_.CommandLine -like ('*' + $marker + '*') } | "
        "Select-Object -First 1 -ExpandProperty ProcessId; "
        "if ($p) { Write-Output $p }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return bool(result.stdout.strip())


def mark_running_job_lost(job: dict[str, object]) -> None:
    job_id = int(job["id"])
    end_time = utc_now()
    elapsed = elapsed_seconds(str(job.get("start_time") or ""), end_time)
    log_path = Path(str(job.get("run_dir") or "")) / "logs" / "train.log"
    if log_path.exists():
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write("\n[LoRA-Studio] running status reconciled: training process was not found. Marking job stopped.\n")
    try:
        collect_job_results(job_id)
    except Exception as exc:
        if log_path.exists():
            with log_path.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(f"[LoRA-Studio] result import during stale-process reconciliation failed: {exc}\n")
    with connect() as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET status = 'stopped', process_id = NULL,
                end_time = COALESCE(end_time, ?),
                elapsed_seconds = COALESCE(elapsed_seconds, ?),
                return_code = COALESCE(return_code, 4294967295),
                updated_at = ?
            WHERE id = ? AND status = 'running'
            """,
            (end_time, elapsed, end_time, job_id),
        )


def start_job(job_id: int, acknowledge_trigger_mismatch: bool = False) -> int:
    with connect() as conn:
        running = conn.execute(
            "SELECT id FROM training_jobs WHERE status = 'running' AND id != ? LIMIT 1",
            (job_id,),
        ).fetchone()
        if running:
            raise RuntimeError(f"Job #{running['id']} is already running.")

        job = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
        if job is None:
            raise ValueError(f"Job not found: {job_id}")
        if job["status"] not in RUNNABLE_STATUSES:
            raise RuntimeError(f"Job status is not runnable: {job['status']}")
        if "config_dirty" in job.keys() and int(job["config_dirty"] or 0):
            raise RuntimeError("Job settings were changed after Prepare. Run Prepare Files again before starting.")

    ensure_job_prepared(job_id)
    validate_job_ready(job_id, acknowledge_trigger_mismatch=acknowledge_trigger_mismatch)

    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")

    command_argv_path = Path(job["run_dir"]) / "config" / "command_argv.json"
    argv = json.loads(command_argv_path.read_text(encoding="utf-8"))
    environment = latest_environment()
    if environment is None:
        raise RuntimeError("sd-scripts environment is not registered.")
    sd_scripts_path = Path(environment["sd_scripts_path"])
    log_path = Path(job["run_dir"]) / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    archive_existing_log(log_path)

    start_time = utc_now()
    log_handle = log_path.open("ab")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    process = subprocess.Popen(
        argv,
        cwd=str(sd_scripts_path),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        shell=False,
        env=env,
    )
    with connect() as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET status = 'running', start_time = ?, end_time = NULL,
                elapsed_seconds = NULL, process_id = ?, return_code = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (start_time, process.pid, start_time, job_id),
        )

    thread = threading.Thread(
        target=monitor_process,
        args=(job_id, process, log_handle, start_time, log_path),
        daemon=True,
    )
    thread.start()
    return process.pid


def ensure_job_prepared(job_id: int) -> None:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
    if dataset is None:
        raise RuntimeError("Dataset not found.")

    config_dir = Path(job["run_dir"]) / "config"
    required = [
        config_dir / "dataset_config.toml",
        config_dir / "sample_prompts.txt",
        config_dir / "command_argv.json",
    ]
    if job["status"] == "draft" or not all(path.exists() for path in required):
        files = prepare_job_files(dict(job), dict(dataset))
        with connect() as conn:
            conn.execute(
                "UPDATE training_jobs SET command_line = ?, status = 'prepared', config_dirty = 0, updated_at = ? WHERE id = ?",
                (files["command"], utc_now(), job_id),
            )


def validate_job_ready(job_id: int, acknowledge_trigger_mismatch: bool = False) -> None:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    if "config_dirty" in job.keys() and int(job["config_dirty"] or 0):
        raise RuntimeError("Job settings were changed after Prepare. Run Prepare Files again before starting.")
    dataset = fetch_one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
    if dataset is None:
        raise RuntimeError("Dataset not found.")
    dataset_path = Path(dataset["path"])
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset path does not exist: {dataset_path}")
    if int(dataset["image_count"] or 0) <= 0:
        raise RuntimeError("Dataset image_count must be greater than 0.")
    analysis = fetch_one("SELECT * FROM dataset_analysis WHERE dataset_id = ?", (dataset["id"],))
    if (
        analysis is not None
        and analysis["trigger_consistency_label"] == "ERROR"
        and not acknowledge_trigger_mismatch
    ):
        raise RuntimeError("Trigger mismatch must be acknowledged before running this job.")
    if not Path(job["base_model_path"]).exists():
        raise RuntimeError(f"Base model path does not exist: {job['base_model_path']}")

    environment = latest_environment()
    if environment is None:
        raise RuntimeError("sd-scripts environment is not registered.")
    sd_scripts_path = Path(environment["sd_scripts_path"])
    if not sd_scripts_path.exists():
        raise RuntimeError(f"sd-scripts path does not exist: {sd_scripts_path}")
    training_script = sd_scripts_path / job["training_script"]
    if not training_script.exists():
        raise RuntimeError(f"Training script does not exist: {training_script}")
    if not Path(environment["venv_python_path"]).exists():
        raise RuntimeError(f"venv python does not exist: {environment['venv_python_path']}")

    config_dir = Path(job["run_dir"]) / "config"
    for filename in ("sample_prompts.txt", "dataset_config.toml", "command_argv.json"):
        path = config_dir / filename
        if not path.exists():
            raise RuntimeError(f"Prepared file does not exist: {path}")


def monitor_process(
    job_id: int,
    process: subprocess.Popen[bytes],
    log_handle,
    start_time_text: str,
    log_path: Path,
) -> None:
    return_code = process.wait()
    log_handle.close()
    end_time = utc_now()
    elapsed = elapsed_seconds(start_time_text, end_time)
    status = "completed" if return_code == 0 else "failed"

    current = fetch_one("SELECT status FROM training_jobs WHERE id = ?", (job_id,))
    if current is not None and current["status"] == "stopped":
        status = "stopped"

    try:
        imported = collect_job_results(job_id)
    except Exception as exc:
        imported = {"models": 0, "samples": 0}
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(f"\n[LoRA-Studio] result import failed: {exc}\n")
    if status == "completed" and imported["models"] == 0 and not has_model_outputs(job_id):
        status = "failed"
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write("\n[LoRA-Studio] training ended without LoRA outputs; marking job failed.\n")

    with connect() as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET status = ?, return_code = ?, end_time = ?, elapsed_seconds = ?,
                process_id = NULL, updated_at = ?
            WHERE id = ?
            """,
            (status, return_code, end_time, elapsed, end_time, job_id),
        )


def stop_job(job_id: int) -> None:
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    end_time = utc_now()
    elapsed = elapsed_seconds(job["start_time"], end_time) if job["start_time"] else None
    with connect() as conn:
        conn.execute(
            """
            UPDATE training_jobs
            SET status = 'stopped', end_time = ?, elapsed_seconds = ?,
                process_id = NULL, updated_at = ?
            WHERE id = ?
            """,
            (end_time, elapsed, end_time, job_id),
        )
    pid = job["process_id"]
    if pid:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    kill_job_processes(dict(job))


def kill_job_processes(job: dict) -> None:
    run_dir = str(job.get("run_dir") or "")
    if not run_dir:
        return
    marker = Path(run_dir).name
    if not marker.startswith("job_"):
        return
    script = (
        "$marker = " + json.dumps(marker) + "; "
        "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
        "Where-Object { $_.CommandLine -like ('*' + $marker + '*') } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, check=False)


def archive_existing_log(log_path: Path) -> None:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = log_path.with_name(f"train_{stamp}.log")
    try:
        log_path.replace(archive_path)
    except OSError:
        pass


def has_model_outputs(job_id: int) -> bool:
    row = fetch_one("SELECT id FROM training_outputs WHERE job_id = ? AND file_type = 'model' LIMIT 1", (job_id,))
    return row is not None


def elapsed_seconds(start_time_text: str | None, end_time_text: str) -> int | None:
    if not start_time_text:
        return None
    try:
        start = datetime.fromisoformat(start_time_text)
        end = datetime.fromisoformat(end_time_text)
    except ValueError:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return max(0, int((end - start).total_seconds()))


def read_log_tail(job: dict, max_lines: int = 200) -> str:
    log_path = Path(job["run_dir"]) / "logs" / "train.log"
    if not log_path.exists():
        return ""
    try:
        data = log_path.read_bytes()
    except OSError:
        return ""
    lines = decode_log_bytes(data).splitlines()
    return "\n".join(lines[-max_lines:])


def decode_log_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "cp932", "shift_jis"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")
