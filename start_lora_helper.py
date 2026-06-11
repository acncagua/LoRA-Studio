from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import webbrowser

from app import settings
from app.db import init_db


def sd_scripts_ready() -> bool:
    sd_scripts = settings.SD_SCRIPTS_DIR
    venv_python = sd_scripts / "venv" / "Scripts" / "python.exe"
    required_files = [
        sd_scripts / ".git",
        sd_scripts / "sdxl_train_network.py",
        sd_scripts / "train_network.py",
        venv_python,
    ]
    return all(path.exists() for path in required_files)


def ensure_sd_scripts_installed() -> None:
    if sd_scripts_ready():
        return

    setup_script = settings.ROOT_DIR / "scripts" / "setup_sd_scripts.ps1"
    if not setup_script.exists():
        raise FileNotFoundError(f"sd-scripts setup script not found: {setup_script}")

    print(f"sd-scripts is not ready. Installing {settings.SD_SCRIPTS_RELEASE_TAG} ...")
    result = subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(setup_script),
            "-InstallRoot",
            str(settings.EXTERNAL_DIR),
            "-ReleaseTag",
            settings.SD_SCRIPTS_RELEASE_TAG,
        ],
        cwd=settings.ROOT_DIR,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sd-scripts setup failed with exit code {result.returncode}")


def find_listening_pids(port: int) -> set[int]:
    if sys.platform != "win32":
        return set()

    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return set()

    pids: set[int] = set()
    port_pattern = re.compile(rf"^(?:0\.0\.0\.0|127\.0\.0\.1|\[?::\]?|\[?::1\]?|[^\s]+):{port}$")
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local_address, state, pid_text = parts[1], parts[3], parts[4]
        if state.upper() != "LISTENING" or not pid_text.isdigit():
            continue
        if port_pattern.match(local_address):
            pids.add(int(pid_text))
    return pids


def release_port(port: int) -> None:
    current_pid = os.getpid()
    for pid in sorted(find_listening_pids(port)):
        if pid == current_pid:
            continue
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start LoRA-Studio local web app.")
    parser.add_argument("--host", default=settings.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=settings.DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--skip-sd-scripts-setup", action="store_true")
    args = parser.parse_args()

    init_db()
    if not args.skip_sd_scripts_setup:
        ensure_sd_scripts_installed()
    url = f"http://{args.host}:{args.port}"
    if args.port != 7865:
        release_port(7865)
    release_port(args.port)
    if not args.no_browser:
        webbrowser.open(url)
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
