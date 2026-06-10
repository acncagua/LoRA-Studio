param([string]$PythonCmd = "py -3.10")
$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
if (-not (Test-Path ".\.venv")) { Invoke-Expression "$PythonCmd -m venv .venv" }
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe -c "from app.db import init_db; init_db(); print('initialized')"
