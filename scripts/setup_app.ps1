param([string]$PythonCmd = "")

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

function Test-PythonCmd([string]$Command) {
    try {
        Invoke-Expression "$Command -c `"import sys; print(sys.version_info[:2])`"" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Resolve-PythonCmd([string]$RequestedCommand) {
    if (-not [string]::IsNullOrWhiteSpace($RequestedCommand)) {
        if (Test-PythonCmd $RequestedCommand) { return $RequestedCommand }
        throw "Requested Python command is not available: $RequestedCommand"
    }
    foreach ($candidate in @("py -3.10", "py -3.12", "python")) {
        if (Test-PythonCmd $candidate) { return $candidate }
    }
    throw "No usable Python command found. Install Python 3.10, Python 3.12, or provide python on PATH."
}

$PythonCmd = Resolve-PythonCmd $PythonCmd
if ((Test-Path ".\.venv") -and -not (Test-PythonCmd ".\.venv\Scripts\python.exe")) {
    Remove-Item -Recurse -Force ".\.venv"
}
if (-not (Test-Path ".\.venv")) {
    Invoke-Expression "$PythonCmd -m venv .venv"
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe -c "from app.db import init_db; init_db(); print('initialized')"
