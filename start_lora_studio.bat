@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_PYTHON=.venv\Scripts\python.exe"
set "VENV_SITE=.venv\Lib\site-packages"
set "APP_SCRIPT=start_lora_studio.py"
set "APP_PYTHON="
set "APP_PYTHONPATH="

powershell -NoProfile -ExecutionPolicy Bypass -Command "$py=$env:VENV_PYTHON; if((Test-Path -LiteralPath $py)){ & $py -c 'import fastapi, uvicorn, jinja2' *> $null; exit $LASTEXITCODE } exit 1"
if not errorlevel 1 (
    set "APP_PYTHON=%VENV_PYTHON%"
    goto start_app
)

call :resolve_env_python
if defined APP_PYTHON goto start_app

echo [LoRA-Studio] .venv is missing or incomplete. Running setup_app.ps1...
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\setup_app.ps1"
if errorlevel 1 (
    echo [LoRA-Studio] setup failed.
    pause
    exit /b 1
)
set "APP_PYTHON=%VENV_PYTHON%"

:start_app
echo [LoRA-Studio] Starting LoRA-Studio...
set "LORA_STUDIO_APP_PYTHON=%APP_PYTHON%"
set "LORA_STUDIO_APP_PYTHON_ARGS=%APP_PYTHON_ARGS%"
set "LORA_STUDIO_APP_SCRIPT=%APP_SCRIPT%"
set "LORA_STUDIO_APP_PYTHONPATH=%APP_PYTHONPATH%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$py=$env:LORA_STUDIO_APP_PYTHON; $script=$env:LORA_STUDIO_APP_SCRIPT; if($env:LORA_STUDIO_APP_PYTHONPATH){ $env:PYTHONPATH=$env:LORA_STUDIO_APP_PYTHONPATH + ';' + $env:PYTHONPATH }; if($env:LORA_STUDIO_APP_PYTHON_ARGS){ & $py $env:LORA_STUDIO_APP_PYTHON_ARGS $script @args } else { & $py $script @args }" %*
exit /b %ERRORLEVEL%

:resolve_env_python
if not exist "%VENV_SITE%" exit /b 0
set "CANDIDATE_PYTHON="
if not defined CANDIDATE_PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "CANDIDATE_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined CANDIDATE_PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "CANDIDATE_PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined CANDIDATE_PYTHON if exist "%ProgramFiles%\Python312\python.exe" set "CANDIDATE_PYTHON=%ProgramFiles%\Python312\python.exe"
if not defined CANDIDATE_PYTHON if exist "%ProgramFiles%\Python310\python.exe" set "CANDIDATE_PYTHON=%ProgramFiles%\Python310\python.exe"
if not defined CANDIDATE_PYTHON (
    set "CHECK_PYTHON=py"
    set "CHECK_PYTHON_ARGS=-3.12"
    goto check_candidate_python
)
set "CHECK_PYTHON=%CANDIDATE_PYTHON%"
set "CHECK_PYTHON_ARGS="
:check_candidate_python
set "CHECK_PYTHONPATH=%VENV_SITE%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:PYTHONPATH=$env:CHECK_PYTHONPATH + ';' + $env:PYTHONPATH; if($env:CHECK_PYTHON_ARGS){ & $env:CHECK_PYTHON $env:CHECK_PYTHON_ARGS -c 'import fastapi, uvicorn, jinja2' *> $null } else { & $env:CHECK_PYTHON -c 'import fastapi, uvicorn, jinja2' *> $null }; exit $LASTEXITCODE"
if errorlevel 1 exit /b 0
echo [LoRA-Studio] Using venv site-packages with detected Python.
set "APP_PYTHON=%CHECK_PYTHON%"
set "APP_PYTHON_ARGS=%CHECK_PYTHON_ARGS%"
set "APP_PYTHONPATH=%VENV_SITE%"
exit /b 0
