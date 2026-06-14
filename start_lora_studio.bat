@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_PYTHON=.venv\Scripts\python.exe"
set "VENV_SITE=.venv\Lib\site-packages"
set "APP_SCRIPT=start_lora_studio.py"
set "APP_PYTHON="
set "APP_PYTHONPATH="

powershell -NoProfile -ExecutionPolicy Bypass -Command "$py=$env:VENV_PYTHON; $item=Get-Item -LiteralPath $py -ErrorAction SilentlyContinue; if($item -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0)){ & $py -c 'import fastapi, uvicorn, jinja2' *> $null; exit $LASTEXITCODE } exit 1"
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
echo [LoRA-Studio] Python: %APP_PYTHON% %APP_PYTHON_ARGS%
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\start_lora_studio_app.ps1" -Python "%APP_PYTHON%" -PythonArgs "%APP_PYTHON_ARGS%" -Script "%APP_SCRIPT%" -PythonPath "%APP_PYTHONPATH%" %*
exit /b %ERRORLEVEL%

:resolve_env_python
if not exist "%VENV_SITE%" exit /b 0
set "CANDIDATE_PYTHON="
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$names=@('Python312','Python310'); $roots=@((Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\Python'), [Environment]::GetFolderPath('ProgramFiles')); foreach($root in $roots){ foreach($name in $names){ $p=Join-Path (Join-Path $root $name) 'python.exe'; if(Test-Path -LiteralPath $p){ Write-Output $p; exit 0 } } }"`) do set "CANDIDATE_PYTHON=%%P"
if not defined CANDIDATE_PYTHON (
    set "CHECK_PYTHON=py"
    set "CHECK_PYTHON_ARGS=-3.12"
    goto check_candidate_python
)
set "CHECK_PYTHON=%CANDIDATE_PYTHON%"
set "CHECK_PYTHON_ARGS="
:check_candidate_python
set "CHECK_PYTHONPATH=%VENV_SITE%"
echo [LoRA-Studio] Using venv site-packages with detected Python.
set "APP_PYTHON=%CHECK_PYTHON%"
set "APP_PYTHON_ARGS=%CHECK_PYTHON_ARGS%"
set "APP_PYTHONPATH=%VENV_SITE%"
exit /b 0
