param(
    [string]$InstallRoot = ".\external",
    [string]$PythonCmd = "",
    [string]$CudaProfile = "cu128",
    [string]$MixedPrecision = "bf16",
    [string]$ReleaseTag = "v0.10.5",
    [bool]$InstallOptionalOptimizerDeps = $true
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$RepoUrl = "https://github.com/kohya-ss/sd-scripts.git"
$ExpectedCommitPrefix = "a1b48df"
$Root = (Resolve-Path ".").Path
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "setup_sd_scripts.log"

function Write-Log([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format o), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

function Invoke-LoggedNative([string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory = "") {
    $previousErrorActionPreference = $ErrorActionPreference
    $previousLocation = (Get-Location).Path
    $ErrorActionPreference = "Continue"
    try {
        if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory)) {
            Set-Location -LiteralPath $WorkingDirectory
        }
        & $FilePath @Arguments *>&1 | Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
    } finally {
        Set-Location -LiteralPath $previousLocation
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $FilePath $($Arguments -join ' ')"
    }
}

function Test-PythonCmd([string]$Command) {
    try {
        Invoke-Expression "& $Command -c `"import sys; print(sys.version_info[:2])`"" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Quote-CommandPath([string]$Path) {
    return "`"$Path`""
}

function Add-PythonCandidate([System.Collections.Generic.List[string]]$Candidates, [string]$Command) {
    if ([string]::IsNullOrWhiteSpace($Command)) { return }
    if (-not $Candidates.Contains($Command)) {
        $Candidates.Add($Command)
    }
}

function Get-ScannedPythonCommands([string]$MajorMinor) {
    $commands = New-Object 'System.Collections.Generic.List[string]'
    $roots = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python"),
        $env:ProgramFiles,
        (Join-Path $env:ProgramFiles "Python")
    )
    foreach ($root in $roots) {
        if ([string]::IsNullOrWhiteSpace($root) -or -not (Test-Path $root)) { continue }
        Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^Python(\d+)$' } |
            ForEach-Object {
                $digits = $Matches[1]
                if ($digits.Length -ge 2) {
                    $version = "{0}.{1}" -f $digits.Substring(0, 1), $digits.Substring(1)
                    if ($version -eq $MajorMinor) {
                        $python = Join-Path $_.FullName "python.exe"
                        if (Test-Path $python) { Add-PythonCandidate $commands (Quote-CommandPath $python) }
                    }
                }
            }
    }
    return $commands
}

function Get-GeneratedPythonCommands([string]$FolderName) {
    $commands = New-Object 'System.Collections.Generic.List[string]'
    $paths = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\$FolderName\python.exe"),
        (Join-Path $env:ProgramFiles "$FolderName\python.exe")
    )
    foreach ($path in $paths) {
        if (-not [string]::IsNullOrWhiteSpace($path) -and (Test-Path $path)) {
            Add-PythonCandidate $commands (Quote-CommandPath $path)
        }
    }
    return $commands
}

function Get-OtherScannedPythonCommands {
    $commands = New-Object 'System.Collections.Generic.List[string]'
    $roots = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python"),
        $env:ProgramFiles,
        (Join-Path $env:ProgramFiles "Python")
    )
    foreach ($root in $roots) {
        if ([string]::IsNullOrWhiteSpace($root) -or -not (Test-Path $root)) { continue }
        Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^Python(\d+)$' } |
            Sort-Object Name |
            ForEach-Object {
                $python = Join-Path $_.FullName "python.exe"
                if (Test-Path $python) { Add-PythonCandidate $commands (Quote-CommandPath $python) }
            }
    }
    return $commands
}

function Resolve-PythonCmd([string]$RequestedCommand) {
    if (-not [string]::IsNullOrWhiteSpace($RequestedCommand)) {
        if (Test-PythonCmd $RequestedCommand) { return $RequestedCommand }
        throw "Requested Python command is not available: $RequestedCommand"
    }
    $pythonSettingPath = Join-Path $Root "data\python_cmd.txt"
    if (Test-Path $pythonSettingPath) {
        $configured = (Get-Content -LiteralPath $pythonSettingPath -Encoding UTF8 -Raw).Trim()
        if (-not [string]::IsNullOrWhiteSpace($configured)) {
            $command = if (Test-Path $configured) { Quote-CommandPath $configured } else { $configured }
            if (Test-PythonCmd $command) { return $command }
            throw "Configured Python command is not available: $configured"
        }
    }
    foreach ($manual in @($env:LORA_STUDIO_PYTHON_EXE, $env:LORA_STUDIO_PYTHON)) {
        if (-not [string]::IsNullOrWhiteSpace($manual)) {
            $command = if (Test-Path $manual) { Quote-CommandPath $manual } else { $manual }
            if (Test-PythonCmd $command) { return $command }
            throw "Configured Python command is not available: $manual"
        }
    }

    $candidates = New-Object 'System.Collections.Generic.List[string]'
    foreach ($candidate in (Get-GeneratedPythonCommands "Python310")) { Add-PythonCandidate $candidates $candidate }
    foreach ($candidate in (Get-ScannedPythonCommands "3.10")) { Add-PythonCandidate $candidates $candidate }
    Add-PythonCandidate $candidates "py -3.10"
    foreach ($candidate in (Get-GeneratedPythonCommands "Python312")) { Add-PythonCandidate $candidates $candidate }
    foreach ($candidate in (Get-ScannedPythonCommands "3.12")) { Add-PythonCandidate $candidates $candidate }
    Add-PythonCandidate $candidates "py -3.12"
    foreach ($candidate in (Get-OtherScannedPythonCommands)) { Add-PythonCandidate $candidates $candidate }
    Add-PythonCandidate $candidates "python"

    foreach ($candidate in $candidates) {
        if (Test-PythonCmd $candidate) { return $candidate }
    }
    throw "No usable Python command found. Install Python 3.10, Python 3.12, or provide python on PATH."
}

$PythonCmd = Resolve-PythonCmd $PythonCmd
Write-Log "setup start: release=$ReleaseTag cuda=$CudaProfile mixed_precision=$MixedPrecision"
Write-Log "python command: $PythonCmd"
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$SdScriptsPath = Join-Path $InstallRoot "sd-scripts"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git が見つかりません。" }
if (Test-Path $SdScriptsPath) {
    Write-Log "fetching existing sd-scripts"
    Invoke-LoggedNative "git" @("-C", $SdScriptsPath, "fetch", "--tags", "origin")
} else {
    Write-Log "cloning sd-scripts"
    Invoke-LoggedNative "git" @("clone", $RepoUrl, $SdScriptsPath)
}

$SdScriptsPath = (Resolve-Path $SdScriptsPath).Path
Invoke-LoggedNative "git" @("-C", $SdScriptsPath, "checkout", $ReleaseTag)
$Commit = (git -C $SdScriptsPath rev-parse --short HEAD).Trim()
if ($ReleaseTag -eq "v0.10.5" -and -not $Commit.StartsWith($ExpectedCommitPrefix)) { throw "sd-scripts $ReleaseTag のcommitが想定と異なります: $Commit" }

$VenvPath = Join-Path $SdScriptsPath "venv"
if (-not (Test-Path $VenvPath)) {
    Write-Log "creating venv"
    Invoke-Expression "& $PythonCmd -m venv `"$VenvPath`""
}
$Python = Join-Path $VenvPath "Scripts\python.exe"
$Pip = Join-Path $VenvPath "Scripts\pip.exe"
$Accelerate = Join-Path $VenvPath "Scripts\accelerate.exe"

Invoke-LoggedNative $Python @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
switch ($CudaProfile) {
    "cu128" { Invoke-LoggedNative $Pip @("install", "torch==2.8.0", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128") }
    "cu129" { Invoke-LoggedNative $Pip @("install", "torch==2.8.0", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu129") }
    "cu124" { Invoke-LoggedNative $Pip @("install", "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu124") }
    "cu121" { Invoke-LoggedNative $Pip @("install", "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu121") }
    default { throw "Unsupported CudaProfile: $CudaProfile" }
}
Invoke-LoggedNative $Pip @("install", "--upgrade", "-r", "requirements.txt") $SdScriptsPath
if ($InstallOptionalOptimizerDeps) {
    Write-Log "installing optional optimizer dependencies"
    Invoke-LoggedNative $Python @("-m", "pip", "install", "dadaptation", "prodigyopt", "lion-pytorch")
} else {
    Write-Log "skip optional optimizer dependencies"
}

$AccelerateDir = Join-Path $env:USERPROFILE ".cache\huggingface\accelerate"
New-Item -ItemType Directory -Force -Path $AccelerateDir | Out-Null
$AccelerateConfig = Join-Path $AccelerateDir "default_config.yaml"
@"
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: $MixedPrecision
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"@ | Set-Content -LiteralPath $AccelerateConfig -Encoding UTF8

Invoke-LoggedNative $Python @("-c", "import json, torch; print(json.dumps({'torch_version': torch.__version__, 'torch_cuda_version': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, ensure_ascii=False))")
Invoke-LoggedNative $Python @((Join-Path $SdScriptsPath "sdxl_train_network.py"), "--help")
Invoke-LoggedNative $Python @((Join-Path $SdScriptsPath "train_network.py"), "--help")

$Result = [pscustomobject]@{ repo_url=$RepoUrl; release_tag=$ReleaseTag; commit=$Commit; sd_scripts_path=(Resolve-Path $SdScriptsPath).Path; venv_python_path=(Resolve-Path $Python).Path; venv_accelerate_path=(Resolve-Path $Accelerate).Path; cuda_profile=$CudaProfile; mixed_precision=$MixedPrecision; accelerate_config=$AccelerateConfig; log_path=$LogPath; completed_at=(Get-Date -Format o) }
$Result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $LogDir "setup_sd_scripts_result.json") -Encoding UTF8
Write-Log "setup complete: commit=$Commit"
