param([string]$PythonCmd = "")

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

function Test-PythonCmd([string]$Command) {
    try {
        $pathCommand = $Command.Trim().Trim('"').Trim("'")
        if (Test-Path -LiteralPath $pathCommand) {
            & $pathCommand --version
        } else {
            Invoke-Expression "& $Command --version"
        }
        return ($? -or $LASTEXITCODE -eq 0)
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
        $command = if (Test-Path $RequestedCommand) { Quote-CommandPath $RequestedCommand } else { $RequestedCommand }
        if (Test-PythonCmd $command) { return $command }
        throw "Requested Python command is not available: $RequestedCommand"
    }
    $pythonSettingPath = ".\data\python_cmd.txt"
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
if ((Test-Path ".\.venv") -and -not (Test-PythonCmd ".\.venv\Scripts\python.exe")) {
    Remove-Item -Recurse -Force ".\.venv"
}
if (-not (Test-Path ".\.venv")) {
    Invoke-Expression "& $PythonCmd -m venv .venv"
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe -c "from app.db import init_db; init_db(); print('initialized')"
