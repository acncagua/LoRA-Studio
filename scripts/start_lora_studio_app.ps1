param(
    [Parameter(Mandatory = $true)]
    [string]$Python,

    [string]$PythonArgs = "",

    [Parameter(Mandatory = $true)]
    [string]$Script,

    [string]$PythonPath = "",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = "Stop"

if ($PythonPath) {
    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$PythonPath;$env:PYTHONPATH"
    } else {
        $env:PYTHONPATH = $PythonPath
    }
}

$argsList = @()
if ($PythonArgs) {
    $argsList += $PythonArgs.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
}
$argsList += $Script
if ($AppArgs) {
    $argsList += $AppArgs
}

& $Python @argsList
exit $LASTEXITCODE
