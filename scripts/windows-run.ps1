#Requires -Version 7.3
# Run project tasks without activating a venv or changing persistent settings.
# Use raw script arguments: advanced parameter binding consumes -- / --flag.
$TaskArgs = @($args)
if ($TaskArgs.Count -eq 0) { $TaskArgs = @('doctor') }

$ErrorActionPreference = 'Stop'
$PSNativeCommandArgumentPassing = 'Standard'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$pythonPrefix = @()

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    $pythonCommand = Get-Command python -CommandType Application -All -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -notmatch '\\Microsoft\\WindowsApps\\' } |
        Select-Object -First 1
    if ($pythonCommand) {
        $pythonPath = $pythonCommand.Source
    } else {
        $launcher = Get-Command py -CommandType Application -ErrorAction SilentlyContinue
        if (-not $launcher) {
            throw 'No Python interpreter found. Use python or py -3.12; python3 aliases are not supported.'
        }
        $pythonPath = $launcher.Source
        $pythonPrefix = @('-3.12')
    }
}

$originalShell = $env:SHELL
$originalLocation = Get-Location
$exitCode = 1
try {
    # This affects only this process and its children, never User/Machine env.
    $env:SHELL = (Get-Process -Id $PID).Path
    Set-Location -LiteralPath $projectRoot
    & $pythonPath @pythonPrefix (Join-Path $projectRoot 'run.py') @TaskArgs
    $exitCode = $LASTEXITCODE
} finally {
    $env:SHELL = $originalShell
    Set-Location -LiteralPath $originalLocation.Path
}
exit $exitCode
