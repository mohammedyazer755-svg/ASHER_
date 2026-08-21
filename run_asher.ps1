[CmdletBinding()]
param(
    [ValidateSet('ui', 'text', 'voice')]
    [string]$Mode = 'ui',
    [switch]$Live,
    [string]$RuntimeDir
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw 'The project environment is missing. Run .\setup_asher.ps1 first.'
}

$arguments = @('-B', 'main.py', "--$Mode")
if ($Live) { $arguments += '--live' }
if ($RuntimeDir) { $arguments += @('--runtime-dir', $RuntimeDir) }
Write-Host ("Launching ASHER in {0} mode (live={1}). Use the local emergency stop for active sessions." -f $Mode, $Live) -ForegroundColor Cyan
& $VenvPython @arguments
exit $LASTEXITCODE
