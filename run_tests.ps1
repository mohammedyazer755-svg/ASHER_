[CmdletBinding()]
param(
    [switch]$Fast
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw 'The project environment is missing. Run .\setup_asher.ps1 first.'
}

# Keep the verification command hermetic even when a developer has live
# provider variables in their interactive shell.  Values are restored before
# returning so this script never changes the caller's environment.
$savedDryRun = $env:ASHER_DRY_RUN
$savedOpenAi = $env:OPENAI_API_KEY
$savedRemoteOllama = $env:ASHER_ALLOW_REMOTE_OLLAMA
$savedQtPlatform = $env:QT_QPA_PLATFORM
try {
    $env:ASHER_DRY_RUN = 'true'
    Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:ASHER_ALLOW_REMOTE_OLLAMA -ErrorAction SilentlyContinue
    if (-not $env:QT_QPA_PLATFORM) { $env:QT_QPA_PLATFORM = 'offscreen' }

    Write-Host 'Running syntax/import checks...' -ForegroundColor Cyan
    & $VenvPython -B -m compileall -q asher main.py
    if ($LASTEXITCODE -ne 0) { throw 'Python syntax compilation failed.' }

    Write-Host 'Running the complete unittest suite...' -ForegroundColor Cyan
    & $VenvPython -B -m unittest discover -s tests -p 'test_*.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'The unittest suite failed.' }

    if (-not $Fast) {
        Write-Host 'Running the deterministic legacy voice smoke test...' -ForegroundColor Cyan
        & $VenvPython -B test_voice_accuracy.py
        if ($LASTEXITCODE -ne 0) { throw 'The voice accuracy smoke test failed.' }
    }
} finally {
    if ($null -eq $savedDryRun) { Remove-Item Env:ASHER_DRY_RUN -ErrorAction SilentlyContinue } else { $env:ASHER_DRY_RUN = $savedDryRun }
    if ($null -eq $savedOpenAi) { Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue } else { $env:OPENAI_API_KEY = $savedOpenAi }
    if ($null -eq $savedRemoteOllama) { Remove-Item Env:ASHER_ALLOW_REMOTE_OLLAMA -ErrorAction SilentlyContinue } else { $env:ASHER_ALLOW_REMOTE_OLLAMA = $savedRemoteOllama }
    if ($null -eq $savedQtPlatform) { Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue } else { $env:QT_QPA_PLATFORM = $savedQtPlatform }
}

Write-Host 'All requested checks completed.' -ForegroundColor Green
