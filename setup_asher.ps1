[CmdletBinding()]
param(
    [switch]$SkipOptional
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

function Resolve-Python {
    foreach ($candidate in @('py', 'python')) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            try {
                $version = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
                $parts = $version.Trim().Split('.')
                if ([int]$parts[0] -gt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 11)) {
                    return $candidate
                }
            } catch { }
        }
    }
    throw 'Python 3.11 or newer was not found. Install Python from python.org and retry.'
}

$SystemPython = Resolve-Python
$VenvPath = Join-Path $ProjectRoot '.venv'
$VenvPython = Join-Path $VenvPath 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host 'Creating the private project virtual environment...' -ForegroundColor Cyan
    & $SystemPython -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}

Write-Host 'Updating pip in .venv...' -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed.' }

Write-Host 'Installing deterministic core dependencies...' -ForegroundColor Cyan
& $VenvPython -m pip install -r (Join-Path $ProjectRoot 'requirements-core.txt')
if ($LASTEXITCODE -ne 0) { throw 'Core dependency installation failed.' }

if (-not $SkipOptional) {
    Write-Host 'Attempting optional UI, voice, ML, and Windows adapters...' -ForegroundColor Cyan
    try {
        & $VenvPython -m pip install -r (Join-Path $ProjectRoot 'requirements-optional.txt')
        if ($LASTEXITCODE -ne 0) { throw 'optional pip command returned a failure' }
    } catch {
        Write-Warning ('Optional dependencies were not all installed. ASHER remains usable in text/dry-run mode. Details: ' + $_.Exception.Message)
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot '.env'))) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot '.env.example') -Destination (Join-Path $ProjectRoot '.env')
    Write-Host 'Created .env from .env.example. It contains placeholders only; add secrets locally if desired.' -ForegroundColor Yellow
} else {
    Write-Host '.env already exists; it was not overwritten.' -ForegroundColor DarkYellow
}

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($null -eq $ollama) {
    Write-Warning 'Ollama was not found on PATH. Install it separately for the local Qwen fallback.'
} else {
    Write-Host 'Ollama detected. The configured model is checked only when ASHER runs.' -ForegroundColor Green
}

Write-Host ''
Write-Host 'ASHER setup is ready.' -ForegroundColor Green
Write-Host 'Start safely with: .\run_asher.ps1 -Mode text'
Write-Host 'Run tests with:    .\run_tests.ps1'
