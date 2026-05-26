[CmdletBinding()]
param(
    [string]$BackendHost = "127.0.0.1",
    [int]$BackendPort = 8000,
    [string]$FrontendHost = "127.0.0.1",
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
$webRoot = Join-Path $projectRoot "web"

if (-not (Test-Path -Path $webRoot -PathType Container)) {
    throw "[run_webapp] Missing frontend directory: $webRoot"
}

if (-not (Test-Path -Path (Join-Path $webRoot "node_modules") -PathType Container)) {
    Write-Host "[run_webapp] Installing frontend dependencies..."
    Push-Location $webRoot
    try {
        npm install
    }
    finally {
        Pop-Location
    }
}

$backendArgs = @(
    "-m",
    "uvicorn",
    "kalorie.webapi.main:create_app",
    "--factory",
    "--host",
    $BackendHost,
    "--port",
    "$BackendPort"
)

Write-Host "[run_webapp] Starting backend on http://$BackendHost`:$BackendPort"
$backendProcess = Start-Process -FilePath "python" -ArgumentList $backendArgs -WorkingDirectory $projectRoot -PassThru

try {
    Start-Sleep -Seconds 2
    if ($backendProcess.HasExited) {
        throw "[run_webapp] Backend exited early with code $($backendProcess.ExitCode)."
    }

    if ([string]::IsNullOrWhiteSpace($env:VITE_API_BASE_URL)) {
        $env:VITE_API_BASE_URL = "http://$BackendHost`:$BackendPort"
    }

    Write-Host "[run_webapp] Starting frontend on http://$FrontendHost`:$FrontendPort"
    Write-Host "[run_webapp] Using VITE_API_BASE_URL=$($env:VITE_API_BASE_URL)"

    Push-Location $webRoot
    try {
        npm run dev -- --host $FrontendHost --port $FrontendPort
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force
        Write-Host "[run_webapp] Stopped backend ($($backendProcess.Id))."
    }
}
