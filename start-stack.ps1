param(
    [int]$ApiPort = 8000,
    [int]$WebPort = 5173,
    [int]$PollIntervalSeconds = 600,
    [string]$ModelName = "",
    [switch]$NoPoller
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$SrcRoot = Join-Path $Root "src"
$WebRoot = Join-Path $Root "web"
$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($PreviousPythonPath) { "$SrcRoot;$PreviousPythonPath" } else { $SrcRoot }

$script:Children = [System.Collections.Generic.List[object]]::new()
$script:Stopping = $false

function Resolve-Executable {
    param([Parameter(Mandatory = $true)][string]$Name)
    $command = Get-Command $Name -ErrorAction Stop
    return $command.Source
}

function Resolve-FirstExecutable {
    param([Parameter(Mandatory = $true)][string[]]$Names)

    foreach ($name in $Names) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            return $command.Source
        }
    }

    throw "None of these executables were found: $($Names -join ', ')"
}

function Start-StackProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    Write-Host "Starting $Name..." -ForegroundColor Cyan
    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -NoNewWindow `
        -PassThru
    $script:Children.Add([pscustomobject]@{ Name = $Name; Process = $process }) | Out-Null
    Write-Host "  $Name pid=$($process.Id)" -ForegroundColor DarkGray
}

function Get-ChildProcessIds {
    param([Parameter(Mandatory = $true)][int]$ParentProcessId)
    Get-CimInstance Win32_Process -Filter "ParentProcessId = $ParentProcessId" |
        Select-Object -ExpandProperty ProcessId
}

function Stop-ProcessTree {
    param([Parameter(Mandatory = $true)][int]$ProcessId)

    foreach ($childId in Get-ChildProcessIds -ParentProcessId $ProcessId) {
        Stop-ProcessTree -ProcessId ([int]$childId)
    }

    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return
    }

    try {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Warning "Failed to stop process ${ProcessId}: $($_.Exception.Message)"
    }
}

function Stop-Stack {
    if ($script:Stopping) {
        return
    }
    $script:Stopping = $true
    Write-Host ""
    Write-Host "Stopping Kalorie2 stack..." -ForegroundColor Yellow

    foreach ($child in @($script:Children)) {
        $process = $child.Process
        if ($null -ne (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
            Write-Host "  stopping $($child.Name) pid=$($process.Id)" -ForegroundColor DarkGray
            Stop-ProcessTree -ProcessId $process.Id
        }
    }

    $env:PYTHONPATH = $PreviousPythonPath
}

try {
    $python = Resolve-Executable "python"
    $npm = Resolve-FirstExecutable @("npm.cmd", "npm")

    Start-StackProcess `
        -Name "api" `
        -FilePath $python `
        -ArgumentList @(
            "-m", "uvicorn",
            "kalorie2.webapi.main:create_app",
            "--factory",
            "--host", "127.0.0.1",
            "--port", "$ApiPort"
        ) `
        -WorkingDirectory $Root

    Start-StackProcess `
        -Name "web" `
        -FilePath $npm `
        -ArgumentList @(
            "run", "dev", "--",
            "--host", "127.0.0.1",
            "--port", "$WebPort",
            "--strictPort"
        ) `
        -WorkingDirectory $WebRoot

    if (-not $NoPoller) {
        $pollerArgs = @(
            "-m",
            "kalorie2.market_poller",
            "loop",
            "--interval-seconds",
            "$PollIntervalSeconds"
        )
        if ($ModelName.Trim()) {
            $pollerArgs += @("--model-name", $ModelName)
        }
        Start-StackProcess `
            -Name "poller" `
            -FilePath $python `
            -ArgumentList $pollerArgs `
            -WorkingDirectory $Root
    }

    Write-Host ""
    Write-Host "Kalorie2 stack is running." -ForegroundColor Green
    Write-Host "  API: http://127.0.0.1:$ApiPort" -ForegroundColor DarkGray
    Write-Host "  Web: http://127.0.0.1:$WebPort" -ForegroundColor DarkGray
    if (-not $NoPoller) {
        Write-Host "  Poller: every $PollIntervalSeconds seconds" -ForegroundColor DarkGray
    }
    Write-Host "Press Ctrl+C to stop all processes." -ForegroundColor DarkGray

    while ($true) {
        Start-Sleep -Seconds 1
        foreach ($child in @($script:Children)) {
            if ($child.Process.HasExited) {
                throw "$($child.Name) exited with code $($child.Process.ExitCode)"
            }
        }
    }
} finally {
    Stop-Stack
}
