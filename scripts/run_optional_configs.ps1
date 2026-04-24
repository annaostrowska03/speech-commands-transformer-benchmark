param(
    [string]$PythonExe = ".\.venv311\Scripts\python.exe",
    [string]$ConfigDir = ".\configs",
    [string]$OutputDir = ".\outputs",
    [int[]]$Seeds = @(42, 123, 2026, 2137),
    [string[]]$ConfigNames = @(
        "mobilenetv2_full_baseline.yaml",
        "mobilenetv2_full_specaugment.yaml",
        "resnet18_full_batch32.yaml",
        "resnet18_full_lr0003.yaml",
        "resnet18_full_unknown_detector.yaml"
    )
)

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path $ConfigDir)) {
    throw "Config directory not found: $ConfigDir"
}

if (-not (Test-Path $OutputDir)) {
    New-Item -Path $OutputDir -ItemType Directory -Force | Out-Null
}

foreach ($configName in $ConfigNames) {
    $configPath = Join-Path $ConfigDir $configName
    if (-not (Test-Path $configPath)) {
        throw "Config file not found: $configPath"
    }

    Write-Host "Running OPTIONAL config: $configName | seeds: $($Seeds -join ', ')" -ForegroundColor Yellow
    $args = @("src/train.py", "--config", $configPath, "--seeds") + ($Seeds | ForEach-Object { "$_" })
    & $PythonExe @args
    if ($LASTEXITCODE -ne 0) {
        throw "Optional experiment failed for config: $configName"
    }
}

Write-Host "All optional experiments completed." -ForegroundColor Green
