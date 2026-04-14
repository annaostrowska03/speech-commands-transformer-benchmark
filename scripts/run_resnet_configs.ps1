param(
    [string]$PythonExe = ".\.venv311\Scripts\python.exe",
    [string]$ConfigDir = ".\configs",
    [string]$OutputDir = ".\outputs",
    [int[]]$Seeds = @(42, 123, 2026, 2137),
    [string[]]$ConfigNames = @(
        "resnet18_full_baseline.yaml",
        "resnet18_full_nmels128.yaml",
        "resnet18_full_optimizer_sgd.yaml",
        "resnet18_full_dropout_p03.yaml",
        "resnet18_full_dropout_p05.yaml",
        "resnet18_full_specaugment.yaml",
        "resnet18_full_balancing_loss.yaml",
        "resnet18_full_balancing_undersample.yaml",
        "resnet18_full_balancing_loss_undersample.yaml"
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

    Write-Host "Running config: $configName | seeds: $($Seeds -join ', ')" -ForegroundColor Cyan
    $args = @("src/train.py", "--config", $configPath, "--seeds") + ($Seeds | ForEach-Object { "$_" })
    & $PythonExe @args
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed for config: $configName"
    }
}

Write-Host "All real experiments completed." -ForegroundColor Green
