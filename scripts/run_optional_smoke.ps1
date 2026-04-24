param(
    [string]$PythonExe = ".\.venv311\Scripts\python.exe",
    [string]$ConfigDir = ".\configs",
    [string[]]$ConfigNames = @(
        "mobilenetv2_smoke_baseline.yaml",
        "mobilenetv2_smoke_specaugment.yaml",
        "resnet18_smoke_unknown_detector.yaml"
    )
)

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path $ConfigDir)) {
    throw "Config directory not found: $ConfigDir"
}

foreach ($configName in $ConfigNames) {
    $configPath = Join-Path $ConfigDir $configName
    if (-not (Test-Path $configPath)) {
        throw "Config file not found: $configPath"
    }

    Write-Host "Running OPTIONAL smoke config: $configName" -ForegroundColor Yellow
    & $PythonExe "src/train.py" "--config" $configPath
    if ($LASTEXITCODE -ne 0) {
        throw "Optional smoke experiment failed for config: $configName"
    }
}

Write-Host "All optional smoke experiments completed." -ForegroundColor Green
