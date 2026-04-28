param(
    [string]$PythonExe = ".\.venv311\Scripts\python.exe",
    [string]$ConfigDir = ".\configs",
    [string[]]$ConfigNames = @(
        "ast_full_baseline.yaml",
        "ast_full_frozen_backbone.yaml",
        "ast_full_dropout_p03.yaml",
        "ast_full_balancing_loss.yaml",
        "ast_full_balancing_undersample.yaml",
        "ast_full_balancing_loss_undersample.yaml"
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

    Write-Host "Running AST config: $configName" -ForegroundColor Cyan
    & $PythonExe "src/train_ast.py" "--config" $configPath
    if ($LASTEXITCODE -ne 0) {
        throw "AST experiment failed for config: $configName"
    }
}

Write-Host "All AST experiments completed." -ForegroundColor Green
