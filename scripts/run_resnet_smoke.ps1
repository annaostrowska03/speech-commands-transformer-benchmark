param(
    [string]$PythonExe = ".\.venv311\Scripts\python.exe",
    [int]$Epochs = 1,
    [int]$BatchSize = 64,
    [int]$NumWorkers = 0
)

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

function Run-Experiment {
    param(
        [string]$Name,
        [string[]]$ExtraArgs = @()
    )

    Write-Host "Running experiment: $Name" -ForegroundColor Cyan

    $args = @(
        "src/train.py",
        "--epochs", $Epochs,
        "--batch_size", $BatchSize,
        "--num_workers", $NumWorkers,
        "--disable_mlflow",
        "--experiment_name", $Name,
        "--checkpoint_path", "outputs/$Name.pt"
    ) + $ExtraArgs

    & $PythonExe @args
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $Name"
    }
}

Run-Experiment -Name "resnet18_smoke_baseline"
Run-Experiment -Name "resnet18_smoke_balancing_loss" -ExtraArgs @("--balancing", "loss")
Run-Experiment -Name "resnet18_smoke_balancing_undersample" -ExtraArgs @("--balancing", "undersample", "--unknown_keep_prob", "0.35")
Run-Experiment -Name "resnet18_smoke_balancing_loss_undersample" -ExtraArgs @("--balancing", "loss+undersample", "--unknown_keep_prob", "0.35")

Write-Host "All smoke experiments completed." -ForegroundColor Green
