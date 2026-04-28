param(
    [string]$PythonExe = ".\.venv311\Scripts\python.exe",
    [switch]$IncludeOptional,
    [switch]$Force
)

function Write-Section {
    param([string]$Title)

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
}

function Run-Config {
    param(
        [string]$Trainer,
        [string]$ConfigPath,
        [string]$ExperimentName,
        [string]$SummaryPath
    )

    if (Test-Path $ConfigPath) {
        if ((Test-Path $SummaryPath) -and (-not $Force)) {
            Write-Host "Skipping completed experiment: $ExperimentName" -ForegroundColor DarkYellow
            return
        }
        Write-Host "Running config: $ConfigPath" -ForegroundColor Yellow
        & $PythonExe $Trainer "--config" $ConfigPath
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment failed for config: $ConfigPath"
        }
    }
    else {
        Write-Host "Skipping missing config: $ConfigPath" -ForegroundColor DarkYellow
    }
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

Write-Host "Force mode enabled: $($Force.IsPresent)"
Write-Host "Optional experiments enabled: $($IncludeOptional.IsPresent)"

Write-Section "Project readiness check"
& $PythonExe "scripts/check_project_ready.py"
if ($LASTEXITCODE -ne 0) {
    throw "Project readiness check failed."
}

Write-Section "ResNet full experiments"
Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_baseline.yaml" -ExperimentName "resnet18_full_baseline" -SummaryPath "outputs/resnet18/resnet18_full_baseline/summary_all_seeds.json"
Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_nmels128.yaml" -ExperimentName "resnet18_full_nmels128" -SummaryPath "outputs/resnet18/resnet18_full_nmels128/summary_all_seeds.json"
Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_optimizer_sgd.yaml" -ExperimentName "resnet18_full_optimizer_sgd" -SummaryPath "outputs/resnet18/resnet18_full_optimizer_sgd/summary_all_seeds.json"
Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_dropout_p03.yaml" -ExperimentName "resnet18_full_dropout_p03" -SummaryPath "outputs/resnet18/resnet18_full_dropout_p03/summary_all_seeds.json"
Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_dropout_p05.yaml" -ExperimentName "resnet18_full_dropout_p05" -SummaryPath "outputs/resnet18/resnet18_full_dropout_p05/summary_all_seeds.json"
Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_specaugment.yaml" -ExperimentName "resnet18_full_specaugment" -SummaryPath "outputs/resnet18/resnet18_full_specaugment/summary_all_seeds.json"
Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_balancing_loss.yaml" -ExperimentName "resnet18_full_balancing_loss" -SummaryPath "outputs/resnet18/resnet18_full_balancing_loss/summary_all_seeds.json"
Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_balancing_undersample.yaml" -ExperimentName "resnet18_full_balancing_undersample" -SummaryPath "outputs/resnet18/resnet18_full_balancing_undersample/summary_all_seeds.json"
Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_balancing_loss_undersample.yaml" -ExperimentName "resnet18_full_balancing_loss_undersample" -SummaryPath "outputs/resnet18/resnet18_full_balancing_loss_undersample/summary_all_seeds.json"

Write-Section "AST full experiments"
Run-Config -Trainer "src/train_ast.py" -ConfigPath "configs/ast_full_baseline.yaml" -ExperimentName "ast_full_baseline" -SummaryPath "outputs/ast/ast_full_baseline/summary_all_seeds.json"
Run-Config -Trainer "src/train_ast.py" -ConfigPath "configs/ast_full_frozen_backbone.yaml" -ExperimentName "ast_full_frozen_backbone" -SummaryPath "outputs/ast/ast_full_frozen_backbone/summary_all_seeds.json"
Run-Config -Trainer "src/train_ast.py" -ConfigPath "configs/ast_full_dropout_p03.yaml" -ExperimentName "ast_full_dropout_p03" -SummaryPath "outputs/ast/ast_full_dropout_p03/summary_all_seeds.json"
Run-Config -Trainer "src/train_ast.py" -ConfigPath "configs/ast_full_balancing_loss.yaml" -ExperimentName "ast_full_balancing_loss" -SummaryPath "outputs/ast/ast_full_balancing_loss/summary_all_seeds.json"
Run-Config -Trainer "src/train_ast.py" -ConfigPath "configs/ast_full_balancing_undersample.yaml" -ExperimentName "ast_full_balancing_undersample" -SummaryPath "outputs/ast/ast_full_balancing_undersample/summary_all_seeds.json"
Run-Config -Trainer "src/train_ast.py" -ConfigPath "configs/ast_full_balancing_loss_undersample.yaml" -ExperimentName "ast_full_balancing_loss_undersample" -SummaryPath "outputs/ast/ast_full_balancing_loss_undersample/summary_all_seeds.json"

if ($IncludeOptional) {
    Write-Section "Optional full experiments"
    Run-Config -Trainer "src/train.py" -ConfigPath "configs/mobilenetv2_full_baseline.yaml" -ExperimentName "mobilenetv2_full_baseline" -SummaryPath "outputs/mobilenetv2/mobilenetv2_full_baseline/summary_all_seeds.json"
    Run-Config -Trainer "src/train.py" -ConfigPath "configs/mobilenetv2_full_specaugment.yaml" -ExperimentName "mobilenetv2_full_specaugment" -SummaryPath "outputs/mobilenetv2/mobilenetv2_full_specaugment/summary_all_seeds.json"
    Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_baseline_no_audio_tweaks.yaml" -ExperimentName "resnet18_full_baseline_no_audio_tweaks" -SummaryPath "outputs/resnet18_no_audio_tweaks/resnet18_full_baseline_no_audio_tweaks/summary_all_seeds.json"
    Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_batch32.yaml" -ExperimentName "resnet18_full_batch32" -SummaryPath "outputs/resnet18/resnet18_full_batch32/summary_all_seeds.json"
    Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_lr0003.yaml" -ExperimentName "resnet18_full_lr0003" -SummaryPath "outputs/resnet18/resnet18_full_lr0003/summary_all_seeds.json"
    Run-Config -Trainer "src/train.py" -ConfigPath "configs/resnet18_full_unknown_detector.yaml" -ExperimentName "resnet18_full_unknown_detector" -SummaryPath "outputs/resnet18/resnet18_full_unknown_detector/summary_all_seeds.json"
}

Write-Section "Reporting"
& $PythonExe "src/reporting.py" "--outputs_dir" "outputs" "--analysis_dir" "outputs/analysis"
if ($LASTEXITCODE -ne 0) {
    throw "Reporting failed."
}

Write-Host ""
Write-Host "All full experiments completed. Analysis outputs: outputs/analysis" -ForegroundColor Green
