#!/usr/bin/env bash
set -e

INCLUDE_OPTIONAL=false
FORCE=false

for arg in "$@"; do
    case "$arg" in
        --include-optional)
            INCLUDE_OPTIONAL=true
            ;;
        --force)
            FORCE=true
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash scripts/run_all_full_experiments.sh [--include-optional] [--force]"
            exit 1
            ;;
    esac
done

run_config() {
    local trainer="$1"
    local config_path="$2"
    local experiment_name="$3"
    local summary_path="$4"

    if [[ -f "$config_path" ]]; then
        if [[ -f "$summary_path" && "$FORCE" != true ]]; then
            echo "Skipping completed experiment: $experiment_name"
            return
        fi
        echo "Running config: $config_path"
        python "$trainer" --config "$config_path"
    else
        echo "Skipping missing config: $config_path"
    fi
}

echo "Force mode enabled: $FORCE"
echo "Optional experiments enabled: $INCLUDE_OPTIONAL"

echo
echo "========================================"
echo "Project readiness check"
echo "========================================"
python scripts/check_project_ready.py

echo
echo "========================================"
echo "ResNet full experiments"
echo "========================================"
run_config "src/train.py" "configs/resnet18_full_baseline.yaml" "resnet18_full_baseline" "outputs/resnet18/resnet18_full_baseline/summary_all_seeds.json"
run_config "src/train.py" "configs/resnet18_full_nmels128.yaml" "resnet18_full_nmels128" "outputs/resnet18/resnet18_full_nmels128/summary_all_seeds.json"
run_config "src/train.py" "configs/resnet18_full_optimizer_sgd.yaml" "resnet18_full_optimizer_sgd" "outputs/resnet18/resnet18_full_optimizer_sgd/summary_all_seeds.json"
run_config "src/train.py" "configs/resnet18_full_dropout_p03.yaml" "resnet18_full_dropout_p03" "outputs/resnet18/resnet18_full_dropout_p03/summary_all_seeds.json"
run_config "src/train.py" "configs/resnet18_full_dropout_p05.yaml" "resnet18_full_dropout_p05" "outputs/resnet18/resnet18_full_dropout_p05/summary_all_seeds.json"
run_config "src/train.py" "configs/resnet18_full_specaugment.yaml" "resnet18_full_specaugment" "outputs/resnet18/resnet18_full_specaugment/summary_all_seeds.json"
run_config "src/train.py" "configs/resnet18_full_balancing_loss.yaml" "resnet18_full_balancing_loss" "outputs/resnet18/resnet18_full_balancing_loss/summary_all_seeds.json"
run_config "src/train.py" "configs/resnet18_full_balancing_undersample.yaml" "resnet18_full_balancing_undersample" "outputs/resnet18/resnet18_full_balancing_undersample/summary_all_seeds.json"
run_config "src/train.py" "configs/resnet18_full_balancing_loss_undersample.yaml" "resnet18_full_balancing_loss_undersample" "outputs/resnet18/resnet18_full_balancing_loss_undersample/summary_all_seeds.json"

echo
echo "========================================"
echo "AST full experiments"
echo "========================================"
run_config "src/train_ast.py" "configs/ast_full_baseline.yaml" "ast_full_baseline" "outputs/ast/ast_full_baseline/summary_all_seeds.json"
run_config "src/train_ast.py" "configs/ast_full_frozen_backbone.yaml" "ast_full_frozen_backbone" "outputs/ast/ast_full_frozen_backbone/summary_all_seeds.json"
run_config "src/train_ast.py" "configs/ast_full_dropout_p03.yaml" "ast_full_dropout_p03" "outputs/ast/ast_full_dropout_p03/summary_all_seeds.json"
run_config "src/train_ast.py" "configs/ast_full_balancing_loss.yaml" "ast_full_balancing_loss" "outputs/ast/ast_full_balancing_loss/summary_all_seeds.json"
run_config "src/train_ast.py" "configs/ast_full_balancing_undersample.yaml" "ast_full_balancing_undersample" "outputs/ast/ast_full_balancing_undersample/summary_all_seeds.json"
run_config "src/train_ast.py" "configs/ast_full_balancing_loss_undersample.yaml" "ast_full_balancing_loss_undersample" "outputs/ast/ast_full_balancing_loss_undersample/summary_all_seeds.json"

if [[ "$INCLUDE_OPTIONAL" == true ]]; then
    echo
    echo "========================================"
    echo "Optional full experiments"
    echo "========================================"
    run_config "src/train.py" "configs/mobilenetv2_full_baseline.yaml" "mobilenetv2_full_baseline" "outputs/mobilenetv2/mobilenetv2_full_baseline/summary_all_seeds.json"
    run_config "src/train.py" "configs/mobilenetv2_full_specaugment.yaml" "mobilenetv2_full_specaugment" "outputs/mobilenetv2/mobilenetv2_full_specaugment/summary_all_seeds.json"
    run_config "src/train.py" "configs/resnet18_full_baseline_no_audio_tweaks.yaml" "resnet18_full_baseline_no_audio_tweaks" "outputs/resnet18_no_audio_tweaks/resnet18_full_baseline_no_audio_tweaks/summary_all_seeds.json"
    run_config "src/train.py" "configs/resnet18_full_batch32.yaml" "resnet18_full_batch32" "outputs/resnet18/resnet18_full_batch32/summary_all_seeds.json"
    run_config "src/train.py" "configs/resnet18_full_lr0003.yaml" "resnet18_full_lr0003" "outputs/resnet18/resnet18_full_lr0003/summary_all_seeds.json"
    run_config "src/train.py" "configs/resnet18_full_unknown_detector.yaml" "resnet18_full_unknown_detector" "outputs/resnet18/resnet18_full_unknown_detector/summary_all_seeds.json"
fi

echo
echo "========================================"
echo "Reporting"
echo "========================================"
python src/reporting.py --outputs_dir outputs --analysis_dir outputs/analysis

echo
echo "All full experiments completed. Analysis outputs: outputs/analysis"
