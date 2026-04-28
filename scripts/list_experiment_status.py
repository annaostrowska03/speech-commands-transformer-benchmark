from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    (
        "Main ResNet",
        "resnet18_full_baseline",
        "configs/resnet18_full_baseline.yaml",
        "outputs/resnet18/resnet18_full_baseline/summary_all_seeds.json",
    ),
    (
        "Main ResNet",
        "resnet18_full_nmels128",
        "configs/resnet18_full_nmels128.yaml",
        "outputs/resnet18/resnet18_full_nmels128/summary_all_seeds.json",
    ),
    (
        "Main ResNet",
        "resnet18_full_optimizer_sgd",
        "configs/resnet18_full_optimizer_sgd.yaml",
        "outputs/resnet18/resnet18_full_optimizer_sgd/summary_all_seeds.json",
    ),
    (
        "Main ResNet",
        "resnet18_full_dropout_p03",
        "configs/resnet18_full_dropout_p03.yaml",
        "outputs/resnet18/resnet18_full_dropout_p03/summary_all_seeds.json",
    ),
    (
        "Main ResNet",
        "resnet18_full_dropout_p05",
        "configs/resnet18_full_dropout_p05.yaml",
        "outputs/resnet18/resnet18_full_dropout_p05/summary_all_seeds.json",
    ),
    (
        "Main ResNet",
        "resnet18_full_specaugment",
        "configs/resnet18_full_specaugment.yaml",
        "outputs/resnet18/resnet18_full_specaugment/summary_all_seeds.json",
    ),
    (
        "Main ResNet",
        "resnet18_full_balancing_loss",
        "configs/resnet18_full_balancing_loss.yaml",
        "outputs/resnet18/resnet18_full_balancing_loss/summary_all_seeds.json",
    ),
    (
        "Main ResNet",
        "resnet18_full_balancing_undersample",
        "configs/resnet18_full_balancing_undersample.yaml",
        "outputs/resnet18/resnet18_full_balancing_undersample/summary_all_seeds.json",
    ),
    (
        "Main ResNet",
        "resnet18_full_balancing_loss_undersample",
        "configs/resnet18_full_balancing_loss_undersample.yaml",
        "outputs/resnet18/resnet18_full_balancing_loss_undersample/summary_all_seeds.json",
    ),
    (
        "AST",
        "ast_full_baseline",
        "configs/ast_full_baseline.yaml",
        "outputs/ast/ast_full_baseline/summary_all_seeds.json",
    ),
    (
        "AST",
        "ast_full_frozen_backbone",
        "configs/ast_full_frozen_backbone.yaml",
        "outputs/ast/ast_full_frozen_backbone/summary_all_seeds.json",
    ),
    (
        "AST",
        "ast_full_dropout_p03",
        "configs/ast_full_dropout_p03.yaml",
        "outputs/ast/ast_full_dropout_p03/summary_all_seeds.json",
    ),
    (
        "AST",
        "ast_full_balancing_loss",
        "configs/ast_full_balancing_loss.yaml",
        "outputs/ast/ast_full_balancing_loss/summary_all_seeds.json",
    ),
    (
        "AST",
        "ast_full_balancing_undersample",
        "configs/ast_full_balancing_undersample.yaml",
        "outputs/ast/ast_full_balancing_undersample/summary_all_seeds.json",
    ),
    (
        "AST",
        "ast_full_balancing_loss_undersample",
        "configs/ast_full_balancing_loss_undersample.yaml",
        "outputs/ast/ast_full_balancing_loss_undersample/summary_all_seeds.json",
    ),
    (
        "Optional",
        "mobilenetv2_full_baseline",
        "configs/mobilenetv2_full_baseline.yaml",
        "outputs/mobilenetv2/mobilenetv2_full_baseline/summary_all_seeds.json",
    ),
    (
        "Optional",
        "mobilenetv2_full_specaugment",
        "configs/mobilenetv2_full_specaugment.yaml",
        "outputs/mobilenetv2/mobilenetv2_full_specaugment/summary_all_seeds.json",
    ),
    (
        "Optional",
        "resnet18_full_baseline_no_audio_tweaks",
        "configs/resnet18_full_baseline_no_audio_tweaks.yaml",
        "outputs/resnet18_no_audio_tweaks/resnet18_full_baseline_no_audio_tweaks/summary_all_seeds.json",
    ),
    (
        "Optional",
        "resnet18_full_batch32",
        "configs/resnet18_full_batch32.yaml",
        "outputs/resnet18/resnet18_full_batch32/summary_all_seeds.json",
    ),
    (
        "Optional",
        "resnet18_full_lr0003",
        "configs/resnet18_full_lr0003.yaml",
        "outputs/resnet18/resnet18_full_lr0003/summary_all_seeds.json",
    ),
    (
        "Optional",
        "resnet18_full_unknown_detector",
        "configs/resnet18_full_unknown_detector.yaml",
        "outputs/resnet18/resnet18_full_unknown_detector/summary_all_seeds.json",
    ),
]


def status_for(config_exists, summary_exists):
    if summary_exists:
        return "DONE"
    if config_exists:
        return "READY_TO_RUN"
    return "MISSING_CONFIG"


def format_bool(value):
    return "yes" if value else "no"


def print_table(rows):
    headers = ["group", "experiment_name", "config_exists", "summary_exists", "status"]
    widths = [
        max(len(str(row[index])) for row in [headers] + rows)
        for index in range(len(headers))
    ]

    def format_row(row):
        return "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))

    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in rows:
        print(format_row(row))


def main():
    rows = []
    totals = {
        "DONE": 0,
        "READY_TO_RUN": 0,
        "MISSING_CONFIG": 0,
    }

    for group, experiment_name, config_path, summary_path in EXPERIMENTS:
        config_exists = (ROOT_DIR / config_path).exists()
        summary_exists = (ROOT_DIR / summary_path).exists()
        status = status_for(config_exists, summary_exists)
        totals[status] += 1
        rows.append(
            [
                group,
                experiment_name,
                format_bool(config_exists),
                format_bool(summary_exists),
                status,
            ]
        )

    print("Experiment status")
    print(f"Repository root: {ROOT_DIR}")
    print()
    print_table(rows)
    print()
    print("Totals")
    print(f"- done count: {totals['DONE']}")
    print(f"- ready_to_run count: {totals['READY_TO_RUN']}")
    print(f"- missing_config count: {totals['MISSING_CONFIG']}")


if __name__ == "__main__":
    main()
