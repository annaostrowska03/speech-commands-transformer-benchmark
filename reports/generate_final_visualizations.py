import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/deeplearning_transformers_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
VIS_DIR = ROOT / "reports" / "visualizations"

CLASS_NAMES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]

AST_ORDER = [
    "ast_full_baseline",
    "ast_full_frozen_backbone",
    "ast_full_dropout_p03",
    "ast_full_balancing_loss",
    "ast_full_balancing_undersample",
    "ast_full_balancing_loss_undersample",
]

STRATEGY_EXPERIMENTS = [
    "resnet18_full_balancing_loss",
    "resnet18_full_balancing_undersample",
    "resnet18_full_balancing_loss_undersample",
    "resnet18_full_unknown_detector",
    "ast_full_balancing_loss",
    "ast_full_balancing_undersample",
    "ast_full_balancing_loss_undersample",
]

ARCH_COLORS = {
    "resnet18": "#4C72B0",
    "mobilenetv2": "#DD8452",
    "AST": "#55A868",
    "resnet18_no_audio_tweaks": "#C44E52",
}


class PlotLog:
    def __init__(self):
        self.generated = []
        self.skipped = []
        self.missing = []

    def warn(self, figure_name, message):
        print(f"WARNING [{figure_name}]: {message}")
        self.missing.append((figure_name, message))

    def skip(self, figure_name, message):
        print(f"SKIP [{figure_name}]: {message}")
        self.skipped.append((figure_name, message))

    def made(self, path):
        print(f"GENERATED: {path}")
        self.generated.append(str(path))


def read_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def nested(payload, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def as_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def metric_dict(value):
    if isinstance(value, dict):
        return as_float(value.get("mean")), as_float(value.get("std"))
    return as_float(value), None


def infer_architecture(summary_path, payload):
    if payload.get("model"):
        return payload["model"]
    if "outputs/ast/" in summary_path.as_posix() or str(summary_path).startswith(str(OUTPUTS / "ast")):
        return "AST"
    return payload.get("model_name") or summary_path.parent.parent.name


def collect_experiment_rows():
    rows = []
    aggregate_bases = set()
    for summary_path in sorted(OUTPUTS.glob("*/*/summary_all_seeds.json")):
        payload = read_json(summary_path)
        aggregate_bases.add(summary_path.parent.resolve())
        architecture = infer_architecture(summary_path, payload)
        experiment_name = payload.get("experiment_name") or summary_path.parent.name

        if "aggregate" in payload:
            test_acc_mean = as_float(nested(payload, "aggregate", "test", "acc", "mean"))
            test_acc_std = as_float(nested(payload, "aggregate", "test", "acc", "std"))
            test_f1_mean = as_float(nested(payload, "aggregate", "test", "macro_f1", "mean"))
            test_f1_std = as_float(nested(payload, "aggregate", "test", "macro_f1", "std"))
            val_acc_mean = as_float(nested(payload, "aggregate", "best_validation", "acc", "mean"))
            val_acc_std = as_float(nested(payload, "aggregate", "best_validation", "acc", "std"))
            train_time_mean = as_float(nested(payload, "aggregate", "timing", "total_training_time_sec", "mean"))
            latency_mean = as_float(nested(payload, "aggregate", "test", "inference_latency_ms", "mean"))
            trainable_params = as_float(nested(payload, "aggregate", "model_parameters", "trainable", "mean"))
            seeds = payload.get("seeds", [])
        else:
            test_acc_mean, test_acc_std = metric_dict(payload.get("test_accuracy"))
            test_f1_mean, test_f1_std = metric_dict(payload.get("test_f1_macro"))
            val_values = [
                as_float(run.get("val_accuracy_best_epoch"))
                for run in payload.get("runs", [])
                if as_float(run.get("val_accuracy_best_epoch")) is not None
            ]
            train_values = [
                as_float(run.get("training_time_seconds"))
                for run in payload.get("runs", [])
                if as_float(run.get("training_time_seconds")) is not None
            ]
            latency_values = [
                as_float(run.get("inference_latency_ms"))
                for run in payload.get("runs", [])
                if as_float(run.get("inference_latency_ms")) is not None
            ]
            params_values = [
                as_float(run.get("trainable_params"))
                for run in payload.get("runs", [])
                if as_float(run.get("trainable_params")) is not None
            ]
            val_acc_mean, val_acc_std = mean_std(val_values)
            train_time_mean, _ = mean_std(train_values)
            latency_mean, _ = mean_std(latency_values)
            trainable_params, _ = mean_std(params_values)
            seeds = payload.get("seeds", [])

        rows.append(
            {
                "architecture": architecture,
                "experiment_name": experiment_name,
                "summary_path": summary_path,
                "seeds": seeds,
                "test_acc_mean": test_acc_mean,
                "test_acc_std": test_acc_std,
                "test_f1_mean": test_f1_mean,
                "test_f1_std": test_f1_std,
                "val_acc_mean": val_acc_mean,
                "val_acc_std": val_acc_std,
                "train_time_mean": train_time_mean,
                "latency_mean": latency_mean,
                "trainable_params": trainable_params,
            }
        )

    seed_groups = {}
    for summary_path in sorted(OUTPUTS.glob("*/*/summary_seed*.json")):
        run_dir = summary_path.parent
        base_name = strip_seed_suffix(run_dir.name)
        aggregate_dir = run_dir.parent / base_name
        if aggregate_dir.resolve() in aggregate_bases:
            continue
        seed_groups.setdefault((run_dir.parent, base_name), []).append(summary_path)

    for (model_dir, base_name), summary_paths in sorted(seed_groups.items(), key=lambda item: str(item[0])):
        row = row_from_seed_summaries(model_dir, base_name, summary_paths)
        if row is not None:
            rows.append(row)
    return rows


def strip_seed_suffix(name):
    parts = name.rsplit("_seed", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


def row_from_seed_summaries(model_dir, base_name, summary_paths):
    payloads = []
    for path in summary_paths:
        payload = read_json(path)
        payloads.append((path, payload))
    if not payloads:
        return None

    first_payload = payloads[0][1]
    architecture = first_payload.get("model") or first_payload.get("model_name") or model_dir.name
    if model_dir.name == "ast":
        architecture = "AST"

    test_acc_values = [as_float(payload.get("test_accuracy") or nested(payload, "test", "acc")) for _, payload in payloads]
    test_f1_values = [as_float(payload.get("test_f1_macro") or nested(payload, "test", "macro_f1")) for _, payload in payloads]
    val_acc_values = [
        as_float(payload.get("val_accuracy_best_epoch") or nested(payload, "best_validation", "acc"))
        for _, payload in payloads
    ]
    train_time_values = [
        as_float(payload.get("training_time_seconds") or nested(payload, "timing", "total_training_time_sec"))
        for _, payload in payloads
    ]
    latency_values = [
        as_float(payload.get("inference_latency_ms") or nested(payload, "test", "inference_latency_ms"))
        for _, payload in payloads
    ]
    params_values = [
        as_float(payload.get("trainable_params") or nested(payload, "model_parameters", "trainable"))
        for _, payload in payloads
    ]
    seeds = [payload.get("seed") for _, payload in payloads if payload.get("seed") is not None]

    test_acc_mean, test_acc_std = mean_std(test_acc_values)
    test_f1_mean, test_f1_std = mean_std(test_f1_values)
    val_acc_mean, val_acc_std = mean_std(val_acc_values)
    train_time_mean, _ = mean_std(train_time_values)
    latency_mean, _ = mean_std(latency_values)
    trainable_params, _ = mean_std(params_values)

    return {
        "architecture": architecture,
        "experiment_name": base_name,
        "summary_path": model_dir / base_name / "summary_all_seeds.json",
        "seed_summary_paths": summary_paths,
        "seeds": seeds,
        "test_acc_mean": test_acc_mean,
        "test_acc_std": test_acc_std,
        "test_f1_mean": test_f1_mean,
        "test_f1_std": test_f1_std,
        "val_acc_mean": val_acc_mean,
        "val_acc_std": val_acc_std,
        "train_time_mean": train_time_mean,
        "latency_mean": latency_mean,
        "trainable_params": trainable_params,
    }


def mean_std(values):
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def percent(value):
    return value * 100.0 if value is not None and value <= 1.5 else value


def minutes(value):
    return value / 60.0 if value is not None else None


def short_label(name):
    return (
        name.replace("resnet18_full_", "")
        .replace("mobilenetv2_full_", "mobilenetv2_")
        .replace("ast_full_", "")
        .replace("_", "\n")
    )


def output_path(filename):
    path = VIS_DIR / filename
    if not path.exists():
        return path
    final_path = path.with_name(f"{path.stem}_final{path.suffix}")
    print(f"NOTICE: {path.name} already exists; writing improved version to {final_path.name}")
    return final_path


def source_desc(row):
    if row.get("seed_summary_paths"):
        return ", ".join(str(path) for path in row["seed_summary_paths"])
    return str(row["summary_path"])


def save_figure(fig, filename, log):
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    path = output_path(filename)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.made(path)


def require_rows(rows, figure_name, log, metric_keys):
    filtered = []
    for row in rows:
        if all(row.get(key) is not None for key in metric_keys):
            filtered.append(row)
    if not filtered:
        log.warn(figure_name, f"No rows have required metrics: {metric_keys}")
    return filtered


def best_by_architecture(rows, metric_key):
    best = {}
    for row in rows:
        value = row.get(metric_key)
        arch = row.get("architecture")
        if arch not in {"resnet18", "mobilenetv2", "AST"} or value is None:
            continue
        if arch not in best or value > best[arch].get(metric_key, float("-inf")):
            best[arch] = row
    return [best[arch] for arch in ["resnet18", "mobilenetv2", "AST"] if arch in best]


def plot_best_architecture(rows, metric_key, std_key, filename, title, ylabel, log):
    figure_name = filename
    selected = best_by_architecture(rows, metric_key)
    selected = require_rows(selected, figure_name, log, [metric_key])
    if len(selected) < 2:
        log.skip(figure_name, "Need at least two architectures with data")
        return

    print(f"\nBuilding {figure_name}")
    for row in selected:
        print(f"- {source_desc(row)} | {metric_key}={row[metric_key]} std={row.get(std_key)}")

    labels = [row["architecture"] for row in selected]
    values = [percent(row[metric_key]) for row in selected]
    errors = [percent(row.get(std_key)) or 0.0 for row in selected]
    colors = [ARCH_COLORS.get(row["architecture"], "#777777") for row in selected]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, yerr=errors, capsize=5, color=colors, edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(max(0, min(values) - 8), min(100, max(values) + 3))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, row, value in zip(bars, selected, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.2f}", ha="center", va="bottom")
        ax.text(bar.get_x() + bar.get_width() / 2, ax.get_ylim()[0] + 0.3, short_label(row["experiment_name"]), ha="center", va="bottom", fontsize=8)
    save_figure(fig, filename, log)


def plot_ordered_ablation(rows, experiments, metric_key, std_key, filename, title, ylabel, log):
    by_name = {row["experiment_name"]: row for row in rows}
    selected = [by_name[name] for name in experiments if name in by_name and by_name[name].get(metric_key) is not None]
    missing = [name for name in experiments if name not in by_name or by_name[name].get(metric_key) is None]
    if missing:
        log.warn(filename, f"Missing experiments or metric: {', '.join(missing)}")
    if not selected:
        log.skip(filename, "No requested experiments available")
        return

    print(f"\nBuilding {filename}")
    for row in selected:
        print(f"- {source_desc(row)} | {metric_key}={row[metric_key]} std={row.get(std_key)}")

    labels = [short_label(row["experiment_name"]) for row in selected]
    values = [percent(row[metric_key]) for row in selected]
    errors = [percent(row.get(std_key)) or 0.0 for row in selected]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(labels, values, xerr=errors, capsize=4, color="#55A868", edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(ylabel)
    ax.set_xlim(max(0, min(values) - 8), min(100, max(values) + 3))
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    save_figure(fig, filename, log)


def plot_efficiency(rows, filename, title, only_arch=None, log=None):
    figure_name = filename
    selected = []
    for row in rows:
        if only_arch and row["architecture"] != only_arch:
            continue
        x_value = minutes(row.get("train_time_mean"))
        x_label = "Training time mean (min)"
        if x_value is None:
            x_value = row.get("latency_mean")
            x_label = "Inference latency mean (ms)"
        y_value = percent(row.get("test_f1_mean"))
        if x_value is not None and y_value is not None:
            selected.append((row, x_value, y_value, x_label))
    if not selected:
        log.skip(figure_name, "No rows with macro-F1 and training time or latency")
        return

    x_label = selected[0][3]
    print(f"\nBuilding {figure_name}")
    for row, x_value, y_value, _ in selected:
        print(f"- {source_desc(row)} | x={x_value} | macro_f1={row['test_f1_mean']}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for row, x_value, y_value, _ in selected:
        color = ARCH_COLORS.get(row["architecture"], "#777777")
        ax.scatter(x_value, y_value, s=85, color=color, edgecolor="white", linewidth=0.7, label=row["architecture"])
        ax.annotate(short_label(row["experiment_name"]), (x_value, y_value), textcoords="offset points", xytext=(5, 4), fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Architecture")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Test macro-F1 mean (%)")
    ax.grid(True, linestyle="--", alpha=0.35)
    save_figure(fig, filename, log)


def plot_strategy_rows(rows, log):
    filename = "silence_unknown_strategies.png"
    by_name = {row["experiment_name"]: row for row in rows}
    selected = [by_name[name] for name in STRATEGY_EXPERIMENTS if name in by_name and by_name[name].get("test_f1_mean") is not None]
    missing = [name for name in STRATEGY_EXPERIMENTS if name not in by_name or by_name[name].get("test_f1_mean") is None]
    if missing:
        log.warn(filename, f"Missing strategy experiments or macro-F1: {', '.join(missing)}")
    if not selected:
        log.skip(filename, "No silence/unknown strategy rows available")
        return

    print(f"\nBuilding {filename}")
    for row in selected:
        print(f"- {source_desc(row)} | macro_f1={row['test_f1_mean']} acc={row.get('test_acc_mean')}")

    labels = [f"{row['architecture']}\n{short_label(row['experiment_name'])}" for row in selected]
    f1_values = [percent(row["test_f1_mean"]) for row in selected]
    acc_values = [percent(row["test_acc_mean"]) if row.get("test_acc_mean") is not None else None for row in selected]

    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = list(range(len(selected)))
    width = 0.38
    ax.bar([x - width / 2 for x in x_positions], f1_values, width=width, label="Macro-F1", color="#55A868", edgecolor="white")
    if all(value is not None for value in acc_values):
        ax.bar([x + width / 2 for x in x_positions], acc_values, width=width, label="Accuracy", color="#4C72B0", edgecolor="white")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Metric mean (%)")
    ax.set_title("Silence / Unknown Related Strategies", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_figure(fig, filename, log)


def load_confusion_matrix(path):
    payload = read_json(path)
    if "matrix" in payload:
        matrix = payload["matrix"]
        labels = payload.get("labels", CLASS_NAMES)
    else:
        matrix = payload.get("confusion_matrix")
        labels = payload.get("class_names", CLASS_NAMES)
    if not matrix:
        return None, None
    return matrix, labels


def normalize_matrix(matrix):
    normalized = []
    for row in matrix:
        total = sum(row)
        if total:
            normalized.append([value / total for value in row])
        else:
            normalized.append([0.0 for value in row])
    return normalized


def best_seed_confusion(row):
    model_dir = row["summary_path"].parent.parent
    base_name = row["experiment_name"]
    candidates = []
    for summary_path in sorted(model_dir.glob(f"{base_name}_seed*/summary_seed*.json")):
        payload = read_json(summary_path)
        f1 = as_float(payload.get("test_f1_macro") or nested(payload, "test", "macro_f1"))
        seed = payload.get("seed")
        confusion_path = summary_path.parent / f"confusion_matrix_seed{seed}.json"
        if seed is not None and f1 is not None and confusion_path.exists():
            candidates.append((f1, seed, confusion_path, summary_path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def draw_confusion(ax, matrix, labels, title):
    normalized = normalize_matrix(matrix)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i, row in enumerate(normalized):
        for j, value in enumerate(row):
            if value >= 0.2:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white" if value > 0.55 else "black")
    return image


def plot_best_ast_confusion(rows, log):
    filename = "best_ast_confusion_matrix.png"
    ast_rows = [row for row in rows if row["architecture"] == "AST" and row.get("test_f1_mean") is not None]
    if not ast_rows:
        log.skip(filename, "No AST rows with macro-F1")
        return
    best = max(ast_rows, key=lambda row: row["test_f1_mean"])
    candidate = best_seed_confusion(best)
    if candidate is None:
        log.warn(filename, f"No seed confusion matrix found for {best['experiment_name']}")
        return
    f1, seed, confusion_path, summary_path = candidate
    matrix, labels = load_confusion_matrix(confusion_path)
    if matrix is None:
        log.warn(filename, f"Could not read confusion matrix: {confusion_path}")
        return

    print(f"\nBuilding {filename}")
    print(f"- {source_desc(best)} | selected best seed {seed} from {summary_path} | confusion={confusion_path} | seed_f1={f1}")

    fig, ax = plt.subplots(figsize=(8, 7))
    image = draw_confusion(ax, matrix, labels, f"Best AST Confusion Matrix\n{best['experiment_name']} seed {seed}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized share")
    save_figure(fig, filename, log)


def plot_best_model_confusions(rows, log):
    filename = "best_model_confusion_matrices.png"
    selected = best_by_architecture(rows, "test_f1_mean")
    selected = [row for row in selected if row["architecture"] in {"resnet18", "mobilenetv2", "AST"}]
    if len(selected) < 2:
        log.skip(filename, "Need at least two architectures with confusion matrices")
        return

    items = []
    for row in selected:
        candidate = best_seed_confusion(row)
        if candidate is None:
            log.warn(filename, f"No seed confusion matrix found for {row['experiment_name']}")
            continue
        f1, seed, confusion_path, summary_path = candidate
        matrix, labels = load_confusion_matrix(confusion_path)
        if matrix is None:
            log.warn(filename, f"Could not read confusion matrix: {confusion_path}")
            continue
        items.append((row, seed, f1, confusion_path, matrix, labels))
    if len(items) < 2:
        log.skip(filename, "Too few readable confusion matrices")
        return

    print(f"\nBuilding {filename}")
    for row, seed, f1, confusion_path, _, _ in items:
        print(f"- {source_desc(row)} | selected seed {seed} | confusion={confusion_path} | seed_f1={f1}")

    fig, axes = plt.subplots(1, len(items), figsize=(7 * len(items), 6.5))
    if len(items) == 1:
        axes = [axes]
    last_image = None
    for ax, (row, seed, _, _, matrix, labels) in zip(axes, items):
        title = f"{row['architecture']}\n{row['experiment_name']} seed {seed}"
        last_image = draw_confusion(ax, matrix, labels, title)
    fig.colorbar(last_image, ax=axes, fraction=0.025, pad=0.02, label="Row-normalized share")
    save_figure(fig, filename, log)


def plot_validation_vs_test(rows, log):
    filename = "validation_vs_test_accuracy.png"
    selected = require_rows(rows, filename, log, ["val_acc_mean", "test_acc_mean"])
    if not selected:
        log.skip(filename, "No rows with both validation and test accuracy")
        return

    print(f"\nBuilding {filename}")
    for row in selected:
        print(f"- {source_desc(row)} | val_acc={row['val_acc_mean']} test_acc={row['test_acc_mean']}")

    fig, ax = plt.subplots(figsize=(8, 7))
    for row in selected:
        x_value = percent(row["val_acc_mean"])
        y_value = percent(row["test_acc_mean"])
        color = ARCH_COLORS.get(row["architecture"], "#777777")
        ax.scatter(x_value, y_value, s=75, color=color, edgecolor="white", linewidth=0.7, label=row["architecture"])
        ax.annotate(short_label(row["experiment_name"]), (x_value, y_value), textcoords="offset points", xytext=(4, 3), fontsize=7)
    values = [percent(row["val_acc_mean"]) for row in selected] + [percent(row["test_acc_mean"]) for row in selected]
    low = max(0, min(values) - 2)
    high = min(100, max(values) + 1)
    ax.plot([low, high], [low, high], linestyle="--", color="black", alpha=0.5, label="val = test")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Architecture")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Validation accuracy mean (%)")
    ax.set_ylabel("Test accuracy mean (%)")
    ax.set_title("Validation vs Test Accuracy", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    save_figure(fig, filename, log)


def main():
    log = PlotLog()
    rows = collect_experiment_rows()
    if not rows:
        log.warn("all", f"No summary_all_seeds.json files found under {OUTPUTS}")
    else:
        print(f"Loaded {len(rows)} aggregate summaries")

    plot_best_architecture(
        rows,
        "test_f1_mean",
        "test_f1_std",
        "best_architecture_macro_f1.png",
        "Best Configuration per Architecture by Macro-F1",
        "Test macro-F1 mean (%)",
        log,
    )
    plot_best_architecture(
        rows,
        "test_acc_mean",
        "test_acc_std",
        "best_architecture_accuracy.png",
        "Best Configuration per Architecture by Accuracy",
        "Test accuracy mean (%)",
        log,
    )
    plot_ordered_ablation(
        rows,
        AST_ORDER,
        "test_f1_mean",
        "test_f1_std",
        "ast_ablation_macro_f1.png",
        "AST Ablation: Test Macro-F1",
        "Test macro-F1 mean (%)",
        log,
    )
    plot_ordered_ablation(
        rows,
        AST_ORDER,
        "test_acc_mean",
        "test_acc_std",
        "ast_ablation_accuracy.png",
        "AST Ablation: Test Accuracy",
        "Test accuracy mean (%)",
        log,
    )
    plot_efficiency(
        rows,
        "ast_efficiency_tradeoff.png",
        "AST Efficiency Trade-off",
        only_arch="AST",
        log=log,
    )
    plot_efficiency(
        rows,
        "global_efficiency_tradeoff.png",
        "Global Efficiency Trade-off",
        log=log,
    )
    plot_strategy_rows(rows, log)
    plot_best_ast_confusion(rows, log)
    plot_best_model_confusions(rows, log)
    plot_validation_vs_test(rows, log)

    print("\nSummary")
    print(f"Generated ({len(log.generated)}):")
    for path in log.generated:
        print(f"- {path}")
    print(f"Skipped ({len(log.skipped)}):")
    for name, message in log.skipped:
        print(f"- {name}: {message}")
    print(f"Missing / warnings ({len(log.missing)}):")
    for name, message in log.missing:
        print(f"- {name}: {message}")


if __name__ == "__main__":
    main()
