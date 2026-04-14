import argparse
import csv
import json
import re
from pathlib import Path


def read_json(path_obj):
    with open(path_obj, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def safe_get(payload, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value, digits=4):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def aggregate_candidate_for_run_dir(run_dir):
    base_name = re.sub(r"_seed\d+$", "", run_dir.name)
    return run_dir.parent / base_name / "summary_all_seeds.json"


def row_from_aggregate_summary(summary_path):
    payload = read_json(summary_path)
    aggregate = payload.get("aggregate", {})

    return {
        "model": payload.get("model"),
        "experiment_name": payload.get("experiment_name"),
        "num_runs": int(payload.get("num_runs", 0)),
        "test_acc_mean": to_float_or_none(safe_get(aggregate, "test", "acc", "mean")),
        "test_acc_std": to_float_or_none(safe_get(aggregate, "test", "acc", "std")),
        "test_macro_f1_mean": to_float_or_none(safe_get(aggregate, "test", "macro_f1", "mean")),
        "test_macro_f1_std": to_float_or_none(safe_get(aggregate, "test", "macro_f1", "std")),
        "val_acc_mean": to_float_or_none(safe_get(aggregate, "best_validation", "acc", "mean")),
        "val_acc_std": to_float_or_none(safe_get(aggregate, "best_validation", "acc", "std")),
        "inference_latency_ms_mean": to_float_or_none(safe_get(aggregate, "test", "inference_latency_ms", "mean")),
        "training_time_sec_mean": to_float_or_none(safe_get(aggregate, "timing", "total_training_time_sec", "mean")),
        "trainable_params": to_float_or_none(safe_get(aggregate, "model_parameters", "trainable", "mean")),
        "unknown_recall_mean": to_float_or_none(safe_get(aggregate, "unknown_silence_analysis", "unknown", "recall", "mean")),
        "silence_recall_mean": to_float_or_none(safe_get(aggregate, "unknown_silence_analysis", "silence", "recall", "mean")),
        "source": "summary_all_seeds",
        "summary_path": str(summary_path),
    }


def row_from_single_seed_summary(summary_path):
    payload = read_json(summary_path)
    test_payload = payload.get("test", {}) or {}
    best_validation = payload.get("best_validation", {}) or {}
    timing = payload.get("timing", {}) or {}
    model_parameters = payload.get("model_parameters", {}) or {}
    unknown_silence = payload.get("unknown_silence_analysis", {}) or {}

    return {
        "model": payload.get("model") or payload.get("model_name"),
        "experiment_name": payload.get("experiment_name"),
        "num_runs": 1,
        "test_acc_mean": to_float_or_none(test_payload.get("acc")),
        "test_acc_std": 0.0,
        "test_macro_f1_mean": to_float_or_none(test_payload.get("macro_f1")),
        "test_macro_f1_std": 0.0,
        "val_acc_mean": to_float_or_none(best_validation.get("acc")),
        "val_acc_std": 0.0,
        "inference_latency_ms_mean": to_float_or_none(test_payload.get("inference_latency_ms")),
        "training_time_sec_mean": to_float_or_none(timing.get("total_training_time_sec")),
        "trainable_params": to_float_or_none(model_parameters.get("trainable")),
        "unknown_recall_mean": to_float_or_none(safe_get(unknown_silence, "unknown", "recall")),
        "silence_recall_mean": to_float_or_none(safe_get(unknown_silence, "silence", "recall")),
        "source": "summary_seed",
        "summary_path": str(summary_path),
    }


def collect_leaderboard_rows(outputs_dir):
    rows = []
    aggregate_paths = sorted(outputs_dir.glob("*/*/summary_all_seeds.json"))

    covered_bases = set()
    for summary_path in aggregate_paths:
        rows.append(row_from_aggregate_summary(summary_path))
        covered_bases.add(summary_path.parent.resolve())

    for summary_path in sorted(outputs_dir.glob("*/*/summary_seed*.json")):
        run_dir = summary_path.parent.resolve()
        aggregate_candidate = aggregate_candidate_for_run_dir(summary_path.parent)
        if aggregate_candidate.exists():
            continue
        if run_dir in covered_bases:
            continue
        rows.append(row_from_single_seed_summary(summary_path))

    def sort_key(row):
        test_acc = row.get("test_acc_mean")
        test_f1 = row.get("test_macro_f1_mean")
        return (
            -1.0 if test_acc is None else -float(test_acc),
            -1.0 if test_f1 is None else -float(test_f1),
            str(row.get("model") or ""),
            str(row.get("experiment_name") or ""),
        )

    rows.sort(key=sort_key)
    return rows


def collect_top_confusions(outputs_dir):
    rows = []
    for error_path in sorted(outputs_dir.glob("*/*/error_analysis_seed*.json")):
        payload = read_json(error_path)
        model_name = error_path.parent.parent.name
        experiment_name = error_path.parent.name
        seed_match = re.search(r"seed(\d+)", error_path.name)
        seed = int(seed_match.group(1)) if seed_match else None

        for confusion in payload.get("top_confusions", [])[:10]:
            rows.append(
                {
                    "model": model_name,
                    "experiment_name": experiment_name,
                    "seed": seed,
                    "true_class": confusion.get("true_class"),
                    "predicted_as": confusion.get("predicted_as"),
                    "count": confusion.get("count"),
                    "rate_within_true": confusion.get("rate_within_true"),
                    "error_analysis_path": str(error_path),
                }
            )
    return rows


def collect_unknown_silence_rows(outputs_dir):
    rows = []

    for aggregate_path in sorted(outputs_dir.glob("*/*/summary_all_seeds.json")):
        payload = read_json(aggregate_path)
        aggregate = payload.get("aggregate", {})
        model_name = payload.get("model")
        experiment_name = payload.get("experiment_name")
        for class_name in ("unknown", "silence"):
            rows.append(
                {
                    "model": model_name,
                    "experiment_name": experiment_name,
                    "scope": "aggregate",
                    "class": class_name,
                    "precision_mean": to_float_or_none(
                        safe_get(aggregate, "unknown_silence_analysis", class_name, "precision", "mean")
                    ),
                    "recall_mean": to_float_or_none(
                        safe_get(aggregate, "unknown_silence_analysis", class_name, "recall", "mean")
                    ),
                    "source_path": str(aggregate_path),
                }
            )

    for error_path in sorted(outputs_dir.glob("*/*/error_analysis_seed*.json")):
        payload = read_json(error_path)
        model_name = error_path.parent.parent.name
        experiment_name = error_path.parent.name
        seed_match = re.search(r"seed(\d+)", error_path.name)
        seed = int(seed_match.group(1)) if seed_match else None
        analysis = payload.get("unknown_silence_analysis", {}) or {}

        for class_name in ("unknown", "silence"):
            class_payload = analysis.get(class_name) or {}
            rows.append(
                {
                    "model": model_name,
                    "experiment_name": experiment_name,
                    "scope": f"seed_{seed}" if seed is not None else "seed",
                    "class": class_name,
                    "precision_mean": to_float_or_none(class_payload.get("precision")),
                    "recall_mean": to_float_or_none(class_payload.get("recall")),
                    "source_path": str(error_path),
                }
            )

    return rows


def write_csv(rows, output_path, fieldnames):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_table(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| Model | Experiment | Runs | Test Acc (mean±std) | Test Macro-F1 (mean±std) | Val Acc (mean±std) | Latency [ms] | Train time [s] | Trainable params | Unknown recall | Silence recall |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            "| "
            + f"{row.get('model') or ''} | "
            + f"{row.get('experiment_name') or ''} | "
            + f"{row.get('num_runs') or ''} | "
            + f"{format_float(row.get('test_acc_mean'))}±{format_float(row.get('test_acc_std'))} | "
            + f"{format_float(row.get('test_macro_f1_mean'))}±{format_float(row.get('test_macro_f1_std'))} | "
            + f"{format_float(row.get('val_acc_mean'))}±{format_float(row.get('val_acc_std'))} | "
            + f"{format_float(row.get('inference_latency_ms_mean'), digits=3)} | "
            + f"{format_float(row.get('training_time_sec_mean'), digits=2)} | "
            + f"{format_float(row.get('trainable_params'), digits=0)} | "
            + f"{format_float(row.get('unknown_recall_mean'))} | "
            + f"{format_float(row.get('silence_recall_mean'))} |"
        )

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build experiment summary tables from outputs directory")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Root outputs directory")
    parser.add_argument("--analysis_dir", type=str, default="outputs/analysis", help="Directory for generated reports")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory does not exist: {outputs_dir}")

    leaderboard_rows = collect_leaderboard_rows(outputs_dir)
    leaderboard_fields = [
        "model",
        "experiment_name",
        "num_runs",
        "test_acc_mean",
        "test_acc_std",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "val_acc_mean",
        "val_acc_std",
        "inference_latency_ms_mean",
        "training_time_sec_mean",
        "trainable_params",
        "unknown_recall_mean",
        "silence_recall_mean",
        "source",
        "summary_path",
    ]
    write_csv(leaderboard_rows, analysis_dir / "leaderboard.csv", leaderboard_fields)
    write_markdown_table(leaderboard_rows, analysis_dir / "leaderboard.md")
    with open(analysis_dir / "leaderboard.json", "w", encoding="utf-8") as file_obj:
        json.dump(leaderboard_rows, file_obj, indent=2)

    confusion_rows = collect_top_confusions(outputs_dir)
    confusion_fields = [
        "model",
        "experiment_name",
        "seed",
        "true_class",
        "predicted_as",
        "count",
        "rate_within_true",
        "error_analysis_path",
    ]
    write_csv(confusion_rows, analysis_dir / "top_confusions.csv", confusion_fields)

    unknown_silence_rows = collect_unknown_silence_rows(outputs_dir)
    unknown_silence_fields = [
        "model",
        "experiment_name",
        "scope",
        "class",
        "precision_mean",
        "recall_mean",
        "source_path",
    ]
    write_csv(unknown_silence_rows, analysis_dir / "unknown_silence.csv", unknown_silence_fields)

    print(f"Leaderboard rows: {len(leaderboard_rows)}")
    print(f"Top confusion rows: {len(confusion_rows)}")
    print(f"Unknown/silence rows: {len(unknown_silence_rows)}")
    print(f"Saved reports to: {analysis_dir}")


if __name__ == "__main__":
    main()
