import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SpeechCommandsDataset
from models import get_available_models, get_model
import time
import argparse
import random
import numpy as np
import contextlib
import csv
import json
import re
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import mlflow
except ImportError:
    mlflow = None


CLASS_NAMES = SpeechCommandsDataset.TARGET_WORDS + ["unknown", "silence"]
UNKNOWN_CLASS_INDEX = CLASS_NAMES.index("unknown")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_weights(labels, num_classes=12):
    class_counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    total = class_counts.sum()
    weights = total / torch.clamp(class_counts, min=1.0)
    weights = weights / weights.mean()
    return weights

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

def evaluate(model, dataloader, criterion, device, measure_latency=False, return_predictions=False):
    """
    Evaluates the model on a given split and returns loss/accuracy and macro metrics.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    inference_time_sec = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            if measure_latency and device.type == "cuda":
                torch.cuda.synchronize()
            batch_start = time.perf_counter() if measure_latency else None

            outputs = model(inputs)

            if measure_latency and device.type == "cuda":
                torch.cuda.synchronize()
            if measure_latency and batch_start is not None:
                inference_time_sec += time.perf_counter() - batch_start

            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(predicted.cpu().tolist())

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    metrics = {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }
    if measure_latency:
        metrics["inference_time_sec"] = float(inference_time_sec)
        metrics["inference_latency_ms"] = float((inference_time_sec / total) * 1000.0) if total > 0 else None

    if return_predictions:
        return running_loss / total, correct / total, metrics, all_labels, all_preds
    return running_loss / total, correct / total, metrics


def count_model_parameters(model):
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total_params), int(trainable_params)


def build_confusion_analysis(all_labels, all_preds, class_names):
    if len(all_labels) == 0:
        return None, None

    num_classes = len(class_names)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    cm = cm.astype(np.int64)

    row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
    cm_normalized = np.divide(cm, np.maximum(row_sums, 1.0))

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    per_class_metrics = []
    for class_index, class_name in enumerate(class_names):
        per_class_metrics.append(
            {
                "class_index": int(class_index),
                "class_name": class_name,
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
            }
        )

    top_confusions = []
    for true_index in range(num_classes):
        for pred_index in range(num_classes):
            if true_index == pred_index:
                continue
            count = int(cm[true_index, pred_index])
            if count == 0:
                continue
            row_total = int(cm[true_index, :].sum())
            top_confusions.append(
                {
                    "true_class": class_names[true_index],
                    "predicted_as": class_names[pred_index],
                    "count": count,
                    "rate_within_true": float(count / row_total) if row_total > 0 else 0.0,
                }
            )
    top_confusions = sorted(top_confusions, key=lambda item: item["count"], reverse=True)[:15]

    unknown_silence_analysis = {}
    for focus_class in ("unknown", "silence"):
        if focus_class not in class_names:
            continue
        idx = class_names.index(focus_class)
        tp = int(cm[idx, idx])
        true_total = int(cm[idx, :].sum())
        predicted_total = int(cm[:, idx].sum())

        row_without_diag = cm[idx, :].copy()
        row_without_diag[idx] = 0
        most_confused_as_idx = int(np.argmax(row_without_diag)) if row_without_diag.sum() > 0 else None

        col_without_diag = cm[:, idx].copy()
        col_without_diag[idx] = 0
        most_common_source_idx = int(np.argmax(col_without_diag)) if col_without_diag.sum() > 0 else None

        unknown_silence_analysis[focus_class] = {
            "precision": float(tp / predicted_total) if predicted_total > 0 else 0.0,
            "recall": float(tp / true_total) if true_total > 0 else 0.0,
            "support": true_total,
            "predicted_total": predicted_total,
            "most_confused_as": class_names[most_confused_as_idx] if most_confused_as_idx is not None else None,
            "most_common_source_as_prediction": class_names[most_common_source_idx] if most_common_source_idx is not None else None,
        }

    payload = {
        "class_names": list(class_names),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "per_class_metrics": per_class_metrics,
        "top_confusions": top_confusions,
    }
    return payload, unknown_silence_analysis


def save_confusion_matrix_plot(confusion_payload, output_path, title):
    if plt is None:
        return

    class_names = confusion_payload["class_names"]
    matrix = np.array(confusion_payload["confusion_matrix_normalized"], dtype=np.float64)

    figure_size = max(8, len(class_names) * 0.8)
    fig, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)

    color_threshold = 0.5
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            axis.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= color_threshold else "black",
                fontsize=8,
            )

    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def map_labels_to_unknown_binary(labels, unknown_class_index=UNKNOWN_CLASS_INDEX):
    return (labels == unknown_class_index).long()


def train_unknown_detector_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        binary_labels = map_labels_to_unknown_binary(labels)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, binary_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predicted = outputs.argmax(dim=1)
        total += binary_labels.size(0)
        correct += predicted.eq(binary_labels).sum().item()

    return running_loss / max(total, 1), correct / max(total, 1)


def evaluate_unknown_detector(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            binary_labels = map_labels_to_unknown_binary(labels)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predicted = (probabilities >= threshold).long()

            loss = criterion(outputs, binary_labels)
            running_loss += loss.item() * inputs.size(0)

            total += binary_labels.size(0)
            correct += predicted.eq(binary_labels).sum().item()
            all_true.extend(binary_labels.cpu().tolist())
            all_pred.extend(predicted.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true,
        all_pred,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "acc": float(correct / max(total, 1)),
        "threshold": float(threshold),
    }
    return running_loss / max(total, 1), correct / max(total, 1), metrics


def train_separate_unknown_detector(
    args,
    train_loader,
    val_loader,
    device,
    checkpoint_path,
    summary_path,
):
    detector_model_name = args.unknown_detector_model if args.unknown_detector_model else args.model
    detector_lr = args.unknown_detector_lr if args.unknown_detector_lr is not None else args.lr
    detector_dropout = args.unknown_detector_dropout if args.unknown_detector_dropout is not None else args.dropout

    detector_model = get_model(
        detector_model_name,
        num_classes=2,
        input_channels=1,
        use_pretrained=args.use_pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=detector_dropout,
    ).to(device)

    total_params, trainable_param_count = count_model_parameters(detector_model)
    trainable_param_iterator = filter(lambda parameter: parameter.requires_grad, detector_model.parameters())
    if args.optimizer == "sgd":
        optimizer = optim.SGD(trainable_param_iterator, lr=detector_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(trainable_param_iterator, lr=detector_lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    detector_epochs = int(max(1, args.unknown_detector_epochs))
    detector_patience = int(max(2, min(args.patience, detector_epochs)))

    best_val_acc = float("-inf")
    best_state = None
    best_epoch = None
    patience_counter = 0

    for epoch in range(detector_epochs):
        train_loss, train_acc = train_unknown_detector_one_epoch(detector_model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_metrics = evaluate_unknown_detector(
            detector_model,
            val_loader,
            criterion,
            device,
            threshold=args.unknown_detector_threshold,
        )

        print(
            f"[UnknownDetector] Epoch {epoch}: "
            f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val F1={val_metrics['f1']:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in detector_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= detector_patience:
            print(f"[UnknownDetector] Early stopping after {epoch + 1} epochs.")
            break

    if best_state is not None:
        detector_model.load_state_dict(best_state)

    checkpoint_payload = {
        "model_state_dict": detector_model.state_dict(),
        "model": detector_model_name,
        "seed": int(args.seed),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_val_acc": float(best_val_acc) if best_val_acc != float("-inf") else None,
    }
    torch.save(checkpoint_payload, str(checkpoint_path))

    _, val_acc, val_metrics = evaluate_unknown_detector(
        detector_model,
        val_loader,
        criterion,
        device,
        threshold=args.unknown_detector_threshold,
    )
    summary_payload = {
        "model": detector_model_name,
        "seed": int(args.seed),
        "threshold": float(args.unknown_detector_threshold),
        "epochs_configured": detector_epochs,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_val_acc": float(best_val_acc) if best_val_acc != float("-inf") else None,
        "val_binary_metrics": val_metrics,
        "model_parameters": {
            "total": int(total_params),
            "trainable": int(trainable_param_count),
        },
        "checkpoint": str(checkpoint_path),
    }
    write_run_summary(summary_payload, summary_path)
    return detector_model, summary_payload


def evaluate_with_unknown_detector(
    model,
    unknown_detector_model,
    dataloader,
    criterion,
    device,
    threshold=0.5,
    unknown_class_index=UNKNOWN_CLASS_INDEX,
    measure_latency=False,
):
    model.eval()
    unknown_detector_model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_unknown_true = []
    all_unknown_pred = []
    override_count = 0
    inference_time_sec = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            if measure_latency and device.type == "cuda":
                torch.cuda.synchronize()
            batch_start = time.perf_counter() if measure_latency else None

            main_outputs = model(inputs)
            unknown_outputs = unknown_detector_model(inputs)

            if measure_latency and device.type == "cuda":
                torch.cuda.synchronize()
            if measure_latency and batch_start is not None:
                inference_time_sec += time.perf_counter() - batch_start

            loss = criterion(main_outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            main_predicted = main_outputs.argmax(dim=1)
            unknown_probabilities = torch.softmax(unknown_outputs, dim=1)[:, 1]
            predicted_unknown = unknown_probabilities >= threshold

            final_predicted = main_predicted.clone()
            override_count += int((predicted_unknown & (main_predicted != unknown_class_index)).sum().item())
            final_predicted[predicted_unknown] = unknown_class_index

            total += labels.size(0)
            correct += final_predicted.eq(labels).sum().item()

            binary_true = map_labels_to_unknown_binary(labels, unknown_class_index=unknown_class_index)
            binary_pred = predicted_unknown.long()
            all_unknown_true.extend(binary_true.cpu().tolist())
            all_unknown_pred.extend(binary_pred.cpu().tolist())

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(final_predicted.cpu().tolist())

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    unknown_precision, unknown_recall, unknown_f1, _ = precision_recall_fscore_support(
        all_unknown_true,
        all_unknown_pred,
        average="binary",
        zero_division=0,
    )
    unknown_positive_rate = float(np.mean(all_unknown_pred)) if len(all_unknown_pred) > 0 else 0.0

    metrics = {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "unknown_detector_precision": float(unknown_precision),
        "unknown_detector_recall": float(unknown_recall),
        "unknown_detector_f1": float(unknown_f1),
        "unknown_detector_acc": float(np.mean(np.array(all_unknown_true) == np.array(all_unknown_pred))) if len(all_unknown_true) > 0 else 0.0,
        "unknown_detector_threshold": float(threshold),
        "unknown_detector_positive_rate": float(unknown_positive_rate),
        "overridden_to_unknown_count": int(override_count),
        "overridden_to_unknown_rate": float(override_count / max(total, 1)),
    }
    if measure_latency:
        metrics["inference_time_sec"] = float(inference_time_sec)
        metrics["inference_latency_ms"] = float((inference_time_sec / total) * 1000.0) if total > 0 else None

    return running_loss / max(total, 1), correct / max(total, 1), metrics, all_labels, all_preds


def add_seed_suffix_to_filename(file_name, seed):
    path_obj = Path(file_name)
    seed_tag = f"seed{seed}"
    stem = path_obj.stem if path_obj.suffix else path_obj.name
    if re.search(rf"(?:^|_){seed_tag}$", stem):
        seeded_stem = stem
    else:
        seeded_stem = f"{stem}_{seed_tag}"

    return f"{seeded_stem}{path_obj.suffix}" if path_obj.suffix else seeded_stem


def make_json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return str(value)


def resolve_run_output_paths(args):
    """Create the run output directory and return a dict of output file paths."""
    checkpoint_name = Path(args.checkpoint_path).name if args.checkpoint_path else "best_model.pt"
    checkpoint_name = add_seed_suffix_to_filename(checkpoint_name, args.seed)
    run_output_dir = Path("outputs") / args.model / args.experiment_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "checkpoint": run_output_dir / checkpoint_name,
        "history_csv": run_output_dir / f"history_seed{args.seed}.csv",
        "summary_json": run_output_dir / f"summary_seed{args.seed}.json",
        "config_yaml": run_output_dir / f"config_seed{args.seed}.yaml",
        "confusion_matrix_json": run_output_dir / f"confusion_matrix_seed{args.seed}.json",
        "confusion_matrix_png": run_output_dir / f"confusion_matrix_seed{args.seed}.png",
        "error_analysis_json": run_output_dir / f"error_analysis_seed{args.seed}.json",
        "unknown_detector_checkpoint": run_output_dir / f"unknown_detector_seed{args.seed}.pt",
        "unknown_detector_summary_json": run_output_dir / f"unknown_detector_summary_seed{args.seed}.json",
    }
    return run_output_dir, output_paths


def write_epoch_history_csv(history_rows, csv_path):
    """Write per-epoch training metrics to a CSV file."""
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_macro_precision",
        "val_macro_recall",
        "val_macro_f1",
        "val_f1_macro",
        "epoch_time_seconds",
        "learning_rate_current",
        "is_best",
        "best_val_acc_so_far",
        "patience_counter",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)


def write_run_summary(summary_payload, summary_path):
    """Serialise ``summary_payload`` as indented JSON to ``summary_path``."""
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary_payload, file_obj, indent=2)


def write_run_config(args, config_path):
    """Serialise the resolved argument namespace to a YAML file."""
    config_payload = make_json_safe(vars(args))
    with open(config_path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config_payload, file_obj, sort_keys=True)


def load_model_state_dict_from_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    model.load_state_dict(checkpoint)
    return {}


def aggregate_numeric(values):
    if len(values) == 0:
        return None
    array = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n": int(array.size),
    }


def aggregate_metric_group(run_summaries, group_key, metric_keys):
    group_payload = {}
    for metric_name in metric_keys:
        metric_values = []
        for summary in run_summaries:
            metric_value = summary.get(group_key, {}).get(metric_name)
            if metric_value is not None:
                metric_values.append(metric_value)
        group_payload[metric_name] = aggregate_numeric(metric_values)
    return group_payload


def aggregate_unknown_silence(run_summaries):
    focus_payload = {}
    for class_name in ("unknown", "silence"):
        class_payload = {}
        for metric_name in ("precision", "recall"):
            values = []
            for summary in run_summaries:
                focus_metrics = summary.get("unknown_silence_analysis") or {}
                metric_value = focus_metrics.get(class_name, {}).get(metric_name)
                if metric_value is not None:
                    values.append(metric_value)
            class_payload[metric_name] = aggregate_numeric(values)
        focus_payload[class_name] = class_payload
    return focus_payload


def save_multi_seed_summary(base_args, run_results):
    """Aggregate per-seed results and write ``summary_all_seeds.json``.

    Skipped when fewer than two seeds were run. Returns the path to the
    aggregate summary, or ``None`` if no summary was written.
    """
    if len(run_results) < 2:
        return None

    aggregate_output_dir = Path("outputs") / base_args.model / base_args.experiment_name
    aggregate_output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_summary_path = aggregate_output_dir / "summary_all_seeds.json"

    run_summaries = [run_result["summary"] for run_result in run_results]
    seeds = [int(run_result["seed"]) for run_result in run_results]

    per_seed_runs = []
    for run_result in run_results:
        summary = run_result["summary"]
        per_seed_runs.append(
            {
                "seed": int(run_result["seed"]),
                "experiment_name": run_result["experiment_name"],
                "run_output_dir": run_result["run_output_dir"],
                "summary_path": run_result["summary_path"],
                "artifacts": summary.get("artifacts", {}),
                "epochs": summary.get("epochs", {}),
                "best_validation": summary.get("best_validation", {}),
                "test": summary.get("test", {}),
                "timing": summary.get("timing", {}),
                "model_parameters": summary.get("model_parameters", {}),
                "unknown_silence_analysis": summary.get("unknown_silence_analysis", {}),
            }
        )

    aggregate_payload = {
        "model": base_args.model,
        "experiment_name": base_args.experiment_name,
        "seeds": seeds,
        "num_runs": len(run_results),
        "aggregate": {
            "best_validation": aggregate_metric_group(
                run_summaries,
                "best_validation",
                ["acc", "loss", "macro_precision", "macro_recall", "macro_f1"],
            ),
            "test": aggregate_metric_group(
                run_summaries,
                "test",
                ["acc", "loss", "macro_precision", "macro_recall", "macro_f1", "inference_latency_ms"],
            ),
            "timing": aggregate_metric_group(
                run_summaries,
                "timing",
                ["total_training_time_sec", "mean_epoch_time_seconds", "max_epoch_time_seconds"],
            ),
            "model_parameters": aggregate_metric_group(
                run_summaries,
                "model_parameters",
                ["total", "trainable"],
            ),
            "unknown_silence_analysis": aggregate_unknown_silence(run_summaries),
        },
        "per_seed_runs": per_seed_runs,
    }

    write_run_summary(aggregate_payload, aggregate_summary_path)
    return str(aggregate_summary_path)

def run_experiment(args):
    """Execute a single training run for the given seed and configuration.

    Builds datasets and data loaders, instantiates the model, runs the training
    loop with early stopping, evaluates on the test split (when enabled), saves
    all artefacts (checkpoint, history CSV, summary JSON, confusion matrix), and
    returns a result dict suitable for :func:`save_multi_seed_summary`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    unknown_keep_prob = args.unknown_keep_prob if args.use_unknown_undersampling else 1.0
    run_output_dir, output_paths = resolve_run_output_paths(args)
    checkpoint_path = output_paths["checkpoint"]

    train_ds = SpeechCommandsDataset(
        root_dir=args.data_path,
        split='train', 
        n_mels=args.n_mels,
        apply_augment=args.use_augment,
        time_mask=args.time_mask,
        freq_mask=args.freq_mask,
        silence_train_samples=args.silence_train_samples,
        silence_eval_samples=args.silence_eval_samples,
        unknown_keep_prob=unknown_keep_prob,
    )
    val_ds = SpeechCommandsDataset(
        root_dir=args.data_path,
        split='val',
        n_mels=args.n_mels,
        silence_train_samples=args.silence_train_samples,
        silence_eval_samples=args.silence_eval_samples,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=args.num_workers > 0,
    )

    test_loader = None
    if args.run_test:
        test_ds = SpeechCommandsDataset(
            root_dir=args.data_path,
            split='test',
            n_mels=args.n_mels,
            silence_train_samples=args.silence_train_samples,
            silence_eval_samples=args.silence_eval_samples,
            include_silence_in_test=args.include_silence_in_test,
            silence_test_samples=args.silence_test_samples,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=args.num_workers > 0,
        )

    model = get_model(
        args.model,
        num_classes=12,
        input_channels=1,
        use_pretrained=args.use_pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    ).to(device)
    total_params, trainable_param_count = count_model_parameters(model)

    if args.use_weighted_loss:
        weights = get_class_weights(train_ds.labels, num_classes=12).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    trainable_param_iterator = filter(lambda parameter: parameter.requires_grad, model.parameters())
    if args.optimizer == "sgd":
        optimizer = optim.SGD(trainable_param_iterator, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(trainable_param_iterator, lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = float("-inf")
    best_val_loss = None
    best_val_metrics = None
    best_epoch = None
    patience_counter = 0
    start_time = time.time()
    epoch_history = []
    test_payload = None
    confusion_payload = None
    unknown_silence_payload = None
    unknown_detector_model = None
    unknown_detector_summary = None
    
    print(f"Starting training on {gpu_name}...")
    print(f"Run seed: {args.seed}")
    if args.use_unknown_undersampling:
        print(f"Unknown keep probability (train): {unknown_keep_prob:.3f}")
    else:
        print(
            "Unknown undersampling disabled; "
            f"effective unknown keep probability (train): {unknown_keep_prob:.3f} "
            f"(configured: {args.unknown_keep_prob:.3f})"
        )
    print(f"Model parameters: total={total_params:,}, trainable={trainable_param_count:,}")
    print(f"Separate unknown detector enabled: {args.use_separate_unknown_detector}")
    print(f"Include synthetic silence in test split: {args.include_silence_in_test}")
    if args.include_silence_in_test:
        print(f"Synthetic silence samples in test split: {args.silence_test_samples}")
    print(f"Run outputs directory: {run_output_dir}")

    if args.use_separate_unknown_detector:
        print("Training separate binary unknown detector...")
        unknown_detector_model, unknown_detector_summary = train_separate_unknown_detector(
            args=args,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_path=output_paths["unknown_detector_checkpoint"],
            summary_path=output_paths["unknown_detector_summary_json"],
        )
        print(
            "[UnknownDetector] "
            f"Val Acc={unknown_detector_summary['val_binary_metrics']['acc']:.4f}, "
            f"Val F1={unknown_detector_summary['val_binary_metrics']['f1']:.4f}, "
            f"Threshold={unknown_detector_summary['threshold']:.2f}"
        )

    use_mlflow = (mlflow is not None) and (not args.disable_mlflow)
    if mlflow is None and not args.disable_mlflow:
        print("MLflow is not installed. Continuing without MLflow logging.")

    run_context = mlflow.start_run(run_name=f"{args.model}_{args.experiment_name}") if use_mlflow else contextlib.nullcontext()
    run_id = f"{args.model}_{args.experiment_name}_seed{args.seed}"
    with run_context as active_run:
        if use_mlflow:
            if active_run is not None:
                run_id = active_run.info.run_id
            mlflow.log_params(vars(args))
            mlflow.log_param("device_name", gpu_name)
            mlflow.log_param("device", device.type)
            mlflow.log_param("run_id", run_id)
            mlflow.log_param("run_output_dir", str(run_output_dir))
            mlflow.log_param("checkpoint_path_resolved", str(checkpoint_path))

        for epoch in range(args.epochs):
            epoch_start = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_metrics = evaluate(model, val_loader, criterion, device)
            current_lr = float(optimizer.param_groups[0]["lr"])

            epoch_duration = time.time() - epoch_start

            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_macro_precision": val_metrics["macro_precision"],
                        "val_macro_recall": val_metrics["macro_recall"],
                        "val_macro_f1": val_metrics["macro_f1"],
                        "learning_rate_current": current_lr,
                        "epoch_time_seconds": epoch_duration,
                    },
                    step=epoch,
                )

            print(
                f"Epoch {epoch}: "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Acc: {val_acc:.4f}, "
                f"Val Macro-F1: {val_metrics['macro_f1']:.4f}, "
                f"Time: {epoch_duration:.2f}s"
            )

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_val_metrics = dict(val_metrics)
                best_epoch = epoch
                patience_counter = 0
                checkpoint_payload = {
                    "model_state_dict": model.state_dict(),
                    "seed": int(args.seed),
                    "model": args.model,
                    "experiment_name": args.experiment_name,
                    "best_epoch": int(epoch),
                    "best_val_acc": float(val_acc),
                    "best_val_loss": float(val_loss),
                    "best_val_metrics": make_json_safe(val_metrics),
                }
                torch.save(checkpoint_payload, str(checkpoint_path))
                print(f"New best model saved (Val Acc: {val_acc:.4f}) -> {checkpoint_path}")
            else:
                patience_counter += 1

            epoch_history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "train_accuracy": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                    "val_macro_precision": float(val_metrics["macro_precision"]),
                    "val_macro_recall": float(val_metrics["macro_recall"]),
                    "val_macro_f1": float(val_metrics["macro_f1"]),
                    "val_f1_macro": float(val_metrics["macro_f1"]),
                    "epoch_time_seconds": float(epoch_duration),
                    "learning_rate_current": current_lr,
                    "is_best": int(is_best),
                    "best_val_acc_so_far": float(best_val_acc),
                    "patience_counter": int(patience_counter),
                }
            )

            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        if use_mlflow:
            if best_val_acc != float("-inf"):
                mlflow.log_metric("best_val_acc", best_val_acc)
            if best_epoch is not None:
                mlflow.log_metric("best_epoch", best_epoch)

        if args.run_test and test_loader is not None:
            load_model_state_dict_from_checkpoint(model, checkpoint_path, device)
            if unknown_detector_model is not None:
                test_loss, test_acc, test_metrics, test_labels, test_preds = evaluate_with_unknown_detector(
                    model,
                    unknown_detector_model,
                    test_loader,
                    criterion,
                    device,
                    threshold=args.unknown_detector_threshold,
                    unknown_class_index=UNKNOWN_CLASS_INDEX,
                    measure_latency=True,
                )
            else:
                test_loss, test_acc, test_metrics, test_labels, test_preds = evaluate(
                    model,
                    test_loader,
                    criterion,
                    device,
                    measure_latency=True,
                    return_predictions=True,
                )
            print(
                f"Test: Acc={test_acc:.4f}, Loss={test_loss:.4f}, "
                f"Macro-Precision={test_metrics['macro_precision']:.4f}, "
                f"Macro-Recall={test_metrics['macro_recall']:.4f}, "
                f"Macro-F1={test_metrics['macro_f1']:.4f}"
            )

            confusion_payload, unknown_silence_payload = build_confusion_analysis(
                test_labels,
                test_preds,
                CLASS_NAMES,
            )
            if confusion_payload is not None:
                write_run_summary(confusion_payload, output_paths["confusion_matrix_json"])
                save_confusion_matrix_plot(
                    confusion_payload,
                    output_paths["confusion_matrix_png"],
                    title=f"Confusion Matrix ({args.model}, seed={args.seed})",
                )
                write_run_summary(
                    {
                        "unknown_silence_analysis": unknown_silence_payload,
                        "top_confusions": confusion_payload.get("top_confusions", []),
                    },
                    output_paths["error_analysis_json"],
                )

            test_payload = {
                "loss": float(test_loss),
                "acc": float(test_acc),
                "macro_precision": float(test_metrics["macro_precision"]),
                "macro_recall": float(test_metrics["macro_recall"]),
                "macro_f1": float(test_metrics["macro_f1"]),
                "inference_time_sec": float(test_metrics["inference_time_sec"]),
                "inference_latency_ms": float(test_metrics["inference_latency_ms"]) if test_metrics["inference_latency_ms"] is not None else None,
                "used_separate_unknown_detector": bool(unknown_detector_model is not None),
            }
            if unknown_detector_model is not None:
                test_payload.update(
                    {
                        "unknown_detector_precision": float(test_metrics["unknown_detector_precision"]),
                        "unknown_detector_recall": float(test_metrics["unknown_detector_recall"]),
                        "unknown_detector_f1": float(test_metrics["unknown_detector_f1"]),
                        "unknown_detector_acc": float(test_metrics["unknown_detector_acc"]),
                        "unknown_detector_threshold": float(test_metrics["unknown_detector_threshold"]),
                        "unknown_detector_positive_rate": float(test_metrics["unknown_detector_positive_rate"]),
                        "overridden_to_unknown_count": int(test_metrics["overridden_to_unknown_count"]),
                        "overridden_to_unknown_rate": float(test_metrics["overridden_to_unknown_rate"]),
                    }
                )
            if use_mlflow:
                metrics_to_log = {
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_macro_precision": test_metrics["macro_precision"],
                    "test_macro_recall": test_metrics["macro_recall"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_inference_latency_ms": test_metrics["inference_latency_ms"] if test_metrics["inference_latency_ms"] is not None else 0.0,
                }
                if unknown_detector_model is not None:
                    metrics_to_log.update(
                        {
                            "test_unknown_detector_precision": test_metrics["unknown_detector_precision"],
                            "test_unknown_detector_recall": test_metrics["unknown_detector_recall"],
                            "test_unknown_detector_f1": test_metrics["unknown_detector_f1"],
                            "test_unknown_detector_acc": test_metrics["unknown_detector_acc"],
                            "test_unknown_detector_positive_rate": test_metrics["unknown_detector_positive_rate"],
                            "test_overridden_to_unknown_rate": test_metrics["overridden_to_unknown_rate"],
                        }
                    )
                mlflow.log_metrics(metrics_to_log)

        total_time = time.time() - start_time
        if use_mlflow:
            mlflow.log_metric("total_training_time_sec", total_time)

        write_run_config(args, output_paths["config_yaml"])
        write_epoch_history_csv(epoch_history, output_paths["history_csv"])

        epoch_times = [row["epoch_time_seconds"] for row in epoch_history]
        summary_payload = {
            "run_id": run_id,
            "model_name": args.model,
            "optimizer": args.optimizer,
            "learning_rate": float(args.lr),
            "dropout": float(args.dropout),
            "batch_size": int(args.batch_size),
            "device": device.type,
            "model_parameters": {
                "total": int(total_params),
                "trainable": int(trainable_param_count),
            },
            "model": args.model,
            "experiment_name": args.experiment_name,
            "seed": int(args.seed),
            "device_name": gpu_name,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "val_accuracy_best_epoch": float(best_val_acc) if best_val_acc != float("-inf") else None,
            "test_accuracy": float(test_payload["acc"]) if test_payload is not None else None,
            "test_f1_macro": float(test_payload["macro_f1"]) if test_payload is not None else None,
            "training_time_seconds": float(total_time),
            "inference_latency_ms": float(test_payload["inference_latency_ms"]) if test_payload is not None and test_payload.get("inference_latency_ms") is not None else None,
            "run_output_dir": str(run_output_dir),
            "artifacts": {key: str(path_obj) for key, path_obj in output_paths.items()},
            "dataset_sizes": {
                "train": len(train_ds),
                "val": len(val_ds),
                "test": len(test_loader.dataset) if test_loader is not None else 0,
            },
            "epochs": {
                "configured": int(args.epochs),
                "ran": int(len(epoch_history)),
                "early_stopped": bool(len(epoch_history) < args.epochs),
                "best_epoch": int(best_epoch) if best_epoch is not None else None,
            },
            "best_validation": {
                "acc": float(best_val_acc) if best_val_acc != float("-inf") else None,
                "loss": float(best_val_loss) if best_val_loss is not None else None,
                "macro_precision": float(best_val_metrics["macro_precision"]) if best_val_metrics is not None else None,
                "macro_recall": float(best_val_metrics["macro_recall"]) if best_val_metrics is not None else None,
                "macro_f1": float(best_val_metrics["macro_f1"]) if best_val_metrics is not None else None,
            },
            "test": test_payload,
            "unknown_silence_analysis": unknown_silence_payload,
            "separate_unknown_detector": {
                "enabled": bool(args.use_separate_unknown_detector),
                "threshold": float(args.unknown_detector_threshold),
                "summary": unknown_detector_summary,
            },
            "timing": {
                "total_training_time_sec": float(total_time),
                "mean_epoch_time_seconds": float(np.mean(epoch_times)) if len(epoch_times) > 0 else 0.0,
                "max_epoch_time_seconds": float(np.max(epoch_times)) if len(epoch_times) > 0 else 0.0,
            },
            "args": make_json_safe(vars(args)),
        }
        write_run_summary(summary_payload, output_paths["summary_json"])

        print(f"Saved run config -> {output_paths['config_yaml']}")
        print(f"Saved epoch history -> {output_paths['history_csv']}")
        print(f"Saved run summary -> {output_paths['summary_json']}")
        if confusion_payload is not None:
            print(f"Saved confusion matrix JSON -> {output_paths['confusion_matrix_json']}")
            if plt is not None:
                print(f"Saved confusion matrix PNG -> {output_paths['confusion_matrix_png']}")
            print(f"Saved error analysis -> {output_paths['error_analysis_json']}")
        if unknown_detector_summary is not None:
            print(f"Saved unknown detector checkpoint -> {output_paths['unknown_detector_checkpoint']}")
            print(f"Saved unknown detector summary -> {output_paths['unknown_detector_summary_json']}")
        print(f"Training finished in {total_time:.2f} seconds.")

        return {
            "seed": int(args.seed),
            "experiment_name": args.experiment_name,
            "run_output_dir": str(run_output_dir),
            "summary_path": str(output_paths["summary_json"]),
            "summary": summary_payload,
        }


def load_yaml_config(config_path):
    """Load a YAML config file and return it as a dict.

    Applies a backward-compatibility shim: if the file contains a scalar ``seed``
    key but no ``seeds`` list, it is converted to ``seeds: [seed]``.
    """
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level key-value mapping")

    if "seed" in config and "seeds" not in config:
        config["seeds"] = [int(config["seed"])]
    config.pop("seed", None)

    return config


def build_parser():
    """Build and return the argument parser for the CNN training script."""
    parser = argparse.ArgumentParser(description="Speech Commands Training Script")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')

    available_models = get_available_models()

    parser.add_argument('--data_path', type=str, default='.//data//train', help='Path to dataset root')
    parser.add_argument('--experiment_name', type=str, default='baseline', help='Experiment name used in output directory')
    parser.add_argument('--model', type=str, default='resnet18', choices=available_models, help=f"Model name ({', '.join(available_models)})")

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer choice')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout before classifier head')
    parser.add_argument('--n_mels', type=int, default=64, choices=[64, 128], help='Mel filterbank bins')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], help='List of random seeds; each seed runs a full experiment')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze all layers except modified conv1 and fc')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pre-trained weights')

    parser.add_argument('--use_augment', action='store_true', help='Enable SpecAugment')
    parser.add_argument('--augment', type=str, choices=['true', 'false', 'True', 'False'], help='Compatibility flag: set augmentation on/off')
    parser.add_argument('--time_mask', type=int, default=20, help='SpecAugment time mask')
    parser.add_argument('--freq_mask', type=int, default=8, help='SpecAugment frequency mask')
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use weighted CrossEntropy')
    parser.add_argument('--use_unknown_undersampling', action='store_true', help='Undersample unknown class in train split')
    parser.add_argument('--unknown_keep_prob', type=float, default=0.35, help='Keep probability for unknown class when undersampling is enabled')
    parser.add_argument('--use_separate_unknown_detector', action='store_true', help='Train a separate binary network for unknown-vs-rest and use it to override final predictions')
    parser.add_argument('--unknown_detector_model', type=str, default=None, choices=available_models, help='Model for separate unknown detector (defaults to main model)')
    parser.add_argument('--unknown_detector_epochs', type=int, default=8, help='Epochs for separate unknown detector training')
    parser.add_argument('--unknown_detector_lr', type=float, default=None, help='Learning rate for separate unknown detector (defaults to --lr)')
    parser.add_argument('--unknown_detector_dropout', type=float, default=None, help='Dropout for separate unknown detector (defaults to --dropout)')
    parser.add_argument('--unknown_detector_threshold', type=float, default=0.5, help='Probability threshold for predicting unknown in separate detector')
    parser.add_argument('--balancing', type=str, default='none', choices=['none', 'loss', 'undersample', 'loss+undersample'], help='Class balancing strategy')
    parser.add_argument('--silence_train_samples', type=int, default=2300, help='Synthetic silence samples for train')
    parser.add_argument('--silence_eval_samples', type=int, default=250, help='Synthetic silence samples for validation')
    parser.add_argument('--include_silence_in_test', action='store_true', help='Include synthetic silence samples in test split')
    parser.add_argument('--silence_test_samples', type=int, default=250, help='Synthetic silence samples for test split when --include_silence_in_test is enabled')
    parser.add_argument('--run_test', action='store_true', help='Evaluate the best checkpoint on official test split')
    parser.add_argument('--checkpoint_path', type=str, default='best_model.pt', help='Base checkpoint filename (saved under outputs/{model}/{experiment_name} with automatic seed suffix)')
    parser.add_argument('--disable_mlflow', action='store_true', help='Disable MLflow logging')

    return parser


def parse_args_with_config():
    """Parse CLI arguments, applying YAML config file defaults when ``--config`` is supplied."""
    parser = build_parser()

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    if config_args.config:
        config_values = load_yaml_config(config_args.config)
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config_values.keys()) - valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")
        parser.set_defaults(**config_values)

    return parser.parse_args()


def expand_seed_runs(args):
    """Expand a multi-seed args namespace into a list of per-seed namespaces.

    When multiple seeds are specified, each seed's namespace gets its own
    ``experiment_name`` suffixed with ``_seed{seed}`` so output directories
    remain distinct.
    """
    seeds = args.seeds if args.seeds is not None else [42]
    if len(seeds) == 0:
        raise ValueError("seeds must contain at least one value")
    seeds = [int(seed) for seed in seeds]

    runs = []
    for seed in seeds:
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = seed

        if len(seeds) > 1:
            # Each seed gets its own output directory
            run_args.experiment_name = f"{args.experiment_name}_seed{seed}"

        runs.append(run_args)

    return runs

if __name__ == "__main__":
    args = parse_args_with_config()

    if args.augment is not None:
        args.use_augment = args.augment.lower() == 'true'
    if args.balancing in {'loss', 'loss+undersample'}:
        args.use_weighted_loss = True
    if args.balancing in {'undersample', 'loss+undersample'}:
        args.use_unknown_undersampling = True

    runs = expand_seed_runs(args)
    if len(runs) > 1:
        print(f"Multi-seed mode enabled. Running seeds: {[run.seed for run in runs]}")

    run_results = []
    for index, run_args in enumerate(runs, start=1):
        if len(runs) > 1:
            print(f"Seed run {index}/{len(runs)} (seed={run_args.seed})")
        set_seed(run_args.seed)
        run_result = run_experiment(run_args)
        run_results.append(run_result)

    aggregate_summary_path = save_multi_seed_summary(args, run_results)
    if aggregate_summary_path is not None:
        print(f"Saved multi-seed aggregate summary -> {aggregate_summary_path}")