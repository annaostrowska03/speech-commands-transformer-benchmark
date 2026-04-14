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
from sklearn.metrics import precision_recall_fscore_support
import yaml

try:
    import mlflow
except ImportError:
    mlflow = None


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

def evaluate(model, dataloader, criterion, device, measure_latency=False):
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

    return running_loss / total, correct / total, metrics


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
    checkpoint_name = Path(args.checkpoint_path).name if args.checkpoint_path else "best_model.pt"
    checkpoint_name = add_seed_suffix_to_filename(checkpoint_name, args.seed)
    run_output_dir = Path("outputs") / args.model / args.experiment_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "checkpoint": run_output_dir / checkpoint_name,
        "history_csv": run_output_dir / f"history_seed{args.seed}.csv",
        "summary_json": run_output_dir / f"summary_seed{args.seed}.json",
        "config_yaml": run_output_dir / f"config_seed{args.seed}.yaml",
    }
    return run_output_dir, output_paths


def write_epoch_history_csv(history_rows, csv_path):
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
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary_payload, file_obj, indent=2)


def write_run_config(args, config_path):
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


def save_multi_seed_summary(base_args, run_results):
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
        },
        "per_seed_runs": per_seed_runs,
    }

    write_run_summary(aggregate_payload, aggregate_summary_path)
    return str(aggregate_summary_path)

def run_experiment(args):
    """
    Main experiment runner: setup, training loop, and logging.
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

    if args.use_weighted_loss:
        weights = get_class_weights(train_ds.labels, num_classes=12).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    trainable_params = filter(lambda parameter: parameter.requires_grad, model.parameters())
    if args.optimizer == "sgd":
        optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = float("-inf")
    best_val_loss = None
    best_val_metrics = None
    best_epoch = None
    patience_counter = 0
    start_time = time.time()
    epoch_history = []
    test_payload = None
    
    print(f"Starting training on {gpu_name}...")
    print(f"Run seed: {args.seed}")
    print(f"Unknown keep probability (train): {unknown_keep_prob:.3f}")
    print(f"Run outputs directory: {run_output_dir}")

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

            # Early Stopping
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
            test_loss, test_acc, test_metrics = evaluate(model, test_loader, criterion, device, measure_latency=True)
            print(
                f"Test: Acc={test_acc:.4f}, Loss={test_loss:.4f}, "
                f"Macro-Precision={test_metrics['macro_precision']:.4f}, "
                f"Macro-Recall={test_metrics['macro_recall']:.4f}, "
                f"Macro-F1={test_metrics['macro_f1']:.4f}"
            )
            test_payload = {
                "loss": float(test_loss),
                "acc": float(test_acc),
                "macro_precision": float(test_metrics["macro_precision"]),
                "macro_recall": float(test_metrics["macro_recall"]),
                "macro_f1": float(test_metrics["macro_f1"]),
                "inference_time_sec": float(test_metrics["inference_time_sec"]),
                "inference_latency_ms": float(test_metrics["inference_latency_ms"]) if test_metrics["inference_latency_ms"] is not None else None,
            }
            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "test_macro_precision": test_metrics["macro_precision"],
                        "test_macro_recall": test_metrics["macro_recall"],
                        "test_macro_f1": test_metrics["macro_f1"],
                        "test_inference_latency_ms": test_metrics["inference_latency_ms"] if test_metrics["inference_latency_ms"] is not None else 0.0,
                    }
                )

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
        print(f"Training finished in {total_time:.2f} seconds.")

        return {
            "seed": int(args.seed),
            "experiment_name": args.experiment_name,
            "run_output_dir": str(run_output_dir),
            "summary_path": str(output_paths["summary_json"]),
            "summary": summary_payload,
        }


def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level key-value mapping")

    # Backward compatibility for older configs that used `seed`.
    if "seed" in config and "seeds" not in config:
        config["seeds"] = [int(config["seed"])]
    config.pop("seed", None)

    return config


def build_parser():
    parser = argparse.ArgumentParser(description="Speech Commands Training Script")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')

    available_models = get_available_models()
    
    # Data params
    parser.add_argument('--data_path', type=str, default='.//data//train', help='Path to dataset root')
    parser.add_argument('--experiment_name', type=str, default='baseline', help='MLflow run name')
    parser.add_argument('--model', type=str, default='resnet18', choices=available_models, help=f"Model name ({', '.join(available_models)})")
    
    # Hyperparams
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer choice')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout before classifier head')
    parser.add_argument('--n_mels', type=int, default=64, choices=[64, 128], help='Mel filterbank bins')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], help='List of seeds for execution (can contain one element)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze all layers except modified conv1 and fc')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pre-trained weights')

    # Augmentation and balancing
    parser.add_argument('--use_augment', action='store_true', help='Enable SpecAugment')
    parser.add_argument('--augment', type=str, choices=['true', 'false', 'True', 'False'], help='Compatibility flag: set augmentation on/off')
    parser.add_argument('--time_mask', type=int, default=20, help='SpecAugment time mask')
    parser.add_argument('--freq_mask', type=int, default=8, help='SpecAugment frequency mask')
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use weighted CrossEntropy')
    parser.add_argument('--use_unknown_undersampling', action='store_true', help='Undersample unknown class in train split')
    parser.add_argument('--unknown_keep_prob', type=float, default=0.35, help='Keep probability for unknown class when undersampling is enabled')
    parser.add_argument('--balancing', type=str, default='none', choices=['none', 'loss', 'undersample', 'loss+undersample'], help='Compatibility flag for class balancing')
    parser.add_argument('--silence_train_samples', type=int, default=2300, help='Synthetic silence samples for train')
    parser.add_argument('--silence_eval_samples', type=int, default=250, help='Synthetic silence samples for validation')
    parser.add_argument('--run_test', action='store_true', help='Evaluate the best checkpoint on official test split')
    parser.add_argument('--checkpoint_path', type=str, default='best_model.pt', help='Base checkpoint filename (saved under outputs/{model}/{experiment_name} with automatic seed suffix)')
    parser.add_argument('--disable_mlflow', action='store_true', help='Disable MLflow logging')

    return parser


def parse_args_with_config():
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
    seeds = args.seeds if args.seeds is not None else [42]
    if len(seeds) == 0:
        raise ValueError("seeds must contain at least one value")
    seeds = [int(seed) for seed in seeds]

    runs = []
    for seed in seeds:
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = seed

        if len(seeds) > 1:
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