import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from dataset import SpeechCommandsDataset
from models import get_model

try:
    from transformers import AutoFeatureExtractor, get_cosine_schedule_with_warmup
except ImportError:
    AutoFeatureExtractor = None
    get_cosine_schedule_with_warmup = None

try:
    import yaml
except ImportError:
    yaml = None


CLASS_NAMES = SpeechCommandsDataset.TARGET_WORDS + ["unknown", "silence"]
NUM_CLASSES = 12


def require_transformers_for_ast():
    if AutoFeatureExtractor is None or get_cosine_schedule_with_warmup is None:
        raise ImportError(
            "The 'transformers' package is required to train AST. "
            "Install it with `pip install transformers`."
        )


def load_yaml_config(config_path):
    if config_path is None:
        return {}
    if yaml is None:
        raise ImportError("PyYAML is required to use --config. Install it with `pip install pyyaml`.")
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must contain a mapping of argument names to values: {config_path}")
    return config


def save_yaml_config(config, output_path):
    if yaml is None:
        raise ImportError("PyYAML is required to save YAML config files.")
    with open(output_path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, sort_keys=True)


def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Train HuggingFace AST on Speech Commands")
    parser.add_argument("--config", type=str, default=defaults.get("config"))
    parser.add_argument("--experiment_name", type=str, default=defaults.get("experiment_name", "ast_full_baseline"))
    parser.add_argument("--model", type=str, choices=["ast"], default=defaults.get("model", "ast"))
    parser.add_argument("--data_path", type=str, default=defaults.get("data_path", "./data/train"))
    parser.add_argument("--output_dir", type=str, default=defaults.get("output_dir", "outputs/ast"))
    parser.add_argument("--seeds", type=int, nargs="+", default=defaults.get("seeds", [42]))

    parser.add_argument("--model_name", type=str, default=defaults.get("model_name", "MIT/ast-finetuned-audioset-10-10-0.4593"))
    parser.add_argument(
        "--feature_extractor_name",
        type=str,
        default=defaults.get("feature_extractor_name", "MIT/ast-finetuned-audioset-10-10-0.4593"),
    )
    parser.add_argument("--use_pretrained", action=argparse.BooleanOptionalAction, default=defaults.get("use_pretrained", True))
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=defaults.get("freeze_backbone", False))
    parser.add_argument("--dropout", type=float, default=defaults.get("dropout", 0.1))

    parser.add_argument("--lr", type=float, default=defaults.get("lr", 5e-5))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 0.01))
    parser.add_argument("--warmup_ratio", type=float, default=defaults.get("warmup_ratio", 0.1))
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 16))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 50))
    parser.add_argument("--patience", type=int, default=defaults.get("patience", 5))
    parser.add_argument("--num_workers", type=int, default=defaults.get("num_workers", 4))
    parser.add_argument("--run_test", action=argparse.BooleanOptionalAction, default=defaults.get("run_test", True))

    parser.add_argument(
        "--balancing",
        type=str,
        choices=["none", "loss", "undersample", "loss+undersample"],
        default=defaults.get("balancing", "none"),
    )
    parser.add_argument("--unknown_keep_prob", type=float, default=defaults.get("unknown_keep_prob", 0.35))
    parser.add_argument("--silence_train_samples", type=int, default=defaults.get("silence_train_samples", 2300))
    parser.add_argument("--silence_eval_samples", type=int, default=defaults.get("silence_eval_samples", 250))
    parser.add_argument("--include_silence_in_test", action=argparse.BooleanOptionalAction, default=defaults.get("include_silence_in_test", False))
    parser.add_argument("--silence_test_samples", type=int, default=defaults.get("silence_test_samples", 250))

    parser.add_argument("--use_mlflow", action=argparse.BooleanOptionalAction, default=defaults.get("use_mlflow", False))
    parser.add_argument("--mlflow_experiment", type=str, default=defaults.get("mlflow_experiment", "speech-commands-ast"))
    return parser


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining_args = config_parser.parse_known_args()
    defaults = load_yaml_config(config_args.config)
    defaults["config"] = config_args.config
    parser = build_parser(defaults)
    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(defaults.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {unknown_keys}")
    args = parser.parse_args(remaining_args)
    if isinstance(args.seeds, int):
        args.seeds = [args.seeds]
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if not (0.0 <= args.warmup_ratio <= 1.0):
        raise ValueError("warmup_ratio must be in [0, 1]")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device, torch.cuda.get_device_name(0)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def count_model_parameters(model):
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total_params), int(trainable_params)


def get_class_weights(labels, num_classes=NUM_CLASSES):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32)


def make_collate_fn(feature_extractor):
    def collate_fn(batch):
        waveforms, labels = zip(*batch)
        waveform_arrays = [
            waveform.detach().cpu().numpy() if torch.is_tensor(waveform) else np.asarray(waveform)
            for waveform in waveforms
        ]
        features = feature_extractor(
            waveform_arrays,
            sampling_rate=16000,
            return_tensors="pt",
        )
        return features["input_values"], torch.tensor(labels, dtype=torch.long)

    return collate_fn


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for input_values, labels in dataloader:
        input_values = input_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_values=input_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, dataloader, criterion, device, measure_latency=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    inference_seconds = 0.0

    with torch.no_grad():
        for input_values, labels in dataloader:
            input_values = input_values.to(device)
            labels = labels.to(device)

            if measure_latency and device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            logits = model(input_values=input_values)
            if measure_latency and device.type == "cuda":
                torch.cuda.synchronize()
            if measure_latency:
                inference_seconds += time.perf_counter() - start_time

            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            total += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )
    latency_ms = (inference_seconds / total) * 1000.0 if measure_latency and total > 0 else None
    metrics = {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    return metrics, all_labels, all_preds, latency_ms


def save_history(history, output_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_macro_precision",
        "val_macro_recall",
        "val_macro_f1",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_confusion_outputs(labels, preds, output_dir, seed):
    matrix = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    payload = {
        "labels": CLASS_NAMES,
        "matrix": matrix.tolist(),
    }
    json_path = output_dir / f"confusion_matrix_seed{seed}.json"
    png_path = output_dir / f"confusion_matrix_seed{seed}.png"
    with open(json_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("AST Speech Commands Confusion Matrix")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def maybe_start_mlflow(args, seed, device, device_name):
    if not args.use_mlflow:
        return None
    try:
        import mlflow
    except ImportError:
        print("MLflow requested but unavailable; continuing without MLflow.")
        return None

    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.start_run(run_name=f"{args.experiment_name}_seed{seed}")
    mlflow.log_params(
        {
            "seed": seed,
            "model_name": args.model_name,
            "feature_extractor_name": args.feature_extractor_name,
            "lr": args.lr,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "device": str(device),
            "device_name": device_name,
            "balancing": args.balancing,
        }
    )
    return mlflow


def build_dataloaders(args, feature_extractor):
    use_unknown_undersampling = args.balancing in {"undersample", "loss+undersample"}
    train_unknown_keep_prob = args.unknown_keep_prob if use_unknown_undersampling else 1.0
    pin_memory = torch.cuda.is_available()
    collate_fn = make_collate_fn(feature_extractor)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_fn,
    }

    train_ds = SpeechCommandsDataset(
        root_dir=args.data_path,
        split="train",
        silence_train_samples=args.silence_train_samples,
        silence_eval_samples=args.silence_eval_samples,
        unknown_keep_prob=train_unknown_keep_prob,
        return_waveform=True,
    )
    val_ds = SpeechCommandsDataset(
        root_dir=args.data_path,
        split="val",
        silence_train_samples=args.silence_train_samples,
        silence_eval_samples=args.silence_eval_samples,
        return_waveform=True,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    test_loader = None
    if args.run_test:
        test_ds = SpeechCommandsDataset(
            root_dir=args.data_path,
            split="test",
            silence_train_samples=args.silence_train_samples,
            silence_eval_samples=args.silence_eval_samples,
            include_silence_in_test=args.include_silence_in_test,
            silence_test_samples=args.silence_test_samples,
            return_waveform=True,
        )
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_ds, train_loader, val_loader, test_loader


def run_seed(args, seed):
    require_transformers_for_ast()
    set_seed(seed)
    device, device_name = get_device()
    run_id = f"{args.experiment_name}_seed{seed}"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.feature_extractor_name)
    train_ds, train_loader, val_loader, test_loader = build_dataloaders(args, feature_extractor)

    model = get_model(
        args.model,
        num_classes=NUM_CLASSES,
        use_pretrained=args.use_pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
        model_name=args.model_name,
    ).to(device)
    total_params, trainable_params = count_model_parameters(model)

    if args.balancing in {"loss", "loss+undersample"}:
        criterion = nn.CrossEntropyLoss(weight=get_class_weights(train_ds.labels).to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    mlflow = maybe_start_mlflow(args, seed, device, device_name)
    try:
        best_val_acc = float("-inf")
        best_epoch = None
        patience_counter = 0
        history = []
        start_time = time.time()
        checkpoint_path = output_dir / f"best_model_seed{seed}.pt"

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            history.append(row)
            save_history(history, output_dir / f"history_seed{seed}.csv")

            if mlflow is not None:
                mlflow.log_metrics(row, step=epoch)

            print(
                f"[seed {seed}] epoch {epoch:03d}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_accuracy": best_val_acc,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[seed {seed}] early stopping after epoch {epoch}")
                    break

        training_time_seconds = time.time() - start_time
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = None
        inference_latency_ms = None
        if args.run_test and test_loader is not None:
            test_metrics, test_labels, test_preds, inference_latency_ms = evaluate(
                model,
                test_loader,
                criterion,
                device,
                measure_latency=True,
            )
            save_confusion_outputs(test_labels, test_preds, output_dir, seed)
            if mlflow is not None:
                mlflow_metrics = {
                    "test_accuracy": test_metrics["accuracy"],
                    "test_f1_macro": test_metrics["macro_f1"],
                }
                if inference_latency_ms is not None:
                    mlflow_metrics["inference_latency_ms"] = inference_latency_ms
                mlflow.log_metrics(mlflow_metrics)

        config_payload = vars(args).copy()
        config_payload["seed"] = seed
        save_yaml_config(config_payload, output_dir / f"config_seed{seed}.yaml")

        summary = {
            "run_id": run_id,
            "model_name": args.model_name,
            "seed": seed,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "device": str(device),
            "device_name": device_name,
            "best_epoch": best_epoch,
            "val_accuracy_best_epoch": best_val_acc,
            "test_accuracy": test_metrics["accuracy"] if test_metrics else None,
            "test_f1_macro": test_metrics["macro_f1"] if test_metrics else None,
            "training_time_seconds": training_time_seconds,
            "inference_latency_ms": inference_latency_ms,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
        with open(output_dir / f"summary_seed{seed}.json", "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2)

        if mlflow is not None:
            mlflow.log_dict(summary, f"summary_seed{seed}.json")

        return summary
    finally:
        if mlflow is not None:
            mlflow.end_run()


def summarize_metric(summaries, metric_name):
    values = [summary[metric_name] for summary in summaries if summary.get(metric_name) is not None]
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def save_aggregate_summary(args, summaries):
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_name": args.experiment_name,
        "seeds": [summary["seed"] for summary in summaries],
        "runs": summaries,
        "test_accuracy": summarize_metric(summaries, "test_accuracy"),
        "test_f1_macro": summarize_metric(summaries, "test_f1_macro"),
    }
    with open(output_dir / "summary_all_seeds.json", "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def main():
    args = parse_args()
    summaries = []
    for seed in args.seeds:
        summaries.append(run_seed(args, seed))
    save_aggregate_summary(args, summaries)


if __name__ == "__main__":
    main()
