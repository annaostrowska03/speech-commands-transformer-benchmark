import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset import SpeechCommandsDataset
from models import get_available_models, get_model
from train import (
    CLASS_NAMES,
    UNKNOWN_CLASS_INDEX,
    build_confusion_analysis,
    evaluate,
    evaluate_with_unknown_detector,
    load_model_state_dict_from_checkpoint,
    make_json_safe,
    set_seed,
)


def load_yaml_mapping(config_path):
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level key-value mapping")
    return config


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate an existing checkpoint on Speech Commands test split"
    )

    available_models = get_available_models()

    parser.add_argument("--config", type=str, default=None, help="Path to config_seed*.yaml from a finished run")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data_path", type=str, default=".//data//train", help="Path to dataset root")

    parser.add_argument("--model", type=str, default="resnet18", choices=available_models, help="Model name")
    parser.add_argument("--n_mels", type=int, default=64, choices=[64, 128], help="Mel filterbank bins")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout before classifier head")
    parser.add_argument("--use_pretrained", action="store_true", help="Use pretrained model initialization")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone (kept for architecture parity)")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for test loader")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    parser.add_argument("--include_silence_in_test", action="store_true", help="Include synthetic silence in test split")
    parser.add_argument("--silence_eval_samples", type=int, default=250, help="Synthetic silence count for validation split")
    parser.add_argument("--silence_test_samples", type=int, default=250, help="Synthetic silence count for test split when enabled")
    parser.add_argument("--compare_silence_modes", action="store_true", help="Evaluate both modes: official test and test+synthetic-silence")

    parser.add_argument("--use_separate_unknown_detector", action="store_true", help="Use additional unknown detector during evaluation")
    parser.add_argument("--unknown_detector_model", type=str, default=None, choices=available_models, help="Model for unknown detector")
    parser.add_argument("--unknown_detector_dropout", type=float, default=None, help="Dropout for unknown detector model")
    parser.add_argument("--unknown_detector_checkpoint", type=str, default=None, help="Path to unknown detector checkpoint")
    parser.add_argument("--unknown_detector_threshold", type=float, default=0.5, help="Threshold for unknown detector override")

    parser.add_argument("--eval_seed", type=int, default=42, help="Seed for deterministic test-time silence sampling")
    parser.add_argument("--output_json", type=str, default=None, help="Optional JSON path for evaluation payload")

    return parser


def parse_args_with_optional_config():
    parser = build_parser()

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    if config_args.config:
        config_values = load_yaml_mapping(config_args.config)
        valid_keys = {action.dest for action in parser._actions}
        filtered_config = {
            key: value
            for key, value in config_values.items()
            if key in valid_keys
        }
        parser.set_defaults(**filtered_config)

    args = parser.parse_args()

    if args.checkpoint_path is None and args.config is not None:
        config_parent = Path(args.config).resolve().parent
        candidate_paths = sorted(config_parent.glob("best_model_seed*.pt"))
        if len(candidate_paths) == 1:
            args.checkpoint_path = str(candidate_paths[0])

    if args.checkpoint_path is None:
        parser.error("checkpoint path is required: pass --checkpoint_path or provide --config with a resolvable best_model_seed*.pt")

    if args.use_separate_unknown_detector and args.unknown_detector_checkpoint is None:
        checkpoint_parent = Path(args.checkpoint_path).resolve().parent
        candidate_paths = sorted(checkpoint_parent.glob("unknown_detector_seed*.pt"))
        if len(candidate_paths) == 1:
            args.unknown_detector_checkpoint = str(candidate_paths[0])

    if args.use_separate_unknown_detector and args.unknown_detector_checkpoint is None:
        parser.error("--use_separate_unknown_detector requires --unknown_detector_checkpoint (or auto-detection in checkpoint directory)")

    return args


def build_test_loader(args, include_silence_in_test):
    test_ds = SpeechCommandsDataset(
        root_dir=args.data_path,
        split="test",
        n_mels=args.n_mels,
        silence_eval_samples=args.silence_eval_samples,
        include_silence_in_test=include_silence_in_test,
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
    return test_loader


def build_main_model(args, device):
    model = get_model(
        args.model,
        num_classes=12,
        input_channels=1,
        use_pretrained=args.use_pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    ).to(device)
    return model


def build_unknown_detector_model(args, device):
    detector_model_name = args.unknown_detector_model if args.unknown_detector_model else args.model
    detector_dropout = args.unknown_detector_dropout if args.unknown_detector_dropout is not None else args.dropout

    detector_model = get_model(
        detector_model_name,
        num_classes=2,
        input_channels=1,
        use_pretrained=args.use_pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=detector_dropout,
    ).to(device)
    return detector_model


def evaluate_mode(args, include_silence_in_test):
    set_seed(int(args.eval_seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    test_loader = build_test_loader(args, include_silence_in_test=include_silence_in_test)

    model = build_main_model(args, device)
    load_model_state_dict_from_checkpoint(model, args.checkpoint_path, device)

    criterion = nn.CrossEntropyLoss()

    unknown_detector_model = None
    if args.use_separate_unknown_detector:
        unknown_detector_model = build_unknown_detector_model(args, device)
        load_model_state_dict_from_checkpoint(unknown_detector_model, args.unknown_detector_checkpoint, device)

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

    confusion_payload, unknown_silence_payload = build_confusion_analysis(
        test_labels,
        test_preds,
        CLASS_NAMES,
    )

    mode_name = "test_with_synthetic_silence" if include_silence_in_test else "official_test_without_silence"
    result = {
        "mode": mode_name,
        "include_silence_in_test": bool(include_silence_in_test),
        "test_size": int(len(test_loader.dataset)),
        "device": device.type,
        "device_name": gpu_name,
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "model": args.model,
        "n_mels": int(args.n_mels),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "test": {
            "loss": float(test_loss),
            "acc": float(test_acc),
            "macro_precision": float(test_metrics["macro_precision"]),
            "macro_recall": float(test_metrics["macro_recall"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "inference_time_sec": float(test_metrics["inference_time_sec"]),
            "inference_latency_ms": float(test_metrics["inference_latency_ms"]) if test_metrics["inference_latency_ms"] is not None else None,
            "used_separate_unknown_detector": bool(unknown_detector_model is not None),
        },
        "unknown_silence_analysis": unknown_silence_payload,
        "confusion_matrix": confusion_payload,
    }

    if unknown_detector_model is not None:
        result["test"].update(
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

    return result


def print_short_result(result):
    test_payload = result["test"]
    print(
        f"[{result['mode']}] "
        f"size={result['test_size']} | "
        f"acc={test_payload['acc']:.4f} | "
        f"macro_f1={test_payload['macro_f1']:.4f} | "
        f"latency_ms={test_payload['inference_latency_ms']:.4f}"
    )


if __name__ == "__main__":
    args = parse_args_with_optional_config()

    if args.compare_silence_modes:
        result_without_silence = evaluate_mode(args, include_silence_in_test=False)
        result_with_silence = evaluate_mode(args, include_silence_in_test=True)

        print_short_result(result_without_silence)
        print_short_result(result_with_silence)

        delta_acc = result_with_silence["test"]["acc"] - result_without_silence["test"]["acc"]
        delta_f1 = result_with_silence["test"]["macro_f1"] - result_without_silence["test"]["macro_f1"]
        print(f"[delta silence - official] acc={delta_acc:+.4f}, macro_f1={delta_f1:+.4f}")

        payload = {
            "compare_silence_modes": True,
            "official_test_without_silence": result_without_silence,
            "test_with_synthetic_silence": result_with_silence,
            "delta": {
                "acc": float(delta_acc),
                "macro_f1": float(delta_f1),
            },
        }
    else:
        single_result = evaluate_mode(args, include_silence_in_test=bool(args.include_silence_in_test))
        print_short_result(single_result)
        payload = {
            "compare_silence_modes": False,
            "result": single_result,
        }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(make_json_safe(payload), file_obj, indent=2)
        print(f"Saved evaluation payload -> {output_path}")
