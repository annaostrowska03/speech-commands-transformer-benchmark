import importlib
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

REQUIRED_SOURCE_FILES = [
    "src/train.py",
    "src/train_ast.py",
    "src/models.py",
    "src/dataset.py",
    "src/reporting.py",
]

REQUIRED_AST_CONFIGS = [
    "configs/ast_full_baseline.yaml",
    "configs/ast_full_frozen_backbone.yaml",
    "configs/ast_full_dropout_p03.yaml",
    "configs/ast_full_balancing_loss.yaml",
    "configs/ast_full_balancing_undersample.yaml",
    "configs/ast_full_balancing_loss_undersample.yaml",
]

OPTIONAL_RESNET_CONFIGS = [
    "configs/resnet18_full_baseline.yaml",
    "configs/resnet18_full_nmels128.yaml",
    "configs/resnet18_full_optimizer_sgd.yaml",
    "configs/resnet18_full_dropout_p03.yaml",
    "configs/resnet18_full_dropout_p05.yaml",
    "configs/resnet18_full_specaugment.yaml",
    "configs/resnet18_full_balancing_loss.yaml",
    "configs/resnet18_full_balancing_undersample.yaml",
    "configs/resnet18_full_balancing_loss_undersample.yaml",
]

REQUIRED_RUNNER_SCRIPTS = [
    "scripts/run_ast_configs.sh",
    "scripts/run_ast_configs.ps1",
]

REQUIRED_IMPORTS = [
    "torch",
    "torchaudio",
    "transformers",
    "sklearn",
    "yaml",
]


class CheckReport:
    def __init__(self):
        self.passed = []
        self.warnings = []
        self.errors = []

    def pass_(self, message):
        self.passed.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)


def check_required_files(report, paths, label, missing_is_error=True):
    for rel_path in paths:
        path = ROOT_DIR / rel_path
        if path.exists():
            report.pass_(f"{label} exists: {rel_path}")
        elif missing_is_error:
            report.error(f"Missing {label}: {rel_path}")
        else:
            report.warn(f"Missing optional {label}: {rel_path}")


def check_imports(report):
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            report.pass_(f"Python import available: {module_name}")
        except ImportError as exc:
            report.error(f"Missing Python import: {module_name} ({exc})")


def check_model_registry(report):
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    try:
        from models import get_available_models
    except Exception as exc:
        report.error(f"Could not import model registry from src/models.py: {exc}")
        return

    try:
        available_models = set(get_available_models())
    except Exception as exc:
        report.error(f"Could not read model registry: {exc}")
        return

    for model_name in ["ast", "resnet18", "mobilenetv2"]:
        if model_name in available_models:
            report.pass_(f"Model registered: {model_name}")
        else:
            report.error(f"Model missing from registry: {model_name}")


def check_dataset_structure(report):
    data_path = ROOT_DIR / "data" / "train"
    if not data_path.exists():
        report.warn("Dataset is not available locally: data/train does not exist")
        return

    report.pass_("Dataset path exists: data/train")

    audio_path = data_path / "audio"
    if audio_path.exists():
        report.pass_("Dataset audio directory exists: data/train/audio")
    else:
        report.warn("data/train exists, but data/train/audio does not exist")

    for split_file in ["validation_list.txt", "testing_list.txt"]:
        split_path = data_path / split_file
        if split_path.exists():
            report.pass_(f"Split file exists: data/train/{split_file}")
        else:
            report.warn(f"Split file not found: data/train/{split_file}")


def check_device(report):
    try:
        import torch
    except ImportError:
        report.warn("Skipping CUDA/MPS check because torch is not importable")
        return

    cuda_available = torch.cuda.is_available()
    report.pass_(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        try:
            report.pass_(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
        except Exception as exc:
            report.warn(f"CUDA is available, but GPU name could not be read: {exc}")

    if hasattr(torch.backends, "mps"):
        try:
            report.pass_(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")
        except Exception as exc:
            report.warn(f"MPS availability could not be checked: {exc}")


def print_section(title, items):
    print(f"\n{title} ({len(items)})")
    print("-" * (len(title) + 4 + len(str(len(items)))))
    if not items:
        print("None")
        return
    for item in items:
        print(f"- {item}")


def main():
    report = CheckReport()

    check_required_files(report, REQUIRED_SOURCE_FILES, "source file", missing_is_error=True)
    check_required_files(report, REQUIRED_AST_CONFIGS, "AST config", missing_is_error=True)
    check_required_files(report, OPTIONAL_RESNET_CONFIGS, "ResNet config", missing_is_error=False)
    check_required_files(report, REQUIRED_RUNNER_SCRIPTS, "runner script", missing_is_error=True)
    check_imports(report)
    check_model_registry(report)
    check_dataset_structure(report)
    check_device(report)

    print("Project readiness check")
    print(f"Repository root: {ROOT_DIR}")
    print_section("PASSED checks", report.passed)
    print_section("WARNINGS", report.warnings)
    print_section("ERRORS", report.errors)

    if report.errors:
        print("\nResult: NOT READY")
        return 1

    print("\nResult: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
