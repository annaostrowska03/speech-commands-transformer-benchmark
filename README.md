# Speech Commands Classification with Transformers

**Project II: Deep Learning** — Comparison of CNN and Transformer architectures for speech command recognition.

**Authors**: 

- Anna Ostrowska 

- Igor Rudolf
---

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone https://github.com/annaostrowska03/DeepLearning_transformers.git
cd DeepLearning_transformers

python -m venv .venv
source .venv/Scripts/activate  # Windows
# or: source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Speech Commands Dataset

Download the data from https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data, place in \data folder.
The dataset is provided as two 7z archives:
- `data/train.7z` 
- `data/test.7z`

#### Extract (option A)

```python
import py7zr
from pathlib import Path

data_dir = Path("data")

# Extract train.7z
with py7zr.SevenZipFile(data_dir / "train.7z", "r") as archive:
    archive.extractall(path=data_dir)

# Extract test.7z
with py7zr.SevenZipFile(data_dir / "test.7z", "r") as archive:
    archive.extractall(path=data_dir)
```

#### Option B: Command Line (Windows)

If you have 7-Zip installed:

```bash
cd data
7z x train.7z
7z x test.7z
cd ..
```

---
## Training

### Experiments

Run a single full experiment:

```bash
python src/train.py --config configs/resnet18_full_baseline.yaml
```

Run all prepared full configs:

```powershell
./scripts/run_resnet_configs.ps1
```

Full configs define `seeds: [42, 123, 2026, 2137]`, so one command runs all seeds.

To force single-seed run, pass one-element list to `--seeds`:

```bash
python src/train.py --config configs/resnet18_full_baseline.yaml --seeds 42
```

Select model from CLI (available names come from `src/models.py` registry):

```bash
python src/train.py --model resnet18 --config configs/resnet18_full_baseline.yaml
```

Example for MobileNetV2:

```bash
python src/train.py --model mobilenetv2 --config configs/mobilenetv2_full_baseline.yaml
```

Or set model directly in YAML config:

```yaml
model: resnet18
```

Available full-experiment configs:
- `configs/resnet18_full_baseline.yaml`
- `configs/resnet18_full_nmels128.yaml`
- `configs/resnet18_full_optimizer_sgd.yaml`
- `configs/resnet18_full_dropout_p03.yaml`
- `configs/resnet18_full_dropout_p05.yaml`
- `configs/resnet18_full_specaugment.yaml`
- `configs/resnet18_full_balancing_loss.yaml`
- `configs/resnet18_full_balancing_undersample.yaml`
- `configs/resnet18_full_balancing_loss_undersample.yaml`

Optional full-experiment configs (additional ablations + MobileNetV2):
- `configs/mobilenetv2_full_baseline.yaml`
- `configs/mobilenetv2_full_specaugment.yaml`
- `configs/resnet18_full_batch32.yaml`
- `configs/resnet18_full_lr0003.yaml`
- `configs/resnet18_full_unknown_detector.yaml`

Run optional full configs:

```powershell
./scripts/run_optional_configs.ps1
```

Run artifacts are saved per run in:
- `outputs/{model}/{experiment_name}/`

Per-seed files saved inside each run directory:
- `best_model_seed{seed}.pt` (best checkpoint)
- `history_seed{seed}.csv` (full epoch-by-epoch history)
- `summary_seed{seed}.json` (best metrics, test metrics, timing, dataset sizes, args)
- `config_seed{seed}.yaml` (resolved run config)
- `confusion_matrix_seed{seed}.json` (raw + normalized confusion matrix, per-class metrics)
- `confusion_matrix_seed{seed}.png` (row-normalized confusion matrix plot)
- `error_analysis_seed{seed}.json` (top confusions + `unknown`/`silence` focused diagnostics)
- `unknown_detector_seed{seed}.pt` (separate binary unknown detector checkpoint, if enabled)
- `unknown_detector_summary_seed{seed}.json` (metrics/params for separate unknown detector, if enabled)

When running multiple seeds, an aggregate file is also generated:
- `outputs/{model}/{base_experiment_name}/summary_all_seeds.json` (mean/std/min/max across seeds + per-seed artifact index)

Build report-ready tables from saved outputs:

```bash
python src/reporting.py --outputs_dir outputs --analysis_dir outputs/analysis
```

Generated analysis files:
- `outputs/analysis/leaderboard.csv`
- `outputs/analysis/leaderboard.md`
- `outputs/analysis/leaderboard.json`
- `outputs/analysis/top_confusions.csv`
- `outputs/analysis/unknown_silence.csv`

### Smoke sanity checks (fast)

Run quick debug experiment:

```bash
python src/train.py --config configs/resnet18_smoke_baseline.yaml
```

Optional smoke configs:
- `configs/mobilenetv2_smoke_baseline.yaml`
- `configs/mobilenetv2_smoke_specaugment.yaml`
- `configs/resnet18_smoke_unknown_detector.yaml`

Run optional smoke configs:

```powershell
./scripts/run_optional_smoke.ps1
```

### Useful balancing modes

```bash
python src/train.py --epochs 10 --balancing loss

# unknown undersampling only
python src/train.py --epochs 10 --balancing undersample --unknown_keep_prob 0.35

# weighted loss + unknown undersampling
python src/train.py --epochs 10 --balancing loss+undersample --unknown_keep_prob 0.35
```

The script reports accuracy and macro metrics (`macro_precision`, `macro_recall`, `macro_f1`) for validation,
and for test if `--run_test` is enabled.

### Test split variants: with and without synthetic silence

By default, test split follows the official list only (no synthetic `silence`).
You can optionally add synthetic silence generated from `_background_noise_`:

```bash
python src/train.py --config configs/resnet18_full_baseline.yaml --run_test --include_silence_in_test --silence_test_samples 250
```

This lets you keep backward-compatible results and also report an alternative 12-class test setup.

### Evaluate already-trained checkpoints (no retraining)

Use a saved checkpoint and (optionally) saved run config to re-run test evaluation later:

```bash
python src/evaluate_checkpoint.py --config outputs/resnet18/resnet18_full_baseline_seed42/config_seed42.yaml --checkpoint_path outputs/resnet18/resnet18_full_baseline_seed42/best_model_seed42.pt
```

Compare both test approaches in one command:

```bash
python src/evaluate_checkpoint.py --config outputs/resnet18/resnet18_full_baseline_seed42/config_seed42.yaml --checkpoint_path outputs/resnet18/resnet18_full_baseline_seed42/best_model_seed42.pt --compare_silence_modes --output_json outputs/analysis/eval_seed42_compare_silence.json
```

### Separate Unknown Detector (optional)

Enable a second binary network for `unknown` vs `non-unknown` and use it to override the final class prediction:

```bash
python src/train.py --config configs/resnet18_full_baseline.yaml --use_separate_unknown_detector --unknown_detector_threshold 0.5
```

Key flags:
- `--use_separate_unknown_detector`
- `--unknown_detector_model` (defaults to main model)
- `--unknown_detector_epochs` (default: 8)
- `--unknown_detector_lr` (defaults to `--lr`)
- `--unknown_detector_dropout` (defaults to `--dropout`)
- `--unknown_detector_threshold` (default: 0.5)

### Reproducibility checklist

- Use config files committed to the repository.
- Run experiments with fixed multi-seed lists (default full configs: `[42, 123, 2026, 2137]`).
- Keep generated `summary_seed*.json` and `summary_all_seeds.json` as source of report numbers.
- Build final tables from raw outputs using `src/reporting.py`.