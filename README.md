# Speech Commands Classification with Transformers

**Project II: Deep Learning** — Comparison of CNN and Transformer architectures for speech command recognition.

**Authors**: 

* [Anna Ostrowska ](https://github.com/annaostrowska03)
* [Igor Rudolf](https://github.com/IgorRudolf)
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
- `configs/resnet18_full_baseline_no_audio_tweaks.yaml`
- `configs/resnet18_full_batch32.yaml`
- `configs/resnet18_full_lr0003.yaml`
- `configs/resnet18_full_unknown_detector.yaml`

`resnet18_full_baseline_no_audio_tweaks.yaml` is a comparison baseline that uses
`model: resnet18_no_audio_tweaks` (no audio-specific stem adaptation; 1-channel input is repeated to 3 channels).

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

### Audio Spectrogram Transformer (AST)

AST training uses HuggingFace `AutoFeatureExtractor` on raw 1-second waveforms. It does not use the
`torchaudio` MelSpectrogram preprocessing path used by ResNet/MobileNet.

Run a single AST full experiment:

```bash
python src/train_ast.py --config configs/ast_full_baseline.yaml
```

Run all AST full configs:

```bash
# Linux/Mac
bash scripts/run_ast_configs.sh
```

```powershell
# Windows PowerShell
./scripts/run_ast_configs.ps1
```

Full AST configs use `seeds: [42, 123, 2026, 2137]`.

AST outputs are saved per seed in:
- `outputs/ast/{experiment_name}_seed{seed}/`

Aggregate summaries are saved in:
- `outputs/ast/{experiment_name}/summary_all_seeds.json`

If GPU memory is insufficient, reduce `batch_size` from `16` to `8` in the AST YAML configs.

### Full experiment orchestration

Full training should be run on a CUDA GPU machine.

Before full GPU training, run:

```bash
python scripts/check_project_ready.py
```

Inspect what is already completed:

```bash
python scripts/list_experiment_status.py
```

Run all missing main full experiments on Linux/Mac:

```bash
bash scripts/run_all_full_experiments.sh
```

Include optional experiments:

```bash
bash scripts/run_all_full_experiments.sh --include-optional
```

Force rerunning even completed experiments:

```bash
bash scripts/run_all_full_experiments.sh --force
```

On Windows PowerShell:

```powershell
./scripts/run_all_full_experiments.ps1
./scripts/run_all_full_experiments.ps1 -IncludeOptional
./scripts/run_all_full_experiments.ps1 -Force
```

Rebuild report tables without retraining:

```bash
bash scripts/collect_results.sh
```

or:

```powershell
./scripts/collect_results.ps1
```

If AST GPU memory is insufficient, reduce `batch_size` from `16` to `8` in the AST YAML configs.

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

When `--output_json` is provided, the script also exports confusion matrix artifacts next to the payload:
- single mode: `*_confusion_matrix.json` and `*_confusion_matrix.png`
- compare mode: `*_official_confusion_matrix.json/.png` and `*_with_silence_confusion_matrix.json/.png`

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
