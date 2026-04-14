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

Full configs define `seeds: [42, 123, 2026, 2137]`, so one command runs all seeds.

To force single-seed run, pass one-element list to `--seeds`:

```bash
python src/train.py --config configs/resnet18_full_baseline.yaml --seeds 42
```

Select model from CLI (available names come from `src/models.py` registry):

```bash
python src/train.py --model resnet18 --config configs/resnet18_full_baseline.yaml
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

Run artifacts are saved per run in:
- `outputs/{model}/{experiment_name}/`

Per-seed files saved inside each run directory:
- `best_model_seed{seed}.pt` (best checkpoint)
- `history_seed{seed}.csv` (full epoch-by-epoch history)
- `summary_seed{seed}.json` (best metrics, test metrics, timing, dataset sizes, args)
- `config_seed{seed}.yaml` (resolved run config)

### Smoke sanity checks (fast)

Run quick debug experiment:

```bash
python src/train.py --config configs/resnet18_smoke_baseline.yaml
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