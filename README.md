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