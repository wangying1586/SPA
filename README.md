# Spherical Procrustes Alignment for Reliable Medical Audio Diagnosis

This repository adds Spherical Procrustes Alignment (SPA) to the official PAFA codebase for ICBHI 2017 respiratory sound classification.

Supported tasks:

- 4-class: Normal, Crackle, Wheeze, Both
- 2-class: Normal, Abnormal
- Patient-wise 5-fold training and evaluation
- PAFA baseline and PAFA + SPA

## 1. Installation

### 1.1 Create the project

Download and extract this package, then run:

```bash
cd PAFA_SPA_ICBHI
bash bootstrap.sh "$HOME/PAFA-SPA"
cd "$HOME/PAFA-SPA"
```

The command clones the official PAFA repository and installs the SPA files into it.

To install the SPA files into an existing PAFA clone:

```bash
cd PAFA_SPA_ICBHI
python install_overlay.py --target /path/to/pafa
cd /path/to/pafa
```

### 1.2 Create the Python environment

```bash
conda env create -f environment.yml
conda activate pafa-spa
```

Install PyTorch 2.6 with CUDA 12.4:

```bash
pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining dependencies:

```bash
pip install -r requirements-spa.txt
```

Check the environment:

```bash
python - <<'PY'
import torch
import torchaudio

print("PyTorch:", torch.__version__)
print("Torchaudio:", torchaudio.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

## 2. BEATs Checkpoint

Download `BEATs_iter3+ (AS2M)` from the Microsoft BEATs repository:

https://github.com/microsoft/unilm/tree/master/beats

Place the checkpoint at:

```text
pretrained_models/BEATs_iter3_plus_AS2M.pt
```

Create the directory and copy the checkpoint:

```bash
mkdir -p pretrained_models
cp /path/to/BEATs_iter3_plus_AS2M.pt \
  pretrained_models/BEATs_iter3_plus_AS2M.pt
```

## 3. ICBHI 2017 Dataset

Download the dataset:

```bash
mkdir -p downloads
wget --no-check-certificate \
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip \
  -O downloads/ICBHI_final_database.zip
```

Prepare the PAFA directory structure:

```bash
python tools/prepare_icbhi.py \
  --archive downloads/ICBHI_final_database.zip \
  --repo_root .
```

The prepared dataset must contain:

```text
data/icbhi_dataset/
├── audio_test_data/
├── metadata.txt
├── official_split.txt
└── patient_list_foldwise.txt
```

For an already extracted dataset:

```bash
python tools/prepare_icbhi.py \
  --source_dir /path/to/ICBHI_final_database \
  --repo_root .
```

Verify the installation:

```bash
python tools/smoke_test_spa.py
python -m unittest tests/test_spa.py -v
python tools/verify_install.py --repo_root .
```

## 4. Configuration

The shell scripts use the following defaults:

```text
GPU=0
SEED=1
EPOCHS=100
BATCH_SIZE=32
NUM_WORKERS=8
DATA_FOLDER=./data
SAVE_ROOT=./experiments
```

Override any value before the command:

```bash
GPU=1 BATCH_SIZE=16 NUM_WORKERS=4 \
  bash scripts/train_pafa_spa_4class_5fold.sh
```

The default SPA settings are:

```text
prototype momentum      0.99
geometric temperature  2.0
alignment loss weight  0.5
fusion weight           0.5
spherical scale         16.0
label smoothing         0.1
```

## 5. Patient-Wise 5-Fold Training and Evaluation

Run all commands from the repository root:

```bash
cd "$HOME/PAFA-SPA"
```

### 5.1 PAFA baseline: 4-class

```bash
bash scripts/train_pafa_4class_5fold.sh
bash scripts/eval_pafa_4class_5fold.sh
```

### 5.2 PAFA + SPA: 4-class

```bash
bash scripts/train_pafa_spa_4class_5fold.sh
bash scripts/eval_pafa_spa_4class_5fold.sh
```

### 5.3 PAFA baseline: 2-class

```bash
bash scripts/train_pafa_2class_5fold.sh
bash scripts/eval_pafa_2class_5fold.sh
```

### 5.4 PAFA + SPA: 2-class

```bash
bash scripts/train_pafa_spa_2class_5fold.sh
bash scripts/eval_pafa_spa_2class_5fold.sh
```

### 5.5 Run the complete comparison

4-class:

```bash
GPU=0 bash scripts/run_all_5fold.sh 4
```

2-class:

```bash
GPU=0 bash scripts/run_all_5fold.sh 2
```

### 5.6 Summarize the results

```bash
bash scripts/summarize_5fold.sh
```

The summary files are written to:

```text
experiments/summary/
├── runs.csv
├── summary.csv
└── summary.json
```

## 6. Referenced Open-Source Repositories

- PAFA: https://github.com/wa976/pafa
- BEATs: https://github.com/microsoft/unilm/tree/master/beats

## 7. Citation

```bibtex
@inproceedings{wang2026spherical,
title={Spherical Procrustes Alignment for Reliable Medical Audio Diagnosis},
author={Ying Wang and Guoheng Huang and Chan-Tong Lam and Xiaochen Yuan},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=czRY4Qj153}
}
```
