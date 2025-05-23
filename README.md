# 🚀 ml25521

It's Friday afternoon now.

---

## 📋 Table of Contents

- [🚀 ml25521](#-ml25521)
  - [📋 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [🗂 Repository Layout](#-repository-layout)
  - [⚡ Quick-Start](#-quick-start)
    - [1️⃣ Clone](#1️⃣clone)
    - [2️⃣ Create a Python 3.10+ virtual environment](#2️⃣create-a-python-310-virtual-environment)
    - [3️⃣ Install requirements](#3️⃣install-requirements)
  - [🏋️‍♀️ Model Training](#️️-model-training)
  - [🔄 Loading Saved Artifacts](#-loading-saved-artifacts)
  - [🛠 Dataset Utilities](#-dataset-utilities)
  - [🌐 Data Source](#-data-source)

---

## ✨ Features

- **Two classic regressors** – plain Linear Regression (`models/linear.py`) and Support Vector Regression (`models/svr.py`).  
- **Three datasets** – MNIST (sub-sampled to regression targets), a cleaned UCI bundle, and a synthetic Credit dataset.  
- **One-command training** that serialises both model weights and feature scalers to `model_params/`.  
- **Minimal dependency set** (see `piplist.txt`) to keep the environment footprint small.

---

## 🗂 Repository Layout

```
ml25521/
├── data/                # Datasets (not tracked in Git)
├── model_params/        # Saved *.npz weights & scalers
├── model_test/
│   └── commends/        # Evaluation & data-loading helpers
├── models/              # Training scripts (linear.py, svr.py)
├── piplist.txt          # Python dependencies
└── README.md            # Project documentation
```

---

## ⚡ Quick-Start

### 1️⃣ Clone

```bash
git clone https://github.com/ZhaoShuai2332/ml25521.git
cd ml25521
```

### 2️⃣ Create a Python 3.10+ virtual environment

```bash
python -m venv env
# Linux / macOS
source env/bin/activate
# Windows
env\Scripts\activate
```

### 3️⃣ Install requirements

```bash
pip install --upgrade pip
pip install -r piplist.txt
```

---

## 🏋️‍♀️ Model Training

```bash
python models/<model_name>.py --name <dataset_name>
```

| `<model_name>` | `<dataset_name>` |
| -------------- | ---------------- |
| `linear`       | `mnist` \| `uci` \| `credit` |
| `svr`          | `mnist` \| `uci` \| `credit` |

Artifacts are written to `model_params/` as:

```
{dataset_name}_{model_name}_params.npz
{dataset_name}_scaler_params.npz
```

---

## 🔄 Loading Saved Artifacts

```python
import os, numpy as np

root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

params = np.load(
    os.path.join(root, "model_params",
                 f"{dataset_name}_{model_name}_params.npz")
)

scaler_params = np.load(
    os.path.join(root, "model_params",
                 f"{dataset_name}_scaler_params.npz")
)
```

---

## 🛠 Dataset Utilities

```python
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model_test.commends import dataset_selecter

train_X, test_X, train_y, test_y = dataset_selecter(name="mnist")
```

Valid dataset names: `mnist`, `uci`, `credit`.

---

## 🌐 Data Source

Large raw datasets reside in **`data/datasets/`** but are *excluded* from version control to keep the repository lightweight.

But you can access them on my [onedrive](https://stummuac-my.sharepoint.com/:f:/r/personal/21901260_stu_mmu_ac_uk/Documents/datasets?csf=1&web=1&e=1Ki0Pk).

