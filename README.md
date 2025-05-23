
# 📁 `ml25521` Project Documentation

## 📝 Description  
It's Friday afternoon now!

---

## 🚀 Quickstart Guide

### 1. Clone the Repository  
```bash
git clone https://github.com/ZhaoShuai2332/ml25521.git
cd ml25521
```

### 2. Setup Python Environment  
Use Python 3.10+ for best compatibility.

```bash
python -m venv env
source env/bin/activate     # For Unix/macOS  
env\Scripts\activate        # For Windows
```

### 3. Install Dependencies  
Ensure `pip` is updated:

```bash
pip install --upgrade pip
pip install -r piplist.txt
```

---

## 🧠 Model Training

### Supported Models  
- `linear`: Linear Regression  
- `svr`: Support Vector Regression  

### Supported Datasets  
- `mnist`  
- `uci`  
- `credit`

### Run Training  
```bash
python models/<model_name>.py --name <dataset_name>
```

Example:
```bash
python models/linear.py --name mnist
```

Trained models will be saved in the `model_params/` directory.

---

## 💾 Load Saved Parameters

### Load Model Parameters
```python
import numpy as np
import os

params_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "model_params", 
    f"{dataset_name}_{model_name}_params.npz"
)
params = np.load(params_path)
```

### Load Scaler Parameters
```python
scaler_params_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "model_params", 
    f"{dataset_name}_scaler_params.npz"
)
scaler_params = np.load(scaler_params_path)
```

---

## 📊 Dataset Loading

Use the data loading utilities provided under `model_test/commends`.

```python
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model_test.commends import dataset_selecter

train_X, test_X, train_y, test_y = dataset_selecter(name=<dataset_name>)
```

Valid `dataset_name` values:
- `mnist`
- `uci`
- `credit`

---

## 📁 Data Source

Training datasets are stored in `data/datasets`.  
**Note**: Due to file size, datasets are not uploaded to the repository.