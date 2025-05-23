# ml25521

## Description  
It is Friday afternoon.

## Usage

### 1. Clone the Repository  
Clone the project repository from the following URL:

```bash
git clone https://github.com/ZhaoShuai2332/ml25521.git
```

### 2. Setup Execution Environment  
Navigate to the project root directory and create a Python virtual environment. It is recommended to use Python version 3.10 or higher for compatibility.

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### 3. Install Dependencies  
Ensure that `pip` is installed and updated, then install the required packages listed in `piplist.txt`:

```bash
pip install -r piplist.txt
```

### 4. Train Models  
This project supports training on three datasets — MNIST, UCI, and Credit — using either Linear Regression or SVR models.

To train a model, run the following command from the project root:

```bash
python models/<model_name>.py --name <dataset_name>
```

- `<model_name>` options:  
  - `linear` — for Linear Regression  
  - `svr` — for Support Vector Regression  

- `<dataset_name>` options:  
  - `mnist`  
  - `uci`  
  - `credit`

This command will execute the training process, saving model parameters upon completion.

### 5. Load Pre-trained Model Parameters  
Trained model parameters are saved in the `model_params` directory in `.npz` format. Use the following Python snippet to load model parameters:

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

### 6. Load Scaler Parameters  
To maintain consistent feature scaling, scaler parameters for each dataset are also saved in `model_params`. Load them with:

```python
import numpy as np
import os

scaler_params_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "model_params", 
    f"{dataset_name}_scaler_params.npz"
)
scaler_params = np.load(scaler_params_path)
```