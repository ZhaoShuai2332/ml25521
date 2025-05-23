# ml25521
## Description:
周五下午了，想想晚上吃什么。
## Usage:
### 1. Fetch project:
You can cloning this project form [here](https://github.com/ZhaoShuai2332/ml25521.git).
### 2. Execution environment:
Skipping to the root path of the project and create python environment (Python 3.10+ is better).
### 3. Confirm you have installed pip commend and execute this commend:

```shell pip install -r piplist.txt ``` 
### 4. Training models:
 * This project supports training mnist, uci and credit datasets on Linear Regression and SVR.
 * If there is no other modifies of models, you can execute commend on this format on the root path of the project:

    ```shell python models/<model_name>.py --name <dataset_name>```

    Where the ```model_name``` can be setted as ```svr``` or ```linear```;
    The ```dataset_name``` can be setted as ```mnist```, ```uci```and ```credit```
### 5. Loading models parameters:
* The peremeters of model training are saved on ```model_params```, you can load model parameters by using numpy:
    ```python
    import numpy as np
    import os, sys

    self.params_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_params", f"{dataset_name}_{model_name}_params.npz")
    params = np.load(params_path)
    ```
* Also you can load scaler parameters by:
    ```python
    import numpy as np
    import os, sys

    self.params_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_params", f"{dataset_name}_scaler_params.npz")
    params = np.load(params_path)
    ```


