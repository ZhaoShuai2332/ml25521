import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

from models_fix_point.fix_interface.fix_point_transfer import FixedPoint, parse_float
import numpy as np

# print(FixedPoint(0.00000000000026897151, 64, from_float=True))
import os,sys, numpy as np

input_dir = os.path.join(parent_dir, "model_params")
output_dir = os.path.join(parent_dir, "models_fix_point", "fix_params")
os.makedirs(output_dir, exist_ok=True)

# dataset_names = ["mnist", "uci", "credit"]
# model_names = ["linear", "svr", 'min_max_scaler', 'scaler']
# params_keys = {
#     "linear": ["coef", "intercept"], 
#     "svr" : ["coef", "intercept", "suppost_vertors", "support", "n_support"], 
#     "min_max_scaler": ["min", "scale", "data_min", "data_max"],
#     "scaler": ["min", "scale", "mean", "n_samples"]
#     }

def parse_params_name(dataset_name : str, model_name : str, params_keys : list):
    params = np.load(os.path.join(input_dir, f"{dataset_name}_{model_name}_params.npz"))
    new_params = {}
    for k in params.files:
        arr = params[k]
        if k in params_keys:
            fixed_arr = parse_float(arr, frac_bits=32)
            # fixed_arr = arr
            new_params[k] = fixed_arr
            # print(f"Converted {k}: {fixed_arr}")
        else:
            new_params[k] = arr
    return new_params
                
        
