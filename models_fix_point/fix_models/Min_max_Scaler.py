import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

import numpy as np


class Min_Max_Scaler:
    def __init__(self, name):
        self.param_path = os.path.join(parent_dir, "model_params", f"{name}_min_max_scaler_params.npz")
        self.y_params_path = os.path.join(parent_dir, "model_params", f"{name}_y_min_max_scaler_params.npz")
        params_np = np.load(self.param_path)
        y_params_np = np.load(self.y_params_path)
        self.params = {
            "min": params_np["min"],
            "scale": params_np["scale"],
            "data_min": params_np["data_min"],
            "data_max": params_np["data_max"],
        }
        self.y_params = {
            "min": y_params_np["min"],
            "scale": y_params_np["scale"],
            "data_min": y_params_np["data_min"],
            "data_max": y_params_np["data_max"],
        }
    
    def scaler_x(self, x: np.ndarray):
        x_min = self.params["data_min"]
        x_scale = self.params["scale"]
        x_centered = x - x_min
        x_scaled = x_centered * x_scale
        
        return x_scaled
    
    def scaler_y(self, y: np.ndarray):
        y_min = self.y_params["data_min"]
        y_scale = self.y_params["scale"]
        y_centered = y - y_min
        y_scaled = y_centered * y_scale
        return y_scaled
        

    