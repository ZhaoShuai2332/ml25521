import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

from models_fix_point.fix_interface.fix_point_transfer import FixedPoint, parse_float, multiply_fixed_array, parse_fix
import numpy as np
import pandas as pd
from model_test.commends import dataset_selecter

def process_and_save_datasets():
    name_list = ["mnist", "uci", "credit"]
    # root_dir = os.path.dirname(os.path.dirname(__file__))

    for name in name_list:
        train_X, test_X, train_y, test_y = dataset_selecter(name)
        
        # Convert to float32 and then to fixed point
        datasets = [train_X, test_X, train_y, test_y]
        datasets = [parse_float(data.astype(np.float32)) for data in datasets]

        return datasets
    
def parse_float_data(name):
    train_X, test_X, train_y, test_y = dataset_selecter(name)
        
    datasets = [train_X, test_X, train_y, test_y]
    datasets = [parse_float(data.astype(np.float32)) for data in datasets]

    return datasets










