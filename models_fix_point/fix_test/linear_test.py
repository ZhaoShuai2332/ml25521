import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

from models_fix_point.fix_interface.fix_point_transfer import FixedPoint, parse_float, multiply_fixed_array, parse_fix
import numpy as np
from models_fix_point.fix_models.Linear_fix import Linear_fix
from models_fix_point.fix_params.fix_params import parse_params_name
from models_fix_point.fix_datasets.fix_data import parse_float_data
from model_test.commends import parse_args

dataset_name = parse_args().name
train_X, test_X, train_y, test_y = parse_float_data(dataset_name)
params = parse_params_name(dataset_name, "linear", ["coef", "intercept"])
linear_model = Linear_fix(train_X, train_y, test_X, test_y, params["coef"], params["intercept"])
print(parse_fix(linear_model.predict()))
