import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models_fix_point.fix_interface.fix_point_transfer import FixedPoint, parse_float, multiply_fixed_array, parse_fix
from models.LinearModel import Linear
from model_test import commends



def test_prase_float(float_array: np.ndarray):
    fixed_array = parse_float(float_array)
    # print(fixed_array)
    return fixed_array

# a = test_prase_float(np.array([-3.14, 2.71, -8.97, -4.58]))
# b = test_prase_float(np.array([2.71, -5.67, 9.01, 3.45]))
# c = multiply_fixed_array(a, b)
# print(np.array([-3.14, 2.71, -8.97, -4.58]) * np.array([2.71, -5.67, 9.01, 3.45]))
# print(parse_fix(c))
# print(f"diff: {np.array([-3.14, 2.71, -8.97, -4.58]) * np.array([2.71, -5.67, 9.01, 3.45]) - parse_fix(c)}")




