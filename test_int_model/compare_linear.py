import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.LinearModel import Linear
import numpy as np
from model_test import commends
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_regression() -> np.ndarray:
    args = commends.parse_args()
    train_X, test_X, train_y, test_y = commends.dataset_selecter(args.name)
    linear = Linear(args.name, train_X, train_y, test_X, test_y, load_integer=True) 
    linear.fit()
    res = linear.predict()
    return res, test_y

def logist_regression(linear_predit: np.ndarray) -> np.ndarray:
    res = linear_predit.copy()
    res = 1 / (1 + np.exp(-res))
    return res

def sign_bit(x: np.ndarray) -> np.ndarray:
    """
    Extract the sign bit (MSB) of each element in x.
    Works for signed integers and IEEE-754 floats.
    Returns an array of uint8 {0,1}: 0 if x>=0, 1 if x<0.
    """
    arr = np.asarray(x)
    bits = arr.dtype.itemsize * 8
    uint_type = f'uint{bits}'
    arr_uint = arr.view(uint_type)
    return ((arr_uint >> (bits - 1)) & 1).astype(np.uint8)


def f_logr(linear_pred: np.ndarray, t: float) -> np.ndarray:
    """
    Logistic‐regression "sign" prediction on each element of linear_pred,
    with decision threshold ln(t/(1-t)).
    
    Inputs:
      linear_pred : np.ndarray
        Your M-module's output w̃⊙z + b (can be int or float).
      t : float
        A probability in (0,1), e.g. 0.1, 0.5, 0.9…
    
    Returns:
      np.ndarray of uint8 {0,1}, same shape as linear_pred.
        0 ↦ non-negative (predict "class 1"), 
        1 ↦ negative       (predict "class 0").
    """
    thresh = np.log(t / (1.0 - t))
    diff = linear_pred - thresh
    return sign_bit(diff)

def compare_errors(linear_pred: np.ndarray, y: np.ndarray, t: float):
    log_pred = logist_regression(linear_pred)
    f_log = f_logr(linear_pred, t)
    
    print("\nOriginal Metrics:")
    mse = mean_squared_error(y, linear_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, linear_pred)
    r2 = r2_score(y, linear_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # Evaluate logistic regression predictions
    print("\nLogistic Regression Metrics:")
    mse = mean_squared_error(y, log_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, log_pred)
    r2 = r2_score(y, log_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # Evaluate f_logr predictions 
    print("\nF_logr Metrics:")
    mse = mean_squared_error(y, f_log)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, f_log)
    r2 = r2_score(y, f_log)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")



if __name__ == "__main__":
    pred, y  = get_regression()
    compare_errors(pred, y, 0.1)
    
    

