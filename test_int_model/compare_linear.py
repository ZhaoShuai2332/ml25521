import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    log_loss, brier_score_loss, matthews_corrcoef
)
from models.LinearModel import Linear
from model_test import commends

def get_regression() -> (np.ndarray, np.ndarray):
    '''Load dataset, train Linear model, and return predictions and true labels.'''
    args = commends.parse_args()
    train_X, test_X, train_y, test_y = commends.dataset_selecter(args.name)
    linear = Linear(args.name, train_X, train_y, test_X, test_y, load_integer=True)
    linear.fit()
    preds = linear.predict()
    # print("All prediction values:")
    # np.set_printoptions(threshold=np.inf)
    # print(preds) 
    return preds, test_y, args.name


def sign_bit(x: np.ndarray) -> np.ndarray:
    '''Extract MSB sign bit: 0 if x >= 0, 1 if x < 0.'''
    arr = np.asarray(x)
    bits = arr.dtype.itemsize * 8
    uint_type = f'uint{bits}'
    arr_uint = arr.view(uint_type)
    return ((arr_uint >> (bits - 1)) & 1).astype(np.uint8)


def logistic_pred(linear_pred: np.ndarray, t: float) -> (np.ndarray, np.ndarray):
    '''
    Compute sigmoid probabilities and binary decision using threshold t.
    Returns tuple (probabilities, binary_preds).
    '''
    # print(linear_pred)
    prob = 1 / (1 + np.exp(-linear_pred))
    # print(prob)
    binary = sign_bit(prob - t)
    # binary = sign_bit(prob - 0.5)
    return prob, binary


def f_logr_pred(linear_pred: np.ndarray, t: float) -> np.ndarray:
    '''
    Compute binary decision based on linear_pred and logit threshold log(t/(1-t)).
    Returns binary_preds.
    '''
    with np.errstate(divide='ignore'):
        thresh = np.log(t / (1.0 - t))
    return sign_bit(linear_pred - thresh)


def compare_metrics(linear_pred: np.ndarray, y: np.ndarray, t: float) -> pd.DataFrame:
    '''
    Compute and compare metrics for two methods:
      - Logistic: sigmoid + threshold t
      - F_logr:   linear_pred + logit threshold
    Returns DataFrame with metrics side by side.
    '''
    # Normalize true labels to numeric type to avoid mixed-type label errors
    y = np.asarray(y)
    try:
        y = y.astype(int)
    except Exception:
        pass
    prob, logi_bits = logistic_pred(linear_pred, t)
    flog_bits = f_logr_pred(linear_pred, t)

    methods = {'Logistic': logi_bits, 'F_logr': flog_bits}

    # Regression metrics
    reg_funcs = {
        'MSE':  lambda yt, yp: mean_squared_error(yt, yp),
        'RMSE': lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        'MAE':  lambda yt, yp: mean_absolute_error(yt, yp),
        'R²':   lambda yt, yp: r2_score(yt, yp),
    }

    # Classification metrics
    cls_funcs = {
        'Accuracy':      lambda yt, yp, p: accuracy_score(yt, yp),
        'Precision':     lambda yt, yp, p: precision_score(yt, yp, zero_division=0),
        'Recall':        lambda yt, yp, p: recall_score(yt, yp, zero_division=0),
        'F1 Score':      lambda yt, yp, p: f1_score(yt, yp, zero_division=0),
        'ROC AUC':       lambda yt, yp, p: roc_auc_score(yt, p),
        'Avg Precision': lambda yt, yp, p: average_precision_score(yt, p),
        'Log Loss':      lambda yt, yp, p: log_loss(yt, p),
        'Brier Loss':    lambda yt, yp, p: brier_score_loss(yt, p),
        'MCC':           lambda yt, yp, p: matthews_corrcoef(yt, yp),
        'L2_distance':   lambda yt, yp, p: np.linalg.norm(yt - yp),
    }

    rows = []
    for name, preds in methods.items():
        for m_name, func in reg_funcs.items():
            rows.append({'Method': name, 'Metric': m_name, 'Value': func(y, preds)})
        for m_name, func in cls_funcs.items():
            try:
                value = func(y, preds, prob)
            except ValueError:
                # Skip metrics incompatible with multiclass; set as NaN
                value = np.nan
            rows.append({'Method': name, 'Metric': m_name, 'Value': value})

    df = pd.DataFrame(rows)
    return df.pivot(index='Metric', columns='Method', values='Value')

if __name__ == '__main__':
    np.random.seed(42)
    
    preds, y, name = get_regression()
    thresholds = np.linspace(0.01, 0.99, 20)
    
    with open(f'{name}_Linear_comparison_results.md', 'w', encoding='utf-8') as f:
        f.write("# Linear Model Comparison Results\n\n")
        f.write(f"- Dataset: {name}\n")
        f.write(f"- Number of samples: {len(y)}\n")
        f.write(f"- Number of thresholds: {len(thresholds)}\n")
        f.write("- Threshold range: 0.01-0.99\n\n")
        
        for t in thresholds:
            df = compare_metrics(preds, y, t=t)
            f.write(f"\n### Threshold = {t:.2f}\n\n")
            try:
                f.write(df.to_markdown())
            except (ImportError, ValueError):
                f.write(df.to_string())
            f.write("\n")
            
