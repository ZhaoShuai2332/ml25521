import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.metrics import accuracy_score
from models.LinearModel import Linear
from model_test.commends import parse_args, dataset_selecter
from sklearnex import patch_sklearn, unpatch_sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
patch_sklearn()

def linear(name: str):
    train_X, test_X, train_y, test_y = dataset_selecter(name)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    model = Linear(name, train_X, train_y, test_X, test_y)
    model.fit()
    predictions = model.predict()
    # np.set_printoptions(threshold=np.inf)
    # print(predictions) 
    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    print("Regression Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"\nModel Parameters:")
    print(f"Coefficients: {model.get_params()[0]}")
    print(f"Intercept: {model.get_params()[1]}")

if __name__ == "__main__":
    args = parse_args()
    linear(args.name)

