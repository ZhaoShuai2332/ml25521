import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.SVR import SVRModel
from sklearn.metrics import accuracy_score
import numpy as np
from model_test.commends import parse_args, dataset_selecter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()

def svr(name: str):
    train_X, test_X, train_y, test_y = dataset_selecter(name)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    model = SVRModel(name, train_X, train_y, test_X, test_y)
    model.fit()
    predictions = model.predict()
    predicted_labels = np.round(predictions).astype(int)
    test_labels = test_y.astype(int)
    accuracy = accuracy_score(test_labels, predicted_labels)
    mse = mean_squared_error(test_labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_labels, predictions)
    r2 = r2_score(test_labels, predictions)
    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nRegression Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

if __name__ == "__main__":
    args = parse_args()
    svr(args.name)

