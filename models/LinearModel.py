import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

class Linear:
    def __init__(self, name: str,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray,  y_test: np.ndarray,
                 load_integer: bool = False):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.params_path = os.path.join(base_dir, "model_params", f"{name}_linear_params.npz")
        self.scaler_path = os.path.join(base_dir, "model_params", f"{name}_min_max_scaler_params.npz")
        # 新增 y_scaler 的保存路径
        self.y_scaler_path = os.path.join(base_dir, "model_params", f"{name}_y_min_max_scaler_params.npz")

        # X 的归一化范围
        # min_max_range = {
        #     "mnist":(-0.001, 0.001),
        #     "uci":(-0.001, 0.001),
        #     "credit":(-0.00001,0.00001)
        # }

        # 原始数据
        self.X_train, raw_y_train = X_train, y_train
        self.X_test,  raw_y_test  = X_test,  y_test

        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_train = self.y_scaler.fit_transform(raw_y_train.reshape(-1, 1)).ravel()
        self.y_test  = self.y_scaler.transform( raw_y_test.reshape(-1, 1)).ravel()

        self.pipeline = Pipeline([
            # ("scaler", MinMaxScaler(feature_range=min_max_range[name])),  # 处理 X
            ("scaler", MinMaxScaler()),
            ("regressor", LinearRegression())
        ])

        self.loaded = False
        if load_integer and os.path.exists(self.params_path) and os.path.exists(self.scaler_path):
            print(f"Loading model params from {self.params_path}")
            params = np.load(self.params_path)
            reg = self.pipeline.named_steps["regressor"]
            reg.coef_ = params["coef"]
            reg.intercept_ = params["intercept"]

            print(f"Loading scaler params from {self.scaler_path}")
            scaler_params = np.load(self.scaler_path)
            scaler = self.pipeline.named_steps["scaler"]
            scaler.min_ = scaler_params["min"]
            scaler.scale_ = scaler_params["scale"]
            scaler.data_min_ = scaler_params["data_min"]
            scaler.data_max_ = scaler_params["data_max"]
            scaler.data_range_ = scaler.data_max_ - scaler.data_min_
            scaler.n_samples_seen_ = self.X_train.shape[0]

            # 同时加载 y_scaler
            print(f"Loading y_scaler params from {self.y_scaler_path}")
            y_params = np.load(self.y_scaler_path)
            self.y_scaler.min_ = y_params["min"]
            self.y_scaler.scale_ = y_params["scale"]
            self.y_scaler.data_min_ = y_params["data_min"]
            self.y_scaler.data_max_ = y_params["data_max"]
            self.y_scaler.data_range_ = self.y_scaler.data_max_ - self.y_scaler.data_min_
            self.y_scaler.n_samples_seen_ = self.y_train.shape[0]

            self.loaded = True
        else:
            print(f"No pre-saved params found for X or y; will fit from scratch.")

    def fit(self):
        if self.loaded:
            print("Parameters already loaded; skipping fit.")
            return

        self.pipeline.fit(self.X_train, self.y_train)

        reg = self.pipeline.named_steps["regressor"]
        np.savez(self.params_path,
                 coef=reg.coef_,
                 intercept=reg.intercept_)

        scaler = self.pipeline.named_steps["scaler"]
        np.savez(self.scaler_path,
                 min=scaler.min_,
                 scale=scaler.scale_,
                 data_min=scaler.data_min_,
                 data_max=scaler.data_max_)

        np.savez(self.y_scaler_path,
                 min=self.y_scaler.min_,
                 scale=self.y_scaler.scale_,
                 data_min=self.y_scaler.data_min_,
                 data_max=self.y_scaler.data_max_)

    def predict(self) -> np.ndarray:
        return self.pipeline.predict(self.X_test)

    def predict_original_scale(self) -> np.ndarray:
        y_pred_scaled = self.pipeline.predict(self.X_test).reshape(-1,1)
        return self.y_scaler.inverse_transform(y_pred_scaled).ravel()

    def get_params(self):
        reg = self.pipeline.named_steps["regressor"]
        return reg.coef_, reg.intercept_
