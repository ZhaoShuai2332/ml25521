import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# class Linear:
#     def __init__(self, name: str,
#                  X_train: np.ndarray, y_train: np.ndarray,
#                  X_test: np.ndarray,  y_test: np.ndarray):
#         base_dir = os.path.dirname(os.path.dirname(__file__))
#         self.params_path = os.path.join(base_dir, "model_params", f"{name}_linear_params.npz")
#         self.scaler_path = os.path.join(base_dir, "model_params", f"{name}_standard_scaler_params.npz")

#         self.X_train, self.y_train = X_train, y_train
#         self.X_test,  self.y_test  = X_test,  y_test

#         self.pipeline = Pipeline([
#             ("scaler", StandardScaler()),
#             ("regressor", LinearRegression())
#         ])

#         self.loaded = False
#         if os.path.exists(self.params_path) and os.path.exists(self.scaler_path):
#             print(f"Loading model params from {self.params_path}")
#             params = np.load(self.params_path)
#             reg = self.pipeline.named_steps["regressor"]
#             reg.coef_ = params["coef"]
#             reg.intercept_ = params["intercept"]

#             print(f"Loading scaler params from {self.scaler_path}")
#             scaler_params = np.load(self.scaler_path)
#             scaler = self.pipeline.named_steps["scaler"]
#             scaler.mean_ = scaler_params["mean"]
#             scaler.scale_ = scaler_params["scale"]
#             scaler.var_ = scaler.scale_ ** 2
#             scaler.n_samples_seen_ = X_train.shape[0]

#             self.loaded = True

#     def fit(self):
#         if self.loaded:
#             print("Parameters already loaded; skipping fit.")
#             return

#         self.pipeline.fit(self.X_train, self.y_train)

#         reg = self.pipeline.named_steps["regressor"]
#         np.savez(self.params_path,
#                  coef=reg.coef_,
#                  intercept=reg.intercept_)

#         scaler = self.pipeline.named_steps["scaler"]
#         np.savez(self.scaler_path,
#                  mean = scaler.mean_,
#                  scale= scaler.scale_)

#     def predict(self) -> np.ndarray:
#         return self.pipeline.predict(self.X_test)

#     def get_params(self):
#         reg = self.pipeline.named_steps["regressor"]
#         return reg.coef_, reg.intercept_

class Linear:
    def __init__(self, name: str,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray,  y_test: np.ndarray):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.params_path = os.path.join(base_dir, "model_params", f"{name}_linear_params.npz")
        self.scaler_path = os.path.join(base_dir, "model_params", f"{name}_min_max_scaler_params.npz")

        self.X_train, self.y_train = X_train, y_train
        self.X_test,  self.y_test  = X_test,  y_test

        self.pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("regressor", LinearRegression())
        ])

        self.loaded = False
        if os.path.exists(self.params_path) and os.path.exists(self.scaler_path):
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
            scaler.n_samples_seen_ = X_train.shape[0]

            self.loaded = True

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

    def predict(self) -> np.ndarray:
        return self.pipeline.predict(self.X_test)

    def get_params(self):
        reg = self.pipeline.named_steps["regressor"]
        return reg.coef_, reg.intercept_

# class Linear:
#     def __init__(self, name:str, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray):
#         self.params_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_params", f"{name}_linear_params.npz")
#         self.scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_params", f"{name}_scaler_params.npz")

#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_test = X_test
#         self.y_test = y_test
#         if os.path.exists(self.params_path):
#             params = np.load(self.params_path)
#             self.coef_ = params["coef"]
#             self.intercept_ = params["intercept"]
#         else:
#             self.coef_ = None
#             self.intercept_ = None
#         self.model = LinearRegression()
        
#     def fit(self):
#         if self.coef_ is None or self.intercept_ is None:
#             self.model.fit(self.X_train, self.y_train)
#             # self.coef_ = np.round(self.model.coef_).astype(int)
#             # self.intercept_ = np.round(self.model.intercept_).astype(int)
#             self.coef_ = self.model.coef_
#             self.intercept_ = self.model.intercept_
#             np.savez(self.params_path, coef=self.coef_, intercept=self.intercept_)
#         else:
#             self.model.coef_ = self.coef_
#             self.model.intercept_ = self.intercept_

#     def predict(self) -> np.ndarray:
#         return self.model.predict(self.X_test)

#     def get_params(self):
#         return self.coef_, self.intercept_
