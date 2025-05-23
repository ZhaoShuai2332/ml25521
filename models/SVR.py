import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class SVRModel:
    """
    SVRModel encapsulates a linear-kernel SVR with feature standardization.
    Parameters are cached to avoid retraining, including scaler parameters.
    """
    def __init__(self,
                 name: str,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray):
        # Base directory for parameter storage
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.params_path = os.path.join(base_dir, "model_params", f"{name}_svr_params.npz")
        self.scaler_path = os.path.join(base_dir, "model_params", f"{name}_scaler_params.npz")

        # Data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Initialize placeholders for model parameters
        self.coef_ = None
        self.intercept_ = None
        self.support_vectors_ = None
        self.support_ = None
        self.n_support_ = None

        # Components
        self.scaler = StandardScaler()
        self.model = SVR(
            kernel='linear',
            C=1.0,
            epsilon=0.1,
            tol=1e-3,
            max_iter=1000,
            verbose=True,
            shrinking=True,
            cache_size=200
        )

        # Load existing parameters if available
        self.load_params()

    def load_params(self) -> bool:
        """
        Load cached SVR and scaler parameters from disk.
        Returns True if both scaler and SVR params were loaded, False otherwise.
        """
        if os.path.exists(self.params_path) and os.path.exists(self.scaler_path):
            print(f"Loading parameters from {self.params_path}")
            svr_params = np.load(self.params_path)
            # SVR params
            self.coef_ = svr_params["coef"]
            self.intercept_ = svr_params["intercept"]
            self.support_vectors_ = svr_params["support_vectors"]
            self.support_ = svr_params["support"]
            self.n_support_ = svr_params["n_support"]

        if os.path.exists(self.scaler_path):
            print(f"Loading scaler parameters from {self.scaler_path}")
            scaler_params = np.load(self.scaler_path)
            # Scaler params
            self.scaler.mean_ = scaler_params["mean"]
            self.scaler.scale_ = scaler_params["scale"]
            self.scaler.var_ = scaler_params.get("var", None)
            self.scaler.n_samples_seen_ = scaler_params.get("n_samples", None)

    def fit(self):
        """
        Train the SVR and scaler if parameters are not cached. Otherwise, do nothing.
        """
        if self.coef_ is None:
            # Fit scaler then transform training data
            self.scaler.fit(self.X_train)
            X_scaled = self.scaler.transform(self.X_train)

            # Fit SVR on scaled data
            self.model.fit(X_scaled, self.y_train)

            # Cache SVR params
            self.coef_ = self.model.coef_
            self.intercept_ = self.model.intercept_
            self.support_vectors_ = self.model.support_vectors_
            self.support_ = self.model.support_
            self.n_support_ = self.model.n_support_
            np.savez(
                self.params_path,
                coef=self.coef_,
                intercept=self.intercept_,
                support_vectors=self.support_vectors_,
                support=self.support_,
                n_support=self.n_support_
            )

            # Cache scaler params
            np.savez(
                self.scaler_path,
                mean=self.scaler.mean_,
                scale=self.scaler.scale_,
                var=getattr(self.scaler, 'var_', None),
                n_samples=self.scaler.n_samples_seen_
            )
        else:
            print(f"Parameters already cached. Loading parameters from {self.params_path}")

    def predict(self) -> np.ndarray:
        """
        Predict on the test set using loaded or trained parameters.
        """
        if self.coef_ is None:
            raise ValueError("Model parameters not loaded or trained. Call fit() first.")

        X_scaled = self.scaler.transform(self.X_test)
        return np.dot(X_scaled, self.coef_.T) + self.intercept_

    def get_params(self) -> dict:
        """
        Return the cached SVR and scaler parameters.
        """
        return {
            'svr': {
                'coef': self.coef_,
                'intercept': self.intercept_,
                'support_vectors': self.support_vectors_,
                'support': self.support_,
                'n_support': self.n_support_
            },
            'scaler': {
                'mean': getattr(self.scaler, 'mean_', None),
                'scale': getattr(self.scaler, 'scale_', None),
                'var': getattr(self.scaler, 'var_', None),
                'n_samples': getattr(self.scaler, 'n_samples_seen_', None)
            }
        }
