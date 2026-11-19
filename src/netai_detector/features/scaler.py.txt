"""
Feature Scaler — save/load StandardScaler
CRITICAL: Must use SAME scaler at train AND inference time
"""
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path


class FeatureScaler:
    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        self._fitted = False

    def fit(self, X: np.ndarray):
        self.scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path):
        joblib.dump({"scaler": self.scaler, "type": self.scaler_type}, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        data = joblib.load(path)
        instance = cls(scaler_type=data["type"])
        instance.scaler = data["scaler"]
        instance._fitted = True
        return instance
