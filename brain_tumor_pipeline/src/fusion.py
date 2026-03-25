from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureFusionScaler:
    def __init__(self, use_standard_scaler: bool = True):
        self.use_standard_scaler = use_standard_scaler
        self.scaler: Optional[StandardScaler] = None

    def fit(self, X_train_parts: Sequence[np.ndarray]) -> None:
        X = np.concatenate(X_train_parts, axis=1)
        if self.use_standard_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(X)

    def transform(self, X_parts: Sequence[np.ndarray]) -> np.ndarray:
        X = np.concatenate(X_parts, axis=1)
        if self.use_standard_scaler and self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def fit_transform(self, X_train_parts: Sequence[np.ndarray]) -> np.ndarray:
        self.fit(X_train_parts)
        return self.transform(X_train_parts)

