import numpy as np
import pandas as pd
from typing import Tuple

class StandardScaler:
    """
    IMPLEMENTS STANDARD SCALING (Z-SCORE NORMALIZATION) FROM SCRATCH.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8 # AVOID DIVISION BY ZERO
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 36) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    SPLITS DATA ARRAYS INTO RANDOM TRAIN AND TEST SUBSETS.
    """
    np.random.seed(seed)
    indices = np.random.permutation(X.shape[0])
    test_samples = int(X.shape[0] * test_size)
    
    test_idx, train_idx = indices[:test_samples], indices[test_samples:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]