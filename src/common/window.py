import numpy as np
from typing import Tuple

class WindowGenerator:
    """Generates sliding window sequences from time series data."""
    
    @staticmethod
    def create(values: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output pairs from time series. Args: values: 1D array of time series values, window_size: Number of time steps in each input sequence. Returns: Tuple of (X, y) where X is input sequences and y is target values."""
        X, y = [], []
        for i in range(len(values) - window_size):
            X.append(values[i:i + window_size])
            y.append(values[i + window_size])
        return np.array(X), np.array(y)

# Keep this for compatibility with training scripts
create_windows = WindowGenerator.create
