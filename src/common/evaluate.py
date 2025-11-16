import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from typing import Tuple

class ModelEvaluator:
    """Evaluates and visualizes forecasting model performance."""
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Forecast Evaluation") -> Tuple[float, float]:
        """Calculate metrics and plot predictions vs actuals. Args: y_true: Actual values, y_pred: Predicted values, title: Plot title. Returns: Tuple of (MAE, RMSE)."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label="Actual", marker='o')
        plt.plot(y_pred, label="Predicted", marker='x')
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("Production")
        plt.legend()
        plt.grid()
        plt.show()
        
        return mae, rmse
