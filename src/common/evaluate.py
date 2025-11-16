import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

class ModelEvaluator:
    """Evaluates and visualizes forecasting model performance."""
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Forecast Evaluation") -> Tuple[float, float]:
        """Calculate metrics and plot predictions vs actuals. Args: y_true: Actual values, y_pred: Predicted values, title: Plot title. Returns: Tuple of (MAE, RMSE)."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
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
    
    @staticmethod
    def tune_hyperparameters(values: np.ndarray, window_sizes: List[int], epochs_list: List[int], batch_sizes: List[int], model_builder, test_split: float = 0.2) -> Dict[tuple, Tuple[float, float]]:
        """Compare different hyperparameter combinations and return metrics. Args: values: Time series data, window_sizes: List of window sizes to test, epochs_list: List of epoch values to test, batch_sizes: List of batch sizes to test, model_builder: Function to build model (e.g., ST53Model.build), test_split: Fraction for testing. Returns: Dict mapping (window_size, epochs, batch_size) to (MAE, RMSE)."""
        from src.common.window import WindowGenerator
        results = {}
        split_idx = int(len(values) * (1 - test_split))
        
        print("Tuning hyperparameters...\n")
        for ws in window_sizes:
            for ep in epochs_list:
                for bs in batch_sizes:
                    X, y = WindowGenerator.create(values, ws)
                    X = X.reshape((-1, ws, 1))
                    X_train, y_train = X[:split_idx-ws], y[:split_idx-ws]
                    X_test, y_test = X[split_idx-ws:], y[split_idx-ws:]
                    
                    model = model_builder(ws)
                    model.fit(X_train, y_train, epochs=ep, batch_size=bs, verbose=0)
                    y_pred = model.predict(X_test).flatten()
                    
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    results[(ws, ep, bs)] = (mae, rmse)
                    print(f"window={ws}, epochs={ep}, batch={bs}: MAE={mae:.2f}, RMSE={rmse:.2f}")
        
        best_config = min(results.keys(), key=lambda k: results[k][0])
        print(f"\n✓ Best: window={best_config[0]}, epochs={best_config[1]}, batch={best_config[2]} (MAE={results[best_config][0]:.2f}, RMSE={results[best_config][1]:.2f})")
        return results
