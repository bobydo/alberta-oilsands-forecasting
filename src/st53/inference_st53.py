import numpy as np
import joblib
import keras
from typing import List

class ST53Predictor:
    """Loads trained ST53 model and makes predictions."""
    
    def __init__(self, model_path: str):
        """Load model and metadata from disk. Args: model_path: Directory containing model files."""
        self.model = keras.models.load_model(f"{model_path}/st53_model.keras")
        self.window_size = joblib.load(f"{model_path}/st53_meta.pkl")["window"]
    
    def predict(self, values: List[float]) -> float:
        """Predict next value given recent history. Args: values: List of recent production values (length must equal window_size). Returns: Predicted next production value."""
        arr = np.array(values).reshape(1, self.window_size, 1)
        return float(self.model.predict(arr)[0][0])
