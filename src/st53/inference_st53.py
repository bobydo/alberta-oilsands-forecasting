import numpy as np
import joblib
import keras
import os
from typing import List, Any
from src.common.logger import FileLogger

class ST53Predictor:
    """Loads trained ST53 model and makes predictions."""
    
    def __init__(self, model_path: str):
        """Load model and metadata from disk. Args: model_path: Directory containing model files."""
        self.logger = FileLogger.setup("inference_st53")
        
        try:
            self.logger.info(f"Initializing ST53Predictor from: {model_path}")
            
            if not os.path.exists(model_path):
                error_msg = f"Model directory not found: {model_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            model_file = f"{model_path}/st53_model.keras"
            meta_file = f"{model_path}/st53_meta.pkl"
            
            if not os.path.exists(model_file):
                error_msg = f"Model file not found: {model_file}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            if not os.path.exists(meta_file):
                error_msg = f"Metadata file not found: {meta_file}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            self.model: Any = keras.models.load_model(model_file)
            meta = joblib.load(meta_file)
            
            if "window" not in meta:
                error_msg = "Metadata missing 'window' key"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.window_size = meta["window"]
            self.logger.info(f"Model loaded successfully with window_size={self.window_size}")
            
        except FileNotFoundError as e:
            self.logger.error(f"Model loading error: {e}")
            raise FileNotFoundError(f"Model loading error: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize ST53Predictor: {e}", exc_info=True)
            raise Exception(f"Failed to initialize ST53Predictor: {e}")
    
    def predict(self, values: List[float]) -> float:
        """Predict next value given recent history. Args: values: List of recent production values (length must equal window_size). Returns: Predicted next production value."""
        try:
            self.logger.info(f"Making prediction with {len(values)} input values")
            
            if len(values) != self.window_size:
                error_msg = f"Expected {self.window_size} values, got {len(values)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not all(isinstance(v, (int, float)) for v in values):
                error_msg = "All values must be numeric (int or float)"
                self.logger.error(error_msg)
                raise TypeError(error_msg)
            
            arr = np.array(values).reshape(1, self.window_size, 1)
            prediction = self.model.predict(arr, verbose=0)
            result = float(prediction[0][0])
            
            self.logger.info(f"Prediction successful: {result}")
            return result
            
        except ValueError as e:
            self.logger.error(f"Prediction input error: {e}")
            raise ValueError(f"Prediction input error: {e}")
        except TypeError as e:
            self.logger.error(f"Prediction type error: {e}")
            raise TypeError(f"Prediction type error: {e}")
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            raise Exception(f"Prediction failed: {e}")
