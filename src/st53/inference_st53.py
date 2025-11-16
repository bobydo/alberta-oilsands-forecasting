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
            scaler_file = f"{model_path}/st53_scaler.pkl"
            meta_file = f"{model_path}/st53_meta.pkl"
            
            if not os.path.exists(model_file):
                error_msg = f"Model file not found: {model_file}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            if not os.path.exists(scaler_file):
                error_msg = f"Scaler file not found: {scaler_file}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            if not os.path.exists(meta_file):
                error_msg = f"Metadata file not found: {meta_file}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            self.model: Any = keras.models.load_model(model_file)
            self.scaler = joblib.load(scaler_file)  # Load scaler for data normalization
            meta = joblib.load(meta_file)
            
            if "window" not in meta:
                error_msg = "Metadata missing 'window' key"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.window_size = meta["window"]
            self.logger.info(f"Model loaded successfully with window_size={self.window_size}")
            self.logger.info(f"Scaler loaded successfully")
            
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
            
            # CRITICAL FIX: Scale input data before prediction
            # API Problem: User sent [8232, 8086, 7690, 8227, 8423, 8405, 8477, 7892] (Cenovus Sunrise data)
            # Without scaling: Model predicted 73.39 m³ (completely wrong!)
            # Root cause: Model was trained on scaled data (0-1), but API sent raw data (8000+)
            # Solution: Scale input → predict → inverse-scale output
            
            # Step 1: Scale input values from raw (e.g., 8000 m³) to 0-1 range
            # Why Scale to [0, 1] for Neural Networks? => Gradient descent steps: fewer for larger steps, training faster
            values_array = np.array(values).reshape(-1, 1)
            values_scaled = self.scaler.transform(values_array).flatten()
            self.logger.info(f"Scaled input from range [{min(values):.2f}, {max(values):.2f}] to [0, 1]")
            
            # Step 2: Reshape for LSTM input: (1 batch, 8 timesteps, 1 feature)
            arr = values_scaled.reshape(1, self.window_size, 1)
            
            # Step 3: Get prediction (will be in scaled 0-1 range)
            prediction_scaled = self.model.predict(arr, verbose=0)
            
            # Step 4: Inverse-scale prediction back to original range (e.g., 8000 m³)
            prediction_original = self.scaler.inverse_transform(prediction_scaled)[0][0]
            result = float(prediction_original)
            
            self.logger.info(f"Prediction successful: {result:.2f} m³")
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
