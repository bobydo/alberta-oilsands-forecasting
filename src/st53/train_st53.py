import sys
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from src.st53.preprocess_st53 import ST53DataProcessor
from src.st53.model_st53 import ST53Model
from src.common.window import WindowGenerator
from src.common.logger import FileLogger
import os

class ST53Trainer:
    """Trains LSTM model on ST53 bitumen production data."""
    # From python -m tuning.tune_st53
    # ✓ Best: window=8, epochs=40, batch=8 (MAE=7757.21, RMSE=13733.49)
    def __init__(self, window_size: int = 8, epochs: int = 40, batch_size: int = 8):
        """Initialize trainer with hyperparameters. Args: window_size: Number of time steps for input sequence, epochs: Training iterations, batch_size: Samples per gradient update."""
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.logger = FileLogger.setup("train_st53")
    
    def train(self, xls_path: str, output_dir: str):
        """Train model and save artifacts. Args: xls_path: Path to ST53 Excel file, output_dir: Directory to save model and metadata."""
        try:
            self.logger.info(f"Starting ST53 model training")
            self.logger.info(f"Hyperparameters: window_size={self.window_size}, epochs={self.epochs}, batch_size={self.batch_size}")
            
            # Load and preprocess data
            df = ST53DataProcessor.load(xls_path)
            values = np.array(df["Bitumen"].astype(float).values)
            self.logger.info(f"Loaded {len(values)} data points")
            
            if len(values) < self.window_size + 1:
                error_msg = f"Insufficient data: need at least {self.window_size + 1} samples, got {len(values)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # CRITICAL: Scale data to 0-1 range for neural network training
            # Problem: Without scaling, model was trained on raw values (8,000-40,000 m³)
            # This caused API predictions to be completely wrong (73 m³ instead of ~8,000 m³)
            # Neural networks learn best when input data is normalized to similar ranges
            scaler = MinMaxScaler(feature_range=(0, 1))
            values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
            self.logger.info(f"Scaled data from range [{values.min():.2f}, {values.max():.2f}] to [0, 1]")
            
            # Create windowed sequences from SCALED data
            X, y = WindowGenerator.create(values_scaled, self.window_size)
            # reshapes the data into the 3D format required by LSTM neural networks. (445, 8, 1)
            # -1: "Auto-calculate this dimension" - Python figures out it should be 445 based on the total array size
            # self.window_size (8): Number of time steps in each sequence (8 months of historical data)
            # 1: Number of features per timestep (just 1 feature: bitumen production)            
            X = X.reshape((-1, self.window_size, 1))
            self.logger.info(f"Created {X.shape[0]} training samples")
            
            if X.shape[0] == 0:
                error_msg = "No training samples generated from windowed data"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Build and train model
            model = ST53Model.build(self.window_size)
            self.logger.info("Starting model training...")
            #Complete Training Flow training = 40 epochs × 56 batches/epoch = 2,240 weight updates
            model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)
            self.logger.info("Model training completed")
            
            # Save model, scaler, and metadata
            os.makedirs(output_dir, exist_ok=True)
            model_file = f"{output_dir}/st53_model.keras"
            scaler_file = f"{output_dir}/st53_scaler.pkl"
            meta_file = f"{output_dir}/st53_meta.pkl"
            
            model.save(model_file)
            joblib.dump(scaler, scaler_file)  # Save scaler for inference
            joblib.dump({"window": self.window_size}, meta_file)
            self.logger.info(f"Model saved to {model_file}")
            self.logger.info(f"Scaler saved to {scaler_file}")
            self.logger.info(f"Metadata saved to {meta_file}")
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found error: {e}")
            print(f"Error: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Validation error: {e}")
            print(f"Error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            print(f"Training failed: {e}")
            raise


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_st53.py <input_xls> <output_dir>")
        sys.exit(1)
    
    try:
        trainer = ST53Trainer()
        trainer.train(sys.argv[1], sys.argv[2])
        print(f"\n✓ Model trained successfully and saved to {sys.argv[2]}")
        trainer.logger.info("Training completed successfully")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        sys.exit(1)
