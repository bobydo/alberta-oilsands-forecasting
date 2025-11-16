import sys
import numpy as np
import joblib
from src.st53.preprocess_st53 import ST53DataProcessor
from src.st53.model_st53 import ST53Model
from src.common.window import create_windows

class ST53Trainer:
    """Trains LSTM model on ST53 bitumen production data."""
    
    def __init__(self, window_size: int = 6, epochs: int = 40, batch_size: int = 16):
        """Initialize trainer with hyperparameters. Args: window_size: Number of time steps for input sequence, epochs: Training iterations, batch_size: Samples per gradient update."""
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
    
    def train(self, xls_path: str, output_dir: str):
        """Train model and save artifacts. Args: xls_path: Path to ST53 Excel file, output_dir: Directory to save model and metadata."""
        # Load and preprocess data
        df = ST53DataProcessor.load(xls_path)
        values = np.array(df["Bitumen"].astype(float).values)
        
        # Create windowed sequences
        X, y = create_windows(values, self.window_size)
        X = X.reshape((-1, self.window_size, 1))
        
        # Build and train model
        model = ST53Model.build(self.window_size)
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)
        
        # Save model and metadata
        model.save(f"{output_dir}/st53_model.keras")
        joblib.dump({"window": self.window_size}, f"{output_dir}/st53_meta.pkl")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_st53.py <input_xls> <output_dir>")
        sys.exit(1)
    trainer = ST53Trainer()
    trainer.train(sys.argv[1], sys.argv[2])
