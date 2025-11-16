import sys
import joblib
from src.st39.preprocess_st39 import ST39DataProcessor
from src.st39.model_st39 import ST39Model
from src.common.window import create_windows

class ST39Trainer:
    """Trains LSTM model on ST39 mining production data."""
    
    def __init__(self, window_size: int = 6, epochs: int = 40, batch_size: int = 16):
        """Initialize trainer with hyperparameters. Args: window_size: Number of time steps for input sequence, epochs: Training iterations, batch_size: Samples per gradient update."""
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
    
    def train(self, xls_path: str, output_dir: str):
        """Train model and save artifacts. Args: xls_path: Path to ST39 Excel file, output_dir: Directory to save model and metadata."""
        # Load and preprocess data
        df = ST39DataProcessor.load(xls_path)
        values = df["Production"].astype(float).to_numpy()
        
        # Create windowed sequences
        X, y = create_windows(values, self.window_size)
        X = X.reshape((-1, self.window_size, 1))
        
        # Build and train model
        model = ST39Model.build(self.window_size)
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)
        
        # Save model and metadata
        model.save(f"{output_dir}/st39_model.keras")
        joblib.dump({"window": self.window_size}, f"{output_dir}/st39_meta.pkl")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_st39.py <input_xls> <output_dir>")
        sys.exit(1)
    trainer = ST39Trainer()
    trainer.train(sys.argv[1], sys.argv[2])
