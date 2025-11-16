"""
Train ST53 model on LARGE SAGD operations only (> 15,000 m³/month average)

Problem: Original model trained on ALL operators (mean: 7,538 m³, median: 2,532 m³)
         When predicting for Cenovus (30,000-40,000 m³), model regresses to training mean

Solution: Filter training data to only large operations similar to Cenovus scale
"""
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from src.st53.preprocess_st53 import ST53DataProcessor
from src.st53.model_st53 import ST53Model
from src.common.window import WindowGenerator
from src.common.logger import FileLogger
import os

class ST53LargeOpsTrainer:
    """Trains LSTM model on large SAGD operations only."""
    
    def __init__(self, window_size: int = 8, epochs: int = 40, batch_size: int = 8, min_production: float = 15000):
        """
        Args:
            window_size: Number of time steps for input sequence
            epochs: Training iterations
            batch_size: Samples per gradient update
            min_production: Minimum average production to include operator (m³/month)
        """
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.min_production = min_production
        self.logger = FileLogger.setup("train_st53_large_ops")
    
    def train(self, xls_path: str, output_dir: str):
        """Train model on large operators only and save artifacts."""
        try:
            self.logger.info(f"Starting ST53 LARGE OPS model training")
            self.logger.info(f"Hyperparameters: window_size={self.window_size}, epochs={self.epochs}, batch_size={self.batch_size}")
            self.logger.info(f"Filtering for operators with average production >= {self.min_production} m³/month")
            
            # Load and preprocess data
            df = ST53DataProcessor.load(xls_path)
            self.logger.info(f"Loaded {len(df)} total records")
            
            # CRITICAL FIX: Filter for large operations only
            # Group by operator+scheme and calculate average production
            df['Bitumen'] = df['Bitumen'].astype(float)
            operator_avg = df.groupby(['Operator', 'Scheme Name'])['Bitumen'].mean()
            large_operators = operator_avg[operator_avg >= self.min_production].index
            
            self.logger.info(f"Found {len(large_operators)} large operations:")
            for (operator, scheme), avg in operator_avg[operator_avg >= self.min_production].items():
                self.logger.info(f"  {operator} - {scheme}: {avg:.2f} m³/month avg")
            
            # Filter dataset to only large operators
            df_filtered = df[df.apply(lambda row: (row['Operator'], row['Scheme Name']) in large_operators, axis=1)]
            values = df_filtered["Bitumen"].values
            self.logger.info(f"Filtered to {len(values)} data points from large operations")
            self.logger.info(f"New data range: [{values.min():.2f}, {values.max():.2f}] m³")
            self.logger.info(f"New mean: {values.mean():.2f} m³, median: {np.median(values):.2f} m³")
            
            if len(values) < self.window_size + 1:
                error_msg = f"Insufficient data: need at least {self.window_size + 1} samples, got {len(values)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Scale data to 0-1 range
            scaler = MinMaxScaler(feature_range=(0, 1))
            values_scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
            self.logger.info(f"Scaled data to [0, 1] range")
            
            # Create windowed sequences
            X, y = WindowGenerator.create(values_scaled, self.window_size)
            X = X.reshape((-1, self.window_size, 1))
            self.logger.info(f"Created {X.shape[0]} training samples")
            
            if X.shape[0] == 0:
                error_msg = "No training samples generated from windowed data"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Build and train model
            model = ST53Model.build(self.window_size)
            self.logger.info("Starting model training...")
            model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
            self.logger.info("Model training completed")
            
            # Save model, scaler, and metadata
            os.makedirs(output_dir, exist_ok=True)
            model_file = f"{output_dir}/st53_model.keras"
            scaler_file = f"{output_dir}/st53_scaler.pkl"
            meta_file = f"{output_dir}/st53_meta.pkl"
            
            model.save(model_file)
            joblib.dump(scaler, scaler_file)
            joblib.dump({"window": self.window_size, "min_production": self.min_production}, meta_file)
            
            self.logger.info(f"✅ Model saved to {model_file}")
            self.logger.info(f"✅ Scaler saved to {scaler_file}")
            self.logger.info(f"✅ Metadata saved to {meta_file}")
            
            print("\n" + "="*70)
            print("✅ LARGE OPS MODEL TRAINING COMPLETE")
            print("="*70)
            print(f"Training data: {len(values)} points from {len(large_operators)} large operations")
            print(f"Data range: {values.min():.0f} - {values.max():.0f} m³/month")
            print(f"Mean production: {values.mean():.0f} m³/month")
            print(f"Model saved to: {model_file}")
            
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
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python train_st53_large_ops.py <input_xls> <output_dir>")
        print("Example: python train_st53_large_ops.py data/st53/ST53_2024-12.xls models/")
        sys.exit(1)
    
    try:
        trainer = ST53LargeOpsTrainer(min_production=15000)  # Only ops producing > 15,000 m³/month
        trainer.train(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
