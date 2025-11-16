"""MLflow experiment tracking for model training.

This script demonstrates how to track model experiments with MLflow.
Logs parameters, metrics, and artifacts for every training run.

Setup MLflow:
pip install mlflow

Usage:
1. Start MLflow UI: mlflow ui
2. Open browser: http://localhost:5000
3. Run training with tracking enabled

To integrate with training, add this to train_st53.py
"""

import mlflow
import joblib
import os
from datetime import datetime


class ST53ExperimentTracker:
    """Track ST53 model training experiments with MLflow"""
    
    def __init__(self, experiment_name: str = "ST53_SAGD_Forecasting"):
        """Initialize MLflow experiment
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        mlflow.set_experiment(experiment_name)
        self.run = None
    
    def start_run(self, run_name: str | None = None):
        """Start a new MLflow run
        
        Args:
            run_name: Optional name for this run
        """
        if run_name is None:
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run = mlflow.start_run(run_name=run_name)
        print(f"🔬 Started MLflow experiment: {run_name}")
        return self.run
    
    def log_hyperparameters(self, params: dict):
        """Log model hyperparameters
        
        Args:
            params: Dictionary of hyperparameters
        """
        mlflow.log_params(params)
        print(f"📝 Logged {len(params)} hyperparameters")
    
    def log_training_metrics(self, metrics: dict):
        """Log training metrics
        
        Args:
            metrics: Dictionary of metrics (loss, accuracy, etc.)
        """
        mlflow.log_metrics(metrics)
        print(f"📊 Logged {len(metrics)} metrics")
    
    def log_epoch_metrics(self, epoch: int, metrics: dict):
        """Log metrics for a specific epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics for this epoch
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=epoch)
    
    def log_model(self, model, model_path: str):
        """Log Keras model as artifact
        
        Args:
            model: Keras model
            model_path: Path where model is saved
        """
        # Log the saved model file as an artifact instead of using mlflow.keras
        mlflow.log_artifact(model_path, "model")
        print(f"💾 Logged model to MLflow")
    
    def log_scaler(self, scaler_path: str):
        """Log scaler artifact
        
        Args:
            scaler_path: Path to scaler pickle file
        """
        mlflow.log_artifact(scaler_path, "preprocessing")
        print(f"📦 Logged scaler to MLflow")
    
    def log_data_info(self, data_info: dict):
        """Log dataset information
        
        Args:
            data_info: Dictionary with data statistics
        """
        for key, value in data_info.items():
            mlflow.log_param(f"data_{key}", value)
    
    def log_validation_results(self, validation_results: dict):
        """Log model validation results
        
        Args:
            validation_results: Dictionary of validation metrics
        """
        for key, value in validation_results.items():
            mlflow.log_metric(f"validation_{key}", value)
    
    def log_tag(self, key: str, value: str):
        """Add a tag to the run
        
        Args:
            key: Tag key
            value: Tag value
        """
        mlflow.set_tag(key, value)
    
    def end_run(self):
        """End the current MLflow run"""
        if self.run:
            mlflow.end_run()
            print("✅ MLflow experiment completed")


def example_integration_with_training():
    """
    Example showing how to integrate MLflow tracking into train_st53.py
    
    Add this code to your training script:
    """
    
    code_example = '''
# At the top of train_st53.py
from track_experiments import ST53ExperimentTracker

# In the train() method, after loading data:
tracker = ST53ExperimentTracker()
tracker.start_run(run_name=f"st53_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# Log hyperparameters
tracker.log_hyperparameters({
    "window_size": self.window_size,
    "epochs": self.epochs,
    "batch_size": self.batch_size,
    "lstm_units_1": 64,
    "lstm_units_2": 32,
    "activation": "sigmoid",
    "optimizer": "adam",
    "loss": "mse"
})

# Log data information
tracker.log_data_info({
    "total_samples": len(values),
    "training_samples": len(X),
    "data_min": scaler.data_min_[0],
    "data_max": scaler.data_max_[0]
})

# During training, log epoch metrics
for epoch in range(self.epochs):
    # ... training code ...
    tracker.log_epoch_metrics(epoch, {
        "loss": history.history['loss'][epoch]
    })

# After training
tracker.log_training_metrics({
    "final_loss": history.history['loss'][-1],
    "training_time_seconds": training_time
})

# Log model and artifacts
tracker.log_model(model, model_file)
tracker.log_scaler(scaler_file)

# Add tags for easy filtering
tracker.log_tag("model_type", "LSTM")
tracker.log_tag("data_version", "ST53_2024-12")
tracker.log_tag("status", "production" if validation_passed else "experimental")

# End tracking
tracker.end_run()
'''
    
    print("="*70)
    print("MLFLOW INTEGRATION EXAMPLE")
    print("="*70)
    print(code_example)
    print("="*70)
    print("\nTo view experiments:")
    print("1. Run: mlflow ui")
    print("2. Open: http://localhost:5000")
    print("3. Compare runs, parameters, and metrics visually")


if __name__ == "__main__":
    example_integration_with_training()
