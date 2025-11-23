"""
Example Usage: Autoencoder Anomaly Detection
============================================
This script demonstrates how to use the TimeSeriesAutoencoder class
for detecting anomalies in oil sands production data.

Workflow:
1. Generate synthetic production data (normal + anomalies)
2. Build and train the autoencoder on normal data
3. Set anomaly detection threshold
4. Detect anomalies in test data
5. Visualize results

Author: Your Name
Date: 2025-11-23
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autoencoder_model import TimeSeriesAutoencoder
from data_generator import SyntheticProductionDataGenerator
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory for saving plots
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logger.info(f"Created output directory: {OUTPUT_DIR}")


def main():
    """
    Main function demonstrating complete anomaly detection workflow.
    """
    print("\n" + "="*70)
    print("AUTOENCODER ANOMALY DETECTION FOR OIL SANDS PRODUCTION")
    print("="*70 + "\n")
    
    # ========================================================================
    # STEP 1: Generate Synthetic Data
    # ========================================================================
    print("STEP 1: Generating Synthetic Production Data")
    print("-" * 70)
    
    data_generator = SyntheticProductionDataGenerator(
        num_samples=2000,
        num_features=10,
        anomaly_ratio=0.05,  # 5% anomalies
        random_seed=42
    )
    
    # Generate data
    X, y_true = data_generator.generate_data()
    feature_names = data_generator.get_feature_names()
    
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    print(f"Normal samples: {(y_true == 0).sum()}")
    print(f"Anomalous samples: {(y_true == 1).sum()}")
    print(f"Features: {', '.join(feature_names)}")
    
    # Split data into train (normal only), validation, and test sets
    train_size = 1000
    val_size = 300
    
    # Training data: only normal samples
    X_train = X[y_true == 0][:train_size]
    
    # Validation data: normal samples
    X_val = X[y_true == 0][train_size:train_size + val_size]
    
    # Test data: mix of normal and anomalous samples
    X_test = X[train_size + val_size:]
    y_test = y_true[train_size + val_size:]
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples (normal only)")
    print(f"  Validation: {len(X_val)} samples (normal only)")
    print(f"  Test: {len(X_test)} samples ({(y_test == 0).sum()} normal, {(y_test == 1).sum()} anomalies)")
    
    # ========================================================================
    # STEP 2: Build and Train Autoencoder
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Building and Training Autoencoder")
    print("-" * 70)
    
    # Initialize autoencoder
    autoencoder = TimeSeriesAutoencoder(
        input_dim=X.shape[1],
        encoding_dim=4  # Compress 10 features to 4 latent dimensions
    )
    
    # Build model architecture
    autoencoder.build_model()
    autoencoder.get_model_summary()
    
    # Train the autoencoder
    print("\nTraining autoencoder...")
    history = autoencoder.train(
        X_train=X_train,
        X_val=X_val,
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    # Plot training history
    autoencoder.plot_training_history()
    
    # ========================================================================
    # STEP 3: Set Anomaly Detection Threshold
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Setting Anomaly Detection Threshold")
    print("-" * 70)
    
    # Use 95th percentile of normal data reconstruction errors
    threshold = autoencoder.set_threshold(X_train, percentile=95)
    print(f"Threshold set at: {threshold:.6f}")
    
    # ========================================================================
    # STEP 4: Detect Anomalies in Test Data
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: Detecting Anomalies in Test Data")
    print("-" * 70)
    
    # Predict anomalies
    y_pred, reconstruction_errors = autoencoder.predict_anomalies(X_test)
    
    # Calculate performance metrics
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Predicted")
    print(f"                Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:7d}")
    print(f"       Anomaly  {cm[1,0]:6d}  {cm[1,1]:7d}")
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, reconstruction_errors)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # ========================================================================
    # STEP 5: Visualize Results
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: Visualizing Results")
    print("-" * 70)
    
    # Plot reconstruction error distribution
    autoencoder.plot_reconstruction_error_distribution(
        X_normal=X_train,
        X_test=X_test,
        anomaly_labels=y_pred
    )
    
    # Plot ROC curve
    plot_roc_curve(y_test, reconstruction_errors)
    
    # Plot reconstruction errors over time
    plot_errors_timeline(reconstruction_errors, y_test, y_pred, threshold)
    
    # Analyze specific anomalies
    analyze_detected_anomalies(X_test, y_test, y_pred, reconstruction_errors, feature_names)
    
    # ========================================================================
    # STEP 6: Explore Latent Space (Optional)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: Exploring Latent Space Representation")
    print("-" * 70)
    
    # Get latent representations
    latent_train = autoencoder.get_latent_representation(X_train[:500])
    latent_test = autoencoder.get_latent_representation(X_test)
    
    # Visualize latent space (first 2 dimensions)
    plot_latent_space(latent_train, latent_test, y_test, y_pred)
    
    # ========================================================================
    # STEP 7: Save Model (Optional)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: Saving Trained Model")
    print("-" * 70)
    
    model_path = "trained_autoencoder.keras"
    autoencoder.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n" + "="*70)
    print("ANOMALY DETECTION COMPLETE!")
    print("="*70 + "\n")


def plot_roc_curve(y_true, scores):
    """Plot ROC curve for anomaly detection."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Anomaly Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"ROC curve saved to {os.path.join(OUTPUT_DIR, 'roc_curve.png')}")


def plot_errors_timeline(errors, y_true, y_pred, threshold):
    """Plot reconstruction errors over time."""
    plt.figure(figsize=(14, 6))
    
    # Plot all errors
    plt.plot(errors, label='Reconstruction Error', color='blue', alpha=0.6, linewidth=1)
    
    # Highlight true anomalies
    anomaly_indices = np.where(y_true == 1)[0]
    plt.scatter(anomaly_indices, errors[anomaly_indices], 
               color='red', s=100, marker='x', label='True Anomalies', zorder=5)
    
    # Highlight false positives
    false_positive_indices = np.where((y_pred == 1) & (y_true == 0))[0]
    if len(false_positive_indices) > 0:
        plt.scatter(false_positive_indices, errors[false_positive_indices],
                   color='orange', s=100, marker='o', alpha=0.6, 
                   label='False Positives', zorder=4)
    
    # Plot threshold
    plt.axhline(threshold, color='green', linestyle='--', linewidth=2, 
               label=f'Threshold = {threshold:.4f}')
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Reconstruction Error', fontsize=12)
    plt.title('Reconstruction Errors Over Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'errors_timeline.png'), dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Errors timeline saved to {os.path.join(OUTPUT_DIR, 'errors_timeline.png')}")


def plot_latent_space(latent_train, latent_test, y_test, y_pred):
    """Visualize the latent space representation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Training data in latent space
    axes[0].scatter(latent_train[:, 0], latent_train[:, 1], 
                   c='green', alpha=0.5, s=30, label='Normal Training Data')
    axes[0].set_xlabel('Latent Dimension 1', fontsize=12)
    axes[0].set_ylabel('Latent Dimension 2', fontsize=12)
    axes[0].set_title('Latent Space - Training Data', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Test data in latent space (colored by true labels)
    normal_mask = y_test == 0
    anomaly_mask = y_test == 1
    
    axes[1].scatter(latent_test[normal_mask, 0], latent_test[normal_mask, 1],
                   c='blue', alpha=0.5, s=30, label='Normal Test Data')
    axes[1].scatter(latent_test[anomaly_mask, 0], latent_test[anomaly_mask, 1],
                   c='red', alpha=0.7, s=50, marker='x', label='True Anomalies')
    axes[1].set_xlabel('Latent Dimension 1', fontsize=12)
    axes[1].set_ylabel('Latent Dimension 2', fontsize=12)
    axes[1].set_title('Latent Space - Test Data', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'latent_space.png'), dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Latent space visualization saved to {os.path.join(OUTPUT_DIR, 'latent_space.png')}")


def analyze_detected_anomalies(X_test, y_true, y_pred, errors, feature_names, top_n=5):
    """Analyze top detected anomalies."""
    print(f"\nAnalyzing Top {top_n} Detected Anomalies:")
    print("-" * 70)
    
    # Get indices of top errors
    top_indices = np.argsort(errors)[-top_n:][::-1]
    
    for i, idx in enumerate(top_indices, 1):
        true_label = "ANOMALY" if y_true[idx] == 1 else "Normal"
        pred_label = "ANOMALY" if y_pred[idx] == 1 else "Normal"
        error = errors[idx]
        
        print(f"\n{i}. Sample #{idx}")
        print(f"   Reconstruction Error: {error:.6f}")
        print(f"   True Label: {true_label}")
        print(f"   Predicted Label: {pred_label}")
        print(f"   Status: {'✓ Correct' if y_true[idx] == y_pred[idx] else '✗ Incorrect'}")
        
        # Show feature values
        print(f"   Feature Values:")
        for j, (fname, val) in enumerate(zip(feature_names, X_test[idx])):
            print(f"      {fname}: {val:.2f}")


if __name__ == "__main__":
    main()
