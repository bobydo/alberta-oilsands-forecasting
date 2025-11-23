"""
Autoencoder for Anomaly Detection in Time Series Data
======================================================
This module implements an autoencoder neural network for detecting anomalies
in oil sands production data using reconstruction error.

Author: Your Name
Date: 2025-11-23
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeSeriesAutoencoder:
    """
    Autoencoder model for time series anomaly detection.
    
    The autoencoder learns to reconstruct normal data patterns. When it encounters
    anomalous data, the reconstruction error will be significantly higher.
    
    Architecture:
        Encoder: Input → Dense(64) → Dense(32) → Dense(16) → Latent(8)
        Decoder: Latent(8) → Dense(16) → Dense(32) → Dense(64) → Output
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 8):
        """
        Initialize the autoencoder model.
        
        Args:
            input_dim (int): Number of input features
            encoding_dim (int): Dimension of the encoded (latent) representation
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.history = None
        
        logger.info(f"Initialized TimeSeriesAutoencoder with input_dim={input_dim}, encoding_dim={encoding_dim}")
    
    def build_model(self):
        """
        Build the autoencoder architecture with encoder and decoder components.
        """
        # Input layer
        input_layer = keras.Input(shape=(self.input_dim,))
        
        # Encoder layers (compress the data)
        encoded = layers.Dense(64, activation='relu', name='encoder_dense_1')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(32, activation='relu', name='encoder_dense_2')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(16, activation='relu', name='encoder_dense_3')(encoded)
        
        # Latent space (bottleneck) - the compressed representation
        latent = layers.Dense(self.encoding_dim, activation='relu', name='latent_space')(encoded)
        
        # Decoder layers (reconstruct the data)
        decoded = layers.Dense(16, activation='relu', name='decoder_dense_1')(latent)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(32, activation='relu', name='decoder_dense_2')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(64, activation='relu', name='decoder_dense_3')(decoded)
        
        # Output layer (same dimension as input)
        output_layer = layers.Dense(self.input_dim, activation='linear', name='output')(decoded)
        
        # Complete autoencoder model
        self.model = Model(inputs=input_layer, outputs=output_layer, name='autoencoder')
        
        # Encoder model (for extracting latent representations)
        self.encoder = Model(inputs=input_layer, outputs=latent, name='encoder')
        
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean Squared Error for reconstruction
            metrics=['mae']  # Mean Absolute Error
        )
        
        logger.info("Autoencoder model built successfully")
        logger.info(f"Total parameters: {self.model.count_params():,}")
    
    def get_model_summary(self):
        """Print the model architecture summary."""
        if self.model is None:
            logger.warning("Model not built yet. Call build_model() first.")
            return
        
        print("\n" + "="*70)
        print("AUTOENCODER MODEL SUMMARY")
        print("="*70)
        self.model.summary()
        print("\n" + "="*70)
        print("ENCODER MODEL SUMMARY")
        print("="*70)
        self.encoder.summary()
    
    def preprocess_data(self, X, fit_scaler=False):
        """
        Normalize the input data using StandardScaler.
        
        Args:
            X (np.ndarray): Input features
            fit_scaler (bool): Whether to fit the scaler (True for training data)
        
        Returns:
            np.ndarray: Normalized features
        """
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Fitted scaler on training data")
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def train(self, X_train, X_val=None, epochs=100, batch_size=32, verbose=1):
        """
        Train the autoencoder model.
        
        Args:
            X_train (np.ndarray): Training data (normal data only)
            X_val (np.ndarray): Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
        
        Returns:
            keras.callbacks.History: Training history
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() first.")
            raise ValueError("Model must be built before training")
        
        # Preprocess training data
        X_train_scaled = self.preprocess_data(X_train, fit_scaler=True)
        
        # Preprocess validation data if provided
        validation_data = None
        if X_val is not None:
            X_val_scaled = self.preprocess_data(X_val, fit_scaler=False)
            validation_data = (X_val_scaled, X_val_scaled)
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Reduce learning rate when learning plateaus
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        logger.info(f"Starting training for {epochs} epochs with batch_size={batch_size}")
        
        # Train the model (input = output for autoencoder)
        self.history = self.model.fit(
            X_train_scaled, X_train_scaled,  # Autoencoder learns to reconstruct input
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        logger.info("Training completed successfully")
        return self.history
    
    def calculate_reconstruction_error(self, X):
        """
        Calculate reconstruction error (MSE) for each sample.
        
        Args:
            X (np.ndarray): Input data
        
        Returns:
            np.ndarray: Reconstruction errors for each sample
        """
        X_scaled = self.preprocess_data(X, fit_scaler=False)
        X_reconstructed = self.model.predict(X_scaled, verbose=0)
        
        # Calculate Mean Squared Error for each sample
        reconstruction_errors = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
        
        return reconstruction_errors
    
    def set_threshold(self, X_normal, percentile=95):
        """
        Set the anomaly detection threshold based on normal data.
        
        Args:
            X_normal (np.ndarray): Normal (non-anomalous) data
            percentile (float): Percentile to use as threshold (e.g., 95 means 5% false positive rate)
        
        Returns:
            float: The threshold value
        """
        reconstruction_errors = self.calculate_reconstruction_error(X_normal)
        self.threshold = np.percentile(reconstruction_errors, percentile)
        
        logger.info(f"Threshold set at {percentile}th percentile: {self.threshold:.6f}")
        logger.info(f"Min error: {reconstruction_errors.min():.6f}, Max error: {reconstruction_errors.max():.6f}")
        
        return self.threshold
    
    def predict_anomalies(self, X, threshold=None):
        """
        Predict anomalies based on reconstruction error.
        
        Args:
            X (np.ndarray): Input data to check for anomalies
            threshold (float): Custom threshold (uses self.threshold if None)
        
        Returns:
            tuple: (anomaly_labels, reconstruction_errors)
                - anomaly_labels: 1 for anomaly, 0 for normal
                - reconstruction_errors: reconstruction error for each sample
        """
        if threshold is None:
            if self.threshold is None:
                logger.error("Threshold not set. Call set_threshold() first or provide a threshold.")
                raise ValueError("Threshold must be set before predicting anomalies")
            threshold = self.threshold
        
        reconstruction_errors = self.calculate_reconstruction_error(X)
        anomaly_labels = (reconstruction_errors > threshold).astype(int)
        
        num_anomalies = anomaly_labels.sum()
        anomaly_rate = (num_anomalies / len(X)) * 100
        
        logger.info(f"Detected {num_anomalies} anomalies ({anomaly_rate:.2f}% of data)")
        
        return anomaly_labels, reconstruction_errors
    
    def get_latent_representation(self, X):
        """
        Get the latent (encoded) representation of the input data.
        
        Args:
            X (np.ndarray): Input data
        
        Returns:
            np.ndarray: Latent representations
        """
        if self.encoder is None:
            logger.error("Encoder not built. Call build_model() first.")
            raise ValueError("Encoder must be built before extracting latent representations")
        
        X_scaled = self.preprocess_data(X, fit_scaler=False)
        latent_repr = self.encoder.predict(X_scaled, verbose=0)
        
        return latent_repr
    
    def plot_training_history(self, figsize=(12, 4), save_path='output'):
        """
        Plot training and validation loss curves.
        
        Args:
            figsize (tuple): Figure size (width, height)
            save_path (str): Directory to save the plot
        """
        if self.history is None:
            logger.warning("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[1].plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        if 'val_mae' in self.history.history:
            axes[1].plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Mean Absolute Error During Training', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            output_file = os.path.join(save_path, 'training_history.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {output_file}")
        
        plt.show()
    
    def plot_reconstruction_error_distribution(self, X_normal, X_test, anomaly_labels=None, bins=50, save_path='output'):
        """
        Plot the distribution of reconstruction errors for normal and test data.
        
        Args:
            X_normal (np.ndarray): Normal training data
            X_test (np.ndarray): Test data
            anomaly_labels (np.ndarray): Binary labels (1=anomaly, 0=normal)
            bins (int): Number of histogram bins
            save_path (str): Directory to save the plot
        """
        normal_errors = self.calculate_reconstruction_error(X_normal)
        test_errors = self.calculate_reconstruction_error(X_test)
        
        plt.figure(figsize=(12, 6))
        
        # Plot histograms
        plt.hist(normal_errors, bins=bins, alpha=0.6, label='Normal Data', color='green', edgecolor='black')
        
        if anomaly_labels is not None:
            normal_test = test_errors[anomaly_labels == 0]
            anomalous_test = test_errors[anomaly_labels == 1]
            
            if len(normal_test) > 0:
                plt.hist(normal_test, bins=bins, alpha=0.6, label='Test (Normal)', color='blue', edgecolor='black')
            if len(anomalous_test) > 0:
                plt.hist(anomalous_test, bins=bins, alpha=0.6, label='Test (Anomaly)', color='red', edgecolor='black')
        else:
            plt.hist(test_errors, bins=bins, alpha=0.6, label='Test Data', color='blue', edgecolor='black')
        
        # Plot threshold line
        if self.threshold is not None:
            plt.axvline(self.threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold = {self.threshold:.4f}')
        
        plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Reconstruction Errors', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            output_file = os.path.join(save_path, 'reconstruction_error_distribution.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Reconstruction error distribution plot saved to {output_file}")
        
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            logger.error("No model to save. Build and train the model first.")
            return
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        # Rebuild encoder from loaded model
        self.encoder = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('latent_space').output,
            name='encoder'
        )
