"""
Autoencoder Anomaly Detection Module
====================================
Deep learning-based anomaly detection using autoencoder neural networks.

Classes:
    TimeSeriesAutoencoder: Main autoencoder model for anomaly detection
    SyntheticProductionDataGenerator: Generate synthetic data for testing

Example:
    >>> from autoencoder_model import TimeSeriesAutoencoder
    >>> autoencoder = TimeSeriesAutoencoder(input_dim=10, encoding_dim=4)
    >>> autoencoder.build_model()
    >>> autoencoder.train(X_train, epochs=100)
    >>> y_pred, errors = autoencoder.predict_anomalies(X_test)
"""

from .autoencoder_model import TimeSeriesAutoencoder
from .data_generator import SyntheticProductionDataGenerator

__version__ = '1.0.0'
__all__ = ['TimeSeriesAutoencoder', 'SyntheticProductionDataGenerator']
