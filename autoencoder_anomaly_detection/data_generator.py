"""
Synthetic Data Generator for Anomaly Detection
==============================================
This module generates synthetic oil sands production data with injected anomalies
for testing autoencoder anomaly detection.

Author: Your Name
Date: 2025-11-23
"""

import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class SyntheticProductionDataGenerator:
    """
    Generate synthetic time series data simulating oil sands production metrics
    with controllable anomaly injection.
    
    Features simulated:
    - Production volume (m³/month)
    - Steam injection rate
    - Temperature
    - Pressure
    - Viscosity
    - Water content
    - Gas oil ratio
    - Energy consumption
    - Equipment efficiency
    - Operating hours
    """
    
    def __init__(self, num_samples: int = 1000, num_features: int = 10, 
                 anomaly_ratio: float = 0.05, random_seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            num_samples (int): Total number of samples to generate
            num_features (int): Number of features to generate
            anomaly_ratio (float): Proportion of anomalous samples (0.0 to 1.0)
            random_seed (int): Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_features = num_features
        self.anomaly_ratio = anomaly_ratio
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        self.feature_names = [
            'production_volume',
            'steam_injection_rate',
            'temperature',
            'pressure',
            'viscosity',
            'water_content',
            'gas_oil_ratio',
            'energy_consumption',
            'equipment_efficiency',
            'operating_hours'
        ][:num_features]
        
        # Normal operating ranges for each feature
        self.normal_ranges = {
            'production_volume': (25000, 35000),        # m³/month
            'steam_injection_rate': (800, 1200),        # tonnes/day
            'temperature': (200, 250),                   # °C
            'pressure': (3000, 4000),                    # kPa
            'viscosity': (50, 150),                      # cP
            'water_content': (10, 30),                   # %
            'gas_oil_ratio': (20, 40),                   # m³/m³
            'energy_consumption': (5000, 8000),          # MWh
            'equipment_efficiency': (85, 95),            # %
            'operating_hours': (650, 720)                # hours/month
        }
    
    def generate_normal_data(self, n_samples: int) -> np.ndarray:
        """
        Generate normal (non-anomalous) production data.
        
        Args:
            n_samples (int): Number of normal samples to generate
        
        Returns:
            np.ndarray: Normal data with shape (n_samples, num_features)
        """
        data = np.zeros((n_samples, self.num_features))
        
        for i, feature_name in enumerate(self.feature_names):
            min_val, max_val = self.normal_ranges[feature_name]
            mean_val = (min_val + max_val) / 2
            std_val = (max_val - min_val) / 6  # ~99.7% within range
            
            # Generate data with slight correlations (more realistic)
            base = np.random.normal(mean_val, std_val, n_samples)
            
            # Add some temporal autocorrelation
            for j in range(1, n_samples):
                base[j] = 0.7 * base[j] + 0.3 * base[j-1]
            
            data[:, i] = base
        
        return data
    
    def inject_anomalies(self, data: np.ndarray, n_anomalies: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject anomalies into the data.
        
        Anomaly types:
        1. Point anomalies: Single extreme values
        2. Contextual anomalies: Values unusual in context
        3. Collective anomalies: Sequences of unusual patterns
        
        Args:
            data (np.ndarray): Normal data
            n_anomalies (int): Number of anomalies to inject
        
        Returns:
            tuple: (data_with_anomalies, labels)
                - data_with_anomalies: Modified data with anomalies
                - labels: Binary array (0=normal, 1=anomaly)
        """
        data_copy = data.copy()
        labels = np.zeros(len(data), dtype=int)
        
        # Randomly select indices for anomalies
        anomaly_indices = np.random.choice(len(data), size=n_anomalies, replace=False)
        labels[anomaly_indices] = 1
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['point', 'multi_feature', 'shift'])
            
            if anomaly_type == 'point':
                # Single feature extreme value
                feature_idx = np.random.randint(0, self.num_features)
                feature_name = self.feature_names[feature_idx]
                min_val, max_val = self.normal_ranges[feature_name]
                
                # Make it extremely high or low
                if np.random.rand() > 0.5:
                    data_copy[idx, feature_idx] = max_val * np.random.uniform(1.5, 2.5)
                else:
                    data_copy[idx, feature_idx] = min_val * np.random.uniform(0.3, 0.7)
            
            elif anomaly_type == 'multi_feature':
                # Multiple features simultaneously anomalous
                num_affected = np.random.randint(2, min(4, self.num_features))
                affected_features = np.random.choice(self.num_features, num_affected, replace=False)
                
                for feature_idx in affected_features:
                    feature_name = self.feature_names[feature_idx]
                    min_val, max_val = self.normal_ranges[feature_name]
                    
                    if np.random.rand() > 0.5:
                        data_copy[idx, feature_idx] = max_val * np.random.uniform(1.3, 2.0)
                    else:
                        data_copy[idx, feature_idx] = min_val * np.random.uniform(0.4, 0.8)
            
            elif anomaly_type == 'shift':
                # Shift all features by unusual amount
                shift_factor = np.random.uniform(-0.3, 0.3)
                for feature_idx in range(self.num_features):
                    feature_name = self.feature_names[feature_idx]
                    min_val, max_val = self.normal_ranges[feature_name]
                    mean_val = (min_val + max_val) / 2
                    
                    # Shift from mean
                    data_copy[idx, feature_idx] += shift_factor * mean_val
        
        logger.info(f"Injected {n_anomalies} anomalies ({n_anomalies/len(data)*100:.2f}%)")
        
        return data_copy, labels
    
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete dataset with normal data and injected anomalies.
        
        Returns:
            tuple: (X, y)
                - X: Feature matrix (num_samples, num_features)
                - y: Binary labels (0=normal, 1=anomaly)
        """
        logger.info(f"Generating {self.num_samples} samples with {self.num_features} features")
        
        # Generate normal data
        data = self.generate_normal_data(self.num_samples)
        
        # Calculate number of anomalies
        n_anomalies = int(self.num_samples * self.anomaly_ratio)
        
        # Inject anomalies
        X, y = self.inject_anomalies(data, n_anomalies)
        
        logger.info(f"Data generation complete: {len(X)} samples, {y.sum()} anomalies")
        
        return X, y
    
    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return self.feature_names
    
    def to_dataframe(self, X: np.ndarray, y: np.ndarray = None) -> pd.DataFrame:
        """
        Convert numpy arrays to pandas DataFrame.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels (optional)
        
        Returns:
            pd.DataFrame: Data in DataFrame format
        """
        df = pd.DataFrame(X, columns=self.feature_names)
        
        if y is not None:
            df['is_anomaly'] = y
        
        return df
