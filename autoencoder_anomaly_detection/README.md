# Autoencoder Anomaly Detection

**Deep learning-based anomaly detection using autoencoder neural networks for oil sands production monitoring.**

## 📋 Overview

This module demonstrates how to use **autoencoder neural networks** for detecting anomalies in time series production data. Autoencoders learn to compress and reconstruct normal operating patterns. When they encounter anomalous data, the reconstruction error is significantly higher, allowing us to detect unusual behavior.

### Key Concepts

- **Autoencoder**: A neural network that learns to compress data into a lower-dimensional representation (encoding) and then reconstruct it (decoding)
- **Reconstruction Error**: The difference between original and reconstructed data (MSE)
- **Anomaly Detection**: Samples with high reconstruction error are flagged as anomalies
- **Latent Space**: The compressed representation learned by the encoder

## 🏗️ Architecture

```
Input (10 features)
    ↓
Encoder: Dense(64) → Dense(32) → Dense(16) → Latent(8)
    ↓ (Bottleneck: 8 dimensions)
Decoder: Latent(8) → Dense(16) → Dense(32) → Dense(64) → Output(10)
    ↓
Reconstruction Loss (MSE)
```

**Key Points:**
- **Bottleneck**: Forces the model to learn compressed representations
- **Symmetric Architecture**: Encoder mirrors decoder for balanced learning
- **Dropout Layers**: Prevents overfitting during training
- **MSE Loss**: Measures reconstruction quality

## 📁 Files

```
autoencoder_anomaly_detection/
├── autoencoder_model.py      # TimeSeriesAutoencoder class
├── data_generator.py          # Synthetic data generation
├── example_usage.py           # Complete example workflow
├── README.md                  # This file
└── trained_autoencoder.keras  # Saved model (generated after training)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r reqirements.txt
```

### 2. Run the Example

```bash
python autoencoder_anomaly_detection/example_usage.py
```

This will:
1. ✅ Generate synthetic production data with anomalies
2. ✅ Build and train the autoencoder
3. ✅ Set anomaly detection threshold
4. ✅ Detect anomalies in test data
5. ✅ Visualize results (training curves, ROC curve, error distributions)
6. ✅ Save the trained model

## 📊 Example Workflow

### Step 1: Generate Data

```python
from data_generator import SyntheticProductionDataGenerator

# Create data generator
generator = SyntheticProductionDataGenerator(
    num_samples=2000,
    num_features=10,
    anomaly_ratio=0.05,  # 5% anomalies
    random_seed=42
)

# Generate data
X, y = generator.generate_data()
```

### Step 2: Build Autoencoder

```python
from autoencoder_model import TimeSeriesAutoencoder

# Initialize autoencoder
autoencoder = TimeSeriesAutoencoder(
    input_dim=10,      # Number of features
    encoding_dim=4     # Latent dimension (compression)
)

# Build model
autoencoder.build_model()
autoencoder.get_model_summary()
```

### Step 3: Train on Normal Data

```python
# Train only on normal (non-anomalous) data
X_train_normal = X[y == 0][:1000]
X_val_normal = X[y == 0][1000:1300]

# Train the autoencoder
autoencoder.train(
    X_train=X_train_normal,
    X_val=X_val_normal,
    epochs=100,
    batch_size=32
)
```

### Step 4: Set Threshold

```python
# Set threshold at 95th percentile (5% false positive rate)
threshold = autoencoder.set_threshold(X_train_normal, percentile=95)
print(f"Threshold: {threshold:.6f}")
```

### Step 5: Detect Anomalies

```python
# Predict on test data
X_test = X[1300:]
y_test = y[1300:]

y_pred, reconstruction_errors = autoencoder.predict_anomalies(X_test)

# Evaluate performance
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

### Step 6: Visualize Results

```python
# Plot reconstruction error distribution
autoencoder.plot_reconstruction_error_distribution(
    X_normal=X_train_normal,
    X_test=X_test,
    anomaly_labels=y_pred
)

# Plot training history
autoencoder.plot_training_history()
```

## 🎯 Use Cases

### 1. Production Monitoring
- Detect unusual production volumes
- Identify equipment malfunctions
- Monitor steam injection anomalies

### 2. Predictive Maintenance
- Detect early signs of equipment failure
- Identify gradual performance degradation
- Trigger maintenance alerts

### 3. Process Optimization
- Identify suboptimal operating conditions
- Detect process deviations
- Monitor efficiency metrics

### 4. Quality Control
- Detect abnormal product quality
- Monitor contamination levels
- Identify process variations

## 📈 Performance Metrics

The example achieves:
- **ROC-AUC**: ~0.95-0.98 on synthetic data
- **Precision**: ~85-90%
- **Recall**: ~80-90%
- **F1-Score**: ~85-90%

*Performance varies based on anomaly type and threshold setting.*

## 🔧 Customization

### Change Model Architecture

```python
# Modify in autoencoder_model.py build_model()
encoded = layers.Dense(128, activation='relu')(input_layer)  # Larger
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)
latent = layers.Dense(16, activation='relu')(encoded)  # Larger latent space
```

### Adjust Training Parameters

```python
autoencoder.train(
    X_train=X_train,
    X_val=X_val,
    epochs=200,          # More epochs
    batch_size=64,       # Larger batches
    verbose=1
)
```

### Tune Threshold

```python
# More strict (fewer false positives, may miss some anomalies)
threshold = autoencoder.set_threshold(X_normal, percentile=99)

# More lenient (more false positives, catches more anomalies)
threshold = autoencoder.set_threshold(X_normal, percentile=90)
```

## 🎓 Key Learnings

### Why Autoencoders for Anomaly Detection?

1. **Unsupervised Learning**: No need for labeled anomaly data
2. **Captures Complex Patterns**: Learns non-linear relationships
3. **Dimensionality Reduction**: Compresses data to essential features
4. **Reconstruction-Based**: Simple and interpretable anomaly score

### Limitations

- Requires sufficient normal data for training
- May not detect novel anomaly types
- Threshold selection affects performance
- Computationally more expensive than statistical methods

### Best Practices

1. ✅ Train only on normal data (no anomalies in training set)
2. ✅ Normalize/scale features before training
3. ✅ Use early stopping to prevent overfitting
4. ✅ Validate threshold on separate validation set
5. ✅ Monitor reconstruction errors over time
6. ✅ Retrain periodically with recent normal data

## 📚 References

- **Original Paper**: Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks.
- **Anomaly Detection**: Chalapathy, R., & Chawla, S. (2019). Deep learning for anomaly detection: A survey.
- **TensorFlow/Keras**: https://www.tensorflow.org/tutorials/generative/autoencoder

## 🤝 Contributing

Feel free to extend this example with:
- Variational Autoencoders (VAE)
- LSTM Autoencoders for time series
- Convolutional Autoencoders for spatial data
- Ensemble methods combining multiple autoencoders

## 📝 License

This example is part of the Alberta Oil Sands Forecasting project.

---

**Note**: This is an educational example using synthetic data. For production use, train on real historical production data and validate thoroughly before deployment.
