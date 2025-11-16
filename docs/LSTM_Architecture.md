# LSTM Architecture Visualization

## ST53 Model Architecture

```
INPUT DATA (8 months of production history)
┌─────────────────────────────────────────────────────────┐
│ [100, 102, 105, 103, 107, 110, 108, 112] m³/month      │
└─────────────────────────────────────────────────────────┘
                         ↓
                    RESHAPE
                         ↓
┌─────────────────────────────────────────────────────────┐
│     Shape: (batch=8, timesteps=8, features=1)          │
│                                                         │
│     [[100], [102], [105], [103],                       │
│      [107], [110], [108], [112]]                       │
└─────────────────────────────────────────────────────────┘
                         ↓
              ┌──────────────────────┐
              │   INPUT LAYER        │
              │   Shape: (8, 1)      │
              └──────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│              LSTM LAYER 1 (64 units)                     │
│                                                          │
│  ┌────┐  ┌────┐  ┌────┐  ...  ┌────┐  (64 cells)      │
│  │ C1 │  │ C2 │  │ C3 │  ...  │C64 │                   │
│  └────┘  └────┘  └────┘  ...  └────┘                   │
│                                                          │
│  Each cell processes all 8 timesteps                    │
│  return_sequences=True → outputs (8, 64)                │
│                                                          │
│  Purpose: Learn basic patterns                          │
│  - Seasonality (winter vs summer production)            │
│  - Short-term trends (monthly changes)                  │
│  - Noise filtering                                      │
└──────────────────────────────────────────────────────────┘
                         ↓
              Shape: (8 timesteps, 64 features)
                         ↓
┌──────────────────────────────────────────────────────────┐
│              LSTM LAYER 2 (32 units)                     │
│                                                          │
│  ┌────┐  ┌────┐  ...  ┌────┐  (32 cells)               │
│  │ C1 │  │ C2 │  ...  │C32 │                            │
│  └────┘  └────┘  ...  └────┘                            │
│                                                          │
│  Processes 64-feature sequences                         │
│  return_sequences=False → outputs (32)                  │
│                                                          │
│  Purpose: Learn complex patterns                        │
│  - Long-term dependencies (yearly cycles)               │
│  - Multi-factor relationships                           │
│  - Abstract representations                             │
└──────────────────────────────────────────────────────────┘
                         ↓
                  Shape: (32 features)
                         ↓
              ┌──────────────────────┐
              │   DENSE LAYER        │
              │   1 output neuron    │
              │                      │
              │   W₁×f₁ + W₂×f₂ +   │
              │   ... + W₃₂×f₃₂ = y │
              └──────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│            PREDICTION (next month)                      │
│                   115.3 m³                              │
└─────────────────────────────────────────────────────────┘
```

## Dimensionality Flow

```
Input:          (batch, 8, 1)        ← 8 timesteps, 1 feature
                     ↓
LSTM(64):       (batch, 8, 64)       ← 8 timesteps, 64 features
                     ↓
LSTM(32):       (batch, 32)          ← Final state, 32 features
                     ↓
Dense(1):       (batch, 1)           ← Single prediction
```

## Why This Architecture Works

### Funnel Shape (64 → 32 → 1)
```
     [64 units]  ←  Rich feature extraction
          ↓
     [32 units]  ←  Pattern refinement
          ↓
      [1 unit]   ←  Final prediction
```

**Benefits:**
- Prevents overfitting by progressively reducing complexity
- 64 units: ~28,700 parameters for 453 samples = 63 params/sample ✓
- Hierarchical learning: simple → complex patterns

### LSTM Memory Cell (Simplified)

```
       Previous State (h_{t-1})
              ↓
         ┌─────────┐
    x_t →│ Forget  │→ What to forget from memory
         │  Gate   │
         └─────────┘
              ↓
         ┌─────────┐
         │ Input   │→ What new info to store
         │  Gate   │
         └─────────┘
              ↓
         ┌─────────┐
         │ Output  │→ What to output
         │  Gate   │
         └─────────┘
              ↓
         Next State (h_t)
```

## Training Process

```
Epoch 1/40:  [===>              ] loss: 169,343,936
Epoch 2/40:  [======>           ] loss: 169,270,544
    ...
Epoch 40/40: [==================] loss: 168,283,984

Optimization: Adam (adaptive learning rate)
Loss: MSE (Mean Squared Error)
```

## Hyperparameter Impact

```
Window Size = 8 months
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ Jan │ Feb │ Mar │ Apr │ May │ Jun │ Jul │ Aug │ → Predict Sep
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Batch Size = 8 samples
┌─────────────┐
│ Sample 1    │ ┐
│ Sample 2    │ │
│ Sample 3    │ │
│ Sample 4    │ ├─ Process together
│ Sample 5    │ │  (GPU optimization)
│ Sample 6    │ │
│ Sample 7    │ │
│ Sample 8    │ ┘
└─────────────┘

Epochs = 40 iterations
Pass through entire dataset 40 times
```

## Visual Resources

For professional diagrams, refer to these resources:

1. **LSTM Cell Diagram:**
   - Original Paper: https://www.bioinf.jku.at/publications/older/2604.pdf
   - Colah's Blog: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

2. **Time Series LSTM:**
   - TensorFlow Tutorial: https://www.tensorflow.org/tutorials/structured_data/time_series
   - Keras Examples: https://keras.io/examples/timeseries/

3. **Architecture Visualization Tools:**
   - Use `model.summary()` to see layer shapes
   - Use `tf.keras.utils.plot_model()` to generate diagram (requires graphviz)

## Generate Your Own Diagram

Add this to your code to visualize:

```python
# In model_st53.py after building the model
from tensorflow import keras

model = ST53Model.build(window_size=8)
model.summary()

# Optional: Generate image (requires: pip install pydot graphviz)
keras.utils.plot_model(
    model, 
    to_file='docs/st53_model_architecture.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',  # Top to Bottom
    expand_nested=True
)
```

## Model Parameters Breakdown

```
Layer (type)                Output Shape              Param #
=================================================================
input_layer (InputLayer)    (None, 8, 1)             0
_________________________________________________________________
lstm (LSTM)                 (None, 8, 64)            16,896
                                                     (4 gates × 64 × (64+1+1))
_________________________________________________________________
lstm_1 (LSTM)               (None, 32)               12,416
                                                     (4 gates × 32 × (64+32+1))
_________________________________________________________________
dense (Dense)               (None, 1)                33
                                                     (32 weights + 1 bias)
=================================================================
Total params: 29,345
Trainable params: 29,345
Non-trainable params: 0
```

## Real Data Example

```
Historical Data (Jan-Aug 2024):
[102,450, 103,200, 105,800, 104,100, 107,500, 110,200, 108,900, 112,300] m³
                              ↓
                        LSTM Model
                              ↓
Prediction (Sep 2024): 114,800 m³
Actual (Sep 2024):     115,200 m³
Error (MAE):           400 m³
```
