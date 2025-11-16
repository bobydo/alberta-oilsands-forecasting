# Keras 3.x: High-level neural network API that uses TensorFlow as backend engine
# We only import keras, but TensorFlow runs behind the scenes to do the actual computation
import keras
from src.common.logger import FileLogger

class ST53Model:
    """Builds LSTM neural network for ST53 time series prediction."""  
    @staticmethod
    def build(window_size: int) -> keras.Model:
        """Create and compile LSTM model. Args: window_size: Number of time steps in input sequence. Returns: Compiled Keras model."""
        # Why Keras?
        # High-level API: Keras (now integrated into TensorFlow) provides a simple, intuitive interface for building neural networks
        # Fast prototyping: Build complex models with just a few lines of code
        # Industry standard: Widely used in production for time series forecasting
        # Built-in LSTM support: Perfect for sequential data like your monthly production values
        logger = FileLogger.setup("model_st53")
        
        try:
            logger.info(f"Building LSTM model with window_size={window_size}")
            
            if not isinstance(window_size, int) or window_size <= 0:
                error_msg = f"window_size must be a positive integer, got {window_size}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            model = keras.Sequential([
                # Input layer: Expects shape (window_size=8, features=1)
                # - 8 timesteps = 8 months of historical production data
                # - 1 feature = single value (bitumen production in m³)
                keras.layers.Input((window_size, 1)),
                
                # First LSTM layer: 64 memory cells with sequence output
                # - 64 units: Enough capacity to learn patterns without overfitting (good for ~453 samples)
                # - return_sequences=True: Outputs full sequence (8 timesteps × 64 features) for next LSTM
                # - Purpose: Captures basic patterns like seasonality, short-term trends
                # - Why 64? Sweet spot between model complexity and dataset size (64 params/sample ratio)
                keras.layers.LSTM(64, return_sequences=True),
                
                # Second LSTM layer: 32 memory cells (funnel architecture)
                # - 32 units: Half the size for dimensionality reduction, prevents overfitting
                # - return_sequences=False (default): Outputs only final state (32 features)
                # - Purpose: Learns complex long-term dependencies and relationships
                # - Why stack? Two layers learn hierarchical patterns: basic → complex
                keras.layers.LSTM(32),
                
                # Output layer: Single prediction value
                # - Dense(1): Fully connected layer that combines all 32 LSTM outputs → 1 value
                # - Sigmoid activation: Constrains output to [0, 1] range (matches scaled training data)
                # - Why sigmoid? Training data is scaled to [0, 1], so predictions must also be [0, 1]
                # - Problem solved: Without activation, model could output negative (e.g., -0.0187 → -763 m³)
                # - Purpose: Final forecast for next month's bitumen production (scaled, then inverse-transformed)
                keras.layers.Dense(1, activation='sigmoid')
            ])
            # Why Adam?
            # Combines best features of other optimizers (momentum + adaptive learning rates)
            # Automatically adjusts learning speed for each paramete
            # MSE (Mean Squared Error) measures how wrong your predictions are.
            model.compile(optimizer="adam", loss="mse")
            
            logger.info("Model built and compiled successfully")
            return model
            
        except ValueError as e:
            logger.error(f"Model build error: {e}")
            raise ValueError(f"Model build error: {e}")
        except Exception as e:
            logger.error(f"Failed to build ST53 model: {e}", exc_info=True)
            raise Exception(f"Failed to build ST53 model: {e}")