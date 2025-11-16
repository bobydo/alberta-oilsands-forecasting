import keras

class ST39Model:
    """Builds LSTM neural network for ST39 time series prediction."""
    
    @staticmethod
    def build(window_size: int) -> keras.Model:
        """Create and compile LSTM model. Args: window_size: Number of time steps in input sequence. Returns: Compiled Keras model."""
        model = keras.Sequential([
            keras.layers.Input((window_size, 1)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.LSTM(32),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        return model
