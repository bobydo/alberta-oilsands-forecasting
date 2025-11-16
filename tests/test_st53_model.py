"""Unit tests for ST53 LSTM model.

Run with: pytest tests/test_st53_model.py -v
Or: python -m pytest tests/test_st53_model.py -v
"""

import pytest
import numpy as np
import os
from src.st53.inference_st53 import ST53Predictor
from src.st53.model_st53 import ST53Model


class TestST53Predictor:
    """Test suite for ST53 prediction model"""
    
    @pytest.fixture
    def predictor(self):
        """Load model once for all tests"""
        return ST53Predictor("models")
    
    def test_no_negative_predictions(self, predictor):
        """Model should NEVER predict negative production values"""
        test_inputs = [
            [8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700],  # Increasing
            [30000, 29000, 28000, 27000, 26000, 25000, 24000, 23000],  # Decreasing
            [10000, 10500, 9500, 10200, 9800, 10100, 9900, 10000],  # Volatile
            [1000] * 8,  # Very low
            [40000] * 8,  # Very high
        ]
        
        for values in test_inputs:
            prediction = predictor.predict(values)
            assert prediction >= 0, f"Negative prediction: {prediction} for input {values}"
    
    def test_predictions_in_reasonable_range(self, predictor):
        """Predictions should be within reasonable bounds (0-50,000 m³)"""
        test_inputs = [
            [8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700],
            [15000, 15500, 15200, 15800, 15300, 15900, 15400, 16000],
            [25000, 24500, 25500, 24800, 25200, 24700, 25300, 24900],
        ]
        
        for values in test_inputs:
            prediction = predictor.predict(values)
            assert 0 <= prediction <= 50000, f"Unrealistic prediction: {prediction} for input {values}"
    
    def test_prediction_consistency(self, predictor):
        """Same input should always give same output (deterministic)"""
        input_data = [8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700]
        
        pred1 = predictor.predict(input_data)
        pred2 = predictor.predict(input_data)
        pred3 = predictor.predict(input_data)
        
        assert abs(pred1 - pred2) < 0.01, f"Non-deterministic predictions: {pred1} vs {pred2}"
        assert abs(pred2 - pred3) < 0.01, f"Non-deterministic predictions: {pred2} vs {pred3}"
    
    def test_prediction_near_input_average(self, predictor):
        """For stable inputs, prediction should be close to average"""
        stable_input = [8000] * 8
        avg = 8000
        
        prediction = predictor.predict(stable_input)
        
        # Prediction should be within 50% of average for stable input
        assert 0.5 * avg <= prediction <= 1.5 * avg, \
            f"Prediction {prediction} too far from average {avg}"
    
    def test_wrong_input_length_raises_error(self, predictor):
        """Should raise ValueError for incorrect input length"""
        with pytest.raises(ValueError, match="Expected 8 values"):
            predictor.predict([8000, 8100, 8200])  # Only 3 values
        
        with pytest.raises(ValueError, match="Expected 8 values"):
            predictor.predict([8000] * 10)  # 10 values
    
    def test_non_numeric_input_raises_error(self, predictor):
        """Should raise TypeError for non-numeric input"""
        with pytest.raises(TypeError, match="must be numeric"):
            predictor.predict(["8000", "8100", "8200", "8300", "8400", "8500", "8600", "8700"])
        
        with pytest.raises(TypeError, match="must be numeric"):
            predictor.predict([8000, 8100, None, 8300, 8400, 8500, 8600, 8700])


class TestST53ModelArchitecture:
    """Test model architecture and configuration"""
    
    def test_model_output_activation_is_sigmoid(self):
        """Output layer should have sigmoid activation to constrain [0,1]"""
        import keras
        
        model = keras.models.load_model("models/st53_model.keras")
        last_layer = model.layers[-1]
        
        assert hasattr(last_layer, 'activation'), "Output layer should have activation"
        activation_name = last_layer.activation.__name__
        assert activation_name == 'sigmoid', \
            f"Output activation should be 'sigmoid', got '{activation_name}'"
    
    def test_model_output_range_with_sigmoid(self):
        """With sigmoid activation, model output should be in [0,1] range"""
        import keras
        
        model = keras.models.load_model("models/st53_model.keras")
        
        # Test with random scaled inputs
        test_inputs = np.random.rand(100, 8, 1)
        outputs = model.predict(test_inputs, verbose=0)
        
        assert outputs.min() >= 0, f"Model output minimum {outputs.min()} is negative!"
        assert outputs.max() <= 1, f"Model output maximum {outputs.max()} exceeds 1!"
    
    def test_model_builder_creates_valid_model(self):
        """ST53Model.build() should create valid Keras model"""
        model = ST53Model.build(window_size=8)
        
        assert model is not None
        assert len(model.layers) == 4  # Input, LSTM, LSTM, Dense
        
        # Check output layer
        last_layer = model.layers[-1]
        assert last_layer.activation.__name__ == 'sigmoid'


class TestST53Scaler:
    """Test data scaling functionality"""
    
    @pytest.fixture
    def predictor(self):
        return ST53Predictor("models")
    
    def test_scaler_loaded_correctly(self, predictor):
        """Scaler should be loaded and have valid parameters"""
        scaler = predictor.scaler
        
        assert hasattr(scaler, 'data_min_'), "Scaler missing data_min_"
        assert hasattr(scaler, 'data_max_'), "Scaler missing data_max_"
        assert scaler.data_min_[0] >= 0, "Scaler data_min should be non-negative"
        assert scaler.data_max_[0] > scaler.data_min_[0], "data_max should be > data_min"
    
    def test_scaler_transforms_to_0_1_range(self, predictor):
        """Scaler should transform values to [0,1] range"""
        scaler = predictor.scaler
        
        # Test with values in expected range
        test_values = np.array([[5000], [10000], [20000], [30000]])
        scaled = scaler.transform(test_values)
        
        assert scaled.min() >= 0, f"Scaled minimum {scaled.min()} is negative"
        assert scaled.max() <= 1, f"Scaled maximum {scaled.max()} exceeds 1"


class TestRealWorldData:
    """Test with real Cenovus production data"""
    
    @pytest.fixture
    def predictor(self):
        return ST53Predictor("models")
    
    def test_cenovus_sunrise_prediction(self, predictor):
        """Test with real Cenovus Sunrise data"""
        sunrise_data = [8232.27, 8086.21, 7690.36, 8227.58, 8423.98, 8405.0, 8477.05, 7892.54]
        prediction = predictor.predict(sunrise_data)
        
        assert prediction >= 0, f"Negative prediction: {prediction}"
        assert 5000 <= prediction <= 12000, \
            f"Sunrise prediction {prediction} outside expected range [5000, 12000]"
    
    def test_cenovus_foster_creek_prediction(self, predictor):
        """Test with real Cenovus Foster Creek data"""
        foster_creek_data = [30717.12, 32122.67, 30897.38, 30137.87, 29907.24, 31504.45, 30909.04, 30768.31]
        prediction = predictor.predict(foster_creek_data)
        
        assert prediction >= 0, f"Negative prediction: {prediction}"
        assert 25000 <= prediction <= 35000, \
            f"Foster Creek prediction {prediction} outside expected range [25000, 35000]"
    
    def test_cenovus_christina_lake_prediction(self, predictor):
        """Test with real Cenovus Christina Lake data"""
        christina_lake_data = [38440.48, 38453.59, 38345.22, 23973.98, 40922.27, 40339.68, 38701.56, 37223.34]
        prediction = predictor.predict(christina_lake_data)
        
        assert prediction >= 0, f"Negative prediction: {prediction}"
        assert 20000 <= prediction <= 45000, \
            f"Christina Lake prediction {prediction} outside expected range [20000, 45000]"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
