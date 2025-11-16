"""
Unit tests for ST53 SAGD production forecasting model.

Tests cover:
- Model architecture validation
- Data preprocessing pipeline
- Prediction accuracy
- Edge cases and error handling
- Integration tests
"""

import unittest
import numpy as np
import os
import tempfile
import shutil
from src.st53.model_st53 import ST53Model
from src.st53.inference_st53 import ST53Predictor
from src.st53.preprocess_st53 import ST53DataProcessor
from src.common.window import WindowGenerator
from sklearn.preprocessing import MinMaxScaler
import joblib


class TestST53Model(unittest.TestCase):
    """Test suite for ST53 model architecture."""
    
    def test_model_build_valid_window(self):
        """Test model builds successfully with valid window size."""
        model = ST53Model.build(window_size=8)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 3)  # 2 LSTM + 1 Dense
    
    def test_model_has_sigmoid_activation(self):
        """Test output layer has sigmoid activation to prevent negative predictions."""
        model = ST53Model.build(window_size=8)
        output_layer = model.layers[-1]
        config = output_layer.get_config()
        self.assertEqual(config['activation'], 'sigmoid')
    
    def test_model_invalid_window_size(self):
        """Test model raises error for invalid window sizes."""
        with self.assertRaises(ValueError):
            ST53Model.build(window_size=0)
        with self.assertRaises(ValueError):
            ST53Model.build(window_size=-5)
    
    def test_model_output_shape(self):
        """Test model outputs single value prediction."""
        model = ST53Model.build(window_size=8)
        test_input = np.random.rand(1, 8, 1)  # (batch, timesteps, features)
        prediction = model.predict(test_input, verbose=0)
        self.assertEqual(prediction.shape, (1, 1))


class TestST53Predictor(unittest.TestCase):
    """Test suite for ST53 predictor inference."""
    
    @classmethod
    def setUpClass(cls):
        """Create temporary model files for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create a simple test model
        model = ST53Model.build(window_size=8)
        model.save(f"{cls.temp_dir}/st53_model.keras")
        
        # Create test scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        test_data = np.array([[8791.24], [40922.27]])  # Min and max from large ops
        scaler.fit(test_data)
        joblib.dump(scaler, f"{cls.temp_dir}/st53_scaler.pkl")
        
        # Create metadata
        joblib.dump({"window": 8}, f"{cls.temp_dir}/st53_meta.pkl")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        shutil.rmtree(cls.temp_dir)
    
    def test_predictor_initialization(self):
        """Test predictor loads model correctly."""
        predictor = ST53Predictor(self.temp_dir)
        self.assertEqual(predictor.window_size, 8)
        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.scaler)
    
    def test_predict_valid_input(self):
        """Test prediction with valid input data."""
        predictor = ST53Predictor(self.temp_dir)
        test_values = [30000.0] * 8
        prediction = predictor.predict(test_values)
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)  # No negative predictions
    
    def test_predict_wrong_length(self):
        """Test prediction raises error for wrong input length."""
        predictor = ST53Predictor(self.temp_dir)
        with self.assertRaises(ValueError):
            predictor.predict([30000.0] * 5)  # Wrong length
    
    def test_predict_non_numeric(self):
        """Test prediction raises error for non-numeric input."""
        predictor = ST53Predictor(self.temp_dir)
        with self.assertRaises(TypeError):
            predictor.predict([30000.0, "invalid", 30000.0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0])
    
    def test_no_negative_predictions(self):
        """Test model never predicts negative values."""
        predictor = ST53Predictor(self.temp_dir)
        # Test with various input patterns
        test_cases = [
            [10000.0] * 8,  # Constant
            [i * 1000 for i in range(1, 9)],  # Increasing
            [i * 1000 for i in range(8, 0, -1)],  # Decreasing
            [10000.0, 20000.0] * 4,  # Oscillating
        ]
        for test_input in test_cases:
            prediction = predictor.predict(test_input)
            self.assertGreaterEqual(prediction, 0.0, 
                f"Negative prediction for input: {test_input}")


class TestWindowGenerator(unittest.TestCase):
    """Test suite for time series windowing."""
    
    def test_window_creation(self):
        """Test window generator creates correct shapes."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        X, y = WindowGenerator.create(data, window_size=3)
        self.assertEqual(X.shape[0], 7)  # 10 - 3 = 7 windows
        self.assertEqual(X.shape[1], 3)  # window_size
    
    def test_window_values(self):
        """Test window generator produces correct values."""
        data = np.array([1, 2, 3, 4, 5])
        X, y = WindowGenerator.create(data, window_size=2)
        np.testing.assert_array_equal(X[0], [1, 2])
        np.testing.assert_array_equal(y[0], 3)
        np.testing.assert_array_equal(X[1], [2, 3])
        np.testing.assert_array_equal(y[1], 4)


class TestST53DataProcessor(unittest.TestCase):
    """Test suite for ST53 data preprocessing."""
    
    def test_processor_loads_real_data(self):
        """Test processor can load real ST53 Excel file."""
        if os.path.exists("data/st53/ST53_2024-12.xls"):
            df = ST53DataProcessor.load("data/st53/ST53_2024-12.xls")
            self.assertGreater(len(df), 0)
            self.assertIn("Bitumen", df.columns)
            self.assertIn("Operator", df.columns)
            self.assertIn("Scheme Name", df.columns)
        else:
            self.skipTest("ST53 data file not found")
    
    def test_processor_file_not_found(self):
        """Test processor raises error for missing file."""
        with self.assertRaises(FileNotFoundError):
            ST53DataProcessor.load("nonexistent_file.xls")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def test_full_prediction_pipeline(self):
        """Test complete pipeline from data to prediction."""
        if not os.path.exists("models/st53_model.keras"):
            self.skipTest("Trained model not found")
        
        predictor = ST53Predictor("models")
        
        # Test with real Cenovus Christina Lake data
        test_input = [38440.48, 38453.59, 38345.22, 23973.98, 
                      40922.27, 40339.68, 38701.56, 37223.34]
        
        prediction = predictor.predict(test_input)
        
        # Validate prediction is reasonable for large operations
        self.assertGreater(prediction, 20000, "Prediction too low for Christina Lake")
        self.assertLess(prediction, 50000, "Prediction unrealistically high")
        self.assertIsInstance(prediction, float)
    
    def test_multiple_predictions_consistency(self):
        """Test model produces consistent predictions for same input."""
        if not os.path.exists("models/st53_model.keras"):
            self.skipTest("Trained model not found")
        
        predictor = ST53Predictor("models")
        test_input = [30000.0] * 8
        
        # Run prediction multiple times
        predictions = [predictor.predict(test_input) for _ in range(5)]
        
        # All predictions should be identical (deterministic model)
        for pred in predictions[1:]:
            self.assertAlmostEqual(pred, predictions[0], places=2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_predict_all_zeros(self):
        """Test prediction with all zero input."""
        if not os.path.exists("models/st53_model.keras"):
            self.skipTest("Trained model not found")
        
        predictor = ST53Predictor("models")
        prediction = predictor.predict([0.0] * 8)
        self.assertGreaterEqual(prediction, 0.0)
    
    def test_predict_very_large_values(self):
        """Test prediction with values at upper bound."""
        if not os.path.exists("models/st53_model.keras"):
            self.skipTest("Trained model not found")
        
        predictor = ST53Predictor("models")
        # Test with maximum production values
        prediction = predictor.predict([45000.0] * 8)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLess(prediction, 50000.0)


def run_tests():
    """Run all tests and display results."""
    print("="*70)
    print("ST53 MODEL UNIT TESTS")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestST53Model))
    suite.addTests(loader.loadTestsFromTestCase(TestST53Predictor))
    suite.addTests(loader.loadTestsFromTestCase(TestWindowGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestST53DataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
