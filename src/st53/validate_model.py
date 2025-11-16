"""Validation pipeline to catch issues after model training.
Run this after training to verify model behavior before deployment.
Usage: python -m src.st53.validate_model
"""

import numpy as np
from src.st53.inference_st53 import ST53Predictor
from src.common.logger import FileLogger
import sys

def validate_model(model_path: str = "models") -> bool:
    """Run comprehensive validation checks on trained model.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        True if all checks pass, False otherwise
    """
    logger = FileLogger.setup("validate_st53")
    logger.info("Starting ST53 model validation")
    
    issues = []
    warnings = []
    
    try:
        predictor = ST53Predictor(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # ============================================
    # Test 1: Check for negative predictions
    # ============================================
    logger.info("Test 1: Checking for negative predictions")
    test_cases = [
        ([1000] * 8, "Very low production (1000 m³)"),
        ([5000] * 8, "Low production (5000 m³)"),
        ([10000] * 8, "Medium-low production (10000 m³)"),
        ([20000] * 8, "Medium production (20000 m³)"),
        ([30000] * 8, "High production (30000 m³)"),
        ([40000] * 8, "Very high production (40000 m³)"),
        ([8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700], "Steady increase"),
        ([8700, 8600, 8500, 8400, 8300, 8200, 8100, 8000], "Steady decrease"),
        ([10000, 12000, 9000, 13000, 8500, 14000, 8000, 15000], "Volatile pattern"),
        ([30717.12, 32122.67, 30897.38, 30137.87, 29907.24, 31504.45, 30909.04, 30768.31], "Cenovus Foster Creek"),
    ]
    
    negative_found = False
    for values, description in test_cases:
        try:
            pred = predictor.predict(values)
            if pred < 0:
                issues.append(f"❌ CRITICAL: Negative prediction for '{description}': {pred:.2f} m³")
                negative_found = True
            elif pred > 50000:
                warnings.append(f"⚠️  Unusually high prediction for '{description}': {pred:.2f} m³")
            else:
                logger.info(f"✅ {description}: {pred:.2f} m³")
        except Exception as e:
            issues.append(f"❌ CRITICAL: Prediction failed for '{description}': {e}")
    
    if not negative_found:
        logger.info("✅ Test 1 PASSED: No negative predictions found")
    
    # ============================================
    # Test 2: Check output layer activation
    # ============================================
    logger.info("Test 2: Checking output layer activation function")
    model = predictor.model
    last_layer = model.layers[-1]
    
    if hasattr(last_layer, 'activation'):
        activation_name = last_layer.activation.__name__
        if activation_name == 'linear':
            issues.append("❌ CRITICAL: Output layer has NO activation (should be 'sigmoid')")
            issues.append("   This allows negative predictions! Add activation='sigmoid' to Dense layer.")
        elif activation_name == 'sigmoid':
            logger.info(f"✅ Output activation: {activation_name} (correct for [0,1] scaled data)")
        else:
            warnings.append(f"⚠️  Output activation: {activation_name} (expected 'sigmoid' for scaled data)")
    else:
        warnings.append("⚠️  Could not verify output layer activation")
    
    # ============================================
    # Test 3: Check model output range
    # ============================================
    logger.info("Test 3: Checking raw model output range (should be [0,1] with sigmoid)")
    test_inputs = np.random.rand(10, 8, 1)  # 10 random scaled inputs
    raw_outputs = model.predict(test_inputs, verbose=0)
    
    min_output = raw_outputs.min()
    max_output = raw_outputs.max()
    
    if min_output < 0 or max_output > 1:
        issues.append(f"❌ CRITICAL: Model outputs outside [0,1] range: [{min_output:.4f}, {max_output:.4f}]")
        issues.append("   This indicates missing sigmoid activation or incorrect architecture")
    else:
        logger.info(f"✅ Model output range: [{min_output:.4f}, {max_output:.4f}] (within [0,1])")
    
    # ============================================
    # Test 4: Check scaler consistency
    # ============================================
    logger.info("Test 4: Checking scaler min/max values")
    scaler = predictor.scaler
    
    if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
        data_min = scaler.data_min_[0]
        data_max = scaler.data_max_[0]
        logger.info(f"✅ Scaler range: [{data_min:.2f}, {data_max:.2f}] m³")
        
        if data_min < 0:
            warnings.append(f"⚠️  Scaler data_min is negative: {data_min:.2f}")
        if data_max > 100000:
            warnings.append(f"⚠️  Scaler data_max unusually high: {data_max:.2f}")
    else:
        warnings.append("⚠️  Could not verify scaler parameters")
    
    # ============================================
    # Test 5: Check prediction consistency
    # ============================================
    logger.info("Test 5: Checking prediction consistency (same input = same output)")
    test_input = [8000.0, 8100.0, 8200.0, 8300.0, 8400.0, 8500.0, 8600.0, 8700.0]
    pred1 = predictor.predict(test_input)
    pred2 = predictor.predict(test_input)
    
    if abs(pred1 - pred2) > 0.01:
        issues.append(f"❌ CRITICAL: Predictions not deterministic: {pred1:.2f} vs {pred2:.2f}")
    else:
        logger.info(f"✅ Predictions are deterministic: {pred1:.2f} m³")
    
    # ============================================
    # Test 6: Check trend following
    # ============================================
    logger.info("Test 6: Checking if model follows input trends")
    increasing = [5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0]
    decreasing = [12000.0, 11000.0, 10000.0, 9000.0, 8000.0, 7000.0, 6000.0, 5000.0]
    
    pred_inc = predictor.predict(increasing)
    pred_dec = predictor.predict(decreasing)
    
    avg_inc = sum(increasing) / len(increasing)
    avg_dec = sum(decreasing) / len(decreasing)
    
    # For increasing trend, prediction should be >= average
    # For decreasing trend, prediction should be <= average
    if pred_inc < avg_inc * 0.8:
        warnings.append(f"⚠️  Increasing trend prediction too low: {pred_inc:.2f} (avg: {avg_inc:.2f})")
    if pred_dec > avg_dec * 1.2:
        warnings.append(f"⚠️  Decreasing trend prediction too high: {pred_dec:.2f} (avg: {avg_dec:.2f})")
    
    logger.info(f"✅ Increasing trend → {pred_inc:.2f} m³, Decreasing trend → {pred_dec:.2f} m³")
    
    # ============================================
    # Print Summary Report
    # ============================================
    print("\n" + "="*70)
    print("ST53 MODEL VALIDATION REPORT")
    print("="*70)
    
    if not issues and not warnings:
        print("\n✅ ALL VALIDATION CHECKS PASSED!")
        print("\nModel is ready for deployment.")
        logger.info("Validation completed successfully - all checks passed")
        return True
    
    if warnings:
        print(f"\n⚠️  {len(warnings)} WARNING(S) FOUND:\n")
        for warning in warnings:
            print(warning)
    
    if issues:
        print(f"\n🚨 {len(issues)} CRITICAL ISSUE(S) FOUND:\n")
        for issue in issues:
            print(issue)
        print("\n❌ MODEL VALIDATION FAILED!")
        print("Fix these issues before deploying to production.")
        logger.error(f"Validation failed with {len(issues)} critical issues")
        return False
    
    print("\n⚠️  MODEL HAS WARNINGS - Review before deployment")
    logger.warning(f"Validation completed with {len(warnings)} warnings")
    return True


if __name__ == "__main__":
    success = validate_model()
    sys.exit(0 if success else 1)
