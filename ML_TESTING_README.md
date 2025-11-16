# ML Testing & Validation Tools

This directory contains tools for validating, testing, and analyzing the ST53 LSTM forecasting model.

## 📁 Files Created

### 1. **src/st53/validate_model.py** - Automated Validation Pipeline
Runs comprehensive checks after model training to catch issues before deployment.

**Usage:**
```bash
# After training, run validation
python -m src.st53.validate_model
```

**What it checks:**
- ✅ No negative predictions
- ✅ Output layer has sigmoid activation
- ✅ Model outputs in [0,1] range
- ✅ Scaler consistency
- ✅ Prediction determinism
- ✅ Trend following behavior

**Example output:**
```
======================================================================
ST53 MODEL VALIDATION REPORT
======================================================================

✅ ALL VALIDATION CHECKS PASSED!

Model is ready for deployment.
```

---

### 2. **tests/test_st53_model.py** - Unit Tests with Pytest
Comprehensive test suite for automated testing.

**Setup:**
```bash
pip install pytest
```

**Usage:**
```bash
# Run all tests
pytest tests/test_st53_model.py -v

# Run specific test class
pytest tests/test_st53_model.py::TestST53Predictor -v

# Run with coverage
pytest tests/test_st53_model.py --cov=src.st53 -v
```

**Test coverage:**
- Model predictions (negative values, range checks)
- Architecture validation (sigmoid activation)
- Scaler functionality
- Real Cenovus data tests
- Input validation

**Example output:**
```
tests/test_st53_model.py::TestST53Predictor::test_no_negative_predictions PASSED
tests/test_st53_model.py::TestST53Predictor::test_predictions_in_reasonable_range PASSED
tests/test_st53_model.py::TestST53ModelArchitecture::test_model_output_activation_is_sigmoid PASSED

========================= 15 passed in 3.42s ==========================
```

---

### 3. **analyze_with_llm.py** - AI-Powered Analysis
Uses local LLM (Ollama/LM Studio) to intelligently analyze model predictions.

**Setup Ollama:**
```bash
# 1. Install Ollama
# Download from: https://ollama.ai/

# 2. Pull a model
ollama pull llama3.2

# 3. Server should auto-start, or run:
ollama serve
```

**Usage:**
```bash
python analyze_with_llm.py
```

**What it does:**
1. Runs 6 test scenarios (real Cenovus data + synthetic patterns)
2. Sends results to local LLM
3. Gets intelligent analysis and recommendations
4. Saves report to `model_analysis_report.json`

**Example output:**
```
🤖 Sending test results to Ollama LLM for analysis...
   Using model: llama3.2

🤖 LLM ANALYSIS:

Based on the test results, here are my findings:

1. CRITICAL ISSUES FOUND:
   - Foster Creek prediction is NEGATIVE (-763.34 m³)
   - This indicates missing sigmoid activation in output layer
   
2. TREND FOLLOWING:
   - Model correctly predicts decline for decreasing trend
   - Handles volatile data reasonably well
   
3. RECOMMENDATIONS:
   - Add activation='sigmoid' to final Dense layer
   - Retrain model with constrained output
   - Implement max(0, prediction) clipping as safety net

4. DEPLOYMENT READINESS: NOT READY
   - Fix negative prediction bug before production
```

---

### 4. **track_experiments.py** - MLflow Experiment Tracking
Track all training experiments with parameters, metrics, and artifacts.

**Setup:**
```bash
pip install mlflow
```

**Usage:**
```bash
# 1. Start MLflow UI (in separate terminal)
mlflow ui

# 2. Open browser
http://localhost:5000

# 3. Integration example provided in file
python track_experiments.py  # Shows how to integrate
```

**Benefits:**
- 📊 Compare multiple training runs
- 📈 Visualize loss curves over epochs
- 🔍 Filter by hyperparameters
- 💾 Version models with metadata
- 🏷️ Tag production-ready models

**Dashboard features:**
- See all experiments in one place
- Compare hyperparameters side-by-side
- Download best performing models
- Track data versions

---

## 🚀 Recommended Workflow

### After Every Training:
```bash
# 1. Train model
python -m src.st53.train_st53 data/st53/ST53_2024-12.xls models/

# 2. Validate automatically
python -m src.st53.validate_model

# 3. Run unit tests
pytest tests/test_st53_model.py -v
```

### Before Production Deployment:
```bash
# 1. Run all checks
python -m src.st53.validate_model
pytest tests/test_st53_model.py -v

# 2. Get AI analysis
python analyze_with_llm.py

# 3. Review MLflow experiments
mlflow ui  # Compare with previous versions
```

### During Development:
```bash
# Quick test cycle
pytest tests/test_st53_model.py -v --tb=short

# Watch mode (re-run on file changes)
pytest tests/test_st53_model.py -v --looponfail
```

---

## 📊 Continuous Integration (Optional)

If you want to add GitHub Actions (future):

1. Create `.github/workflows/ml-tests.yml`
2. Configure to run on every push
3. Automatically runs validation + pytest
4. Prevents merging broken models

---

## 🎯 Summary

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `validate_model.py` | Automated checks | After every training |
| `test_st53_model.py` | Unit testing | Before commits/deployment |
| `analyze_with_llm.py` | AI-powered analysis | When investigating issues |
| `track_experiments.py` | Experiment tracking | During hyperparameter tuning |

---

## 💡 Best Practices

1. **Always run validation after training**
   ```bash
   python -m src.st53.train_st53 ... && python -m src.st53.validate_model
   ```

2. **Run tests before committing code**
   ```bash
   pytest tests/ -v
   ```

3. **Use LLM analysis when predictions seem wrong**
   ```bash
   python analyze_with_llm.py
   ```

4. **Track experiments when tuning hyperparameters**
   - Integrate MLflow into training script
   - Compare runs visually

5. **Automate everything**
   - Add validation to training scripts
   - Set up pre-commit hooks for tests
   - Use CI/CD for automatic testing

---

## 🐛 Troubleshooting

### Ollama connection error:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Pytest not found:
```bash
pip install pytest
```

### MLflow UI not starting:
```bash
# Check if port 5000 is available
mlflow ui --port 5001
```

---

## 📚 Learn More

- [Pytest Documentation](https://docs.pytest.org/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Ollama Documentation](https://github.com/ollama/ollama)

---

**Created for interview preparation and production ML workflows! 🚀**
