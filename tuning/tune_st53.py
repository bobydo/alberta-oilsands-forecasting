"""Hyperparameter tuning script for ST53 model."""
import numpy as np
from src.st53.preprocess_st53 import ST53DataProcessor
from src.st53.model_st53 import ST53Model
from src.common.evaluate import ModelEvaluator

if __name__ == "__main__":
    # Load data
    df = ST53DataProcessor.load("data/st53/ST53_2024-12.xls")
    values = np.array(df["Bitumen"].astype(float).values)
    
    # Test hyperparameter combinations: 
    # window_sizes (months of history to predict next month)

    # epochs_list 
    # Think of studying for an exam:
    # 1 epoch = Reading the entire textbook once
    # Epoch 1: First time reading → learn basic concepts
    # Epoch 2: Second time reading → remember better, catch things you missed
    # Epoch 40: Read 40 times → know it extremely well
    # Overfitting 200 : Memorized exact textbook examples, can't handle variations

    # batch_sizes (samples per update)
    
    # Small batches (8):
    # ✓ More frequent updates = faster learning, better generalization
    # ✓ Less memory usage
    # ✗ Slower training (more computations)
    # ✗ Noisier updates (less stable)

    # Large batches (32+):
    # ✓ Faster training (fewer updates, more parallel processing)
    # ✓ Smoother, more stable updates
    # ✗ Needs more memory
    # ✗ Can get stuck in local minima
    # ✗ May overfit

    """ 
    ST53 dataset has ~450 samples (months × operators)
    test_split=0.1 (90/10) - More training data
    test_split=0.2 (80/20) - Standard balance ✓
    test_split=0.3 (70/30) - More rigorous validation    """
    results = ModelEvaluator.tune_hyperparameters(
        values=values,
        window_sizes=[6, 8, 12],
        epochs_list=[20, 40],
        batch_sizes=[8, 16],
        model_builder=ST53Model.build,
        test_split=0.2 #means 80% training, 20% validation
    )
