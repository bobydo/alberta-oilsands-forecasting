"""Generate visual diagram of ST53 LSTM model architecture."""

import os
import sys
from src.st53.model_st53 import ST53Model

# Add common Graphviz installation paths to PATH (Windows)
graphviz_paths = [
    r"C:\Program Files\Graphviz\bin",
    r"C:\Program Files (x86)\Graphviz\bin",
    r"C:\Graphviz\bin",
]

for path in graphviz_paths:
    if os.path.exists(path):
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        print(f"✓ Added Graphviz to PATH: {path}")
        break

# Build the model
model = ST53Model.build(window_size=8)

# Print text summary
print("\n" + "="*70)
print("ST53 LSTM Model Architecture Summary")
print("="*70 + "\n")
model.summary()

print("\n" + "="*70)
print("Model Configuration")
print("="*70)
print(f"Total Parameters: {model.count_params():,}")
print(f"Window Size: 8 months")
print(f"Input Shape: (batch_size, 8, 1)")
print(f"Output Shape: (batch_size, 1)")
print(f"Optimizer: Adam")
print(f"Loss Function: MSE (Mean Squared Error)")

# Generate PNG diagram
try:
    import keras
    
    os.makedirs("docs/images", exist_ok=True)
    
    keras.utils.plot_model(
        model,
        to_file='docs/images/st53_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        rankdir='TB',  # Top to Bottom
        expand_nested=True,
        dpi=150
    )
    print("\n✓ Architecture diagram saved to: docs/images/st53_architecture.png")
    
except Exception as e:
    print(f"\n✗ Failed to generate PNG diagram: {e}")
    print("\nTroubleshooting:")
    print("1. Verify Graphviz is installed from: https://graphviz.org/download/")
    print("2. Add Graphviz bin folder to Windows PATH")
    print("3. Restart VS Code/PowerShell after changing PATH")
    print(f"4. Test with: dot -V")
