"""Test the trained model on new droplets"""
import sys
sys.path.append('.')

import numpy as np
from tensorflow import keras
from src.physics.young_laplace import PendantDropSolver
from src.models.data_preparation import prepare_training_data

print("="*70)
print("TESTING TRAINED MODEL")
print("="*70)

# Load the best model (without compilation to avoid error)
print("\n1. Loading trained model...")
model = keras.models.load_model('models/pendant_drop_model_best.h5', compile=False)
print("   ✅ Model loaded successfully")

# Test on 5 different droplets
test_cases = [
    {'Bo': 0.3, 'pL': 2.0, 'name': 'Homework droplet'},
    {'Bo': 0.5, 'pL': 3.0, 'name': 'Medium droplet'},
    {'Bo': 1.0, 'pL': 2.5, 'name': 'Large Bo'},
    {'Bo': 2.0, 'pL': 4.0, 'name': 'Very large'},
    {'Bo': 0.15, 'pL': 1.8, 'name': 'Small droplet'},
]

print("\n2. Testing on 5 new droplets...")
print("-"*70)

for i, test in enumerate(test_cases, 1):
    # Generate droplet
    solver = PendantDropSolver(Bo=test['Bo'], pL_tilde=test['pL'])
    shape = solver.solve()
    
    # Prepare for model
    X_test, y_true = prepare_training_data([shape])
    
    # Predict
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate errors
    error_Bo = abs(y_true[0, 0] - y_pred[0, 0])
    error_pL = abs(y_true[0, 1] - y_pred[0, 1])
    
    print(f"\nTest {i}: {test['name']}")
    print(f"  True:      Bo={y_true[0, 0]:.4f}, p̃_L={y_true[0, 1]:.4f}")
    print(f"  Predicted: Bo={y_pred[0, 0]:.4f}, p̃_L={y_pred[0, 1]:.4f}")
    print(f"  Error:     Bo={error_Bo:.6f}, p̃_L={error_pL:.6f}")
    
    if error_Bo < 0.1 and error_pL < 0.2:
        print(f"  ✅ Excellent prediction!")
    elif error_Bo < 0.2 and error_pL < 0.4:
        print(f"  ✅ Good prediction!")
    else:
        print(f"  ⚠️  Fair prediction")

print("\n" + "="*70)
print("MODEL TEST COMPLETE!")
print("="*70)
print("\nYour neural network can now predict surface tension from droplet shapes!")
print("This model is ready to be fine-tuned on real lab images when available.")
print("="*70)
