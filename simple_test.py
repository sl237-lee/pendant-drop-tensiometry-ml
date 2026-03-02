"""Simple test of trained model"""
import sys
sys.path.append('.')

print("Loading libraries...")
from tensorflow import keras
from src.physics.young_laplace import PendantDropSolver
from src.models.data_preparation import prepare_training_data

print("✅ Libraries loaded\n")

print("="*70)
print("TESTING TRAINED MODEL")
print("="*70)

# Load model
print("\n1. Loading model...")
model = keras.models.load_model('models/pendant_drop_model_best.h5', compile=False)
print("   ✅ Model loaded\n")

# Test one droplet
print("2. Testing one droplet (Bo=0.3, pL=2.0)...")
solver = PendantDropSolver(Bo=0.3, pL_tilde=2.0)
shape = solver.solve()

X_test, y_true = prepare_training_data([shape])
y_pred = model.predict(X_test, verbose=0)

print(f"\n   True:      Bo={y_true[0,0]:.4f}, p̃_L={y_true[0,1]:.4f}")
print(f"   Predicted: Bo={y_pred[0,0]:.4f}, p̃_L={y_pred[0,1]:.4f}")
print(f"   Error:     Bo={abs(y_true[0,0]-y_pred[0,0]):.6f}, p̃_L={abs(y_true[0,1]-y_pred[0,1]):.6f}")

if abs(y_true[0,0]-y_pred[0,0]) < 0.1:
    print("   ✅ Excellent prediction!")
else:
    print("   ⚠️ Fair prediction")

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)
