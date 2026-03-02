import sys
print("Python:", sys.version)

print("Loading numpy...")
import numpy as np
print("✅ NumPy OK")

print("Loading TensorFlow...")
import tensorflow as tf
print("✅ TensorFlow OK")

print("Loading Keras...")
from tensorflow import keras
print("✅ Keras OK")

print("Loading model...")
model = keras.models.load_model('models/pendant_drop_model_best.h5', compile=False)
print("✅ Model loaded!")

print("\nAll systems working!")
