"""
Neural network architecture for pendant drop tensiometry
Based on Kratz & Kierfeld (2020) Figure 1
"""
import tensorflow as tf
from tensorflow import keras


class PendantDropNN:
    """Deep neural network for surface tension prediction"""
    
    def __init__(self, input_dim=452, learning_rate=1.0):
        """
        Args:
            input_dim: Input dimension (226 points × 2 coordinates = 452)
            learning_rate: Learning rate for Adadelta optimizer
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = None
    
    def build_model(self):
        """
        Build the neural network architecture from paper (Figure 1)
        
        Architecture:
            Input (452) → Dense(512) → Dense(1024) → Dense(256) → Dense(16) → Output(2)
        """
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=(self.input_dim,)),
            
            # Hidden layers with LeakyReLU activation
            keras.layers.Dense(512, activation=keras.layers.LeakyReLU(alpha=0.01)),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(1024, activation=keras.layers.LeakyReLU(alpha=0.01)),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(256, activation=keras.layers.LeakyReLU(alpha=0.01)),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(16, activation=keras.layers.LeakyReLU(alpha=0.01)),
            
            # Output layer: [Bo, pL_tilde]
            keras.layers.Dense(2, activation='linear')
        ])
        
        # Compile with Adadelta optimizer (from paper)
        model.compile(
            optimizer=keras.optimizers.Adadelta(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def summary(self):
        """Print model architecture"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
