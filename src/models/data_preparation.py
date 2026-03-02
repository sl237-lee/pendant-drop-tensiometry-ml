"""
Prepare data for neural network training
"""
import numpy as np
from pathlib import Path


def prepare_training_data(dataset, n_points=226):
    """
    Convert dataset to model input format
    
    Args:
        dataset: List of shape dictionaries
        n_points: Number of points to use (226 from paper)
    
    Returns:
        X: Input array (N, 452) - flattened (r, z) coordinates
        y: Output array (N, 2) - [Bo, pL_tilde]
    """
    X = []
    y = []
    
    for shape in dataset:
        # Get coordinates
        r = shape['r']
        z = shape['z']
        
        # Create (r, z) pairs
        coords = np.column_stack([r, z])
        
        # Pad or truncate to n_points
        if len(coords) < n_points:
            # Pad with zeros
            padding = np.zeros((n_points - len(coords), 2))
            coords = np.vstack([coords, padding])
        else:
            # Truncate
            coords = coords[:n_points, :]
        
        # Flatten to 1D vector (452 elements)
        X.append(coords.flatten())
        
        # Output labels
        y.append([shape['Bo'], shape['pL_tilde']])
    
    return np.array(X), np.array(y)


def normalize_data(X_train, X_val=None, X_test=None):
    """
    Normalize input data (optional but can help training)
    
    Args:
        X_train, X_val, X_test: Input arrays
    
    Returns:
        Normalized arrays and normalization parameters
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8  # Avoid division by zero
    
    X_train_norm = (X_train - mean) / std
    
    if X_val is not None:
        X_val_norm = (X_val - mean) / std
    else:
        X_val_norm = None
    
    if X_test is not None:
        X_test_norm = (X_test - mean) / std
    else:
        X_test_norm = None
    
    return X_train_norm, X_val_norm, X_test_norm, mean, std
