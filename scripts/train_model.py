"""
Train neural network on synthetic data

Usage:
    python scripts/train_model.py --epochs 100 --batch_size 100
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.file_io import load_dataset
from src.models.architecture import PendantDropNN
from src.models.data_preparation import prepare_training_data
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description='Train pendant drop neural network')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='pendant_drop_model', help='Model save name')
    args = parser.parse_args()
    
    print("="*70)
    print("PENDANT DROP NEURAL NETWORK TRAINING")
    print("="*70)
    
    # Load datasets
    print("\n1. Loading datasets...")
    train_dataset = load_dataset('data/synthetic/training/class2_10000.pkl')
    val_dataset = load_dataset('data/synthetic/validation/class2_2000.pkl')
    test_dataset = load_dataset('data/synthetic/test/class2_1000.pkl')
    
    # Prepare data
    print("\n2. Preparing data for neural network...")
    X_train, y_train = prepare_training_data(train_dataset)
    X_val, y_val = prepare_training_data(val_dataset)
    X_test, y_test = prepare_training_data(test_dataset)
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Input dimension: {X_train.shape[1]}")
    
    # Build model
    print("\n3. Building neural network...")
    nn = PendantDropNN(input_dim=X_train.shape[1], learning_rate=args.learning_rate)
    model = nn.build_model()
    print(model.summary())
    
    # Callbacks
    print("\n4. Setting up training callbacks...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'models/{args.model_name}_best.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    # Train
    print("\n5. Training model...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print("-"*70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss (MSE): {test_loss:.6e}")
    print(f"   Test MAE: {test_mae:.6e}")
    
    # Make predictions
    print("\n7. Making predictions on test set...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate errors
    errors_Bo = np.abs(y_pred[:, 0] - y_test[:, 0])
    errors_pL = np.abs(y_pred[:, 1] - y_test[:, 1])
    
    print(f"\n   Bo predictions:")
    print(f"      Mean Absolute Error: {np.mean(errors_Bo):.6f}")
    print(f"      Std: {np.std(errors_Bo):.6f}")
    print(f"      Max Error: {np.max(errors_Bo):.6f}")
    
    print(f"\n   pL predictions:")
    print(f"      Mean Absolute Error: {np.mean(errors_pL):.6f}")
    print(f"      Std: {np.std(errors_pL):.6f}")
    print(f"      Max Error: {np.max(errors_pL):.6f}")
    
    # Save final model
    print(f"\n8. Saving model to models/{args.model_name}_final.h5...")
    model.save(f'models/{args.model_name}_final.h5')
    
    # Plot training history
    print("\n9. Creating training history plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_yscale('log')
    axes[0].set_title('Training History - Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[1].set_title('Training History - MAE', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
    print("   Saved to: results/training_history.png")
    
    # Plot prediction accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bo predictions
    axes[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5, s=10)
    axes[0].plot([y_test[:, 0].min(), y_test[:, 0].max()], 
                 [y_test[:, 0].min(), y_test[:, 0].max()], 
                 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('True Bo', fontsize=12)
    axes[0].set_ylabel('Predicted Bo', fontsize=12)
    axes[0].set_title('Bo Prediction Accuracy', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # pL predictions
    axes[1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5, s=10)
    axes[1].plot([y_test[:, 1].min(), y_test[:, 1].max()], 
                 [y_test[:, 1].min(), y_test[:, 1].max()], 
                 'r--', linewidth=2, label='Perfect prediction')
    axes[1].set_xlabel('True p̃_L', fontsize=12)
    axes[1].set_ylabel('Predicted p̃_L', fontsize=12)
    axes[1].set_title('p̃_L Prediction Accuracy', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/prediction_accuracy.png', dpi=150, bbox_inches='tight')
    print("   Saved to: results/prediction_accuracy.png")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"✅ Model saved: models/{args.model_name}_final.h5")
    print(f"✅ Best model: models/{args.model_name}_best.h5")
    print(f"✅ Training plots: results/training_history.png")
    print(f"✅ Accuracy plots: results/prediction_accuracy.png")
    print("="*70)


if __name__ == '__main__':
    main()
