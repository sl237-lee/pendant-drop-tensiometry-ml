"""
Predict surface tension from droplet image
"""
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path

from src.preprocessing.edge_detection import DropletImageProcessor


def prepare_image_data(r_vals, z_vals, n_points=226):
    """Prepare image coordinates for model input"""
    coords = np.column_stack([r_vals, z_vals])
    
    if len(coords) < n_points:
        padding = np.zeros((n_points - len(coords), 2))
        coords = np.vstack([coords, padding])
    else:
        coords = coords[:n_points, :]
    
    return coords.flatten().reshape(1, -1)


def predict_surface_tension_from_image(image_path, pixel_to_mm=1.0, 
                                       capillary_diameter_mm=2.7,
                                       density_diff=1000.0):
    """
    Complete pipeline: Image → Surface Tension
    """
    
    print("="*70)
    print("PENDANT DROP SURFACE TENSION PREDICTION")
    print("="*70)
    
    # Step 1: Load model
    print("\n1. Loading trained model...")
    model = keras.models.load_model('models/pendant_drop_model_best.h5', compile=False)
    print("   ✅ Model loaded")
    
    # Step 2: Process image
    print(f"\n2. Processing image: {image_path}")
    processor = DropletImageProcessor()
    
    try:
        r_pixels, z_pixels, contour, img_prep = processor.process_image(image_path)
        print(f"   ✅ Extracted {len(r_pixels)} points from droplet edge")
    except Exception as e:
        print(f"   ❌ Error processing image: {e}")
        return None, None
    
    # Step 3: Convert to physical units and normalize
    print("\n3. Converting to dimensionless coordinates...")
    r_mm = r_pixels * pixel_to_mm
    z_mm = z_pixels * pixel_to_mm
    
    a = capillary_diameter_mm
    r_norm = r_mm / a
    z_norm = z_mm / a
    
    X_input = prepare_image_data(r_norm, z_norm)
    
    print(f"   ✅ Normalized to dimensionless coordinates")
    
    # Step 4: Predict
    print("\n4. Predicting surface tension...")
    prediction = model.predict(X_input, verbose=0)
    Bo_pred = prediction[0, 0]
    pL_pred = prediction[0, 1]
    
    print(f"   Predicted Bo (Δρ̃): {Bo_pred:.4f}")
    print(f"   Predicted p̃_L: {pL_pred:.4f}")
    
    # Step 5: Convert to physical surface tension
    print("\n5. Converting to physical units...")
    g = 9.81
    a_m = a / 1000
    
    gamma = (density_diff * g * a_m**2) / Bo_pred
    gamma_mN = gamma * 1000
    
    print(f"   Surface tension: {gamma_mN:.2f} mN/m")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_prep, cmap='gray')
    axes[0].plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2)
    axes[0].set_title('Detected Droplet Edge')
    axes[0].axis('off')
    
    axes[1].plot(r_norm, z_norm, 'b-', linewidth=2)
    axes[1].plot(-r_norm, z_norm, 'b-', linewidth=2)
    axes[1].set_xlabel('r (dimensionless)')
    axes[1].set_ylabel('z (dimensionless)')
    axes[1].set_title('Extracted Shape')
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].axis('off')
    results_text = f"""
    RESULTS
    
    Predicted Parameters:
      Bo (Δρ̃): {Bo_pred:.4f}
      p̃_L: {pL_pred:.4f}
    
    Physical Properties:
      Surface Tension: {gamma_mN:.2f} mN/m
      Capillary diameter: {capillary_diameter_mm:.2f} mm
      Density difference: {density_diff:.0f} kg/m³
    
    (Assuming water in air at 20°C,
     true γ ≈ 72 mN/m)
    """
    axes[2].text(0.1, 0.5, results_text, fontsize=11, 
                 verticalalignment='center', family='monospace')
    axes[2].set_title('Prediction Results', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('results/image_prediction.png', dpi=150, bbox_inches='tight')
    print("\n   ✅ Saved visualization to: results/image_prediction.png")
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETE!")
    print("="*70)
    
    results = {
        'Bo': Bo_pred,
        'pL': pL_pred,
        'surface_tension_mN_m': gamma_mN,
        'r_norm': r_norm,
        'z_norm': z_norm
    }
    
    return gamma_mN, results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str)
    parser.add_argument('--pixel_to_mm', type=float, default=0.05)
    parser.add_argument('--capillary_mm', type=float, default=2.7)
    parser.add_argument('--density', type=float, default=1000.0)
    
    args = parser.parse_args()
    
    gamma, results = predict_surface_tension_from_image(
        args.image,
        pixel_to_mm=args.pixel_to_mm,
        capillary_diameter_mm=args.capillary_mm,
        density_diff=args.density
    )
    
    if gamma is not None:
        print(f"\n🎯 Final Answer: Surface Tension = {gamma:.2f} mN/m")
