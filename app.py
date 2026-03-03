import gradio as gr
import sys
sys.path.append('.')

from tensorflow import keras
from src.preprocessing.edge_detection import DropletImageProcessor
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import cv2

# Load model
print("Loading model...")
model = keras.models.load_model('models/pendant_drop_model_best.h5', compile=False)
processor = DropletImageProcessor()
print("✅ Model loaded!")

def predict_surface_tension(image, pixel_to_mm, capillary_mm, density):
    """Predict surface tension from uploaded image"""
    try:
        # Save uploaded image temporarily
        temp_path = "temp_upload.png"
        image.save(temp_path)
        
        # Process image
        r, z, contour, img_prep = processor.process_image(temp_path, pixel_to_mm)
        
        if r is None or len(r) == 0:
            return "❌ **Error:** Could not detect droplet edge. Please check image quality.", None
        
        # Normalize coordinates
        a = capillary_mm
        r_norm = (r * pixel_to_mm) / a
        z_norm = (z * pixel_to_mm) / a
        
        # Prepare for model
        coords = np.column_stack([r_norm, z_norm])
        if len(coords) < 226:
            padding = np.zeros((226 - len(coords), 2))
            coords = np.vstack([coords, padding])
        else:
            coords = coords[:226, :]
        X = coords.flatten().reshape(1, -1)
        
        # Predict
        pred = model.predict(X, verbose=0)
        Bo = pred[0, 0]
        pL = pred[0, 1]
        
        # Convert to surface tension
        g = 9.81
        gamma = (density * g * (capillary_mm/1000)**2) / Bo
        gamma_mN = gamma * 1000
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Original with contour
        ax1.imshow(img_prep, cmap='gray')
        if contour is not None and len(contour) > 0:
            # Fix: Handle contour shape properly
            if contour.ndim == 3:
                contour = contour.reshape(-1, 2)
            ax1.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, label='Detected Edge')
        ax1.set_title('Detected Droplet Edge', fontsize=14, fontweight='bold')
        ax1.axis('off')
        ax1.legend()
        
        # Plot 2: Extracted shape
        ax2.plot(r_norm, z_norm, 'b-', linewidth=2, label='Right side')
        ax2.plot(-r_norm, z_norm, 'b--', linewidth=2, label='Left side (mirrored)')
        ax2.set_xlabel('r (dimensionless)', fontsize=12)
        ax2.set_ylabel('z (dimensionless)', fontsize=12)
        ax2.set_title('Extracted Normalized Shape', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        
        # Format results
        results_text = f"""
# Prediction Results

## Predicted Parameters:
- **Bond Number (Bo):** {Bo:.4f}
- **Apex Pressure (p̃_L):** {pL:.4f}

## Physical Properties:
- **Surface Tension:** **{gamma_mN:.2f} mN/m** 
- **Capillary Diameter:** {capillary_mm:.2f} mm
- **Density Difference:** {density:.0f} kg/m³

## Processing Info:
- **Edge Points Extracted:** {len(r)} points
- **Calibration:** {pixel_to_mm:.4f} mm/pixel

---
### Reference Values:
- Water at 20°C: γ ≈ 72 mN/m
- Ethanol at 20°C: γ ≈ 22 mN/m
- Mercury at 20°C: γ ≈ 486 mN/m
        """
        
        return results_text, result_image
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"""
❌ **Error Processing Image**

**Error message:** {str(e)}

**Please check:**
- Image shows a clear droplet
- Good contrast between droplet and background
- Calibration values are reasonable
- Image format is PNG or JPG

<details>
<summary>Technical Details (click to expand)</summary>
```
{error_details}
```
</details>
        """
        return error_msg, None

# Create Gradio interface
with gr.Blocks(title="Neural Tensiometry") as demo:
    gr.Markdown("""
    # Neural Tensiometry: AI-Powered Surface Tension Measurement
    
    Upload a droplet image to measure surface tension in **<1 second** using deep learning!
    
    ### Image Requirements:
    - Clear droplet photo (PNG or JPG)
    - Good contrast between droplet and background
    - Droplet should be clearly visible
    
    ### Calibration:
    - `pixel_to_mm`: Measure something of known size in your image to calculate this
    - `capillary_mm`: Diameter of the capillary tube in millimeters
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="�� Upload Droplet Image")
            
            with gr.Accordion("⚙️ Calibration Settings", open=True):
                pixel_to_mm = gr.Number(
                    value=0.05, 
                    label="Pixel to mm ratio",
                    info="How many mm per pixel? (measure something with known size)"
                )
                capillary_mm = gr.Number(
                    value=2.7, 
                    label="Capillary diameter (mm)",
                    info="Diameter of the capillary tube"
                )
                density = gr.Number(
                    value=1000, 
                    label="Density difference (kg/m³)",
                    info="For water in air: 1000 kg/m³"
                )
            
            predict_btn = gr.Button("Predict Surface Tension", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 💡 Quick Tips:
            - Default values work for most water droplets
            - Adjust `pixel_to_mm` if results seem off
            - Try the test image: `data/test_droplet_image.png`
            """)
        
        with gr.Column(scale=1):
            results_output = gr.Markdown(label="Results")
            image_output = gr.Image(label="📊 Visualization")
    
    gr.Markdown("""
    ---
    ### Example Fluids:
    | Fluid | Density (kg/m³) | Expected γ (mN/m) |
    |-------|----------------|-------------------|
    | Water | 1000 | ~72 |
    | Ethanol | 789 | ~22 |
    | Glycerol | 1260 | ~63 |
    | Mercury | 13,600 | ~486 |
    
    ### Links:
    - [GitHub Repository](https://github.com/sl237-lee/pendant-drop-tensiometry-ml)
    - [Documentation](https://github.com/sl237-lee/pendant-drop-tensiometry-ml/blob/main/README.md)
    - [Project Summary](https://github.com/sl237-lee/pendant-drop-tensiometry-ml/blob/main/PROJECT_SUMMARY.md)
    
    ---
    **Built by Seungryul Andrew Lee | Machine Learning in Physics**
    """)
    
    predict_btn.click(
        fn=predict_surface_tension,
        inputs=[image_input, pixel_to_mm, capillary_mm, density],
        outputs=[results_output, image_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
