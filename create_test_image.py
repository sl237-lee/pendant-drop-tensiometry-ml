"""Create synthetic droplet image for testing"""
import sys
sys.path.append('.')

import numpy as np
import cv2
from src.physics.young_laplace import PendantDropSolver

print("Creating synthetic droplet image...")

# Generate a droplet
solver = PendantDropSolver(Bo=0.3, pL_tilde=2.0)
shape = solver.solve()

# Create image (512x512)
img_size = (512, 512)
img = np.zeros(img_size, dtype=np.uint8)

# Scale coordinates to image
r_vals = shape['r']
z_vals = shape['z']

# Scale to fit image (with some margin)
scale = 150
r_pixels = (r_vals * scale + img_size[1] // 2).astype(int)
z_pixels = (z_vals * scale + 100).astype(int)

# Draw droplet (both sides)
for i in range(len(r_pixels) - 1):
    cv2.line(img, (r_pixels[i], z_pixels[i]), 
             (r_pixels[i+1], z_pixels[i+1]), 255, 2)
    cv2.line(img, (img_size[1] - r_pixels[i], z_pixels[i]), 
             (img_size[1] - r_pixels[i+1], z_pixels[i+1]), 255, 2)

# Fill droplet
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.fillPoly(img, contours, 255)

# Add some noise to make it realistic
noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
img_noisy = cv2.add(img, noise)

# Save
cv2.imwrite('data/test_droplet_image.png', img_noisy)
print("✅ Created test image: data/test_droplet_image.png")
print(f"   True Bo: {shape['Bo']}")
print(f"   True pL: {shape['pL_tilde']}")
print("\nNow test with: python predict_from_image.py data/test_droplet_image.png")
