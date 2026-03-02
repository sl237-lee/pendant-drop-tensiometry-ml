"""
Image preprocessing for pendant drop images
"""
import cv2
import numpy as np
from pathlib import Path


class DropletImageProcessor:
    """Process droplet images to extract coordinates"""
    
    def __init__(self):
        self.canny_low = 50
        self.canny_high = 150
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def detect_edges(self, img):
        """Detect edges using Canny"""
        edges = cv2.Canny(img, self.canny_low, self.canny_high)
        return edges
    
    def extract_contour(self, edges):
        """Extract largest contour (should be droplet)"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            raise ValueError("No contours found in image!")
        
        # Get largest contour
        droplet_contour = max(contours, key=cv2.contourArea)
        return droplet_contour
    
    def contour_to_coordinates(self, contour, pixel_to_mm=1.0):
        """
        Convert contour to (r, z) coordinates
        
        Args:
            contour: OpenCV contour
            pixel_to_mm: Calibration factor (pixels per mm)
        
        Returns:
            r_vals, z_vals: Coordinate arrays
        """
        points = contour.reshape(-1, 2)
        
        # Find apex (topmost point)
        apex_idx = np.argmin(points[:, 1])
        apex = points[apex_idx]
        
        # Use right side of droplet
        right_side = points[points[:, 0] >= apex[0]]
        
        # Convert to cylindrical coordinates centered at apex
        r_vals = (right_side[:, 0] - apex[0]) * pixel_to_mm
        z_vals = (right_side[:, 1] - apex[1]) * pixel_to_mm
        
        # Sort by z
        sort_idx = np.argsort(z_vals)
        r_vals = r_vals[sort_idx]
        z_vals = z_vals[sort_idx]
        
        return r_vals, z_vals
    
    def process_image(self, image_path, pixel_to_mm=1.0):
        """
        Complete pipeline: image → coordinates
        
        Args:
            image_path: Path to droplet image
            pixel_to_mm: Calibration factor
        
        Returns:
            r_vals, z_vals, contour, preprocessed_image
        """
        # Preprocess
        img_prep = self.preprocess_image(image_path)
        
        # Detect edges
        edges = self.detect_edges(img_prep)
        
        # Extract contour
        contour = self.extract_contour(edges)
        
        # Convert to coordinates
        r, z = self.contour_to_coordinates(contour, pixel_to_mm)
        
        return r, z, contour, img_prep
