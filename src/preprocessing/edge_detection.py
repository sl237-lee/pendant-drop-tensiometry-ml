"""
Image preprocessing for pendant drop images
Improved for real lab images with complex backgrounds
"""
import cv2
import numpy as np
from pathlib import Path


class DropletImageProcessor:
    """Process droplet images to extract coordinates"""

    def __init__(self):
        self.canny_low = 30
        self.canny_high = 100

    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Bilateral filter: smooths interior glare while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)

        return enhanced

    def detect_edges(self, img):
        """Detect edges using Canny with auto-threshold"""
        # Use Otsu threshold to auto-pick good Canny thresholds
        otsu_val, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low = otsu_val * 0.5
        high = otsu_val

        edges = cv2.Canny(img, low, high)

        # Close small gaps in the edge
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        return edges

    def extract_contour(self, edges, img_shape):
        """
        Extract the droplet contour.
        Filters out contours that are too small, too close to image border,
        or that don't look like a pendant drop (tall and narrow).
        """
        h, w = img_shape[:2]
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            raise ValueError("No contours found in image!")

        # Score each contour: prefer large, not touching border, roughly droplet-shaped
        best_contour = None
        best_score = -1

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Skip tiny contours (noise)
            if area < 500:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)

            # Skip contours touching the image border (likely background/capillary)
            margin = 5
            if x <= margin or y <= margin or (x + cw) >= (w - margin) or (y + ch) >= (h - margin):
                continue

            # Prefer contours in the lower 2/3 of the image (pendant drop hangs down)
            center_y = y + ch / 2
            vertical_position_score = center_y / h  # higher = lower in image = better

            # Prefer taller contours (pendant drops are tall relative to width)
            aspect_score = ch / max(cw, 1)

            # Combined score weighted by area
            score = area * vertical_position_score * min(aspect_score, 2.0)

            if score > best_score:
                best_score = score
                best_contour = cnt

        # Fallback: if no contour passed filters, just use the largest
        if best_contour is None:
            best_contour = max(contours, key=cv2.contourArea)

        return best_contour

    def contour_to_coordinates(self, contour, pixel_to_mm=1.0):
        """
        Convert contour to (r, z) coordinates.
        Apex is at z=0, z increases downward (physical pendant drop convention).
        """
        points = contour.reshape(-1, 2)

        # Find apex: topmost point (minimum y in image coordinates)
        apex_idx = np.argmin(points[:, 1])
        apex = points[apex_idx]

        # Use right side of droplet only (r >= 0)
        right_side = points[points[:, 0] >= apex[0]]

        if len(right_side) < 10:
            # Fallback: use all points if right side filtering is too aggressive
            right_side = points

        # Convert to physical coordinates centered at apex
        r_vals = (right_side[:, 0] - apex[0]) * pixel_to_mm
        z_vals = (right_side[:, 1] - apex[1]) * pixel_to_mm  # positive = downward

        # Sort by z (top to bottom)
        sort_idx = np.argsort(z_vals)
        r_vals = r_vals[sort_idx]
        z_vals = z_vals[sort_idx]

        # Remove negative z points (above apex, part of capillary)
        mask = z_vals >= 0
        r_vals = r_vals[mask]
        z_vals = z_vals[mask]

        # Remove outlier r values (noise spikes)
        if len(r_vals) > 10:
            r_median = np.median(r_vals)
            r_std = np.std(r_vals)
            mask = np.abs(r_vals - r_median) < 3 * r_std
            r_vals = r_vals[mask]
            z_vals = z_vals[mask]

        return r_vals, z_vals

    def process_image(self, image_path, pixel_to_mm=1.0):
        """
        Complete pipeline: image → coordinates

        Args:
            image_path: Path to droplet image
            pixel_to_mm: Calibration factor (mm per pixel)

        Returns:
            r_vals, z_vals, contour, preprocessed_image
        """
        img_prep = self.preprocess_image(image_path)
        edges = self.detect_edges(img_prep)
        contour = self.extract_contour(edges, img_prep.shape)
        r, z = self.contour_to_coordinates(contour, pixel_to_mm)

        return r, z, contour, img_prep