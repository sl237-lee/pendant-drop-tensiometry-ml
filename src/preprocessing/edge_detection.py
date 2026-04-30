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
        self.default_top_crop_fraction = 0.20

    def preprocess_image(self, image_path, top_crop_fraction=None):
        """Load and preprocess image.

        A small top crop helps remove the capillary stem above the pendant drop.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        crop_fraction = self.default_top_crop_fraction if top_crop_fraction is None else top_crop_fraction
        crop_fraction = float(np.clip(crop_fraction, 0.0, 0.5))
        if crop_fraction > 0:
            crop_y = int(img.shape[0] * crop_fraction)
            img = img[crop_y:, :]

        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Auto-detect low contrast images and apply stronger enhancement
        contrast = gray.std()
        if contrast < 40:
            # Low contrast image (e.g. alcohol) — apply stronger CLAHE first
            clahe_strong = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
            gray = clahe_strong.apply(gray)

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
            vertical_position_score = center_y / h

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

    def contour_to_coordinates(self, contour, pixel_to_mm=1.0, n_points=226):
        """
        Convert contour to (r, z) coordinates.
        Apex is at z=0, z increases downward (physical pendant drop convention).
        """
        points = contour.reshape(-1, 2)

        # Reconstruct a single-valued radial profile by taking the left/right
        # boundary at each image row. This is much more stable than sampling an
        # arbitrary side of the contour.
        row_map = {}
        for x, y in points:
            row_map.setdefault(int(y), []).append(int(x))

        rows = []
        for y in sorted(row_map):
            xs = np.asarray(row_map[y])
            if xs.size < 2:
                continue
            rows.append((y, xs.min(), xs.max()))

        if len(rows) < 5:
            # Fallback to the previous right-side extraction if the contour is sparse.
            M = cv2.moments(contour)
            if M['m00'] != 0:
                center_x = int(M['m10'] / M['m00'])
            else:
                center_x = int(np.mean(points[:, 0]))
            right_side = points[points[:, 0] >= center_x]
            if len(right_side) < 10:
                right_side = points
            apex_y = int(np.min(right_side[:, 1]))
            r_vals = (right_side[:, 0] - center_x) * pixel_to_mm
            z_vals = (right_side[:, 1] - apex_y) * pixel_to_mm
            sort_idx = np.argsort(z_vals)
            return r_vals[sort_idx], z_vals[sort_idx]

        rows = np.asarray(rows, dtype=float)
        z_pix = rows[:, 0]
        r_pix = (rows[:, 2] - rows[:, 1]) / 2.0

        # Smooth the radial profile to suppress edge speckle and glare.
        if len(r_pix) >= 5:
            kernel = np.ones(5, dtype=float) / 5.0
            r_pix = np.convolve(r_pix, kernel, mode='same')

        # Keep only the lower pendant profile, then sample it on a fixed grid
        # so the model sees a consistent input length and ordering.
        apex_y = z_pix[0]
        z_vals = (z_pix - apex_y) * pixel_to_mm
        r_vals = r_pix * pixel_to_mm

        valid = z_vals >= 0
        r_vals = r_vals[valid]
        z_vals = z_vals[valid]

        if len(r_vals) < 2:
            return r_vals, z_vals

        # Resample onto a uniform z grid to better match the model's training
        # distribution from synthetic ODE solutions.
        z_grid = np.linspace(float(z_vals.min()), float(z_vals.max()), n_points)
        r_grid = np.interp(z_grid, z_vals, r_vals)

        return r_grid, z_grid

    def process_image(self, image_path, pixel_to_mm=1.0, top_crop_fraction=None):
        """
        Complete pipeline: image -> coordinates

        Args:
            image_path: Path to droplet image
            pixel_to_mm: Calibration factor (mm per pixel)
            top_crop_fraction: Fraction of the image height to crop from the top

        Returns:
            r_vals, z_vals, contour, preprocessed_image
        """
        img_prep = self.preprocess_image(image_path, top_crop_fraction=top_crop_fraction)
        edges = self.detect_edges(img_prep)
        contour = self.extract_contour(edges, img_prep.shape)
        r, z = self.contour_to_coordinates(contour, pixel_to_mm)

        return r, z, contour, img_prep
