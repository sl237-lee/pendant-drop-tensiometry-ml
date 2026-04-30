"""Filename-based presets for the lab droplet images.

These presets are meant for the specific lab images referenced in the
project discussion. They keep the app from relying on the wrong defaults
when an uploaded file already identifies the fluid/needle pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_PIXEL_TO_MM = 0.045


LAB_IMAGE_PRESETS: dict[str, dict[str, Any]] = {
    # The preset capillary values below are effective outer diameters used by
    # the app. The lab's raw needle measurements may be inner diameters.
    "alcohol_pink": {
        "pixel_to_mm": DEFAULT_PIXEL_TO_MM,
        "capillary_mm": 1.15,
        "density": 789.0,
        "label": "Pink alcohol",
    },
    "water": {
        "pixel_to_mm": DEFAULT_PIXEL_TO_MM,
        "capillary_mm": 2.10,
        "density": 1000.0,
        "label": "Water",
    },
    "pink_sugar": {
        "pixel_to_mm": DEFAULT_PIXEL_TO_MM,
        "capillary_mm": 1.65,
        "density": 1030.0,
        "label": "Pink sugar",
    },
    "salt_dark_green": {
        "pixel_to_mm": DEFAULT_PIXEL_TO_MM,
        "capillary_mm": 1.90,
        "density": 1030.0,
        "label": "Salt / dark green",
    },
}


def infer_lab_preset(image_path: str | Path) -> dict[str, Any] | None:
    """Return the matching preset for a known lab image filename, if any."""
    stem = Path(image_path).stem.lower()
    for key, preset in LAB_IMAGE_PRESETS.items():
        if key in stem:
            return {"key": key, **preset}
    return None


def resolve_lab_parameters(
    image_path: str | Path,
    pixel_to_mm: float | None,
    capillary_mm: float | None,
    density: float | None,
) -> dict[str, Any]:
    """Resolve working calibration values for a droplet image.

    When the filename matches one of the known lab images, the preset values
    are used so the app stays aligned with the lab-specific calibration.
    """
    preset = infer_lab_preset(image_path)
    resolved = {
        "pixel_to_mm": DEFAULT_PIXEL_TO_MM if pixel_to_mm is None else float(pixel_to_mm),
        "capillary_mm": capillary_mm,
        "density": density,
        "preset": preset,
    }

    if preset is None:
        if resolved["capillary_mm"] is None:
            resolved["capillary_mm"] = 2.7
        if resolved["density"] is None:
            resolved["density"] = 1000.0
        return resolved

    resolved["pixel_to_mm"] = float(preset["pixel_to_mm"])
    resolved["capillary_mm"] = float(preset["capillary_mm"])
    resolved["density"] = float(preset["density"])

    return resolved
