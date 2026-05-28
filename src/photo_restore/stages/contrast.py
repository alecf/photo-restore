"""Classical (no-ML) pre-processing: contrast/levels normalization.

Runs before the ML stages. On faded scans this is often the single biggest
visible win and costs almost nothing. Operates on luminance so it never shifts
color balance — safe for both color and grayscale inputs, and it can't colorize.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def normalize(array: np.ndarray, *, cutoff: float = 0.5) -> np.ndarray:
    """Auto-contrast an RGB uint8 array, preserving hue.

    `cutoff` is the percent of the lightest/darkest pixels to clip before
    stretching — a small clip avoids a single dust speck or blown highlight from
    anchoring the range. Stretch is computed on luminance and applied as a single
    curve to all channels so colors aren't pulled apart (and grayscale stays
    grayscale).
    """
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected H x W x 3 RGB array, got shape {array.shape}")

    luminance = _luminance(array)
    lo, hi = _clip_bounds(luminance, cutoff)
    if hi <= lo:
        return array.copy()

    scale = 255.0 / (hi - lo)
    stretched = (array.astype(np.float32) - lo) * scale
    result: np.ndarray = np.clip(stretched, 0, 255).astype(np.uint8)
    return result


def autocontrast_pil(array: np.ndarray, *, cutoff: float = 0.5) -> np.ndarray:
    """Pillow's autocontrast as an alternative path (per-channel on color).

    Kept for experimentation; `normalize` is the default because operating on a
    shared luminance curve is safer for faithful, hue-preserving restoration.
    """
    im = Image.fromarray(array, mode="RGB")
    out = ImageOps.autocontrast(im, cutoff=cutoff)
    return np.asarray(out, dtype=np.uint8)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return rgb.astype(np.float32) @ weights


def _clip_bounds(luminance: np.ndarray, cutoff: float) -> tuple[float, float]:
    lo = float(np.percentile(luminance, cutoff))
    hi = float(np.percentile(luminance, 100.0 - cutoff))
    return lo, hi
