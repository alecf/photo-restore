"""Orchestrates the restoration stages for a single image.

contrast -> build upscaled background -> composite restored faces onto it.

Faces are restored at 512 and composited onto the *already upscaled* background,
so they never pass through the super-resolution GAN (which would make their
texture diverge from the rest of the image). The ML stages import torch lazily,
so --help, --dry-run, and contrast/resize-only runs don't pay its slow import.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from photo_restore import imageio, resolution
from photo_restore.imageio import LoadedImage
from photo_restore.resolution import Target
from photo_restore.stages import contrast


@dataclass
class Config:
    target: Target
    strength: str = "conservative"
    fidelity: float | None = None  # CodeFormer fidelity for strength="balanced"
    do_face: bool = True
    do_contrast: bool = True
    face_blend: float = 0.8  # 1.0 = fully restored face, 0.0 = original
    face_restore_threshold: int = 500  # skip faces larger than this (px); 0 = restore all
    face_grain: bool = True  # match film grain so the face reads as the same scan
    device: str = "mps"
    upscaler: str = "realesrgan-x4plus"


def restore_image(loaded: LoadedImage, config: Config) -> Image.Image:
    """Run the pipeline and return a PIL image at the exact target size.

    Grayscale inputs are collapsed back to a single channel at the end, so no
    model-introduced tint can survive — the "never colorize" guarantee.
    """
    array = loaded.array
    orig_h, orig_w = array.shape[0], array.shape[1]
    target_w, target_h = resolution.resolve_dimensions(config.target, orig_w, orig_h)

    if config.do_contrast:
        array = contrast.normalize(array)

    background = _build_background(array, orig_w, orig_h, target_w, target_h, config)

    if config.do_face:
        from photo_restore.stages import faces

        result = faces.restore_onto(
            array,
            background,
            scale_ratio=target_w / orig_w,
            strength=config.strength,
            device=config.device,
            fidelity=config.fidelity,
            blend=config.face_blend,
            restore_threshold=config.face_restore_threshold,
            grain=config.face_grain,
        )
    else:
        result = background

    image = imageio.to_pil(result, grayscale=loaded.is_grayscale)
    if image.size != (target_w, target_h):
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return image


def _build_background(
    array: np.ndarray,
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
    config: Config,
) -> np.ndarray:
    """The whole image at target size, upscaled by the SR model when enlarging."""
    if resolution.needs_enlargement(orig_w, orig_h, target_w, target_h):
        from photo_restore.stages import upscale

        array = upscale.upscale(array, weight_name=config.upscaler, device=config.device)
    if (array.shape[1], array.shape[0]) == (target_w, target_h):
        return array
    resized = Image.fromarray(array, "RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)
