"""Orchestrates the four restoration stages for a single image.

Stage order: contrast -> faces -> upscale -> fit-to-target. Each is skippable.
The ML stages are only imported when actually used, so a contrast-only or
resize-only run needs no torch.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from photo_restore import imageio, resolution
from photo_restore.imageio import LoadedImage
from photo_restore.resolution import Target
from photo_restore.stages import contrast


@dataclass
class Config:
    target: Target
    strength: str = "conservative"
    do_face: bool = True
    do_contrast: bool = True
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

    if config.do_face:
        from photo_restore.stages import faces

        array = faces.restore(array, strength=config.strength, device=config.device)

    if resolution.needs_enlargement(orig_w, orig_h, target_w, target_h):
        from photo_restore.stages import upscale

        array = upscale.upscale(array, weight_name=config.upscaler, device=config.device)

    image = imageio.to_pil(array, grayscale=loaded.is_grayscale)
    if image.size != (target_w, target_h):
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return image
