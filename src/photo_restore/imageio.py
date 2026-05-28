"""Image I/O: load, save, stream to stdout, detect grayscale, carry metadata.

Uses Pillow + NumPy only (no torch), so it loads without the ML stack.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

# Extensions we treat as input images in directory mode.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Grayscale detection tolerance: max allowed per-pixel channel spread (0-255)
# for an RGB image to still count as "really grayscale" (a scanned B&W photo).
_GRAY_TOLERANCE = 6


@dataclass
class LoadedImage:
    """An image plus the provenance we need to save it faithfully."""

    array: np.ndarray  # H x W x 3, uint8, RGB (grayscale is expanded to 3 channels)
    is_grayscale: bool
    source_format: str | None  # PIL format of the input, e.g. "JPEG"
    exif: bytes | None  # raw EXIF blob to carry over, if any


def load(path: Path) -> LoadedImage:
    """Load an image as RGB uint8, recording whether it is really grayscale."""
    with Image.open(path) as im:
        source_format = im.format
        exif = im.info.get("exif")
        oriented = ImageOps.exif_transpose(im) or im  # honor orientation, then drop it
        gray = _is_grayscale_mode(oriented.mode)
        rgb = oriented.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
    if not gray:
        gray = _looks_grayscale(array)
    return LoadedImage(array=array, is_grayscale=gray, source_format=source_format, exif=exif)


def to_pil(array: np.ndarray, *, grayscale: bool) -> Image.Image:
    """Build a PIL image from an RGB array, collapsing to L if grayscale.

    Collapsing guarantees no model-introduced color tint survives — the
    "never colorize" invariant.
    """
    im = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    if grayscale:
        im = im.convert("L")
    return im


def save(
    image: Image.Image,
    dest: Path,
    *,
    fmt: str | None = None,
    quality: int = 95,
    exif: bytes | None = None,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_format = fmt or _format_for_suffix(dest.suffix) or "PNG"
    with dest.open("wb") as fh:
        _encode(image, fh, save_format, quality=quality, exif=exif)


def write_stream(
    image: Image.Image,
    stream: BinaryIO,
    *,
    fmt: str = "PNG",
    quality: int = 95,
    exif: bytes | None = None,
) -> None:
    """Write encoded image bytes to a binary stream (e.g. stdout when piped)."""
    _encode(image, stream, fmt, quality=quality, exif=exif)


def stdout_is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def stdout_buffer() -> BinaryIO:
    return sys.stdout.buffer


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def iter_images(root: Path, *, recurse: bool = True) -> list[Path]:
    walker = root.rglob("*") if recurse else root.glob("*")
    return sorted(p for p in walker if p.is_file() and is_image_file(p))


def _encode(
    image: Image.Image,
    fh: BinaryIO,
    fmt: str,
    *,
    quality: int,
    exif: bytes | None,
) -> None:
    fmt = fmt.upper()
    params: dict[str, object] = {}
    if fmt in {"JPEG", "JPG"}:
        fmt = "JPEG"
        params["quality"] = quality
        params["optimize"] = True
        if image.mode not in {"L", "RGB"}:
            image = image.convert("RGB")
    if exif is not None and fmt in {"JPEG", "PNG", "TIFF", "WEBP"}:
        params["exif"] = exif
    image.save(fh, format=fmt, **params)


def _format_for_suffix(suffix: str) -> str | None:
    s = suffix.lower()
    if s in {".jpg", ".jpeg"}:
        return "JPEG"
    if s == ".png":
        return "PNG"
    if s in {".tif", ".tiff"}:
        return "TIFF"
    if s == ".webp":
        return "WEBP"
    if s == ".bmp":
        return "BMP"
    return None


def _is_grayscale_mode(mode: str) -> bool:
    return mode in {"1", "L", "LA", "I", "I;16"}


def _looks_grayscale(rgb: np.ndarray) -> bool:
    """True if an RGB array is visually grayscale (a B&W scan stored as RGB)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return False
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    return bool(spread.max() <= _GRAY_TOLERANCE)
