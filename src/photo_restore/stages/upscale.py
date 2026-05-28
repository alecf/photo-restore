"""Super-resolution via spandrel, defaulting to Real-ESRGAN x4plus.

spandrel loads ~20 SR architectures (SwinIR, HAT, DAT, ...) behind one
interface, so swapping the upscaler later means changing a weight name, not this
code. Heavy ML libraries are imported lazily inside `upscale`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from photo_restore import models

_MISSING_ML = (
    "Upscaling needs the ML extra. Install it with:\n"
    '    uv pip install -e ".[ml]"   (or: pip install "photo-restore[ml]")'
)


def upscale(
    array: np.ndarray, *, weight_name: str = "realesrgan-x4plus", device: str = "mps"
) -> np.ndarray:
    """Run the super-resolution model once at its native factor.

    Returns an enlarged RGB uint8 array; the caller resamples to the exact
    target. Raises a clear, actionable error on out-of-memory.
    """
    try:
        import torch
    except ImportError as err:  # pragma: no cover - exercised only without the ml extra
        raise RuntimeError(_MISSING_ML) from err

    model = _load_model(weight_name, device)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)

    try:
        with torch.no_grad():
            out = model(tensor)
    except RuntimeError as err:
        if _is_oom(err):
            raise RuntimeError(
                "ran out of memory upscaling this image. Try a smaller --size, or "
                "--device cpu (slower but more headroom)."
            ) from err
        raise

    out = out.clamp(0.0, 1.0).mul(255.0).round().squeeze(0).permute(1, 2, 0)
    result: np.ndarray = out.to("cpu", torch.uint8).numpy()
    return result


@lru_cache(maxsize=2)
def _load_model(weight_name: str, device: str):  # type: ignore[no-untyped-def]
    try:
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader
    except ImportError as err:  # pragma: no cover
        raise RuntimeError(_MISSING_ML) from err

    path = models.ensure_weight(weight_name)
    descriptor = ModelLoader().load_from_file(str(path))
    if not isinstance(descriptor, ImageModelDescriptor):
        raise RuntimeError(f"{weight_name} is not a single-image super-resolution model")
    return descriptor.to(torch.device(device)).eval()


def _is_oom(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg
