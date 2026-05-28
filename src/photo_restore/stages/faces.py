"""Face restoration via GFPGAN (conservative) or CodeFormer (balanced).

Both detect faces, restore each crop, and paste it back. If no faces are found
the input is returned unchanged. Identity preservation is the priority, so the
default is GFPGAN v1.4 and CodeFormer is run at a high fidelity weight.

Heavy ML libraries are imported lazily inside `restore` so the rest of the tool
loads without them.
"""

from __future__ import annotations

import numpy as np

from photo_restore import models

# strength -> (weight name, GFPGANer arch, codeformer fidelity weight)
_STRENGTH = {
    "conservative": ("gfpgan-v1.4", "clean", 0.5),
    "balanced": ("codeformer", "CodeFormer", 0.75),
}

_MISSING_ML = (
    "Face restoration needs the ML extra. Install it with:\n"
    '    uv pip install -e ".[ml]"   (or: pip install "photo-restore[ml]")'
)


def restore(
    array: np.ndarray, *, strength: str = "conservative", device: str = "mps"
) -> np.ndarray:
    """Restore faces in an RGB uint8 array. Returns RGB uint8 of the same size.

    `strength` selects the model: 'conservative' (GFPGAN v1.4, identity-faithful)
    or 'balanced' (CodeFormer at fidelity weight 0.75 — sharper, recovers more
    from heavy degradation, slightly higher risk of subtle facial change).
    """
    if strength not in _STRENGTH:
        raise ValueError(f"unknown strength {strength!r}; choose from {sorted(_STRENGTH)}")

    try:
        import cv2
        from gfpgan import GFPGANer
    except ImportError as err:  # pragma: no cover - exercised only without the ml extra
        raise RuntimeError(_MISSING_ML) from err

    weight_name, arch, fidelity = _STRENGTH[strength]
    model_path = str(models.ensure_weight(weight_name))

    restorer = GFPGANer(
        model_path=model_path,
        upscale=1,  # we control scaling separately in the upscale stage
        arch=arch,
        channel_multiplier=2,
        bg_upsampler=None,
        device=device,
    )

    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    _, _, restored_bgr = restorer.enhance(
        bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=fidelity,
    )
    if restored_bgr is None:
        return array  # no faces detected / nothing to paste back
    rgb: np.ndarray = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    return rgb
