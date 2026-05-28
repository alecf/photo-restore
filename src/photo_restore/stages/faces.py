"""Face restoration via spandrel + facexlib.

facexlib detects, aligns, and pastes faces back; spandrel runs the restoration
network. This avoids the `gfpgan`/`basicsr` packages, which don't build on
modern Python.

Two strengths:

- 'conservative' (default): GFPGAN v1.4, the most identity-faithful option,
  permissively licensed (Apache-2.0 / BSD).
- 'balanced': CodeFormer, run at a high fidelity weight so it stays faithful to
  the input face while recovering more detail. CodeFormer's architecture lives
  in `spandrel_extra_arches` (the `balanced` install extra) and ships under a
  non-commercial license — see README.

If no faces are found the input is returned unchanged. Heavy ML libraries are
imported lazily inside `restore`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from photo_restore import models

# strength -> (weight name, default fidelity weight, needs spandrel_extra_arches)
# A fidelity of None means the network has no fidelity knob (GFPGAN); we then use
# spandrel's standard call. A float means CodeFormer's `weight` (1.0 = stay
# faithful to the input, 0.0 = freely reconstruct), injected via the raw model.
_STRENGTH: dict[str, tuple[str, float | None, bool]] = {
    "conservative": ("gfpgan-v1.4", None, False),
    "balanced": ("codeformer", 0.8, True),
}

_MISSING_ML = (
    "Face restoration needs the ML extra. Install it with:\n"
    '    uv pip install -e ".[ml]"   (or: pip install "photo-restore[ml]")'
)

_MISSING_BALANCED = (
    "--strength balanced uses CodeFormer, whose architecture ships in the "
    "separate `balanced` extra (non-commercial license). Install it with:\n"
    '    uv pip install -e ".[ml,balanced]"\n'
    "Or use the permissively licensed default (--strength conservative)."
)

_extra_arches_registered = False


def restore(
    array: np.ndarray,
    *,
    strength: str = "conservative",
    device: str = "mps",
    fidelity: float | None = None,
) -> np.ndarray:
    """Restore faces in an RGB uint8 array. Returns RGB uint8 of the same size.

    `fidelity` overrides the default CodeFormer fidelity for 'balanced'; it is
    ignored for 'conservative' (GFPGAN has no such knob).
    """
    if strength not in _STRENGTH:
        raise ValueError(f"unknown strength {strength!r}; choose from {sorted(_STRENGTH)}")

    try:
        import cv2
        import torch
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    except ImportError as err:  # pragma: no cover - exercised only without the ml extra
        raise RuntimeError(_MISSING_ML) from err

    weight_name, default_fidelity, needs_extra = _STRENGTH[strength]
    if needs_extra:
        _register_extra_arches()
    fid = default_fidelity if fidelity is None else fidelity

    model = _load_model(weight_name, device)

    helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device,
    )
    helper.clean_all()
    helper.read_image(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    num_faces = helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
    if num_faces == 0:
        return array  # nothing to restore; leave the image untouched

    helper.align_warp_face()
    for cropped_bgr in helper.cropped_faces:
        face_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(face_rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div(255.0)
            .to(device)
        )
        with torch.no_grad():
            if fid is None:
                out = model(tensor)  # spandrel's standard call (GFPGAN)
            else:
                # Replicate spandrel's path but inject the fidelity weight. The
                # crop is already 512x512 square, so spandrel's pad step is a
                # no-op; the only difference is `weight=fid` instead of 0.5.
                out = model.model(tensor, weight=fid)[0].clamp(0.0, 1.0)
        restored_rgb = (
            out.mul(255.0).round().squeeze(0).permute(1, 2, 0).to("cpu", torch.uint8)
        ).numpy()
        helper.add_restored_face(cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR))

    helper.get_inverse_affine(None)
    pasted_bgr = helper.paste_faces_to_input_image()
    result: np.ndarray = cv2.cvtColor(pasted_bgr, cv2.COLOR_BGR2RGB)
    return result


def _register_extra_arches() -> None:
    global _extra_arches_registered
    if _extra_arches_registered:
        return
    try:
        from spandrel import MAIN_REGISTRY
        from spandrel_extra_arches import EXTRA_REGISTRY
    except ImportError as err:
        raise RuntimeError(_MISSING_BALANCED) from err
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    _extra_arches_registered = True


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
        raise RuntimeError(f"{weight_name} is not a usable image model")
    return descriptor.to(torch.device(device)).eval()
