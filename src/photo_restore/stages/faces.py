"""Face restoration via spandrel + facexlib.

facexlib detects, aligns, and pastes faces back; spandrel runs the restoration
network (GFPGAN v1.4 by default, CodeFormer for `--strength balanced`). This
avoids the `gfpgan`/`basicsr` packages, which don't build on modern Python.

If no faces are found the input is returned unchanged. Identity preservation is
the priority, so the default is GFPGAN v1.4.

Heavy ML libraries are imported lazily inside `restore`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from photo_restore import models

# strength -> weight registry name
_STRENGTH = {
    "conservative": "gfpgan-v1.4",
    "balanced": "codeformer",
}

_MISSING_ML = (
    "Face restoration needs the ML extra. Install it with:\n"
    '    uv pip install -e ".[ml]"   (or: pip install "photo-restore[ml]")'
)


def restore(
    array: np.ndarray, *, strength: str = "conservative", device: str = "mps"
) -> np.ndarray:
    """Restore faces in an RGB uint8 array. Returns RGB uint8 of the same size.

    `strength` selects the network: 'conservative' (GFPGAN v1.4, identity-faithful)
    or 'balanced' (CodeFormer — sharper, recovers more from heavy degradation,
    slightly higher risk of subtle facial change).
    """
    if strength not in _STRENGTH:
        raise ValueError(f"unknown strength {strength!r}; choose from {sorted(_STRENGTH)}")

    try:
        import cv2
        import torch
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    except ImportError as err:  # pragma: no cover - exercised only without the ml extra
        raise RuntimeError(_MISSING_ML) from err

    model = _load_model(_STRENGTH[strength], device)

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
            out = model(tensor)
        restored_rgb = (
            out.clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .squeeze(0)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
        ).numpy()
        helper.add_restored_face(cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR))

    helper.get_inverse_affine(None)
    pasted_bgr = helper.paste_faces_to_input_image()
    result: np.ndarray = cv2.cvtColor(pasted_bgr, cv2.COLOR_BGR2RGB)
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
        raise RuntimeError(f"{weight_name} is not a usable image model")
    return descriptor.to(torch.device(device)).eval()
