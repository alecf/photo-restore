"""Face restoration via spandrel + facexlib.

facexlib detects and aligns faces; spandrel runs the restoration network; the
restored faces are composited onto a caller-supplied (already upscaled)
background. Compositing this way means the face never passes through the
super-resolution GAN, so its texture doesn't diverge from the background.

Three knobs fight the "photoshopped face" look, since these nets *regenerate*
the face from a learned prior:

- size gating: faces already larger than a threshold in the source are left to
  the gentler background upscaler instead of being regenerated.
- blend: the regenerated face is mixed back over the original crop, retaining
  real skin texture.
- grain match: synthetic film grain matched to the original crop is added so the
  face reads as the same photo/scan as everything around it.

Strengths: 'conservative' = GFPGAN v1.4 (identity-faithful, permissive license);
'balanced' = CodeFormer at a high fidelity weight (sharper; needs the `balanced`
extra, non-commercial license). Heavy ML libraries are imported lazily.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

from photo_restore import models

# strength -> (weight name, default fidelity weight, needs spandrel_extra_arches)
_STRENGTH: dict[str, tuple[str, float | None, bool]] = {
    "conservative": ("gfpgan-v1.4", None, False),
    "balanced": ("codeformer", 0.8, True),
}

_MISSING_ML = (
    "PyTorch isn't importable — the install looks incomplete. Reinstall with:\n"
    "    uv pip install -e .   (or: pip install photo-restore)"
)

_MISSING_BALANCED = (
    "--strength balanced uses CodeFormer, whose architecture ships in the "
    "separate `balanced` extra (non-commercial license). Install it with:\n"
    '    uv pip install -e ".[balanced]"\n'
    "Or use the permissively licensed default (--strength conservative)."
)

_extra_arches_registered = False


def _should_restore(face_px: float, threshold: int) -> bool:
    """Whether a face this many source-pixels wide/tall should be regenerated.

    `threshold <= 0` disables gating (restore every face). Otherwise faces larger
    than the threshold are already detailed enough that regeneration tends to
    look synthetic, so they're left to the background upscaler.
    """
    if threshold <= 0:
        return True
    return face_px <= threshold


def _blend(restored: np.ndarray, original: np.ndarray, alpha: float) -> np.ndarray:
    """Mix the restored face over the original crop. alpha=1 fully restored, 0 original."""
    a = float(np.clip(alpha, 0.0, 1.0))
    out = restored.astype(np.float32) * a + original.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _match_color(restored: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Give the restored face the *color* of the reference (source) crop while
    keeping its own brightness/detail.

    These nets are trained on color faces and inject blue-ish eyes / reddish lips
    even into B&W or sepia inputs. Taking luma from the restored face and
    chroma from the source neutralizes that: grayscale stays gray, sepia stays
    sepia, real color stays the source's real color.
    """
    import cv2

    out = cv2.cvtColor(restored, cv2.COLOR_RGB2YCrCb)
    ref = cv2.cvtColor(reference, cv2.COLOR_RGB2YCrCb)
    out[:, :, 1] = ref[:, :, 1]  # Cr
    out[:, :, 2] = ref[:, :, 2]  # Cb
    result: np.ndarray = cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)
    return result


def _match_grain(
    face: np.ndarray,
    reference: np.ndarray,
    *,
    strength: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add film grain to `face` matched to the high-frequency noise of `reference`.

    The same noise is applied to all channels so a grayscale face stays gray.
    """
    import cv2

    if rng is None:
        rng = np.random.default_rng()
    ref_luma = reference.astype(np.float32).mean(axis=2)
    high_freq = ref_luma - cv2.GaussianBlur(ref_luma, (0, 0), 1.0)
    noise_std = min(float(np.std(high_freq)), 20.0) * strength
    if noise_std <= 0.0:
        return face
    noise = rng.normal(0.0, noise_std, size=face.shape[:2])[:, :, None]
    out = face.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def restore_onto(
    array: np.ndarray,
    background: np.ndarray,
    *,
    scale_ratio: float,
    strength: str = "conservative",
    device: str = "mps",
    fidelity: float | None = None,
    blend: float = 0.8,
    restore_threshold: int = 500,
    grain: bool = True,
    match_color: bool = True,
) -> np.ndarray:
    """Detect faces in `array` (native res), restore them, and composite onto
    `background` (an RGB uint8 image already upscaled to the target size).

    `scale_ratio` is target_width / source_width (== height ratio, aspect is
    preserved). Returns RGB uint8 at the background's size. If no faces are
    restored, `background` is returned unchanged.
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

    # facexlib initializes its detector with torchvision's deprecated
    # `pretrained=` API, which spams two UserWarnings on every call. Silence just
    # those (matched by message) while the helper spins up.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message=r".*deprecated since 0\.13.*"
        )
        helper = FaceRestoreHelper(
            upscale_factor=scale_ratio,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
        )
    helper.clean_all()
    helper.read_image(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    if helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5) == 0:
        return background

    helper.align_warp_face()
    rng = np.random.default_rng()
    kept_affines: list[np.ndarray] = []
    kept_faces_bgr: list[np.ndarray] = []
    for cropped_bgr, affine, det in zip(
        helper.cropped_faces, helper.affine_matrices, helper.det_faces, strict=True
    ):
        x1, y1, x2, y2 = det[:4]
        if not _should_restore(max(x2 - x1, y2 - y1), restore_threshold):
            continue  # already detailed; leave the upscaled background face alone

        crop_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(crop_rgb))
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
                out = model.model(tensor, weight=fid)[0].clamp(0.0, 1.0)  # CodeFormer + fidelity
        restored_rgb = (
            out.mul(255.0).round().squeeze(0).permute(1, 2, 0).to("cpu", torch.uint8)
        ).numpy()

        if match_color:
            restored_rgb = _match_color(restored_rgb, crop_rgb)
        restored_rgb = _blend(restored_rgb, crop_rgb, blend)
        if grain:
            restored_rgb = _match_grain(restored_rgb, crop_rgb, rng=rng)
        kept_affines.append(affine)
        kept_faces_bgr.append(cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR))

    if not kept_faces_bgr:
        return background

    # Composite only the kept faces onto the supplied upscaled background.
    helper.affine_matrices = kept_affines
    helper.inverse_affine_matrices = []
    helper.restored_faces = []
    helper.get_inverse_affine(None)
    for face_bgr in kept_faces_bgr:
        helper.add_restored_face(face_bgr)

    pasted_bgr = helper.paste_faces_to_input_image(
        upsample_img=cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
    )
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
