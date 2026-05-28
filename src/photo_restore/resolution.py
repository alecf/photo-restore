"""Pure resolution math: turn a `--scale`/`--size` request into exact target
dimensions, always preserving aspect ratio.

This module has no image or ML dependencies on purpose — it is the most
correctness-sensitive logic in the tool and is unit-tested in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SCALE_RE = re.compile(r"^(\d+(?:\.\d+)?)x?$", re.IGNORECASE)
_SIZE_RE = re.compile(r"^(\d+)?x(\d+)?$", re.IGNORECASE)


class ResolutionError(ValueError):
    """Raised when a --scale/--size value can't be parsed or makes no sense."""


@dataclass(frozen=True)
class Target:
    """A resolved request for output size, independent of any input image."""

    kind: str  # "same" | "scale" | "size"
    factor: float | None = None  # for kind == "scale"
    width: int | None = None  # for kind == "size" (bounding box)
    height: int | None = None  # for kind == "size" (bounding box)


def parse_scale(value: str) -> Target:
    """Parse `--scale`: 'same', '2x', '3', '4x', '1.5x'."""
    v = value.strip().lower()
    if v == "same":
        return Target(kind="same")
    m = _SCALE_RE.match(v)
    if not m:
        raise ResolutionError(
            f"invalid --scale {value!r}: expected 'same' or a factor like '2x', '3', '4x'"
        )
    factor = float(m.group(1))
    if factor <= 0:
        raise ResolutionError(f"invalid --scale {value!r}: factor must be positive")
    return Target(kind="scale", factor=factor)


def parse_size(value: str) -> Target:
    """Parse `--size`: 'WxH' (fit inside box), 'Wx' (width only), 'xH' (height only)."""
    v = value.strip().lower()
    m = _SIZE_RE.match(v)
    if not m or (m.group(1) is None and m.group(2) is None):
        raise ResolutionError(
            f"invalid --size {value!r}: expected 'WxH', 'Wx', or 'xH' "
            "(e.g. '2000x2000', '2000x', 'x1500')"
        )
    width = int(m.group(1)) if m.group(1) is not None else None
    height = int(m.group(2)) if m.group(2) is not None else None
    for dim in (width, height):
        if dim is not None and dim <= 0:
            raise ResolutionError(f"invalid --size {value!r}: dimensions must be positive")
    return Target(kind="size", width=width, height=height)


def resolve_dimensions(target: Target, orig_w: int, orig_h: int) -> tuple[int, int]:
    """Resolve a Target against an actual image size to exact output dimensions.

    Aspect ratio is always preserved. For a `size` bounding box, the image is
    scaled so it fits *inside* the box (the more constraining dimension wins).
    """
    if orig_w <= 0 or orig_h <= 0:
        raise ResolutionError(f"invalid source dimensions: {orig_w}x{orig_h}")

    if target.kind == "same":
        return orig_w, orig_h

    if target.kind == "scale":
        assert target.factor is not None
        return _round_dim(orig_w * target.factor), _round_dim(orig_h * target.factor)

    # kind == "size": compute the scale factor each present constraint implies,
    # then take the smallest so the result fits inside the box.
    factors: list[float] = []
    if target.width is not None:
        factors.append(target.width / orig_w)
    if target.height is not None:
        factors.append(target.height / orig_h)
    factor = min(factors)
    return _round_dim(orig_w * factor), _round_dim(orig_h * factor)


def needs_enlargement(orig_w: int, orig_h: int, target_w: int, target_h: int) -> bool:
    """Whether the target is larger than the source on either axis.

    When false, the super-resolution model is skipped entirely: the target is at
    or below the source size, so we only restore and Lanczos-resample. When true,
    we run the model at its native factor and then Lanczos to the exact target.
    """
    return target_w > orig_w or target_h > orig_h


def _round_dim(value: float) -> int:
    return max(1, round(value))
