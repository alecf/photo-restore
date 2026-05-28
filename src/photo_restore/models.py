"""Model-weight registry: where to download each weight, where to cache it, and
how to verify it.

Weights are large (60-350 MB) and never committed. They download on first use to
``$XDG_CACHE_HOME/photo-restore`` (default ``~/.cache/photo-restore``) and are
reused afterwards.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Weight:
    name: str
    url: str
    filename: str
    # SHA256 is verified when known. Left None until pinned against a real
    # download — fabricating a hash would reject valid files, which is worse
    # than validating by successful load.
    sha256: str | None = None
    # A sanity floor so an HTML error page or truncated download is rejected.
    min_bytes: int = 1_000_000


REGISTRY: dict[str, Weight] = {
    "gfpgan-v1.4": Weight(
        name="gfpgan-v1.4",
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        filename="GFPGANv1.4.pth",
        min_bytes=300_000_000,
    ),
    "realesrgan-x4plus": Weight(
        name="realesrgan-x4plus",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        filename="RealESRGAN_x4plus.pth",
        min_bytes=60_000_000,
    ),
    "codeformer": Weight(
        name="codeformer",
        url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        filename="codeformer.pth",
        min_bytes=300_000_000,
    ),
}


class WeightError(RuntimeError):
    """Raised when a weight can't be downloaded or fails verification."""


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    d = root / "photo-restore"
    d.mkdir(parents=True, exist_ok=True)
    return d


def weight_path(name: str) -> Path:
    return cache_dir() / REGISTRY[name].filename


def is_cached(name: str) -> bool:
    p = weight_path(name)
    return p.exists() and p.stat().st_size >= REGISTRY[name].min_bytes


def ensure_weight(name: str, *, retries: int = 1) -> Path:
    """Return the local path to a weight, downloading + verifying it if needed."""
    if name not in REGISTRY:
        raise WeightError(f"unknown model {name!r}; known: {', '.join(sorted(REGISTRY))}")
    if is_cached(name):
        return weight_path(name)

    weight = REGISTRY[name]
    dest = weight_path(name)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            _download(weight, dest)
            _verify(weight, dest)
            return dest
        except Exception as err:
            last_err = err
            dest.unlink(missing_ok=True)
            if attempt < retries:
                print(f"retrying download of {weight.name}...", file=sys.stderr)
    raise WeightError(f"failed to fetch {weight.name} from {weight.url}: {last_err}") from last_err


def _download(weight: Weight, dest: Path) -> None:
    print(f"downloading {weight.name} -> {dest}", file=sys.stderr)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(weight.url) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)


def _verify(weight: Weight, dest: Path) -> None:
    size = dest.stat().st_size
    if size < weight.min_bytes:
        raise WeightError(f"{weight.name} download too small ({size} bytes); likely an error page")
    if weight.sha256 is not None:
        digest = _sha256(dest)
        if digest != weight.sha256:
            raise WeightError(
                f"{weight.name} sha256 mismatch: expected {weight.sha256}, got {digest}"
            )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
