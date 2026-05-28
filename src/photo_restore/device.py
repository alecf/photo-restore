"""Compute-device selection. Defaults to the GPU (MPS) on Apple Silicon.

Importing this module is cheap; it only touches `torch` when a device is
actually resolved, so the CLI and tests load without the ML stack installed.
"""

from __future__ import annotations

import os
import sys

DeviceChoice = str  # "auto" | "mps" | "cpu"


class DeviceError(RuntimeError):
    """Raised when a forced device is unavailable."""


def enable_mps_fallback() -> None:
    """Let ops MPS doesn't implement fall back per-op to CPU instead of crashing.

    Set before any torch op runs. Honored only if the user hasn't set it.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(choice: DeviceChoice = "auto", *, announce: bool = True) -> str:
    """Resolve a device choice to a concrete torch device string.

    - "auto"  -> "mps" if available (the default on this machine), else "cpu".
    - "mps"   -> "mps", or raise DeviceError if MPS is genuinely unavailable
                 (so a misconfigured env can't silently cost 20x the runtime).
    - "cpu"   -> "cpu".
    """
    enable_mps_fallback()

    import torch

    mps_available = torch.backends.mps.is_available()

    if choice == "cpu":
        device = "cpu"
    elif choice == "mps":
        if not mps_available:
            raise DeviceError(
                "MPS was requested but is not available. Use --device auto to fall "
                "back to CPU, or check your PyTorch install supports Apple Silicon."
            )
        device = "mps"
    elif choice == "auto":
        device = "mps" if mps_available else "cpu"
    else:
        raise DeviceError(f"unknown device choice {choice!r}")

    if announce:
        print(f"device: {device}", file=sys.stderr)
    return device
