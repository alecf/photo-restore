from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def color_image(tmp_path: Path) -> Path:
    arr = np.zeros((30, 40, 3), dtype=np.uint8)
    arr[..., 0] = 200  # clearly red -> not grayscale
    arr[..., 1] = 50
    arr[..., 2] = 10
    path = tmp_path / "color.png"
    Image.fromarray(arr, "RGB").save(path)
    return path


@pytest.fixture
def gray_image(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    arr = rng.integers(40, 210, size=(30, 40), dtype=np.uint8)
    path = tmp_path / "gray.png"
    Image.fromarray(arr, "L").save(path)
    return path


@pytest.fixture
def gray_as_rgb_image(tmp_path: Path) -> Path:
    """A B&W scan stored as RGB (R==G==B) — should be detected as grayscale."""
    rng = np.random.default_rng(1)
    g = rng.integers(40, 210, size=(30, 40), dtype=np.uint8)
    arr = np.stack([g, g, g], axis=-1)
    path = tmp_path / "gray_rgb.png"
    Image.fromarray(arr, "RGB").save(path)
    return path
