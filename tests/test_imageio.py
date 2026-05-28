from pathlib import Path

import numpy as np
from PIL import Image

from photo_restore import imageio


def test_detects_color(color_image: Path):
    loaded = imageio.load(color_image)
    assert loaded.is_grayscale is False
    assert loaded.array.shape == (30, 40, 3)


def test_detects_grayscale_mode(gray_image: Path):
    loaded = imageio.load(gray_image)
    assert loaded.is_grayscale is True
    # expanded to 3 channels for a uniform pipeline
    assert loaded.array.shape == (30, 40, 3)


def test_detects_grayscale_stored_as_rgb(gray_as_rgb_image: Path):
    loaded = imageio.load(gray_as_rgb_image)
    assert loaded.is_grayscale is True


def test_to_pil_collapses_grayscale():
    arr = np.dstack([np.full((4, 4), 120, np.uint8)] * 3)
    im = imageio.to_pil(arr, grayscale=True)
    assert im.mode == "L"


def test_to_pil_keeps_color():
    arr = np.zeros((4, 4, 3), np.uint8)
    im = imageio.to_pil(arr, grayscale=False)
    assert im.mode == "RGB"


def test_save_infers_format_from_suffix(tmp_path: Path):
    im = Image.new("L", (8, 8), 100)
    dest = tmp_path / "out.jpg"
    imageio.save(im, dest)
    with Image.open(dest) as reread:
        assert reread.format == "JPEG"


def test_iter_images_filters_and_sorts(tmp_path: Path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.JPG").write_bytes(b"x")
    found = imageio.iter_images(tmp_path, recurse=True)
    names = [p.name for p in found]
    assert names == ["a.png", "c.JPG"]
    flat = imageio.iter_images(tmp_path, recurse=False)
    assert [p.name for p in flat] == ["a.png"]
