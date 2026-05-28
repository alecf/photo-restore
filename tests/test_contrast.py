import numpy as np

from photo_restore.stages import contrast


def test_normalize_stretches_low_contrast_range():
    # A washed-out image using only values 100..150 should expand toward 0..255.
    rng = np.random.default_rng(0)
    arr = rng.integers(100, 150, size=(20, 20, 3), dtype=np.uint8)
    out = contrast.normalize(arr)
    assert out.min() < arr.min()
    assert out.max() > arr.max()
    assert out.dtype == np.uint8
    assert out.shape == arr.shape


def test_normalize_keeps_grayscale_grayscale():
    g = np.random.default_rng(1).integers(80, 160, size=(16, 16), dtype=np.uint8)
    arr = np.stack([g, g, g], axis=-1)
    out = contrast.normalize(arr)
    # all three channels must stay identical -> no color introduced
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])


def test_normalize_handles_flat_image():
    arr = np.full((8, 8, 3), 128, dtype=np.uint8)
    out = contrast.normalize(arr)
    assert out.shape == arr.shape  # no divide-by-zero, returns a copy
