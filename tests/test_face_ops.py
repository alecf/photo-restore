"""Unit tests for the pure (no-ML) face-compositing helpers."""

import numpy as np

from photo_restore.stages.faces import _blend, _match_grain, _should_restore


class TestShouldRestore:
    def test_small_face_is_restored(self):
        assert _should_restore(120, 500) is True

    def test_face_at_threshold_is_restored(self):
        assert _should_restore(500, 500) is True

    def test_large_face_is_skipped(self):
        assert _should_restore(800, 500) is False

    def test_threshold_zero_disables_gating(self):
        assert _should_restore(5000, 0) is True


class TestBlend:
    def test_alpha_one_is_fully_restored(self):
        restored = np.full((4, 4, 3), 200, np.uint8)
        original = np.full((4, 4, 3), 50, np.uint8)
        assert np.array_equal(_blend(restored, original, 1.0), restored)

    def test_alpha_zero_is_original(self):
        restored = np.full((4, 4, 3), 200, np.uint8)
        original = np.full((4, 4, 3), 50, np.uint8)
        assert np.array_equal(_blend(restored, original, 0.0), original)

    def test_alpha_half_is_midpoint(self):
        restored = np.full((4, 4, 3), 200, np.uint8)
        original = np.full((4, 4, 3), 100, np.uint8)
        out = _blend(restored, original, 0.5)
        assert np.all(out == 150)
        assert out.dtype == np.uint8

    def test_out_of_range_alpha_is_clamped(self):
        restored = np.full((2, 2, 3), 200, np.uint8)
        original = np.full((2, 2, 3), 50, np.uint8)
        assert np.array_equal(_blend(restored, original, 5.0), restored)


class TestMatchGrain:
    def test_preserves_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        face = np.full((32, 32, 3), 128, np.uint8)
        noisy = (rng.integers(80, 170, (32, 32, 3))).astype(np.uint8)
        out = _match_grain(face, noisy, rng=rng)
        assert out.shape == face.shape
        assert out.dtype == np.uint8

    def test_adds_variation_when_reference_is_noisy(self):
        rng = np.random.default_rng(1)
        face = np.full((64, 64, 3), 128, np.uint8)
        noisy = (rng.normal(128, 30, (64, 64, 3)).clip(0, 255)).astype(np.uint8)
        out = _match_grain(face, noisy, rng=np.random.default_rng(1))
        # a flat face should gain texture (nonzero variance) from grain matching
        assert out.astype(np.int16).std() > 0

    def test_clean_reference_adds_little(self):
        rng = np.random.default_rng(2)
        face = np.full((32, 32, 3), 128, np.uint8)
        clean = np.full((32, 32, 3), 200, np.uint8)  # no high-freq content
        out = _match_grain(face, clean, rng=rng)
        assert np.allclose(out, face, atol=2)

    def test_grayscale_stays_grayscale(self):
        # identical noise across channels keeps R==G==B
        rng = np.random.default_rng(3)
        g = np.full((32, 32), 128, np.uint8)
        face = np.stack([g, g, g], axis=-1)
        ref_g = (np.random.default_rng(9).normal(128, 25, (32, 32)).clip(0, 255)).astype(np.uint8)
        ref = np.stack([ref_g, ref_g, ref_g], axis=-1)
        out = _match_grain(face, ref, rng=rng)
        assert np.array_equal(out[..., 0], out[..., 1])
        assert np.array_equal(out[..., 1], out[..., 2])
