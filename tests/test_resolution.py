import pytest

from photo_restore.resolution import (
    ResolutionError,
    needs_enlargement,
    parse_scale,
    parse_size,
    resolve_dimensions,
)


def test_parse_scale_same():
    assert parse_scale("same").kind == "same"


@pytest.mark.parametrize(("text", "factor"), [("2x", 2.0), ("3", 3.0), ("4x", 4.0), ("1.5x", 1.5)])
def test_parse_scale_factor(text, factor):
    t = parse_scale(text)
    assert t.kind == "scale"
    assert t.factor == factor


@pytest.mark.parametrize("bad", ["", "x", "-2x", "0", "abc", "2.x.3"])
def test_parse_scale_rejects_garbage(bad):
    with pytest.raises(ResolutionError):
        parse_scale(bad)


@pytest.mark.parametrize(
    ("text", "w", "h"),
    [("2000x1500", 2000, 1500), ("2000x", 2000, None), ("x1500", None, 1500)],
)
def test_parse_size(text, w, h):
    t = parse_size(text)
    assert t.kind == "size"
    assert t.width == w
    assert t.height == h


@pytest.mark.parametrize("bad", ["", "x", "2000", "axb", "-1x2", "0x0"])
def test_parse_size_rejects_garbage(bad):
    with pytest.raises(ResolutionError):
        parse_size(bad)


def test_same_keeps_dimensions():
    assert resolve_dimensions(parse_scale("same"), 800, 600) == (800, 600)


def test_scale_multiplies_both_axes():
    assert resolve_dimensions(parse_scale("2x"), 800, 600) == (1600, 1200)


def test_size_box_preserves_aspect_ratio_width_bound():
    # 800x600 (4:3) into a 2000x2000 box -> width-bound at... height is the limit
    # actually 4:3 wider than tall, so width hits first: factor = min(2000/800, 2000/600)
    out_w, out_h = resolve_dimensions(parse_size("2000x2000"), 800, 600)
    assert (out_w, out_h) == (2000, 1500)
    assert out_w / out_h == pytest.approx(800 / 600)


def test_size_box_never_stretches_tall_image():
    out_w, out_h = resolve_dimensions(parse_size("1000x1000"), 600, 800)
    assert (out_w, out_h) == (750, 1000)
    assert out_w / out_h == pytest.approx(600 / 800)


def test_size_width_only_locks_aspect():
    out_w, out_h = resolve_dimensions(parse_size("400x"), 800, 600)
    assert (out_w, out_h) == (400, 300)


def test_size_height_only_locks_aspect():
    out_w, out_h = resolve_dimensions(parse_size("x300"), 800, 600)
    assert (out_w, out_h) == (400, 300)


def test_needs_enlargement():
    assert needs_enlargement(800, 600, 1600, 1200) is True
    assert needs_enlargement(800, 600, 800, 600) is False
    assert needs_enlargement(800, 600, 400, 300) is False
    # larger on a single axis still counts
    assert needs_enlargement(800, 600, 800, 601) is True


def test_invalid_source_dimensions():
    with pytest.raises(ResolutionError):
        resolve_dimensions(parse_scale("same"), 0, 100)
