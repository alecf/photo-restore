from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from photo_restore import imageio
from photo_restore.cli import app

runner = CliRunner()


def test_list_models_exits_clean():
    result = runner.invoke(app, ["--list-models"])
    assert result.exit_code == 0
    assert "realesrgan-x4plus" in result.output
    assert "gfpgan-v1.4" in result.output


def test_missing_input_is_error():
    result = runner.invoke(app, [])
    assert result.exit_code != 0


def test_scale_and_size_conflict(color_image: Path):
    result = runner.invoke(app, [str(color_image), "--scale", "2x", "--size", "100x100"])
    assert result.exit_code != 0
    assert "not both" in result.output


def test_directory_requires_output(tmp_path: Path, color_image: Path):
    # color_image lives in tmp_path; pass the directory as input
    result = runner.invoke(app, [str(color_image.parent)])
    assert result.exit_code != 0
    assert "needs -o" in result.output


def test_dry_run_reports_without_working(color_image: Path, tmp_path: Path):
    out = tmp_path / "out.png"
    result = runner.invoke(app, [str(color_image), "-o", str(out), "--dry-run"])
    assert result.exit_code == 0
    assert "would restore" in result.output
    assert not out.exists()


def test_skip_existing_unless_overwrite(color_image: Path, tmp_path: Path):
    out = tmp_path / "exists.png"
    out.write_bytes(b"already here")
    result = runner.invoke(app, [str(color_image), "-o", str(out), "--no-face", "--scale", "same"])
    assert result.exit_code == 0
    assert "skip (exists)" in result.output
    assert out.read_bytes() == b"already here"


def test_contrast_only_run_without_ml(gray_as_rgb_image: Path, tmp_path: Path):
    # --no-face + same size needs no torch; exercises the full I/O + collapse path.
    out = tmp_path / "out.png"
    result = runner.invoke(
        app, [str(gray_as_rgb_image), "-o", str(out), "--no-face", "--scale", "same"]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    with Image.open(out) as im:
        assert im.mode == "L"  # grayscale preserved, never colorized
        assert im.size == (40, 30)


def test_tty_refusal(monkeypatch, color_image: Path):
    monkeypatch.setattr(imageio, "stdout_is_tty", lambda: True)
    result = runner.invoke(app, [str(color_image), "--no-face", "--scale", "same"])
    assert result.exit_code != 0
    assert "terminal" in result.output


def test_error_message_names_exception_type(tmp_path: Path):
    # A failing file must report *what* went wrong. Some exceptions (e.g.
    # spandrel's UnsupportedModelError) have an empty str(), so the handler must
    # include the exception type name, not just its message.
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not a real png")
    out = tmp_path / "out.png"
    result = runner.invoke(app, [str(bad), "-o", str(out), "--no-face", "--scale", "same"])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "UnidentifiedImageError" in result.output


def test_directory_batch_contrast_only(tmp_path: Path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("L", (20, 16), 120).save(in_dir / name)
    out_dir = tmp_path / "out"
    result = runner.invoke(app, [str(in_dir), "-o", str(out_dir), "--no-face", "--scale", "same"])
    assert result.exit_code == 0, result.output
    assert (out_dir / "a.png").exists()
    assert (out_dir / "b.png").exists()
