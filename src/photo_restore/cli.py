"""Command-line interface for photo-restore.

    restore-photos INPUT [-o OUTPUT] [options]

INPUT is a single image or a directory. With no -o, a single image is streamed
to stdout when stdout is piped (PNG by default); writing raw image bytes to a
terminal is refused. A directory mirrors filenames into the -o directory and is
resumable (existing outputs are skipped unless --overwrite).
"""

import sys
import traceback
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image

from photo_restore import imageio, models
from photo_restore.device import DeviceError, resolve_device
from photo_restore.pipeline import Config, restore_image
from photo_restore.resolution import ResolutionError, Target, parse_scale, parse_size

app = typer.Typer(add_completion=False)


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _resolve_target(scale: str | None, size: str | None) -> Target:
    if scale is not None and size is not None:
        raise typer.BadParameter("pass either --scale or --size, not both")
    try:
        if size is not None:
            return parse_size(size)
        return parse_scale(scale if scale is not None else "same")
    except ResolutionError as err:
        raise typer.BadParameter(str(err)) from err


def _may_enlarge(target: Target) -> bool:
    if target.kind == "scale":
        return (target.factor or 1.0) > 1.0
    return target.kind == "size"


def _list_models() -> None:
    for name, weight in sorted(models.REGISTRY.items()):
        status = "cached" if models.is_cached(name) else "not downloaded"
        _eprint(f"{name:22} {status:16} {weight.url}")


def _download_models() -> None:
    for name in sorted(models.REGISTRY):
        models.ensure_weight(name)
    _eprint("all model weights are cached.")


def _process(loaded: imageio.LoadedImage, config: Config) -> Image.Image:
    return restore_image(loaded, config)


def _emit(
    image: Image.Image,
    *,
    out_path: Path | None,
    fmt: str | None,
    quality: int,
    exif: bytes | None,
) -> None:
    if out_path is None:
        imageio.write_stream(
            image, imageio.stdout_buffer(), fmt=(fmt or "PNG"), quality=quality, exif=exif
        )
        return
    imageio.save(image, out_path, fmt=fmt, quality=quality, exif=exif)


def _run_single(
    in_path: Path,
    out_path: Path | None,
    config: Config,
    *,
    fmt: str | None,
    quality: int,
    overwrite: bool,
    dry_run: bool,
    debug: bool = False,
) -> bool:
    if out_path is not None and out_path.exists() and not overwrite:
        _eprint(f"skip (exists): {out_path}")
        return True
    if dry_run:
        dest = out_path if out_path is not None else Path("<stdout>")
        _eprint(f"would restore {in_path} -> {dest}")
        return True
    try:
        loaded = imageio.load(in_path)
        image = _process(loaded, config)
        _emit(image, out_path=out_path, fmt=fmt, quality=quality, exif=loaded.exif)
        if out_path is not None:
            _eprint(f"restored {in_path} -> {out_path}")
        return True
    except Exception as err:
        # Always name the exception type: some errors (e.g. spandrel's
        # UnsupportedModelError) have an empty message, which would otherwise
        # print as "error: <path>:" with nothing useful after it.
        detail = f"{type(err).__name__}: {err}" if str(err) else type(err).__name__
        _eprint(f"error: {in_path}: {detail}")
        if debug:
            traceback.print_exc()
        return False


def _run_directory(
    in_dir: Path,
    out_dir: Path,
    config: Config,
    *,
    fmt: str | None,
    quality: int,
    overwrite: bool,
    recurse: bool,
    dry_run: bool,
    debug: bool = False,
) -> bool:
    files = imageio.iter_images(in_dir, recurse=recurse)
    if not files:
        _eprint(f"no images found in {in_dir}")
        return True
    all_ok = True
    for in_path in files:
        rel = in_path.relative_to(in_dir)
        out_path = out_dir / rel
        if fmt is not None:
            out_path = out_path.with_suffix("." + fmt.lower().replace("jpeg", "jpg"))
        ok = _run_single(
            in_path,
            out_path,
            config,
            fmt=fmt,
            quality=quality,
            overwrite=overwrite,
            dry_run=dry_run,
            debug=debug,
        )
        all_ok = all_ok and ok
    return all_ok


@app.command()
def main(
    input: Annotated[
        Path | None,
        typer.Argument(help="Image file or directory to restore."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Output file or directory. Omit to stream to stdout."),
    ] = None,
    scale: Annotated[
        str | None,
        typer.Option("--scale", help="'same' (default), '2x', '3x', '4x'."),
    ] = None,
    size: Annotated[
        str | None,
        typer.Option("--size", help="Fit inside a box: 'WxH', 'Wx', or 'xH'."),
    ] = None,
    strength: Annotated[
        str,
        typer.Option("--strength", help="Face model: 'conservative' or 'balanced'."),
    ] = "conservative",
    fidelity: Annotated[
        float | None,
        typer.Option(
            "--fidelity",
            help="CodeFormer fidelity for --strength balanced: 1.0=most faithful, "
            "0.0=most invented. Default 0.8.",
        ),
    ] = None,
    no_face: Annotated[bool, typer.Option("--no-face", help="Skip face restoration.")] = False,
    no_contrast: Annotated[
        bool, typer.Option("--no-contrast", help="Skip contrast normalization.")
    ] = False,
    device: Annotated[
        str, typer.Option("--device", help="'auto' (GPU), 'mps', or 'cpu'.")
    ] = "auto",
    fmt: Annotated[
        str | None, typer.Option("--format", help="Output format: 'png' or 'jpeg'.")
    ] = None,
    quality: Annotated[int, typer.Option("--quality", help="JPEG quality (1-100).")] = 95,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Reprocess files that already have output.")
    ] = False,
    no_recurse: Annotated[
        bool, typer.Option("--no-recurse", help="Don't descend into subdirectories.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would happen; do no work.")
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Print full tracebacks on per-file errors.")
    ] = False,
    list_models: Annotated[
        bool, typer.Option("--list-models", help="List models and cache status, then exit.")
    ] = False,
    download_models: Annotated[
        bool, typer.Option("--download-models", help="Pre-download all weights, then exit.")
    ] = False,
) -> None:
    """Restore old family-photo scans: faithful upscaling, face restoration, contrast."""
    if list_models:
        _list_models()
        raise typer.Exit(0)
    if download_models:
        _download_models()
        raise typer.Exit(0)

    if input is None:
        raise typer.BadParameter("INPUT is required (a file or directory)")
    if not input.exists():
        raise typer.BadParameter(f"input not found: {input}")

    target = _resolve_target(scale, size)
    fmt_norm = _normalize_format(fmt)
    if fidelity is not None and not (0.0 <= fidelity <= 1.0):
        raise typer.BadParameter("--fidelity must be between 0.0 and 1.0")

    # Validate the input/output arrangement *before* touching the device, so the
    # clean "directory needs -o" / TTY-refusal errors fire even in a base install
    # without torch (resolving the device imports it).
    is_dir = input.is_dir()
    out_path: Path | None = output
    if is_dir:
        if output is None:
            raise typer.BadParameter("a directory input needs -o OUTPUT_DIR")
    else:
        if output is not None and output.exists() and output.is_dir():
            out_path = output / input.name
            if fmt_norm is not None:
                out_path = out_path.with_suffix("." + fmt_norm.lower().replace("jpeg", "jpg"))
        if out_path is None and imageio.stdout_is_tty():
            raise typer.BadParameter(
                "refusing to write image bytes to a terminal. Use -o FILE, or redirect "
                "stdout (e.g. `restore-photos in.jpg > out.png`)."
            )

    # Resolve the compute device only if a stage will actually run on it, so a
    # contrast-only / downscale-only run works without the ML stack installed.
    device_str = "cpu"
    if not dry_run and (not no_face or _may_enlarge(target)):
        try:
            device_str = resolve_device(device)
        except DeviceError as err:
            raise typer.BadParameter(str(err)) from err

    config = Config(
        target=target,
        strength=strength,
        fidelity=fidelity,
        do_face=not no_face,
        do_contrast=not no_contrast,
        device=device_str,
    )

    if is_dir:
        assert output is not None
        ok = _run_directory(
            input,
            output,
            config,
            fmt=fmt_norm,
            quality=quality,
            overwrite=overwrite,
            recurse=not no_recurse,
            dry_run=dry_run,
            debug=debug,
        )
    else:
        ok = _run_single(
            input,
            out_path,
            config,
            fmt=fmt_norm,
            quality=quality,
            overwrite=overwrite,
            dry_run=dry_run,
            debug=debug,
        )
    raise typer.Exit(0 if ok else 1)


def _normalize_format(fmt: str | None) -> str | None:
    if fmt is None:
        return None
    f = fmt.strip().lower()
    if f in {"jpg", "jpeg"}:
        return "JPEG"
    if f == "png":
        return "PNG"
    raise typer.BadParameter(f"unsupported --format {fmt!r}; use 'png' or 'jpeg'")


if __name__ == "__main__":
    app()
