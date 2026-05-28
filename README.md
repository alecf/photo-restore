# photo-restore

A local command-line tool for restoring **scans of old family photos** on Apple
Silicon. It upscales, restores faces, and normalizes contrast while staying
**faithful to the original** — it does not re-imagine faces and it never
colorizes. Grayscale in, grayscale out.

Built in Python so you can iterate on models quickly, with a clean core to grow
toward a native Mac app later.

## What it does

| Stage | What | Model |
|---|---|---|
| Contrast | Auto levels/contrast on faded scans (hue-preserving) | classical, no ML |
| Faces | Detect → restore → paste back; no-op if no faces | GFPGAN v1.4 (default) |
| Upscale | Super-resolution on the whole image | Real-ESRGAN x4plus (via `spandrel`) |
| Fit | Lanczos resize to your exact target, aspect ratio preserved | Pillow |

The defaults are tuned to **preserve identity**. The newer diffusion restorers
(DiffBIR, SUPIR) look flashier but invent plausible-but-wrong faces, so they are
deliberately not used here. For sharper recovery on heavily degraded photos,
`--strength balanced` switches the face model to CodeFormer at a high fidelity
weight.

## Requirements

- macOS on Apple Silicon (M1/M2/M3…). Runs on the GPU via PyTorch **MPS** by
  default. (Other platforms fall back to CPU.)
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

## Install

```bash
git clone https://github.com/alecf/photo-restore
cd photo-restore

# Create a virtual env and install with the ML extra (the models).
uv venv
uv pip install -e ".[ml,dev]"
```

> **Note on the `ml` extra:** it installs `torch`, `spandrel`, `gfpgan`, and
> friends (a few GB). The base install (`uv pip install -e .`) gives you the
> CLI, contrast, and resizing without the heavy stack — handy for development.

Model weights (60–350 MB each) download automatically on first use to
`~/.cache/photo-restore/`. Pre-download them with:

```bash
restore-photos --download-models
```

## Usage

```bash
# Single file -> explicit output
restore-photos old.jpg -o restored.jpg

# Pipe to stdout (PNG); refuses to dump binary to a terminal
restore-photos old.jpg > restored.png

# A whole directory -> mirrored into another directory (resumable)
restore-photos ./scans -o ./restored

# 2x upscale
restore-photos old.jpg -o big.jpg --scale 2x

# Fit inside a 2000x2000 box, aspect ratio preserved (never stretched)
restore-photos old.jpg -o web.jpg --size 2000x2000

# Constrain width only
restore-photos old.jpg -o w1600.jpg --size 1600x
```

### Resolution

Aspect ratio is **always** preserved — output is never stretched or skewed.

| Flag | Meaning |
|---|---|
| `--scale same` (default) | restore quality, keep original dimensions |
| `--scale 2x` / `3x` / `4x` | multiply both dimensions |
| `--size WxH` | fit inside this box (the longer side hits the limit) |
| `--size Wx` / `xH` | constrain width or height only |

The model only enlarges at fixed factors, so the tool upscales at the native
factor and then Lanczos-resizes down to your exact target.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `-o, --output` | stdout | Output file or directory |
| `--strength` | `conservative` | `conservative` (GFPGAN) or `balanced` (CodeFormer) |
| `--no-face` | off | Skip face restoration |
| `--no-contrast` | off | Skip contrast normalization |
| `--device` | `auto` | `auto` (GPU), `mps`, or `cpu` |
| `--format` | by extension / PNG | `png` or `jpeg` |
| `--quality` | `95` | JPEG quality |
| `--overwrite` | off | Reprocess files that already have output |
| `--no-recurse` | off | Don't descend into subdirectories |
| `--dry-run` | off | Report what would happen; do nothing |

### Models

```bash
restore-photos --list-models       # show models + cache status
restore-photos --download-models   # pre-warm the cache (e.g. before going offline)
```

## Development

```bash
uv pip install -e ".[dev]"   # light: CLI + pure logic, no torch
uv run pytest                # tests (pure logic + I/O; ML stages are smoke-tested)
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

The resolution math, CLI dispatch, grayscale handling, and contrast pass have
real unit tests and run without the ML stack. The ML stages lazy-import torch,
so the test suite and a contrast-only/resize-only run work in a base install.

## Known issues

`gfpgan`/`basicsr` historically import `torchvision.transforms.functional_tensor`,
which was removed in torchvision ≥ 0.17. The `ml` extra pins
`torchvision>=0.16,<0.17` to avoid the breakage. If you bump torchvision and see
an import error from `basicsr`, that's the cause — either keep the pin or apply
the one-line shim that re-exports `functional_tensor` from
`torchvision.transforms.functional`.

## License

MIT — see [LICENSE](LICENSE).
