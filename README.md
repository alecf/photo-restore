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
| Upscale | Super-resolution to build the target-size background | Real-ESRGAN x4plus (via `spandrel`) |
| Faces | Detect → restore → composite onto the upscaled background | GFPGAN v1.4 (default) |
| Fit | Lanczos resize to your exact target, aspect ratio preserved | Pillow |

The defaults are tuned to **preserve identity**. The newer diffusion restorers
(DiffBIR, SUPIR) look flashier but invent plausible-but-wrong faces, so they are
deliberately not used here. For sharper recovery on heavily degraded photos,
`--strength balanced` switches the face model to CodeFormer at a high fidelity
weight.

**Keeping faces from looking "photoshopped."** Face restorers *regenerate* the
face from a learned prior, so on already-sharp inputs they can look smoother than
the rest of the photo. Three defaults counter this: restored faces are
composited onto the upscaled background (they never pass through the upscaler, so
textures match); the result is blended back over the original (`--face-blend`) to
keep real skin texture; matched film grain is added (`--face-grain`); and faces
already large in the source are left alone (`--face-restore-threshold`). Tune
these per photo.

Because these nets are trained on color faces, they paint blue-ish eyes and
reddish lips even onto black-and-white or sepia scans. By default
(`--match-face-color`) the restored face takes its *color* from the source crop
while keeping the model's recovered detail — so B&W stays gray, sepia stays
sepia, and color photos keep their own (not invented) tones.

## Requirements

- macOS on Apple Silicon (M1/M2/M3…). Runs on the GPU via PyTorch **MPS** by
  default. (Other platforms fall back to CPU.)
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

## Install

```bash
git clone https://github.com/alecf/photo-restore
cd photo-restore

# Create a virtual env and install. This pulls the ML stack (torch, spandrel,
# facexlib, opencv) — a few GB — because running the models is the whole point.
uv venv
uv pip install -e .

# Optional: add CodeFormer for `--strength balanced` (non-commercial license).
uv pip install -e ".[balanced]"
```

> **Note on the `balanced` extra:** `--strength balanced` uses CodeFormer, whose
> architecture ships in `spandrel_extra_arches` under a **non-commercial
> license**. It is the one piece kept opt-in, so the default (`conservative`,
> GFPGAN + Real-ESRGAN, permissively licensed) carries no such restriction.
> Without this extra, `--strength balanced` fails with an explanatory message.

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
| `--fidelity` | `0.8` | CodeFormer fidelity for `balanced`: `1.0` = most faithful to the input face, `0.0` = most freely reconstructed. Keep high for family photos. |
| `--face-blend` | `0.8` | Blend of the restored face over the original: `1.0` = fully restored, `0.0` = original. Lower keeps more real skin texture (less "photoshopped"). |
| `--face-restore-threshold` | `500` | Skip regenerating faces already larger than this many **source** pixels — they look synthetic when regenerated, so they're left to the background upscaler. `0` restores every face. |
| `--face-grain` / `--no-face-grain` | on | Add film grain matched to the source so the restored face reads as the same scan. |
| `--match-face-color` / `--no-match-face-color` | on | Recolor the restored face to the source's color (B&W stays gray, sepia stays sepia) instead of the model's invented blue eyes / red lips. |
| `--no-face` | off | Skip face restoration |
| `--no-contrast` | off | Skip contrast normalization |
| `--device` | `auto` | `auto` (GPU), `mps`, or `cpu` |
| `--format` | by extension / PNG | `png` or `jpeg` |
| `--quality` | `95` | JPEG quality |
| `--overwrite` | off | Reprocess files that already have output |
| `--no-recurse` | off | Don't descend into subdirectories |
| `--dry-run` | off | Report what would happen; do nothing |
| `--debug` | off | Print full tracebacks on per-file errors |

### Models

```bash
restore-photos --list-models       # show models + cache status
restore-photos --download-models   # pre-warm the cache (e.g. before going offline)
```

## Development

```bash
uv pip install -e ".[dev]"   # adds pytest, ruff, mypy on top of the core stack
uv run pytest                # tests (pure logic + I/O; ML stages are smoke-tested)
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

The resolution math, CLI dispatch, grayscale handling, and contrast pass have
real unit tests that need no model. The ML stages import torch lazily, so
`--help`, `--dry-run`, and contrast/resize-only runs start fast.

## How the models are loaded

The popular `gfpgan` and `realesrgan` pip packages both depend on `basicsr`,
which does not build on modern Python (3.12+) and imports a torchvision module
that was removed in torchvision ≥ 0.17. Rather than fight that, this tool does
not depend on any of them. Instead:

- **`spandrel`** loads and runs every network — the Real-ESRGAN upscaler *and*
  the face-restoration nets (GFPGAN v1.4, CodeFormer) — from their original
  `.pth` weights through one interface. Swapping in SwinIR/HAT/DAT later is a
  weight-registry change, not new inference code.
- **`facexlib`** handles face detection, alignment, and paste-back (the same
  components GFPGAN uses internally), and installs cleanly with no `basicsr`.

The result is a modern, buildable stack on current Python and torchvision.

## License

MIT — see [LICENSE](LICENSE).
