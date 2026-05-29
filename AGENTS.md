# photo-restore

A local command-line tool that restores **scans of old family photos** on Apple
Silicon: upscaling, face restoration, and contrast — while staying **faithful to
the original** (it does not re-imagine faces) and **never colorizing**. Images in,
enhanced images out. Built in Python so models are easy to swap and try; the core
is kept clean enough to grow toward a native Mac app later.

The original design discussion lives in `docs/plans/2026-05-28-photo-restore-design.md`.

## Tech Stack

- **Runtime / packaging**: Python ≥3.11, managed with [`uv`](https://docs.astral.sh/uv/). `hatchling` build backend, `src/` layout.
- **CLI**: `typer` (single root command; entry point `restore-photos = photo_restore.cli:app`).
- **ML stack (core deps, not optional)**: `torch` + `torchvision` (MPS), `spandrel` (loads/runs every network), `facexlib` (face detect/align/paste), `opencv-python`. Running the models is the whole point, so these are required dependencies.
- **Imaging**: Pillow + NumPy for I/O, contrast, and resampling.
- **Lint/format/types**: `ruff` (line length 100, double quotes, 4-space indent) and `mypy --strict`. `pytest` for tests.
- **Optional extra**: `balanced` adds `spandrel-extra-arches` for CodeFormer (non-commercial license — see below).

## Project Structure

```
src/photo_restore/
  cli.py            # typer CLI: arg parsing, TTY/stdout logic, single vs batch dispatch
  pipeline.py       # Config + restore_image(): orchestrates the stages
  resolution.py     # PURE scale/size math (no imaging) — heavily unit-tested
  device.py         # MPS/CPU selection; sets PYTORCH_ENABLE_MPS_FALLBACK
  models.py         # weight registry (pinned URLs) + cache/download to ~/.cache/photo-restore
  imageio.py        # load/save, stdout streaming, grayscale detection, EXIF carry-over
  stages/
    contrast.py     # classical (no-ML) levels normalization
    upscale.py      # super-resolution via spandrel (Real-ESRGAN x4plus)
    faces.py        # face restoration + compositing; pure helpers (_blend/_match_*/_should_restore)
tests/              # pure logic gets real assertions; ML stages are smoke-tested
docs/plans/         # design docs
```

## Common Commands

```bash
uv venv
uv pip install -e ".[dev]"          # full tool + test/lint tools
uv pip install -e ".[balanced,dev]" # + CodeFormer (--strength balanced)
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
restore-photos --download-models    # pre-warm the weight cache
```

**Verify before claiming done:** `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`.

## Architecture: the pipeline

`pipeline.restore_image` runs, each stage skippable:

1. **Contrast** — classical levels normalization (no ML), hue-preserving.
2. **Background** — upscale the whole image (Real-ESRGAN x4plus via spandrel, only when enlarging), Lanczos to the exact target.
3. **Faces** — detect on the source, restore each 512 crop, and **composite onto the upscaled background**.

ML stages **lazy-import torch** so `--help`/`--dry-run`/contrast-only runs stay fast.

## Learnings (read before touching the model code)

**Never add `gfpgan`, `realesrgan`, or `basicsr`.** `basicsr` does not build on
Python ≥3.12 (its `setup.py` throws `KeyError: '__version__'`) and imports
`torchvision.transforms.functional_tensor`, removed in torchvision ≥0.17. Both
`gfpgan` and `realesrgan` drag it in. Instead: **spandrel runs every network**
(it loads a `.pth` and auto-detects the architecture) and **facexlib** does face
detection/alignment/paste-back — the same pieces gfpgan uses internally, minus
the unbuildable dependency.

**spandrel usage.** `ModelLoader().load_from_file(path)` → `ImageModelDescriptor`.
Call `descriptor(tensor)` with a `BCHW` RGB tensor in `[0, 1]`; it returns `[0, 1]`
and clamps/pads internally. `descriptor.model` is the raw `nn.Module` when you need
to pass custom forward args.

**CodeFormer (`--strength balanced`) is special.** Its architecture lives in
`spandrel_extra_arches` (the opt-in `balanced` extra), kept separate because
CodeFormer ships under a **non-commercial license** — keep it out of core deps.
Register it with `MAIN_REGISTRY.add(*EXTRA_REGISTRY)`. spandrel's default call runs
it at fidelity `weight=0.5` (quality-biased, drifts from identity); to control
fidelity, call the raw model: `descriptor.model(tensor, weight=w)[0].clamp(0, 1)`
(high `w` = faithful to the input). GFPGAN (the conservative default) is
permissively licensed and has no such knob.

**Faces composite onto the upscaled background — they never go through the SR
GAN.** facexlib's `paste_faces_to_input_image(upsample_img=bg)` with a float
`upscale_factor = target_w / orig_w` does this. Running the restored face through
Real-ESRGAN makes its texture diverge from the background ("photoshopped" look).

**Face restorers regenerate the face from a color-trained prior**, so they trend
smooth and inject blue-ish eyes / reddish lips even on B&W or sepia inputs. Four
defaults counter this; keep them on unless a user opts out: composite-not-upscale
(above), `--face-blend` (mix restored over original to keep skin texture),
`--face-grain` (match source grain), `--face-restore-threshold` (skip regenerating
faces already large in the source), `--match-face-color` (take the face's chroma
from the source crop, keep the model's luma). **Avoid diffusion restorers
(DiffBIR/SUPIR)** — they hallucinate identity, which is the one thing this tool
must not do.

**Never colorize.** Grayscale is detected at load (`mode L`, or RGB whose channels
are within tolerance) and the final image is collapsed back to `L`. Sepia/tinted
scans take the color path, so `--match-face-color` is what keeps invented face
colors out of them.

**Device defaults to the GPU.** `device.resolve_device("auto")` picks MPS when
available and sets `PYTORCH_ENABLE_MPS_FALLBACK=1`. `--device mps` errors loudly
rather than silently falling back to CPU (which would be ~20× slower).

**Resolution math is pure and exact.** SR models emit a fixed integer scale (4×);
upscale at the native factor, then Lanczos to the exact target. Aspect ratio is
never altered. Keep this logic in `resolution.py` and unit-tested.

**facexlib spams torchvision `pretrained=` deprecation warnings** on every run;
they're suppressed by message in `faces.py`. Suppress by message, never blanket-
silence warnings.

## Conventions

- **Pure helpers get real unit tests** (resolution math; the `faces` helpers
  `_should_restore`/`_blend`/`_match_grain`/`_match_color`; contrast). ML stages get
  smoke tests only. These helpers are testable without a GPU because the ML stack
  (incl. `cv2`/`numpy`) is a core dependency.
- **Weights are never committed.** They auto-download to `~/.cache/photo-restore/`
  from the pinned registry in `models.py`.
- **Never commit photos.** `.gitignore` blocks `*.jpg/*.jpeg/*.png` tree-wide and
  the `inputs/`/`outputs/` work directories — inputs are personal family photos.
  Check `git status` before committing.
- `from __future__ import annotations` in modules; `mypy --strict` must pass.
- Keep this file current: when a non-obvious decision is made or corrected, record
  the timeless principle here (and the *why*), not the incident.
