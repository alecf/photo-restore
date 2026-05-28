# photo-restore — design

_Date: 2026-05-28_

A local CLI that restores scans of old family photos on Apple Silicon (M2 Max,
GPU via PyTorch MPS). It upscales, restores faces, and normalizes contrast while
**preserving identity** and **never colorizing**. Built to iterate on models
quickly in Python now, with a clean enough core to grow toward a native Mac app
later.

## Goals & constraints

- **Faithful, not imaginative.** Output should look like a cleaned-up version of
  the same photo — not a re-imagined one. This rules out diffusion restorers
  (DiffBIR, SUPIR, StableSR), which hallucinate plausible-but-wrong faces.
- **Never colorize.** Grayscale in → grayscale out, guaranteed by detecting
  grayscale input and stripping any introduced tint at the end.
- **Runs on the GPU by default** (MPS) on an M2 Max; CPU only as an explicit
  override or genuine fallback.
- **Aspect ratio is sacred.** Output is never stretched or skewed.

## Pipeline

Each photo flows through up to four stages, each individually skippable:

1. **Pre-process (classical, no ML):** auto-contrast / levels normalization and
   light denoise (Pillow + NumPy). Often the single biggest visible win on faded
   scans, and nearly free.
2. **Face restoration (GFPGAN v1.4):** detects faces, restores each, pastes back.
   No faces found → no-op. Chosen as the most identity-faithful option; the more
   aggressive CodeFormer (`w` fidelity weight) is available as `--strength
   balanced`.
3. **Upscale (Real-ESRGAN x4plus via `spandrel`):** runs on the whole image.
   Loaded through spandrel so SwinIR / HAT / DAT can be swapped in later without
   touching orchestration. Only runs when the target needs enlargement.
4. **Fit to target (Lanczos):** ML models emit fixed integer scales (4×). We
   upscale at the native factor, then Lanczos-resize to the exact target,
   always preserving aspect ratio.

All ML runs on MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1` so any unimplemented op
falls back per-op to CPU instead of crashing.

## CLI

```
restore-photos INPUT [-o OUTPUT] [options]
```

- **Single file:** `-o out.jpg`, or pipe to a non-TTY stdout (streams image
  bytes; PNG default). TTY + no `-o` → friendly refusal, never raw binary.
- **Directory:** `-o OUTDIR` mirrors filenames, recurses by default, skips
  existing outputs (resumable) unless `--overwrite`. Progress to **stderr**.

### Resolution (aspect ratio always preserved)

| Flag | Meaning |
|---|---|
| `--scale same` (default) | restore quality, keep original dimensions |
| `--scale 2x` / `3x` / `4x` | multiply both dimensions |
| `--size WxH` | **fit inside** this box (longer side hits the limit) |
| `--size Wx` / `xH` | constrain width or height only |

Mechanism: pick the smallest model native factor that meets-or-exceeds the
target, then Lanczos to the exact target. Never upscale past need; never distort.

### Other flags

`--strength conservative|balanced`, `--no-face`, `--no-contrast`,
`--device auto|mps|cpu`, `--format png|jpeg`, `--quality N`, `--overwrite`,
`--no-recurse`, `--dry-run`.

### Device (default = GPU)

- `auto` (default) → MPS whenever available (always, on this machine); CPU only
  if MPS is genuinely absent.
- `mps` → force MPS, **error loudly** if unavailable (no silent slow fallback).
- `cpu` → force CPU (debugging / comparison).
- Resolved device printed once to stderr at startup.

## Weights

Not in git (60–350 MB each). Downloaded on first use to
`$XDG_CACHE_HOME/photo-restore` (default `~/.cache/photo-restore`), reused after.
`restore-photos models --list` / `--download-all` pre-warms the cache. Each entry
pins a URL; SHA256 is verified when known, otherwise validated by load.

- GFPGAN v1.4 — `TencentARC/GFPGAN` release asset.
- Real-ESRGAN x4plus — `xinntao/Real-ESRGAN` release asset.

## Layout

```
photo-restore/
  pyproject.toml            # deps + console_scripts → restore-photos
  README.md
  src/photo_restore/
    cli.py                  # arg parsing, TTY logic, single vs batch dispatch
    pipeline.py             # orchestrates the four stages
    device.py               # MPS/CPU selection + fallback env
    models.py               # weight registry: URLs, cache, download/verify
    imageio.py              # load/save, format detect, stdout streaming,
                            #   grayscale detection, EXIF carry-over
    resolution.py           # pure scale/size math (fit-to-box, aspect)
    stages/
      contrast.py           # classical pre-process
      faces.py              # GFPGAN wrapper (lazy ML import)
      upscale.py            # spandrel + Real-ESRGAN (lazy ML import)
  tests/                    # pure logic gets real assertions; ML gets smoke tests
```

The ML stack (`torch`, `spandrel`, `facexlib`, `opencv`) is a **core dependency**
— running the models is the whole point of the tool. The stages still
**lazy-import** torch so `--help`, `--dry-run`, and contrast/resize-only runs
don't pay its slow import cost. Only CodeFormer (`spandrel_extra_arches`) is an
opt-in extra, because of its non-commercial license.

## Error handling

- **Batch isolation:** a bad file logs to stderr and the batch continues; exit
  code reflects whether any file failed.
- Unreadable / non-image input → clear message.
- OOM on a huge scan → caught, hints `--device cpu` or a smaller `--size`.
- Download failure / SHA mismatch → retry once, then a clear error naming the
  model and URL.

## Testing

- **Real assertions:** resolution math (every `--scale`/`--size` form → correct
  dimensions, aspect ratio never altered), CLI arg parsing, TTY-refusal,
  directory mirroring + skip-existing, grayscale detection, contrast pass.
- **Smoke tests:** each ML stage loads its model, runs on a tiny fixture, asserts
  output shape, aspect preserved, grayscale stays grayscale. No pixel-quality
  assertions.

## Model-loading decision

The `gfpgan` and `realesrgan` pip packages both depend on `basicsr`, which does
not build on Python ≥ 3.12 (its `setup.py` fails) and imports
`torchvision.transforms.functional_tensor`, removed in torchvision ≥ 0.17. We
therefore depend on **neither**. `spandrel` runs every network (upscaler and
face nets) from their original `.pth` weights, and `facexlib` provides face
detection / alignment / paste-back without `basicsr`. This keeps the stack
buildable on current Python and torchvision and matches the "spandrel for
everything" principle, at the cost of orchestrating face crop → net → paste
ourselves (a few lines in `stages/faces.py`).

## Out of scope for v1

Colorization, scratch/inpainting models, GUI, native Mac app (the core is kept
clean to grow toward one later).
