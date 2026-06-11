# anime-descale-assist

`anime-descale-assist` is a conservative native-resolution auditor for anime
sources. The goal is not a one-click descale button. The goal is to automate the
boring parts: sparse sampling, scale-signature scoring, clustering, zone
summaries, QC previews, and a VapourSynth script skeleton that a human can
review.

The first MVP intentionally stays small:

- sample a video with `ffmpeg`, or analyze an existing folder of `.pgm`,
  `.ppm`, `.pnm`, or uncompressed `.bmp` frames
- test candidate native heights and resize kernels with an edge-weighted,
  complexity-normalized round-trip signature score
- group sampled frames into scale-signature clusters
- write `analysis.json`, `zones.json`, `summary.md`, and grayscale QC triptychs
- emit a starter `.vpy` with candidate descale zones

This is a decision-support tool. Low-confidence areas should remain passthrough
until a human reviews the QC output.

## Install for development

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,vs]"
.venv\Scripts\python.exe -m vapoursynth config
```

For this workspace, keeping Python local is recommended:

```powershell
.venv\Scripts\python.exe -m anime_descale_assist --help
.venv\Scripts\descale-assist.exe analyze input.mkv --out work/input
.venv\Scripts\descale-assist.exe probe-vs sample\720p --out work\sample-vs
.venv\Scripts\pytest.exe
.venv\Scripts\ruff.exe check .
```

The local `.venv/` directory is ignored by git. Video extraction uses `ffmpeg`
from `PATH` when available, otherwise it falls back to the bundled ffmpeg
installed by `imageio-ffmpeg`. VapourSynth is only needed when you use the
emitted `.vpy` script.

## Usage

Analyze a video:

```bash
descale-assist analyze input.mkv --out work/input
```

Analyze a directory of PGM samples:

```bash
descale-assist analyze samples/ --out work/samples --no-extract
```

Sample directories are scanned recursively, so nested folders such as
`sample/720p/show-name/*.bmp` work directly.

Emit a VapourSynth starter script:

```bash
descale-assist emit-vpy work/input --out work/input/filter.vpy
```

Print a short report:

```bash
descale-assist qc work/input
```

Probe samples with VapourSynth and `vapoursynth-descale`:

```powershell
.\.venv\Scripts\descale-assist.exe probe-vs sample\720p --out work\sample-vs
.\.venv\Scripts\vspipe.exe --info --arg samples=sample\720p --arg height=720 --arg kernel=bilinear scripts\sample_roundtrip.vpy -
```

## Current limits

- The MVP uses a normalized resize round-trip heuristic, not the real inverse
  kernels from `vapoursynth-descale`.
- The timeline zones are based on sparse samples, so they are candidates, not
  final cut-accurate ranges.
- Credit masks, fractional source heights, sample-grid shifts, and border
  handling are planned but not implemented yet.

## Roadmap

1. Sparse sampling and scale-signature clustering.
2. Native-height/kernel candidate reports with QC previews.
3. Mixed-content timeline classifier and merged zones.
4. VapourSynth/descale backend for exact candidate validation.
5. Credit/overlay mask generation with original-resolution merge-back.
