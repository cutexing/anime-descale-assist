# anime-descale-assist

`anime-descale-assist` is a VapourSynth native-resolution probe for anime
sources. The goal is not a one-click descale button. The goal is to automate the
VapourSynth/descale candidate check that a human can review before building a
final filter chain.

The current path intentionally stays small:

- probe an existing folder of `.pgm`, `.ppm`, `.pnm`, or uncompressed `.bmp`
  frames
- test candidate native heights, kernels, and sample-grid shifts through
  VapourSynth and `vapoursynth-descale`
- write `vs_probe.json` with ranked height and kernel candidates
- print a compact candidate summary in the terminal

This is a decision-support tool. Low-confidence candidates should remain under
human review.

## Install for development

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,vs]"
.venv\Scripts\python.exe -m vapoursynth config
```

For this workspace, keeping Python local is recommended:

```powershell
.venv\Scripts\python.exe -m anime_descale_assist --help
.venv\Scripts\descale-assist.exe probe-vs sample\720p --out work\sample-vs
.venv\Scripts\pytest.exe
.venv\Scripts\ruff.exe check .
```

The local `.venv/` directory is ignored by git. VapourSynth and
`vapoursynth-descale` are required for probing.

## Usage

Probe a directory of samples with VapourSynth and `vapoursynth-descale`:

```bash
descale-assist probe-vs samples/ --out work/samples-vs
```

Sample directories are scanned recursively, so nested folders such as
`sample/720p/show-name/*.bmp` work directly.

Probe a single sample:

```bash
descale-assist probe-vs sample/frame0001.pgm --out work/frame0001-vs
```

Print VapourSynth plugin status:

```bash
descale-assist vs-env
```

Use `vspipe` for a focused round-trip check:

```powershell
.\.venv\Scripts\vspipe.exe --info --arg samples=sample\765p --arg height=765 --arg kernel=catrom --arg src_top=-0.5 scripts\sample_roundtrip.vpy -
```

## Current limits

- Input samples must already be extracted as supported image files.
- Results are sample-level probes, not final cut-accurate timeline zones.
- Credit masks, fractional source heights, and border handling are planned but
  not implemented yet.

## Roadmap

1. VapourSynth/descale candidate validation.
2. Better low-confidence diagnostics.
3. Credit/overlay mask generation with original-resolution merge-back.
