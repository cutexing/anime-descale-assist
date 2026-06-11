from pathlib import Path

import vapoursynth as vs

from anime_descale_assist.vapoursynth_backend import (
    load_sample_clip,
    roundtrip_clip,
    to_float_clip,
)

core = vs.core

samples = globals().get("samples", "sample/720p")
height = int(globals().get("height", "720"))
kernel = globals().get("kernel", "bilinear")

sample_path = Path(samples)
if not sample_path.is_absolute() and not sample_path.exists():
    script_dir = Path(__file__).resolve().parent
    for base in (Path.cwd(), Path.cwd().parent, script_dir, script_dir.parent):
        candidate = base / sample_path
        if candidate.exists():
            sample_path = candidate
            break

src8, info = load_sample_clip(sample_path, max_samples=int(globals().get("max_samples", "8")))
src = to_float_clip(src8)
native_width = round(src.width * height / src.height)
reconstruction = roundtrip_clip(src, kernel, native_width, height)
reconstruction8 = core.resize.Bicubic(
    reconstruction,
    format=vs.GRAY8,
    range_in_s="full",
    range_s="full",
    dither_type="none",
)

# Output alternates source and reconstruction frames.
out = core.std.Interleave([src8, reconstruction8])
out.set_output()
