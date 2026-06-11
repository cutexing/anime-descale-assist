from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .images import GrayImage, list_sample_images, read_image

VS_KERNELS = ("bilinear", "catrom", "bicubicsharp", "mitchell", "lanczos3", "spline36")
VS_HEIGHTS = (720, 756, 810, 864, 900, 936, 960, 1008)


@dataclass(frozen=True)
class SampleClipInfo:
    paths: list[str]
    width: int
    height: int
    length: int


@dataclass(frozen=True)
class VSCandidate:
    height: int
    width: int
    kernel: str
    score: float
    raw_error: float
    frame_errors: list[float]


@dataclass(frozen=True)
class VSProbeResult:
    source: str
    source_width: int
    source_height: int
    sample_count: int
    score_exponent: float
    best: VSCandidate
    candidates: list[VSCandidate]
    height_summary: list[VSCandidate]
    kernel_summary: list[VSCandidate]
    height_confidence: float
    kernel_confidence: float
    classification: str

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def _imports() -> tuple[Any, Any]:
    try:
        import numpy as np
        import vapoursynth as vs
    except ImportError as exc:
        raise RuntimeError(
            "VapourSynth probing requires the vs extra: "
            "python -m pip install -e \".[vs]\""
        ) from exc
    return np, vs


def ensure_descale() -> None:
    _np, vs = _imports()
    if not hasattr(vs.core, "descale"):
        raise RuntimeError(
            "core.descale is not available. Install vapoursynth-descale into this venv."
        )


def _pack_rows(image: GrayImage) -> list[bytes]:
    return [bytes(row) for row in image]


def _sample_paths(path: Path, max_samples: int) -> list[Path]:
    if path.is_dir():
        paths = list_sample_images(path)
    elif path.is_file():
        paths = [path]
    else:
        raise FileNotFoundError(path)
    if not paths:
        raise ValueError(f"no supported sample images found in {path}")
    return paths[:max_samples]


def load_sample_clip(path: Path | str, max_samples: int = 12) -> tuple[Any, SampleClipInfo]:
    _np, vs = _imports()
    source_path = Path(path)
    paths = _sample_paths(source_path, max_samples)
    packed_frames: list[list[bytes]] = []
    width: int | None = None
    height: int | None = None

    for sample_path in paths:
        image, image_width, image_height = read_image(sample_path)
        if width is None:
            width = image_width
            height = image_height
        elif image_width != width or image_height != height:
            raise ValueError(
                f"sample dimensions differ: {sample_path} is "
                f"{image_width}x{image_height}, expected {width}x{height}"
            )
        packed_frames.append(_pack_rows(image))

    assert width is not None and height is not None
    core = vs.core
    blank = core.std.BlankClip(
        width=width,
        height=height,
        format=vs.GRAY8,
        length=len(packed_frames),
        color=0,
    )

    def fill_frame(n: int, f: Any) -> Any:
        out = f.copy()
        ptr = out.get_write_ptr(0).value
        stride = out.get_stride(0)
        for y, row in enumerate(packed_frames[n]):
            (ctypes.c_ubyte * width).from_address(ptr + y * stride)[:] = row
        return out

    clip = core.std.ModifyFrame(blank, blank, fill_frame)
    return (
        clip,
        SampleClipInfo(
            paths=[str(item) for item in paths],
            width=width,
            height=height,
            length=len(packed_frames),
        ),
    )


def to_float_clip(clip: Any) -> Any:
    _np, vs = _imports()
    return vs.core.resize.Bicubic(
        clip,
        format=vs.GRAYS,
        range_in_s="full",
        range_s="full",
    )


def frame_to_array(frame: Any) -> Any:
    np, _vs = _imports()
    ptr = frame.get_read_ptr(0).value
    stride = frame.get_stride(0)
    width = frame.width
    height = frame.height

    if frame.format.bytes_per_sample == 4:
        out = np.empty((height, width), dtype=np.float32)
        for y in range(height):
            row = (ctypes.c_float * width).from_address(ptr + y * stride)
            out[y, :] = np.frombuffer(row, dtype=np.float32, count=width)
        return out * 255.0

    out = np.empty((height, width), dtype=np.uint8)
    for y in range(height):
        row = (ctypes.c_ubyte * width).from_address(ptr + y * stride)
        out[y, :] = np.frombuffer(row, dtype=np.uint8, count=width)
    return out.astype(np.float32)


def edge_weighted_mae_array(source: Any, reconstruction: Any) -> float:
    np, _vs = _imports()
    src = source.astype(np.float32, copy=False)
    rec = reconstruction.astype(np.float32, copy=False)
    dx = np.abs(src[1:-1, 2:] - src[1:-1, :-2])
    dy = np.abs(src[2:, 1:-1] - src[:-2, 1:-1])
    weight = 1.0 + np.minimum((dx + dy) / 32.0, 4.0)
    error = np.abs(src[1:-1, 1:-1] - rec[1:-1, 1:-1])
    return float(np.sum(error * weight) / np.sum(weight))


def descale_clip(clip: Any, kernel: str, width: int, height: int) -> Any:
    ensure_descale()
    d = _imports()[1].core.descale
    if kernel == "bilinear":
        return d.Debilinear(clip, width, height)
    if kernel == "catrom":
        return d.Debicubic(clip, width, height, b=0, c=0.5)
    if kernel == "bicubicsharp":
        return d.Debicubic(clip, width, height, b=0, c=0.75)
    if kernel == "mitchell":
        return d.Debicubic(clip, width, height, b=1 / 3, c=1 / 3)
    if kernel == "lanczos3":
        return d.Delanczos(clip, width, height, taps=3)
    if kernel == "spline36":
        return d.Despline36(clip, width, height)
    raise ValueError(f"unsupported VapourSynth descale kernel: {kernel}")


def upscale_clip(clip: Any, kernel: str, width: int, height: int) -> Any:
    ensure_descale()
    d = _imports()[1].core.descale
    if kernel == "bilinear":
        return d.Bilinear(clip, width, height)
    if kernel == "catrom":
        return d.Bicubic(clip, width, height, b=0, c=0.5)
    if kernel == "bicubicsharp":
        return d.Bicubic(clip, width, height, b=0, c=0.75)
    if kernel == "mitchell":
        return d.Bicubic(clip, width, height, b=1 / 3, c=1 / 3)
    if kernel == "lanczos3":
        return d.Lanczos(clip, width, height, taps=3)
    if kernel == "spline36":
        return d.Spline36(clip, width, height)
    raise ValueError(f"unsupported VapourSynth upscale kernel: {kernel}")


def roundtrip_clip(clip: Any, kernel: str, native_width: int, native_height: int) -> Any:
    native = descale_clip(clip, kernel, native_width, native_height)
    return upscale_clip(native, kernel, clip.width, clip.height)


def _height_confidence(best: VSCandidate, height_summary: list[VSCandidate]) -> float:
    second = next((item for item in height_summary if item.height != best.height), None)
    if second is None or second.score <= 1e-9:
        return 0.0
    separation = max(0.0, (second.score - best.score) / second.score)
    raw_quality = max(0.0, min(1.0, (2.0 - best.raw_error) / 2.0))
    return round(max(0.0, min(1.0, separation * 3.5 + raw_quality * 0.1)), 4)


def _kernel_confidence(best: VSCandidate, kernel_summary: list[VSCandidate]) -> float:
    second = next((item for item in kernel_summary if item.kernel != best.kernel), None)
    if second is None or second.score <= 1e-9:
        return 0.0
    separation = max(0.0, (second.score - best.score) / second.score)
    return round(max(0.0, min(1.0, separation * 2.5)), 4)


def probe_samples(
    path: Path | str,
    heights: list[int] | tuple[int, ...] = VS_HEIGHTS,
    kernels: list[str] | tuple[str, ...] = VS_KERNELS,
    max_samples: int = 8,
    score_exponent: float = 4.0,
) -> VSProbeResult:
    ensure_descale()
    source_path = Path(path)
    source8, info = load_sample_clip(source_path, max_samples=max_samples)
    source = to_float_clip(source8)
    source_arrays = [frame_to_array(source.get_frame(n)) for n in range(info.length)]

    candidates: list[VSCandidate] = []
    for height in heights:
        if height >= info.height:
            continue
        width = max(16, round(info.width * height / info.height))
        for kernel in kernels:
            reconstruction = roundtrip_clip(source, kernel, width, height)
            frame_errors = [
                edge_weighted_mae_array(
                    source_arrays[n],
                    frame_to_array(reconstruction.get_frame(n)),
                )
                for n in range(info.length)
            ]
            raw_error = float(median(frame_errors))
            score = raw_error * ((height / info.height) ** score_exponent)
            candidates.append(
                VSCandidate(
                    height=height,
                    width=width,
                    kernel=kernel,
                    score=round(score, 8),
                    raw_error=round(raw_error, 8),
                    frame_errors=[round(value, 8) for value in frame_errors],
                )
            )

    if not candidates:
        raise ValueError("no valid VapourSynth descale candidates were produced")

    ranked = sorted(candidates, key=lambda item: item.score)
    best = ranked[0]
    height_summary = sorted(
        [min((item for item in ranked if item.height == height), key=lambda item: item.score) for height in sorted({item.height for item in ranked})],
        key=lambda item: item.score,
    )
    kernel_summary = sorted(
        [item for item in ranked if item.height == best.height],
        key=lambda item: item.score,
    )
    height_confidence = _height_confidence(best, height_summary)
    kernel_confidence = _kernel_confidence(best, kernel_summary)
    classification = "high_confidence_height" if height_confidence >= 0.75 else "needs_review"

    return VSProbeResult(
        source=str(source_path),
        source_width=info.width,
        source_height=info.height,
        sample_count=info.length,
        score_exponent=score_exponent,
        best=best,
        candidates=ranked,
        height_summary=height_summary,
        kernel_summary=kernel_summary,
        height_confidence=height_confidence,
        kernel_confidence=kernel_confidence,
        classification=classification,
    )


def vapoursynth_environment() -> dict[str, Any]:
    _np, vs = _imports()
    core = vs.core
    plugins = [
        {
            "identifier": plugin.identifier,
            "namespace": plugin.namespace,
            "name": plugin.name,
        }
        for plugin in core.plugins()
    ]
    return {
        "core": str(core),
        "has_descale": hasattr(core, "descale"),
        "plugins": plugins,
    }
