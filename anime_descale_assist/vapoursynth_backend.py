from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

from .images import GrayImage, list_sample_images, read_image

VS_KERNELS = ("bilinear", "catrom", "bicubicsharp", "mitchell", "lanczos3", "spline36")
VS_HEIGHTS = (720, 756, 765, 800, 810, 838, 850, 864, 900, 936, 960, 1008)
VS_SHIFTS = (0.0, -0.5, 0.5)


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
    src_left: float = 0.0
    src_top: float = 0.0
    height_signal: float = 0.0


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


def descale_clip(
    clip: Any,
    kernel: str,
    width: int,
    height: int,
    src_left: float = 0.0,
    src_top: float = 0.0,
) -> Any:
    ensure_descale()
    d = _imports()[1].core.descale
    shift = {"src_left": src_left, "src_top": src_top}
    if kernel == "bilinear":
        return d.Debilinear(clip, width, height, **shift)
    if kernel == "catrom":
        return d.Debicubic(clip, width, height, b=0, c=0.5, **shift)
    if kernel == "bicubicsharp":
        return d.Debicubic(clip, width, height, b=0, c=0.75, **shift)
    if kernel == "mitchell":
        return d.Debicubic(clip, width, height, b=1 / 3, c=1 / 3, **shift)
    if kernel == "lanczos3":
        return d.Delanczos(clip, width, height, taps=3, **shift)
    if kernel == "spline36":
        return d.Despline36(clip, width, height, **shift)
    raise ValueError(f"unsupported VapourSynth descale kernel: {kernel}")


def upscale_clip(
    clip: Any,
    kernel: str,
    width: int,
    height: int,
    src_left: float = 0.0,
    src_top: float = 0.0,
) -> Any:
    ensure_descale()
    d = _imports()[1].core.descale
    shift = {"src_left": src_left, "src_top": src_top}
    if kernel == "bilinear":
        return d.Bilinear(clip, width, height, **shift)
    if kernel == "catrom":
        return d.Bicubic(clip, width, height, b=0, c=0.5, **shift)
    if kernel == "bicubicsharp":
        return d.Bicubic(clip, width, height, b=0, c=0.75, **shift)
    if kernel == "mitchell":
        return d.Bicubic(clip, width, height, b=1 / 3, c=1 / 3, **shift)
    if kernel == "lanczos3":
        return d.Lanczos(clip, width, height, taps=3, **shift)
    if kernel == "spline36":
        return d.Spline36(clip, width, height, **shift)
    raise ValueError(f"unsupported VapourSynth upscale kernel: {kernel}")


def roundtrip_clip(
    clip: Any,
    kernel: str,
    native_width: int,
    native_height: int,
    src_left: float = 0.0,
    src_top: float = 0.0,
) -> Any:
    native = descale_clip(clip, kernel, native_width, native_height, src_left, src_top)
    return upscale_clip(native, kernel, clip.width, clip.height, src_left, src_top)


def _legacy_score_confidence(best: VSCandidate, height_summary: list[VSCandidate]) -> float:
    by_score = sorted(height_summary, key=lambda item: item.score)
    second = next((item for item in by_score if item.height != best.height), None)
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


def annotate_height_signals(height_summary: list[VSCandidate]) -> list[VSCandidate]:
    ordered = sorted(height_summary, key=lambda item: item.height)
    if len(ordered) < 2:
        return [replace(item, height_signal=0.0) for item in ordered]

    annotated: list[VSCandidate] = []
    for index, item in enumerate(ordered):
        signal = 0.0
        if index == 0:
            next_item = ordered[index + 1]
            if next_item.raw_error > item.raw_error:
                signal = (next_item.raw_error - item.raw_error) / max(item.raw_error, 1e-9)
        elif index < len(ordered) - 1:
            previous = ordered[index - 1]
            next_item = ordered[index + 1]
            previous_drop = previous.raw_error - item.raw_error
            next_drop = item.raw_error - next_item.raw_error
            if previous_drop > 0:
                signal = (
                    previous_drop - max(next_drop, 0.0)
                ) / max(item.raw_error, 1e-9)
        annotated.append(replace(item, height_signal=round(max(0.0, signal), 8)))
    return annotated


def select_height_candidate(height_summary: list[VSCandidate]) -> tuple[VSCandidate, str]:
    ordered = annotate_height_signals(height_summary)
    if not ordered:
        raise ValueError("height summary is empty")

    if len(ordered) > 1 and ordered[0].raw_error <= ordered[1].raw_error:
        return ordered[0], "first_local_minimum"

    knees = [item for item in ordered[1:-1] if item.height_signal >= 0.08]
    if knees:
        return max(knees, key=lambda item: item.height_signal), "native_knee"

    return min(ordered, key=lambda item: item.score), "score_fallback"


def height_confidence(best: VSCandidate, height_summary: list[VSCandidate], method: str) -> float:
    if method == "first_local_minimum":
        return max(_legacy_score_confidence(best, height_summary), 0.8)
    if method == "native_knee":
        raw_quality = max(0.0, min(1.0, (2.0 - best.raw_error) / 2.0))
        return round(max(0.0, min(1.0, best.height_signal * 4.5 + raw_quality * 0.1)), 4)
    return _legacy_score_confidence(best, height_summary)


def _score_candidate(
    source: Any,
    source_arrays: list[Any],
    info: SampleClipInfo,
    height: int,
    kernel: str,
    score_exponent: float,
    src_left: float = 0.0,
    src_top: float = 0.0,
) -> VSCandidate:
    width = max(16, round(info.width * height / info.height))
    reconstruction = roundtrip_clip(source, kernel, width, height, src_left, src_top)
    frame_errors = [
        edge_weighted_mae_array(
            source_arrays[n],
            frame_to_array(reconstruction.get_frame(n)),
        )
        for n in range(info.length)
    ]
    raw_error = float(median(frame_errors))
    score = raw_error * ((height / info.height) ** score_exponent)
    return VSCandidate(
        height=height,
        width=width,
        kernel=kernel,
        score=round(score, 8),
        raw_error=round(raw_error, 8),
        frame_errors=[round(value, 8) for value in frame_errors],
        src_left=src_left,
        src_top=src_top,
    )


def _resolve_candidates(
    candidates: list[VSCandidate],
) -> tuple[
    list[VSCandidate],
    VSCandidate,
    list[VSCandidate],
    list[VSCandidate],
    float,
    float,
    str,
]:
    ranked = sorted(candidates, key=lambda item: item.score)
    raw_height_summary = sorted(
        [
            min(
                (item for item in ranked if item.height == height),
                key=lambda item: item.raw_error,
            )
            for height in sorted({item.height for item in ranked})
        ],
        key=lambda item: item.score,
    )
    selected_height, height_method = select_height_candidate(raw_height_summary)
    height_summary = sorted(
        annotate_height_signals(raw_height_summary),
        key=lambda item: (
            0 if item.height == selected_height.height else 1,
            -item.height_signal,
            item.score,
        ),
    )
    best = next(item for item in height_summary if item.height == selected_height.height)
    kernel_summary = sorted(
        [item for item in ranked if item.height == best.height],
        key=lambda item: item.score,
    )
    height_confidence_value = height_confidence(best, height_summary, height_method)
    kernel_confidence = _kernel_confidence(best, kernel_summary)
    classification = "high_confidence_height" if height_confidence_value >= 0.75 else "needs_review"
    return (
        ranked,
        best,
        height_summary,
        kernel_summary,
        height_confidence_value,
        kernel_confidence,
        classification,
    )


def probe_samples(
    path: Path | str,
    heights: list[int] | tuple[int, ...] = VS_HEIGHTS,
    kernels: list[str] | tuple[str, ...] = VS_KERNELS,
    max_samples: int = 8,
    score_exponent: float = 4.0,
    shifts: list[float] | tuple[float, ...] = VS_SHIFTS,
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
        for kernel in kernels:
            candidates.append(
                _score_candidate(
                    source,
                    source_arrays,
                    info,
                    height,
                    kernel,
                    score_exponent,
                )
            )

    if not candidates:
        raise ValueError("no valid VapourSynth descale candidates were produced")

    (
        ranked,
        best,
        height_summary,
        kernel_summary,
        height_confidence_value,
        kernel_confidence,
        classification,
    ) = _resolve_candidates(candidates)

    shift_values = sorted(set(float(value) for value in shifts))
    if height_confidence_value < 0.75 and any(value != 0.0 for value in shift_values):
        base_scores = {
            height: min(
                (item for item in ranked if item.height == height),
                key=lambda item: item.score,
            )
            for height in sorted({item.height for item in ranked})
        }
        refine_heights = [
            height
            for height in sorted(base_scores)
            if height % 4 != 0 and base_scores[height].score <= best.score * 1.4
        ]
        seen = {
            (item.height, item.kernel, item.src_left, item.src_top)
            for item in candidates
        }
        for height in refine_heights:
            for kernel in kernels:
                for src_left in shift_values:
                    for src_top in shift_values:
                        key = (height, kernel, src_left, src_top)
                        if key in seen:
                            continue
                        candidates.append(
                            _score_candidate(
                                source,
                                source_arrays,
                                info,
                                height,
                                kernel,
                                score_exponent,
                                src_left,
                                src_top,
                            )
                        )
                        seen.add(key)
        (
            ranked,
            best,
            height_summary,
            kernel_summary,
            height_confidence_value,
            kernel_confidence,
            classification,
        ) = _resolve_candidates(candidates)

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
        height_confidence=height_confidence_value,
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
