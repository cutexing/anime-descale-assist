from __future__ import annotations

from statistics import median

from .images import GrayImage, fit_width
from .models import CandidateScore, SampleResult
from .resample import normalize_kernel_name, resize_gray


DEFAULT_HEIGHTS = (540, 576, 648, 720, 756, 810, 864, 900, 936, 960, 1008)
DEFAULT_KERNELS = ("bilinear", "catrom", "bicubicsharp", "lanczos3")


def edge_weighted_mae(source: GrayImage, reconstruction: GrayImage) -> float:
    height = min(len(source), len(reconstruction))
    width = min(len(source[0]), len(reconstruction[0]))
    weighted_error = 0.0
    total_weight = 0.0

    for y in range(1, height - 1):
        prev_row = source[y - 1]
        row = source[y]
        next_row = source[y + 1]
        rec_row = reconstruction[y]
        for x in range(1, width - 1):
            dx = abs(row[x + 1] - row[x - 1])
            dy = abs(next_row[x] - prev_row[x])
            edge = dx + dy
            weight = 1.0 + min(edge / 32.0, 4.0)
            weighted_error += abs(row[x] - rec_row[x]) * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0
    return weighted_error / total_weight


def _candidate_size(
    original_width: int,
    original_height: int,
    analysis_width: int,
    analysis_height: int,
    candidate_height: int,
) -> tuple[int, int] | None:
    if candidate_height >= original_height:
        return None
    scaled_height = round(candidate_height * analysis_height / original_height)
    if scaled_height >= analysis_height or scaled_height < 16:
        return None
    scaled_width = max(16, round(analysis_width * scaled_height / analysis_height))
    return scaled_width, scaled_height


def score_candidate(
    image: GrayImage,
    original_width: int,
    original_height: int,
    candidate_height: int,
    kernel: str,
) -> tuple[CandidateScore, GrayImage]:
    analysis_height = len(image)
    analysis_width = len(image[0])
    size = _candidate_size(
        original_width,
        original_height,
        analysis_width,
        analysis_height,
        candidate_height,
    )
    if size is None:
        raise ValueError("candidate is not valid for this source size")

    candidate_width, analysis_candidate_height = size
    down = resize_gray(image, candidate_width, analysis_candidate_height, kernel)
    reconstruction = resize_gray(down, analysis_width, analysis_height, kernel)
    raw_error = edge_weighted_mae(image, reconstruction)
    # Plain down/up round-trip error unfairly favors candidates close to source
    # height because they discard less information. The MVP reports a
    # complexity-normalized signature score, while keeping raw_error in JSON.
    reduction_ratio = candidate_height / original_height
    score = raw_error * (reduction_ratio**2)
    native_width = max(16, round(original_width * candidate_height / original_height))
    return (
        CandidateScore(
            height=candidate_height,
            width=native_width,
            kernel=normalize_kernel_name(kernel),
            score=round(score, 6),
            raw_error=round(raw_error, 6),
        ),
        reconstruction,
    )


def confidence_from_scores(scores: list[CandidateScore]) -> float:
    if len(scores) < 2:
        return 0.0
    best = scores[0].score
    second = scores[1].score
    if second <= 1e-9:
        return 0.0
    separation = max(0.0, (second - best) / second)
    absolute_quality = max(0.0, min(1.0, (18.0 - best) / 18.0))
    return round(max(0.0, min(1.0, separation * 1.7 + absolute_quality * 0.25)), 4)


def classify(confidence: float, best_score: float | None) -> str:
    if best_score is None:
        return "passthrough_or_unknown"
    if confidence >= 0.45 and best_score <= 16.0:
        return "descale_candidate"
    if confidence >= 0.25 and best_score <= 22.0:
        return "needs_review"
    return "passthrough_or_unknown"


def analyze_image(
    image: GrayImage,
    sample_file: str,
    sample_index: int,
    timestamp_seconds: float | None,
    original_width: int,
    original_height: int,
    candidate_heights: list[int],
    kernels: list[str],
    max_analysis_width: int,
) -> tuple[SampleResult, GrayImage | None]:
    analysis_width, analysis_height = fit_width(original_width, original_height, max_analysis_width)
    if analysis_width != original_width or analysis_height != original_height:
        analysis = resize_gray(image, analysis_width, analysis_height, "bilinear")
    else:
        analysis = [row[:] for row in image]

    scores: list[CandidateScore] = []
    reconstructions: dict[tuple[int, str], GrayImage] = {}
    for height in candidate_heights:
        for kernel in kernels:
            try:
                candidate_score, reconstruction = score_candidate(
                    analysis,
                    original_width,
                    original_height,
                    height,
                    kernel,
                )
            except ValueError:
                continue
            scores.append(candidate_score)
            reconstructions[(candidate_score.height, candidate_score.kernel)] = reconstruction

    scores.sort(key=lambda item: item.score)
    best = scores[0] if scores else None
    second = scores[1] if len(scores) > 1 else None
    confidence = confidence_from_scores(scores)
    classification = classify(confidence, best.score if best else None)
    if best and classification != "passthrough_or_unknown":
        label = f"h{best.height}_{best.kernel}"
    else:
        label = "passthrough_or_unknown"

    preview = reconstructions.get((best.height, best.kernel)) if best else None
    return (
        SampleResult(
            sample_file=sample_file,
            sample_index=sample_index,
            timestamp_seconds=timestamp_seconds,
            width=original_width,
            height=original_height,
            analysis_width=analysis_width,
            analysis_height=analysis_height,
            label=label,
            classification=classification,
            confidence=confidence,
            best=best,
            second=second,
            scores=scores,
        ),
        preview,
    )


def median_score(samples: list[SampleResult]) -> float | None:
    values = [sample.best.score for sample in samples if sample.best is not None]
    if not values:
        return None
    return round(float(median(values)), 6)
