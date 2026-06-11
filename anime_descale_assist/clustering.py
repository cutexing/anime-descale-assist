from __future__ import annotations

from collections import defaultdict
from statistics import median

from .models import ClusterResult, SampleResult, ZoneResult
from .scoring import median_score


def cluster_samples(samples: list[SampleResult]) -> list[ClusterResult]:
    if not samples:
        return []

    grouped: dict[str, list[SampleResult]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    clusters: list[ClusterResult] = []
    total = len(samples)
    for label, items in grouped.items():
        confidences = [item.confidence for item in items]
        score = median_score(items)
        representatives = sorted(
            items,
            key=lambda item: (item.confidence, -(item.best.score if item.best else 9999)),
            reverse=True,
        )[:5]
        clusters.append(
            ClusterResult(
                label=label,
                classification=items[0].classification,
                count=len(items),
                percentage=round(len(items) * 100 / total, 2),
                median_confidence=round(float(median(confidences)), 4),
                median_score=score,
                representatives=[item.sample_file for item in representatives],
            )
        )

    clusters.sort(key=lambda item: item.count, reverse=True)
    return clusters


def build_zones(samples: list[SampleResult], sample_interval: float | None) -> list[ZoneResult]:
    if not samples:
        return []

    ordered = sorted(samples, key=lambda item: item.sample_index)
    zones: list[ZoneResult] = []
    run_start = ordered[0]
    run_items = [ordered[0]]

    def flush(items: list[SampleResult]) -> None:
        first = items[0]
        last = items[-1]
        confidences = [item.confidence for item in items]
        best_items = [item for item in items if item.best is not None]
        candidate_height = None
        kernel = None
        if first.label != "passthrough_or_unknown" and best_items:
            candidate_height = best_items[0].best.height if best_items[0].best else None
            kernel = best_items[0].best.kernel if best_items[0].best else None

        start_seconds = first.timestamp_seconds
        end_seconds = last.timestamp_seconds
        if end_seconds is not None and sample_interval:
            end_seconds += sample_interval

        action = "descale" if first.classification == "descale_candidate" else "review"
        if first.classification == "passthrough_or_unknown":
            action = "passthrough"

        zones.append(
            ZoneResult(
                label=first.label,
                classification=first.classification,
                action=action,
                sample_start=first.sample_index,
                sample_end=last.sample_index,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                candidate_height=candidate_height,
                kernel=kernel,
                confidence=round(float(median(confidences)), 4),
            )
        )

    for sample in ordered[1:]:
        previous = run_items[-1]
        if sample.label == previous.label and sample.classification == previous.classification:
            run_items.append(sample)
        else:
            flush(run_items)
            run_start = sample
            run_items = [run_start]
    flush(run_items)

    return zones

