from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import ClusterResult, SampleResult, ZoneResult


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_reports(
    output_dir: Path,
    source: str,
    sample_interval: float | None,
    candidate_heights: list[int],
    kernels: list[str],
    samples: list[SampleResult],
    clusters: list[ClusterResult],
    zones: list[ZoneResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_payload = {
        "schema_version": 1,
        "source": source,
        "sample_interval_seconds": sample_interval,
        "candidate_heights": candidate_heights,
        "kernels": kernels,
        "samples": [asdict(item) for item in samples],
        "clusters": [asdict(item) for item in clusters],
    }
    zones_payload = {
        "schema_version": 1,
        "source": source,
        "sample_interval_seconds": sample_interval,
        "zones": [asdict(item) for item in zones],
    }
    write_json(output_dir / "analysis.json", analysis_payload)
    write_json(output_dir / "zones.json", zones_payload)
    write_summary(output_dir / "summary.md", source, samples, clusters, zones)


def write_summary(
    path: Path,
    source: str,
    samples: list[SampleResult],
    clusters: list[ClusterResult],
    zones: list[ZoneResult],
) -> None:
    lines = [
        "# descale-assist summary",
        "",
        f"Source: `{source}`",
        f"Samples analyzed: **{len(samples)}**",
        "",
        "## Clusters",
        "",
        "| label | class | count | share | median confidence | median score |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for cluster in clusters:
        median_score = "" if cluster.median_score is None else f"{cluster.median_score:.3f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    cluster.label,
                    cluster.classification,
                    str(cluster.count),
                    f"{cluster.percentage:.2f}%",
                    f"{cluster.median_confidence:.3f}",
                    median_score,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Sparse Zones",
            "",
            "| label | action | samples | seconds | confidence |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for zone in zones:
        if zone.start_seconds is None or zone.end_seconds is None:
            seconds = ""
        else:
            seconds = f"{zone.start_seconds:.2f}-{zone.end_seconds:.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    zone.label,
                    zone.action,
                    f"{zone.sample_start}-{zone.sample_end}",
                    seconds,
                    f"{zone.confidence:.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_analysis(work_dir: Path) -> dict[str, Any]:
    return read_json(work_dir / "analysis.json")


def load_zones(work_dir: Path) -> dict[str, Any]:
    return read_json(work_dir / "zones.json")

