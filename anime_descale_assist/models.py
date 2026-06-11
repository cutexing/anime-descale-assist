from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateScore:
    height: int
    width: int
    kernel: str
    score: float
    raw_error: float


@dataclass(frozen=True)
class SampleResult:
    sample_file: str
    sample_index: int
    timestamp_seconds: float | None
    width: int
    height: int
    analysis_width: int
    analysis_height: int
    label: str
    classification: str
    confidence: float
    best: CandidateScore | None
    second: CandidateScore | None
    scores: list[CandidateScore]


@dataclass(frozen=True)
class ClusterResult:
    label: str
    classification: str
    count: int
    percentage: float
    median_confidence: float
    median_score: float | None
    representatives: list[str]


@dataclass(frozen=True)
class ZoneResult:
    label: str
    classification: str
    action: str
    sample_start: int
    sample_end: int
    start_seconds: float | None
    end_seconds: float | None
    candidate_height: int | None
    kernel: str | None
    confidence: float


def to_jsonable(value: Any) -> Any:
    return asdict(value)
