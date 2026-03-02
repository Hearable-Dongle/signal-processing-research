from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MetricRecord:
    name: str
    value: float | None
    higher_is_better: bool
    threshold: float | None = None
    passed: bool | None = None


@dataclass(slots=True)
class SanityArtifactRecord:
    kind: str
    path: str
    note: str = ""


@dataclass(slots=True)
class SubsystemVerificationResult:
    subsystem: str
    status: str
    metrics: list[MetricRecord] = field(default_factory=list)
    artifacts: list[SanityArtifactRecord] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationRunSummary:
    timestamp: str
    out_dir: str
    results: list[SubsystemVerificationResult]
    overall_pass: bool
