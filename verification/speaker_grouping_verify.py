from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .contracts import MetricRecord, SanityArtifactRecord, SubsystemVerificationResult


def verify_speaker_grouping(out_root: Path, max_mixtures: int = 25, chunk_ms: int = 200) -> SubsystemVerificationResult:
    out_dir = out_root / "speaker_grouping"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "speaker_identity_grouping.validate",
        "--max-mixtures",
        str(max_mixtures),
        "--chunk-ms",
        str(chunk_ms),
        "--out-dir",
        str(out_dir),
        "--device",
        "cpu",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return SubsystemVerificationResult(
            subsystem="speaker_grouping",
            status="error",
            details={"stderr": proc.stderr[-4000:], "stdout": proc.stdout[-4000:]},
        )

    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return SubsystemVerificationResult(subsystem="speaker_grouping", status="error", details={"error": "missing summary.json"})

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    overall = summary.get("overall_metrics", {})

    acc = float(overall.get("majority_vote_accuracy", 0.0))
    switch = float(overall.get("switch_rate", 1.0))
    ratio = float(overall.get("speaker_count_ratio", 0.0))
    rtf = float(overall.get("realtime_factor_total", 1e9))

    metrics = [
        MetricRecord("majority_vote_accuracy", acc, True, 0.88, acc >= 0.88),
        MetricRecord("switch_rate", switch, False, 0.15, switch <= 0.15),
        MetricRecord("speaker_count_ratio", ratio, True, None, 0.8 <= ratio <= 1.25),
        MetricRecord("realtime_factor_total", rtf, False, 1.0, rtf <= 1.0),
    ]

    artifacts = [
        SanityArtifactRecord("json", str(summary_path)),
        SanityArtifactRecord("csv", str(out_dir / "per_mixture_metrics.csv")),
        SanityArtifactRecord("csv", str(out_dir / "pair_rows.csv")),
    ]

    status = "pass" if all(m.passed for m in metrics) else "warn"
    return SubsystemVerificationResult(
        subsystem="speaker_grouping",
        status=status,
        metrics=metrics,
        artifacts=artifacts,
        details={"results_dir": str(out_dir.resolve())},
    )
