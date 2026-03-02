from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from .contracts import MetricRecord, SanityArtifactRecord, SubsystemVerificationResult


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_localization(out_root: Path, run_preset: str = "quick", max_scenes: int = 20) -> SubsystemVerificationResult:
    out_dir = out_root / "localization"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "localization.benchmark.run",
        "--preset",
        run_preset,
        "--max-scenes",
        str(max_scenes),
        "--out-root",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return SubsystemVerificationResult(
            subsystem="localization",
            status="error",
            details={"stderr": proc.stderr[-4000:], "stdout": proc.stdout[-4000:]},
        )

    latest = out_dir / "latest"
    if not latest.exists():
        return SubsystemVerificationResult(subsystem="localization", status="error", details={"error": "missing latest symlink"})

    summary_csv = latest / "summary_by_method.csv"
    rows = _read_csv(summary_csv)
    if not rows:
        return SubsystemVerificationResult(subsystem="localization", status="error", details={"error": "empty summary_by_method.csv"})

    def _mean(field: str) -> float:
        vals = []
        for r in rows:
            s = r.get(field, "")
            if s == "" or s.lower() == "nan":
                continue
            vals.append(float(s))
        return float(sum(vals) / len(vals)) if vals else 0.0

    mae = _mean("mae_deg_matched_mean")
    recall = _mean("recall_mean")
    precision = _mean("precision_mean")

    metrics = [
        MetricRecord("mae_deg_matched_mean", mae, higher_is_better=False, threshold=12.0, passed=mae <= 12.0),
        MetricRecord("recall_mean", recall, higher_is_better=True, threshold=0.80, passed=recall >= 0.80),
        MetricRecord("precision_mean", precision, higher_is_better=True, threshold=0.80, passed=precision >= 0.80),
    ]

    artifacts = [
        SanityArtifactRecord("csv", str(summary_csv)),
        SanityArtifactRecord("markdown", str(latest / "README_summary.md")),
        SanityArtifactRecord("plot", str(latest / "overall_method_comparison.png")),
        SanityArtifactRecord("plot", str(latest / "scene_type_mae_comparison.png")),
        SanityArtifactRecord("plot", str(latest / "k_trends.png")),
    ]

    status = "pass" if all(m.passed for m in metrics) else "warn"
    return SubsystemVerificationResult(
        subsystem="localization",
        status=status,
        metrics=metrics,
        artifacts=artifacts,
        details={"results_dir": str(latest.resolve())},
    )
