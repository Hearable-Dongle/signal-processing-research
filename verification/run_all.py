from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .beamforming_verify import verify_beamforming
from .contracts import SubsystemVerificationResult, VerificationRunSummary
from .integrated_verify import verify_integrated
from .localization_verify import verify_localization
from .speaker_grouping_verify import verify_speaker_grouping
from .speaker_id_verify import verify_speaker_identification


def _write_subsystem_csv(path: Path, results: list[SubsystemVerificationResult]) -> None:
    rows: list[dict[str, object]] = []
    for r in results:
        for m in r.metrics:
            rows.append(
                {
                    "subsystem": r.subsystem,
                    "status": r.status,
                    "metric": m.name,
                    "value": m.value,
                    "higher_is_better": m.higher_is_better,
                    "threshold": m.threshold,
                    "passed": m.passed,
                }
            )
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_manifest(path: Path, results: list[SubsystemVerificationResult]) -> None:
    manifest = {
        r.subsystem: [
            {
                "kind": a.kind,
                "path": a.path,
                "note": a.note,
            }
            for a in r.artifacts
        ]
        for r in results
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _summary_md(path: Path, summary: VerificationRunSummary) -> None:
    lines = ["# Verification Summary", "", f"- timestamp: {summary.timestamp}", f"- overall_pass: {summary.overall_pass}", ""]
    lines.append("| subsystem | status | key_metrics |")
    lines.append("|---|---|---|")
    for r in summary.results:
        km = ", ".join(
            f"{m.name}={m.value:.4f}" if isinstance(m.value, float) else f"{m.name}={m.value}"
            for m in r.metrics[:4]
        )
        lines.append(f"| {r.subsystem} | {r.status} | {km} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_verification(out_root: Path, scene_config: str, quick: bool = True) -> VerificationRunSummary:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (out_root / ts).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    loc = verify_localization(out_dir, run_preset="quick" if quick else "full", max_scenes=20 if quick else 200)
    sid = verify_speaker_identification(out_dir, num_scenes=8 if quick else 20)
    grp = verify_speaker_grouping(out_dir, max_mixtures=20 if quick else 100, chunk_ms=200)
    bf = verify_beamforming(out_dir, scene_config=scene_config)
    integ = verify_integrated(out_dir, scene_config=scene_config)

    results = [loc, sid, grp, bf, integ]
    overall = all(r.status == "pass" for r in results)

    summary = VerificationRunSummary(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        out_dir=str(out_dir),
        results=results,
        overall_pass=overall,
    )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": summary.timestamp,
                "out_dir": summary.out_dir,
                "overall_pass": summary.overall_pass,
                "results": [
                    {
                        "subsystem": r.subsystem,
                        "status": r.status,
                        "metrics": [asdict(m) for m in r.metrics],
                        "artifacts": [asdict(a) for a in r.artifacts],
                        "details": r.details,
                    }
                    for r in summary.results
                ],
            },
            f,
            indent=2,
        )

    _write_subsystem_csv(out_dir / "subsystem_scores.csv", results)
    _write_manifest(out_dir / "artifacts_manifest.json", results)
    _summary_md(out_dir / "README_summary.md", summary)

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run verification suite across subsystems")
    p.add_argument("--out-root", default="verification/output")
    p.add_argument("--scene-config", default="simulation/simulations/configs/library_scene/library_k2_scene00.json")
    p.add_argument("--quick", action="store_true", help="Use quick preset")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = run_all_verification(
        out_root=Path(args.out_root),
        scene_config=args.scene_config,
        quick=bool(args.quick),
    )
    print(json.dumps({"overall_pass": summary.overall_pass, "out_dir": summary.out_dir}, indent=2))


if __name__ == "__main__":
    main()
