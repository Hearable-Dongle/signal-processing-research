from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_SINGLE_DOM_ROOT = Path("beamforming/benchmark/testing_specific_angles_single_dom_postloc")
DEFAULT_LOCALIZATION_ONLY_ROOT = Path("realtime_pipeline/output/testing_specific_angles_localization_only_postloc")
DEFAULT_OUT_ROOT = Path("realtime_pipeline/output/postloc_comparison")


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _latest_run(root: Path) -> Path:
    latest = root / "latest"
    if latest.exists():
        return latest.resolve()
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found under {root}")
    return runs[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare post-localization no-separator and localization_only results.")
    parser.add_argument("--single-dom-root", default=str(DEFAULT_SINGLE_DOM_ROOT))
    parser.add_argument("--localization-only-root", default=str(DEFAULT_LOCALIZATION_ONLY_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    single_dom_run = _latest_run(Path(args.single_dom_root))
    localization_only_run = _latest_run(Path(args.localization_only_root))

    single_dom_rows = _read_csv(single_dom_run / "scene_metrics.csv")
    local_rows = _read_csv(localization_only_run / "summary.csv")
    local_by_scene = {str(row["scene_id"]): row for row in local_rows}

    comparison_rows = []
    for row in single_dom_rows:
        scene_id = str(row["recording"]).replace("recording-", "", 1)
        local = local_by_scene.get(scene_id)
        comparison_rows.append(
            {
                "scene_id": scene_id,
                "single_dom_speaker_count_final": row.get("speaker_count_final"),
                "single_dom_active_speakers_avg": row.get("active_speakers_avg"),
                "single_dom_dominant_confidence_avg": row.get("dominant_confidence_avg"),
                "single_dom_fast_rtf": row.get("fast_rtf"),
                "localization_only_final_speakers": local.get("final_speakers") if local else "",
                "localization_only_fast_rtf": local.get("fast_rtf") if local else "",
                "localization_only_slow_rtf": local.get("slow_rtf") if local else "",
                "localization_only_summary_path": local.get("summary_path") if local else "",
            }
        )

    _write_csv(out_root / "comparison.csv", comparison_rows)
    report = {
        "run_id": run_id,
        "single_dominant_root": str(single_dom_run),
        "localization_only_root": str(localization_only_run),
        "comparison_csv": str((out_root / "comparison.csv").resolve()),
        "n_scenes": len(comparison_rows),
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_root.resolve(), target_is_directory=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
