from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_SCENES_ROOT = Path("simulation/simulations/configs/testing_specific_angles")
DEFAULT_COLLECTION_ROOT = Path("beamforming/benchmark/testing_specific_angles_simulated_collection_full")
DEFAULT_OUT_ROOT = Path("beamforming/benchmark/testing_specific_angles_single_dom_postloc")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _load_scene_metrics(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_ground_truth(recording_dir: Path) -> list[dict]:
    metadata_path = recording_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    scene_cfg = Path(str(payload["sceneConfigPath"]))
    metadata = scene_cfg.parent.parent / "assets" / scene_cfg.stem / "scenario_metadata.json"
    if not metadata.exists():
        metadata = Path("simulation/simulations/assets/testing_specific_angles") / scene_cfg.stem / "scenario_metadata.json"
    if not metadata.exists():
        return []
    scene_meta = json.loads(metadata.read_text(encoding="utf-8"))
    return list(scene_meta.get("assets", {}).get("speech", []))


def _interpret_recording(row: dict, recording_dir: Path) -> dict:
    gt = _load_ground_truth(recording_dir)
    expected_turns = len(gt)
    final_count = int(float(row.get("speaker_count_final", 0)))
    active_final = int(float(row.get("active_speaker_count_final", 0)))
    dominant_conf = float(row.get("dominant_confidence_avg", 0.0))
    tracked_avg = float(row.get("tracked_speakers_avg", 0.0))
    active_avg = float(row.get("active_speakers_avg", 0.0))
    assessment = "ok"
    if final_count == 0 or active_final == 0:
        assessment = "no_final_speaker"
    elif dominant_conf < 0.35:
        assessment = "low_confidence"
    elif tracked_avg > 1.5 and expected_turns == 2:
        assessment = "too_many_tracks"
    return {
        "recording": row["recording"],
        "expected_turn_speakers": expected_turns,
        "speaker_count_final": final_count,
        "active_speaker_count_final": active_final,
        "tracked_speakers_avg": tracked_avg,
        "active_speakers_avg": active_avg,
        "dominant_confidence_avg": dominant_conf,
        "assessment": assessment,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single_dominant_no_separator benchmark on testing_specific_angles.")
    parser.add_argument("--scenes-root", default=str(DEFAULT_SCENES_ROOT))
    parser.add_argument("--collection-root", default=str(DEFAULT_COLLECTION_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--methods", nargs="+", default=["mvdr_fd"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--mic-array-profile", choices=["respeaker_v3_0457", "respeaker_cross_0640", "respeaker_xvf3800_0650"], default="respeaker_v3_0457")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    collection_root = Path(args.collection_root)
    if args.force_export or not (collection_root / "collection.json").exists():
        _run(
            [
                sys.executable,
                "-m",
                "beamforming.benchmark.export_simulation_to_collection",
                "--input-path",
                str(Path(args.scenes_root)),
                "--out-dir",
                str(collection_root),
                *(
                    ["--scene-ids", *list(args.scene_ids)]
                    if args.scene_ids
                    else (["--max-scenes", str(int(args.max_scenes))] if args.max_scenes is not None else [])
                ),
            ]
        )

    _run(
        [
            sys.executable,
            "-m",
            "beamforming.benchmark.data_collection_benchmark",
            "--input-path",
            str(collection_root),
            "--out-dir",
            str(out_dir),
            "--methods",
            *list(args.methods),
            "--separation-mode",
            "single_dominant_no_separator",
            "--control-mode",
            "speaker_tracking_mode",
            "--fast-path-reference-mode",
            "speaker_map",
            "--mic-array-profile",
            str(args.mic_array_profile),
            "--workers",
            str(int(args.workers)),
        ]
    )

    scene_metrics = _load_scene_metrics(out_dir / "scene_metrics.csv")
    recording_root = collection_root / "recordings"
    interpretation_rows = [
        _interpret_recording(row, recording_root / str(row["recording"]))
        for row in scene_metrics
    ]
    _write_csv(out_dir / "interpretation_by_recording.csv", interpretation_rows)

    summary = {
        "run_id": run_id,
        "collection_root": str(collection_root.resolve()),
        "benchmark_root": str(out_dir.resolve()),
        "methods": list(args.methods),
        "mic_array_profile": str(args.mic_array_profile),
        "n_recordings": len(scene_metrics),
        "n_ok": sum(1 for row in interpretation_rows if row["assessment"] == "ok"),
        "n_problematic": sum(1 for row in interpretation_rows if row["assessment"] != "ok"),
        "interpretation_csv": str((out_dir / "interpretation_by_recording.csv").resolve()),
    }
    (out_dir / "postloc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir.resolve(), target_is_directory=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
