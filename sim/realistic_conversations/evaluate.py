from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_frame_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "frame_index": int(row["frame_index"]),
                    "start_time_s": float(row["start_time_s"]),
                    "end_time_s": float(row["end_time_s"]),
                    "active_speaker_ids": [int(v) for v in row["active_speaker_ids"].split("|") if v != ""],
                    "primary_speaker_id": None if row["primary_speaker_id"] in {"", "None"} else int(row["primary_speaker_id"]),
                    "overlap": row["overlap"].lower() == "true",
                    "overlap_count": int(row["overlap_count"]),
                    "frame_snr_db": float(row["frame_snr_db"]),
                }
            )
    return rows


def summarize_scene(scene_dir: str | Path) -> dict[str, Any]:
    scene_path = Path(scene_dir)
    frame_rows = _read_frame_csv(scene_path / "frame_ground_truth.csv")
    metadata = json.loads((scene_path / "scenario_metadata.json").read_text(encoding="utf-8"))

    total_frames = max(1, len(frame_rows))
    overlap_ratio = sum(1 for row in frame_rows if row["overlap"]) / total_frames
    speaker_frames: dict[int, int] = {}
    primary_frames: dict[int, int] = {}
    for row in frame_rows:
        for speaker_id in row["active_speaker_ids"]:
            speaker_frames[speaker_id] = speaker_frames.get(speaker_id, 0) + 1
        if row["primary_speaker_id"] is not None:
            sid = int(row["primary_speaker_id"])
            primary_frames[sid] = primary_frames.get(sid, 0) + 1

    snrs = np.asarray([row["frame_snr_db"] for row in frame_rows], dtype=np.float64)
    summary = {
        "scene_name": metadata["scene_name"],
        "preset": metadata["preset"],
        "speaker_count": int(metadata["speaker_count"]),
        "overlap_ratio": float(overlap_ratio),
        "speaker_activity_frames": {str(k): int(v) for k, v in sorted(speaker_frames.items())},
        "primary_activity_frames": {str(k): int(v) for k, v in sorted(primary_frames.items())},
        "snr_distribution_db": {
            "p10": float(np.percentile(snrs, 10)) if snrs.size else 0.0,
            "median": float(np.percentile(snrs, 50)) if snrs.size else 0.0,
            "p90": float(np.percentile(snrs, 90)) if snrs.size else 0.0,
            "mean": float(np.mean(snrs)) if snrs.size else 0.0,
        },
        "num_overlap_intervals": len(metadata.get("overlap_intervals", [])),
        "num_backchannels": int(metadata.get("summary_metrics", {}).get("num_backchannels", 0)),
    }
    return summary


def summarize_root(root_dir: str | Path) -> dict[str, Any]:
    root = Path(root_dir)
    scene_dirs = sorted(path for path in root.iterdir() if path.is_dir() and (path / "frame_ground_truth.csv").exists())
    scene_summaries = [summarize_scene(path) for path in scene_dirs]
    overlap_values = np.asarray([item["overlap_ratio"] for item in scene_summaries], dtype=np.float64)
    snr_means = np.asarray([item["snr_distribution_db"]["mean"] for item in scene_summaries], dtype=np.float64)
    return {
        "num_scenes": len(scene_summaries),
        "scenes": scene_summaries,
        "aggregate": {
            "overlap_ratio_mean": float(np.mean(overlap_values)) if overlap_values.size else 0.0,
            "snr_mean_db": float(np.mean(snr_means)) if snr_means.size else 0.0,
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate realistic conversation simulation outputs")
    parser.add_argument("--input", required=True, help="Scene directory or root directory containing scene subdirectories")
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    input_path = Path(args.input)
    if (input_path / "frame_ground_truth.csv").exists():
        summary = summarize_scene(input_path)
    else:
        summary = summarize_root(input_path)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
