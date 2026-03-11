from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from localization.benchmark.matching import match_predictions
from localization.target_policy import true_target_doas_deg
from simulation.simulation_config import SimulationConfig

from .simulation_runner import run_simulation_pipeline


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_bar(path: Path, rows: list[dict], metric: str, title: str) -> None:
    labels = [f"{r['backend']}\\n{r['tracking_mode']}" for r in rows]
    values = [float(r[metric]) for r in rows]
    plt.figure(figsize=(11, 4))
    plt.bar(np.arange(len(values)), values, width=0.6)
    plt.xticks(np.arange(len(values)), labels, rotation=20)
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_timeline(path: Path, true_doas: list[float], trace: list[dict], title: str) -> None:
    plt.figure(figsize=(11, 4))
    xs: list[float] = []
    ys: list[float] = []
    for frame in trace:
        t = float(frame["frame_index"]) * 0.01
        for item in frame.get("speakers", []):
            xs.append(t)
            ys.append(float(item["direction_degrees"]))
    plt.scatter(xs, ys, s=10, alpha=0.75, label="tracked")
    for idx, doa in enumerate(true_doas):
        plt.axhline(doa, linestyle="--", linewidth=1.2, color=f"C{(idx + 2) % 10}", label=f"GT {idx}: {doa:.1f}")
    plt.ylim(0.0, 360.0)
    plt.xlabel("Time (s)")
    plt.ylabel("DOA (deg)")
    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    seen: set[str] = set()
    uniq_h = []
    uniq_l = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    plt.legend(uniq_h, uniq_l, loc="upper right", ncol=min(3, len(uniq_l)))
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _evaluate_trace(true_doas: list[float], summary: dict) -> dict[str, float]:
    maes: list[float] = []
    churn: list[float] = []
    count_err: list[float] = []
    lifetimes: dict[int, int] = {}
    prev_dirs: list[float] = []
    for frame in summary.get("speaker_map_trace", []):
        dirs = [float(item["direction_degrees"]) for item in frame.get("speakers", [])]
        match = match_predictions(list(true_doas), dirs)
        if match.matched_errors_deg:
            maes.append(float(np.mean(match.matched_errors_deg)))
        count_err.append(float(abs(len(dirs) - len(true_doas))))
        if prev_dirs and dirs:
            jitter = match_predictions(prev_dirs, dirs)
            if jitter.matched_errors_deg:
                churn.append(float(np.mean(jitter.matched_errors_deg)))
        prev_dirs = dirs
        for item in frame.get("speakers", []):
            sid = int(item["speaker_id"])
            lifetimes[sid] = int(lifetimes.get(sid, 0)) + 1
    return {
        "mae_deg": float(np.mean(maes)) if maes else float("nan"),
        "mismatch_rate": float(np.mean(np.asarray(maes, dtype=float) > 15.0)) if maes else float("nan"),
        "map_churn_deg": float(np.mean(churn)) if churn else float("nan"),
        "speaker_count_error": float(np.mean(count_err)) if count_err else float("nan"),
        "mean_track_lifetime_frames": float(np.mean(list(lifetimes.values()))) if lifetimes else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare localization backends and tracking modes on noisy ReSpeaker scenes.")
    p.add_argument(
        "--scenes",
        nargs="+",
        default=[
            "simulation/output/noisy_respeaker_cases_run1/generated_scenes/library_k2_scene19__respeaker_v3_0457__n3.json",
            "simulation/output/noisy_respeaker_cases_run2/generated_scenes/restaurant_k2_scene04__respeaker_v3_0457__n3.json",
        ],
    )
    p.add_argument("--out-dir", default="realtime_pipeline/output/backend_tracking_compare")
    args = p.parse_args()

    scene_paths = [Path(v) for v in args.scenes]
    out_root = Path(args.out_dir)
    backends = ["srp_phat_legacy", "srp_phat_localization", "music_1src"]
    tracking_modes = ["legacy", "multi_peak_v2"]
    rows: list[dict] = []

    for scene_path in scene_paths:
        true_doas = true_target_doas_deg(SimulationConfig.from_file(scene_path))
        for backend in backends:
            for tracking_mode in tracking_modes:
                out_dir = out_root / scene_path.stem / backend / tracking_mode
                summary = run_simulation_pipeline(
                    scene_config_path=scene_path,
                    out_dir=out_dir,
                    use_mock_separation=True,
                    robust_mode=True,
                    capture_trace=True,
                    localization_backend=backend,
                    tracking_mode=tracking_mode,
                )
                metrics = _evaluate_trace(true_doas, summary)
                row = {
                    "scene": scene_path.stem,
                    "backend": backend,
                    "tracking_mode": tracking_mode,
                    **metrics,
                    "out_dir": str(out_dir),
                }
                rows.append(row)
                _plot_timeline(
                    out_dir / "tracked_vs_ground_truth.png",
                    true_doas,
                    summary.get("speaker_map_trace", []),
                    f"{scene_path.stem} :: {backend} :: {tracking_mode}",
                )

    _write_csv(out_root / "summary.csv", rows)
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    for scene in sorted({r["scene"] for r in rows}):
        scene_rows = [r for r in rows if r["scene"] == scene]
        _plot_bar(out_root / scene / "mae_compare.png", scene_rows, "mae_deg", f"{scene}: MAE by backend/tracker")
        _plot_bar(out_root / scene / "mismatch_compare.png", scene_rows, "mismatch_rate", f"{scene}: mismatch rate by backend/tracker")


if __name__ == "__main__":
    main()
