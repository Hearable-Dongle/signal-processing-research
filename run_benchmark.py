from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from realtime_pipeline.localization_strategies.ipd_regressor import CHECKPOINT_PATH, ensure_trained
from realtime_pipeline.simulation_runner import run_simulation_pipeline


SCENES_ROOT = Path("simulation/simulations/configs/testing_specific_angles")
ASSETS_ROOT = Path("simulation/simulations/assets/testing_specific_angles")
OUTPUT_ROOT = Path("realtime_pipeline/output/localization_benchmark")
BASELINE_STRATEGY = "baseline_existing_model"
DEFAULT_STRATEGIES = [
    "snr_weighted_srp_phat",
    "peak_confidence_srp_phat",
    "particle_filter_tracker",
    "neural_mask_gcc_phat",
    "ipd_regressor",
]
BASELINE_CANDIDATES = [
    Path("realtime_pipeline/run_testing_specific_angles_localization_only.py"),
    Path("verification/localization_verify.py"),
    Path("localization/benchmark/run.py"),
]


@dataclass(frozen=True)
class SceneResult:
    strategy: str
    scene_id: str
    truth_deg: list[float]
    pred_deg: list[float]
    errors_deg: list[float]
    gating_rate: float | None
    fast_avg_ms: float
    fast_rtf: float
    out_dir: str


def _read_frame_truth(scene_id: str) -> tuple[list[float], list[float]]:
    path = ASSETS_ROOT / scene_id / "frame_ground_truth.csv"
    scene_cfg = json.loads((SCENES_ROOT / f"{scene_id}.json").read_text(encoding="utf-8"))
    mic_center = np.asarray(scene_cfg["microphone_array"]["mic_center"][:2], dtype=np.float64)
    truth: list[float] = []
    time_s: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            start_time = float(row.get("start_time_s", row.get("time_sec", 0.0)))
            end_time = float(row.get("end_time_s", start_time + 0.02))
            speaker_positions = row.get("speaker_positions", "")
            primary_id = str(row.get("primary_speaker_id", "")).strip()
            doa = float("nan")
            if speaker_positions and primary_id:
                pos_map = json.loads(speaker_positions)
                if primary_id in pos_map:
                    pos = np.asarray(pos_map[primary_id][:2], dtype=np.float64)
                    delta = pos - mic_center
                    doa = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)
            time_s.extend([start_time, start_time + ((end_time - start_time) * 0.5)])
            truth.extend([doa, doa])
    return time_s, truth


def _angular_error_deg(pred: float, truth: float) -> float:
    return float(abs((float(pred) - float(truth) + 180.0) % 360.0 - 180.0))


def _stable_segments(truth_deg: list[float]) -> list[slice]:
    arr = np.asarray(truth_deg, dtype=np.float64)
    segments: list[slice] = []
    start: int | None = None
    for idx, value in enumerate(arr):
        if not np.isfinite(value):
            if start is not None and idx - start >= 2:
                segments.append(slice(start, idx))
            start = None
            continue
        if start is None:
            start = idx
            continue
        if abs(_angular_error_deg(float(arr[idx - 1]), float(value))) > 1e-6:
            if idx - start >= 2:
                segments.append(slice(start, idx))
            start = idx
    if start is not None and arr.size - start >= 2:
        segments.append(slice(start, arr.size))
    return segments


def _prediction_from_summary(summary: dict, n_frames: int) -> tuple[list[float], float | None]:
    pred = np.full(n_frames, np.nan, dtype=np.float64)
    gated = 0
    active = 0
    for row in summary.get("srp_trace", []):
        idx = int(row["frame_index"])
        if idx >= n_frames:
            continue
        peaks = row.get("peaks_deg", [])
        if peaks:
            pred[idx] = float(peaks[0])
        debug = row.get("debug", {}) or {}
        if "gated" in debug:
            active += 1
            gated += 1 if bool(debug.get("gated")) else 0
    gating_rate = None if active == 0 else float(gated / active)
    return pred.tolist(), gating_rate


def _metrics(errors_deg: list[float], pred_deg: list[float], truth_deg: list[float], gating_rate: float | None, runtime_ms: list[float]) -> dict:
    arr = np.asarray(errors_deg, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    segments = _stable_segments(truth_deg)
    pred_arr = np.asarray(pred_deg, dtype=np.float64)
    stability_vals = []
    for seg in segments:
        block = pred_arr[seg]
        block = block[np.isfinite(block)]
        if block.size >= 2:
            stability_vals.append(float(np.std(block)))
    return {
        "mae_deg": float(np.mean(valid)) if valid.size else float("nan"),
        "acc_at_10": float(np.mean(valid <= 10.0)) if valid.size else float("nan"),
        "acc_at_20": float(np.mean(valid <= 20.0)) if valid.size else float("nan"),
        "median_deg": float(np.median(valid)) if valid.size else float("nan"),
        "p90_deg": float(np.percentile(valid, 90)) if valid.size else float("nan"),
        "stability_deg": float(np.mean(stability_vals)) if stability_vals else float("nan"),
        "gating_rate": gating_rate,
        "fast_avg_ms": float(np.mean(runtime_ms)) if runtime_ms else float("nan"),
    }


def _resolve_baseline_script() -> str:
    existing = [path for path in BASELINE_CANDIDATES if path.exists()]
    existing.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(existing[0].resolve()) if existing else ""


def _run_scene(strategy: str, scene_path: Path, run_root: Path) -> SceneResult:
    scene_id = scene_path.stem
    out_dir = run_root / strategy / scene_id
    summary = run_simulation_pipeline(
        scene_config_path=scene_path,
        out_dir=out_dir,
        use_mock_separation=True,
        capture_trace=True,
        robust_mode=True,
        localization_backend=strategy,
        control_mode="spatial_peak_mode",
        fast_path_reference_mode="srp_peak",
        direction_long_memory_enabled=False,
    )
    _time_s, truth_deg = _read_frame_truth(scene_id)
    pred_deg, gating_rate = _prediction_from_summary(summary, len(truth_deg))
    errors_deg = [
        _angular_error_deg(pred, truth) if np.isfinite(pred) and np.isfinite(truth) else float("nan")
        for pred, truth in zip(pred_deg, truth_deg)
    ]
    return SceneResult(
        strategy=strategy,
        scene_id=scene_id,
        truth_deg=truth_deg,
        pred_deg=pred_deg,
        errors_deg=errors_deg,
        gating_rate=gating_rate,
        fast_avg_ms=float(summary.get("fast_avg_ms", float("nan"))),
        fast_rtf=float(summary.get("fast_rtf", float("nan"))),
        out_dir=str(out_dir.resolve()),
    )


def _run_baseline_script(
    baseline_script: str,
    *,
    out_root: Path,
    scenes_root: Path,
    max_scenes: int | None,
    profile: str,
) -> list[SceneResult]:
    script_path = Path(baseline_script)
    baseline_root = out_root / BASELINE_STRATEGY
    if script_path.name == "run_testing_specific_angles_localization_only.py":
        cmd = [
            sys.executable,
            "-m",
            "realtime_pipeline.run_testing_specific_angles_localization_only",
            "--scenes-root",
            str(scenes_root),
            "--out-root",
            str(baseline_root),
            "--mock-separation",
            "--profile",
            str(profile),
        ]
    else:
        cmd = [
            sys.executable,
            str(script_path),
            "--scenes-root",
            str(scenes_root),
            "--out-root",
            str(baseline_root),
            "--mock-separation",
        ]
    if max_scenes is not None:
        cmd.extend(["--max-scenes", str(int(max_scenes))])
    subprocess.run(cmd, check=True)
    latest = baseline_root / "latest"
    if not latest.exists():
        raise FileNotFoundError(f"Baseline run did not produce latest under {baseline_root}")
    summary_payload = json.loads((latest.resolve() / "summary.json").read_text(encoding="utf-8"))
    scene_results: list[SceneResult] = []
    for row in summary_payload.get("rows", []):
        scene_id = str(row["scene_id"])
        summary = json.loads(Path(str(row["summary_path"])).read_text(encoding="utf-8"))
        _time_s, truth_deg = _read_frame_truth(scene_id)
        pred_deg, gating_rate = _prediction_from_summary(summary, len(truth_deg))
        errors_deg = [
            _angular_error_deg(pred, truth) if np.isfinite(pred) and np.isfinite(truth) else float("nan")
            for pred, truth in zip(pred_deg, truth_deg)
        ]
        scene_results.append(
            SceneResult(
                strategy=BASELINE_STRATEGY,
                scene_id=scene_id,
                truth_deg=truth_deg,
                pred_deg=pred_deg,
                errors_deg=errors_deg,
                gating_rate=gating_rate,
                fast_avg_ms=float(summary.get("fast_avg_ms", float("nan"))),
                fast_rtf=float(summary.get("fast_rtf", float("nan"))),
                out_dir=str(Path(str(row["summary_path"])).parent.resolve()),
            )
        )
    return scene_results


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_bars(out_dir: Path, summary_rows: list[dict]) -> None:
    labels = [row["strategy"] for row in summary_rows]
    metric_map = {
        "mae_by_strategy.png": ("mae_deg", "MAE (deg)"),
        "acc10_by_strategy.png": ("acc_at_10", "Acc@10"),
        "stability_by_strategy.png": ("stability_deg", "Stability (deg std)"),
        "runtime_by_strategy.png": ("fast_avg_ms", "Fast path avg ms"),
    }
    for filename, (field, ylabel) in metric_map.items():
        values = [float(row[field]) if row[field] is not None else float("nan") for row in summary_rows]
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(labels)), values)
        plt.xticks(np.arange(len(labels)), labels, rotation=20)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()
    gating = [float(row["gating_rate"]) if row["gating_rate"] is not None else 0.0 for row in summary_rows]
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(labels)), gating)
    plt.xticks(np.arange(len(labels)), labels, rotation=20)
    plt.ylabel("Gating rate")
    plt.tight_layout()
    plt.savefig(out_dir / "gating_rate_by_strategy.png", dpi=180)
    plt.close()


def _plot_issue_timelines(out_dir: Path, scene_results: list[SceneResult]) -> None:
    by_strategy: dict[str, list[SceneResult]] = {}
    for result in scene_results:
        by_strategy.setdefault(result.strategy, []).append(result)
    issue_dir = out_dir / "timeline_worst_scenes"
    issue_dir.mkdir(parents=True, exist_ok=True)
    for strategy, rows in by_strategy.items():
        worst = max(
            rows,
            key=lambda row: float(np.nanmean(np.asarray(row.errors_deg, dtype=np.float64)))
            if np.isfinite(np.nanmean(np.asarray(row.errors_deg, dtype=np.float64)))
            else -1.0,
        )
        xs = np.arange(len(worst.truth_deg), dtype=np.float64) * 0.01
        plt.figure(figsize=(12, 4))
        plt.plot(xs, worst.truth_deg, label="truth", linewidth=2.0)
        plt.plot(xs, worst.pred_deg, label="pred", linewidth=1.4, alpha=0.85)
        plt.ylim(-5.0, 365.0)
        plt.xlabel("Time (s)")
        plt.ylabel("DOA (deg)")
        plt.title(f"{strategy} worst scene: {worst.scene_id}")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(issue_dir / f"{strategy}__{worst.scene_id}.png", dpi=180)
        plt.close()


def _format_table(rows: list[dict]) -> str:
    lines = [
        "Strategy                  | MAE   | Acc@10 | Acc@20 | Median | P90  | Stability | GatingRate",
        "--------------------------|-------|--------|--------|--------|------|-----------|-----------",
    ]
    for row in rows:
        gating = "N/A" if row["gating_rate"] is None else f"{100.0 * float(row['gating_rate']):.1f}%"
        label = "baseline (existing model)" if row["strategy"] == BASELINE_STRATEGY else row["strategy"]
        lines.append(
            f"{label:<26}| {row['mae_deg']:.1f} | {100.0 * row['acc_at_10']:.1f}% | {100.0 * row['acc_at_20']:.1f}% | {row['median_deg']:.1f} | {row['p90_deg']:.1f} | {row['stability_deg']:.1f} | {gating}"
        )
    return "\n".join(lines)


def _aggregate(scene_results: list[SceneResult], strategies: list[str]) -> tuple[list[dict], dict[str, list[float]]]:
    grouped: dict[str, list[SceneResult]] = {}
    for result in scene_results:
        grouped.setdefault(result.strategy, []).append(result)
    summary_rows: list[dict] = []
    error_arrays: dict[str, list[float]] = {}
    for strategy in [BASELINE_STRATEGY, *strategies]:
        rows = grouped.get(strategy, [])
        all_errors = [value for row in rows for value in row.errors_deg]
        all_pred = [value for row in rows for value in row.pred_deg]
        all_truth = [value for row in rows for value in row.truth_deg]
        gating_values = [row.gating_rate for row in rows if row.gating_rate is not None]
        runtimes = [row.fast_avg_ms for row in rows]
        metrics = _metrics(
            all_errors,
            all_pred,
            all_truth,
            float(np.mean(gating_values)) if gating_values else None,
            runtimes,
        )
        summary_rows.append({"strategy": strategy, **metrics})
        error_arrays[strategy] = [float(v) if np.isfinite(v) else float("nan") for v in all_errors]
    return summary_rows, error_arrays


def _write_scene_csv(path: Path, rows: list[SceneResult]) -> None:
    flat_rows = []
    for row in rows:
        valid = np.asarray(row.errors_deg, dtype=np.float64)
        valid = valid[np.isfinite(valid)]
        flat_rows.append(
            {
                "strategy": row.strategy,
                "scene_id": row.scene_id,
                "mae_deg": float(np.mean(valid)) if valid.size else float("nan"),
                "acc_at_10": float(np.mean(valid <= 10.0)) if valid.size else float("nan"),
                "fast_avg_ms": row.fast_avg_ms,
                "fast_rtf": row.fast_rtf,
                "gating_rate": row.gating_rate if row.gating_rate is not None else "",
                "out_dir": row.out_dir,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark localization strategies on testing_specific_angles scenes.")
    parser.add_argument("--scenes-root", default=str(SCENES_ROOT))
    parser.add_argument("--out-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--skip-ipd-train", action="store_true")
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--baseline-script", default=None)
    parser.add_argument("--profile", default="respeaker_xvf3800_0650", choices=["respeaker_v3_0457", "respeaker_xvf3800_0650"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    scenes = sorted(Path(args.scenes_root).glob("*.json"))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    if not scenes:
        raise FileNotFoundError(f"No scenes found under {args.scenes_root}")

    strategies = list(args.strategies or DEFAULT_STRATEGIES)
    if not args.skip_ipd_train and "ipd_regressor" in strategies:
        ensure_trained(fs=16000, nfft=256, overlap=0.5)
    elif args.skip_ipd_train and "ipd_regressor" in strategies and not CHECKPOINT_PATH.exists():
        strategies = [name for name in strategies if name != "ipd_regressor"]

    baseline_script = str(args.baseline_script) if args.baseline_script else _resolve_baseline_script()
    scene_results: list[SceneResult] = _run_baseline_script(
        baseline_script,
        out_root=out_root,
        scenes_root=Path(args.scenes_root),
        max_scenes=args.max_scenes,
        profile=str(args.profile),
    )
    jobs = [(strategy, scene) for strategy in strategies for scene in scenes]
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_map = {pool.submit(_run_scene, strategy, scene, out_root): (strategy, scene.stem) for strategy, scene in jobs}
        for future in as_completed(future_map):
            strategy, scene_id = future_map[future]
            print(f"completed {strategy} :: {scene_id}")
            scene_results.append(future.result())

    summary_rows, error_arrays = _aggregate(scene_results, strategies)
    summary_rows.sort(key=lambda row: [BASELINE_STRATEGY, *strategies].index(row["strategy"]))
    report_text = _format_table(summary_rows)
    print(report_text)

    _plot_bars(out_root, summary_rows)
    _plot_issue_timelines(out_root, scene_results)
    _write_scene_csv(out_root / "scene_metrics.csv", scene_results)

    payload = {
        "run_id": run_id,
        "baseline_script": baseline_script,
        "profile": str(args.profile),
        "scenes_root": str(Path(args.scenes_root).resolve()),
        "n_scenes": len(scenes),
        "strategies": summary_rows,
        "per_frame_errors_deg": error_arrays,
    }
    _write_json(out_root / "localization_benchmark_results.json", payload)
    (out_root / "localization_benchmark_results.txt").write_text(report_text + "\n", encoding="utf-8")

    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_root.resolve(), target_is_directory=True)


if __name__ == "__main__":
    main()
