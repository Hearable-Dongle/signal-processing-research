from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from localization.benchmark.algo_runner import _build_algorithm
from localization.target_policy import true_target_doas_deg
from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation


def _load_benchmark_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_deg(angle_deg: float) -> float:
    return float(angle_deg % 360.0)


def _window_iter(mic_audio: np.ndarray, fs: int, window_ms: int, hop_ms: int):
    window_samples = max(1, int(fs * window_ms / 1000))
    hop_samples = max(1, int(fs * hop_ms / 1000))
    total = mic_audio.shape[0]
    for start in range(0, max(1, total - window_samples + 1), hop_samples):
        end = min(total, start + window_samples)
        frame = mic_audio[start:end, :]
        if frame.shape[0] < window_samples:
            frame = np.pad(frame, ((0, window_samples - frame.shape[0]), (0, 0)))
        center_s = (start + frame.shape[0] / 2.0) / fs
        yield center_s, frame


def _collect_method_tracks(
    *,
    scene_path: Path,
    method: str,
    method_cfg: dict,
    window_ms: int,
    hop_ms: int,
) -> tuple[list[dict], list[float]]:
    sim_cfg = SimulationConfig.from_file(scene_path)
    mic_audio, mic_pos_abs, _ = run_simulation(sim_cfg)
    center = np.asarray(sim_cfg.microphone_array.mic_center, dtype=float).reshape(3, 1)
    mic_pos_rel = mic_pos_abs - center
    true_doas = true_target_doas_deg(sim_cfg)

    rows: list[dict] = []
    for time_s, frame in _window_iter(mic_audio, sim_cfg.audio.fs, window_ms, hop_ms):
        algo = _build_algorithm(
            method=method,
            mic_pos_rel=mic_pos_rel,
            fs=sim_cfg.audio.fs,
            n_true_targets=len(true_doas),
            cfg=method_cfg,
        )
        estimated_doas_rad, _, _ = algo.process(frame.T)
        estimated_doas_deg = [_normalize_deg(math.degrees(x)) for x in estimated_doas_rad]
        if not estimated_doas_deg:
            rows.append(
                {
                    "time_s": round(time_s, 3),
                    "method": method,
                    "pred_idx": -1,
                    "pred_doa_deg": "",
                }
            )
            continue
        for pred_idx, pred_deg in enumerate(estimated_doas_deg):
            rows.append(
                {
                    "time_s": round(time_s, 3),
                    "method": method,
                    "pred_idx": int(pred_idx),
                    "pred_doa_deg": round(float(pred_deg), 6),
                }
            )
    return rows, true_doas


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_tracks(path: Path, per_method_rows: dict[str, list[dict]], true_doas: list[float], scene_id: str) -> None:
    methods = list(per_method_rows.keys())
    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 3.4 * len(methods)), sharex=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        rows = per_method_rows[method]
        pred_rows = [r for r in rows if str(r.get("pred_doa_deg", "")).strip()]
        xs = [float(r["time_s"]) for r in pred_rows]
        ys = [float(r["pred_doa_deg"]) for r in pred_rows]
        ax.scatter(xs, ys, s=14, alpha=0.8, label=f"{method} predictions")
        for idx, doa in enumerate(true_doas):
            ax.axhline(doa, color=f"C{(idx + 2) % 10}", linestyle="--", linewidth=1.3, label=f"GT {idx}: {doa:.1f} deg")
        ax.set_ylim(0.0, 360.0)
        ax.set_ylabel("DOA (deg)")
        ax.set_title(method)
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        seen: set[str] = set()
        uniq_handles = []
        uniq_labels = []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            uniq_handles.append(h)
            uniq_labels.append(l)
        ax.legend(uniq_handles, uniq_labels, loc="upper right", ncol=min(3, len(uniq_labels)))

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Predicted DOAs vs ground truth: {scene_id}", y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_pipeline_trace(scene_path: Path, summary_path: Path, out_path: Path, label: str) -> None:
    if not summary_path.exists():
        return
    sim_cfg = SimulationConfig.from_file(scene_path)
    true_doas = true_target_doas_deg(sim_cfg)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    trace = summary.get("speaker_map_trace", [])
    if not trace:
        return

    xs: list[float] = []
    ys: list[float] = []
    for frame in trace:
        t = float(frame["frame_index"]) * 0.01
        for item in frame.get("speakers", []):
            xs.append(t)
            ys.append(float(item["direction_degrees"]))

    plt.figure(figsize=(12, 4))
    plt.scatter(xs, ys, s=10, alpha=0.7, label=f"{label} trace")
    for idx, doa in enumerate(true_doas):
        plt.axhline(doa, color=f"C{(idx + 2) % 10}", linestyle="--", linewidth=1.3, label=f"GT {idx}: {doa:.1f} deg")
    plt.ylim(0.0, 360.0)
    plt.xlabel("Time (s)")
    plt.ylabel("DOA (deg)")
    plt.title(f"Pipeline speaker-map trace vs ground truth: {scene_path.stem} [{label}]")
    handles, labels = plt.gca().get_legend_handles_labels()
    seen: set[str] = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_handles.append(h)
        uniq_labels.append(l)
    plt.legend(uniq_handles, uniq_labels, loc="upper right", ncol=min(3, len(uniq_labels)))
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def generate_timeline_artifacts(
    *,
    scenes: list[Path],
    methods: list[str],
    benchmark_config: Path,
    out_root: Path,
    pipeline_results_root: Path | None = None,
    window_ms: int = 400,
    hop_ms: int = 100,
) -> list[Path]:
    cfg = _load_benchmark_cfg(Path(benchmark_config))
    method_cfgs = cfg["methods"]
    out_paths: list[Path] = []

    for scene_path in scenes:
        scene_id = scene_path.stem
        scene_out = out_root / scene_id
        per_method_rows: dict[str, list[dict]] = {}
        common_true_doas: list[float] | None = None
        all_rows: list[dict] = []
        for method in methods:
            rows, true_doas = _collect_method_tracks(
                scene_path=scene_path,
                method=method,
                method_cfg=method_cfgs[method],
                window_ms=int(window_ms),
                hop_ms=int(hop_ms),
            )
            per_method_rows[method] = rows
            common_true_doas = true_doas
            all_rows.extend(rows)
        _write_csv(scene_out / "timeline_predictions.csv", all_rows)
        plot_path = scene_out / "predictions_vs_ground_truth.png"
        _plot_tracks(plot_path, per_method_rows, common_true_doas or [], scene_id)
        out_paths.append(plot_path)

        if pipeline_results_root is not None:
            pipeline_root = Path(pipeline_results_root) / scene_id
            _plot_pipeline_trace(
                scene_path,
                pipeline_root / "baseline" / "summary.json",
                scene_out / "pipeline_trace_baseline.png",
                "baseline",
            )
            _plot_pipeline_trace(
                scene_path,
                pipeline_root / "robust" / "summary.json",
                scene_out / "pipeline_trace_robust.png",
                "robust",
            )
    return out_paths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot noisy ReSpeaker localization predictions over time.")
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=["SRP-PHAT", "GMDA", "SSZ"])
    p.add_argument("--benchmark-config", default="localization/benchmark/configs/default.json")
    p.add_argument("--window-ms", type=int, default=400)
    p.add_argument("--hop-ms", type=int, default=100)
    p.add_argument("--out-root", default="simulation/output/noisy_respeaker_timeline_plots")
    p.add_argument("--pipeline-results-root", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_paths = generate_timeline_artifacts(
        scenes=[Path(raw_scene) for raw_scene in args.scenes],
        methods=list(args.methods),
        benchmark_config=Path(args.benchmark_config),
        out_root=Path(args.out_root),
        pipeline_results_root=Path(args.pipeline_results_root) if args.pipeline_results_root else None,
        window_ms=int(args.window_ms),
        hop_ms=int(args.hop_ms),
    )
    for path in out_paths:
        print(str(path))


if __name__ == "__main__":
    main()
