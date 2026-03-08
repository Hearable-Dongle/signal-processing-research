from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from localization.benchmark.algo_runner import run_method_on_scene
from localization.benchmark.matching import match_predictions
from localization.benchmark.metrics import compute_scene_metrics
from localization.target_policy import true_target_doas_deg
from realtime_pipeline.simulation_runner import run_simulation_pipeline
from simulation.mic_array_profiles import mic_positions_xyz
from simulation.plot_noisy_respeaker_timelines import generate_timeline_artifacts
from simulation.simulation_config import MicrophoneArray, SimulationConfig, SimulationSource
from simulation.target_policy import is_speech_target


def _load_benchmark_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _available_noise_paths(sim_cfg: SimulationConfig) -> list[str]:
    seen = {str(src.audio_path) for src in sim_cfg.audio.sources}
    root = Path("/home/mkeller/data/librimix/wham_noise/tr")
    if not root.exists():
        return []
    paths: list[str] = []
    for wav in sorted(root.rglob("*.wav")):
        rel = str(wav.relative_to(root.parent.parent))
        if rel not in seen:
            paths.append(rel)
    return paths


def _room_position(
    room_dims: list[float],
    center: list[float],
    angle_deg: float,
    radius_scale: float,
    z: float,
) -> list[float]:
    max_radius = max(0.4, min(float(room_dims[0]), float(room_dims[1])) * 0.5 - 0.45)
    radius = min(max_radius, max_radius * radius_scale)
    theta = np.deg2rad(angle_deg)
    x = float(center[0]) + radius * float(np.cos(theta))
    y = float(center[1]) + radius * float(np.sin(theta))
    x = float(np.clip(x, 0.25, float(room_dims[0]) - 0.25))
    y = float(np.clip(y, 0.25, float(room_dims[1]) - 0.25))
    z = float(np.clip(z, 0.4, float(room_dims[2]) - 0.25))
    return [round(x, 3), round(y, 3), round(z, 3)]


def build_noisy_respeaker_variant(
    scene_path: Path,
    *,
    out_dir: Path,
    profile: str,
    noise_count: int,
    seed: int,
) -> Path:
    rng = random.Random(seed)
    sim_cfg = SimulationConfig.from_file(scene_path)

    rel_positions = mic_positions_xyz(profile)
    sim_cfg.microphone_array = MicrophoneArray(
        mic_center=list(sim_cfg.microphone_array.mic_center),
        mic_radius=float(np.max(np.linalg.norm(rel_positions[:, :2], axis=1))),
        mic_count=int(rel_positions.shape[0]),
        mic_positions=rel_positions.tolist(),
    )

    noise_pool = _available_noise_paths(sim_cfg)
    if len(noise_pool) < noise_count:
        raise RuntimeError(f"Only found {len(noise_pool)} candidate noise files, need {noise_count}")

    speech_sources = [src for src in sim_cfg.audio.sources if is_speech_target(src)]
    for src in sim_cfg.audio.sources:
        if not is_speech_target(src):
            src.gain = round(float(src.gain) * 1.8, 3)
            src.classification = "noise"

    if speech_sources:
        speech_angles = sorted(true_target_doas_deg(sim_cfg))
    else:
        speech_angles = []

    candidate_angles = [35.0, 110.0, 195.0, 285.0, 330.0]
    chosen_noise = rng.sample(noise_pool, noise_count)
    for idx, noise_audio in enumerate(chosen_noise):
        angle = candidate_angles[idx % len(candidate_angles)]
        if speech_angles:
            angle = min(
                candidate_angles,
                key=lambda a: min(abs(((a - s + 180.0) % 360.0) - 180.0) for s in speech_angles),
            )
            candidate_angles = [a for a in candidate_angles if a != angle] or candidate_angles
        gain = [0.35, 0.6, 0.9, 0.5, 0.75][idx % 5]
        loc = _room_position(
            sim_cfg.room.dimensions,
            sim_cfg.microphone_array.mic_center,
            angle_deg=angle,
            radius_scale=0.78 + 0.08 * idx,
            z=0.7 + 0.45 * (idx % 3),
        )
        sim_cfg.audio.sources.append(
            SimulationSource(
                loc=loc,
                audio_path=noise_audio,
                gain=gain,
                classification="noise",
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_path.stem}__{profile}__n{noise_count}.json"
    sim_cfg.to_file(out_path)
    return out_path


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_method_mae(path: Path, rows: list[dict]) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("mae_deg_matched") is not None]
    if not ok_rows:
        return
    plt.figure(figsize=(8, 4))
    labels = [str(r["method"]) for r in ok_rows]
    values = [float(r["mae_deg_matched"]) for r in ok_rows]
    plt.bar(np.arange(len(labels)), values, width=0.6)
    plt.xticks(np.arange(len(labels)), labels, rotation=20)
    plt.ylabel("Matched MAE (deg)")
    plt.title("Localization stress results")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close()


def _pipeline_trace_metrics(scene_path: Path, mode_out: Path, summary: dict) -> dict[str, float]:
    true_doas = true_target_doas_deg(SimulationConfig.from_file(scene_path))
    frame_maes: list[float] = []
    churn_values: list[float] = []
    prev_dirs: list[float] = []
    trace_rows: list[dict] = []
    for frame in summary.get("speaker_map_trace", []):
        dirs = [float(item["direction_degrees"]) for item in frame.get("speakers", [])]
        match = match_predictions(list(true_doas), dirs)
        mae = float(np.mean(match.matched_errors_deg)) if match.matched_errors_deg else float("nan")
        if np.isfinite(mae):
            frame_maes.append(mae)
        churn = float("nan")
        if prev_dirs and dirs:
            jitter_match = match_predictions(prev_dirs, dirs)
            if jitter_match.matched_errors_deg:
                churn = float(np.mean(jitter_match.matched_errors_deg))
                churn_values.append(churn)
        prev_dirs = dirs
        trace_rows.append(
            {
                "frame_index": int(frame["frame_index"]),
                "num_speakers": len(dirs),
                "mae_deg": mae,
                "map_churn_deg": churn,
            }
        )
    _write_csv(mode_out / "speaker_map_trace_metrics.csv", trace_rows)
    return {
        "mae_deg": float(np.mean(frame_maes)) if frame_maes else float("nan"),
        "map_churn_deg": float(np.mean(churn_values)) if churn_values else float("nan"),
        "mismatch_rate": float(np.mean(np.asarray(frame_maes, dtype=float) > 15.0)) if frame_maes else 0.0,
    }


def analyze_case(
    scene_path: Path,
    *,
    out_root: Path,
    methods: list[str],
    benchmark_cfg_path: Path,
) -> dict:
    case_dir = out_root / scene_path.stem
    case_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_benchmark_cfg(benchmark_cfg_path)
    method_cfgs = cfg["methods"]
    sim_cfg = SimulationConfig.from_file(scene_path)
    localization_rows: list[dict] = []
    for method in methods:
        result = run_method_on_scene(method, method_cfgs[method], sim_cfg)
        match = match_predictions(result.true_doas_deg, result.estimated_doas_deg)
        metrics = compute_scene_metrics(match=match, n_true=len(result.true_doas_deg), n_pred=len(result.estimated_doas_deg))
        localization_rows.append(
            {
                "scene_id": scene_path.stem,
                "method": method,
                "status": "ok",
                "true_doas_deg": result.true_doas_deg,
                "pred_doas_deg": result.estimated_doas_deg,
                "mae_deg_matched": metrics.mae_deg_matched,
                "rmse_deg_matched": metrics.rmse_deg_matched,
                "misses": metrics.misses,
                "false_alarms": metrics.false_alarms,
                "runtime_seconds": result.runtime_seconds,
            }
        )
    _write_csv(case_dir / "localization_metrics.csv", localization_rows)
    _plot_method_mae(case_dir / "localization_mae.png", localization_rows)

    pipeline_rows: list[dict] = []
    pipeline_summary: dict[str, dict[str, float]] = {}
    for mode in ["baseline", "robust"]:
        mode_out = case_dir / mode
        summary = run_simulation_pipeline(
            scene_config_path=scene_path,
            out_dir=mode_out,
            use_mock_separation=True,
            robust_mode=(mode == "robust"),
            capture_trace=True,
        )
        metrics = _pipeline_trace_metrics(scene_path, mode_out, summary)
        pipeline_summary[mode] = metrics
        pipeline_rows.append({"mode": mode, **metrics})
    _write_csv(case_dir / "pipeline_metrics.csv", pipeline_rows)

    report = {
        "scene_path": str(scene_path.resolve()),
        "localization": localization_rows,
        "pipeline": pipeline_summary,
    }
    with (case_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate noisy ReSpeaker scene variants and run localization stress checks.")
    p.add_argument(
        "--scenes",
        nargs="+",
        default=[
            "simulation/simulations/configs/library_scene/library_k2_scene19.json",
            "simulation/simulations/configs/restaurant_scene/restaurant_k2_scene04.json",
            "simulation/simulations/configs/restaurant_scene/restaurant_k3_scene34.json",
        ],
    )
    p.add_argument("--out-root", default="simulation/output/noisy_respeaker_cases")
    p.add_argument("--profile", choices=["respeaker_v3_0457", "respeaker_cross_0640"], default="respeaker_v3_0457")
    p.add_argument("--noise-count", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--benchmark-config", default="localization/benchmark/configs/default.json")
    p.add_argument("--methods", nargs="+", default=["SRP-PHAT", "GMDA", "SSZ"])
    p.add_argument("--timeline-window-ms", type=int, default=400)
    p.add_argument("--timeline-hop-ms", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_root)
    generated_dir = out_root / "generated_scenes"
    reports: list[dict] = []
    generated_scenes: list[Path] = []
    for idx, raw_scene in enumerate(args.scenes):
        source_scene = Path(raw_scene)
        generated_scene = build_noisy_respeaker_variant(
            source_scene,
            out_dir=generated_dir,
            profile=args.profile,
            noise_count=args.noise_count,
            seed=args.seed + idx,
        )
        generated_scenes.append(generated_scene)
        reports.append(
            analyze_case(
                generated_scene,
                out_root=out_root / "results",
                methods=list(args.methods),
                benchmark_cfg_path=Path(args.benchmark_config),
            )
        )

    timeline_paths = generate_timeline_artifacts(
        scenes=generated_scenes,
        methods=list(args.methods),
        benchmark_config=Path(args.benchmark_config),
        out_root=out_root / "results",
        pipeline_results_root=out_root / "results",
        window_ms=int(args.timeline_window_ms),
        hop_ms=int(args.timeline_hop_ms),
    )

    summary = {
        "profile": args.profile,
        "noise_count": int(args.noise_count),
        "cases": reports,
        "timeline_plots": [str(path) for path in timeline_paths],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
