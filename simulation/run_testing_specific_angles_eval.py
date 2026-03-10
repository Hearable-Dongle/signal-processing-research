from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from localization.benchmark.algo_runner import run_method_on_scene
from localization.benchmark.matching import match_predictions
from localization.benchmark.metrics import compute_scene_metrics
from realtime_pipeline.simulation_runner import run_simulation_pipeline
from simulation.mic_array_profiles import mic_positions_xyz
from simulation.plot_noisy_respeaker_timelines import generate_timeline_artifacts
from simulation.simulation_config import MicrophoneArray, SimulationConfig


DEFAULT_SCENES_ROOT = Path("simulation/simulations/configs/testing_specific_angles")
DEFAULT_ASSETS_ROOT = Path("simulation/simulations/assets/testing_specific_angles")
DEFAULT_OUT_ROOT = Path("simulation/output/testing_specific_angles_eval")
DEFAULT_PROFILE = "respeaker_v3_0457"


def _load_benchmark_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_json_value(v) for k, v in value.items()}
    return value


def _load_scene_metadata(scene_path: Path, assets_root: Path) -> dict:
    scene_dir = assets_root / scene_path.stem
    metadata_path = scene_dir / "scenario_metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _with_profile(scene_path: Path, out_dir: Path, profile: str) -> Path:
    sim_cfg = SimulationConfig.from_file(scene_path)
    rel_positions = mic_positions_xyz(profile)
    sim_cfg.microphone_array = MicrophoneArray(
        mic_center=list(sim_cfg.microphone_array.mic_center),
        mic_radius=float(np.max(np.linalg.norm(rel_positions[:, :2], axis=1))),
        mic_count=int(rel_positions.shape[0]),
        mic_positions=rel_positions.tolist(),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_path.stem}__{profile}.json"
    sim_cfg.to_file(out_path)
    return out_path


def _plot_sweep_heatmap(path: Path, rows: list[dict], title: str, metric_key: str) -> None:
    if not rows:
        return
    main_angles = sorted({int(r["main_angle_deg"]) for r in rows})
    noise_keys = sorted({str(r["noise_label"]) for r in rows})
    if not main_angles or not noise_keys:
        return
    grid = np.full((len(noise_keys), len(main_angles)), np.nan, dtype=float)
    for row in rows:
        y = noise_keys.index(str(row["noise_label"]))
        x = main_angles.index(int(row["main_angle_deg"]))
        val = row.get(metric_key)
        if val is None:
            continue
        grid[y, x] = float(val)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, max(4, 0.6 * len(noise_keys))))
    im = plt.imshow(grid, aspect="auto", interpolation="nearest")
    plt.xticks(np.arange(len(main_angles)), [str(v) for v in main_angles])
    plt.yticks(np.arange(len(noise_keys)), noise_keys)
    plt.xlabel("Main speaker angle (deg)")
    plt.ylabel("Noise layout")
    plt.title(title)
    plt.colorbar(im, label=metric_key)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def analyze_scene(
    scene_path: Path,
    *,
    out_root: Path,
    methods: list[str],
    benchmark_cfg_path: Path,
    metadata: dict,
    beamforming_mode: str,
) -> dict:
    case_dir = out_root / scene_path.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    method_cfgs = _load_benchmark_cfg(benchmark_cfg_path)["methods"]
    sim_cfg = SimulationConfig.from_file(scene_path)
    localization_rows: list[dict] = []
    for method in methods:
        result = run_method_on_scene(method, method_cfgs[method], sim_cfg)
        match = match_predictions(result.true_doas_deg, result.estimated_doas_deg)
        metrics = compute_scene_metrics(match=match, n_true=len(result.true_doas_deg), n_pred=len(result.estimated_doas_deg))
        localization_rows.append(
            {
                "scene_id": scene_path.stem,
                "scene_type": "testing_specific_angles",
                "method": method,
                "status": "ok",
                "true_doas_deg": result.true_doas_deg,
                "pred_doas_deg": result.estimated_doas_deg,
                "mae_deg_matched": metrics.mae_deg_matched,
                "rmse_deg_matched": metrics.rmse_deg_matched,
                "misses": metrics.misses,
                "false_alarms": metrics.false_alarms,
                "runtime_seconds": result.runtime_seconds,
                "main_angle_deg": metadata.get("main_angle_deg"),
                "secondary_angle_deg": metadata.get("secondary_angle_deg"),
                "noise_layout_type": metadata.get("noise_layout_type"),
                "noise_angles_deg": metadata.get("noise_angles_deg"),
            }
        )
    _write_csv(case_dir / "localization_metrics.csv", localization_rows)

    pipeline_rows: list[dict] = []
    pipeline_summary: dict[str, dict] = {}
    for mode in ["baseline", "robust"]:
        mode_out = case_dir / mode
        summary = run_simulation_pipeline(
            scene_config_path=scene_path,
            out_dir=mode_out,
            use_mock_separation=True,
            robust_mode=(mode == "robust"),
            capture_trace=True,
            beamforming_mode=str(beamforming_mode),
        )
        trace = summary.get("speaker_map_trace", [])
        trace_counts = [len(frame.get("speakers", [])) for frame in trace]
        pipeline_row = {
            "mode": mode,
            "mean_num_tracks": float(np.mean(trace_counts)) if trace_counts else float("nan"),
            "num_trace_frames": len(trace),
        }
        pipeline_rows.append(pipeline_row)
        pipeline_summary[mode] = summary
    _write_csv(case_dir / "pipeline_metrics.csv", pipeline_rows)

    report = {
        "scene_path": str(scene_path.resolve()),
        "scene_type": "testing_specific_angles",
        "geometry": {
            "main_angle_deg": metadata.get("main_angle_deg"),
            "secondary_angle_deg": metadata.get("secondary_angle_deg"),
            "noise_layout_type": metadata.get("noise_layout_type"),
            "noise_angles_deg": metadata.get("noise_angles_deg"),
        },
        "localization": localization_rows,
        "pipeline": _normalize_json_value(pipeline_summary),
    }
    with (case_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate testing-specific-angle scenes with ReSpeaker geometry")
    parser.add_argument("--scenes-root", default=str(DEFAULT_SCENES_ROOT))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--profile", choices=["respeaker_v3_0457", "respeaker_cross_0640"], default=DEFAULT_PROFILE)
    parser.add_argument("--beamforming-mode", choices=["mvdr_fd", "gsc_fd", "delay_sum"], default="mvdr_fd")
    parser.add_argument("--benchmark-config", default="localization/benchmark/configs/default.json")
    parser.add_argument("--methods", nargs="+", default=["SRP-PHAT", "GMDA", "SSZ"])
    parser.add_argument("--timeline-window-ms", type=int, default=400)
    parser.add_argument("--timeline-hop-ms", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenes_root = Path(args.scenes_root)
    assets_root = Path(args.assets_root)
    out_root = Path(args.out_root)
    generated_dir = out_root / "generated_scenes"
    results_dir = out_root / "results"
    scene_paths = sorted(scenes_root.glob("*.json"))
    if not scene_paths:
        raise RuntimeError(f"No scenes found under {scenes_root}")

    profiled_paths: list[Path] = []
    reports: list[dict] = []
    summary_rows: list[dict] = []
    for scene_path in scene_paths:
        metadata = _load_scene_metadata(scene_path, assets_root)
        profiled_scene = _with_profile(scene_path, generated_dir, args.profile)
        profiled_paths.append(profiled_scene)
        report = analyze_scene(
            profiled_scene,
            out_root=results_dir,
            methods=list(args.methods),
            benchmark_cfg_path=Path(args.benchmark_config),
            metadata=metadata,
            beamforming_mode=str(args.beamforming_mode),
        )
        reports.append(report)
        geometry = report.get("geometry", {})
        best_row = next((row for row in report["localization"] if row["method"] == args.methods[0]), report["localization"][0])
        summary_rows.append(
            {
                "scene_id": profiled_scene.stem,
                "main_angle_deg": geometry.get("main_angle_deg"),
                "secondary_angle_deg": geometry.get("secondary_angle_deg"),
                "noise_layout_type": geometry.get("noise_layout_type"),
                "noise_angles_deg": geometry.get("noise_angles_deg"),
                "noise_label": "/".join(str(v) for v in (geometry.get("noise_angles_deg") or [])),
                "reference_method": best_row["method"],
                "reference_mae_deg_matched": best_row["mae_deg_matched"],
            }
        )

    timeline_paths = generate_timeline_artifacts(
        scenes=profiled_paths,
        methods=list(args.methods),
        benchmark_config=Path(args.benchmark_config),
        out_root=results_dir,
        pipeline_results_root=results_dir,
        window_ms=int(args.timeline_window_ms),
        hop_ms=int(args.timeline_hop_ms),
    )
    _write_csv(out_root / "sweep_summary.csv", summary_rows)
    _plot_sweep_heatmap(
        out_root / "sweep_reference_mae.png",
        summary_rows,
        title=f"Testing-specific-angles reference MAE [{args.methods[0]}]",
        metric_key="reference_mae_deg_matched",
    )

    summary = {
        "scene_type": "testing_specific_angles",
        "profile": args.profile,
        "beamforming_mode": str(args.beamforming_mode),
        "num_scenes": len(profiled_paths),
        "timeline_plots": [str(path) for path in timeline_paths],
        "reports": reports,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(_normalize_json_value(summary), f, indent=2)
    print(json.dumps({"num_scenes": len(profiled_paths), "out_root": str(out_root.resolve())}, indent=2))


if __name__ == "__main__":
    main()
