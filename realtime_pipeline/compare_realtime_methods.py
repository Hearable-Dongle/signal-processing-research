from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from sim.realistic_conversations.config import build_preset
from sim.realistic_conversations.generator import generate_scenario
from simulation.simulation_config import SimulationConfig

from .plot_scene_setup import render_scene_setup
from .simulation_runner import run_simulation_pipeline


METHOD_LOCALIZATION_ONLY = "localization_only"
METHOD_SPATIAL_BASELINE = "spatial_baseline"
METHOD_SPEAKER_TRACKING = "speaker_tracking"
METHOD_SPEAKER_TRACKING_LONG_MEMORY = "speaker_tracking_long_memory"
COMPARISON_METHODS = (
    METHOD_LOCALIZATION_ONLY,
    METHOD_SPATIAL_BASELINE,
    METHOD_SPEAKER_TRACKING,
    METHOD_SPEAKER_TRACKING_LONG_MEMORY,
)


@dataclass(frozen=True)
class MethodPreset:
    method_id: str
    label: str
    control_mode: str
    fast_path_reference_mode: str
    direction_long_memory_enabled: bool
    direction_long_memory_window_ms: float = 60000.0


def get_method_preset(method_id: str) -> MethodPreset:
    method = str(method_id).strip().lower()
    if method == METHOD_LOCALIZATION_ONLY:
        return MethodPreset(
            method_id=method,
            label="Localization Only",
            control_mode="spatial_peak_mode",
            fast_path_reference_mode="srp_peak",
            direction_long_memory_enabled=False,
        )
    if method == METHOD_SPATIAL_BASELINE:
        return MethodPreset(
            method_id=method,
            label="Spatial Baseline",
            control_mode="spatial_peak_mode",
            fast_path_reference_mode="speaker_map",
            direction_long_memory_enabled=False,
        )
    if method == METHOD_SPEAKER_TRACKING:
        return MethodPreset(
            method_id=method,
            label="Speaker Tracking",
            control_mode="speaker_tracking_mode",
            fast_path_reference_mode="speaker_map",
            direction_long_memory_enabled=False,
        )
    if method == METHOD_SPEAKER_TRACKING_LONG_MEMORY:
        return MethodPreset(
            method_id=method,
            label="Speaker Tracking + Long Memory",
            control_mode="speaker_tracking_mode",
            fast_path_reference_mode="speaker_map",
            direction_long_memory_enabled=True,
        )
    raise ValueError(f"Unsupported comparison method: {method_id}")


def _normalize_angle_deg(v: float) -> float:
    return float(v % 360.0)


def _circular_distance_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _noise_snr_shift_db(noise_scale: float) -> float:
    scale = max(float(noise_scale), 1e-6)
    return float(-20.0 * math.log10(scale))


def generate_sequential_restaurant_scene(
    *,
    out_root: str | Path,
    duration_sec: float = 60.0,
    noise_scale: float = 0.4,
    seed: int = 20260314,
    sample_rate: int = 16000,
    frame_ms: int = 20,
) -> dict[str, str]:
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = build_preset("restaurant_meeting")
    cfg.turn_taking.min_speakers = 2
    cfg.turn_taking.max_speakers = 2
    cfg.turn_taking.utterance_sec_range = [2.8, 6.2]
    cfg.turn_taking.pause_sec_range = [0.25, 1.2]
    cfg.turn_taking.overlap_sec_range = [0.05, 0.12]
    cfg.turn_taking.overlap_probability = 0.02
    cfg.turn_taking.interruption_probability = 0.0
    cfg.turn_taking.persistence_probability = 0.9
    cfg.turn_taking.backchannel_probability = 0.0
    cfg.render.duration_sec = float(duration_sec)
    cfg.render.sample_rate = int(sample_rate)
    cfg.render.frame_ms = int(frame_ms)

    snr_shift_db = _noise_snr_shift_db(noise_scale)
    base_snr = np.asarray(cfg.noise.base_snr_db_range, dtype=float)
    cfg.noise.base_snr_db_range = [float(base_snr[0] + snr_shift_db), float(base_snr[1] + snr_shift_db)]

    scene_name = f"restaurant_meeting_seq_k2_noise{str(noise_scale).replace('.', 'p')}x_{int(duration_sec)}s"
    result = generate_scenario(
        preset="restaurant_meeting",
        out_dir=out_root,
        seed=int(seed),
        duration_sec=float(duration_sec),
        sample_rate=int(sample_rate),
        frame_ms=int(frame_ms),
        export_audio=False,
        scene_name=scene_name,
        config_override=cfg,
    )
    return {
        "scene_name": scene_name,
        "scene_dir": str(result.scene_dir.resolve()),
        "scene_config": str(result.scene_config_path.resolve()),
        "scenario_metadata": str(result.metadata_path.resolve()),
        "frame_truth": str(result.frame_truth_path.resolve()),
        "metrics_summary": str(result.metrics_path.resolve()),
    }


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_active_speaker_ground_truth(
    *,
    scene_config_path: str | Path,
    scenario_metadata_path: str | Path,
    frame_step_ms: float,
) -> dict[str, object]:
    scene_cfg = SimulationConfig.from_file(scene_config_path)
    meta = _load_json(scenario_metadata_path)
    mic_center = np.asarray(scene_cfg.microphone_array.mic_center[:2], dtype=float)

    doa_by_speaker: dict[int, float] = {}
    for row in meta.get("assets", {}).get("render_segments", []):
        if row.get("classification") != "speech" or "speaker_id" not in row:
            continue
        sid = int(row["speaker_id"])
        pos = np.asarray(row.get("position_m", row.get("loc", [0.0, 0.0]))[:2], dtype=float)
        delta = pos - mic_center
        doa_by_speaker[sid] = _normalize_angle_deg(np.degrees(np.arctan2(delta[1], delta[0])))

    speech_events = sorted(
        meta.get("assets", {}).get("speech_events", []),
        key=lambda row: (float(row.get("start_sec", 0.0)), float(row.get("end_sec", 0.0))),
    )
    duration_s = float(meta.get("config", {}).get("render", {}).get("duration_sec", scene_cfg.audio.duration))
    num_frames = max(1, int(round(duration_s * 1000.0 / max(float(frame_step_ms), 1.0))))
    time_s = np.arange(num_frames, dtype=np.float64) * (float(frame_step_ms) / 1000.0)

    active_speaker_ids = np.full(num_frames, -1, dtype=np.int32)
    active_directions_deg = np.full(num_frames, np.nan, dtype=np.float64)

    for idx, t_sec in enumerate(time_s):
        active_rows = [
            row
            for row in speech_events
            if float(row.get("start_sec", 0.0)) <= float(t_sec) < float(row.get("end_sec", 0.0))
        ]
        if not active_rows:
            continue
        active_rows.sort(key=lambda row: (float(row.get("start_sec", 0.0)), float(row.get("end_sec", 0.0))))
        chosen = active_rows[-1]
        sid = int(chosen["speaker_id"])
        active_speaker_ids[idx] = sid
        active_directions_deg[idx] = float(doa_by_speaker.get(sid, np.nan))

    return {
        "time_s": time_s.tolist(),
        "active_speaker_ids": active_speaker_ids.tolist(),
        "active_directions_deg": active_directions_deg.tolist(),
        "doa_by_speaker": {str(k): float(v) for k, v in sorted(doa_by_speaker.items())},
    }


def _prediction_from_summary(summary: dict, active_gt: dict[str, object], *, method_id: str) -> list[float]:
    active_speaker_ids = np.asarray(active_gt["active_speaker_ids"], dtype=int)
    active_directions = np.asarray(active_gt["active_directions_deg"], dtype=float)
    predicted = np.full(active_speaker_ids.shape[0], np.nan, dtype=np.float64)

    if method_id == METHOD_LOCALIZATION_ONLY:
        trace = summary.get("srp_trace", [])
        for row in trace:
            frame_idx = int(row["frame_index"])
            peaks = row.get("peaks_deg", [])
            if frame_idx >= predicted.shape[0] or active_speaker_ids[frame_idx] < 0 or not peaks:
                continue
            predicted[frame_idx] = float(peaks[0])
        return predicted.tolist()

    trace = summary.get("speaker_map_trace", [])
    for row in trace:
        frame_idx = int(row["frame_index"])
        if frame_idx >= predicted.shape[0] or active_speaker_ids[frame_idx] < 0:
            continue
        speakers = row.get("speakers", [])
        if not speakers:
            continue
        gt_deg = float(active_directions[frame_idx])
        best = min(
            speakers,
            key=lambda item: (
                _circular_distance_deg(float(item.get("direction_degrees", 0.0)), gt_deg),
                -float(item.get("gain_weight", 0.0)),
            ),
        )
        predicted[frame_idx] = float(best.get("direction_degrees", np.nan))
    return predicted.tolist()


def _plot_active_speaker_timelines(
    *,
    out_path: Path,
    methods: list[MethodPreset],
    method_summaries: dict[str, dict],
    active_gt: dict[str, object],
) -> None:
    time_s = np.asarray(active_gt["time_s"], dtype=float)
    gt_speakers = np.asarray(active_gt["active_speaker_ids"], dtype=int)
    gt_deg = np.asarray(active_gt["active_directions_deg"], dtype=float)
    speaker_ids = sorted({int(v) for v in gt_speakers if int(v) >= 0})
    cmap = plt.get_cmap("tab10")
    colors = {sid: cmap(idx % 10) for idx, sid in enumerate(speaker_ids)}

    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 2.8 * len(methods)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, preset in zip(axes, methods):
        pred_deg = np.asarray(method_summaries[preset.method_id]["active_prediction_deg"], dtype=float)
        for sid in speaker_ids:
            mask = gt_speakers == sid
            gt_series = np.where(mask, gt_deg, np.nan)
            pred_series = np.where(mask, pred_deg, np.nan)
            ax.plot(time_s, gt_series, color=colors[sid], linewidth=2.0, label=f"spk {sid} gt" if ax is axes[0] else None)
            ax.plot(time_s, pred_series, color=colors[sid], linewidth=1.6, linestyle="--", alpha=0.85, label=f"spk {sid} pred" if ax is axes[0] else None)
        ax.set_title(preset.label)
        ax.set_ylabel("deg")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-5.0, 365.0)
    axes[-1].set_xlabel("time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, ncol=min(4, len(labels)), loc="upper right")
    fig.suptitle("Active speaker direction over time: ground truth vs predicted")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_waveforms(*, out_path: Path, methods: list[MethodPreset], output_root: Path) -> None:
    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 2.2 * len(methods)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, preset in zip(axes, methods):
        y, sr = sf.read(output_root / preset.method_id / "enhanced_fast_path.wav", always_2d=False)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        t = np.arange(y.shape[0], dtype=np.float64) / max(int(sr), 1)
        ax.plot(t, y, linewidth=0.6)
        ax.set_ylabel(preset.method_id)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Enhanced output waveforms")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_spectrograms(*, out_path: Path, methods: list[MethodPreset], output_root: Path) -> None:
    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 2.8 * len(methods)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, preset in zip(axes, methods):
        y, sr = sf.read(output_root / preset.method_id / "enhanced_fast_path.wav", always_2d=False)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        ax.specgram(y, NFFT=512, Fs=int(sr), noverlap=256, cmap="magma")
        ax.set_ylabel(preset.method_id)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Enhanced output spectrograms")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _run_methods(
    *,
    scene_config_path: Path,
    output_root: Path,
    methods: list[MethodPreset],
    use_mock_separation: bool,
) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for preset in methods:
        method_out = output_root / preset.method_id
        summary = run_simulation_pipeline(
            scene_config_path=scene_config_path,
            out_dir=method_out,
            use_mock_separation=use_mock_separation,
            capture_trace=True,
            control_mode=preset.control_mode,
            fast_path_reference_mode=preset.fast_path_reference_mode,
            direction_long_memory_enabled=preset.direction_long_memory_enabled,
            direction_long_memory_window_ms=preset.direction_long_memory_window_ms,
        )
        summary["method_id"] = preset.method_id
        summary["label"] = preset.label
        summaries[preset.method_id] = summary
    return summaries


def run_comparison_suite(
    *,
    out_dir: str | Path,
    methods: list[str] | None = None,
    scene_config_path: str | Path | None = None,
    scene_asset_root: str | Path | None = None,
    duration_sec: float = 60.0,
    noise_scale: float = 0.4,
    seed: int = 20260314,
    use_mock_separation: bool = False,
) -> dict:
    output_root = Path(out_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    visualizations_root = output_root / "visualizations"
    visualizations_root.mkdir(parents=True, exist_ok=True)

    method_ids = list(methods or COMPARISON_METHODS)
    method_presets = [get_method_preset(method_id) for method_id in method_ids]

    if scene_config_path is None:
        scene_info = generate_sequential_restaurant_scene(
            out_root=scene_asset_root or (output_root.parent / f"{output_root.name}_assets"),
            duration_sec=float(duration_sec),
            noise_scale=float(noise_scale),
            seed=int(seed),
        )
        scene_config = Path(scene_info["scene_config"])
        scenario_metadata = Path(scene_info["scenario_metadata"])
        metrics_summary = Path(scene_info["metrics_summary"])
    else:
        scene_config = Path(scene_config_path).resolve()
        scenario_metadata = scene_config.parent / "scenario_metadata.json"
        metrics_summary = scene_config.parent / "metrics_summary.json"
        scene_info = {
            "scene_config": str(scene_config),
            "scenario_metadata": str(scenario_metadata.resolve()) if scenario_metadata.exists() else "",
            "metrics_summary": str(metrics_summary.resolve()) if metrics_summary.exists() else "",
            "scene_dir": str(scene_config.parent.resolve()),
            "scene_name": scene_config.parent.name,
        }

    summaries = _run_methods(
        scene_config_path=scene_config,
        output_root=output_root,
        methods=method_presets,
        use_mock_separation=bool(use_mock_separation),
    )

    active_gt = build_active_speaker_ground_truth(
        scene_config_path=scene_config,
        scenario_metadata_path=scenario_metadata,
        frame_step_ms=10.0,
    )
    for preset in method_presets:
        summaries[preset.method_id]["active_prediction_deg"] = _prediction_from_summary(
            summaries[preset.method_id],
            active_gt,
            method_id=preset.method_id,
        )

    _plot_active_speaker_timelines(
        out_path=visualizations_root / "active_speaker_direction_compare.png",
        methods=method_presets,
        method_summaries=summaries,
        active_gt=active_gt,
    )
    _plot_waveforms(out_path=visualizations_root / "waveforms.png", methods=method_presets, output_root=output_root)
    _plot_spectrograms(out_path=visualizations_root / "spectrograms.png", methods=method_presets, output_root=output_root)
    render_scene_setup(
        scene_config_path=scene_config,
        scenario_metadata_path=scenario_metadata if scenario_metadata.exists() else None,
        comparison_summary_path=None,
        out_path=visualizations_root / "scene_layout_user_view.png",
    )

    metrics_payload = _load_json(metrics_summary) if metrics_summary.exists() else {}
    comparison_summary = {
        "scene": scene_info,
        "methods": [
            {
                "method_id": preset.method_id,
                "label": preset.label,
                "control_mode": preset.control_mode,
                "fast_path_reference_mode": preset.fast_path_reference_mode,
                "direction_long_memory_enabled": bool(preset.direction_long_memory_enabled),
                "direction_long_memory_window_ms": float(preset.direction_long_memory_window_ms),
                "summary_path": str((output_root / preset.method_id / "summary.json").resolve()),
                "enhanced_wav": str((output_root / preset.method_id / "enhanced_fast_path.wav").resolve()),
                "fast_rtf": float(summaries[preset.method_id]["fast_rtf"]),
                "slow_rtf": float(summaries[preset.method_id]["slow_rtf"]),
                "final_speakers": len(summaries[preset.method_id]["speaker_map_final"]),
            }
            for preset in method_presets
        ],
        "scene_metrics": metrics_payload,
        "active_ground_truth_path": str((output_root / "active_speaker_ground_truth.json").resolve()),
        "visualizations": {
            "active_speaker_direction_compare": str((visualizations_root / "active_speaker_direction_compare.png").resolve()),
            "waveforms": str((visualizations_root / "waveforms.png").resolve()),
            "spectrograms": str((visualizations_root / "spectrograms.png").resolve()),
            "scene_layout_user_view": str((visualizations_root / "scene_layout_user_view.png").resolve()),
        },
        "use_mock_separation": bool(use_mock_separation),
    }
    (output_root / "active_speaker_ground_truth.json").write_text(json.dumps(active_gt, indent=2), encoding="utf-8")
    (output_root / "comparison_summary.json").write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
    return comparison_summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a stable four-method realtime comparison suite.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scene-config", default=None)
    parser.add_argument("--scene-asset-root", default=None)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--noise-scale", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=20260314)
    parser.add_argument("--methods", nargs="+", choices=list(COMPARISON_METHODS), default=list(COMPARISON_METHODS))
    parser.add_argument("--mock-separation", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = run_comparison_suite(
        out_dir=args.out_dir,
        methods=[str(v) for v in args.methods],
        scene_config_path=args.scene_config,
        scene_asset_root=args.scene_asset_root,
        duration_sec=float(args.duration_sec),
        noise_scale=float(args.noise_scale),
        seed=int(args.seed),
        use_mock_separation=bool(args.mock_separation),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
