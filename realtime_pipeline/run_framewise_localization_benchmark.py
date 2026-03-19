from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from realtime_pipeline.compare_realtime_methods import build_framewise_speech_ground_truth
from realtime_pipeline.simulation_runner import run_simulation_pipeline
from simulation.mic_array_profiles import SUPPORTED_MIC_ARRAY_PROFILES, mic_positions_xyz
from simulation.simulation_config import MicrophoneArray, SimulationConfig


DEFAULT_SCENES_ROOT = Path("simulation/simulations/configs/testing_specific_angles")
DEFAULT_ASSETS_ROOT = Path("simulation/simulations/assets/testing_specific_angles")
DEFAULT_OUT_ROOT = Path("realtime_pipeline/output/framewise_localization_benchmark")
DEFAULT_PROFILE = "respeaker_v3_0457"
DEFAULT_STRATEGIES = [
    "srp_phat_localization",
]


@dataclass(frozen=True)
class JobResult:
    strategy: str
    scene_id: str
    active_metrics: dict[str, float | None]
    topk_metrics: dict[str, float | None]
    scene_summary: dict[str, object]
    predictions_csv: str
    timeline_png: str


def _normalize_deg(value: float) -> float:
    return float(value % 360.0)


def _angular_error_deg(pred: float, truth: float) -> float:
    return float(abs((float(pred) - float(truth) + 180.0) % 360.0 - 180.0))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _scale_noise_gains(sim_cfg: SimulationConfig, noise_gain_scale: float) -> SimulationConfig:
    if abs(float(noise_gain_scale) - 1.0) < 1e-12:
        return sim_cfg
    for source in sim_cfg.audio.sources:
        if str(getattr(source, "classification", "")).strip().lower() == "noise":
            source.gain = float(source.gain) * float(noise_gain_scale)
    return sim_cfg


def _stage_scene(
    scene_path: Path,
    staging_root: Path,
    profile: str,
    assets_root: Path,
    noise_gain_scale: float,
) -> tuple[Path, Path]:
    stage_dir = staging_root / scene_path.stem
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_scene = stage_dir / scene_path.name
    sim_cfg = SimulationConfig.from_file(scene_path)
    sim_cfg = _scale_noise_gains(sim_cfg, float(noise_gain_scale))
    rel_positions = mic_positions_xyz(profile)
    sim_cfg.microphone_array = MicrophoneArray(
        mic_center=list(sim_cfg.microphone_array.mic_center),
        mic_radius=float(np.max(np.linalg.norm(rel_positions[:, :2], axis=1))),
        mic_count=int(rel_positions.shape[0]),
        mic_positions=rel_positions.tolist(),
    )
    sim_cfg.to_file(staged_scene)

    asset_dir = assets_root / scene_path.stem
    for filename in ("scenario_metadata.json", "metrics_summary.json", "frame_ground_truth.csv"):
        src = asset_dir / filename
        if src.exists():
            shutil.copy2(src, stage_dir / filename)
    return staged_scene, stage_dir / "scenario_metadata.json"


def _extract_trace_rows(summary: dict, n_frames: int) -> tuple[list[list[float]], list[float], list[bool], list[list[float] | None]]:
    pred_lists: list[list[float]] = [[] for _ in range(n_frames)]
    active_pred = [float("nan")] * n_frames
    gated = [False] * n_frames
    scores: list[list[float] | None] = [None] * n_frames
    for row in summary.get("srp_trace", []):
        idx = int(row.get("frame_index", -1))
        if idx < 0 or idx >= n_frames:
            continue
        peaks = [float(v) for v in row.get("peaks_deg", [])]
        peak_scores = row.get("peak_scores")
        pred_lists[idx] = peaks
        active_pred[idx] = float(peaks[0]) if peaks else float("nan")
        if peak_scores is not None:
            scores[idx] = [float(v) for v in peak_scores]
        debug = row.get("debug", {}) or {}
        gated[idx] = bool(debug.get("gated", False))
    return pred_lists, active_pred, gated, scores


def _static_segments(active_truth_deg: list[float]) -> list[slice]:
    arr = np.asarray(active_truth_deg, dtype=np.float64)
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
        if _angular_error_deg(float(arr[idx - 1]), float(value)) > 1e-9:
            if idx - start >= 2:
                segments.append(slice(start, idx))
            start = idx
    if start is not None and arr.size - start >= 2:
        segments.append(slice(start, arr.size))
    return segments


def _match_topk(preds: list[float], truths: list[float]) -> tuple[list[float], int, int]:
    preds = [float(v) for v in preds if np.isfinite(v)]
    truths = [float(v) for v in truths if np.isfinite(v)]
    if not preds or not truths:
        return [], len(truths), len(preds)

    best_errors: list[float] | None = None
    best_cost: float | None = None

    def _search(remaining_preds: list[float], remaining_truths: list[float], acc: list[float]) -> None:
        nonlocal best_errors, best_cost
        if not remaining_preds or not remaining_truths:
            cost = float(sum(acc))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_errors = list(acc)
            return
        first_truth = remaining_truths[0]
        for idx, pred in enumerate(remaining_preds):
            err = _angular_error_deg(pred, first_truth)
            _search(remaining_preds[:idx] + remaining_preds[idx + 1 :], remaining_truths[1:], acc + [err])
        cost = float(sum(acc))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_errors = list(acc)

    if len(preds) <= len(truths):
        _search(preds, truths, [])
    else:
        def _search_truths(remaining_preds: list[float], remaining_truths: list[float], acc: list[float]) -> None:
            nonlocal best_errors, best_cost
            if not remaining_preds or not remaining_truths:
                cost = float(sum(acc))
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_errors = list(acc)
                return
            first_pred = remaining_preds[0]
            for idx, truth in enumerate(remaining_truths):
                err = _angular_error_deg(first_pred, truth)
                _search_truths(remaining_preds[1:], remaining_truths[:idx] + remaining_truths[idx + 1 :], acc + [err])
            cost = float(sum(acc))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_errors = list(acc)

        _search_truths(preds, truths, [])
    errors = [] if best_errors is None else best_errors
    matches = len(errors)
    return errors, max(0, len(truths) - matches), max(0, len(preds) - matches)


def _compute_active_metrics(active_truth_deg: list[float], active_pred_deg: list[float], gated_flags: list[bool]) -> dict[str, float | None]:
    errors: list[float] = []
    false_alarm_frames = 0
    no_target_frames = 0
    coverage_hits = 0
    evaluable_frames = 0
    pred_arr = np.asarray(active_pred_deg, dtype=np.float64)
    truth_arr = np.asarray(active_truth_deg, dtype=np.float64)
    for pred, truth in zip(pred_arr, truth_arr):
        if np.isfinite(truth):
            evaluable_frames += 1
            if np.isfinite(pred):
                coverage_hits += 1
                errors.append(_angular_error_deg(float(pred), float(truth)))
        elif np.isfinite(pred):
            no_target_frames += 1
            false_alarm_frames += 1
        else:
            no_target_frames += 1

    valid = np.asarray(errors, dtype=np.float64)
    stability_vals = []
    for seg in _static_segments(active_truth_deg):
        block = pred_arr[seg]
        block = block[np.isfinite(block)]
        if block.size >= 2:
            stability_vals.append(float(np.std(block)))
    gating_mask = np.asarray(gated_flags[: len(active_truth_deg)], dtype=bool)
    target_mask = np.isfinite(truth_arr)
    gated_targets = gating_mask[target_mask]
    return {
        "evaluable_frames": float(evaluable_frames),
        "coverage_rate": float(coverage_hits / evaluable_frames) if evaluable_frames else None,
        "gating_rate": float(np.mean(gated_targets)) if gated_targets.size else None,
        "false_alarm_rate": float(false_alarm_frames / no_target_frames) if no_target_frames else None,
        "mae_deg": float(np.mean(valid)) if valid.size else None,
        "acc_at_10": float(np.mean(valid <= 10.0)) if valid.size else None,
        "acc_at_20": float(np.mean(valid <= 20.0)) if valid.size else None,
        "acc_at_25": float(np.mean(valid <= 25.0)) if valid.size else None,
        "median_deg": float(np.median(valid)) if valid.size else None,
        "p90_deg": float(np.percentile(valid, 90)) if valid.size else None,
        "stability_deg": float(np.mean(stability_vals)) if stability_vals else None,
    }


def _compute_topk_metrics(pred_lists: list[list[float]], truth_lists: list[list[float]]) -> dict[str, float | None]:
    matched_errors: list[float] = []
    matched_frames = 0
    target_frames = 0
    misses = 0
    false_alarms = 0
    false_alarm_frames = 0
    no_target_frames = 0
    covered_frames = 0
    for preds, truths in zip(pred_lists, truth_lists):
        preds_clean = [float(v) for v in preds if np.isfinite(v)]
        truths_clean = [float(v) for v in truths if np.isfinite(v)]
        if truths_clean:
            target_frames += 1
            if preds_clean:
                covered_frames += 1
            errors, frame_misses, frame_false_alarms = _match_topk(preds_clean, truths_clean)
            if errors:
                matched_frames += 1
                matched_errors.extend(errors)
            misses += frame_misses
            false_alarms += frame_false_alarms
        else:
            no_target_frames += 1
            if preds_clean:
                false_alarm_frames += 1

    valid = np.asarray(matched_errors, dtype=np.float64)
    return {
        "target_frames": float(target_frames),
        "coverage_rate": float(covered_frames / target_frames) if target_frames else None,
        "false_alarm_rate": float(false_alarm_frames / no_target_frames) if no_target_frames else None,
        "miss_rate": float(misses / max(1, sum(len([v for v in truths if np.isfinite(v)]) for truths in truth_lists))) if truth_lists else None,
        "frame_match_rate": float(matched_frames / target_frames) if target_frames else None,
        "mae_deg": float(np.mean(valid)) if valid.size else None,
        "acc_at_10": float(np.mean(valid <= 10.0)) if valid.size else None,
        "acc_at_20": float(np.mean(valid <= 20.0)) if valid.size else None,
        "acc_at_25": float(np.mean(valid <= 25.0)) if valid.size else None,
        "median_deg": float(np.median(valid)) if valid.size else None,
        "p90_deg": float(np.percentile(valid, 90)) if valid.size else None,
    }


def _plot_timeline(out_path: Path, time_s: list[float], active_truth_deg: list[float], active_pred_deg: list[float], strategy: str, scene_id: str) -> None:
    xs = np.asarray(time_s, dtype=np.float64)
    truth = np.asarray(active_truth_deg, dtype=np.float64)
    pred = np.asarray(active_pred_deg, dtype=np.float64)
    plt.figure(figsize=(12, 4))
    plt.plot(xs, truth, label="active truth", linewidth=2.0)
    plt.plot(xs, pred, label="pred", linewidth=1.3, alpha=0.85)
    plt.ylim(-5.0, 365.0)
    plt.xlabel("Time (s)")
    plt.ylabel("DOA (deg)")
    plt.title(f"{strategy} :: {scene_id}")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _strategy_run_config(strategy: str) -> dict[str, object]:
    return {
        "localization_backend": str(strategy),
        "beamforming_mode": "delay_sum",
        "control_mode": "spatial_peak_mode",
        "fast_path_reference_mode": "srp_peak",
        "direction_long_memory_enabled": False,
    }


def _run_job(
    *,
    strategy: str,
    scene_path: str,
    assets_root: str,
    out_root: str,
    profile: str,
    noise_gain_scale: float,
    fast_frame_ms: int,
    slow_chunk_ms: int,
    localization_window_ms: int,
    localization_hop_ms: int,
    localization_grid_size: int,
    srp_overlap: float,
    srp_freq_min_hz: int,
    srp_freq_max_hz: int,
    localization_vad_enabled: bool,
    localization_snr_gating_enabled: bool,
    localization_snr_threshold_db: float,
    localization_msc_variance_enabled: bool,
    localization_msc_history_frames: int,
    localization_hsda_enabled: bool,
    localization_hsda_window_frames: int,
    input_downsample_rate_hz: int | None,
    localization_track_hold_frames: int,
    localization_max_assoc_distance_deg: float,
    localization_velocity_alpha: float,
    localization_angle_alpha: float,
    capon_spectrum_ema_alpha: float,
    capon_peak_min_sharpness: float,
    capon_peak_min_margin: float,
    capon_hold_frames: int,
    capon_freq_bin_subsample_stride: int,
    capon_freq_bin_min_hz: int | None,
    capon_freq_bin_max_hz: int | None,
    capon_use_cholesky_solve: bool,
    capon_covariance_ema_alpha: float,
    capon_full_scan_every_n_updates: int,
    capon_local_refine_enabled: bool,
    capon_local_refine_half_width_deg: float,
) -> JobResult:
    scene_cfg_path = Path(scene_path)
    bench_root = Path(out_root)
    job_root = bench_root / strategy / scene_cfg_path.stem
    staged_scene, staged_meta = _stage_scene(
        scene_cfg_path,
        bench_root / "_staged_scenes" / strategy,
        profile,
        Path(assets_root),
        float(noise_gain_scale),
    )
    config = _strategy_run_config(strategy)
    summary = run_simulation_pipeline(
        scene_config_path=staged_scene,
        out_dir=job_root,
        use_mock_separation=True,
        capture_trace=True,
        robust_mode=True,
        localization_backend=str(config["localization_backend"]),
        beamforming_mode=str(config["beamforming_mode"]),
        control_mode=str(config["control_mode"]),
        fast_path_reference_mode=str(config["fast_path_reference_mode"]),
        direction_long_memory_enabled=bool(config["direction_long_memory_enabled"]),
        fast_frame_ms=int(fast_frame_ms),
        slow_chunk_ms=int(slow_chunk_ms),
        localization_window_ms=int(localization_window_ms),
        localization_hop_ms=int(localization_hop_ms),
        localization_grid_size=int(localization_grid_size),
        srp_overlap=float(srp_overlap),
        srp_freq_min_hz=int(srp_freq_min_hz),
        srp_freq_max_hz=int(srp_freq_max_hz),
        localization_vad_enabled=bool(localization_vad_enabled),
        localization_track_hold_frames=int(localization_track_hold_frames),
        localization_max_assoc_distance_deg=float(localization_max_assoc_distance_deg),
        localization_velocity_alpha=float(localization_velocity_alpha),
        localization_angle_alpha=float(localization_angle_alpha),
        capon_spectrum_ema_alpha=float(capon_spectrum_ema_alpha),
        capon_peak_min_sharpness=float(capon_peak_min_sharpness),
        capon_peak_min_margin=float(capon_peak_min_margin),
        capon_hold_frames=int(capon_hold_frames),
        capon_freq_bin_subsample_stride=int(capon_freq_bin_subsample_stride),
        capon_freq_bin_min_hz=(None if capon_freq_bin_min_hz is None else int(capon_freq_bin_min_hz)),
        capon_freq_bin_max_hz=(None if capon_freq_bin_max_hz is None else int(capon_freq_bin_max_hz)),
        capon_use_cholesky_solve=bool(capon_use_cholesky_solve),
        capon_covariance_ema_alpha=float(capon_covariance_ema_alpha),
        capon_full_scan_every_n_updates=int(capon_full_scan_every_n_updates),
        capon_local_refine_enabled=bool(capon_local_refine_enabled),
        capon_local_refine_half_width_deg=float(capon_local_refine_half_width_deg),
        input_downsample_rate_hz=(None if input_downsample_rate_hz is None else int(input_downsample_rate_hz)),
        localization_snr_gating_enabled=bool(localization_snr_gating_enabled),
        localization_snr_threshold_db=float(localization_snr_threshold_db),
        localization_msc_variance_enabled=bool(localization_msc_variance_enabled),
        localization_msc_history_frames=int(localization_msc_history_frames),
        localization_hsda_enabled=bool(localization_hsda_enabled),
        localization_hsda_window_frames=int(localization_hsda_window_frames),
    )
    gt = build_framewise_speech_ground_truth(
        scene_config_path=staged_scene,
        scenario_metadata_path=staged_meta,
        frame_step_ms=float(fast_frame_ms),
    )
    n_frames = len(gt["time_s"])
    pred_lists, active_pred_deg, gated_flags, scores = _extract_trace_rows(summary, n_frames)
    active_truth = [float(v) if v is not None else float("nan") for v in gt["active_directions_deg"]]
    truth_lists = [[float(v) for v in row] for row in gt["active_target_directions_deg"]]
    active_errors = [
        _angular_error_deg(float(pred), float(truth)) if np.isfinite(pred) and np.isfinite(truth) else math.nan
        for pred, truth in zip(active_pred_deg, active_truth)
    ]

    per_frame_rows = []
    for idx in range(n_frames):
        per_frame_rows.append(
            {
                "frame_index": idx,
                "timestamp_ms": int(round(float(gt["time_s"][idx]) * 1000.0)),
                "active_speaker_id": int(gt["active_speaker_ids"][idx]),
                "active_truth_deg": "" if not np.isfinite(active_truth[idx]) else float(active_truth[idx]),
                "active_target_ids": json.dumps(gt["active_target_speaker_ids"][idx]),
                "active_target_deg": json.dumps(truth_lists[idx]),
                "pred_primary_deg": "" if not np.isfinite(active_pred_deg[idx]) else float(active_pred_deg[idx]),
                "pred_peaks_deg": json.dumps(pred_lists[idx]),
                "pred_peak_scores": json.dumps(scores[idx]) if scores[idx] is not None else "",
                "gated": int(bool(gated_flags[idx])),
            }
        )
    predictions_csv = bench_root / "per_frame_predictions" / f"{strategy}__{scene_cfg_path.stem}.csv"
    _write_csv(predictions_csv, per_frame_rows)
    errors_path = bench_root / "per_frame_errors" / f"{strategy}__{scene_cfg_path.stem}.npy"
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(errors_path, np.asarray(active_errors, dtype=np.float64))

    active_metrics = _compute_active_metrics(active_truth, active_pred_deg, gated_flags)
    topk_metrics = _compute_topk_metrics(pred_lists, truth_lists)

    scene_summary = {
        "strategy": strategy,
        "scene_id": scene_cfg_path.stem,
        "scene_config": str(staged_scene.resolve()),
        "summary_json": str((job_root / "summary.json").resolve()),
        "prediction_csv": str(predictions_csv.resolve()),
        "active_error_npy": str(errors_path.resolve()),
        "fast_frame_ms": int(fast_frame_ms),
        "slow_chunk_ms": int(slow_chunk_ms),
        "localization_window_ms": int(localization_window_ms),
        "localization_hop_ms": int(localization_hop_ms),
        "localization_grid_size": int(localization_grid_size),
        "srp_overlap": float(srp_overlap),
        "srp_freq_min_hz": int(srp_freq_min_hz),
        "srp_freq_max_hz": int(srp_freq_max_hz),
        "input_downsample_rate_hz": (None if input_downsample_rate_hz is None else int(input_downsample_rate_hz)),
        "localization_track_hold_frames": int(localization_track_hold_frames),
        "localization_max_assoc_distance_deg": float(localization_max_assoc_distance_deg),
        "localization_velocity_alpha": float(localization_velocity_alpha),
        "localization_angle_alpha": float(localization_angle_alpha),
        "capon_spectrum_ema_alpha": float(capon_spectrum_ema_alpha),
        "capon_peak_min_sharpness": float(capon_peak_min_sharpness),
        "capon_peak_min_margin": float(capon_peak_min_margin),
        "capon_hold_frames": int(capon_hold_frames),
        "capon_freq_bin_subsample_stride": int(capon_freq_bin_subsample_stride),
        "capon_freq_bin_min_hz": (None if capon_freq_bin_min_hz is None else int(capon_freq_bin_min_hz)),
        "capon_freq_bin_max_hz": (None if capon_freq_bin_max_hz is None else int(capon_freq_bin_max_hz)),
        "capon_use_cholesky_solve": bool(capon_use_cholesky_solve),
        "capon_covariance_ema_alpha": float(capon_covariance_ema_alpha),
        "capon_full_scan_every_n_updates": int(capon_full_scan_every_n_updates),
        "capon_local_refine_enabled": bool(capon_local_refine_enabled),
        "capon_local_refine_half_width_deg": float(capon_local_refine_half_width_deg),
        "localization_vad_enabled": bool(localization_vad_enabled),
        "localization_snr_gating_enabled": bool(localization_snr_gating_enabled),
        "localization_snr_threshold_db": float(localization_snr_threshold_db),
        "localization_msc_variance_enabled": bool(localization_msc_variance_enabled),
        "localization_msc_history_frames": int(localization_msc_history_frames),
        "localization_hsda_enabled": bool(localization_hsda_enabled),
        "localization_hsda_window_frames": int(localization_hsda_window_frames),
        "noise_gain_scale": float(noise_gain_scale),
        "fast_avg_ms": float(summary.get("fast_avg_ms", math.nan)),
        "fast_rtf": float(summary.get("fast_rtf", math.nan)),
        "active_metrics": active_metrics,
        "topk_metrics": topk_metrics,
    }
    scene_summary_path = bench_root / "scene_summaries" / f"{strategy}__{scene_cfg_path.stem}.json"
    _write_json(scene_summary_path, scene_summary)

    timeline_png = bench_root / "visualizations" / f"{strategy}__{scene_cfg_path.stem}__timeline.png"
    _plot_timeline(timeline_png, gt["time_s"], active_truth, active_pred_deg, strategy, scene_cfg_path.stem)

    return JobResult(
        strategy=strategy,
        scene_id=scene_cfg_path.stem,
        active_metrics=active_metrics,
        topk_metrics=topk_metrics,
        scene_summary=scene_summary,
        predictions_csv=str(predictions_csv.resolve()),
        timeline_png=str(timeline_png.resolve()),
    )


def _mean_metric(rows: list[dict[str, float | None]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    if not vals:
        return None
    return float(np.mean(vals))


def _aggregate(results: list[JobResult], strategies: list[str]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[JobResult]] = {}
    for result in results:
        grouped.setdefault(result.strategy, []).append(result)

    def _aggregate_mode(mode_key: str) -> list[dict[str, object]]:
        rows = []
        for strategy in strategies:
            strategy_rows = grouped.get(strategy, [])
            metric_rows = [getattr(result, mode_key) for result in strategy_rows]
            rows.append(
                {
                    "strategy": strategy,
                    "mae_deg": _mean_metric(metric_rows, "mae_deg"),
                    "acc_at_10": _mean_metric(metric_rows, "acc_at_10"),
                    "acc_at_20": _mean_metric(metric_rows, "acc_at_20"),
                    "acc_at_25": _mean_metric(metric_rows, "acc_at_25"),
                    "median_deg": _mean_metric(metric_rows, "median_deg"),
                    "p90_deg": _mean_metric(metric_rows, "p90_deg"),
                    "coverage_rate": _mean_metric(metric_rows, "coverage_rate"),
                    "gating_rate": _mean_metric(metric_rows, "gating_rate"),
                    "false_alarm_rate": _mean_metric(metric_rows, "false_alarm_rate"),
                    "stability_deg": _mean_metric(metric_rows, "stability_deg"),
                    "frame_match_rate": _mean_metric(metric_rows, "frame_match_rate"),
                    "miss_rate": _mean_metric(metric_rows, "miss_rate"),
                    "fast_avg_ms": float(np.mean([float(r.scene_summary["fast_avg_ms"]) for r in strategy_rows])) if strategy_rows else None,
                }
            )
        return rows

    return {
        "active_speaker_per_frame": _aggregate_mode("active_metrics"),
        "top_k_per_frame": _aggregate_mode("topk_metrics"),
    }


def _plot_metric_bars(out_root: Path, rows: list[dict[str, object]], prefix: str) -> None:
    labels = [str(row["strategy"]) for row in rows]
    metrics = {
        f"{prefix}_mae_by_strategy.png": ("mae_deg", "MAE (deg)"),
        f"{prefix}_acc25_by_strategy.png": ("acc_at_25", "Acc@25"),
        f"{prefix}_coverage_by_strategy.png": ("coverage_rate", "Coverage"),
        f"{prefix}_runtime_by_strategy.png": ("fast_avg_ms", "Fast avg ms"),
    }
    for filename, (field, ylabel) in metrics.items():
        values = [float(row[field]) if row.get(field) is not None else math.nan for row in rows]
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(labels)), values)
        plt.xticks(np.arange(len(labels)), labels, rotation=20)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out_root / filename, dpi=180)
        plt.close()


def _format_table(rows: list[dict[str, object]], title: str) -> str:
    lines = [
        title,
        "Strategy                  | MAE   | Acc@10 | Acc@20 | Acc@25 | Median | P90  | Coverage | Gating | Stability | FAR",
        "--------------------------|-------|--------|--------|--------|--------|------|----------|--------|-----------|-----",
    ]
    for row in rows:
        def _fmt_pct(value: object) -> str:
            if value is None or not np.isfinite(float(value)):
                return "N/A"
            return f"{100.0 * float(value):.1f}%"

        def _fmt_num(value: object) -> str:
            if value is None or not np.isfinite(float(value)):
                return "N/A"
            return f"{float(value):.1f}"

        lines.append(
            f"{str(row['strategy']):<26}| {_fmt_num(row['mae_deg'])} | {_fmt_pct(row['acc_at_10'])} | {_fmt_pct(row['acc_at_20'])} | {_fmt_pct(row['acc_at_25'])} | {_fmt_num(row['median_deg'])} | {_fmt_num(row['p90_deg'])} | {_fmt_pct(row['coverage_rate'])} | {_fmt_pct(row['gating_rate'])} | {_fmt_num(row['stability_deg'])} | {_fmt_pct(row['false_alarm_rate'])}"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime-parity framewise localization benchmark.")
    parser.add_argument("--scenes-root", default=str(DEFAULT_SCENES_ROOT))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--run-label", default="framewise_realtime")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=list(SUPPORTED_MIC_ARRAY_PROFILES))
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--eval-mode", choices=["active_speaker_per_frame", "top_k_per_frame", "both"], default="both")
    parser.add_argument("--fast-frame-ms", type=int, default=10)
    parser.add_argument("--slow-chunk-ms", type=int, default=200)
    parser.add_argument("--localization-window-ms", type=int, default=160)
    parser.add_argument("--localization-hop-ms", type=int, default=50)
    parser.add_argument("--localization-grid-size", type=int, default=72)
    parser.add_argument("--srp-overlap", type=float, default=0.2)
    parser.add_argument("--srp-freq-min-hz", type=int, default=1200)
    parser.add_argument("--srp-freq-max-hz", type=int, default=5400)
    parser.add_argument("--input-downsample-rate-hz", type=int, default=None)
    parser.add_argument("--localization-vad-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--localization-snr-gating-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--localization-snr-threshold-db", type=float, default=3.0)
    parser.add_argument("--localization-msc-variance-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--localization-msc-history-frames", type=int, default=6)
    parser.add_argument("--localization-hsda-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--localization-hsda-window-frames", type=int, default=5)
    parser.add_argument("--localization-track-hold-frames", type=int, default=5)
    parser.add_argument("--localization-max-assoc-distance-deg", type=float, default=20.0)
    parser.add_argument("--localization-velocity-alpha", type=float, default=0.35)
    parser.add_argument("--localization-angle-alpha", type=float, default=0.30)
    parser.add_argument("--capon-spectrum-ema-alpha", type=float, default=0.78)
    parser.add_argument("--capon-peak-min-sharpness", type=float, default=0.12)
    parser.add_argument("--capon-peak-min-margin", type=float, default=0.04)
    parser.add_argument("--capon-hold-frames", type=int, default=2)
    parser.add_argument("--capon-freq-bin-subsample-stride", type=int, default=1)
    parser.add_argument("--capon-freq-bin-min-hz", type=int, default=None)
    parser.add_argument("--capon-freq-bin-max-hz", type=int, default=None)
    parser.add_argument("--capon-use-cholesky-solve", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--capon-covariance-ema-alpha", type=float, default=0.0)
    parser.add_argument("--capon-full-scan-every-n-updates", type=int, default=1)
    parser.add_argument("--capon-local-refine-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--capon-local-refine-half-width-deg", type=float, default=30.0)
    parser.add_argument("--noise-gain-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / f"{run_id}_{args.run_label}"
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = sorted(Path(args.scenes_root).glob("*.json"))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    if not scenes:
        raise FileNotFoundError(f"No scenes found under {args.scenes_root}")

    strategies = list(args.strategies or DEFAULT_STRATEGIES)
    jobs = [
        {
            "strategy": strategy,
            "scene_path": str(scene.resolve()),
            "assets_root": str(Path(args.assets_root).resolve()),
            "out_root": str(out_root.resolve()),
            "profile": str(args.profile),
            "noise_gain_scale": float(args.noise_gain_scale),
            "fast_frame_ms": int(args.fast_frame_ms),
            "slow_chunk_ms": int(args.slow_chunk_ms),
            "localization_window_ms": int(args.localization_window_ms),
            "localization_hop_ms": int(args.localization_hop_ms),
            "localization_grid_size": int(args.localization_grid_size),
            "srp_overlap": float(args.srp_overlap),
            "srp_freq_min_hz": int(args.srp_freq_min_hz),
            "srp_freq_max_hz": int(args.srp_freq_max_hz),
            "input_downsample_rate_hz": (None if args.input_downsample_rate_hz is None else int(args.input_downsample_rate_hz)),
            "localization_vad_enabled": bool(args.localization_vad_enabled),
            "localization_snr_gating_enabled": bool(args.localization_snr_gating_enabled),
            "localization_snr_threshold_db": float(args.localization_snr_threshold_db),
            "localization_msc_variance_enabled": bool(args.localization_msc_variance_enabled),
            "localization_msc_history_frames": int(args.localization_msc_history_frames),
            "localization_hsda_enabled": bool(args.localization_hsda_enabled),
            "localization_hsda_window_frames": int(args.localization_hsda_window_frames),
            "localization_track_hold_frames": int(args.localization_track_hold_frames),
            "localization_max_assoc_distance_deg": float(args.localization_max_assoc_distance_deg),
            "localization_velocity_alpha": float(args.localization_velocity_alpha),
            "localization_angle_alpha": float(args.localization_angle_alpha),
            "capon_spectrum_ema_alpha": float(args.capon_spectrum_ema_alpha),
            "capon_peak_min_sharpness": float(args.capon_peak_min_sharpness),
            "capon_peak_min_margin": float(args.capon_peak_min_margin),
            "capon_hold_frames": int(args.capon_hold_frames),
            "capon_freq_bin_subsample_stride": int(args.capon_freq_bin_subsample_stride),
            "capon_freq_bin_min_hz": (None if args.capon_freq_bin_min_hz is None else int(args.capon_freq_bin_min_hz)),
            "capon_freq_bin_max_hz": (None if args.capon_freq_bin_max_hz is None else int(args.capon_freq_bin_max_hz)),
            "capon_use_cholesky_solve": bool(args.capon_use_cholesky_solve),
            "capon_covariance_ema_alpha": float(args.capon_covariance_ema_alpha),
            "capon_full_scan_every_n_updates": int(args.capon_full_scan_every_n_updates),
            "capon_local_refine_enabled": bool(args.capon_local_refine_enabled),
            "capon_local_refine_half_width_deg": float(args.capon_local_refine_half_width_deg),
        }
        for strategy in strategies
        for scene in scenes
    ]

    results: list[JobResult] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_map = {pool.submit(_run_job, **job): (job["strategy"], Path(job["scene_path"]).stem) for job in jobs}
        for future in as_completed(future_map):
            strategy, scene_id = future_map[future]
            print(f"completed {strategy} :: {scene_id}")
            results.append(future.result())

    aggregates = _aggregate(results, strategies)
    selected_modes = (
        [args.eval_mode]
        if args.eval_mode != "both"
        else ["active_speaker_per_frame", "top_k_per_frame"]
    )
    report_parts = []
    for mode in selected_modes:
        rows = aggregates[mode]
        report_parts.append(_format_table(rows, f"[{mode}]"))
        _plot_metric_bars(out_root, rows, mode)

    payload = {
        "run_id": run_id,
        "run_label": str(args.run_label),
        "profile": str(args.profile),
        "scenes_root": str(Path(args.scenes_root).resolve()),
        "assets_root": str(Path(args.assets_root).resolve()),
        "n_scenes": len(scenes),
        "strategies": strategies,
        "fast_frame_ms": int(args.fast_frame_ms),
        "slow_chunk_ms": int(args.slow_chunk_ms),
        "localization_window_ms": int(args.localization_window_ms),
        "localization_hop_ms": int(args.localization_hop_ms),
        "localization_grid_size": int(args.localization_grid_size),
        "srp_overlap": float(args.srp_overlap),
        "srp_freq_min_hz": int(args.srp_freq_min_hz),
        "srp_freq_max_hz": int(args.srp_freq_max_hz),
        "input_downsample_rate_hz": (None if args.input_downsample_rate_hz is None else int(args.input_downsample_rate_hz)),
        "localization_vad_enabled": bool(args.localization_vad_enabled),
        "localization_snr_gating_enabled": bool(args.localization_snr_gating_enabled),
        "localization_snr_threshold_db": float(args.localization_snr_threshold_db),
        "localization_msc_variance_enabled": bool(args.localization_msc_variance_enabled),
        "localization_msc_history_frames": int(args.localization_msc_history_frames),
        "localization_hsda_enabled": bool(args.localization_hsda_enabled),
        "localization_hsda_window_frames": int(args.localization_hsda_window_frames),
        "localization_track_hold_frames": int(args.localization_track_hold_frames),
        "localization_max_assoc_distance_deg": float(args.localization_max_assoc_distance_deg),
        "localization_velocity_alpha": float(args.localization_velocity_alpha),
        "localization_angle_alpha": float(args.localization_angle_alpha),
        "capon_spectrum_ema_alpha": float(args.capon_spectrum_ema_alpha),
        "capon_peak_min_sharpness": float(args.capon_peak_min_sharpness),
        "capon_peak_min_margin": float(args.capon_peak_min_margin),
        "capon_hold_frames": int(args.capon_hold_frames),
        "capon_freq_bin_subsample_stride": int(args.capon_freq_bin_subsample_stride),
        "capon_freq_bin_min_hz": (None if args.capon_freq_bin_min_hz is None else int(args.capon_freq_bin_min_hz)),
        "capon_freq_bin_max_hz": (None if args.capon_freq_bin_max_hz is None else int(args.capon_freq_bin_max_hz)),
        "capon_use_cholesky_solve": bool(args.capon_use_cholesky_solve),
        "capon_covariance_ema_alpha": float(args.capon_covariance_ema_alpha),
        "capon_full_scan_every_n_updates": int(args.capon_full_scan_every_n_updates),
        "capon_local_refine_enabled": bool(args.capon_local_refine_enabled),
        "capon_local_refine_half_width_deg": float(args.capon_local_refine_half_width_deg),
        "noise_gain_scale": float(args.noise_gain_scale),
        "eval_mode": str(args.eval_mode),
        "aggregates": aggregates,
        "scene_results": [result.scene_summary for result in sorted(results, key=lambda item: (item.strategy, item.scene_id))],
    }
    _write_json(out_root / "benchmark_results.json", payload)
    (out_root / "benchmark_results.txt").write_text("\n\n".join(report_parts) + "\n", encoding="utf-8")
    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_root.resolve(), target_is_directory=True)


if __name__ == "__main__":
    main()
