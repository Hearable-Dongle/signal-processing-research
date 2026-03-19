from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Any

from simulation.mic_array_profiles import mic_positions_xyz

from beamforming.benchmark.data_collection_benchmark import (
    _discover_recordings,
    _extract_if_needed,
    _run_recording_method_job,
)


def _get_optuna():
    import optuna

    return optuna


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [
        float(row[key])
        for row in rows
        if key in row and row[key] is not None and math.isfinite(float(row[key]))
    ]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-gate Capon tuning on real data.")
    parser.add_argument(
        "--input-paths",
        nargs="+",
        required=True,
        help="Real-data collection roots.",
    )
    parser.add_argument(
        "--out-root",
        default="beamforming/benchmark/capon_realdata_tuning",
    )
    parser.add_argument("--study-name", default="capon_realdata_hardgate")
    parser.add_argument("--time-budget-min", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial-job-workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--mic-array-profile", choices=["respeaker_v3_0457", "respeaker_xvf3800_0650"], default="respeaker_xvf3800_0650")
    return parser.parse_args()


def _suggest_params(trial) -> dict[str, Any]:
    full_scan_every = trial.suggest_categorical("capon_full_scan_every_n_updates", [2, 4, 8])
    return {
        "input_downsample_rate_hz": trial.suggest_categorical("input_downsample_rate_hz", [8000]),
        "fast_frame_ms": 10,
        "localization_hop_ms": trial.suggest_categorical("localization_hop_ms", [150, 200, 250]),
        "localization_window_ms": trial.suggest_categorical("localization_window_ms", [160, 200]),
        "localization_grid_size": trial.suggest_categorical("localization_grid_size", [12, 24, 36]),
        "capon_spectrum_ema_alpha": trial.suggest_categorical("capon_spectrum_ema_alpha", [0.65, 0.78]),
        "capon_hold_frames": trial.suggest_categorical("capon_hold_frames", [1, 2]),
        "localization_track_hold_frames": trial.suggest_categorical("localization_track_hold_frames", [2, 3, 5]),
        "localization_max_assoc_distance_deg": trial.suggest_categorical("localization_max_assoc_distance_deg", [10.0, 12.0, 16.0, 20.0]),
        "localization_velocity_alpha": trial.suggest_categorical("localization_velocity_alpha", [0.20, 0.35]),
        "localization_angle_alpha": trial.suggest_categorical("localization_angle_alpha", [0.18, 0.30]),
        "capon_peak_min_sharpness": trial.suggest_categorical("capon_peak_min_sharpness", [0.04, 0.08]),
        "capon_peak_min_margin": trial.suggest_categorical("capon_peak_min_margin", [0.01, 0.02]),
        "capon_freq_bin_subsample_stride": trial.suggest_categorical("capon_freq_bin_subsample_stride", [1, 2, 4]),
        "capon_freq_bin_min_hz": trial.suggest_categorical("capon_freq_bin_min_hz", [800, 1000]),
        "capon_freq_bin_max_hz": trial.suggest_categorical("capon_freq_bin_max_hz", [2200, 2600]),
        "capon_use_cholesky_solve": True,
        "capon_covariance_ema_alpha": trial.suggest_categorical("capon_covariance_ema_alpha", [0.85, 0.93, 0.97]),
        "capon_full_scan_every_n_updates": full_scan_every,
        "capon_local_refine_enabled": True,
        "capon_local_refine_half_width_deg": trial.suggest_categorical("capon_local_refine_half_width_deg", [12.0, 20.0, 30.0]),
    }


def _objective_factory(
    *,
    recordings_by_dataset: list[tuple[str, list[tuple[str, Path]]]],
    out_dir: Path,
    mic_array_profile: str,
    mic_geometry_xyz,
    trial_job_workers: int,
):
    def objective(trial):
        params = _suggest_params(trial)
        trial_dir = out_dir / "trials" / f"trial_{trial.number:04d}"
        rows: list[dict[str, Any]] = []
        jobs: list[dict[str, Any]] = []
        for dataset_label, recordings in recordings_by_dataset:
            dataset_out = trial_dir / dataset_label
            for recording_id, recording_dir in recordings:
                jobs.append({
                    "recording_id": recording_id,
                    "recording_dir": recording_dir,
                    "method": "delay_sum",
                    "out_dir": dataset_out,
                    "mic_array_profile": mic_array_profile,
                    "mic_geometry_xyz": mic_geometry_xyz,
                    "algorithm_mode": "speaker_tracking_single_active",
                    "separation_mode": "single_dominant_no_separator",
                    "localization_window_ms": int(params["localization_window_ms"]),
                    "localization_hop_ms": int(params["localization_hop_ms"]),
                    "fast_frame_ms": int(params["fast_frame_ms"]),
                    "localization_grid_size": int(params["localization_grid_size"]),
                    "localization_overlap": 0.2,
                    "localization_freq_low_hz": 1200,
                    "localization_freq_high_hz": 5400,
                    "localization_pair_selection_mode": "all",
                    "localization_vad_enabled": False,
                    "capon_peak_min_sharpness": float(params["capon_peak_min_sharpness"]),
                    "capon_peak_min_margin": float(params["capon_peak_min_margin"]),
                    "localization_track_hold_frames": int(params["localization_track_hold_frames"]),
                    "localization_velocity_alpha": float(params["localization_velocity_alpha"]),
                    "localization_angle_alpha": float(params["localization_angle_alpha"]),
                    "srp_peak_ema_alpha": 0.35,
                    "srp_peak_hold_frames": 4,
                    "capon_spectrum_ema_alpha": float(params["capon_spectrum_ema_alpha"]),
                    "capon_hold_frames": int(params["capon_hold_frames"]),
                    "localization_max_assoc_distance_deg": float(params["localization_max_assoc_distance_deg"]),
                    "speaker_history_size": 8,
                    "speaker_activation_min_predictions": 3,
                    "speaker_match_window_deg": 18.0,
                    "single_active_min_observation_score": 0.2,
                    "centroid_association_mode": "hard_window",
                    "centroid_association_sigma_deg": 10.0,
                    "centroid_association_min_score": 0.15,
                    "own_voice_suppression_mode": "off",
                    "suppressed_user_voice_doa_deg": None,
                    "suppressed_user_match_window_deg": 33.0,
                    "suppressed_user_null_on_frames": 3,
                    "suppressed_user_null_off_frames": 10,
                    "suppressed_user_gate_attenuation_db": 18.0,
                    "suppressed_user_target_conflict_deg": 30.0,
                    "suppressed_user_speaker_name": None,
                    "ground_truth_transition_ignore_sec": 1.5,
                    "enhancement_tier": "custom",
                    "output_enhancer_mode": "off",
                    "postfilter_enabled": True,
                    "postfilter_method": "rnnoise",
                    "postfilter_noise_source": "tracked_mono",
                    "postfilter_input_source": "beamformed_mono",
                    "postfilter_noise_ema_alpha": 0.02,
                    "postfilter_speech_ema_alpha": 0.01,
                    "postfilter_gain_floor": 0.18,
                    "postfilter_gain_ema_alpha": 0.35,
                    "postfilter_dd_alpha": 0.98,
                    "postfilter_noise_update_speech_scale": 0.0,
                    "postfilter_oversubtraction_alpha": 1.0,
                    "postfilter_spectral_floor_beta": 0.01,
                    "postfilter_gain_max_step_db": 2.5,
                    "rnnoise_wet_mix": 0.9,
                    "rnnoise_output_lowpass_cutoff_hz": 7500.0,
                    "rnnoise_output_notch_freq_hz": 500.0,
                    "rnnoise_output_notch_q": 20.0,
                    "rnnoise_residual_ema_enabled": False,
                    "rnnoise_residual_ema_alpha": 0.0,
                    "beamformer_snapshot_frame_indices": None,
                    "delay_sum_update_min_delta_deg": 3.0,
                    "delay_sum_crossfade_frames": 1,
                    "delay_sum_use_smoothed_doa": True,
                    "delay_sum_subtractive_alpha": 0.9,
                    "delay_sum_subtractive_interferer_doa_deg": None,
                    "delay_sum_subtractive_multi_offset_deg": 10.0,
                    "delay_sum_subtractive_use_suppressed_user_doa": True,
                    "delay_sum_subtractive_output_clip_guard": True,
                    "slow_chunk_ms": 2000,
                    "slow_chunk_hop_ms": 1000,
                    "fast_path_reference_mode": "speaker_map",
                    "output_normalization_enabled": True,
                    "output_allow_amplification": False,
                    "localization_backend": "capon_1src",
                    "tracking_mode": "doa_centroid_v1",
                    "control_mode": "speaker_tracking_mode",
                    "direction_long_memory_enabled": False,
                    "identity_backend": "mfcc_legacy",
                    "identity_speaker_embedding_model": "wavlm_base_plus_sv",
                    "max_speakers_hint": 1,
                    "assume_single_speaker": True,
                    "use_ground_truth_doa_override": False,
                    "mvdr_hop_ms": None,
                    "fd_analysis_window_ms": 32.0,
                    "robust_target_band_width_deg": 12.0,
                    "robust_target_band_max_freq_hz": 4000.0,
                    "robust_target_band_condition_limit": 5000.0,
                    "fd_noise_covariance_mode": "estimated_target_subtractive_frozen",
                    "target_activity_rnn_update_mode": "estimated_target_activity",
                    "target_activity_detector_mode": "localization_peak_confidence",
                    "target_activity_detector_backend": "webrtc_fused",
                    "target_activity_update_every_n_fast_frames": 2,
                    "fd_cov_ema_alpha": 0.08,
                    "fd_diag_load": 1e-3,
                    "fd_trace_diagonal_loading_factor": 1e-2,
                    "fd_identity_blend_alpha": 0.0,
                    "beamformer_rnn_skip_refresh_when_clean": False,
                    "beamformer_rnn_dirty_threshold": 0.03,
                    "beamformer_rnn_dirty_eps": 1e-8,
                    "beamformer_rnn_dirty_stat": "max",
                    "beamformer_sparse_solve_enabled": False,
                    "beamformer_sparse_solve_stride": 1,
                    "beamformer_sparse_solve_min_freq_hz": 200.0,
                    "beamformer_sparse_solve_interp": "linear_complex",
                    "beamformer_weight_reuse_enabled": True,
                    "beamformer_weight_smoothing_alpha": 1.0,
                    "beamformer_doa_refresh_tolerance_deg": 5.0,
                    "target_activity_low_threshold": 0.35,
                    "target_activity_high_threshold": 0.55,
                    "target_activity_enter_frames": 2,
                    "target_activity_exit_frames": 6,
                    "fd_cov_update_scale_target_active": 0.0,
                    "fd_cov_update_scale_target_inactive": 1.0,
                    "input_downsample_rate_hz": int(params["input_downsample_rate_hz"]),
                    "capon_freq_bin_subsample_stride": int(params["capon_freq_bin_subsample_stride"]),
                    "capon_freq_bin_min_hz": int(params["capon_freq_bin_min_hz"]),
                    "capon_freq_bin_max_hz": int(params["capon_freq_bin_max_hz"]),
                    "capon_use_cholesky_solve": bool(params["capon_use_cholesky_solve"]),
                    "capon_covariance_ema_alpha": float(params["capon_covariance_ema_alpha"]),
                    "capon_full_scan_every_n_updates": int(params["capon_full_scan_every_n_updates"]),
                    "capon_local_refine_enabled": bool(params["capon_local_refine_enabled"]),
                    "capon_local_refine_half_width_deg": float(params["capon_local_refine_half_width_deg"]),
                    "dataset_label": dataset_label,
                })
        with ProcessPoolExecutor(max_workers=max(1, int(trial_job_workers))) as pool:
            future_map = {
                pool.submit(_run_recording_method_job, **{k: v for k, v in job.items() if k != "dataset_label"}): job["dataset_label"]
                for job in jobs
            }
            for future in as_completed(future_map):
                row = future.result()
                row["dataset_label"] = str(future_map[future])
                rows.append(row)

        fast_rtf_mean = _mean(rows, "fast_rtf")
        acc10_mean = _mean(rows, "gt_trace_active_acc_10")
        acc20_mean = _mean(rows, "gt_trace_active_acc_20")
        mae_mean = _mean(rows, "gt_trace_active_mae_deg")
        valid = bool(math.isfinite(fast_rtf_mean) and fast_rtf_mean < 0.3)
        objective = (
            (0.0 if valid else 1000.0)
            + (0.0 if not math.isfinite(mae_mean) else mae_mean)
            + 100.0 * (1.0 - (0.0 if not math.isfinite(acc10_mean) else acc10_mean))
            + 30.0 * (1.0 - (0.0 if not math.isfinite(acc20_mean) else acc20_mean))
            + 0.1 * (0.0 if not math.isfinite(fast_rtf_mean) else fast_rtf_mean)
        )

        trial.set_user_attr("valid_under_rtf_gate", valid)
        trial.set_user_attr("fast_rtf_mean", fast_rtf_mean)
        trial.set_user_attr("gt_trace_active_acc_10_mean", acc10_mean)
        trial.set_user_attr("gt_trace_active_acc_20_mean", acc20_mean)
        trial.set_user_attr("gt_trace_active_mae_mean", mae_mean)
        trial.set_user_attr("params_echo", params)
        _write_json(
            trial_dir / "trial_summary.json",
            {
                "trial_number": int(trial.number),
                "objective": float(objective),
                "valid_under_rtf_gate": bool(valid),
                "fast_rtf_mean": fast_rtf_mean,
                "gt_trace_active_acc_10_mean": acc10_mean,
                "gt_trace_active_acc_20_mean": acc20_mean,
                "gt_trace_active_mae_mean": mae_mean,
                "params": params,
            },
        )
        _write_csv(trial_dir / "trial_rows.csv", rows)
        return float(objective)

    return objective


def _resolve_recordings(input_paths: list[str]) -> list[tuple[str, list[tuple[str, Path]]]]:
    groups: list[tuple[str, list[tuple[str, Path]]]] = []
    temp_dirs = []
    for raw_path in input_paths:
        input_path = Path(raw_path).resolve()
        extracted_root, temp_dir = _extract_if_needed(input_path)
        if temp_dir is not None:
            temp_dirs.append(temp_dir)
        resolved_root, recordings = _discover_recordings(extracted_root.resolve())
        groups.append((resolved_root.name, recordings))
    return groups, temp_dirs


def main() -> None:
    args = _parse_args()
    optuna = _get_optuna()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root).resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings_by_dataset, temp_dirs = _resolve_recordings(list(args.input_paths))
    mic_geometry_xyz = mic_positions_xyz(str(args.mic_array_profile))
    study = optuna.create_study(
        study_name=str(args.study_name),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(args.seed)),
    )
    study.optimize(
        _objective_factory(
            recordings_by_dataset=recordings_by_dataset,
            out_dir=out_dir,
            mic_array_profile=str(args.mic_array_profile),
            mic_geometry_xyz=mic_geometry_xyz,
            trial_job_workers=int(args.trial_job_workers),
        ),
        timeout=int(float(args.time_budget_min) * 60.0),
        show_progress_bar=False,
        catch=(Exception,),
    )

    trial_rows: list[dict[str, Any]] = []
    valid_trials: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "trial_number": int(trial.number),
            "state": str(trial.state.name),
            "value": (None if trial.value is None else float(trial.value)),
            "valid_under_rtf_gate": bool(trial.user_attrs.get("valid_under_rtf_gate", False)),
            "fast_rtf_mean": trial.user_attrs.get("fast_rtf_mean"),
            "gt_trace_active_acc_10_mean": trial.user_attrs.get("gt_trace_active_acc_10_mean"),
            "gt_trace_active_acc_20_mean": trial.user_attrs.get("gt_trace_active_acc_20_mean"),
            "gt_trace_active_mae_mean": trial.user_attrs.get("gt_trace_active_mae_mean"),
            **{str(k): v for k, v in (trial.user_attrs.get("params_echo") or {}).items()},
        }
        trial_rows.append(row)
        if row["valid_under_rtf_gate"] and row["state"] == "COMPLETE":
            valid_trials.append(row)
    valid_trials.sort(
        key=lambda r: (
            -float(r.get("gt_trace_active_acc_10_mean") or -1.0),
            -float(r.get("gt_trace_active_acc_20_mean") or -1.0),
            float(r.get("gt_trace_active_mae_mean") or 1e9),
            float(r.get("fast_rtf_mean") or 1e9),
        )
    )
    _write_csv(out_dir / "trials_export.csv", trial_rows)
    _write_json(
        out_dir / "study_summary.json",
        {
            "run_id": run_id,
            "study_name": str(args.study_name),
            "time_budget_min": float(args.time_budget_min),
            "input_paths": [str(Path(p).resolve()) for p in args.input_paths],
            "valid_trial_count": len(valid_trials),
            "best_valid_trial": (valid_trials[0] if valid_trials else None),
        },
    )
    if valid_trials:
        _write_json(out_dir / "best_valid_config.json", valid_trials[0])
    else:
        _write_json(out_dir / "no_valid_config_found.json", {"reason": "No trial satisfied mean fast_rtf < 0.3"})

    for temp_dir in temp_dirs:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
