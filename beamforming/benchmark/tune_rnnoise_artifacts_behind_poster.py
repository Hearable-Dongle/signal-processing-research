from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simulation.mic_array_profiles import mic_positions_xyz

from beamforming.benchmark.data_collection_benchmark import _discover_recordings, _extract_if_needed, _run_recording_method_job


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    config: dict[str, Any]
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    root: Path
    recordings: list[tuple[str, Path]]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [
        float(row[key])
        for row in rows
        if key in row and row[key] is not None and math.isfinite(float(row[key]))
    ]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _trial_summary(rows: list[dict[str, Any]], config: dict[str, Any], trial_id: str) -> dict[str, Any]:
    baseline_lsd_guard = float(config.get("_baseline_speech_band_lsd_mean", float("inf")))
    speech_corr = _mean(rows, "speech_envelope_corr")
    speech_lsd = _mean(rows, "speech_band_lsd_mean")
    valid = bool(
        math.isfinite(speech_corr)
        and speech_corr >= 0.80
        and math.isfinite(speech_lsd)
        and speech_lsd <= baseline_lsd_guard + 0.20
    )
    artifact_score = (
        (1000.0 * _mean(rows, "artifact_500hz_cluster_ratio"))
        + (200.0 * _mean(rows, "artifact_hf_jump_count"))
        + (10.0 * _mean(rows, "artifact_hf_flux_p95"))
        + (100.0 * _mean(rows, "artifact_lowband_hum_ratio"))
        + (10.0 * _mean(rows, "artifact_hf_burst_ratio"))
    )
    return {
        "trial_id": trial_id,
        "valid": valid,
        "artifact_score": float(artifact_score),
        "n_rows": len(rows),
        "artifact_500hz_cluster_ratio_mean": _mean(rows, "artifact_500hz_cluster_ratio"),
        "artifact_hf_jump_count_mean": _mean(rows, "artifact_hf_jump_count"),
        "artifact_hf_flux_p95_mean": _mean(rows, "artifact_hf_flux_p95"),
        "artifact_hf_burst_ratio_mean": _mean(rows, "artifact_hf_burst_ratio"),
        "artifact_lowband_hum_ratio_mean": _mean(rows, "artifact_lowband_hum_ratio"),
        "speech_band_lsd_mean_mean": speech_lsd,
        "speech_envelope_corr_mean": speech_corr,
        "fast_rtf_mean": _mean(rows, "fast_rtf"),
        "rnnoise_output_lowpass_cutoff_hz": float(config["rnnoise_output_lowpass_cutoff_hz"]),
        "rnnoise_output_notch_freq_hz": float(config["rnnoise_output_notch_freq_hz"]),
        "rnnoise_output_notch_q": float(config["rnnoise_output_notch_q"]),
        "rnnoise_output_highpass_cutoff_hz": float(config["rnnoise_output_highpass_cutoff_hz"]),
        "rnnoise_vad_blend_gamma": float(config["rnnoise_vad_blend_gamma"]),
        "rnnoise_vad_max_speech_preserve": float(config["rnnoise_vad_max_speech_preserve"]),
        "rnnoise_residual_highband_enabled": bool(config["rnnoise_residual_highband_enabled"]),
        "rnnoise_residual_highband_cutoff_hz": float(config["rnnoise_residual_highband_cutoff_hz"]),
        "rnnoise_residual_highband_gain": float(config["rnnoise_residual_highband_gain"]),
        "rnnoise_residual_jump_limit_enabled": bool(config["rnnoise_residual_jump_limit_enabled"]),
        "rnnoise_residual_jump_limit_band_low_hz": float(config["rnnoise_residual_jump_limit_band_low_hz"]),
        "rnnoise_residual_jump_limit_rise_db_per_frame": float(config["rnnoise_residual_jump_limit_rise_db_per_frame"]),
    }


def _ranking_key(summary: dict[str, Any]) -> tuple[float, float, float]:
    penalty = 0.0 if bool(summary["valid"]) else 1e9
    return (
        penalty + float(summary["artifact_500hz_cluster_ratio_mean"]),
        float(summary["artifact_hf_jump_count_mean"]),
        float(summary["artifact_hf_flux_p95_mean"]),
    )


def _ranking_key_with_hum(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    penalty = 0.0 if bool(summary["valid"]) else 1e9
    return (
        penalty + float(summary["artifact_500hz_cluster_ratio_mean"]),
        float(summary["artifact_hf_jump_count_mean"]),
        float(summary["artifact_hf_flux_p95_mean"]),
        float(summary["artifact_lowband_hum_ratio_mean"]),
    )


def _run_trial(
    *,
    trial_id: str,
    trial_dir: Path,
    datasets: list[DatasetSpec],
    mic_array_profile: str,
    mic_geometry_xyz,
    config: dict[str, Any],
) -> TrialResult:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_trial_dir = trial_dir / dataset.label
        for recording_id, recording_dir in dataset.recordings:
            row = _run_recording_method_job(
                recording_id=recording_id,
                recording_dir=recording_dir,
                method="delay_sum",
                out_dir=dataset_trial_dir,
                mic_array_profile=mic_array_profile,
                mic_geometry_xyz=mic_geometry_xyz,
                algorithm_mode="speaker_tracking_single_active",
                separation_mode="single_dominant_no_separator",
                localization_window_ms=200,
                localization_hop_ms=200,
                fast_frame_ms=10,
                localization_grid_size=36,
                localization_overlap=0.2,
                localization_freq_low_hz=1200,
                localization_freq_high_hz=3000,
                localization_pair_selection_mode="all",
                localization_vad_enabled=False,
                localization_track_hold_frames=5,
                localization_velocity_alpha=0.35,
                localization_angle_alpha=0.30,
                srp_peak_ema_alpha=0.35,
                srp_peak_hold_frames=4,
                capon_spectrum_ema_alpha=0.78,
                capon_hold_frames=1,
                localization_max_assoc_distance_deg=20.0,
                capon_peak_min_sharpness=0.08,
                capon_peak_min_margin=0.02,
                speaker_history_size=8,
                speaker_activation_min_predictions=3,
                speaker_match_window_deg=18.0,
                single_active_min_observation_score=0.2,
                centroid_association_mode="hard_window",
                centroid_association_sigma_deg=10.0,
                centroid_association_min_score=0.15,
                own_voice_suppression_mode="off",
                suppressed_user_voice_doa_deg=None,
                suppressed_user_match_window_deg=33.0,
                suppressed_user_null_on_frames=3,
                suppressed_user_null_off_frames=10,
                suppressed_user_gate_attenuation_db=18.0,
                suppressed_user_target_conflict_deg=30.0,
                suppressed_user_speaker_name=None,
                ground_truth_transition_ignore_sec=0.0,
                enhancement_tier="custom",
                output_enhancer_mode="off",
                postfilter_enabled=True,
                postfilter_method="rnnoise",
                postfilter_noise_source="tracked_mono",
                postfilter_input_source="beamformed_mono",
                postfilter_noise_ema_alpha=0.02,
                postfilter_speech_ema_alpha=0.01,
                postfilter_gain_floor=0.22,
                postfilter_gain_ema_alpha=0.3,
                postfilter_dd_alpha=0.92,
                postfilter_noise_update_speech_scale=0.0,
                postfilter_oversubtraction_alpha=1.0,
                postfilter_spectral_floor_beta=0.01,
                postfilter_gain_max_step_db=2.5,
                rnnoise_wet_mix=0.9,
                rnnoise_input_highpass_enabled=True,
                rnnoise_input_highpass_cutoff_hz=80.0,
                rnnoise_output_highpass_enabled=True,
                rnnoise_output_highpass_cutoff_hz=float(config["rnnoise_output_highpass_cutoff_hz"]),
                rnnoise_output_lowpass_cutoff_hz=float(config["rnnoise_output_lowpass_cutoff_hz"]),
                rnnoise_output_notch_freq_hz=float(config["rnnoise_output_notch_freq_hz"]),
                rnnoise_output_notch_q=float(config["rnnoise_output_notch_q"]),
                rnnoise_vad_adaptive_blend_enabled=True,
                rnnoise_vad_blend_gamma=float(config["rnnoise_vad_blend_gamma"]),
                rnnoise_vad_min_speech_preserve=0.15,
                rnnoise_vad_max_speech_preserve=float(config["rnnoise_vad_max_speech_preserve"]),
                rnnoise_residual_highband_enabled=bool(config["rnnoise_residual_highband_enabled"]),
                rnnoise_residual_highband_cutoff_hz=float(config["rnnoise_residual_highband_cutoff_hz"]),
                rnnoise_residual_highband_gain=float(config["rnnoise_residual_highband_gain"]),
                rnnoise_residual_jump_limit_enabled=bool(config["rnnoise_residual_jump_limit_enabled"]),
                rnnoise_residual_jump_limit_band_low_hz=float(config["rnnoise_residual_jump_limit_band_low_hz"]),
                rnnoise_residual_jump_limit_rise_db_per_frame=float(config["rnnoise_residual_jump_limit_rise_db_per_frame"]),
                rnnoise_residual_ema_enabled=False,
                rnnoise_residual_ema_alpha=0.0,
                beamformer_snapshot_frame_indices=(5, 10, 20),
                delay_sum_update_min_delta_deg=3.0,
                delay_sum_crossfade_frames=1,
                delay_sum_use_smoothed_doa=True,
                delay_sum_subtractive_alpha=0.5,
                delay_sum_subtractive_interferer_doa_deg=None,
                delay_sum_subtractive_multi_offset_deg=10.0,
                delay_sum_subtractive_use_suppressed_user_doa=True,
                delay_sum_subtractive_output_clip_guard=True,
                slow_chunk_ms=2000,
                slow_chunk_hop_ms=1000,
                fast_path_reference_mode="speaker_map",
                output_normalization_enabled=True,
                output_allow_amplification=False,
                localization_backend="capon_1src",
                tracking_mode="doa_centroid_v1",
                control_mode="speaker_tracking_mode",
                direction_long_memory_enabled=False,
                identity_backend="mfcc_legacy",
                identity_speaker_embedding_model="wavlm_base_plus_sv",
                max_speakers_hint=1,
                assume_single_speaker=True,
                use_ground_truth_doa_override=True,
                mvdr_hop_ms=None,
                fd_analysis_window_ms=32.0,
                robust_target_band_width_deg=12.0,
                robust_target_band_max_freq_hz=4000.0,
                robust_target_band_condition_limit=5000.0,
                fd_noise_covariance_mode="estimated_target_subtractive_frozen",
                target_activity_rnn_update_mode="estimated_target_activity",
                target_activity_detector_mode="localization_peak_confidence",
                target_activity_detector_backend="webrtc_fused",
                target_activity_update_every_n_fast_frames=2,
                fd_cov_ema_alpha=0.08,
                fd_diag_load=1e-3,
                fd_trace_diagonal_loading_factor=1e-2,
                fd_identity_blend_alpha=0.0,
                beamformer_rnn_skip_refresh_when_clean=False,
                beamformer_rnn_dirty_threshold=0.03,
                beamformer_rnn_dirty_eps=1e-8,
                beamformer_rnn_dirty_stat="max",
                beamformer_sparse_solve_enabled=False,
                beamformer_sparse_solve_stride=1,
                beamformer_sparse_solve_min_freq_hz=200.0,
                beamformer_sparse_solve_interp="linear_complex",
                beamformer_weight_reuse_enabled=True,
                beamformer_weight_smoothing_alpha=1.0,
                beamformer_doa_refresh_tolerance_deg=5.0,
                target_activity_low_threshold=0.35,
                target_activity_high_threshold=0.55,
                target_activity_enter_frames=2,
                target_activity_exit_frames=6,
                fd_cov_update_scale_target_active=0.0,
                fd_cov_update_scale_target_inactive=1.0,
                input_downsample_rate_hz=16000,
                capon_freq_bin_subsample_stride=2,
                capon_freq_bin_min_hz=1000,
                capon_freq_bin_max_hz=2200,
                capon_use_cholesky_solve=True,
                capon_covariance_ema_alpha=0.97,
                capon_full_scan_every_n_updates=4,
                capon_local_refine_enabled=True,
                capon_local_refine_half_width_deg=12.0,
            )
            row["dataset_label"] = str(dataset.label)
            row["dataset_root"] = str(dataset.root)
            rows.append(row)
    summary = _trial_summary(rows, config, trial_id)
    payload = {"trial_id": trial_id, "config": config, "summary": summary, "rows": rows}
    _write_json(trial_dir / "trial_summary.json", payload)
    return TrialResult(trial_id=trial_id, config=config, rows=rows, summary=summary)


def _stage_configs(stage_name: str, parents: list[TrialResult], baseline_lsd: float) -> list[dict[str, Any]]:
    seed = [
        {
            "_stage": stage_name,
            "_baseline_speech_band_lsd_mean": baseline_lsd,
            "rnnoise_output_lowpass_cutoff_hz": 7500.0,
            "rnnoise_output_notch_freq_hz": 500.0,
            "rnnoise_output_notch_q": 20.0,
            "rnnoise_output_highpass_cutoff_hz": 70.0,
            "rnnoise_vad_blend_gamma": 0.5,
            "rnnoise_vad_max_speech_preserve": 0.95,
            "rnnoise_residual_highband_enabled": False,
            "rnnoise_residual_highband_cutoff_hz": 3000.0,
            "rnnoise_residual_highband_gain": 0.5,
            "rnnoise_residual_jump_limit_enabled": False,
            "rnnoise_residual_jump_limit_band_low_hz": 3000.0,
            "rnnoise_residual_jump_limit_rise_db_per_frame": 4.0,
        }
    ]
    if not parents:
        return seed
    configs: list[dict[str, Any]] = []
    if stage_name == "stage1":
        for parent in parents:
            for notch_freq in (0.0, 480.0, 500.0, 520.0):
                for notch_q in (20.0, 12.0, 8.0, 6.0):
                    for output_hp in (60.0, 70.0, 80.0):
                        cfg = dict(parent.config)
                        cfg.update(
                            {
                                "_stage": stage_name,
                                "_baseline_speech_band_lsd_mean": baseline_lsd,
                                "rnnoise_output_notch_freq_hz": notch_freq,
                                "rnnoise_output_notch_q": notch_q,
                                "rnnoise_output_highpass_cutoff_hz": output_hp,
                                "rnnoise_residual_highband_enabled": False,
                                "rnnoise_residual_jump_limit_enabled": False,
                            }
                        )
                        configs.append(cfg)
    elif stage_name == "stage2":
        for parent in parents:
            for output_lp in (7500.0, 6500.0, 6000.0):
                for gamma in (0.5, 0.75):
                    for max_preserve in (0.85, 0.90, 0.95):
                        cfg = dict(parent.config)
                        cfg.update(
                            {
                                "_stage": stage_name,
                                "_baseline_speech_band_lsd_mean": baseline_lsd,
                                "rnnoise_output_lowpass_cutoff_hz": output_lp,
                                "rnnoise_vad_blend_gamma": gamma,
                                "rnnoise_vad_max_speech_preserve": max_preserve,
                                "rnnoise_residual_highband_enabled": False,
                                "rnnoise_residual_jump_limit_enabled": False,
                            }
                        )
                        configs.append(cfg)
    elif stage_name == "stage3":
        for parent in parents:
            for cutoff in (2500.0, 3000.0, 3500.0):
                for gain in (0.35, 0.5, 0.65):
                    for rise_db in (3.0, 4.0, 6.0):
                        cfg = dict(parent.config)
                        cfg.update(
                            {
                                "_stage": stage_name,
                                "_baseline_speech_band_lsd_mean": baseline_lsd,
                                "rnnoise_residual_highband_enabled": True,
                                "rnnoise_residual_highband_cutoff_hz": cutoff,
                                "rnnoise_residual_highband_gain": gain,
                                "rnnoise_residual_jump_limit_enabled": True,
                                "rnnoise_residual_jump_limit_band_low_hz": 3000.0,
                                "rnnoise_residual_jump_limit_rise_db_per_frame": rise_db,
                            }
                        )
                        configs.append(cfg)
    else:
        raise ValueError(f"unknown stage {stage_name}")
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cfg in configs:
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cfg)
    return deduped


def _shortlist_payload(results: list[TrialResult], baseline: TrialResult) -> list[dict[str, Any]]:
    pool = [result for result in results if bool(result.summary["valid"])]
    if not pool:
        pool = list(results)
    valid_sorted = sorted(pool, key=lambda result: _ranking_key_with_hum(result.summary))
    shortlist: list[TrialResult] = [baseline]
    shortlist.extend(valid_sorted[:3])
    if pool:
        shortlist.append(min(pool, key=lambda result: float(result.summary["artifact_500hz_cluster_ratio_mean"])))
        shortlist.append(min(pool, key=lambda result: float(result.summary["artifact_hf_flux_p95_mean"])))
        notch_off = [result for result in pool if float(result.summary["rnnoise_output_notch_freq_hz"]) <= 0.0]
        if notch_off:
            shortlist.append(min(notch_off, key=lambda result: _ranking_key_with_hum(result.summary)))
    uniq: list[TrialResult] = []
    seen: set[str] = set()
    for result in shortlist:
        if result.trial_id in seen:
            continue
        seen.add(result.trial_id)
        uniq.append(result)
    return [
        {
            "trial_id": result.trial_id,
            "summary": result.summary,
            "trial_dir": str(result.config["_trial_dir"]),
        }
        for result in uniq
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune RNNoise artifact cleanup across one or more real-data collections.")
    parser.add_argument(
        "--input-paths",
        nargs="+",
        default=[
            "data-collection/nano-sympoium/nano-symposium-behind-poster/recordings",
            "data-collection/gym-take-two-mar-17",
        ],
    )
    parser.add_argument("--dataset-labels", nargs="*", default=None)
    parser.add_argument(
        "--out-root",
        default="beamforming/benchmark/rnnoise_artifact_tuning_multi",
    )
    parser.add_argument("--mic-array-profile", choices=["respeaker_v3_0457", "respeaker_xvf3800_0650"], default="respeaker_xvf3800_0650")
    return parser.parse_args()


def _slug(v: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(v).strip().lower())


def _discover_datasets(input_paths: list[str], dataset_labels: list[str] | None) -> tuple[list[DatasetSpec], list[tempfile.TemporaryDirectory[str]]]:
    if dataset_labels is not None and len(dataset_labels) not in {0, len(input_paths)}:
        raise ValueError("--dataset-labels must be omitted or match the number of --input-paths")
    datasets: list[DatasetSpec] = []
    temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
    for idx, input_path in enumerate(input_paths):
        path = Path(input_path).resolve()
        extracted_root, temp_dir = _extract_if_needed(path)
        if temp_dir is not None:
            temp_dirs.append(temp_dir)
        candidate_root = extracted_root.resolve()
        try:
            resolved_root, recordings = _discover_recordings(candidate_root)
        except FileNotFoundError:
            # Allow passing a collection's "recordings/" container directly.
            if candidate_root.is_dir() and candidate_root.name == "recordings":
                items = []
                for recording_dir in sorted(p for p in candidate_root.iterdir() if p.is_dir()):
                    if (recording_dir / "raw").is_dir():
                        items.append((recording_dir.name, recording_dir))
                if not items:
                    raise
                resolved_root, recordings = candidate_root, items
            else:
                raise
        label = (
            str(dataset_labels[idx])
            if dataset_labels is not None and len(dataset_labels) == len(input_paths)
            else _slug(resolved_root.name or path.name or f"dataset_{idx}")
        )
        datasets.append(DatasetSpec(label=label, root=resolved_root, recordings=recordings))
    return datasets, temp_dirs


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    datasets, temp_dirs = _discover_datasets(list(args.input_paths), None if args.dataset_labels in (None, []) else list(args.dataset_labels))
    try:
        mic_geometry_xyz = mic_positions_xyz(str(args.mic_array_profile))
        all_results: list[TrialResult] = []

        baseline_cfg = {
            "_stage": "baseline",
            "_baseline_speech_band_lsd_mean": 0.0,
            "rnnoise_output_lowpass_cutoff_hz": 7500.0,
            "rnnoise_output_notch_freq_hz": 500.0,
            "rnnoise_output_notch_q": 20.0,
            "rnnoise_output_highpass_cutoff_hz": 70.0,
            "rnnoise_vad_blend_gamma": 0.5,
            "rnnoise_vad_max_speech_preserve": 0.95,
            "rnnoise_residual_highband_enabled": False,
            "rnnoise_residual_highband_cutoff_hz": 3000.0,
            "rnnoise_residual_highband_gain": 0.5,
            "rnnoise_residual_jump_limit_enabled": False,
            "rnnoise_residual_jump_limit_band_low_hz": 3000.0,
            "rnnoise_residual_jump_limit_rise_db_per_frame": 4.0,
        }
        baseline_cfg["_trial_dir"] = str((out_root / "trials" / "baseline").resolve())
        baseline = _run_trial(
            trial_id="baseline",
            trial_dir=Path(baseline_cfg["_trial_dir"]),
            datasets=datasets,
            mic_array_profile=str(args.mic_array_profile),
            mic_geometry_xyz=mic_geometry_xyz,
            config=baseline_cfg,
        )
        all_results.append(baseline)
        baseline_lsd = float(baseline.summary["speech_band_lsd_mean_mean"])

        stage1_results: list[TrialResult] = []
        for idx, cfg in enumerate(_stage_configs("stage1", [baseline], baseline_lsd)):
            cfg["_trial_dir"] = str((out_root / "trials" / f"stage1_{idx:03d}").resolve())
            stage1_results.append(
                _run_trial(
                    trial_id=f"stage1_{idx:03d}",
                    trial_dir=Path(cfg["_trial_dir"]),
                    datasets=datasets,
                    mic_array_profile=str(args.mic_array_profile),
                    mic_geometry_xyz=mic_geometry_xyz,
                    config=cfg,
                )
            )
        all_results.extend(stage1_results)
        stage1_top = sorted([r for r in stage1_results if bool(r.summary["valid"])], key=lambda r: _ranking_key_with_hum(r.summary))[:4]
        if not stage1_top:
            stage1_top = sorted(stage1_results, key=lambda r: _ranking_key_with_hum(r.summary))[:4]

        stage2_results: list[TrialResult] = []
        for idx, cfg in enumerate(_stage_configs("stage2", stage1_top, baseline_lsd)):
            cfg["_trial_dir"] = str((out_root / "trials" / f"stage2_{idx:03d}").resolve())
            stage2_results.append(
                _run_trial(
                    trial_id=f"stage2_{idx:03d}",
                    trial_dir=Path(cfg["_trial_dir"]),
                    datasets=datasets,
                    mic_array_profile=str(args.mic_array_profile),
                    mic_geometry_xyz=mic_geometry_xyz,
                    config=cfg,
                )
            )
        all_results.extend(stage2_results)
        stage2_top = sorted([r for r in stage2_results if bool(r.summary["valid"])], key=lambda r: _ranking_key_with_hum(r.summary))[:3]
        if not stage2_top:
            stage2_top = sorted(stage2_results, key=lambda r: _ranking_key_with_hum(r.summary))[:3]

        stage3_results: list[TrialResult] = []
        for idx, cfg in enumerate(_stage_configs("stage3", stage2_top, baseline_lsd)):
            cfg["_trial_dir"] = str((out_root / "trials" / f"stage3_{idx:03d}").resolve())
            stage3_results.append(
                _run_trial(
                    trial_id=f"stage3_{idx:03d}",
                    trial_dir=Path(cfg["_trial_dir"]),
                    datasets=datasets,
                    mic_array_profile=str(args.mic_array_profile),
                    mic_geometry_xyz=mic_geometry_xyz,
                    config=cfg,
                )
            )
        all_results.extend(stage3_results)

        summary_rows = [result.summary for result in all_results]
        _write_csv(out_root / "trial_table.csv", summary_rows)

        valid_results = [result for result in all_results if bool(result.summary["valid"])]
        best = min(valid_results or all_results, key=lambda result: _ranking_key_with_hum(result.summary))
        _write_json(
            out_root / "best_valid_config.json",
            {
                "trial_id": best.trial_id,
                "summary": best.summary,
                "config": {k: v for k, v in best.config.items() if not str(k).startswith("_")},
                "trial_dir": str(best.config["_trial_dir"]),
                "valid_found": bool(valid_results),
            },
        )
        _write_json(out_root / "top_k_listening_set.json", _shortlist_payload(all_results, baseline))
        _write_json(
            out_root / "report.json",
            {
                "input_paths": list(args.input_paths),
                "datasets": [
                    {
                        "label": dataset.label,
                        "resolved_root": str(dataset.root),
                        "n_recordings": len(dataset.recordings),
                    }
                    for dataset in datasets
                ],
                "n_recordings_total": sum(len(dataset.recordings) for dataset in datasets),
                "baseline_trial_id": baseline.trial_id,
                "best_trial_id": best.trial_id,
                "valid_trial_count": len(valid_results),
            },
        )
    finally:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
