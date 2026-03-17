from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from simulation.mic_array_profiles import mic_positions_xyz

from .data_collection_benchmark import (
    _discover_recordings,
    _extract_if_needed,
    _run_recording_method_job,
)


def _get_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific dependency guard
        raise RuntimeError(
            "optuna is not installed in the active environment. Run this from the beamforming environment."
        ) from exc
    return optuna


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row]
    return float(sum(vals) / max(len(vals), 1))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune realtime RTF on a Data Collection export with an Optuna search.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--out-root", default="beamforming/benchmark/tuning_results")
    parser.add_argument("--study-name", default="data_collection_rtf_tuning")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trace-mae-deg", type=float, default=20.0)
    parser.add_argument("--max-final-mae-deg", type=float, default=20.0)
    parser.add_argument("--ground-truth-transition-ignore-sec", type=float, default=1.5)
    parser.add_argument("--mic-array-profile", choices=["respeaker_v3_0457", "respeaker_xvf3800_0650"], default="respeaker_xvf3800_0650")
    parser.add_argument("--methods-space", nargs="+", choices=["mvdr_fd", "gsc_fd", "delay_sum"], default=["mvdr_fd", "delay_sum"])
    parser.add_argument(
        "--localization-backends",
        nargs="+",
        choices=["capon_1src", "capon_multisrc", "capon_mvdr_refine_1src", "music_1src", "srp_phat_localization"],
        default=["capon_1src"],
    )
    return parser.parse_args()


def _objective_factory(
    *,
    optuna,
    recordings: list[tuple[str, Path]],
    out_dir: Path,
    mic_array_profile: str,
    mic_geometry_xyz,
    max_trace_mae_deg: float,
    max_final_mae_deg: float,
    ground_truth_transition_ignore_sec: float,
    methods_space: list[str],
    localization_backends: list[str],
):
    def objective(trial):
        method = trial.suggest_categorical("method", methods_space)
        localization_backend = trial.suggest_categorical("localization_backend", localization_backends)
        localization_window_ms = trial.suggest_categorical("localization_window_ms", [120, 160, 200])
        localization_hop_ms = trial.suggest_categorical("localization_hop_ms", [80, 95, 120])
        localization_freq_low_hz = trial.suggest_categorical("localization_freq_low_hz", [800, 1000, 1200, 1400])
        localization_freq_high_hz = trial.suggest_categorical("localization_freq_high_hz", [2200, 2600, 3000, 3400])
        localization_vad_enabled = trial.suggest_categorical("localization_vad_enabled", [True, False])
        speaker_match_window_deg = trial.suggest_categorical("speaker_match_window_deg", [12.0, 18.0, 25.0, 33.0])
        centroid_association_mode = trial.suggest_categorical("centroid_association_mode", ["hard_window", "gaussian"])
        centroid_association_sigma_deg = (
            trial.suggest_categorical("centroid_association_sigma_deg", [8.0, 10.0, 12.0, 15.0])
            if centroid_association_mode == "gaussian"
            else 10.0
        )
        centroid_association_min_score = (
            trial.suggest_categorical("centroid_association_min_score", [0.10, 0.15, 0.20])
            if centroid_association_mode == "gaussian"
            else 0.15
        )
        capon_peak_min_sharpness = trial.suggest_categorical("capon_peak_min_sharpness", [0.08, 0.10, 0.12])
        capon_peak_min_margin = trial.suggest_categorical("capon_peak_min_margin", [0.02, 0.04, 0.06])

        trial_rows: list[dict[str, Any]] = []
        trial_dir = out_dir / "trials" / f"trial_{trial.number:04d}"
        for recording_id, recording_dir in recordings:
            row = _run_recording_method_job(
                recording_id=recording_id,
                recording_dir=recording_dir,
                method=method,
                out_dir=trial_dir,
                mic_array_profile=mic_array_profile,
                mic_geometry_xyz=mic_geometry_xyz,
                algorithm_mode="speaker_tracking_single_active",
                separation_mode="single_dominant_no_separator",
                localization_window_ms=int(localization_window_ms),
                localization_hop_ms=int(localization_hop_ms),
                localization_overlap=0.2,
                localization_freq_low_hz=int(localization_freq_low_hz),
                localization_freq_high_hz=int(localization_freq_high_hz),
                localization_pair_selection_mode="all",
                localization_vad_enabled=bool(localization_vad_enabled),
                capon_peak_min_sharpness=float(capon_peak_min_sharpness),
                capon_peak_min_margin=float(capon_peak_min_margin),
                speaker_history_size=8,
                speaker_activation_min_predictions=3,
                speaker_match_window_deg=float(speaker_match_window_deg),
                centroid_association_mode=str(centroid_association_mode),
                centroid_association_sigma_deg=float(centroid_association_sigma_deg),
                centroid_association_min_score=float(centroid_association_min_score),
                own_voice_suppression_mode="off",
                suppressed_user_voice_doa_deg=None,
                suppressed_user_match_window_deg=33.0,
                suppressed_user_null_on_frames=3,
                suppressed_user_null_off_frames=10,
                suppressed_user_gate_attenuation_db=18.0,
                suppressed_user_target_conflict_deg=30.0,
                suppressed_user_speaker_name=None,
                ground_truth_transition_ignore_sec=float(ground_truth_transition_ignore_sec),
                slow_chunk_ms=2000,
                slow_chunk_hop_ms=1000,
                fast_path_reference_mode="speaker_map",
                output_normalization_enabled=True,
                output_allow_amplification=False,
                localization_backend=str(localization_backend),
                tracking_mode="doa_centroid_v1",
                control_mode="speaker_tracking_mode",
                direction_long_memory_enabled=False,
                identity_backend="mfcc_legacy",
                identity_speaker_embedding_model="wavlm_base_plus_sv",
                max_speakers_hint=1,
                assume_single_speaker=False,
            )
            trial_rows.append(row)

        fast_rtf_mean = _mean(trial_rows, "fast_rtf")
        trace_mae_mean = _mean(trial_rows, "gt_trace_active_mae_deg")
        final_mae_mean = _mean(trial_rows, "gt_final_mae_deg")
        feasible = bool(trace_mae_mean <= float(max_trace_mae_deg) and final_mae_mean <= float(max_final_mae_deg))
        penalty = 0.0 if feasible else (100.0 + max(0.0, trace_mae_mean - max_trace_mae_deg) + max(0.0, final_mae_mean - max_final_mae_deg))

        trial.set_user_attr("fast_rtf_mean", fast_rtf_mean)
        trial.set_user_attr("gt_trace_active_mae_mean", trace_mae_mean)
        trial.set_user_attr("gt_final_mae_mean", final_mae_mean)
        trial.set_user_attr("feasible", feasible)
        trial.set_user_attr(
            "params_echo",
            {
                "method": method,
                "localization_backend": localization_backend,
                "localization_window_ms": localization_window_ms,
                "localization_hop_ms": localization_hop_ms,
                "localization_freq_low_hz": localization_freq_low_hz,
                "localization_freq_high_hz": localization_freq_high_hz,
                "localization_vad_enabled": localization_vad_enabled,
                "speaker_match_window_deg": speaker_match_window_deg,
                "centroid_association_mode": centroid_association_mode,
                "centroid_association_sigma_deg": centroid_association_sigma_deg,
                "centroid_association_min_score": centroid_association_min_score,
                "capon_peak_min_sharpness": capon_peak_min_sharpness,
                "capon_peak_min_margin": capon_peak_min_margin,
            },
        )
        (trial_dir / "trial_metrics.json").write_text(
            json.dumps(
                {
                    "trial_number": trial.number,
                    "fast_rtf_mean": fast_rtf_mean,
                    "gt_trace_active_mae_mean": trace_mae_mean,
                    "gt_final_mae_mean": final_mae_mean,
                    "feasible": feasible,
                    "params": trial.params,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return float(fast_rtf_mean + penalty)

    return objective


def main() -> None:
    args = _parse_args()
    optuna = _get_optuna()
    input_path = Path(args.input_path).resolve()
    extracted_root, temp_dir = _extract_if_needed(input_path)
    try:
        resolved_root, recordings = _discover_recordings(extracted_root.resolve())
        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_root).resolve() / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        mic_geometry_xyz = mic_positions_xyz(str(args.mic_array_profile))

        study = optuna.create_study(
            study_name=str(args.study_name),
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=int(args.seed)),
        )
        study.enqueue_trial(
            {
                "method": "mvdr_fd",
                "localization_backend": "capon_1src",
                "localization_window_ms": 200,
                "localization_hop_ms": 95,
                "localization_freq_low_hz": 1200,
                "localization_freq_high_hz": 3000,
                "localization_vad_enabled": True,
                "speaker_match_window_deg": 18.0,
                "centroid_association_mode": "hard_window",
                "capon_peak_min_sharpness": 0.12,
                "capon_peak_min_margin": 0.04,
            }
        )
        objective = _objective_factory(
            optuna=optuna,
            recordings=recordings,
            out_dir=out_dir,
            mic_array_profile=str(args.mic_array_profile),
            mic_geometry_xyz=mic_geometry_xyz,
            max_trace_mae_deg=float(args.max_trace_mae_deg),
            max_final_mae_deg=float(args.max_final_mae_deg),
            ground_truth_transition_ignore_sec=float(args.ground_truth_transition_ignore_sec),
            methods_space=list(args.methods_space),
            localization_backends=list(args.localization_backends),
        )
        study.optimize(objective, n_trials=int(args.n_trials), show_progress_bar=False)

        best_trial = study.best_trial
        summary = {
            "input_path": str(input_path),
            "resolved_input_root": str(resolved_root),
            "n_recordings": len(recordings),
            "n_trials": int(args.n_trials),
            "ground_truth_transition_ignore_sec": float(args.ground_truth_transition_ignore_sec),
            "max_trace_mae_deg": float(args.max_trace_mae_deg),
            "max_final_mae_deg": float(args.max_final_mae_deg),
            "best_trial_number": int(best_trial.number),
            "best_objective": float(best_trial.value),
            "best_params": dict(best_trial.params),
            "best_user_attrs": dict(best_trial.user_attrs),
        }
        (out_dir / "study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        trial_rows = []
        for trial in study.trials:
            trial_rows.append(
                {
                    "trial_number": int(trial.number),
                    "state": str(trial.state.name),
                    "value": None if trial.value is None else float(trial.value),
                    "fast_rtf_mean": trial.user_attrs.get("fast_rtf_mean"),
                    "gt_trace_active_mae_mean": trial.user_attrs.get("gt_trace_active_mae_mean"),
                    "gt_final_mae_mean": trial.user_attrs.get("gt_final_mae_mean"),
                    "feasible": trial.user_attrs.get("feasible"),
                    "params": json.dumps(trial.params, sort_keys=True),
                }
            )
        import csv

        with (out_dir / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0].keys()) if trial_rows else ["trial_number"])
            writer.writeheader()
            writer.writerows(trial_rows)
        print(json.dumps({"out_dir": str(out_dir), "best_trial_number": int(best_trial.number), "best_objective": float(best_trial.value)}, indent=2))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
