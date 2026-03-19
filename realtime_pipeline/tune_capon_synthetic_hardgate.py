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

import numpy as np

from localization.tuning.common import load_scene_geometries, stratified_scene_subset
from realtime_pipeline.run_framewise_localization_benchmark import _aggregate, _run_job


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-gate Capon tuning on synthetic framewise scenes.")
    parser.add_argument("--near-scenes-root", default="simulation/simulations/configs/testing_specific_angles_near_target_far_diffuse")
    parser.add_argument("--near-assets-root", default="simulation/simulations/assets/testing_specific_angles_near_target_far_diffuse")
    parser.add_argument("--abab-scenes-root", default="simulation/simulations/configs/testing_specific_angles_reappearing_abab")
    parser.add_argument("--abab-assets-root", default="simulation/simulations/assets/testing_specific_angles_reappearing_abab")
    parser.add_argument("--out-root", default="realtime_pipeline/output/capon_synthetic_tuning")
    parser.add_argument("--study-name", default="capon_synthetic_hardgate")
    parser.add_argument("--time-budget-min", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial-scene-workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    return parser.parse_args()


def _deterministic_abab_subset(scenes_root: Path, assets_root: Path) -> list[Path]:
    scenes = load_scene_geometries(scenes_root=scenes_root, assets_root=assets_root)
    subset = stratified_scene_subset(scenes, per_bucket=1, seed=42)
    if len(subset) >= 6:
        return [scene.scene_path.resolve() for scene in subset[:6]]
    ordered = sorted(scenes, key=lambda item: item.scene_id)
    if len(ordered) <= 6:
        return [scene.scene_path.resolve() for scene in ordered]
    idxs = np.linspace(0, len(ordered) - 1, 6, dtype=int)
    return [ordered[int(idx)].scene_path.resolve() for idx in idxs]


def _suggest_params(trial) -> dict[str, Any]:
    full_scan_every = trial.suggest_categorical("capon_full_scan_every_n_updates", [2, 4, 8])
    return {
        "input_downsample_rate_hz": trial.suggest_categorical("input_downsample_rate_hz", [8000]),
        "fast_frame_ms": 10,
        "localization_hop_ms": trial.suggest_categorical("localization_hop_ms", [150, 200, 250]),
        "localization_window_ms": trial.suggest_categorical("localization_window_ms", [160, 200]),
        "localization_grid_size": trial.suggest_categorical("localization_grid_size", [12, 24, 36]),
        "localization_track_hold_frames": trial.suggest_categorical("localization_track_hold_frames", [2, 3, 5]),
        "localization_max_assoc_distance_deg": trial.suggest_categorical("localization_max_assoc_distance_deg", [10.0, 12.0, 16.0, 20.0]),
        "localization_velocity_alpha": trial.suggest_categorical("localization_velocity_alpha", [0.20, 0.35]),
        "localization_angle_alpha": trial.suggest_categorical("localization_angle_alpha", [0.18, 0.30]),
        "capon_spectrum_ema_alpha": trial.suggest_categorical("capon_spectrum_ema_alpha", [0.65, 0.78]),
        "capon_peak_min_sharpness": trial.suggest_categorical("capon_peak_min_sharpness", [0.04, 0.08]),
        "capon_peak_min_margin": trial.suggest_categorical("capon_peak_min_margin", [0.01, 0.02]),
        "capon_hold_frames": trial.suggest_categorical("capon_hold_frames", [1, 2]),
        "capon_freq_bin_subsample_stride": trial.suggest_categorical("capon_freq_bin_subsample_stride", [1, 2, 4]),
        "capon_freq_bin_min_hz": trial.suggest_categorical("capon_freq_bin_min_hz", [800, 1000]),
        "capon_freq_bin_max_hz": trial.suggest_categorical("capon_freq_bin_max_hz", [2200, 2600]),
        "capon_use_cholesky_solve": True,
        "capon_covariance_ema_alpha": trial.suggest_categorical("capon_covariance_ema_alpha", [0.85, 0.93, 0.97]),
        "capon_full_scan_every_n_updates": full_scan_every,
        "capon_local_refine_enabled": True,
        "capon_local_refine_half_width_deg": trial.suggest_categorical("capon_local_refine_half_width_deg", [12.0, 20.0, 30.0]),
    }


def _job_kwargs_for_scene(
    *,
    strategy: str,
    scene_path: Path,
    assets_root: Path,
    out_root: Path,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "scene_path": str(scene_path.resolve()),
        "assets_root": str(assets_root.resolve()),
        "out_root": str(out_root.resolve()),
        "profile": "respeaker_v3_0457",
        "noise_gain_scale": 1.0,
        "fast_frame_ms": int(params["fast_frame_ms"]),
        "slow_chunk_ms": 200,
        "localization_window_ms": int(params["localization_window_ms"]),
        "localization_hop_ms": int(params["localization_hop_ms"]),
        "localization_grid_size": int(params["localization_grid_size"]),
        "srp_overlap": 0.2,
        "srp_freq_min_hz": 1200,
        "srp_freq_max_hz": 5400,
        "localization_vad_enabled": False,
        "localization_snr_gating_enabled": False,
        "localization_snr_threshold_db": 3.0,
        "localization_msc_variance_enabled": False,
        "localization_msc_history_frames": 6,
        "localization_hsda_enabled": False,
        "localization_hsda_window_frames": 5,
        "input_downsample_rate_hz": int(params["input_downsample_rate_hz"]),
        "localization_track_hold_frames": int(params["localization_track_hold_frames"]),
        "localization_max_assoc_distance_deg": float(params["localization_max_assoc_distance_deg"]),
        "localization_velocity_alpha": float(params["localization_velocity_alpha"]),
        "localization_angle_alpha": float(params["localization_angle_alpha"]),
        "capon_spectrum_ema_alpha": float(params["capon_spectrum_ema_alpha"]),
        "capon_peak_min_sharpness": float(params["capon_peak_min_sharpness"]),
        "capon_peak_min_margin": float(params["capon_peak_min_margin"]),
        "capon_hold_frames": int(params["capon_hold_frames"]),
        "capon_freq_bin_subsample_stride": int(params["capon_freq_bin_subsample_stride"]),
        "capon_freq_bin_min_hz": int(params["capon_freq_bin_min_hz"]),
        "capon_freq_bin_max_hz": int(params["capon_freq_bin_max_hz"]),
        "capon_use_cholesky_solve": bool(params["capon_use_cholesky_solve"]),
        "capon_covariance_ema_alpha": float(params["capon_covariance_ema_alpha"]),
        "capon_full_scan_every_n_updates": int(params["capon_full_scan_every_n_updates"]),
        "capon_local_refine_enabled": bool(params["capon_local_refine_enabled"]),
        "capon_local_refine_half_width_deg": float(params["capon_local_refine_half_width_deg"]),
    }


def _objective_factory(
    *,
    near_scenes: list[Path],
    near_assets_root: Path,
    abab_scenes: list[Path],
    abab_assets_root: Path,
    out_dir: Path,
    trial_scene_workers: int,
):
    def objective(trial):
        params = _suggest_params(trial)
        trial_dir = out_dir / "trials" / f"trial_{trial.number:04d}"
        results = []
        jobs = []
        for scene_path in near_scenes:
            jobs.append(
                _job_kwargs_for_scene(
                    strategy="capon_1src",
                    scene_path=scene_path,
                    assets_root=near_assets_root,
                    out_root=trial_dir / "near_target_far_diffuse",
                    params=params,
                )
            )
        for scene_path in abab_scenes:
            jobs.append(
                _job_kwargs_for_scene(
                    strategy="capon_1src",
                    scene_path=scene_path,
                    assets_root=abab_assets_root,
                    out_root=trial_dir / "reappearing_abab_subset",
                    params=params,
                )
            )

        with ProcessPoolExecutor(max_workers=max(1, int(trial_scene_workers))) as pool:
            futures = [pool.submit(_run_job, **job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())

        aggregates = _aggregate(results, ["capon_1src"])
        active = aggregates["active_speaker_per_frame"][0]
        fast_rtf_mean = (float(active["fast_avg_ms"]) / max(float(params["fast_frame_ms"]), 1.0)) if active.get("fast_avg_ms") is not None else float("nan")
        acc10 = float(active["acc_at_10"]) if active.get("acc_at_10") is not None else float("nan")
        acc20 = float(active["acc_at_20"]) if active.get("acc_at_20") is not None else float("nan")
        mae = float(active["mae_deg"]) if active.get("mae_deg") is not None else float("nan")
        valid = bool(math.isfinite(fast_rtf_mean) and fast_rtf_mean < 0.3)
        objective = (
            (0.0 if valid else 1000.0)
            + (0.0 if not math.isfinite(mae) else mae)
            + 100.0 * (1.0 - (0.0 if not math.isfinite(acc10) else acc10))
            + 30.0 * (1.0 - (0.0 if not math.isfinite(acc20) else acc20))
            + 0.1 * (0.0 if not math.isfinite(fast_rtf_mean) else fast_rtf_mean)
        )
        trial.set_user_attr("valid_under_rtf_gate", valid)
        trial.set_user_attr("fast_rtf_mean", fast_rtf_mean)
        trial.set_user_attr("acc_at_10_mean", acc10)
        trial.set_user_attr("acc_at_20_mean", acc20)
        trial.set_user_attr("mae_deg_mean", mae)
        trial.set_user_attr("params_echo", params)
        _write_json(
            trial_dir / "trial_summary.json",
            {
                "trial_number": int(trial.number),
                "objective": float(objective),
                "valid_under_rtf_gate": bool(valid),
                "fast_rtf_mean": fast_rtf_mean,
                "acc_at_10_mean": acc10,
                "acc_at_20_mean": acc20,
                "mae_deg_mean": mae,
                "params": params,
            },
        )
        return float(objective)

    return objective


def main() -> None:
    args = _parse_args()
    optuna = _get_optuna()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root).resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    near_scenes = sorted(Path(args.near_scenes_root).resolve().glob("*.json"))
    abab_scenes = _deterministic_abab_subset(Path(args.abab_scenes_root).resolve(), Path(args.abab_assets_root).resolve())

    study = optuna.create_study(
        study_name=str(args.study_name),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(args.seed)),
    )
    study.optimize(
        _objective_factory(
            near_scenes=near_scenes,
            near_assets_root=Path(args.near_assets_root).resolve(),
            abab_scenes=abab_scenes,
            abab_assets_root=Path(args.abab_assets_root).resolve(),
            out_dir=out_dir,
            trial_scene_workers=int(args.trial_scene_workers),
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
            "acc_at_10_mean": trial.user_attrs.get("acc_at_10_mean"),
            "acc_at_20_mean": trial.user_attrs.get("acc_at_20_mean"),
            "mae_deg_mean": trial.user_attrs.get("mae_deg_mean"),
            **{str(k): v for k, v in (trial.user_attrs.get("params_echo") or {}).items()},
        }
        trial_rows.append(row)
        if row["valid_under_rtf_gate"] and row["state"] == "COMPLETE":
            valid_trials.append(row)
    valid_trials.sort(
        key=lambda r: (
            -float(r.get("acc_at_10_mean") or -1.0),
            -float(r.get("acc_at_20_mean") or -1.0),
            float(r.get("mae_deg_mean") or 1e9),
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
            "near_scenes_root": str(Path(args.near_scenes_root).resolve()),
            "abab_scenes_root": str(Path(args.abab_scenes_root).resolve()),
            "abab_subset_scene_count": len(abab_scenes),
            "valid_trial_count": len(valid_trials),
            "best_valid_trial": (valid_trials[0] if valid_trials else None),
        },
    )
    if valid_trials:
        _write_json(out_dir / "best_valid_config.json", valid_trials[0])
    else:
        _write_json(out_dir / "no_valid_config_found.json", {"reason": "No trial satisfied mean fast_rtf < 0.3"})


if __name__ == "__main__":
    main()
