from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_OUT_ROOT = Path("beamforming/benchmark/optuna_babble_bootstrap_mvdr")
DEFAULT_BENCHMARK_SCRIPT = Path("beamforming/benchmark/oracle_babble_bootstrap_mvdr_benchmark.py")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for GT-DOA babble-bootstrap MVDR benchmarking.")
    parser.add_argument("--benchmark-script", default=str(DEFAULT_BENCHMARK_SCRIPT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--max-scenes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--profile", default="respeaker_xvf3800_0650")
    parser.add_argument("--methods", nargs="+", default=["delay_sum", "mvdr_fd_bootstrap_oracle_activity", "mvdr_fd_bootstrap_estimated_activity"])
    parser.add_argument("--bootstrap-noise-only-sec", type=float, default=4.0)
    parser.add_argument("--background-babble-count", type=int, default=8)
    parser.add_argument("--background-babble-gain-min", type=float, default=0.045)
    parser.add_argument("--background-babble-gain-max", type=float, default=0.10)
    parser.add_argument("--background-wham-gain", type=float, default=0.16)
    parser.add_argument("--target-activity-detector-backend", default="webrtc_fused", choices=["webrtc_fused", "silero_fused"])
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--regenerate-scenes", action="store_true")
    return parser.parse_args()


def _get_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError("optuna is not installed in the active environment.") from exc
    return optuna


def _load_summary_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def _mean(items: list[float]) -> float:
    return float(sum(items) / max(len(items), 1))


def _objective_factory(args: argparse.Namespace, trial_root: Path):
    optuna = _get_optuna()

    def objective(trial):
        params = {
            "fd_analysis_window_ms": trial.suggest_float("fd_analysis_window_ms", 80.0, 80.0, step=10.0),
            "fd_cov_ema_alpha": trial.suggest_float("fd_cov_ema_alpha", 0.04, 0.25),
            "fd_diag_load": trial.suggest_float("fd_diag_load", 1e-4, 5e-2, log=True),
            "target_activity_low_threshold": trial.suggest_float("target_activity_low_threshold", 0.05, 0.45),
            "target_activity_enter_frames": trial.suggest_int("target_activity_enter_frames", 1, 6),
            "target_activity_exit_frames": trial.suggest_int("target_activity_exit_frames", 1, 8),
            "fd_cov_update_scale_target_active": trial.suggest_float("fd_cov_update_scale_target_active", 0.0, 0.5),
            "fd_cov_update_scale_target_inactive": trial.suggest_float("fd_cov_update_scale_target_inactive", 0.6, 1.5),
            "target_activity_blocker_offset_deg": trial.suggest_float("target_activity_blocker_offset_deg", 75.0, 120.0, step=15.0),
            "target_activity_ratio_floor_db": trial.suggest_float("target_activity_ratio_floor_db", -2.0, 8.0),
            "target_activity_ratio_active_db": trial.suggest_float("target_activity_ratio_active_db", 1.0, 12.0),
            "target_activity_target_rms_floor_scale": trial.suggest_float("target_activity_target_rms_floor_scale", 1.0, 3.5),
            "target_activity_blocker_rms_floor_scale": trial.suggest_float("target_activity_blocker_rms_floor_scale", 0.8, 2.0),
            "target_activity_speech_weight": trial.suggest_float("target_activity_speech_weight", 0.1, 0.8),
            "target_activity_ratio_weight": trial.suggest_float("target_activity_ratio_weight", 0.05, 0.8),
            "target_activity_blocker_weight": trial.suggest_float("target_activity_blocker_weight", 0.0, 0.4),
            "target_activity_vad_mode": trial.suggest_int("target_activity_vad_mode", 0, 3),
            "target_activity_vad_hangover_frames": trial.suggest_int("target_activity_vad_hangover_frames", 0, 6),
            "target_activity_noise_floor_rise_alpha": trial.suggest_float("target_activity_noise_floor_rise_alpha", 0.001, 0.08, log=True),
            "target_activity_noise_floor_fall_alpha": trial.suggest_float("target_activity_noise_floor_fall_alpha", 0.01, 0.25),
            "target_activity_noise_floor_margin_scale": trial.suggest_float("target_activity_noise_floor_margin_scale", 1.0, 2.5),
            "target_activity_rms_scale": trial.suggest_float("target_activity_rms_scale", 1.0, 8.0),
            "target_activity_score_exponent": trial.suggest_float("target_activity_score_exponent", 0.1, 0.9),
        }
        params["target_activity_high_threshold"] = trial.suggest_float(
            "target_activity_high_threshold",
            min(0.75, float(params["target_activity_low_threshold"]) + 0.05),
            0.75,
        )

        out_root = trial_root / f"trial_{trial.number:04d}"
        cmd = [
            sys.executable,
            str(Path(args.benchmark_script).resolve()),
            "--out-root",
            str(out_root),
            "--profile",
            str(args.profile),
            "--workers",
            str(int(args.workers)),
            "--max-scenes",
            str(int(args.max_scenes)),
            "--bootstrap-noise-only-sec",
            str(float(args.bootstrap_noise_only_sec)),
            "--background-babble-count",
            str(int(args.background_babble_count)),
            "--background-babble-gain-min",
            str(float(args.background_babble_gain_min)),
            "--background-babble-gain-max",
            str(float(args.background_babble_gain_max)),
            "--background-wham-gain",
            str(float(args.background_wham_gain)),
            "--fd-analysis-window-ms",
            str(float(params["fd_analysis_window_ms"])),
            "--fd-cov-ema-alpha",
            str(float(params["fd_cov_ema_alpha"])),
            "--fd-diag-load",
            str(float(params["fd_diag_load"])),
            "--target-activity-low-threshold",
            str(float(params["target_activity_low_threshold"])),
            "--target-activity-high-threshold",
            str(float(params["target_activity_high_threshold"])),
            "--target-activity-enter-frames",
            str(int(params["target_activity_enter_frames"])),
            "--target-activity-exit-frames",
            str(int(params["target_activity_exit_frames"])),
            "--fd-cov-update-scale-target-active",
            str(float(params["fd_cov_update_scale_target_active"])),
            "--fd-cov-update-scale-target-inactive",
            str(float(params["fd_cov_update_scale_target_inactive"])),
            "--target-activity-detector-mode",
            "target_blocker_calibrated",
            "--target-activity-detector-backend",
            str(args.target_activity_detector_backend),
            "--target-activity-blocker-offset-deg",
            str(float(params["target_activity_blocker_offset_deg"])),
            "--target-activity-bootstrap-only-calibration",
            "--target-activity-ratio-floor-db",
            str(float(params["target_activity_ratio_floor_db"])),
            "--target-activity-ratio-active-db",
            str(float(params["target_activity_ratio_active_db"])),
            "--target-activity-target-rms-floor-scale",
            str(float(params["target_activity_target_rms_floor_scale"])),
            "--target-activity-blocker-rms-floor-scale",
            str(float(params["target_activity_blocker_rms_floor_scale"])),
            "--target-activity-speech-weight",
            str(float(params["target_activity_speech_weight"])),
            "--target-activity-ratio-weight",
            str(float(params["target_activity_ratio_weight"])),
            "--target-activity-blocker-weight",
            str(float(params["target_activity_blocker_weight"])),
            "--target-activity-vad-mode",
            str(int(params["target_activity_vad_mode"])),
            "--target-activity-vad-hangover-frames",
            str(int(params["target_activity_vad_hangover_frames"])),
            "--target-activity-noise-floor-rise-alpha",
            str(float(params["target_activity_noise_floor_rise_alpha"])),
            "--target-activity-noise-floor-fall-alpha",
            str(float(params["target_activity_noise_floor_fall_alpha"])),
            "--target-activity-noise-floor-margin-scale",
            str(float(params["target_activity_noise_floor_margin_scale"])),
            "--target-activity-rms-scale",
            str(float(params["target_activity_rms_scale"])),
            "--target-activity-score-exponent",
            str(float(params["target_activity_score_exponent"])),
            "--methods",
            *[str(v) for v in args.methods],
        ]
        if args.manifest_path:
            cmd.extend(["--manifest-path", str(args.manifest_path)])
        if args.regenerate_scenes:
            cmd.append("--regenerate-scenes")
        subprocess.run(cmd, check=True)
        summary_rows = _load_summary_rows(out_root / "latest" / "summary_rows.json")
        by_method: dict[str, list[dict[str, Any]]] = {}
        for row in summary_rows:
            by_method.setdefault(str(row["method"]), []).append(row)

        oracle_rows = by_method.get("mvdr_fd_bootstrap_oracle_activity", [])
        estimated_rows = by_method.get("mvdr_fd_bootstrap_estimated_activity", [])
        if not oracle_rows:
            raise RuntimeError("Expected oracle MVDR rows in benchmark output.")
        oracle_speech_snr = _mean([float(row.get("speech_delta_snr_db", 0.0)) for row in oracle_rows])
        oracle_speech_sii = _mean([float(row.get("speech_delta_sii", 0.0)) for row in oracle_rows])
        oracle_bootstrap_nr = _mean([float(row.get("bootstrap_noise_reduction_db", 0.0)) for row in oracle_rows])
        oracle_fast_rtf = _mean([float(row.get("fast_rtf", 0.0)) for row in oracle_rows])
        estimated_gap_penalty = 0.0
        if estimated_rows:
            estimated_speech_snr = _mean([float(row.get("speech_delta_snr_db", 0.0)) for row in estimated_rows])
            estimated_bootstrap_nr = _mean([float(row.get("bootstrap_noise_reduction_db", 0.0)) for row in estimated_rows])
            estimated_gap_penalty = 0.15 * abs(oracle_speech_snr - estimated_speech_snr) + 0.05 * abs(oracle_bootstrap_nr - estimated_bootstrap_nr)
        objective_value = (
            1.0 * oracle_speech_snr
            + 8.0 * oracle_speech_sii
            + 0.08 * oracle_bootstrap_nr
            - 0.15 * oracle_fast_rtf
            - estimated_gap_penalty
        )
        for key, value in params.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("oracle_speech_delta_snr_db", oracle_speech_snr)
        trial.set_user_attr("oracle_speech_delta_sii", oracle_speech_sii)
        trial.set_user_attr("oracle_bootstrap_noise_reduction_db", oracle_bootstrap_nr)
        trial.set_user_attr("oracle_fast_rtf", oracle_fast_rtf)
        trial.set_user_attr("out_root", str((out_root / "latest").resolve()))
        return float(objective_value)

    return objective


def main() -> None:
    args = _parse_args()
    optuna = _get_optuna()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{(out_root / 'study.sqlite3').resolve()}",
        engine_kwargs={"connect_args": {"timeout": 120}},
    )
    study = optuna.create_study(
        study_name="babble_bootstrap_mvdr",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(_objective_factory(args, out_root / "trials"), n_trials=int(args.n_trials), show_progress_bar=False)
    best = study.best_trial
    payload = {
        "run_id": run_id,
        "best_value": float(best.value),
        "best_params": dict(best.params),
        "best_user_attrs": dict(best.user_attrs),
        "n_trials": int(len(study.trials)),
    }
    (out_root / "best_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_root.resolve(), target_is_directory=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
