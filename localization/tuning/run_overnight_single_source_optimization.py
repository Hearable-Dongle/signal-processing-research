from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from localization.tuning.common import write_csv, write_json
from localization.tuning.single_source_common import (
    DEFAULT_BACKEND_CONFIGS,
    DEFAULT_DATASET_WEIGHTS,
    DEFAULT_GYM_ROOT,
    DEFAULT_KITCHENER_ROOT,
    DEFAULT_SIM_DATASETS,
    SUPPORTED_SINGLE_SOURCE_METHODS,
    aggregate_mixed_dataset_rows,
    build_single_source_eval_config,
    evaluate_single_source_jobs,
    export_framewise_timelines,
    export_job_rows,
    load_real_recording_jobs,
    load_simulation_jobs,
    resolve_input_path,
)


DEFAULT_OUT_ROOT = Path("localization/tuning/results/overnight_single_source_optuna")
DEFAULT_METHOD_WORKERS = {
    "srp_phat_localization": 4,
    "capon_1src": 4,
    "capon_mvdr_refine_1src": 4,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overnight mixed real/simulation single-source localization tuning.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--duration-min", type=float, default=720.0)
    parser.add_argument("--poll-interval-sec", type=float, default=600.0)
    parser.add_argument("--methods", nargs="+", default=list(SUPPORTED_SINGLE_SOURCE_METHODS))
    parser.add_argument("--gym-root", default=str(DEFAULT_GYM_ROOT))
    parser.add_argument("--kitchener-root", default=str(DEFAULT_KITCHENER_ROOT))
    parser.add_argument("--sim-near-root", default=str(DEFAULT_SIM_DATASETS["sim_near_target_far_diffuse"][0]))
    parser.add_argument("--sim-near-assets-root", default=str(DEFAULT_SIM_DATASETS["sim_near_target_far_diffuse"][1]))
    parser.add_argument("--sim-silence-root", default=str(DEFAULT_SIM_DATASETS["sim_silence_gaps"][0]))
    parser.add_argument("--sim-silence-assets-root", default=str(DEFAULT_SIM_DATASETS["sim_silence_gaps"][1]))
    parser.add_argument("--top-n-full-eval", type=int, default=3)
    parser.add_argument("--trial-job-workers", type=int, default=1)
    parser.add_argument("--max-real-recordings-per-dataset", type=int, default=None)
    parser.add_argument("--max-sim-scenes-per-dataset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--srp-workers", type=int, default=DEFAULT_METHOD_WORKERS["srp_phat_localization"])
    parser.add_argument("--capon-workers", type=int, default=DEFAULT_METHOD_WORKERS["capon_1src"])
    parser.add_argument("--capon-refine-workers", type=int, default=DEFAULT_METHOD_WORKERS["capon_mvdr_refine_1src"])
    return parser.parse_args()


def _get_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "optuna is not installed in the active environment. Run this from the localization environment that includes optuna."
        ) from exc
    return optuna


def _method_slug(method: str) -> str:
    return str(method).lower().replace("-", "_")


def _storage_url(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve()}"


def _default_trial_params(method: str, cfg: dict[str, Any]) -> dict[str, Any]:
    params = {
        "window_ms": 200,
        "hop_ms": 50,
        "overlap": float(cfg.get("overlap", 0.5)),
        "freq_low_hz": int((cfg.get("freq_range") or [200, 3000])[0]),
        "freq_high_hz": int((cfg.get("freq_range") or [200, 3000])[1]),
    }
    if method in {"srp_phat_localization", "capon_1src", "capon_mvdr_refine_1src"}:
        params["grid_size"] = int(cfg.get("grid_size", 360))
    if method in {"capon_1src", "capon_mvdr_refine_1src"}:
        params["diagonal_loading"] = float(cfg.get("diagonal_loading", 1e-3))
        params["vad_enabled"] = bool(cfg.get("vad_enabled", True))
        params["vad_frame_ms"] = int(cfg.get("vad_frame_ms", 20))
        params["vad_aggressiveness"] = int(cfg.get("vad_aggressiveness", 2))
        params["vad_min_speech_ratio"] = float(cfg.get("vad_min_speech_ratio", 0.2))
        params["capon_spectrum_ema_alpha"] = float(cfg.get("capon_spectrum_ema_alpha", 0.78))
        params["capon_peak_min_sharpness"] = float(cfg.get("capon_peak_min_sharpness", 0.12))
        params["capon_peak_min_margin"] = float(cfg.get("capon_peak_min_margin", 0.04))
        params["capon_hold_frames"] = int(cfg.get("capon_hold_frames", 2))
    if method == "capon_mvdr_refine_1src":
        params["capon_refine_window_deg"] = float(cfg.get("capon_refine_window_deg", 20.0))
        params["capon_refine_step_deg"] = float(cfg.get("capon_refine_step_deg", 2.0))
    return params


def _suggest_eval_config(optuna, trial, method: str, base_cfg: dict[str, Any], fs: int) -> dict[str, Any]:
    window_ms = trial.suggest_int("window_ms", 50, 300, step=50)
    hop_ms = trial.suggest_int("hop_ms", 20, 100, step=5)
    overlap = trial.suggest_float("overlap", 0.25, 0.90, step=0.05)
    freq_low_hz = trial.suggest_int("freq_low_hz", 100, 1500, step=50)
    min_high = max(freq_low_hz + 400, 1500)
    freq_high_hz = trial.suggest_int("freq_high_hz", min_high, 6000, step=100)
    extra: dict[str, Any] = {}
    if method in {"srp_phat_localization", "capon_1src", "capon_mvdr_refine_1src"}:
        extra["grid_size"] = trial.suggest_int("grid_size", 72, 360, step=36)
    if method in {"capon_1src", "capon_mvdr_refine_1src"}:
        extra["diagonal_loading"] = trial.suggest_float("diagonal_loading", 1e-5, 5e-2, log=True)
        extra["vad_enabled"] = trial.suggest_categorical("vad_enabled", [True, False])
        extra["vad_frame_ms"] = trial.suggest_categorical("vad_frame_ms", [10, 20, 30])
        extra["vad_aggressiveness"] = trial.suggest_int("vad_aggressiveness", 0, 3)
        extra["vad_min_speech_ratio"] = trial.suggest_float("vad_min_speech_ratio", 0.05, 0.60, step=0.05)
        extra["capon_spectrum_ema_alpha"] = trial.suggest_float("capon_spectrum_ema_alpha", 0.0, 0.95, step=0.05)
        extra["capon_peak_min_sharpness"] = trial.suggest_float("capon_peak_min_sharpness", 0.0, 0.40, step=0.02)
        extra["capon_peak_min_margin"] = trial.suggest_float("capon_peak_min_margin", 0.0, 0.25, step=0.01)
        extra["capon_hold_frames"] = trial.suggest_int("capon_hold_frames", 0, 8)
    if method == "capon_mvdr_refine_1src":
        extra["capon_refine_window_deg"] = trial.suggest_float("capon_refine_window_deg", 6.0, 40.0, step=2.0)
        extra["capon_refine_step_deg"] = trial.suggest_float("capon_refine_step_deg", 1.0, 5.0, step=1.0)
    return build_single_source_eval_config(
        base_cfg=base_cfg,
        fs=fs,
        window_ms=window_ms,
        hop_ms=hop_ms,
        overlap=overlap,
        freq_low_hz=freq_low_hz,
        freq_high_hz=freq_high_hz,
        extra_updates=extra,
    )


def _resolve_method_workers(args: argparse.Namespace, methods: list[str]) -> dict[str, int]:
    mapping = {
        "srp_phat_localization": int(args.srp_workers),
        "capon_1src": int(args.capon_workers),
        "capon_mvdr_refine_1src": int(args.capon_refine_workers),
    }
    return {method: max(1, mapping[method]) for method in methods}


def _initialize_study(optuna, study_name: str, storage_url: str, default_params: dict[str, Any]):
    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": 120}},
    )
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    if not study.trials:
        study.enqueue_trial(default_params)
    return study


def _build_eval_jobs(args: argparse.Namespace) -> tuple[list[Any], list[Any]]:
    real_jobs = []
    gym_jobs = load_real_recording_jobs("gym", args.gym_root)
    kitchener_jobs = load_real_recording_jobs("kitchener", args.kitchener_root)
    real_jobs.extend(gym_jobs[: args.max_real_recordings_per_dataset] if args.max_real_recordings_per_dataset is not None else gym_jobs)
    real_jobs.extend(
        kitchener_jobs[: args.max_real_recordings_per_dataset] if args.max_real_recordings_per_dataset is not None else kitchener_jobs
    )

    sim_jobs = []
    near_jobs = load_simulation_jobs("sim_near_target_far_diffuse", args.sim_near_root, args.sim_near_assets_root)
    silence_jobs = load_simulation_jobs("sim_silence_gaps", args.sim_silence_root, args.sim_silence_assets_root)
    sim_jobs.extend(near_jobs[: args.max_sim_scenes_per_dataset] if args.max_sim_scenes_per_dataset is not None else near_jobs)
    sim_jobs.extend(
        silence_jobs[: args.max_sim_scenes_per_dataset] if args.max_sim_scenes_per_dataset is not None else silence_jobs
    )
    return real_jobs, sim_jobs


def _objective_factory(
    method: str,
    base_cfg: dict[str, Any],
    subset_jobs: list[Any],
    trial_job_workers: int,
):
    optuna = _get_optuna()
    fs = 16000

    def objective(trial):
        method_cfg = _suggest_eval_config(optuna, trial, method, base_cfg, fs)
        rows = evaluate_single_source_jobs(
            method=method,
            method_cfg=method_cfg,
            jobs=subset_jobs,
            workers=trial_job_workers,
        )
        aggregate = aggregate_mixed_dataset_rows(rows, dataset_weights=DEFAULT_DATASET_WEIGHTS)
        for key, value in aggregate.items():
            if key != "dataset_summaries":
                trial.set_user_attr(key, value)
        trial.set_user_attr("dataset_summaries", aggregate["dataset_summaries"])
        trial.set_user_attr("method_cfg", method_cfg)
        return float(aggregate["weighted_dataset_score"])

    return objective


def _run_worker_loop(
    method: str,
    study_name: str,
    storage_url: str,
    deadline_ts: float,
    objective_args: dict[str, Any],
) -> dict[str, Any]:
    optuna = _get_optuna()
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    objective = _objective_factory(
        method=method,
        base_cfg=objective_args["base_cfg"],
        subset_jobs=objective_args["subset_jobs"],
        trial_job_workers=int(objective_args["trial_job_workers"]),
    )
    completed = 0
    while time.time() < deadline_ts:
        remaining = deadline_ts - time.time()
        timeout = max(1.0, min(remaining, 60.0))
        try:
            study.optimize(objective, n_trials=1, timeout=timeout, catch=(Exception,), show_progress_bar=False)
            completed += 1
        except Exception:
            continue
    return {"method": method, "worker_pid": os.getpid(), "completed_loops": completed}


def _best_trials_summary(study, limit: int) -> list[dict[str, Any]]:
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE" and trial.value is not None]
    complete.sort(key=lambda trial: float(trial.value), reverse=True)
    return [
        {
            "rank": rank,
            "trial_number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for rank, trial in enumerate(complete[:limit], start=1)
    ]


def _full_eval_trial(method: str, trial_summary: dict[str, Any], base_cfg: dict[str, Any], full_jobs: list[Any], trial_job_workers: int) -> dict[str, Any]:
    params = dict(trial_summary["params"])
    method_cfg = build_single_source_eval_config(
        base_cfg=base_cfg,
        fs=16000,
        window_ms=int(params["window_ms"]),
        hop_ms=int(params["hop_ms"]),
        overlap=float(params["overlap"]),
        freq_low_hz=int(params["freq_low_hz"]),
        freq_high_hz=int(params["freq_high_hz"]),
        extra_updates={
            key: params[key]
            for key in (
                "grid_size",
                "diagonal_loading",
                "vad_enabled",
                "vad_frame_ms",
                "vad_aggressiveness",
                "vad_min_speech_ratio",
                "capon_spectrum_ema_alpha",
                "capon_peak_min_sharpness",
                "capon_peak_min_margin",
                "capon_hold_frames",
                "capon_refine_window_deg",
                "capon_refine_step_deg",
            )
            if key in params
        },
    )
    rows = evaluate_single_source_jobs(
        method=method,
        method_cfg=method_cfg,
        jobs=full_jobs,
        workers=trial_job_workers,
    )
    aggregate = aggregate_mixed_dataset_rows(rows, dataset_weights=DEFAULT_DATASET_WEIGHTS)
    return {
        "trial_number": trial_summary["trial_number"],
        "value_subset": trial_summary["value"],
        "method_cfg": method_cfg,
        "job_rows": rows,
        **aggregate,
    }


def _sort_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            0 if bool(row["feasible_under_rtf"]) else 1,
            -float(row["weighted_dataset_score"]),
            float(row["real_mae_deg_mean"] if row["real_mae_deg_mean"] is not None else 1e9),
            float(row["rtf_mean"] if row["rtf_mean"] is not None else 1e9),
        ),
    )


def _atomic_write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def _recommendation_payload(overall_best_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = overall_best_rows[0]
    runners_up = []
    for row in overall_best_rows[1:3]:
        note = "lower score"
        if bool(row["feasible_under_rtf"]) and not bool(best["feasible_under_rtf"]):
            note = "feasible while current leader is not"
        elif row.get("rtf_mean") is not None and best.get("rtf_mean") is not None and float(row["rtf_mean"]) < float(best["rtf_mean"]):
            note = "faster but less accurate"
        runners_up.append(
            {
                "method": row["method"],
                "weighted_dataset_score": row["weighted_dataset_score"],
                "rtf_mean": row["rtf_mean"],
                "real_mae_deg_mean": row["real_mae_deg_mean"],
                "tradeoff_note": note,
            }
        )
    return {
        "current_winning_backend": best["method"],
        "deployable_params": best["method_cfg"],
        "measured_rtf": best["rtf_mean"],
        "feasible_under_rtf": best["feasible_under_rtf"],
        "per_dataset_summary": best["dataset_summaries"],
        "failure_reasons": best["failure_reasons"],
        "runner_up_options": runners_up,
    }


def _recommendation_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"Winning backend: {payload['current_winning_backend']}",
        f"RTF: {payload['measured_rtf']}",
        f"Feasible: {payload['feasible_under_rtf']}",
        "",
        "Per-dataset summary:",
    ]
    for dataset_name, summary in payload["per_dataset_summary"].items():
        lines.append(
            f"- {dataset_name}: score={summary['dataset_score']:.4f}, mae={summary['mae_deg_mean']}, "
            f"acc25={summary['acc_at_25_mean']}, rtf={summary['rtf_mean']}"
        )
    if payload["runner_up_options"]:
        lines.extend(["", "Runners-up:"])
        for row in payload["runner_up_options"]:
            lines.append(
                f"- {row['method']}: score={row['weighted_dataset_score']:.4f}, rtf={row['rtf_mean']}, {row['tradeoff_note']}"
            )
    return "\n".join(lines) + "\n"


def _write_recommendation_files(out_dir: Path, overall_best_rows: list[dict[str, Any]]) -> None:
    payload = _recommendation_payload(overall_best_rows)
    _atomic_write_json(out_dir / "latest_recommendation.json", payload)
    (out_dir / "latest_recommendation.md").write_text(_recommendation_markdown(payload), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    optuna = _get_optuna()
    methods = [str(method) for method in args.methods]
    invalid = [method for method in methods if method not in SUPPORTED_SINGLE_SOURCE_METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods requested: {invalid}")

    real_jobs, sim_jobs = _build_eval_jobs(args)
    all_jobs = real_jobs + sim_jobs
    if not all_jobs:
        raise ValueError("No evaluation jobs discovered")

    cpu_budget = os.cpu_count() or 1
    method_workers = _resolve_method_workers(args, methods)
    total_eval_workers = sum(method_workers.values())
    if total_eval_workers > max(1, cpu_budget - 1):
        print(
            f"Warning: requested per-method workers sum to {total_eval_workers} on a {cpu_budget}-CPU machine. "
            "This is allowed, but may oversubscribe the host."
        )

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "run_id": run_id,
        "duration_min": args.duration_min,
        "poll_interval_sec": args.poll_interval_sec,
        "methods": methods,
        "method_workers": method_workers,
        "trial_job_workers": int(args.trial_job_workers),
        "top_n_full_eval": int(args.top_n_full_eval),
        "gym_root": str(resolve_input_path(args.gym_root).resolve()),
        "kitchener_root": str(resolve_input_path(args.kitchener_root).resolve()),
        "sim_near_root": str(resolve_input_path(args.sim_near_root).resolve()),
        "sim_silence_root": str(resolve_input_path(args.sim_silence_root).resolve()),
        "n_real_jobs": len(real_jobs),
        "n_sim_jobs": len(sim_jobs),
    }
    write_json(out_dir / "run_manifest.json", run_manifest)

    study_infos: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_dir = out_dir / _method_slug(method)
        method_dir.mkdir(parents=True, exist_ok=True)
        db_path = method_dir / "study.db"
        storage_url = _storage_url(db_path)
        study_name = f"overnight_single_source_{_method_slug(method)}"
        study = _initialize_study(optuna, study_name, storage_url, _default_trial_params(method, DEFAULT_BACKEND_CONFIGS[method]))
        study_infos[method] = {
            "method_dir": method_dir,
            "study_name": study_name,
            "storage_url": storage_url,
            "study": study,
        }
        write_json(method_dir / "study_manifest.json", {"method": method, "study_name": study_name, "storage_url": storage_url})

    deadline_ts = time.time() + float(args.duration_min) * 60.0
    poll_interval_sec = max(1.0, float(args.poll_interval_sec))
    monitor_log = out_dir / "monitor.log"
    overall_rows_latest: list[dict[str, Any]] = []

    cycle_index = 0
    while time.time() < deadline_ts:
        cycle_index += 1
        cycle_deadline = min(deadline_ts, time.time() + poll_interval_sec)
        worker_rows: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=sum(method_workers.values())) as pool:
            futures = []
            for method in methods:
                info = study_infos[method]
                for _ in range(method_workers[method]):
                    futures.append(
                        pool.submit(
                            _run_worker_loop,
                            method,
                            info["study_name"],
                            info["storage_url"],
                            cycle_deadline,
                            {
                                "base_cfg": DEFAULT_BACKEND_CONFIGS[method],
                                "subset_jobs": all_jobs,
                                "trial_job_workers": int(args.trial_job_workers),
                            },
                        )
                    )
            for future in concurrent.futures.as_completed(futures):
                worker_rows.append(future.result())
        write_csv(out_dir / "worker_activity_latest.csv", worker_rows)

        overall_best_rows: list[dict[str, Any]] = []
        for method in methods:
            info = study_infos[method]
            study = optuna.load_study(study_name=info["study_name"], storage=info["storage_url"])
            trial_rows = _best_trials_summary(study, limit=max(1, int(args.top_n_full_eval)))
            write_json(info["method_dir"] / "best_trials.json", trial_rows)

            export_rows = []
            export_csv_rows = []
            for trial in study.trials:
                export_rows.append(
                    {
                        "trial_number": trial.number,
                        "state": trial.state.name,
                        "value": trial.value,
                        "params": trial.params,
                        "user_attrs": trial.user_attrs,
                    }
                )
                export_csv_rows.append(
                    {
                        "trial_number": trial.number,
                        "state": trial.state.name,
                        "value": trial.value,
                        "window_ms": trial.params.get("window_ms"),
                        "hop_ms": trial.params.get("hop_ms"),
                        "overlap": trial.params.get("overlap"),
                        "freq_low_hz": trial.params.get("freq_low_hz"),
                        "freq_high_hz": trial.params.get("freq_high_hz"),
                        "grid_size": trial.params.get("grid_size"),
                        "diagonal_loading": trial.params.get("diagonal_loading"),
                        "vad_enabled": trial.params.get("vad_enabled"),
                        "weighted_dataset_score": trial.user_attrs.get("weighted_dataset_score"),
                        "feasible_under_rtf": trial.user_attrs.get("feasible_under_rtf"),
                        "rtf_mean": trial.user_attrs.get("rtf_mean"),
                        "real_mae_deg_mean": trial.user_attrs.get("real_mae_deg_mean"),
                    }
                )
            write_json(info["method_dir"] / "trials_export.json", export_rows)
            write_csv(info["method_dir"] / "trials_export.csv", export_csv_rows)

            full_eval_rows = [
                _full_eval_trial(
                    method=method,
                    trial_summary=trial_summary,
                    base_cfg=DEFAULT_BACKEND_CONFIGS[method],
                    full_jobs=all_jobs,
                    trial_job_workers=max(1, int(args.trial_job_workers)),
                )
                for trial_summary in trial_rows
            ]
            full_eval_rows = _sort_eval_rows(full_eval_rows)
            write_json(
                info["method_dir"] / "best_full_eval.json",
                [{k: v for k, v in row.items() if k != "job_rows"} for row in full_eval_rows],
            )
            write_csv(
                info["method_dir"] / "best_full_eval.csv",
                [
                    {
                        "trial_number": row["trial_number"],
                        "value_subset": row["value_subset"],
                        "weighted_dataset_score": row["weighted_dataset_score"],
                        "feasible_under_rtf": row["feasible_under_rtf"],
                        "rtf_mean": row["rtf_mean"],
                        "real_mae_deg_mean": row["real_mae_deg_mean"],
                        "failure_reasons": ",".join(row["failure_reasons"]),
                        "method_cfg": row["method_cfg"],
                    }
                    for row in full_eval_rows
                ],
            )
            if full_eval_rows:
                best_row = dict(full_eval_rows[0])
                best_payload = {
                    "method": method,
                    "trial_number": best_row["trial_number"],
                    "weighted_dataset_score": best_row["weighted_dataset_score"],
                    "feasible_under_rtf": best_row["feasible_under_rtf"],
                    "rtf_mean": best_row["rtf_mean"],
                    "real_mae_deg_mean": best_row["real_mae_deg_mean"],
                    "failure_reasons": best_row["failure_reasons"],
                    "dataset_summaries": best_row["dataset_summaries"],
                    "method_cfg": best_row["method_cfg"],
                }
                write_json(info["method_dir"] / "best_params.json", best_payload)
                export_job_rows(info["method_dir"] / "best_job_rows.csv", best_row["job_rows"])
                export_framewise_timelines(info["method_dir"] / "best_timeline_rows.csv", best_row["job_rows"])
                overall_best_rows.append(best_payload)

        overall_best_rows = _sort_eval_rows(overall_best_rows)
        if overall_best_rows:
            overall_rows_latest = overall_best_rows
            write_csv(out_dir / "overall_best_configs.csv", overall_best_rows)
            _atomic_write_json(out_dir / "overall_best_configs.json", overall_best_rows)
            _write_recommendation_files(out_dir, overall_best_rows)
            _append_log(
                monitor_log,
                f"{datetime.now(UTC).isoformat()} cycle={cycle_index} leader={overall_best_rows[0]['method']} "
                f"score={overall_best_rows[0]['weighted_dataset_score']:.4f} rtf={overall_best_rows[0]['rtf_mean']}",
            )

        if time.time() >= deadline_ts:
            break

    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir.resolve(), target_is_directory=True)
    if overall_rows_latest:
        _append_log(monitor_log, f"{datetime.now(UTC).isoformat()} complete leader={overall_rows_latest[0]['method']}")
    print(f"Wrote overnight single-source tuning outputs to {out_dir}")


if __name__ == "__main__":
    main()
