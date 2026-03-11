from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from localization.tuning.common import (
    DEFAULT_PROFILE,
    load_benchmark_methods,
    load_scene_geometries,
    stratified_scene_subset,
    write_csv,
    write_json,
)
from localization.tuning.framewise_common import (
    aggregate_framewise_rows,
    build_framewise_eval_config,
    evaluate_method_framewise_on_scenes,
    export_framewise_timelines,
)


DEFAULT_SCENES_ROOT = Path("simulation/simulations/configs/testing_specific_angles")
DEFAULT_ASSETS_ROOT = Path("simulation/simulations/assets/testing_specific_angles")
DEFAULT_BENCHMARK_CONFIG = Path("localization/benchmark/configs/default.json")
DEFAULT_OUT_ROOT = Path("localization/tuning/results/framewise_respeaker_v3_optuna")
METHODS = ("SRP-PHAT", "GMDA", "SSZ")
DEFAULT_METHOD_WORKERS = {
    "SRP-PHAT": 10,
    "GMDA": 8,
    "SSZ": 4,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna framewise localization tuning for ReSpeaker v3.")
    parser.add_argument("--scenes-root", default=str(DEFAULT_SCENES_ROOT))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--benchmark-config", default=str(DEFAULT_BENCHMARK_CONFIG))
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=["respeaker_v3_0457"])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--duration-min", type=float, default=60.0)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--subset-per-bucket", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n-full-eval", type=int, default=5)
    parser.add_argument("--trial-scene-workers", type=int, default=1)
    parser.add_argument("--srp-workers", type=int, default=DEFAULT_METHOD_WORKERS["SRP-PHAT"])
    parser.add_argument("--gmda-workers", type=int, default=DEFAULT_METHOD_WORKERS["GMDA"])
    parser.add_argument("--ssz-workers", type=int, default=DEFAULT_METHOD_WORKERS["SSZ"])
    return parser.parse_args()


def _get_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "optuna is not installed in the active environment. Run this from the localization environment that includes optuna."
        ) from exc
    return optuna


def _resolve_method_workers(args: argparse.Namespace, methods: list[str]) -> dict[str, int]:
    mapping = {
        "SRP-PHAT": int(args.srp_workers),
        "GMDA": int(args.gmda_workers),
        "SSZ": int(args.ssz_workers),
    }
    return {method: max(1, mapping[method]) for method in methods}


def _default_trial_params(method: str, cfg: dict[str, Any]) -> dict[str, Any]:
    params = {
        "window_ms": 200,
        "hop_ms": 50,
        "overlap": float(cfg.get("overlap", 0.5)),
        "freq_low_hz": int((cfg.get("freq_range") or [200, 3000])[0]),
        "freq_high_hz": int((cfg.get("freq_range") or [200, 3000])[1]),
    }
    if method == "GMDA":
        params["power_thresh_percentile"] = int(cfg.get("power_thresh_percentile", 80))
        params["mdl_beta"] = float(cfg.get("mdl_beta", 3.0))
    if method == "SSZ":
        params["epsilon"] = float(cfg.get("epsilon", 0.1))
        params["d_freq"] = int(cfg.get("d_freq", 8))
    return params


def _suggest_eval_config(optuna, trial, method: str, base_cfg: dict[str, Any], fs: int) -> dict[str, Any]:
    window_ms = trial.suggest_int("window_ms", 50, 300, step=50)
    hop_ms = trial.suggest_int("hop_ms", 30, 100, step=5)
    overlap = trial.suggest_float("overlap", 0.25, 0.90, step=0.05)
    freq_low_hz = trial.suggest_int("freq_low_hz", 100, 1500, step=50)
    min_high = max(freq_low_hz + 400, 1500)
    freq_high_hz = trial.suggest_int("freq_high_hz", min_high, 6000, step=100)
    extra: dict[str, Any] = {}
    if method == "GMDA":
        extra["power_thresh_percentile"] = trial.suggest_int("power_thresh_percentile", 50, 98)
        extra["mdl_beta"] = trial.suggest_float("mdl_beta", 0.5, 8.0)
    elif method == "SSZ":
        extra["epsilon"] = trial.suggest_float("epsilon", 0.02, 0.45)
        extra["d_freq"] = trial.suggest_int("d_freq", 1, 16)

    cfg = build_framewise_eval_config(
        base_cfg=base_cfg,
        fs=fs,
        window_ms=window_ms,
        hop_ms=hop_ms,
        overlap=overlap,
        freq_low_hz=freq_low_hz,
        freq_high_hz=freq_high_hz,
        extra_updates=extra,
    )
    return cfg


def _objective_factory(
    method: str,
    base_cfg: dict[str, Any],
    subset_scenes,
    profile: str,
    trial_scene_workers: int,
):
    optuna = _get_optuna()
    fs = 16000

    def objective(trial):
        method_cfg = _suggest_eval_config(optuna, trial, method, base_cfg, fs)
        rows = evaluate_method_framewise_on_scenes(
            method=method,
            method_cfg=method_cfg,
            scenes=subset_scenes,
            profile=profile,
            workers=trial_scene_workers,
        )
        aggregate = aggregate_framewise_rows(rows)
        for key, value in aggregate.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("method_cfg", method_cfg)
        return float(aggregate["balanced_score"])

    return objective


def _storage_url(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve()}"


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


def _run_worker_loop(method: str, study_name: str, storage_url: str, deadline_ts: float, objective_args: dict[str, Any]) -> dict[str, Any]:
    optuna = _get_optuna()
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    objective = _objective_factory(
        method=method,
        base_cfg=objective_args["base_cfg"],
        subset_scenes=objective_args["subset_scenes"],
        profile=objective_args["profile"],
        trial_scene_workers=int(objective_args["trial_scene_workers"]),
    )

    completed = 0
    while time.time() < deadline_ts:
        remaining = deadline_ts - time.time()
        timeout = max(1.0, min(remaining, 600.0))
        try:
            study.optimize(objective, n_trials=1, timeout=timeout, catch=(Exception,), show_progress_bar=False)
            completed += 1
        except Exception:
            continue
    return {"method": method, "worker_pid": os.getpid(), "completed_loops": completed}


def _best_trials_summary(study, limit: int) -> list[dict[str, Any]]:
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE" and trial.value is not None]
    complete.sort(key=lambda trial: float(trial.value), reverse=True)
    rows: list[dict[str, Any]] = []
    for rank, trial in enumerate(complete[:limit], start=1):
        rows.append(
            {
                "rank": rank,
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
        )
    return rows


def _full_eval_trial(method: str, trial_summary: dict[str, Any], base_cfg: dict[str, Any], full_scenes, profile: str, trial_scene_workers: int) -> dict[str, Any]:
    params = dict(trial_summary["params"])
    method_cfg = build_framewise_eval_config(
        base_cfg=base_cfg,
        fs=16000,
        window_ms=int(params["window_ms"]),
        hop_ms=int(params["hop_ms"]),
        overlap=float(params["overlap"]),
        freq_low_hz=int(params["freq_low_hz"]),
        freq_high_hz=int(params["freq_high_hz"]),
        extra_updates={
            key: params[key]
            for key in ("power_thresh_percentile", "mdl_beta", "epsilon", "d_freq")
            if key in params
        },
    )
    rows = evaluate_method_framewise_on_scenes(
        method=method,
        method_cfg=method_cfg,
        scenes=full_scenes,
        profile=profile,
        workers=trial_scene_workers,
    )
    aggregate = aggregate_framewise_rows(rows)
    return {
        "trial_number": trial_summary["trial_number"],
        "value_subset": trial_summary["value"],
        "method_cfg": method_cfg,
        "scene_rows": rows,
        **aggregate,
    }


def _scene_rows_csv_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        payload.append({k: v for k, v in row.items() if k != "timeline_rows"})
    return payload


def main() -> None:
    args = _parse_args()
    optuna = _get_optuna()
    methods = list(args.methods)
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods requested: {invalid}")

    cpu_budget = os.cpu_count() or 1
    method_workers = _resolve_method_workers(args, methods)
    total_eval_workers = sum(method_workers.values())
    if total_eval_workers > max(1, cpu_budget - 2):
        print(
            f"Warning: requested per-method workers sum to {total_eval_workers} on a {cpu_budget}-CPU machine. "
            "This is allowed, but may oversubscribe the host."
        )

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_methods = load_benchmark_methods(Path(args.benchmark_config))
    base_cfgs = {method: dict(all_methods[method]) for method in methods}
    scenes = load_scene_geometries(Path(args.scenes_root), Path(args.assets_root))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    subset_scenes = stratified_scene_subset(scenes, per_bucket=int(args.subset_per_bucket), seed=int(args.seed))
    deadline_ts = time.time() + float(args.duration_min) * 60.0

    write_json(
        out_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "profile": args.profile,
            "duration_min": args.duration_min,
            "benchmark_config": str(Path(args.benchmark_config).resolve()),
            "scenes_root": str(Path(args.scenes_root).resolve()),
            "assets_root": str(Path(args.assets_root).resolve()),
            "methods": methods,
            "subset_per_bucket": args.subset_per_bucket,
            "max_scenes": args.max_scenes,
            "subset_scene_ids": [scene.scene_id for scene in subset_scenes],
            "method_workers": method_workers,
            "trial_scene_workers": int(args.trial_scene_workers),
            "top_n_full_eval": args.top_n_full_eval,
        },
    )

    study_infos: dict[str, dict[str, Any]] = {}
    dashboard_lines: list[str] = []
    for method in methods:
        method_dir = out_dir / method.lower().replace("-", "_")
        method_dir.mkdir(parents=True, exist_ok=True)
        db_path = method_dir / "study.db"
        storage_url = _storage_url(db_path)
        study_name = f"framewise_respeaker_v3_{method.lower().replace('-', '_')}"
        default_params = _default_trial_params(method, base_cfgs[method])
        study = _initialize_study(optuna, study_name, storage_url, default_params)
        dashboard_lines.append(f"{method}: optuna-dashboard {storage_url}")
        study_infos[method] = {
            "method_dir": method_dir,
            "db_path": db_path,
            "storage_url": storage_url,
            "study_name": study_name,
            "study": study,
        }
        write_json(method_dir / "study_manifest.json", {"method": method, "study_name": study_name, "storage_url": storage_url})

    (out_dir / "dashboard_commands.txt").write_text("\n".join(dashboard_lines) + "\n", encoding="utf-8")

    futures = []
    max_workers = sum(method_workers[method] for method in methods)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for method in methods:
            info = study_infos[method]
            for _ in range(method_workers[method]):
                futures.append(
                    pool.submit(
                        _run_worker_loop,
                        method,
                        info["study_name"],
                        info["storage_url"],
                        deadline_ts,
                        {
                            "base_cfg": base_cfgs[method],
                            "subset_scenes": subset_scenes,
                            "profile": args.profile,
                            "trial_scene_workers": int(args.trial_scene_workers),
                        },
                    )
                )

        worker_rows = [future.result() for future in as_completed(futures)]
    write_csv(out_dir / "worker_activity.csv", worker_rows)

    overall_best_rows: list[dict[str, Any]] = []
    for method in methods:
        info = study_infos[method]
        study = optuna.load_study(study_name=info["study_name"], storage=info["storage_url"])
        trial_rows = _best_trials_summary(study, limit=max(int(args.top_n_full_eval), 1))
        write_json(info["method_dir"] / "best_trials.json", trial_rows)

        export_rows: list[dict[str, Any]] = []
        export_csv_rows: list[dict[str, Any]] = []
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
                    "power_thresh_percentile": trial.params.get("power_thresh_percentile"),
                    "mdl_beta": trial.params.get("mdl_beta"),
                    "epsilon": trial.params.get("epsilon"),
                    "d_freq": trial.params.get("d_freq"),
                    "mae_deg_mean": trial.user_attrs.get("mae_deg_mean"),
                    "acc_at_10_mean": trial.user_attrs.get("acc_at_10_mean"),
                    "acc_at_25_mean": trial.user_attrs.get("acc_at_25_mean"),
                    "coverage_rate_mean": trial.user_attrs.get("coverage_rate_mean"),
                    "runtime_ms_mean": trial.user_attrs.get("runtime_ms_mean"),
                }
            )
        write_json(info["method_dir"] / "trials_export.json", export_rows)
        write_csv(info["method_dir"] / "trials_export.csv", export_csv_rows)

        full_eval_rows = []
        for trial_summary in trial_rows:
            full_eval_rows.append(
                _full_eval_trial(
                    method=method,
                    trial_summary=trial_summary,
                    base_cfg=base_cfgs[method],
                    full_scenes=scenes,
                    profile=args.profile,
                    trial_scene_workers=max(1, int(args.trial_scene_workers)),
                )
            )
        full_eval_rows = sorted(full_eval_rows, key=lambda row: (-float(row["balanced_score"]), float(row["mae_deg_mean"] or 1e9)))
        write_json(
            info["method_dir"] / "best_full_eval.json",
            [{k: v for k, v in row.items() if k != "scene_rows"} for row in full_eval_rows],
        )
        write_csv(
            info["method_dir"] / "best_full_eval.csv",
            [
                {
                    "trial_number": row["trial_number"],
                    "value_subset": row["value_subset"],
                    "balanced_score": row["balanced_score"],
                    "mae_deg_mean": row["mae_deg_mean"],
                    "acc_at_10_mean": row["acc_at_10_mean"],
                    "acc_at_25_mean": row["acc_at_25_mean"],
                    "coverage_rate_mean": row["coverage_rate_mean"],
                    "runtime_ms_mean": row["runtime_ms_mean"],
                    "method_cfg": row["method_cfg"],
                }
                for row in full_eval_rows
            ],
        )

        best_row = full_eval_rows[0] if full_eval_rows else None
        if best_row is not None:
            best_payload = {
                "method": method,
                "trial_number": best_row["trial_number"],
                "balanced_score": best_row["balanced_score"],
                "mae_deg_mean": best_row["mae_deg_mean"],
                "acc_at_10_mean": best_row["acc_at_10_mean"],
                "acc_at_25_mean": best_row["acc_at_25_mean"],
                "coverage_rate_mean": best_row["coverage_rate_mean"],
                "runtime_ms_mean": best_row["runtime_ms_mean"],
                "method_cfg": best_row["method_cfg"],
            }
            write_json(info["method_dir"] / "best_params.json", best_payload)
            write_csv(info["method_dir"] / "best_scene_rows.csv", _scene_rows_csv_payload(best_row["scene_rows"]))
            export_framewise_timelines(info["method_dir"] / "best_timeline_rows.csv", best_row["scene_rows"])
            overall_best_rows.append(best_payload)

    overall_best_rows = sorted(overall_best_rows, key=lambda row: (-float(row["balanced_score"]), float(row["mae_deg_mean"] or 1e9), row["method"]))
    write_csv(out_dir / "overall_best_configs.csv", overall_best_rows)
    write_json(out_dir / "overall_best_configs.json", overall_best_rows)

    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir.resolve(), target_is_directory=True)

    print(f"Wrote framewise Optuna tuning outputs to {out_dir}")


if __name__ == "__main__":
    main()
