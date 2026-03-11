from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from localization.tuning.run_optuna_respeaker_v3 import DEFAULT_METHOD_WORKERS, METHODS
from localization.tuning.common import DEFAULT_ASSETS_ROOT, DEFAULT_BENCHMARK_CONFIG, DEFAULT_PROFILE, DEFAULT_SCENES_ROOT, write_json


DEFAULT_OUT_ROOT = Path("localization/tuning/results/respeaker_v3_optuna_launches")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch one Optuna tuning coordinator per localization method.")
    parser.add_argument("--scenes-root", default=str(DEFAULT_SCENES_ROOT))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--benchmark-config", default=str(DEFAULT_BENCHMARK_CONFIG))
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=["respeaker_v3_0457", "respeaker_cross_0640"])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--duration-min", type=float, default=60.0)
    parser.add_argument("--subset-per-bucket", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n-full-eval", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--srp-workers", type=int, default=DEFAULT_METHOD_WORKERS["SRP-PHAT"])
    parser.add_argument("--gmda-workers", type=int, default=DEFAULT_METHOD_WORKERS["GMDA"])
    parser.add_argument("--ssz-workers", type=int, default=DEFAULT_METHOD_WORKERS["SSZ"])
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--poll-sec", type=float, default=10.0)
    return parser.parse_args()


def _workers_for_method(args: argparse.Namespace, method: str) -> int:
    if method == "SRP-PHAT":
        return max(1, int(args.srp_workers))
    if method == "GMDA":
        return max(1, int(args.gmda_workers))
    if method == "SSZ":
        return max(1, int(args.ssz_workers))
    raise ValueError(f"Unsupported method: {method}")


def _slug(method: str) -> str:
    return method.lower().replace("-", "_")


def _trial_count(method_root: Path) -> int | None:
    run_dirs = sorted([path for path in method_root.iterdir() if path.is_dir()])
    if not run_dirs:
        return None
    db_candidates = sorted(run_dirs[-1].glob("*/study.db"))
    if not db_candidates:
        return None
    db_path = db_candidates[0]
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM trials").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return None


def main() -> None:
    args = _parse_args()
    methods = list(args.methods)
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods requested: {invalid}")

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    launch_root = Path(args.out_root) / run_id
    launch_root.mkdir(parents=True, exist_ok=True)
    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(launch_root.resolve(), target_is_directory=True)

    manifest = {
        "run_id": run_id,
        "profile": args.profile,
        "duration_min": args.duration_min,
        "methods": methods,
        "scenes_root": str(Path(args.scenes_root).resolve()),
        "assets_root": str(Path(args.assets_root).resolve()),
        "benchmark_config": str(Path(args.benchmark_config).resolve()),
        "python_bin": args.python_bin,
        "method_workers": {method: _workers_for_method(args, method) for method in methods},
    }
    write_json(launch_root / "launch_manifest.json", manifest)

    procs: dict[str, subprocess.Popen] = {}
    log_paths: dict[str, Path] = {}
    method_out_roots: dict[str, Path] = {}

    for method in methods:
        method_slug = _slug(method)
        method_root = launch_root / method_slug
        method_root.mkdir(parents=True, exist_ok=True)
        method_out_roots[method] = method_root
        log_path = method_root / "launcher.log"
        log_paths[method] = log_path
        log_handle = log_path.open("w", encoding="utf-8")

        cmd = [
            args.python_bin,
            "-m",
            "localization.tuning.run_optuna_respeaker_v3",
            "--scenes-root",
            str(args.scenes_root),
            "--assets-root",
            str(args.assets_root),
            "--benchmark-config",
            str(args.benchmark_config),
            "--profile",
            str(args.profile),
            "--out-root",
            str(method_root),
            "--duration-min",
            str(args.duration_min),
            "--subset-per-bucket",
            str(args.subset_per_bucket),
            "--seed",
            str(args.seed),
            "--top-n-full-eval",
            str(args.top_n_full_eval),
            "--methods",
            method,
        ]
        workers = _workers_for_method(args, method)
        if method == "SRP-PHAT":
            cmd.extend(["--srp-workers", str(workers)])
        elif method == "GMDA":
            cmd.extend(["--gmda-workers", str(workers)])
        elif method == "SSZ":
            cmd.extend(["--ssz-workers", str(workers)])

        env = os.environ.copy()
        env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".:{env['PYTHONPATH']}"
        proc = subprocess.Popen(cmd, cwd=str(Path.cwd()), stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        procs[method] = proc

    status_path = launch_root / "process_status.json"
    while procs:
        status_rows = []
        finished = []
        heartbeat_parts = [f"[{datetime.now(UTC).isoformat()}] launcher heartbeat"]
        for method, proc in procs.items():
            rc = proc.poll()
            trials = _trial_count(method_out_roots[method])
            status_rows.append(
                {
                    "method": method,
                    "pid": proc.pid,
                    "returncode": rc,
                    "running": rc is None,
                    "trial_count": trials,
                    "log_path": str(log_paths[method]),
                    "out_root": str(method_out_roots[method]),
                }
            )
            state = "running" if rc is None else f"exit={rc}"
            trial_text = "trials=?" if trials is None else f"trials={trials}"
            heartbeat_parts.append(f"{method} pid={proc.pid} {state} {trial_text}")
            if rc is not None:
                finished.append(method)
        write_json(status_path, {"updated_at_utc": datetime.now(UTC).isoformat(), "processes": status_rows})
        print(" | ".join(heartbeat_parts), flush=True)
        for method in finished:
            procs.pop(method, None)
        if procs:
            time.sleep(max(1.0, float(args.poll_sec)))

    print(f"Wrote launcher outputs to {launch_root}")


if __name__ == "__main__":
    main()
