from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from localization.benchmark.algo_runner import run_method_on_scene
from localization.benchmark.matching import match_predictions
from localization.benchmark.metrics import compute_scene_metrics
from localization.benchmark.report import generate_reports
from localization.benchmark.scene_loader import discover_scenes, load_simulation_config, scene_targets_count


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_scenes(cases, preset_cfg: dict, seed: int):
    sample_per_bucket = preset_cfg.get("sample_per_bucket")
    if sample_per_bucket is None:
        return list(cases)

    rng = random.Random(seed)
    buckets = defaultdict(list)
    for case in cases:
        buckets[(case.scene_type, case.k)].append(case)

    selected = []
    for key in sorted(buckets.keys()):
        scenes = sorted(buckets[key], key=lambda c: c.scene_id)
        if len(scenes) <= sample_per_bucket:
            selected.extend(scenes)
        else:
            selected.extend(rng.sample(scenes, sample_per_bucket))

    return sorted(selected, key=lambda c: (c.scene_type, c.k, c.scene_id))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark localization methods on simulation scenes.")
    parser.add_argument(
        "--config",
        default="localization/benchmark/configs/default.json",
        help="Benchmark config JSON.",
    )
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["SSZ", "SRP-PHAT", "GMDA", "MUSIC", "NormMUSIC", "CSSM", "WAVES"],
        help="Methods to run.",
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--out-root",
        default="localization/benchmark/results",
        help="Root output dir for benchmark runs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of parallel worker processes for scene-method jobs.",
    )
    return parser.parse_args()


def _run_single_job(
    run_id: str,
    method: str,
    method_cfg: dict,
    scene_id: str,
    scene_type: str,
    k: int,
    scene_path: str,
) -> dict:
    row = {
        "run_id": run_id,
        "method": method,
        "scene_id": scene_id,
        "scene_type": scene_type,
        "k": k,
        "scene_path": scene_path,
    }
    try:
        sim_cfg = load_simulation_config(Path(scene_path))
        result = run_method_on_scene(method, method_cfg, sim_cfg)
        match = match_predictions(result.true_doas_deg, result.estimated_doas_deg)
        metrics = compute_scene_metrics(
            match=match,
            n_true=len(result.true_doas_deg),
            n_pred=len(result.estimated_doas_deg),
        )
        row.update(
            {
                "status": "ok",
                "runtime_seconds": result.runtime_seconds,
                "n_true": metrics.n_true,
                "n_pred": metrics.n_pred,
                "n_matched": metrics.n_matched,
                "misses": metrics.misses,
                "false_alarms": metrics.false_alarms,
                "true_doas_deg": result.true_doas_deg,
                "pred_doas_deg": result.estimated_doas_deg,
                "matched_errors_deg": match.matched_errors_deg,
                "mae_deg_matched": metrics.mae_deg_matched,
                "rmse_deg_matched": metrics.rmse_deg_matched,
                "median_ae_deg": metrics.median_ae_deg,
                "acc_within_5deg": metrics.acc_within_5deg,
                "acc_within_10deg": metrics.acc_within_10deg,
                "acc_within_15deg": metrics.acc_within_15deg,
                "recall": metrics.recall,
                "precision": metrics.precision,
                "f1": metrics.f1,
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=10),
            }
        )
    return row


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(Path(args.config))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_roots = cfg["scene_roots"]
    methods_cfg = cfg["methods"]
    preset_cfg = cfg["presets"][args.preset]

    for method in args.methods:
        if method not in methods_cfg:
            raise ValueError(f"Method '{method}' not found in benchmark config methods: {list(methods_cfg)}")

    cases = discover_scenes(scene_roots)
    selected_cases = _pick_scenes(cases, preset_cfg, args.seed)
    if args.max_scenes is not None:
        selected_cases = selected_cases[: args.max_scenes]

    if not selected_cases:
        raise RuntimeError("No scenes selected for benchmark run.")

    total_jobs = len(selected_cases) * len(args.methods)
    print(f"Run ID: {run_id}")
    print(f"Preset: {args.preset}")
    print(f"Methods: {args.methods}")
    print(f"Scenes selected: {len(selected_cases)}")
    print(f"Total jobs: {total_jobs}")
    print(f"Workers: {args.workers}")
    print(f"Writing outputs to: {out_dir}")

    rows = []
    jobs: list[tuple[str, dict, str, str, int, str]] = []
    for case in selected_cases:
        sim_cfg = load_simulation_config(case.path)
        n_targets = scene_targets_count(sim_cfg)
        if n_targets == 0:
            print(f"Skipping scene with no speech targets: {case.scene_id}")
            continue

        for method in args.methods:
            jobs.append((method, methods_cfg[method], case.scene_id, case.scene_type, case.k, str(case.path)))

    if not jobs:
        raise RuntimeError("No runnable jobs after filtering scenes.")

    completed = 0

    def _run_with_pool(pool):
        nonlocal completed
        fut_to_job = {
            pool.submit(
                _run_single_job,
                run_id,
                method,
                method_cfg,
                scene_id,
                scene_type,
                k,
                scene_path,
            ): (method, scene_id)
            for method, method_cfg, scene_id, scene_type, k, scene_path in jobs
        }

        for fut in concurrent.futures.as_completed(fut_to_job):
            method, scene_id = fut_to_job[fut]
            completed += 1
            print(f"[{completed}/{len(jobs)}] completed {method} :: {scene_id}")
            rows.append(fut.result())

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            _run_with_pool(pool)
    except PermissionError:
        print("ProcessPoolExecutor unavailable in this environment; falling back to ThreadPoolExecutor.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            _run_with_pool(pool)

    rows = sorted(rows, key=lambda r: (str(r.get("scene_type")), int(r.get("k", 0)), str(r.get("scene_id")), str(r.get("method"))))

    raw_jsonl = out_dir / "raw_results.jsonl"
    with raw_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    generate_reports(results_jsonl=raw_jsonl, out_dir=out_dir, run_id=run_id)

    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir.resolve(), target_is_directory=True)

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_err = sum(1 for r in rows if r.get("status") == "error")
    print(f"Done. Success: {n_ok}, Errors: {n_err}")
    print(f"Summary markdown: {out_dir / 'README_summary.md'}")


if __name__ == "__main__":
    main()
