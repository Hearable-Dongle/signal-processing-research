from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from localization.tuning.common import (
    DEFAULT_ASSETS_ROOT,
    DEFAULT_BENCHMARK_CONFIG,
    DEFAULT_PROFILE,
    DEFAULT_SCENES_ROOT,
    SUPPORTED_METHODS,
    aggregate_method_rows,
    evaluate_method_on_scenes,
    load_benchmark_methods,
    load_scene_geometries,
    summarize_by_geometry,
    write_csv,
    write_json,
)


DEFAULT_OUT_ROOT = Path("localization/benchmark/results/respeaker_v3_testing_specific_angles")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank localization methods on testing_specific_angles with ReSpeaker v3.")
    parser.add_argument("--scenes-root", default=str(DEFAULT_SCENES_ROOT))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--benchmark-config", default=str(DEFAULT_BENCHMARK_CONFIG))
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=["respeaker_v3_0457", "respeaker_xvf3800_0650"])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--methods", nargs="+", default=list(SUPPORTED_METHODS))
    return parser.parse_args()


def _plot_heatmap(rows: list[dict], metric_key: str, title: str, out_path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    methods = sorted({str(row["method"]) for row in ok_rows})
    main_angles = sorted({int(row["main_angle_deg"]) for row in ok_rows if row.get("main_angle_deg") is not None})
    if not methods or not main_angles:
        return
    grid = np.full((len(methods), len(main_angles)), np.nan, dtype=float)
    for method_idx, method in enumerate(methods):
        for angle_idx, angle in enumerate(main_angles):
            vals = [
                float(row[metric_key])
                for row in ok_rows
                if row.get("method") == method and row.get("main_angle_deg") == angle and row.get(metric_key) is not None
            ]
            if vals:
                grid[method_idx, angle_idx] = float(sum(vals) / len(vals))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, max(4, 0.65 * len(methods))))
    im = plt.imshow(grid, aspect="auto", interpolation="nearest")
    plt.xticks(np.arange(len(main_angles)), [str(v) for v in main_angles])
    plt.yticks(np.arange(len(methods)), methods)
    plt.xlabel("Main speaker angle (deg)")
    plt.ylabel("Method")
    plt.title(title)
    plt.colorbar(im, label=metric_key)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_method_scores(rankings: list[dict], out_path: Path) -> None:
    if not rankings:
        return
    labels = [str(row["method"]) for row in rankings]
    values = [float(row["balanced_score"]) for row in rankings]
    x = np.arange(len(labels))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(x, values, color="#1f77b4")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Balanced robustness score")
    plt.title("Method ranking on testing_specific_angles")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = _parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_methods = load_benchmark_methods(Path(args.benchmark_config))
    requested_methods = list(args.methods)
    missing = [method for method in requested_methods if method not in all_methods]
    if missing:
        raise ValueError(f"Methods not found in {args.benchmark_config}: {missing}")

    scenes = load_scene_geometries(Path(args.scenes_root), Path(args.assets_root))
    raw_rows: list[dict] = []
    rankings: list[dict] = []
    summary_by_method: list[dict] = []
    summary_by_angle: list[dict] = []
    summary_by_noise: list[dict] = []
    summary_by_noise_angles: list[dict] = []

    for method in requested_methods:
        rows = evaluate_method_on_scenes(
            method=method,
            method_cfg=all_methods[method],
            scenes=scenes,
            profile=args.profile,
            workers=args.workers,
        )
        raw_rows.extend(rows)
        aggregate = aggregate_method_rows(rows)
        aggregate["method"] = method
        summary_by_method.append(aggregate)
        rankings.append({"method": method, **aggregate})

        for row in summarize_by_geometry(rows, "main_angle_deg"):
            summary_by_angle.append({"method": method, **row})
        for row in summarize_by_geometry(rows, "noise_layout_type"):
            summary_by_noise.append({"method": method, **row})
        for row in summarize_by_geometry(rows, "noise_angles_deg"):
            summary_by_noise_angles.append({"method": method, **row})

    rankings = sorted(rankings, key=lambda row: (-float(row["balanced_score"]), float(row["mae_deg_matched_mean"] or 1e9), row["method"]))
    for idx, row in enumerate(rankings, start=1):
        row["rank"] = idx

    write_csv(out_dir / "raw_results.csv", raw_rows)
    write_csv(out_dir / "summary_by_method.csv", summary_by_method)
    write_csv(out_dir / "summary_by_main_angle.csv", summary_by_angle)
    write_csv(out_dir / "summary_by_noise_layout.csv", summary_by_noise)
    write_csv(out_dir / "summary_by_noise_angles.csv", summary_by_noise_angles)
    write_csv(out_dir / "method_rankings.csv", rankings)
    write_json(
        out_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "profile": args.profile,
            "scenes_root": str(Path(args.scenes_root).resolve()),
            "assets_root": str(Path(args.assets_root).resolve()),
            "benchmark_config": str(Path(args.benchmark_config).resolve()),
            "methods": requested_methods,
            "workers": args.workers,
            "n_scenes": len(scenes),
        },
    )
    _plot_heatmap(raw_rows, "mae_deg_matched", "MAE by method and main angle", out_dir / "method_vs_angle_mae.png")
    _plot_heatmap(raw_rows, "acc_within_10deg", "Acc@10 by method and main angle", out_dir / "method_vs_angle_acc10.png")
    _plot_method_scores(rankings, out_dir / "method_rankings.png")

    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir.resolve(), target_is_directory=True)

    print(f"Wrote ranking outputs to {out_dir}")


if __name__ == "__main__":
    main()
