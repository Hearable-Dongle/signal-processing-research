from __future__ import annotations

import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METRIC_FIELDS = [
    "mae_deg_matched",
    "rmse_deg_matched",
    "median_ae_deg",
    "acc_within_5deg",
    "acc_within_10deg",
    "acc_within_15deg",
    "recall",
    "precision",
    "f1",
]


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _read_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row[k] for k in group_keys)
        buckets[key].append(row)

    out: list[dict[str, Any]] = []
    for key, entries in sorted(buckets.items()):
        agg: dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        agg["n_scenes"] = len(entries)
        agg["runtime_seconds_mean"] = _safe_mean([e.get("runtime_seconds") for e in entries])
        agg["n_true_mean"] = _safe_mean([e.get("n_true") for e in entries])
        agg["n_pred_mean"] = _safe_mean([e.get("n_pred") for e in entries])
        agg["misses_mean"] = _safe_mean([e.get("misses") for e in entries])
        agg["false_alarms_mean"] = _safe_mean([e.get("false_alarms") for e in entries])
        for metric in METRIC_FIELDS:
            agg[f"{metric}_mean"] = _safe_mean([e.get(metric) for e in entries])
        out.append(agg)
    return out


def _fmt(v: Any, ndigits: int = 3) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def _markdown_summary(
    summary_by_method: list[dict[str, Any]],
    summary_by_scene_type: list[dict[str, Any]],
    summary_by_k: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    run_id: str,
    graph_paths: dict[str, str],
) -> str:
    lines: list[str] = []
    lines.append(f"# Localization Benchmark Summary ({run_id})")
    lines.append("")
    lines.append("## Overall By Method")
    if graph_paths.get("overall"):
        lines.append("")
        lines.append(f"![Overall method comparison]({graph_paths['overall']})")
    lines.append("")
    lines.append("| method | n_scenes | MAE(deg) | RMSE(deg) | Acc@10 | Recall | Precision | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_by_method:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(r.get("method")),
                    _fmt(r.get("n_scenes"), 0),
                    _fmt(r.get("mae_deg_matched_mean")),
                    _fmt(r.get("rmse_deg_matched_mean")),
                    _fmt(r.get("acc_within_10deg_mean")),
                    _fmt(r.get("recall_mean")),
                    _fmt(r.get("precision_mean")),
                    _fmt(r.get("f1_mean")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## By Scene Type")
    if graph_paths.get("scene_type"):
        lines.append("")
        lines.append(f"![Method comparison by scene type]({graph_paths['scene_type']})")
    lines.append("")
    lines.append("| method | scene_type | n_scenes | MAE(deg) | Acc@10 | F1 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in summary_by_scene_type:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(r.get("method")),
                    _fmt(r.get("scene_type")),
                    _fmt(r.get("n_scenes"), 0),
                    _fmt(r.get("mae_deg_matched_mean")),
                    _fmt(r.get("acc_within_10deg_mean")),
                    _fmt(r.get("f1_mean")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## By Number of Speakers (k)")
    if graph_paths.get("k_trends"):
        lines.append("")
        lines.append(f"![Method trends across k]({graph_paths['k_trends']})")
    lines.append("")
    lines.append("| method | k | n_scenes | MAE(deg) | Acc@10 | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in summary_by_k:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(r.get("method")),
                    _fmt(r.get("k"), 0),
                    _fmt(r.get("n_scenes"), 0),
                    _fmt(r.get("mae_deg_matched_mean")),
                    _fmt(r.get("acc_within_10deg_mean")),
                    _fmt(r.get("f1_mean")),
                ]
            )
            + " |"
        )

    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        lines.append("| method | scene_id | error |")
        lines.append("|---|---|---|")
        for r in failures[:50]:
            lines.append(f"| {r.get('method')} | {r.get('scene_id')} | {str(r.get('error', ''))[:120]} |")

    lines.append("")
    lines.append("Generated by `python -m localization.benchmark.run ...`.")
    return "\n".join(lines) + "\n"


def _plot_overall_method(summary_method: list[dict[str, Any]], out_dir: Path) -> str | None:
    if not summary_method:
        return None
    methods = [str(r["method"]) for r in summary_method]
    mae = [r.get("mae_deg_matched_mean") for r in summary_method]
    acc10 = [r.get("acc_within_10deg_mean") for r in summary_method]
    f1 = [r.get("f1_mean") for r in summary_method]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, [v if v is not None else np.nan for v in mae], width=width, label="MAE (deg)")
    ax.bar(x, [v if v is not None else np.nan for v in acc10], width=width, label="Acc@10")
    ax.bar(x + width, [v if v is not None else np.nan for v in f1], width=width, label="F1")
    ax.set_title("Overall Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "overall_method_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path.name


def _plot_by_scene_type(summary_scene_type: list[dict[str, Any]], out_dir: Path) -> str | None:
    if not summary_scene_type:
        return None
    methods = sorted({str(r["method"]) for r in summary_scene_type})
    scene_types = sorted({str(r["scene_type"]) for r in summary_scene_type})
    if not methods or not scene_types:
        return None

    mae_map: dict[tuple[str, str], float | None] = {}
    for r in summary_scene_type:
        mae_map[(str(r["method"]), str(r["scene_type"]))] = r.get("mae_deg_matched_mean")

    x = np.arange(len(scene_types))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, method in enumerate(methods):
        vals = [mae_map.get((method, scene), np.nan) for scene in scene_types]
        offset = (idx - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=method)
    ax.set_title("MAE by Scene Type and Method")
    ax.set_ylabel("MAE (deg)")
    ax.set_xticks(x)
    ax.set_xticklabels(scene_types)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "scene_type_mae_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path.name


def _plot_k_trends(summary_k: list[dict[str, Any]], out_dir: Path) -> str | None:
    if not summary_k:
        return None
    methods = sorted({str(r["method"]) for r in summary_k})
    ks = sorted({int(r["k"]) for r in summary_k})
    if not methods or not ks:
        return None

    mae_map: dict[tuple[str, int], float | None] = {}
    f1_map: dict[tuple[str, int], float | None] = {}
    for r in summary_k:
        key = (str(r["method"]), int(r["k"]))
        mae_map[key] = r.get("mae_deg_matched_mean")
        f1_map[key] = r.get("f1_mean")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for method in methods:
        mae_vals = [mae_map.get((method, k), np.nan) for k in ks]
        f1_vals = [f1_map.get((method, k), np.nan) for k in ks]
        ax1.plot(ks, mae_vals, marker="o", label=method)
        ax2.plot(ks, f1_vals, marker="o", label=method)

    ax1.set_title("MAE vs k")
    ax1.set_xlabel("k (number of speakers)")
    ax1.set_ylabel("MAE (deg)")
    ax1.grid(alpha=0.3)

    ax2.set_title("F1 vs k")
    ax2.set_xlabel("k (number of speakers)")
    ax2.set_ylabel("F1")
    ax2.grid(alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = out_dir / "k_trends.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path.name


def _generate_plots(
    summary_method: list[dict[str, Any]],
    summary_scene_type: list[dict[str, Any]],
    summary_k: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, str]:
    return {
        "overall": _plot_overall_method(summary_method, out_dir),
        "scene_type": _plot_by_scene_type(summary_scene_type, out_dir),
        "k_trends": _plot_k_trends(summary_k, out_dir),
    }


def generate_reports(results_jsonl: Path, out_dir: Path, run_id: str) -> None:
    rows = _read_results(results_jsonl)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_csv = out_dir / "scene_metrics.csv"
    if rows:
        scene_fields = sorted({k for row in rows for k in row.keys()})
    else:
        scene_fields = []
    _write_csv(scene_csv, rows, scene_fields)

    summary_method = _aggregate(rows, ["method"])
    summary_scene_type = _aggregate(rows, ["method", "scene_type"])
    summary_k = _aggregate(rows, ["method", "k"])

    _write_csv(
        out_dir / "summary_by_method.csv",
        summary_method,
        sorted({k for row in summary_method for k in row.keys()}),
    )
    _write_csv(
        out_dir / "summary_by_scene_type.csv",
        summary_scene_type,
        sorted({k for row in summary_scene_type for k in row.keys()}),
    )
    _write_csv(
        out_dir / "summary_by_k.csv",
        summary_k,
        sorted({k for row in summary_k for k in row.keys()}),
    )

    graph_paths = _generate_plots(summary_method, summary_scene_type, summary_k, out_dir)
    failures = [r for r in rows if r.get("status") == "error"]
    md = _markdown_summary(summary_method, summary_scene_type, summary_k, failures, run_id, graph_paths)
    (out_dir / "README_summary.md").write_text(md, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark summary files from raw JSONL.")
    parser.add_argument("--results", required=True, help="Path to raw_results.jsonl.")
    parser.add_argument("--out", required=True, help="Output directory for summary CSV/MD files.")
    parser.add_argument("--run-id", default="manual", help="Run id label used in markdown title.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_reports(Path(args.results), Path(args.out), args.run_id)


if __name__ == "__main__":
    main()
