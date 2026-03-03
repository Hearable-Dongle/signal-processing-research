from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SCENE_NAME_RE = "_k"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_k(path: Path) -> int:
    name = path.stem
    marker = "_k"
    idx = name.find(marker)
    if idx < 0:
        raise ValueError(f"Unable to parse k from scene filename: {path}")
    rem = name[idx + len(marker) :]
    digits = ""
    for ch in rem:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        raise ValueError(f"Unable to parse k from scene filename: {path}")
    return int(digits)


def _discover_scenes(scene_roots: dict[str, str | Path]) -> list[tuple[str, int, Path]]:
    out: list[tuple[str, int, Path]] = []
    for scene_type, root in scene_roots.items():
        d = Path(root)
        if not d.exists():
            raise FileNotFoundError(f"Scene root not found: {d}")
        for p in sorted(d.glob("*.json")):
            out.append((scene_type, _parse_k(p), p))
    return out


def _filter_scene_types(
    cases: list[tuple[str, int, Path]],
    scene_types: list[str] | None,
) -> list[tuple[str, int, Path]]:
    if not scene_types:
        return cases
    allow = {s.strip() for s in scene_types if s.strip()}
    return [c for c in cases if c[0] in allow]


def _pick_scenes(
    cases: list[tuple[str, int, Path]],
    sample_per_bucket: int | None,
    seed: int,
    max_scenes: int | None,
) -> list[tuple[str, int, Path]]:
    if sample_per_bucket is None:
        selected = list(cases)
    else:
        rng = random.Random(seed)
        buckets: dict[tuple[str, int], list[tuple[str, int, Path]]] = defaultdict(list)
        for c in cases:
            buckets[(c[0], c[1])].append(c)

        selected = []
        for key in sorted(buckets):
            scenes = sorted(buckets[key], key=lambda x: x[2].stem)
            if len(scenes) <= sample_per_bucket:
                selected.extend(scenes)
            else:
                selected.extend(rng.sample(scenes, sample_per_bucket))

    selected = sorted(selected, key=lambda x: (x[0], x[1], x[2].stem))
    if max_scenes is not None:
        selected = selected[:max_scenes]
    return selected


def _run_scene_job(args: argparse.Namespace, scene_path: Path, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "beamforming.main",
        "--config",
        str(args.beamforming_config),
        "--simulation-scene-file",
        str(scene_path),
        "--output",
        str(out_dir),
        "--steering-source",
        args.steering_source,
        "--steering-time",
        args.steering_time,
        "--steering-localization-default",
        args.steering_localization_default,
        "--dynamic-chunk-seconds",
        str(args.dynamic_chunk_seconds),
        "--target-weight-mode",
        args.target_weight_mode,
    ]

    if args.steering_localization_fallbacks:
        cmd += ["--steering-localization-fallbacks", *args.steering_localization_fallbacks]
    if args.localization_methods:
        cmd += ["--localization-methods", *args.localization_methods]
    if args.target_weights_file is not None:
        cmd += ["--target-weights-file", str(args.target_weights_file)]

    subprocess.run(cmd, check=True)


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("beamformer", ""),
            row.get("steering_source", ""),
            row.get("time_mode", ""),
            row.get("localization_method", ""),
        )
        grouped[key].append(row)

    out: list[dict] = []
    for key in sorted(grouped.keys()):
        items = grouped[key]
        snr_raw = [float(r.get("snr_db_raw", r["snr_db"])) for r in items]
        sdr_raw = [float(r.get("si_sdr_db_raw", r["si_sdr_db"])) for r in items]
        rmse_raw = [float(r.get("rmse_raw", r["rmse"])) for r in items]
        snr_norm = [float(r.get("snr_db_norm", r.get("snr_db", 0.0))) for r in items]
        sdr_norm = [float(r.get("si_sdr_db_norm", r.get("si_sdr_db", 0.0))) for r in items]
        rmse_norm = [float(r.get("rmse_norm", r.get("rmse", 0.0))) for r in items]

        out.append(
            {
                "beamformer": key[0],
                "steering_source": key[1],
                "time_mode": key[2],
                "localization_method": key[3],
                "n_rows": len(items),
                "snr_db_raw_mean": sum(snr_raw) / len(snr_raw),
                "si_sdr_db_raw_mean": sum(sdr_raw) / len(sdr_raw),
                "rmse_raw_mean": sum(rmse_raw) / len(rmse_raw),
                "snr_db_norm_mean": sum(snr_norm) / len(snr_norm),
                "si_sdr_db_norm_mean": sum(sdr_norm) / len(sdr_norm),
                "rmse_norm_mean": sum(rmse_norm) / len(rmse_norm),
            }
        )
    return out


def _add_deltas(rows: list[dict]) -> list[dict]:
    baselines: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        if row.get("beamformer") == "Raw Audio (Mean)":
            key = (row.get("scene", ""), row.get("scenario", ""), row.get("time_mode", ""))
            baselines[key] = row

    out: list[dict] = []
    for row in rows:
        key = (row.get("scene", ""), row.get("scenario", ""), row.get("time_mode", ""))
        base = baselines.get(key)
        r = dict(row)
        if base is not None:
            r["delta_snr_db_raw"] = float(row.get("snr_db_raw", row["snr_db"])) - float(base.get("snr_db_raw", base["snr_db"]))
            r["delta_si_sdr_db_raw"] = float(row.get("si_sdr_db_raw", row["si_sdr_db"])) - float(base.get("si_sdr_db_raw", base["si_sdr_db"]))
            r["delta_rmse_raw"] = float(row.get("rmse_raw", row["rmse"])) - float(base.get("rmse_raw", base["rmse"]))
            r["delta_snr_db_norm"] = float(row.get("snr_db_norm", row["snr_db"])) - float(base.get("snr_db_norm", base["snr_db"]))
            r["delta_si_sdr_db_norm"] = float(row.get("si_sdr_db_norm", row["si_sdr_db"])) - float(base.get("si_sdr_db_norm", base["si_sdr_db"]))
            r["delta_rmse_norm"] = float(row.get("rmse_norm", row["rmse"])) - float(base.get("rmse_norm", base["rmse"]))
        else:
            r["delta_snr_db_raw"] = ""
            r["delta_si_sdr_db_raw"] = ""
            r["delta_rmse_raw"] = ""
            r["delta_snr_db_norm"] = ""
            r["delta_si_sdr_db_norm"] = ""
            r["delta_rmse_norm"] = ""
        out.append(r)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_summary(path: Path, run_id: str, scene_count: int, raw_rows: int, agg_rows: list[dict]) -> None:
    lines = []
    lines.append(f"# Beamforming Benchmark Summary ({run_id})")
    lines.append("")
    lines.append(f"- Scenes run: {scene_count}")
    lines.append(f"- Result rows: {raw_rows}")
    lines.append("")
    lines.append("## Aggregate Means (Raw)")
    lines.append("")
    lines.append("| beamformer | steering_source | time_mode | localization_method | n_rows | snr_db_raw_mean | si_sdr_db_raw_mean | rmse_raw_mean |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for r in agg_rows:
        lines.append(
            "| {beamformer} | {steering_source} | {time_mode} | {localization_method} | {n_rows} | {snr_db_raw_mean:.3f} | {si_sdr_db_raw_mean:.3f} | {rmse_raw_mean:.3f} |".format(
                **r
            )
        )

    lines.append("")
    lines.append("## Aggregate Means (Normalized to Reference Loudness)")
    lines.append("")
    lines.append("| beamformer | steering_source | time_mode | localization_method | n_rows | snr_db_norm_mean | si_sdr_db_norm_mean | rmse_norm_mean |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for r in agg_rows:
        lines.append(
            "| {beamformer} | {steering_source} | {time_mode} | {localization_method} | {n_rows} | {snr_db_norm_mean:.3f} | {si_sdr_db_norm_mean:.3f} | {rmse_norm_mean:.3f} |".format(
                **r
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run beamforming evaluation over simulation scene configs.")
    parser.add_argument("--config", default="beamforming/benchmark/configs/default.json")
    parser.add_argument("--beamforming-config", default="beamforming/config/config.json")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--out-root", default="beamforming/benchmark/results")
    parser.add_argument("--scene-types", nargs="+", default=None, help="Filter scene types (e.g., library restaurant)")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of scene jobs to run in parallel.",
    )

    parser.add_argument("--steering-source", choices=["oracle", "localized", "both"], default="localized")
    parser.add_argument("--steering-time", choices=["fixed", "dynamic", "both"], default="fixed")
    parser.add_argument("--localization-methods", nargs="+", default=None)
    parser.add_argument("--steering-localization-default", default="SSZ")
    parser.add_argument("--steering-localization-fallbacks", nargs="+", default=["GMDA"])
    parser.add_argument("--dynamic-chunk-seconds", type=float, default=1.0)
    parser.add_argument("--target-weight-mode", choices=["equal", "config"], default="equal")
    parser.add_argument("--target-weights-file", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_json(Path(args.config))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_id
    runs_dir = out_dir / "scene_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    preset_cfg = cfg["presets"][args.preset]
    cases = _discover_scenes(cfg["scene_roots"])
    cases = _filter_scene_types(cases, args.scene_types)
    selected = _pick_scenes(cases, preset_cfg.get("sample_per_bucket"), args.seed, args.max_scenes)

    if not selected:
        raise RuntimeError("No scenes selected for benchmark run")

    print(f"Run ID: {run_id}")
    print(f"Scenes selected: {len(selected)}")
    print(f"Workers: {args.workers}")
    print(f"Outputs: {out_dir}")

    all_rows: list[dict] = []
    total = len(selected)

    def _run_and_load(scene_path: Path) -> tuple[str, list[dict]]:
        scene_id = scene_path.stem
        scene_out = runs_dir / scene_id
        _run_scene_job(args, scene_path=scene_path, out_dir=scene_out)
        summary_csv = scene_out / "steering_comparison.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing summary CSV for scene: {summary_csv}")
        return scene_id, _load_rows(summary_csv)

    if args.workers <= 1:
        for idx, (_scene_type, _k, scene_path) in enumerate(selected, start=1):
            scene_id, rows = _run_and_load(scene_path)
            print(f"[{idx}/{total}] completed {scene_id}")
            all_rows.extend(rows)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(_run_and_load, scene_path) for _scene_type, _k, scene_path in selected]
            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                scene_id, rows = fut.result()
                completed += 1
                print(f"[{completed}/{total}] completed {scene_id}")
                all_rows.extend(rows)

    with_delta = _add_deltas(all_rows)
    agg = _aggregate(with_delta)

    _write_csv(out_dir / "scene_metrics.csv", with_delta)
    _write_csv(out_dir / "summary_by_method.csv", agg)
    _write_summary(out_dir / "README_summary.md", run_id, len(selected), len(with_delta), agg)

    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir.resolve(), target_is_directory=True)

    print(f"Done. Summary: {out_dir / 'README_summary.md'}")


if __name__ == "__main__":
    main()
