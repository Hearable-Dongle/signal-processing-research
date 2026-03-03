from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from beamforming.benchmark.metrics import compute_metric_bundle, load_audio_mono
from beamforming.benchmark.sanity_report import (
    plot_noise_sweep_trends,
    plot_spectrogram_comparison,
    plot_waveform_comparison,
)
from simulation.simulation_config import SimulationAudio, SimulationConfig
from simulation.simulator import run_simulation
from simulation.target_policy import iter_target_source_indices


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


def _filter_scene_types(cases: list[tuple[str, int, Path]], scene_types: list[str] | None) -> list[tuple[str, int, Path]]:
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

    if args.causal_only:
        cmd.append("--causal-only")
    if args.force_mic_count is not None:
        cmd += ["--force-mic-count", str(args.force_mic_count)]
    if args.force_mic_radius is not None:
        cmd += ["--force-mic-radius", str(args.force_mic_radius)]
    if args.steering_localization_fallbacks:
        cmd += ["--steering-localization-fallbacks", *args.steering_localization_fallbacks]
    if args.localization_methods:
        cmd += ["--localization-methods", *args.localization_methods]
    if args.target_weights_file is not None:
        cmd += ["--target-weights-file", str(args.target_weights_file)]

    if args.verbose_scene_logs:
        subprocess.run(cmd, check=True)
        return

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(
            [
                "beamforming.main failed",
                f"scene={scene_path}",
                "--- stdout (tail) ---",
                proc.stdout[-2000:],
                "--- stderr (tail) ---",
                proc.stderr[-2000:],
            ]
        )
        raise RuntimeError(tail)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _mean(vals: list[float]) -> float:
    good = [v for v in vals if np.isfinite(v)]
    if not good:
        return float("nan")
    return float(sum(good) / len(good))


def _build_ref_audio(sim_cfg: SimulationConfig, source_signals: list[np.ndarray]) -> np.ndarray:
    if not source_signals:
        return np.zeros(1, dtype=np.float64)
    min_len = min(len(s) for s in source_signals)
    ref = np.zeros(min_len, dtype=np.float64)
    for idx in iter_target_source_indices(sim_cfg):
        ref += np.asarray(source_signals[idx][:min_len], dtype=np.float64)
    return ref


def _load_scene_reference(
    scene_cfg_path: Path,
    force_mic_count: int | None,
    force_mic_radius: float | None,
) -> tuple[np.ndarray, int]:
    cfg = SimulationConfig.from_file(scene_cfg_path)
    if force_mic_count is not None:
        cfg.microphone_array.mic_count = int(force_mic_count)
    if force_mic_radius is not None:
        cfg.microphone_array.mic_radius = float(force_mic_radius)
    _mic, _mic_pos, sources = run_simulation(cfg)
    ref = _build_ref_audio(cfg, sources)
    return ref, int(cfg.audio.fs)


def _scenario_audio_dir(scene_out: Path, row: dict[str, str]) -> Path:
    scene = row["scene"]
    scenario = row["scenario"]
    return scene_out / scene / scenario / "audio"


def _beamformer_wav_name(label: str) -> str:
    return label.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".wav"


def _slugify_beamformer(label: str) -> str:
    return label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def _augment_rows_with_intelligibility(
    rows: list[dict[str, str]],
    scene_out: Path,
    scene_cfg_path: Path,
    force_mic_count: int | None,
    force_mic_radius: float | None,
) -> list[dict]:
    ref_audio, fs = _load_scene_reference(scene_cfg_path, force_mic_count, force_mic_radius)
    augmented: list[dict] = []

    for row in rows:
        r = dict(row)
        audio_dir = _scenario_audio_dir(scene_out, row)
        raw_path = audio_dir / "raw_audio_mean.wav"
        if not raw_path.exists():
            raw_path = audio_dir / "mic_raw_audio.wav"
        proc_path = audio_dir / _beamformer_wav_name(row["beamformer"])

        if raw_path.exists() and proc_path.exists():
            raw, sr_raw = load_audio_mono(str(raw_path))
            proc, sr_proc = load_audio_mono(str(proc_path))
            sr = int(sr_proc if sr_proc > 0 else (sr_raw if sr_raw > 0 else fs))
            bundle = compute_metric_bundle(clean_ref=ref_audio, raw_audio=raw, processed_audio=proc, sample_rate=sr)
            r.update(
                {
                    "sii_raw": bundle.sii_raw,
                    "sii_processed": bundle.sii_processed,
                    "delta_sii": bundle.delta_sii,
                    "stoi_raw": bundle.stoi_raw,
                    "stoi_processed": bundle.stoi_processed,
                    "delta_stoi": bundle.delta_stoi,
                    "metric_snr_db_raw_eval": bundle.snr_db_raw,
                    "metric_snr_db_processed_eval": bundle.snr_db_processed,
                    "metric_delta_snr_db_eval": bundle.delta_snr_db,
                    "metric_si_sdr_db_raw_eval": bundle.si_sdr_db_raw,
                    "metric_si_sdr_db_processed_eval": bundle.si_sdr_db_processed,
                    "metric_delta_si_sdr_db_eval": bundle.delta_si_sdr_db,
                }
            )
        else:
            r.update(
                {
                    "sii_raw": float("nan"),
                    "sii_processed": float("nan"),
                    "delta_sii": float("nan"),
                    "stoi_raw": float("nan"),
                    "stoi_processed": float("nan"),
                    "delta_stoi": float("nan"),
                    "metric_snr_db_raw_eval": float("nan"),
                    "metric_snr_db_processed_eval": float("nan"),
                    "metric_delta_snr_db_eval": float("nan"),
                    "metric_si_sdr_db_raw_eval": float("nan"),
                    "metric_si_sdr_db_processed_eval": float("nan"),
                    "metric_delta_si_sdr_db_eval": float("nan"),
                }
            )
        augmented.append(r)

    return augmented


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
        snr_raw = [_safe_float(r.get("snr_db_raw", r.get("snr_db"))) for r in items]
        sdr_raw = [_safe_float(r.get("si_sdr_db_raw", r.get("si_sdr_db"))) for r in items]
        rmse_raw = [_safe_float(r.get("rmse_raw", r.get("rmse"))) for r in items]
        snr_norm = [_safe_float(r.get("snr_db_norm", r.get("snr_db"))) for r in items]
        sdr_norm = [_safe_float(r.get("si_sdr_db_norm", r.get("si_sdr_db"))) for r in items]
        rmse_norm = [_safe_float(r.get("rmse_norm", r.get("rmse"))) for r in items]

        delta_sii = [_safe_float(r.get("delta_sii")) for r in items]
        delta_stoi = [_safe_float(r.get("delta_stoi")) for r in items]
        sii_proc = [_safe_float(r.get("sii_processed")) for r in items]
        stoi_proc = [_safe_float(r.get("stoi_processed")) for r in items]

        out.append(
            {
                "beamformer": key[0],
                "steering_source": key[1],
                "time_mode": key[2],
                "localization_method": key[3],
                "n_rows": len(items),
                "snr_db_raw_mean": _mean(snr_raw),
                "si_sdr_db_raw_mean": _mean(sdr_raw),
                "rmse_raw_mean": _mean(rmse_raw),
                "snr_db_norm_mean": _mean(snr_norm),
                "si_sdr_db_norm_mean": _mean(sdr_norm),
                "rmse_norm_mean": _mean(rmse_norm),
                "delta_sii_mean": _mean(delta_sii),
                "delta_stoi_mean": _mean(delta_stoi),
                "sii_processed_mean": _mean(sii_proc),
                "stoi_processed_mean": _mean(stoi_proc),
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
            r["delta_snr_db_raw"] = _safe_float(row.get("snr_db_raw", row.get("snr_db"))) - _safe_float(
                base.get("snr_db_raw", base.get("snr_db"))
            )
            r["delta_si_sdr_db_raw"] = _safe_float(row.get("si_sdr_db_raw", row.get("si_sdr_db"))) - _safe_float(
                base.get("si_sdr_db_raw", base.get("si_sdr_db"))
            )
            r["delta_rmse_raw"] = _safe_float(row.get("rmse_raw", row.get("rmse"))) - _safe_float(
                base.get("rmse_raw", base.get("rmse"))
            )
            r["delta_snr_db_norm"] = _safe_float(row.get("snr_db_norm", row.get("snr_db"))) - _safe_float(
                base.get("snr_db_norm", base.get("snr_db"))
            )
            r["delta_si_sdr_db_norm"] = _safe_float(row.get("si_sdr_db_norm", row.get("si_sdr_db"))) - _safe_float(
                base.get("si_sdr_db_norm", base.get("si_sdr_db"))
            )
            r["delta_rmse_norm"] = _safe_float(row.get("rmse_norm", row.get("rmse"))) - _safe_float(
                base.get("rmse_norm", base.get("rmse"))
            )
        else:
            r["delta_snr_db_raw"] = float("nan")
            r["delta_si_sdr_db_raw"] = float("nan")
            r["delta_rmse_raw"] = float("nan")
            r["delta_snr_db_norm"] = float("nan")
            r["delta_si_sdr_db_norm"] = float("nan")
            r["delta_rmse_norm"] = float("nan")
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


def _format_float(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    return f"{v:.3f}"


def _write_summary(path: Path, run_id: str, scene_count: int, raw_rows: int, agg_rows: list[dict], top_rows: list[dict]) -> None:
    lines = []
    lines.append(f"# Beamforming Benchmark Summary ({run_id})")
    lines.append("")
    lines.append(f"- Scenes run: {scene_count}")
    lines.append(f"- Result rows: {raw_rows}")
    lines.append("")
    lines.append("## Top Methods (Ranked by Delta SII)")
    lines.append("")
    lines.append("| beamformer | steering_source | delta_sii_mean | delta_si_sdr_db_raw_mean | delta_stoi_mean |")
    lines.append("|---|---|---:|---:|---:|")
    for r in top_rows:
        lines.append(
            f"| {r['beamformer']} | {r['steering_source']} | {_format_float(_safe_float(r.get('delta_sii_mean')))} | "
            f"{_format_float(_safe_float(r.get('delta_si_sdr_db_raw_mean')))} | {_format_float(_safe_float(r.get('delta_stoi_mean')))} |"
        )

    lines.append("")
    lines.append("## Aggregate Means")
    lines.append("")
    lines.append("| beamformer | steering_source | time_mode | localization_method | n_rows | snr_db_raw_mean | si_sdr_db_raw_mean | delta_sii_mean | delta_stoi_mean |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for r in agg_rows:
        lines.append(
            "| {beamformer} | {steering_source} | {time_mode} | {localization_method} | {n_rows} | {snr} | {sdr} | {dsii} | {dstoi} |".format(
                beamformer=r["beamformer"],
                steering_source=r["steering_source"],
                time_mode=r["time_mode"],
                localization_method=r["localization_method"],
                n_rows=r["n_rows"],
                snr=_format_float(_safe_float(r.get("snr_db_raw_mean"))),
                sdr=_format_float(_safe_float(r.get("si_sdr_db_raw_mean"))),
                dsii=_format_float(_safe_float(r.get("delta_sii_mean"))),
                dstoi=_format_float(_safe_float(r.get("delta_stoi_mean"))),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_top_methods(rows: list[dict], topk: int) -> list[dict]:
    candidates = [
        r
        for r in rows
        if r.get("beamformer") != "Raw Audio (Mean)" and r.get("time_mode") == "dynamic" and r.get("steering_source") == "oracle"
    ]
    if not candidates:
        candidates = [r for r in rows if r.get("beamformer") != "Raw Audio (Mean)"]

    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in candidates:
        by_method[r["beamformer"]].append(r)

    scored = []
    for beamformer, items in by_method.items():
        dsii = _mean([_safe_float(x.get("delta_sii")) for x in items])
        dsdr = _mean(
            [
                _safe_float(x.get("delta_si_sdr_db_raw"))
                if np.isfinite(_safe_float(x.get("delta_si_sdr_db_raw")))
                else _safe_float(x.get("metric_delta_si_sdr_db_eval"))
                for x in items
            ]
        )
        dstoi = _mean([_safe_float(x.get("delta_stoi")) for x in items])
        scored.append(
            {
                "beamformer": beamformer,
                "steering_source": "oracle",
                "delta_sii_mean": dsii,
                "delta_si_sdr_db_raw_mean": dsdr,
                "delta_stoi_mean": dstoi,
            }
        )

    scored.sort(
        key=lambda r: (
            -(_safe_float(r.get("delta_sii_mean")) if np.isfinite(_safe_float(r.get("delta_sii_mean"))) else -1e9),
            -(_safe_float(r.get("delta_si_sdr_db_raw_mean")) if np.isfinite(_safe_float(r.get("delta_si_sdr_db_raw_mean"))) else -1e9),
        )
    )
    return scored[: max(1, topk)]


def _apply_noise_target(
    base_scene_path: Path,
    out_path: Path,
    target_snr_db: float,
    force_mic_count: int | None,
    force_mic_radius: float | None,
) -> Path:
    base_data = _load_json(base_scene_path)
    cfg = SimulationConfig.from_file(base_scene_path)
    if force_mic_count is not None:
        cfg.microphone_array.mic_count = int(force_mic_count)
        base_data["microphone_array"]["mic_count"] = int(force_mic_count)
    if force_mic_radius is not None:
        cfg.microphone_array.mic_radius = float(force_mic_radius)
        base_data["microphone_array"]["mic_radius"] = float(force_mic_radius)

    target_idxs = set(iter_target_source_indices(cfg))
    speech_sources = [s for i, s in enumerate(cfg.audio.sources) if i in target_idxs]
    noise_sources = [s for i, s in enumerate(cfg.audio.sources) if i not in target_idxs]

    if noise_sources and speech_sources:
        speech_cfg = SimulationConfig(
            room=cfg.room,
            microphone_array=cfg.microphone_array,
            audio=SimulationAudio(sources=speech_sources, duration=cfg.audio.duration, fs=cfg.audio.fs),
        )
        noise_cfg = SimulationConfig(
            room=cfg.room,
            microphone_array=cfg.microphone_array,
            audio=SimulationAudio(sources=noise_sources, duration=cfg.audio.duration, fs=cfg.audio.fs),
        )

        speech_mic, _, _ = run_simulation(speech_cfg)
        noise_mic, _, _ = run_simulation(noise_cfg)
        speech_mean = np.mean(speech_mic, axis=1)
        noise_mean = np.mean(noise_mic, axis=1)

        # Use active speech regions to avoid over-attenuating noise when speech has long silence.
        speech_abs = np.abs(speech_mean)
        thr = float(np.percentile(speech_abs, 70))
        active = speech_abs >= max(thr, 1e-6)
        if not np.any(active):
            active = np.ones_like(speech_abs, dtype=bool)

        ps = float(np.mean(speech_mean[active] ** 2) + 1e-12)
        pn = float(np.mean(noise_mean[active] ** 2) + 1e-12)
        alpha = math.sqrt(ps / (pn * (10.0 ** (target_snr_db / 10.0))))
        alpha = float(max(alpha, 1e-4))

        for i, src in enumerate(base_data["audio"]["sources"]):
            if i not in target_idxs:
                src["gain"] = float(src.get("gain", 1.0)) * float(alpha)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(base_data, f, indent=2)
    return out_path


def _stratified_scene_pick(
    rows: list[dict],
    selected: list[tuple[str, int, Path]],
    top_beamformer: str,
    per_type: int,
) -> list[tuple[str, Path, str]]:
    type_by_scene = {scene_path.stem: scene_type for scene_type, _k, scene_path in selected}
    path_by_scene = {scene_path.stem: scene_path for _scene_type, _k, scene_path in selected}

    scored_by_type: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        if r.get("beamformer") != top_beamformer:
            continue
        if r.get("time_mode") != "dynamic" or r.get("steering_source") != "oracle":
            continue
        scene = r.get("scene", "")
        dsii = _safe_float(r.get("delta_sii"))
        if scene in type_by_scene and np.isfinite(dsii):
            scored_by_type[type_by_scene[scene]].append((scene, dsii))

    picks: list[tuple[str, Path, str]] = []
    for scene_type, scores in scored_by_type.items():
        uniq = {}
        for scene, val in scores:
            uniq[scene] = val
        ordered = sorted(uniq.items(), key=lambda x: x[1])
        if not ordered:
            continue

        idxs = [0, len(ordered) // 2, len(ordered) - 1]
        seen = set()
        chosen = []
        for idx in idxs:
            scene_id = ordered[idx][0]
            if scene_id not in seen:
                seen.add(scene_id)
                chosen.append(scene_id)
        for scene_id, _val in ordered:
            if len(chosen) >= per_type:
                break
            if scene_id not in seen:
                chosen.append(scene_id)
                seen.add(scene_id)

        for scene_id in chosen[:per_type]:
            picks.append((scene_type, path_by_scene[scene_id], scene_id))
    return picks


def _run_noise_sweep_for_scene(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    scene_path: Path,
    scene_type: str,
    scene_id: str,
    top_beamformers: list[str],
    snr_levels: list[float],
) -> list[dict]:
    rows_out: list[dict] = []
    sweep_root = out_dir / "sanity" / scene_type / scene_id / "noise_sweep"
    tmp_cfg_dir = sweep_root / "configs"

    for snr in tqdm(snr_levels, desc=f"Noise sweep {scene_id}", unit="snr", leave=False):
        snr_tag = f"snr_{int(snr)}dB"
        cfg_path = _apply_noise_target(
            base_scene_path=scene_path,
            out_path=tmp_cfg_dir / f"{scene_id}_{snr_tag}.json",
            target_snr_db=snr,
            force_mic_count=args.force_mic_count,
            force_mic_radius=args.force_mic_radius,
        )

        run_out = sweep_root / "runs" / snr_tag
        _run_scene_job(args, scene_path=cfg_path, out_dir=run_out)

        csv_path = run_out / "steering_comparison.csv"
        if not csv_path.exists():
            continue

        rows = _load_rows(csv_path)
        rows = _augment_rows_with_intelligibility(
            rows=rows,
            scene_out=run_out,
            scene_cfg_path=cfg_path,
            force_mic_count=args.force_mic_count,
            force_mic_radius=args.force_mic_radius,
        )
        rows = _add_deltas(rows)

        beamformer_set = set(top_beamformers)
        for r in rows:
            if r.get("beamformer") not in beamformer_set:
                continue
            if r.get("time_mode") != "dynamic":
                continue
            rows_out.append(
                {
                    "scene_type": scene_type,
                    "scene": scene_id,
                    "snr_db_target": snr,
                    "steering_source": r.get("steering_source", ""),
                    "beamformer": r.get("beamformer", ""),
                    "delta_sii": _safe_float(r.get("delta_sii")),
                    "delta_stoi": _safe_float(r.get("delta_stoi")),
                    "delta_si_sdr_db": _safe_float(
                        r.get("delta_si_sdr_db_raw")
                        if np.isfinite(_safe_float(r.get("delta_si_sdr_db_raw")))
                        else (
                            r.get("delta_si_sdr_db")
                            if np.isfinite(_safe_float(r.get("delta_si_sdr_db")))
                            else r.get("metric_delta_si_sdr_db_eval")
                        )
                    ),
                    "delta_snr_db": _safe_float(
                        r.get("delta_snr_db_raw")
                        if np.isfinite(_safe_float(r.get("delta_snr_db_raw")))
                        else (
                            r.get("delta_snr_db")
                            if np.isfinite(_safe_float(r.get("delta_snr_db")))
                            else r.get("metric_delta_snr_db_eval")
                        )
                    ),
                }
            )

    return rows_out


def _render_sanity_artifacts(
    *,
    out_dir: Path,
    scene_type: str,
    scene_id: str,
    scene_path: Path,
    scene_run_out: Path,
    top_beamformers: list[str],
    force_mic_count: int | None,
    force_mic_radius: float | None,
) -> list[dict]:
    scenario = "dynamic_oracle"
    audio_dir = scene_run_out / scene_id / scenario / "audio"
    if not audio_dir.exists():
        # fallback to any dynamic_oracle_* path where method names changed
        candidates = sorted((scene_run_out / scene_id).glob("dynamic_oracle*/audio"))
        if candidates:
            audio_dir = candidates[0]

    ref, fs = _load_scene_reference(scene_path, force_mic_count, force_mic_radius)

    scene_art_dir = out_dir / "sanity" / scene_type / scene_id
    out: list[dict] = []
    for beamformer in top_beamformers:
        proc_path = audio_dir / _beamformer_wav_name(beamformer)
        if not proc_path.exists():
            continue
        proc, sr = load_audio_mono(str(proc_path))
        sr_use = sr if sr > 0 else fs
        slug = _slugify_beamformer(beamformer)
        wave_png = scene_art_dir / f"waveform_comparison_{slug}.png"
        spec_png = scene_art_dir / f"spectrogram_comparison_{slug}.png"

        plot_waveform_comparison(
            reference=ref,
            processed=proc,
            sample_rate=sr_use,
            title=f"{scene_type}/{scene_id} - {beamformer}",
            out_path=wave_png,
        )
        plot_spectrogram_comparison(
            reference=ref,
            processed=proc,
            sample_rate=sr_use,
            title=f"{scene_type}/{scene_id} - {beamformer}",
            out_path=spec_png,
        )
        out.append(
            {
                "scene_type": scene_type,
                "scene": scene_id,
                "beamformer": beamformer,
                "waveform_plot": str(wave_png),
                "spectrogram_plot": str(spec_png),
            }
        )
    return out


def _write_pr_report(
    out_path: Path,
    run_id: str,
    args: argparse.Namespace,
    top_methods: list[dict],
    agg: list[dict],
    sanity_artifacts: list[dict],
    noise_rows: list[dict],
) -> None:
    top_method_names = {r["beamformer"] for r in top_methods}
    lines: list[str] = []
    stft_window_ms = 10.0
    try:
        bf_cfg = _load_json(Path(args.beamforming_config))
        stft_window_ms = float(bf_cfg.get("beamforming", {}).get("frame_duration", 10.0))
    except Exception:
        stft_window_ms = 10.0
    stft_hop_ms = stft_window_ms / 2.0

    lines.append(f"# PR Report: Beamforming Efficacy ({run_id})")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Subsystem: speaker-location informed beamforming")
    lines.append("- Steering variants: oracle + localized")
    lines.append("- Causality: dynamic chunked steering only")
    lines.append(f"- Microphone array: {args.force_mic_count or 'scene-default'}-mic circular array")
    lines.append(f"- Scene coverage preset: {args.preset}")
    lines.append("")
    lines.append("## Causality and Latency Notes")
    lines.append("")
    lines.append(f"- STFT analysis window: {stft_window_ms:.2f} ms")
    lines.append(f"- STFT hop: {stft_hop_ms:.2f} ms (50% overlap)")
    lines.append(f"- Dynamic steering chunk size: {float(args.dynamic_chunk_seconds):.3f} s")
    lines.append(f"- Causal-only mode enabled: {'yes' if args.causal_only else 'no'}")
    lines.append("- Runtime latency/RTF measurement is not yet instrumented in this benchmark runner.")
    lines.append("")

    lines.append(f"## Top-{len(top_methods)} Beamformers (Ranked by Delta SII)")
    lines.append("")
    lines.append("| rank | beamformer | delta_sii_mean | delta_si_sdr_db_raw_mean | delta_stoi_mean |")
    lines.append("|---:|---|---:|---:|---:|")
    for i, r in enumerate(top_methods, start=1):
        lines.append(
            f"| {i} | {r['beamformer']} | {_format_float(_safe_float(r.get('delta_sii_mean')))} | "
            f"{_format_float(_safe_float(r.get('delta_si_sdr_db_raw_mean')))} | {_format_float(_safe_float(r.get('delta_stoi_mean')))} |"
        )
    lines.append("")

    lines.append("## Oracle vs Localized (Dynamic)")
    lines.append("")
    lines.append("| beamformer | steering_source | n_rows | snr_db_raw_mean | si_sdr_db_raw_mean | delta_sii_mean |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in agg:
        if row.get("time_mode") != "dynamic":
            continue
        if row.get("beamformer") == "Raw Audio (Mean)":
            continue
        if row.get("beamformer") not in top_method_names:
            continue
        lines.append(
            f"| {row['beamformer']} | {row['steering_source']} | {row['n_rows']} | "
            f"{_format_float(_safe_float(row.get('snr_db_raw_mean')))} | {_format_float(_safe_float(row.get('si_sdr_db_raw_mean')))} | "
            f"{_format_float(_safe_float(row.get('delta_sii_mean')))} |"
        )
    lines.append("")

    lines.append("## Sanity Scenes")
    lines.append("")
    for art in sanity_artifacts:
        lines.append(
            f"- {art['scene_type']}/{art['scene']}: waveform=`{art['waveform_plot']}`, spectrogram=`{art['spectrogram_plot']}`"
        )
    lines.append("")

    if noise_rows:
        lines.append("## Noise Sweep (5/10/20/30 dB Input SNR)")
        lines.append("")
        lines.append("| scene_type | scene | beamformer | snr_db_target | steering_source | delta_sii | delta_stoi | delta_si_sdr_db | delta_snr_db |")
        lines.append("|---|---|---|---:|---|---:|---:|---:|---:|")
        for r in sorted(
            noise_rows,
            key=lambda x: (x["scene_type"], x["scene"], x["beamformer"], x["snr_db_target"], x["steering_source"]),
        ):
            lines.append(
                f"| {r['scene_type']} | {r['scene']} | {r['beamformer']} | {int(r['snr_db_target'])} | {r['steering_source']} | "
                f"{_format_float(_safe_float(r.get('delta_sii')))} | {_format_float(_safe_float(r.get('delta_stoi')))} | "
                f"{_format_float(_safe_float(r.get('delta_si_sdr_db')))} | {_format_float(_safe_float(r.get('delta_snr_db')))} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    parser.add_argument("--steering-source", choices=["oracle", "localized", "both"], default="both")
    parser.add_argument("--steering-time", choices=["fixed", "dynamic", "both"], default="dynamic")
    parser.add_argument("--localization-methods", nargs="+", default=None)
    parser.add_argument("--steering-localization-default", default="SSZ")
    parser.add_argument("--steering-localization-fallbacks", nargs="+", default=["GMDA"])
    parser.add_argument("--dynamic-chunk-seconds", type=float, default=1.0)
    parser.add_argument("--target-weight-mode", choices=["equal", "config"], default="equal")
    parser.add_argument("--target-weights-file", type=Path, default=None)

    parser.add_argument("--causal-only", action="store_true")
    parser.add_argument("--force-mic-count", type=int, default=6)
    parser.add_argument("--force-mic-radius", type=float, default=None)

    parser.add_argument("--topk-methods", type=int, default=3)
    parser.add_argument("--sanity-scenes-per-type", type=int, default=3)
    parser.add_argument("--noise-sweep-db", nargs="+", type=float, default=[5.0, 10.0, 20.0, 30.0])
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--verbose-scene-logs", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.causal_only and args.steering_time != "dynamic":
        raise ValueError("--causal-only requires --steering-time dynamic")

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
    scene_meta: dict[str, tuple[str, Path]] = {}

    total = len(selected)

    def _run_and_load(scene_path: Path, scene_type: str) -> tuple[str, list[dict], str, Path]:
        scene_id = scene_path.stem
        scene_out = runs_dir / scene_id
        _run_scene_job(args, scene_path=scene_path, out_dir=scene_out)
        summary_csv = scene_out / "steering_comparison.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing summary CSV for scene: {summary_csv}")
        rows = _load_rows(summary_csv)
        rows = _augment_rows_with_intelligibility(
            rows=rows,
            scene_out=scene_out,
            scene_cfg_path=scene_path,
            force_mic_count=args.force_mic_count,
            force_mic_radius=args.force_mic_radius,
        )
        return scene_id, rows, scene_type, scene_path

    with tqdm(total=total, desc="Scenes", unit="scene") as pbar:
        if args.workers <= 1:
            for scene_type, _k, scene_path in selected:
                scene_id, rows, stype, spath = _run_and_load(scene_path, scene_type)
                scene_meta[scene_id] = (stype, spath)
                all_rows.extend(rows)
                pbar.update(1)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = [pool.submit(_run_and_load, scene_path, scene_type) for scene_type, _k, scene_path in selected]
                for fut in concurrent.futures.as_completed(futures):
                    scene_id, rows, stype, spath = fut.result()
                    scene_meta[scene_id] = (stype, spath)
                    all_rows.extend(rows)
                    pbar.update(1)

    with_delta = _add_deltas(all_rows)
    agg = _aggregate(with_delta)
    top_methods = _select_top_methods(with_delta, args.topk_methods)

    _write_csv(out_dir / "scene_metrics.csv", with_delta)
    _write_csv(out_dir / "summary_by_method.csv", agg)
    _write_csv(out_dir / "top_methods.csv", top_methods)
    _write_summary(out_dir / "README_summary.md", run_id, len(selected), len(with_delta), agg, top_methods)

    sanity_artifacts: list[dict] = []
    noise_rows: list[dict] = []

    if not args.skip_sanity and top_methods:
        top_beamformers = [r["beamformer"] for r in top_methods]
        sanity_picks = _stratified_scene_pick(with_delta, selected, top_beamformers[0], args.sanity_scenes_per_type)

        for scene_type, scene_path, scene_id in tqdm(sanity_picks, desc="Sanity scenes", unit="scene"):
            scene_out = runs_dir / scene_id
            sanity_artifacts.extend(
                _render_sanity_artifacts(
                    out_dir=out_dir,
                    scene_type=scene_type,
                    scene_id=scene_id,
                    scene_path=scene_path,
                    scene_run_out=scene_out,
                    top_beamformers=top_beamformers,
                    force_mic_count=args.force_mic_count,
                    force_mic_radius=args.force_mic_radius,
                )
            )
            noise_rows.extend(
                _run_noise_sweep_for_scene(
                    args=args,
                    out_dir=out_dir,
                    scene_path=scene_path,
                    scene_type=scene_type,
                    scene_id=scene_id,
                    top_beamformers=top_beamformers,
                    snr_levels=list(args.noise_sweep_db),
                )
            )

        if noise_rows:
            _write_csv(out_dir / "sanity" / "noise_sweep_metrics.csv", noise_rows)
            scene_keys = {(r["scene_type"], r["scene"], r["beamformer"]) for r in noise_rows}
            for stype, sid, beamformer in scene_keys:
                scene_rows = [
                    r
                    for r in noise_rows
                    if r["scene_type"] == stype and r["scene"] == sid and r["beamformer"] == beamformer and r["steering_source"] == "oracle"
                ]
                scene_rows = sorted(scene_rows, key=lambda x: x["snr_db_target"])
                if not scene_rows:
                    continue
                plot_noise_sweep_trends(
                    snr_levels=[float(r["snr_db_target"]) for r in scene_rows],
                    delta_sii=[_safe_float(r.get("delta_sii")) for r in scene_rows],
                    delta_si_sdr=[_safe_float(r.get("delta_si_sdr_db")) for r in scene_rows],
                    delta_stoi=[_safe_float(r.get("delta_stoi")) for r in scene_rows],
                    title=f"{stype}/{sid} - {beamformer} Noise Sweep (Oracle)",
                    out_path=out_dir / "sanity" / stype / sid / f"noise_sweep_trends_{_slugify_beamformer(beamformer)}.png",
                )

    _write_pr_report(
        out_path=out_dir / "PR_REPORT.md",
        run_id=run_id,
        args=args,
        top_methods=top_methods,
        agg=agg,
        sanity_artifacts=sanity_artifacts,
        noise_rows=noise_rows,
    )

    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir.resolve(), target_is_directory=True)

    print(f"Done. Summary: {out_dir / 'README_summary.md'}")


if __name__ == "__main__":
    main()
