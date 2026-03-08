from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from direction_assignment import DirectionAssignmentConfig, DirectionAssignmentEngine, build_direction_assignment_input
from direction_assignment.audio_render import StitchAccumulator, write_audio_bundle
from direction_assignment.metrics import compute_track_jump_rate
from direction_assignment.scenario_synth import SceneConfig, generate_scene
from direction_assignment.visualize import plot_doa_timeline, plot_error_histogram, plot_room_topdown, plot_weight_timeline
from localization.benchmark.matching import match_predictions
from localization.target_policy import true_target_doas_deg
from realtime_pipeline.simulation_runner import run_simulation_pipeline
from realtime_pipeline.srp_tracker import SRPPeakTracker
from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation
from speaker_identity_grouping import IdentityChunkInput, IdentityConfig, SpeakerIdentityGrouper


PRIMARY_METRIC = {
    "localization": "mae_deg",
    "grouping": "switch_rate",
    "direction_assignment": "mae_deg",
    "pipeline": "mae_deg",
}


def _bucket_label(k: int) -> str:
    return "3+" if int(k) >= 3 else str(int(k))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _pick_scene_paths(per_bucket: int, seed: int) -> list[Path]:
    roots = [
        Path("simulation/simulations/configs/library_scene"),
        Path("simulation/simulations/configs/restaurant_scene"),
    ]
    rng = random.Random(seed)
    bucketed: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        for path in sorted(root.glob("*.json")):
            stem = path.stem
            if "_k" not in stem:
                continue
            k_str = stem.split("_k", 1)[1].split("_", 1)[0]
            if not k_str.isdigit():
                continue
            bucketed[_bucket_label(int(k_str))].append(path)

    picked: list[Path] = []
    for bucket in ["1", "2", "3+"]:
        files = list(bucketed.get(bucket, []))
        if not files:
            continue
        if len(files) <= per_bucket:
            picked.extend(files)
        else:
            picked.extend(rng.sample(files, per_bucket))
    return sorted(picked)


def _robust_identity_config(sample_rate: int, chunk_ms: int, robust_mode: bool) -> IdentityConfig:
    return IdentityConfig(
        sample_rate_hz=sample_rate,
        chunk_duration_ms=chunk_ms,
        continuity_bonus=0.12 if robust_mode else 0.0,
        switch_penalty=0.2 if robust_mode else 0.0,
        hold_similarity_threshold=0.45 if robust_mode else 1.1,
        carry_forward_chunks=3 if robust_mode else 0,
        confidence_decay=0.85 if robust_mode else 0.0,
    )


def _robust_direction_config(sample_rate: int, chunk_ms: int, robust_mode: bool) -> DirectionAssignmentConfig:
    return DirectionAssignmentConfig(
        sample_rate=sample_rate,
        chunk_ms=chunk_ms,
        transition_penalty_deg=35.0 if robust_mode else 180.0,
        min_confidence_for_switch=0.55 if robust_mode else 0.0,
        hold_confidence_decay=0.9 if robust_mode else 1.0,
        stale_confidence_decay=0.96 if robust_mode else 1.0,
        min_persist_confidence=0.05 if robust_mode else 0.0,
    )


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _norm_float(value: float | np.floating) -> float:
    return float(value) if np.isfinite(float(value)) else float("nan")


def _plot_series(path: Path, xs: list[float], series: dict[str, list[float]], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    for label, ys in series.items():
        if not ys:
            continue
        plt.plot(xs[: len(ys)], ys, label=label, linewidth=1.8)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    if series:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_bar_compare(path: Path, rows: list[dict], metric: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buckets = ["1", "2", "3+"]
    baseline = [next((float(r[metric]) for r in rows if r["bucket"] == b and r["mode"] == "baseline"), np.nan) for b in buckets]
    robust = [next((float(r[metric]) for r in rows if r["bucket"] == b and r["mode"] == "robust"), np.nan) for b in buckets]
    x = np.arange(len(buckets))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, baseline, width, label="baseline")
    plt.bar(x + width / 2, robust, width, label="robust")
    plt.xticks(x, buckets)
    plt.title(title)
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_delta_compare(path: Path, rows: list[dict], metric: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buckets = ["1", "2", "3+"]
    vals = [next((float(r[metric]) for r in rows if r["bucket"] == b), np.nan) for b in buckets]
    x = np.arange(len(buckets))
    plt.figure(figsize=(7, 4))
    plt.bar(x, vals, width=0.5)
    plt.xticks(x, buckets)
    plt.title(title)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _stage_scene_manifest(
    *,
    stage: str,
    scene_id: str,
    bucket: str,
    scene_dir: Path,
    artifact_complete: bool,
    extra: dict | None = None,
) -> dict:
    payload = {
        "stage": stage,
        "scene_id": scene_id,
        "bucket": bucket,
        "scene_dir": str(scene_dir),
        "artifact_complete": bool(artifact_complete),
    }
    if extra:
        payload.update(extra)
    return payload


def run_localization_validation(scene_paths: list[Path], out_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    manifests: list[dict] = []
    delta_rows: list[dict] = []
    for scene_path in scene_paths:
        sim_cfg = SimulationConfig.from_file(scene_path)
        mic_audio, mic_pos, _ = run_simulation(sim_cfg)
        true_doas = true_target_doas_deg(sim_cfg)
        bucket = _bucket_label(len(true_doas))
        frame_samples = max(1, int(sim_cfg.audio.fs * 10 / 1000))
        trackers = {
            "baseline": SRPPeakTracker(
                mic_pos=np.asarray(mic_pos, dtype=float),
                fs=sim_cfg.audio.fs,
                window_ms=40,
                nfft=512,
                overlap=0.5,
                freq_range=(200, 3000),
                max_sources=max(1, len(true_doas)),
                prior_enabled=False,
            ),
            "robust": SRPPeakTracker(
                mic_pos=np.asarray(mic_pos, dtype=float),
                fs=sim_cfg.audio.fs,
                window_ms=40,
                nfft=512,
                overlap=0.5,
                freq_range=(200, 3000),
                max_sources=max(1, len(true_doas)),
                prior_enabled=True,
            ),
        }
        per_frame_rows: list[dict] = []
        xs: list[float] = []
        mean_trace: dict[str, list[float]] = {"baseline": [], "robust": [], "true_mean": []}
        raw_trace: dict[str, list[float]] = {"baseline_raw": [], "robust_raw": []}
        prev_peaks: dict[str, list[float]] = {"baseline": [], "robust": []}
        tracked_by_mode: dict[str, list[list[float]]] = {"baseline": [], "robust": []}
        for frame_idx, start in enumerate(range(0, mic_audio.shape[0], frame_samples)):
            end = min(mic_audio.shape[0], start + frame_samples)
            frame = mic_audio[start:end, :]
            if frame.shape[0] < frame_samples:
                frame = np.pad(frame, ((0, frame_samples - frame.shape[0]), (0, 0)))
            xs.append(start / sim_cfg.audio.fs)
            mean_trace["true_mean"].append(float(np.mean(true_doas)) if true_doas else float("nan"))
            for mode, tracker in trackers.items():
                peaks, _scores, debug = tracker.update(frame)
                match = match_predictions(list(true_doas), list(peaks))
                mae = _mean_or_nan(match.matched_errors_deg)
                continuity = 0.0 if not true_doas else float(len(match.matched_true_indices) / len(true_doas))
                jitter = float("nan")
                if prev_peaks[mode] and peaks:
                    jitter_match = match_predictions(prev_peaks[mode], list(peaks))
                    jitter = _mean_or_nan(jitter_match.matched_errors_deg)
                prev_peaks[mode] = list(peaks)
                tracked_by_mode[mode].append(list(peaks))
                raw_peaks = list(debug.get("raw_peaks_deg", []))
                raw_trace[f"{mode}_raw"].append(float(np.mean(raw_peaks)) if raw_peaks else float("nan"))
                mean_trace[mode].append(float(np.mean(peaks)) if peaks else float("nan"))
                per_frame_rows.append(
                    {
                        "stage": "localization",
                        "scene_id": scene_path.stem,
                        "bucket": bucket,
                        "frame_idx": frame_idx,
                        "timestamp_s": start / sim_cfg.audio.fs,
                        "mode": mode,
                        "num_true": len(true_doas),
                        "num_pred": len(peaks),
                        "mae_deg": mae,
                        "jitter_deg": jitter,
                        "continuity": continuity,
                        "held_tracks": int(debug.get("held_tracks", 0)),
                    }
                )

        scene_dir = out_dir / scene_path.stem
        _write_csv(scene_dir / "per_frame_metrics.csv", per_frame_rows)
        _plot_series(scene_dir / "tracked_doa_compare.png", xs, mean_trace, f"Localization tracked DOA: {scene_path.stem}", "Mean DOA (deg)")
        _plot_series(scene_dir / "raw_vs_tracked.png", xs, {**raw_trace, "robust": mean_trace["robust"]}, f"Raw vs tracked DOA: {scene_path.stem}", "Mean DOA (deg)")

        scene_summary: dict[str, dict[str, float]] = {}
        for mode in ["baseline", "robust"]:
            subset = [row for row in per_frame_rows if row["mode"] == mode]
            scene_summary[mode] = {
                "mae_deg": _mean_or_nan([float(r["mae_deg"]) for r in subset if np.isfinite(r["mae_deg"])]),
                "jitter_deg": _mean_or_nan([float(r["jitter_deg"]) for r in subset if np.isfinite(r["jitter_deg"])]),
                "continuity": _mean_or_nan([float(r["continuity"]) for r in subset]),
                "held_tracks": _mean_or_nan([float(r["held_tracks"]) for r in subset]),
            }
            rows.append(
                {
                    "stage": "localization",
                    "scene_id": scene_path.stem,
                    "bucket": bucket,
                    "mode": mode,
                    **scene_summary[mode],
                }
            )
        delta_rows.append(
            {
                "stage": "localization",
                "scene_id": scene_path.stem,
                "bucket": bucket,
                "delta_mae_deg": _norm_float(scene_summary["robust"]["mae_deg"] - scene_summary["baseline"]["mae_deg"]),
                "improvement_mae_deg": _norm_float(scene_summary["baseline"]["mae_deg"] - scene_summary["robust"]["mae_deg"]),
                "delta_jitter_deg": _norm_float(scene_summary["robust"]["jitter_deg"] - scene_summary["baseline"]["jitter_deg"]),
                "delta_continuity": _norm_float(scene_summary["robust"]["continuity"] - scene_summary["baseline"]["continuity"]),
            }
        )
        _write_json(scene_dir / "scene_summary.json", scene_summary)
        manifests.append(
            _stage_scene_manifest(
                stage="localization",
                scene_id=scene_path.stem,
                bucket=bucket,
                scene_dir=scene_dir,
                artifact_complete=(scene_dir / "per_frame_metrics.csv").exists() and (scene_dir / "tracked_doa_compare.png").exists(),
            )
        )
    return rows, delta_rows, manifests


def run_grouping_validation(out_dir: Path, seed: int, scenes_per_bucket: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    manifests: list[dict] = []
    delta_rows: list[dict] = []
    for bucket_k in [1, 2, 3]:
        for scene_idx in range(scenes_per_bucket):
            scene = generate_scene(
                scene_id=f"grouping_k{bucket_k}_{scene_idx:02d}",
                cfg=SceneConfig(
                    sample_rate=16000,
                    duration_sec=8.0,
                    chunk_ms=200,
                    n_speakers=bucket_k,
                    n_mics=4,
                    mic_spacing_m=0.04,
                    noise_std=0.015,
                    bleed_ratio=0.15,
                    moving_probability=0.5,
                    srp_peak_noise_deg=6.0,
                    srp_num_distractors=1,
                ),
                rng=rng,
            )
            bucket = _bucket_label(bucket_k)
            scene_dir = out_dir / scene.scene_id
            timeline_rows: list[dict] = []
            scene_summary: dict[str, dict[str, float]] = {}
            for mode in ["baseline", "robust"]:
                grouper = SpeakerIdentityGrouper(_robust_identity_config(scene.sample_rate, 200, mode == "robust"))
                by_oracle: dict[int, list[int]] = defaultdict(list)
                switches = 0
                transitions = 0
                for chunk_id, streams in enumerate(scene.separated_streams_by_chunk):
                    out = grouper.update(
                        IdentityChunkInput(
                            chunk_id=chunk_id,
                            timestamp_ms=chunk_id * 200.0,
                            sample_rate_hz=scene.sample_rate,
                            streams=streams,
                        )
                    )
                    for stream_idx, oracle_label in scene.stream_to_oracle_by_chunk[chunk_id].items():
                        sid = out.stream_to_speaker.get(stream_idx)
                        if sid is None:
                            continue
                        seq = by_oracle[int(oracle_label)]
                        if seq:
                            transitions += 1
                            if seq[-1] != int(sid):
                                switches += 1
                        seq.append(int(sid))
                        timeline_rows.append(
                            {
                                "stage": "grouping",
                                "scene_id": scene.scene_id,
                                "bucket": bucket,
                                "mode": mode,
                                "chunk_id": chunk_id,
                                "oracle_label": int(oracle_label),
                                "speaker_id": int(sid),
                                "confidence": float(out.per_stream_confidence.get(stream_idx, 0.0)),
                            }
                        )
                majority_hits = 0
                total = 0
                unique_pred_speakers = set()
                for seq in by_oracle.values():
                    if not seq:
                        continue
                    vals, counts = np.unique(np.asarray(seq, dtype=int), return_counts=True)
                    majority_hits += int(np.max(counts))
                    total += len(seq)
                    unique_pred_speakers.update(int(v) for v in vals.tolist())
                scene_summary[mode] = {
                    "switch_rate": float(switches / transitions) if transitions else 0.0,
                    "stability": float(majority_hits / total) if total else 0.0,
                    "mean_confidence": _mean_or_nan([float(r["confidence"]) for r in timeline_rows if r["mode"] == mode]),
                    "speaker_count_ratio": float(len(unique_pred_speakers) / max(1, bucket_k)),
                }
                rows.append(
                    {
                        "stage": "grouping",
                        "scene_id": scene.scene_id,
                        "bucket": bucket,
                        "mode": mode,
                        **scene_summary[mode],
                    }
                )

            _write_csv(scene_dir / "assignment_timeline.csv", [r for r in timeline_rows if r["scene_id"] == scene.scene_id])
            xs = sorted({float(r["chunk_id"]) * 0.2 for r in timeline_rows if r["scene_id"] == scene.scene_id})
            series: dict[str, list[float]] = {}
            for mode in ["baseline", "robust"]:
                for oracle_label in range(bucket_k):
                    series[f"{mode}:oracle{oracle_label}"] = [
                        float(r["speaker_id"])
                        for r in timeline_rows
                        if r["scene_id"] == scene.scene_id and r["mode"] == mode and int(r["oracle_label"]) == oracle_label
                    ]
            _plot_series(scene_dir / "assignment_trace.png", xs, series, f"Grouping assignments: {scene.scene_id}", "Pred speaker id")
            _write_json(scene_dir / "scene_summary.json", scene_summary)
            delta_rows.append(
                {
                    "stage": "grouping",
                    "scene_id": scene.scene_id,
                    "bucket": bucket,
                    "delta_switch_rate": _norm_float(scene_summary["robust"]["switch_rate"] - scene_summary["baseline"]["switch_rate"]),
                    "improvement_switch_rate": _norm_float(scene_summary["baseline"]["switch_rate"] - scene_summary["robust"]["switch_rate"]),
                    "delta_stability": _norm_float(scene_summary["robust"]["stability"] - scene_summary["baseline"]["stability"]),
                    "delta_mean_confidence": _norm_float(scene_summary["robust"]["mean_confidence"] - scene_summary["baseline"]["mean_confidence"]),
                }
            )
            manifests.append(
                _stage_scene_manifest(
                    stage="grouping",
                    scene_id=scene.scene_id,
                    bucket=bucket,
                    scene_dir=scene_dir,
                    artifact_complete=(scene_dir / "assignment_timeline.csv").exists() and (scene_dir / "assignment_trace.png").exists(),
                )
            )
    return rows, delta_rows, manifests


def run_direction_validation(out_dir: Path, seed: int, scenes_per_bucket: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(seed + 17)
    rows: list[dict] = []
    manifests: list[dict] = []
    delta_rows: list[dict] = []
    for bucket_k in [1, 2, 3]:
        for scene_idx in range(scenes_per_bucket):
            scene = generate_scene(
                scene_id=f"direction_k{bucket_k}_{scene_idx:02d}",
                cfg=SceneConfig(
                    sample_rate=16000,
                    duration_sec=8.0,
                    chunk_ms=200,
                    n_speakers=bucket_k,
                    n_mics=4,
                    mic_spacing_m=0.04,
                    noise_std=0.015,
                    bleed_ratio=0.15,
                    moving_probability=0.6,
                    srp_peak_noise_deg=8.0,
                    srp_num_distractors=2,
                ),
                rng=rng,
            )
            bucket = _bucket_label(bucket_k)
            scene_dir = out_dir / scene.scene_id
            per_chunk_rows: list[dict] = []
            scene_summary: dict[str, dict[str, float]] = {}
            mix_mono = np.mean(scene.raw_mic, axis=1)
            oracle_doa_chunks = [
                {i: float(scene.oracle_doa_deg_by_chunk[chunk_id, i]) for i in range(scene.oracle_sources.shape[0])}
                for chunk_id in range(len(scene.separated_streams_by_chunk))
            ]
            for mode in ["baseline", "robust"]:
                mode_dir = scene_dir / mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                engine = DirectionAssignmentEngine(
                    mic_geometry=scene.mic_geometry_xy,
                    config=_robust_direction_config(scene.sample_rate, 200, mode == "robust"),
                )
                track_history: dict[int, list[float]] = defaultdict(list)
                all_errors: list[float] = []
                chunk_times_s: list[float] = []
                target_speaker_ids_seq: list[list[int]] = []
                target_weights_seq: list[list[float]] = []
                est_doa_chunks: list[dict[int, float]] = []
                stitch = StitchAccumulator(total_samples=scene.raw_mic.shape[0])
                for chunk_id, streams in enumerate(scene.separated_streams_by_chunk):
                    start = chunk_id * scene.chunk_samples
                    end = min(scene.raw_mic.shape[0], start + scene.chunk_samples)
                    payload, _ = build_direction_assignment_input(
                        chunk_id=chunk_id,
                        timestamp_ms=chunk_id * 200.0,
                        raw_mic_chunk=scene.raw_mic[start:end, :],
                        separated_streams=streams,
                        stream_to_speaker=scene.stream_to_oracle_by_chunk[chunk_id],
                        active_speakers=sorted(scene.stream_to_oracle_by_chunk[chunk_id].values()),
                        srp_doa_peaks_deg=list(scene.srp_peaks_by_chunk[chunk_id]),
                        srp_peak_scores=None if scene.srp_scores_by_chunk[chunk_id] is None else list(scene.srp_scores_by_chunk[chunk_id]),
                    )
                    out = engine.update(payload)
                    chunk_errors = []
                    est_map: dict[int, float] = {}
                    for speaker_id, doa in out.speaker_directions_deg.items():
                        true_doa = float(scene.oracle_doa_deg_by_chunk[chunk_id, int(speaker_id)])
                        err = float(abs((doa - true_doa + 180.0) % 360.0 - 180.0))
                        est_map[int(speaker_id)] = float(doa)
                        all_errors.append(err)
                        chunk_errors.append(err)
                        track_history[int(speaker_id)].append(float(doa))
                    for sidx, sid in scene.stream_to_oracle_by_chunk[chunk_id].items():
                        if sidx < 0 or sidx >= len(streams):
                            continue
                        stitch.add_chunk(int(sid), streams[sidx], start, end)
                    est_doa_chunks.append(est_map)
                    chunk_times_s.append(start / scene.sample_rate)
                    target_speaker_ids_seq.append(list(out.target_speaker_ids))
                    target_weights_seq.append(list(out.target_weights))
                    per_chunk_rows.append(
                        {
                            "stage": "direction_assignment",
                            "scene_id": scene.scene_id,
                            "bucket": bucket,
                            "mode": mode,
                            "chunk_id": chunk_id,
                            "timestamp_s": start / scene.sample_rate,
                            "mae_deg": _mean_or_nan(chunk_errors),
                            "num_tracks": len(out.speaker_directions_deg),
                        }
                    )
                scene_summary[mode] = {
                    "mae_deg": _mean_or_nan(all_errors),
                    "jump_rate": compute_track_jump_rate(track_history),
                    "mismatch_rate": float(np.mean(np.asarray(all_errors, dtype=float) > 15.0)) if all_errors else 0.0,
                }
                rows.append(
                    {
                        "stage": "direction_assignment",
                        "scene_id": scene.scene_id,
                        "bucket": bucket,
                        "mode": mode,
                        **scene_summary[mode],
                    }
                )
                _write_csv(mode_dir / "per_chunk_metrics.csv", [r for r in per_chunk_rows if r["scene_id"] == scene.scene_id and r["mode"] == mode])
                plot_doa_timeline(mode_dir / "plots" / "doa_timeline.png", chunk_times_s, oracle_doa_chunks, est_doa_chunks)
                plot_error_histogram(mode_dir / "plots" / "error_histogram.png", all_errors)
                plot_room_topdown(mode_dir / "plots" / "room_topdown.png", scene.mic_geometry_xy, oracle_doa_chunks)
                plot_weight_timeline(mode_dir / "plots" / "weight_timeline.png", chunk_times_s, target_speaker_ids_seq, target_weights_seq)
                write_audio_bundle(
                    out_dir=mode_dir / "audio",
                    sample_rate=scene.sample_rate,
                    mixture_mono=mix_mono,
                    oracle_sources=scene.oracle_sources,
                    estimated_tracks=stitch.render(),
                )

            _write_csv(scene_dir / "per_chunk_metrics_all.csv", [r for r in per_chunk_rows if r["scene_id"] == scene.scene_id])
            _write_json(scene_dir / "scene_summary.json", scene_summary)
            _plot_series(
                scene_dir / "compare_mae_trace.png",
                [float(r["chunk_id"]) * 0.2 for r in per_chunk_rows if r["scene_id"] == scene.scene_id and r["mode"] == "baseline"],
                {
                    "baseline": [float(r["mae_deg"]) for r in per_chunk_rows if r["scene_id"] == scene.scene_id and r["mode"] == "baseline"],
                    "robust": [float(r["mae_deg"]) for r in per_chunk_rows if r["scene_id"] == scene.scene_id and r["mode"] == "robust"],
                },
                f"Direction assignment MAE: {scene.scene_id}",
                "MAE (deg)",
            )
            delta_rows.append(
                {
                    "stage": "direction_assignment",
                    "scene_id": scene.scene_id,
                    "bucket": bucket,
                    "delta_mae_deg": _norm_float(scene_summary["robust"]["mae_deg"] - scene_summary["baseline"]["mae_deg"]),
                    "improvement_mae_deg": _norm_float(scene_summary["baseline"]["mae_deg"] - scene_summary["robust"]["mae_deg"]),
                    "delta_jump_rate": _norm_float(scene_summary["robust"]["jump_rate"] - scene_summary["baseline"]["jump_rate"]),
                    "delta_mismatch_rate": _norm_float(scene_summary["robust"]["mismatch_rate"] - scene_summary["baseline"]["mismatch_rate"]),
                }
            )
            manifests.append(
                _stage_scene_manifest(
                    stage="direction_assignment",
                    scene_id=scene.scene_id,
                    bucket=bucket,
                    scene_dir=scene_dir,
                    artifact_complete=all(
                        [
                            (scene_dir / "baseline" / "audio" / "mixture.wav").exists(),
                            (scene_dir / "robust" / "audio" / "mixture.wav").exists(),
                            (scene_dir / "compare_mae_trace.png").exists(),
                        ]
                    ),
                )
            )
    return rows, delta_rows, manifests


def run_pipeline_validation(scene_paths: list[Path], out_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    manifests: list[dict] = []
    delta_rows: list[dict] = []
    for scene_path in scene_paths:
        sim_cfg = SimulationConfig.from_file(scene_path)
        true_doas = true_target_doas_deg(sim_cfg)
        bucket = _bucket_label(len(true_doas))
        scene_dir = out_dir / scene_path.stem
        trace_rows: list[dict] = []
        scene_summary: dict[str, dict[str, float]] = {}
        for mode in ["baseline", "robust"]:
            mode_out = scene_dir / mode
            summary = run_simulation_pipeline(
                scene_config_path=scene_path,
                out_dir=mode_out,
                use_mock_separation=True,
                robust_mode=(mode == "robust"),
                capture_trace=True,
            )
            frame_maes: list[float] = []
            churn_values: list[float] = []
            prev_dirs: list[float] = []
            for frame in summary.get("speaker_map_trace", []):
                dirs = [float(item["direction_degrees"]) for item in frame.get("speakers", [])]
                match = match_predictions(list(true_doas), dirs)
                mae = _mean_or_nan(match.matched_errors_deg)
                if np.isfinite(mae):
                    frame_maes.append(float(mae))
                churn = float("nan")
                if prev_dirs and dirs:
                    jitter_match = match_predictions(prev_dirs, dirs)
                    churn = _mean_or_nan(jitter_match.matched_errors_deg)
                    if np.isfinite(churn):
                        churn_values.append(float(churn))
                prev_dirs = dirs
                trace_rows.append(
                    {
                        "stage": "pipeline",
                        "scene_id": scene_path.stem,
                        "bucket": bucket,
                        "mode": mode,
                        "frame_index": int(frame["frame_index"]),
                        "num_speakers": len(dirs),
                        "mae_deg": mae,
                        "map_churn_deg": churn,
                    }
                )
            scene_summary[mode] = {
                "mae_deg": _mean_or_nan(frame_maes),
                "map_churn_deg": _mean_or_nan(churn_values),
                "mismatch_rate": float(np.mean(np.asarray(frame_maes, dtype=float) > 15.0)) if frame_maes else 0.0,
            }
            rows.append(
                {
                    "stage": "pipeline",
                    "scene_id": scene_path.stem,
                    "bucket": bucket,
                    "mode": mode,
                    **scene_summary[mode],
                }
            )
        _write_csv(scene_dir / "speaker_map_trace_metrics.csv", [r for r in trace_rows if r["scene_id"] == scene_path.stem])
        _write_json(scene_dir / "scene_summary.json", scene_summary)
        xs = [float(r["frame_index"]) * 0.01 for r in trace_rows if r["scene_id"] == scene_path.stem and r["mode"] == "baseline"]
        _plot_series(
            scene_dir / "pipeline_mae_trace.png",
            xs,
            {
                "baseline": [float(r["mae_deg"]) for r in trace_rows if r["scene_id"] == scene_path.stem and r["mode"] == "baseline"],
                "robust": [float(r["mae_deg"]) for r in trace_rows if r["scene_id"] == scene_path.stem and r["mode"] == "robust"],
            },
            f"Pipeline MAE: {scene_path.stem}",
            "MAE (deg)",
        )
        delta_rows.append(
            {
                "stage": "pipeline",
                "scene_id": scene_path.stem,
                "bucket": bucket,
                "delta_mae_deg": _norm_float(scene_summary["robust"]["mae_deg"] - scene_summary["baseline"]["mae_deg"]),
                "improvement_mae_deg": _norm_float(scene_summary["baseline"]["mae_deg"] - scene_summary["robust"]["mae_deg"]),
                "delta_map_churn_deg": _norm_float(scene_summary["robust"]["map_churn_deg"] - scene_summary["baseline"]["map_churn_deg"]),
                "delta_mismatch_rate": _norm_float(scene_summary["robust"]["mismatch_rate"] - scene_summary["baseline"]["mismatch_rate"]),
            }
        )
        manifests.append(
            _stage_scene_manifest(
                stage="pipeline",
                scene_id=scene_path.stem,
                bucket=bucket,
                scene_dir=scene_dir,
                artifact_complete=all(
                    [
                        (scene_dir / "baseline" / "enhanced_fast_path.wav").exists(),
                        (scene_dir / "robust" / "enhanced_fast_path.wav").exists(),
                        (scene_dir / "baseline" / "raw_mix_mean.wav").exists(),
                        (scene_dir / "robust" / "raw_mix_mean.wav").exists(),
                        (scene_dir / "pipeline_mae_trace.png").exists(),
                    ]
                ),
            )
        )
    return rows, delta_rows, manifests


def summarize_by_bucket(stage_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in stage_rows:
        grouped[(str(row["stage"]), str(row["bucket"]), str(row["mode"]))].append(row)
    summary: list[dict] = []
    for (stage, bucket, mode), rows in sorted(grouped.items()):
        metrics = sorted({key for row in rows for key in row.keys() if key not in {"stage", "scene_id", "bucket", "mode"}})
        out = {"stage": stage, "bucket": bucket, "mode": mode}
        for metric in metrics:
            vals = [float(row[metric]) for row in rows if metric in row and np.isfinite(float(row[metric]))]
            out[metric] = _mean_or_nan(vals)
        summary.append(out)
    return summary


def summarize_deltas_by_bucket(delta_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in delta_rows:
        grouped[(str(row["stage"]), str(row["bucket"]))].append(row)
    summary: list[dict] = []
    for (stage, bucket), rows in sorted(grouped.items()):
        metrics = sorted({key for row in rows for key in row.keys() if key not in {"stage", "scene_id", "bucket"}})
        out = {"stage": stage, "bucket": bucket}
        for metric in metrics:
            vals = [float(row[metric]) for row in rows if metric in row and np.isfinite(float(row[metric]))]
            out[metric] = _mean_or_nan(vals)
        summary.append(out)
    return summary


def _select_showcases(delta_rows: list[dict], manifests: list[dict], showcase_count: int) -> list[dict]:
    manifest_map = {(m["stage"], m["scene_id"]): m for m in manifests}
    selected: list[dict] = []
    for stage in sorted({row["stage"] for row in delta_rows}):
        primary_metric = f"improvement_{PRIMARY_METRIC[stage]}"
        for bucket in ["1", "2", "3+"]:
            candidates = [
                row for row in delta_rows
                if row["stage"] == stage and row["bucket"] == bucket and np.isfinite(float(row.get(primary_metric, float("nan"))))
            ]
            if not candidates:
                continue
            complete = [row for row in candidates if manifest_map.get((stage, row["scene_id"]), {}).get("artifact_complete")]
            if complete:
                candidates = complete
            median_improvement = float(np.median([float(row[primary_metric]) for row in candidates]))
            candidates = sorted(
                candidates,
                key=lambda row: (
                    abs(float(row[primary_metric]) - median_improvement),
                    str(row["scene_id"]),
                ),
            )
            for row in candidates[:showcase_count]:
                manifest = manifest_map.get((stage, row["scene_id"]), {})
                selected.append(
                    {
                        "stage": stage,
                        "bucket": bucket,
                        "scene_id": row["scene_id"],
                        "primary_metric": primary_metric,
                        "primary_improvement": float(row[primary_metric]),
                        "scene_dir": manifest.get("scene_dir", ""),
                    }
                )
    return selected


def _materialize_showcases(out_root: Path, showcases: list[dict]) -> None:
    for item in showcases:
        src = Path(str(item["scene_dir"]))
        if not src.exists():
            continue
        dst = out_root / "showcases" / str(item["stage"]) / str(item["bucket"]) / str(item["scene_id"])
        _link_or_copy(src, dst)


def _write_stage_outputs(out_root: Path, stage: str, stage_rows: list[dict], stage_delta_rows: list[dict]) -> None:
    stage_dir = out_root / stage
    _write_csv(stage_dir / "per_scene_metrics.csv", [row for row in stage_rows if row["stage"] == stage])
    _write_csv(stage_dir / "per_scene_deltas.csv", [row for row in stage_delta_rows if row["stage"] == stage])
    bucket_rows = summarize_by_bucket([row for row in stage_rows if row["stage"] == stage])
    delta_bucket_rows = summarize_deltas_by_bucket([row for row in stage_delta_rows if row["stage"] == stage])
    _write_csv(stage_dir / "summary_by_bucket.csv", bucket_rows)
    _write_csv(stage_dir / "delta_summary_by_bucket.csv", delta_bucket_rows)
    metric = PRIMARY_METRIC.get(stage, "mae_deg")
    if bucket_rows:
        _plot_bar_compare(stage_dir / f"{stage}_compare.png", bucket_rows, metric, f"{stage} baseline vs robust")
    improvement_metric = f"improvement_{metric}"
    if delta_bucket_rows and any(improvement_metric in row for row in delta_bucket_rows):
        _plot_delta_compare(stage_dir / f"{stage}_delta.png", delta_bucket_rows, improvement_metric, f"{stage} improvement by bucket")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run robustness validation across localization, grouping, direction assignment, and full pipeline.")
    parser.add_argument("--scenes-per-bucket", type=int, default=2)
    parser.add_argument("--synthetic-scenes-per-bucket", type=int, default=2)
    parser.add_argument("--showcase-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir or f"realtime_pipeline/output/robustness_validation/{ts}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    scene_paths = _pick_scene_paths(per_bucket=args.scenes_per_bucket, seed=args.seed)
    localization_rows, localization_delta_rows, localization_manifests = run_localization_validation(scene_paths, out_root / "localization")
    grouping_rows, grouping_delta_rows, grouping_manifests = run_grouping_validation(out_root / "grouping", args.seed, args.synthetic_scenes_per_bucket)
    direction_rows, direction_delta_rows, direction_manifests = run_direction_validation(out_root / "direction_assignment", args.seed, args.synthetic_scenes_per_bucket)
    pipeline_rows, pipeline_delta_rows, pipeline_manifests = run_pipeline_validation(scene_paths, out_root / "pipeline")

    all_rows = localization_rows + grouping_rows + direction_rows + pipeline_rows
    all_delta_rows = localization_delta_rows + grouping_delta_rows + direction_delta_rows + pipeline_delta_rows
    all_manifests = localization_manifests + grouping_manifests + direction_manifests + pipeline_manifests

    _write_csv(out_root / "per_scene_metrics.csv", all_rows)
    _write_csv(out_root / "per_scene_deltas.csv", all_delta_rows)
    _write_csv(out_root / "artifact_manifest.csv", all_manifests)

    for stage in ["localization", "grouping", "direction_assignment", "pipeline"]:
        _write_stage_outputs(out_root, stage, all_rows, all_delta_rows)

    bucket_summary = summarize_by_bucket(all_rows)
    delta_bucket_summary = summarize_deltas_by_bucket(all_delta_rows)
    _write_csv(out_root / "summary_by_bucket.csv", bucket_summary)
    _write_csv(out_root / "delta_summary_by_bucket.csv", delta_bucket_summary)

    showcases = _select_showcases(all_delta_rows, all_manifests, args.showcase_count)
    _materialize_showcases(out_root, showcases)
    _write_json(out_root / "showcases.json", showcases)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scene_paths": [str(path) for path in scene_paths],
        "per_scene_metrics_csv": str(out_root / "per_scene_metrics.csv"),
        "per_scene_deltas_csv": str(out_root / "per_scene_deltas.csv"),
        "summary_by_bucket_csv": str(out_root / "summary_by_bucket.csv"),
        "delta_summary_by_bucket_csv": str(out_root / "delta_summary_by_bucket.csv"),
        "artifact_manifest_csv": str(out_root / "artifact_manifest.csv"),
        "showcases_json": str(out_root / "showcases.json"),
        "showcases": showcases,
    }
    _write_json(out_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
