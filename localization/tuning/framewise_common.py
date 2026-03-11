from __future__ import annotations

import concurrent.futures
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from localization.benchmark.algo_runner import _build_algorithm
from localization.tuning.common import DEFAULT_PROFILE, SceneGeometry, apply_profile, write_csv, write_json
from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation


def _normalize_deg(value: float) -> float:
    return float(value % 360.0)


def _angular_error_deg(pred: float, truth: float) -> float:
    return float(abs((float(pred) - float(truth) + 180.0) % 360.0 - 180.0))


def window_ms_to_samples(window_ms: int, fs: int) -> int:
    return max(1, int(round(float(fs) * float(window_ms) / 1000.0)))


def hop_ms_to_samples(hop_ms: int, fs: int) -> int:
    return max(1, int(round(float(fs) * float(hop_ms) / 1000.0)))


def build_framewise_eval_config(
    *,
    base_cfg: dict[str, Any],
    fs: int,
    window_ms: int | None = None,
    hop_ms: int | None = None,
    overlap: float | None = None,
    freq_low_hz: int | None = None,
    freq_high_hz: int | None = None,
    extra_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(base_cfg)
    if window_ms is not None:
        cfg["window_ms"] = int(window_ms)
        cfg["nfft"] = window_ms_to_samples(int(window_ms), fs)
    if hop_ms is not None:
        cfg["hop_ms"] = int(hop_ms)
    if overlap is not None:
        cfg["overlap"] = float(overlap)
    if freq_low_hz is not None or freq_high_hz is not None:
        current = list(cfg.get("freq_range", [200, 3000]))
        if freq_low_hz is not None:
            current[0] = int(freq_low_hz)
        if freq_high_hz is not None:
            current[1] = int(freq_high_hz)
        if current[0] >= current[1]:
            raise ValueError(f"Invalid frequency range: {current}")
        cfg["freq_range"] = current
    if extra_updates:
        cfg.update(extra_updates)
    return cfg


def _extract_speaker_doa_from_metadata(meta: dict, mic_center_xy: np.ndarray) -> dict[int, float]:
    doa_by_speaker: dict[int, float] = {}

    for row in meta.get("assets", {}).get("render_segments", []):
        if row.get("classification") != "speech" or "speaker_id" not in row:
            continue
        sid = int(row["speaker_id"])
        pos = np.asarray(row.get("position_m", row.get("loc", [0.0, 0.0]))[:2], dtype=float)
        delta = pos - mic_center_xy
        doa_by_speaker[sid] = _normalize_deg(np.degrees(np.arctan2(delta[1], delta[0])))

    for row in meta.get("assets", {}).get("speech", []):
        if "speaker_id" not in row:
            continue
        sid = int(row["speaker_id"])
        if "angle_deg" in row:
            doa_by_speaker[sid] = _normalize_deg(float(row["angle_deg"]))
            continue
        pos = np.asarray(row.get("position_m", row.get("loc", [0.0, 0.0]))[:2], dtype=float)
        delta = pos - mic_center_xy
        doa_by_speaker[sid] = _normalize_deg(np.degrees(np.arctan2(delta[1], delta[0])))

    return doa_by_speaker


def _extract_speech_events(meta: dict) -> list[dict[str, float | int]]:
    events = [
        {
            "speaker_id": int(row["speaker_id"]),
            "start_sec": float(row.get("start_sec", 0.0)),
            "end_sec": float(row.get("end_sec", 0.0)),
        }
        for row in meta.get("assets", {}).get("speech_events", [])
        if "speaker_id" in row
    ]
    if events:
        return sorted(events, key=lambda row: (float(row["start_sec"]), float(row["end_sec"])))

    for row in meta.get("assets", {}).get("speech", []):
        if "speaker_id" not in row:
            continue
        active_window = row.get("active_window_sec")
        if not isinstance(active_window, list) or len(active_window) < 2:
            continue
        events.append(
            {
                "speaker_id": int(row["speaker_id"]),
                "start_sec": float(active_window[0]),
                "end_sec": float(active_window[1]),
            }
        )
    return sorted(events, key=lambda row: (float(row["start_sec"]), float(row["end_sec"])))


def _active_speaker_truth(time_s: float, speech_events: list[dict[str, float | int]], doa_by_speaker: dict[int, float]) -> tuple[int | None, float | None]:
    active_rows = [
        row
        for row in speech_events
        if float(row.get("start_sec", 0.0)) <= float(time_s) < float(row.get("end_sec", 0.0))
    ]
    if not active_rows:
        return None, None
    active_rows.sort(key=lambda row: (float(row.get("start_sec", 0.0)), float(row.get("end_sec", 0.0))))
    chosen = active_rows[-1]
    sid = int(chosen["speaker_id"])
    return sid, float(doa_by_speaker.get(sid, math.nan))


def _window_iter(mic_audio: np.ndarray, fs: int, window_ms: int, hop_ms: int):
    window_samples = window_ms_to_samples(window_ms, fs)
    hop_samples = hop_ms_to_samples(hop_ms, fs)
    total = mic_audio.shape[0]
    if total <= 0:
        return
    for start in range(0, max(1, total - window_samples + 1), hop_samples):
        end = min(total, start + window_samples)
        frame = mic_audio[start:end, :]
        if frame.shape[0] < window_samples:
            frame = np.pad(frame, ((0, window_samples - frame.shape[0]), (0, 0)))
        center_s = (start + (frame.shape[0] / 2.0)) / float(fs)
        yield center_s, frame


def evaluate_method_framewise_on_scene(
    *,
    method: str,
    method_cfg: dict[str, Any],
    scene: SceneGeometry,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    sim_cfg = apply_profile(SimulationConfig.from_file(scene.scene_path), profile=profile)
    mic_audio, mic_pos_abs, _ = run_simulation(sim_cfg)
    center = np.asarray(sim_cfg.microphone_array.mic_center, dtype=float).reshape(3, 1)
    mic_pos_rel = mic_pos_abs - center

    metadata: dict[str, Any] = {}
    if scene.metadata_path and Path(scene.metadata_path).exists():
        metadata = json.loads(Path(scene.metadata_path).read_text(encoding="utf-8"))
    mic_center_xy = np.asarray(sim_cfg.microphone_array.mic_center[:2], dtype=float)
    speech_events = _extract_speech_events(metadata)
    doa_by_speaker = _extract_speaker_doa_from_metadata(metadata, mic_center_xy)

    window_ms = int(method_cfg.get("window_ms", 200))
    hop_ms = int(method_cfg.get("hop_ms", max(30, int(round(window_ms / 4)))))
    algo = _build_algorithm(
        method=method,
        mic_pos_rel=mic_pos_rel,
        fs=sim_cfg.audio.fs,
        n_true_targets=1,
        cfg={**method_cfg, "max_sources": 1},
    )

    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    runtimes_ms: list[float] = []
    pred_values: list[float] = []
    truth_values: list[float] = []
    coverage_hits = 0
    eval_frames = 0

    for time_s, frame in _window_iter(mic_audio, sim_cfg.audio.fs, window_ms, hop_ms):
        active_speaker_id, active_truth_deg = _active_speaker_truth(time_s, speech_events, doa_by_speaker)
        if active_speaker_id is None or active_truth_deg is None or not np.isfinite(float(active_truth_deg)):
            continue

        t0 = time.perf_counter()
        estimated_doas_rad, _hist, _history = algo.process(frame.T)
        runtimes_ms.append((time.perf_counter() - t0) * 1000.0)
        pred_deg = float("nan")
        if estimated_doas_rad:
            pred_deg = _normalize_deg(math.degrees(float(estimated_doas_rad[0])))

        eval_frames += 1
        if np.isfinite(pred_deg):
            coverage_hits += 1
            err = _angular_error_deg(pred_deg, float(active_truth_deg))
            errors.append(err)
        else:
            err = math.nan

        rows.append(
            {
                "time_s": round(float(time_s), 6),
                "active_speaker_id": int(active_speaker_id),
                "truth_doa_deg": float(active_truth_deg),
                "pred_doa_deg": None if not np.isfinite(pred_deg) else float(pred_deg),
                "error_deg": None if not np.isfinite(err) else float(err),
                "runtime_ms": float(runtimes_ms[-1]) if runtimes_ms else None,
            }
        )
        pred_values.append(pred_deg)
        truth_values.append(float(active_truth_deg))

    valid = np.asarray(errors, dtype=np.float64)
    return {
        "scene_id": scene.scene_id,
        "scene_path": str(scene.scene_path),
        "scene_type": scene.scene_type,
        "main_angle_deg": scene.main_angle_deg,
        "secondary_angle_deg": scene.secondary_angle_deg,
        "noise_layout_type": scene.noise_layout_type,
        "noise_angles_deg": ",".join(str(v) for v in scene.noise_angles_deg),
        "method": method,
        "status": "ok",
        "window_ms": int(window_ms),
        "hop_ms": int(hop_ms),
        "overlap": float(method_cfg.get("overlap", 0.5)),
        "freq_low_hz": int((method_cfg.get("freq_range") or [200, 3000])[0]),
        "freq_high_hz": int((method_cfg.get("freq_range") or [200, 3000])[1]),
        "eval_frames": int(eval_frames),
        "coverage_rate": float(coverage_hits / eval_frames) if eval_frames else None,
        "mae_deg": float(np.mean(valid)) if valid.size else None,
        "acc_at_10": float(np.mean(valid <= 10.0)) if valid.size else None,
        "acc_at_20": float(np.mean(valid <= 20.0)) if valid.size else None,
        "acc_at_25": float(np.mean(valid <= 25.0)) if valid.size else None,
        "median_deg": float(np.median(valid)) if valid.size else None,
        "p90_deg": float(np.percentile(valid, 90)) if valid.size else None,
        "runtime_ms_mean": float(np.mean(runtimes_ms)) if runtimes_ms else None,
        "timeline_rows": rows,
    }


def evaluate_method_framewise_on_scenes(
    *,
    method: str,
    method_cfg: dict[str, Any],
    scenes: list[SceneGeometry],
    profile: str = DEFAULT_PROFILE,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    max_workers = max(1, workers or max(1, (os.cpu_count() or 1) - 1))
    if max_workers == 1:
        return [
            evaluate_method_framewise_on_scene(method=method, method_cfg=dict(method_cfg), scene=scene, profile=profile)
            for scene in sorted(scenes, key=lambda item: item.scene_id)
        ]

    jobs = [(method, dict(method_cfg), scene, profile) for scene in scenes]
    rows: list[dict[str, Any]] = []

    def _consume(pool):
        fut_to_scene = {
            pool.submit(evaluate_method_framewise_on_scene, method=job_method, method_cfg=cfg, scene=scene, profile=profile_name): scene
            for job_method, cfg, scene, profile_name in jobs
        }
        for future in concurrent.futures.as_completed(fut_to_scene):
            rows.append(future.result())

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            _consume(pool)
    except PermissionError:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            _consume(pool)

    return sorted(rows, key=lambda row: str(row.get("scene_id")))


def aggregate_framewise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        raise ValueError("No successful rows to aggregate")

    mae_values = [float(row["mae_deg"]) for row in ok_rows if row.get("mae_deg") is not None]
    acc25_values = [float(row["acc_at_25"]) for row in ok_rows if row.get("acc_at_25") is not None]
    acc10_values = [float(row["acc_at_10"]) for row in ok_rows if row.get("acc_at_10") is not None]
    coverage_values = [float(row["coverage_rate"]) for row in ok_rows if row.get("coverage_rate") is not None]
    runtime_values = [float(row["runtime_ms_mean"]) for row in ok_rows if row.get("runtime_ms_mean") is not None]

    mae_mean = _safe_mean(mae_values)
    acc25_mean = _safe_mean(acc25_values)
    acc10_mean = _safe_mean(acc10_values)
    coverage_mean = _safe_mean(coverage_values)
    runtime_mean = _safe_mean(runtime_values)

    return {
        "n_scenes": len(ok_rows),
        "mae_deg_mean": mae_mean,
        "acc_at_10_mean": acc10_mean,
        "acc_at_25_mean": acc25_mean,
        "coverage_rate_mean": coverage_mean,
        "runtime_ms_mean": runtime_mean,
        "balanced_score": balanced_framewise_score(
            mae_deg=mae_mean,
            acc25=acc25_mean,
            runtime_ms=runtime_mean,
        ),
    }


def balanced_framewise_score(*, mae_deg: float | None, acc25: float | None, runtime_ms: float | None) -> float:
    mae_term = 1.0 / (1.0 + max(0.0, float(mae_deg if mae_deg is not None else 180.0)))
    acc25_term = _clamp01(acc25 if acc25 is not None else 0.0)
    runtime_term = 1.0 / (1.0 + max(0.0, float(runtime_ms if runtime_ms is not None else 1000.0)))
    return (0.55 * mae_term) + (0.35 * acc25_term) + (0.10 * runtime_term)


def export_framewise_timelines(path: Path, rows: list[dict[str, Any]]) -> None:
    export_rows: list[dict[str, Any]] = []
    for row in rows:
        for timeline_row in row.get("timeline_rows", []):
            export_rows.append(
                {
                    "scene_id": row["scene_id"],
                    "method": row["method"],
                    "window_ms": row["window_ms"],
                    "hop_ms": row["hop_ms"],
                    **timeline_row,
                }
            )
    write_csv(path, export_rows)


def write_framewise_summary(path: Path, payload: Any) -> None:
    write_json(path, payload)


def _safe_mean(values) -> float | None:
    seq = list(values)
    if not seq:
        return None
    return float(sum(seq) / len(seq))


def _safe_pstdev(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(clean) <= 1:
        return 0.0 if clean else None
    return float(statistics.pstdev(clean))


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
