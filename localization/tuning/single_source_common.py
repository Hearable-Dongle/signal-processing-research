from __future__ import annotations

import concurrent.futures
import json
import math
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from localization.benchmark.algo_runner import _build_algorithm
from localization.tuning.common import DEFAULT_PROFILE, SceneGeometry, apply_profile, load_scene_geometries, write_csv, write_json
from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation


DEFAULT_REAL_DATA_ROOT = Path("data-collection")
DEFAULT_GYM_ROOT = DEFAULT_REAL_DATA_ROOT / "gym-take-two-mar-17"
DEFAULT_KITCHENER_ROOT = DEFAULT_REAL_DATA_ROOT / "kitchener-public-library-mar-14"
DEFAULT_SIM_DATASETS = {
    "sim_near_target_far_diffuse": (
        Path("simulation/simulations/configs/testing_specific_angles_near_target_far_diffuse"),
        Path("simulation/simulations/assets/testing_specific_angles"),
    ),
    "sim_silence_gaps": (
        Path("simulation/simulations/configs/testing_specific_angles_silence_gaps"),
        Path("simulation/simulations/assets/testing_specific_angles"),
    ),
}
DEFAULT_DATASET_WEIGHTS = {
    "gym": 0.35,
    "kitchener": 0.35,
    "sim_near_target_far_diffuse": 0.15,
    "sim_silence_gaps": 0.15,
}
DEFAULT_BACKEND_CONFIGS: dict[str, dict[str, Any]] = {
    "srp_phat_localization": {
        "nfft": 512,
        "overlap": 0.5,
        "freq_range": [200, 3000],
        "grid_size": 360,
    },
    "capon_1src": {
        "nfft": 512,
        "overlap": 0.5,
        "freq_range": [200, 3000],
        "grid_size": 360,
        "diagonal_loading": 1e-3,
        "vad_enabled": True,
        "vad_frame_ms": 20,
        "vad_aggressiveness": 2,
        "vad_min_speech_ratio": 0.2,
        "capon_spectrum_ema_alpha": 0.78,
        "capon_peak_min_sharpness": 0.12,
        "capon_peak_min_margin": 0.04,
        "capon_hold_frames": 2,
    },
    "capon_mvdr_refine_1src": {
        "nfft": 512,
        "overlap": 0.5,
        "freq_range": [200, 3000],
        "grid_size": 360,
        "diagonal_loading": 1e-3,
        "vad_enabled": True,
        "vad_frame_ms": 20,
        "vad_aggressiveness": 2,
        "vad_min_speech_ratio": 0.2,
        "capon_spectrum_ema_alpha": 0.78,
        "capon_peak_min_sharpness": 0.12,
        "capon_peak_min_margin": 0.04,
        "capon_hold_frames": 2,
        "capon_refine_window_deg": 20.0,
        "capon_refine_step_deg": 2.0,
    },
}
SUPPORTED_SINGLE_SOURCE_METHODS = tuple(DEFAULT_BACKEND_CONFIGS.keys())


@dataclass(frozen=True)
class FramewiseEvalJob:
    job_id: str
    dataset_name: str
    domain_type: str
    source_kind: str
    input_path: str
    metadata_path: str | None
    mic_array_profile: str
    profile: str = DEFAULT_PROFILE
    scene_type: str | None = None
    main_angle_deg: int | None = None
    secondary_angle_deg: int | None = None
    noise_layout_type: str | None = None
    noise_angles_deg: tuple[int, ...] = ()


def _normalize_deg(value: float) -> float:
    return float(value % 360.0)


def _angular_error_deg(pred: float, truth: float) -> float:
    return float(abs((float(pred) - float(truth) + 180.0) % 360.0 - 180.0))


def window_ms_to_samples(window_ms: int, fs: int) -> int:
    return max(1, int(round(float(fs) * float(window_ms) / 1000.0)))


def hop_ms_to_samples(hop_ms: int, fs: int) -> int:
    return max(1, int(round(float(fs) * float(hop_ms) / 1000.0)))


def build_single_source_eval_config(
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


def _sibling_repo_path(path: Path) -> Path:
    return Path("..") / "signal-processing-research" / path


def resolve_input_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    sibling = _sibling_repo_path(candidate)
    if sibling.exists():
        return sibling
    return candidate


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


def _recording_dir_from_path(path: Path) -> Path | None:
    if path.is_dir() and (path / "raw").is_dir():
        return path
    if path.is_dir() and any(path.glob("*.wav")):
        return path.parent if path.name == "raw" else path
    return None


def _discover_recordings(root: Path) -> list[tuple[str, Path]]:
    if (root / "collection.json").exists():
        recordings_root = root / "recordings"
        items: list[tuple[str, Path]] = []
        for recording_dir in sorted(p for p in recordings_root.iterdir() if p.is_dir()):
            if (recording_dir / "raw").is_dir():
                items.append((recording_dir.name, recording_dir))
        if not items:
            raise FileNotFoundError(f"no recordings found in {recordings_root}")
        return items

    recording_dir = _recording_dir_from_path(root)
    if recording_dir is not None:
        return [(recording_dir.name, recording_dir)]

    raise FileNotFoundError(f"input path is not a collection root or recording dir: {root}")


def _load_multichannel_wavs(raw_dir: Path) -> tuple[np.ndarray, int]:
    wav_paths = sorted(raw_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"no WAV files found in {raw_dir}")
    channels: list[np.ndarray] = []
    sample_rate_hz: int | None = None
    min_len: int | None = None
    for wav_path in wav_paths:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        if sample_rate_hz is None:
            sample_rate_hz = int(sr)
        elif int(sr) != sample_rate_hz:
            raise ValueError(f"sample rate mismatch in {raw_dir}")
        min_len = int(arr.shape[0]) if min_len is None else min(min_len, int(arr.shape[0]))
        channels.append(arr)
    assert sample_rate_hz is not None
    assert min_len is not None
    stacked = np.stack([channel[:min_len] for channel in channels], axis=1)
    return stacked, sample_rate_hz


def _resample_multichannel_audio(mic_audio: np.ndarray, *, in_sample_rate_hz: int, out_sample_rate_hz: int) -> np.ndarray:
    if int(in_sample_rate_hz) == int(out_sample_rate_hz):
        return np.asarray(mic_audio, dtype=np.float32)
    return np.asarray(
        resample_poly(np.asarray(mic_audio, dtype=np.float32), up=int(out_sample_rate_hz), down=int(in_sample_rate_hz), axis=0),
        dtype=np.float32,
    )


def _normalize_data_collection_tracks(metadata: dict[str, Any], duration_s: float) -> tuple[dict[int, float], list[dict[str, float | int]]]:
    doa_by_speaker: dict[int, float] = {}
    events: list[dict[str, float | int]] = []
    speakers = list(metadata.get("speakers", []))
    for speaker_idx, item in enumerate(speakers, start=1):
        periods = list(item.get("speakingPeriods", []))
        for period in periods:
            start_sec = float(period.get("startSec", 0.0))
            end_sec = float(period.get("endSec", 0.0))
            if start_sec == 0.0 and end_sec == 0.0:
                end_sec = float(max(duration_s, 0.0))
            end_sec = max(start_sec, end_sec)
            angle_deg = _normalize_deg(float(period.get("directionDeg", 0.0)))
            doa_by_speaker[speaker_idx] = angle_deg
            events.append(
                {
                    "speaker_id": int(speaker_idx),
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec),
                }
            )
        if not periods and "directionDeg" in item:
            doa_by_speaker[speaker_idx] = _normalize_deg(float(item.get("directionDeg", 0.0)))
            events.append(
                {
                    "speaker_id": int(speaker_idx),
                    "start_sec": 0.0,
                    "end_sec": float(max(duration_s, 0.0)),
                }
            )
    events.sort(key=lambda row: (float(row["start_sec"]), float(row["end_sec"]), int(row["speaker_id"])))
    return doa_by_speaker, events


def _display_angle_to_backend_prediction_deg(angle_deg: float, *, mic_array_profile: str) -> float:
    angle = _normalize_deg(float(angle_deg))
    if str(mic_array_profile) == "respeaker_xvf3800_0650":
        return _normalize_deg(270.0 - angle)
    return angle


def _active_speaker_truth(
    time_s: float,
    speech_events: list[dict[str, float | int]],
    doa_by_speaker: dict[int, float],
) -> tuple[int | None, float | None]:
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
    if total <= window_samples:
        padded = np.pad(mic_audio, ((0, window_samples - total), (0, 0)))
        yield float(window_samples / 2.0) / float(fs), padded
        return
    for start in range(0, total - window_samples + 1, hop_samples):
        end = start + window_samples
        frame = mic_audio[start:end, :]
        center_s = (start + (window_samples / 2.0)) / float(fs)
        yield center_s, frame


def load_real_recording_jobs(dataset_name: str, root: str | Path) -> list[FramewiseEvalJob]:
    resolved_root = resolve_input_path(root)
    jobs: list[FramewiseEvalJob] = []
    for recording_id, recording_dir in _discover_recordings(resolved_root):
        metadata_path = recording_dir / "metadata.json"
        mic_array_profile = "respeaker_xvf3800_0650"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                mic_array_profile = str(metadata.get("micArrayProfile", mic_array_profile))
            except json.JSONDecodeError:
                pass
        jobs.append(
            FramewiseEvalJob(
                job_id=recording_id,
                dataset_name=dataset_name,
                domain_type="real",
                source_kind="recording",
                input_path=str(recording_dir),
                metadata_path=str(metadata_path) if metadata_path.exists() else None,
                mic_array_profile=mic_array_profile,
            )
        )
    return jobs


def load_simulation_jobs(dataset_name: str, scenes_root: str | Path, assets_root: str | Path) -> list[FramewiseEvalJob]:
    jobs: list[FramewiseEvalJob] = []
    scenes = load_scene_geometries(resolve_input_path(scenes_root), resolve_input_path(assets_root))
    for scene in scenes:
        jobs.append(
            FramewiseEvalJob(
                job_id=scene.scene_id,
                dataset_name=dataset_name,
                domain_type="simulation",
                source_kind="simulation_scene",
                input_path=str(scene.scene_path),
                metadata_path=str(scene.metadata_path) if scene.metadata_path else None,
                mic_array_profile=DEFAULT_PROFILE,
                profile=DEFAULT_PROFILE,
                scene_type=scene.scene_type,
                main_angle_deg=scene.main_angle_deg,
                secondary_angle_deg=scene.secondary_angle_deg,
                noise_layout_type=scene.noise_layout_type,
                noise_angles_deg=scene.noise_angles_deg,
            )
        )
    return jobs


def _load_simulation_audio_and_truth(job: FramewiseEvalJob) -> tuple[np.ndarray, np.ndarray, int, dict[int, float], list[dict[str, float | int]], dict[str, Any]]:
    sim_cfg = apply_profile(SimulationConfig.from_file(job.input_path), profile=job.profile)
    mic_audio, mic_pos_abs, _ = run_simulation(sim_cfg)
    center = np.asarray(sim_cfg.microphone_array.mic_center, dtype=float).reshape(3, 1)
    mic_pos_rel = mic_pos_abs - center
    metadata: dict[str, Any] = {}
    if job.metadata_path and Path(job.metadata_path).exists():
        metadata = json.loads(Path(job.metadata_path).read_text(encoding="utf-8"))
    mic_center_xy = np.asarray(sim_cfg.microphone_array.mic_center[:2], dtype=float)
    doa_by_speaker = _extract_speaker_doa_from_metadata(metadata, mic_center_xy)
    speech_events = _extract_speech_events(metadata)
    return mic_audio, mic_pos_rel, int(sim_cfg.audio.fs), doa_by_speaker, speech_events, metadata


def _load_recording_audio_and_truth(job: FramewiseEvalJob, target_fs: int) -> tuple[np.ndarray, np.ndarray, int, dict[int, float], list[dict[str, float | int]], dict[str, Any]]:
    recording_dir = Path(job.input_path)
    mic_audio, fs = _load_multichannel_wavs(recording_dir / "raw")
    duration_s = float(mic_audio.shape[0]) / float(max(fs, 1))
    metadata: dict[str, Any] = {}
    if job.metadata_path and Path(job.metadata_path).exists():
        metadata = json.loads(Path(job.metadata_path).read_text(encoding="utf-8"))
    doa_by_speaker_display, speech_events = _normalize_data_collection_tracks(metadata, duration_s)
    doa_by_speaker = {
        sid: _display_angle_to_backend_prediction_deg(angle_deg, mic_array_profile=job.mic_array_profile)
        for sid, angle_deg in doa_by_speaker_display.items()
    }
    mic_audio = _resample_multichannel_audio(mic_audio, in_sample_rate_hz=fs, out_sample_rate_hz=target_fs)
    mic_radius_m = 0.0325 if str(job.mic_array_profile) == "respeaker_xvf3800_0650" else 0.0457
    angles_deg = np.asarray([90.0, 0.0, 270.0, 180.0], dtype=np.float64)
    mic_pos_rel = np.stack(
        [
            mic_radius_m * np.cos(np.deg2rad(angles_deg)),
            mic_radius_m * np.sin(np.deg2rad(angles_deg)),
            np.zeros_like(angles_deg),
        ],
        axis=0,
    )
    return mic_audio, mic_pos_rel, int(target_fs), doa_by_speaker, speech_events, metadata


def _confidence_from_debug(debug: dict[str, Any], peak_scores: list[float]) -> float | None:
    if peak_scores:
        return float(peak_scores[0])
    for key in ("capon_confidence", "confidence"):
        value = debug.get(key)
        if value is not None:
            return float(value)
    return None


def _decision_mode_from_output(debug: dict[str, Any], has_prediction: bool) -> str:
    mode = str(debug.get("output_mode", "")).strip().lower()
    if mode:
        return mode
    return "accepted" if has_prediction else "abstained"


def evaluate_single_source_job(
    *,
    method: str,
    method_cfg: dict[str, Any],
    job: FramewiseEvalJob,
) -> dict[str, Any]:
    target_fs = 16000
    if job.source_kind == "simulation_scene":
        mic_audio, mic_pos_rel, fs, doa_by_speaker, speech_events, _metadata = _load_simulation_audio_and_truth(job)
    elif job.source_kind == "recording":
        mic_audio, mic_pos_rel, fs, doa_by_speaker, speech_events, _metadata = _load_recording_audio_and_truth(job, target_fs)
    else:
        raise ValueError(f"Unsupported source kind: {job.source_kind}")

    window_ms = int(method_cfg.get("window_ms", 200))
    hop_ms = int(method_cfg.get("hop_ms", max(30, int(round(window_ms / 4)))))
    algo = _build_algorithm(
        method=method,
        mic_pos_rel=mic_pos_rel,
        fs=fs,
        n_true_targets=1,
        cfg={**method_cfg, "max_sources": 1},
    )

    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    runtimes_ms: list[float] = []
    coverage_hits = 0
    eval_frames = 0
    accepted_count = 0
    hold_count = 0
    abstain_count = 0
    confidences: list[float] = []
    confidence_error_pairs: list[tuple[float, float]] = []
    finite_predictions: list[float] = []

    for time_s, frame in _window_iter(mic_audio, fs, window_ms, hop_ms):
        active_speaker_id, active_truth_deg = _active_speaker_truth(time_s, speech_events, doa_by_speaker)
        if active_speaker_id is None or active_truth_deg is None or not np.isfinite(float(active_truth_deg)):
            continue

        t0 = time.perf_counter()
        estimated_doas_rad, _hist, _history = algo.process(frame.T)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        runtimes_ms.append(runtime_ms)

        pred_deg = float("nan")
        if estimated_doas_rad:
            pred_deg = _normalize_deg(math.degrees(float(estimated_doas_rad[0])))
        has_prediction = bool(np.isfinite(pred_deg))
        debug = dict(getattr(algo, "last_debug", {}) or {})
        peak_scores = [float(v) for v in (getattr(algo, "last_peak_scores", []) or [])]
        confidence = _confidence_from_debug(debug, peak_scores)
        decision_mode = _decision_mode_from_output(debug, has_prediction)

        eval_frames += 1
        if decision_mode == "accepted":
            accepted_count += 1
        if decision_mode == "held":
            hold_count += 1
        if not has_prediction:
            abstain_count += 1
            err = math.nan
        else:
            coverage_hits += 1
            err = _angular_error_deg(pred_deg, float(active_truth_deg))
            errors.append(err)
            finite_predictions.append(pred_deg)
            if confidence is not None and np.isfinite(float(confidence)):
                confidences.append(float(confidence))
                confidence_error_pairs.append((float(confidence), float(err)))

        rows.append(
            {
                "time_s": round(float(time_s), 6),
                "active_speaker_id": int(active_speaker_id),
                "truth_doa_deg": float(active_truth_deg),
                "pred_doa_deg": None if not has_prediction else float(pred_deg),
                "confidence": None if confidence is None else float(confidence),
                "decision_mode": decision_mode,
                "error_deg": None if not np.isfinite(err) else float(err),
                "runtime_ms": float(runtime_ms),
            }
        )

    valid = np.asarray(errors, dtype=np.float64)
    runtime_ms_mean = float(np.mean(runtimes_ms)) if runtimes_ms else None
    direction_steps = [
        _angular_error_deg(finite_predictions[idx], finite_predictions[idx - 1])
        for idx in range(1, len(finite_predictions))
    ]
    confidence_summary = summarize_confidence_pairs(confidence_error_pairs)
    rtf = (runtime_ms_mean / float(hop_ms)) if runtime_ms_mean is not None and hop_ms > 0 else None
    feasible_under_rtf = bool(rtf is not None and rtf < 0.3)

    return {
        "job_id": job.job_id,
        "scene_id": job.job_id,
        "dataset_name": job.dataset_name,
        "domain_type": job.domain_type,
        "source_kind": job.source_kind,
        "scene_path": str(job.input_path),
        "scene_type": job.scene_type,
        "main_angle_deg": job.main_angle_deg,
        "secondary_angle_deg": job.secondary_angle_deg,
        "noise_layout_type": job.noise_layout_type,
        "noise_angles_deg": ",".join(str(v) for v in job.noise_angles_deg),
        "method": method,
        "status": "ok",
        "window_ms": int(window_ms),
        "hop_ms": int(hop_ms),
        "overlap": float(method_cfg.get("overlap", 0.5)),
        "freq_low_hz": int((method_cfg.get("freq_range") or [200, 3000])[0]),
        "freq_high_hz": int((method_cfg.get("freq_range") or [200, 3000])[1]),
        "eval_frames": int(eval_frames),
        "coverage_rate": float(coverage_hits / eval_frames) if eval_frames else None,
        "accepted_rate": float(accepted_count / eval_frames) if eval_frames else None,
        "hold_rate": float(hold_count / eval_frames) if eval_frames else None,
        "abstain_rate": float(abstain_count / eval_frames) if eval_frames else None,
        "mae_deg": float(np.mean(valid)) if valid.size else None,
        "acc_at_10": float(np.mean(valid <= 10.0)) if valid.size else None,
        "acc_at_20": float(np.mean(valid <= 20.0)) if valid.size else None,
        "acc_at_25": float(np.mean(valid <= 25.0)) if valid.size else None,
        "median_deg": float(np.median(valid)) if valid.size else None,
        "p90_deg": float(np.percentile(valid, 90)) if valid.size else None,
        "runtime_ms_mean": runtime_ms_mean,
        "rtf": rtf,
        "feasible_under_rtf": feasible_under_rtf,
        "direction_jump_p95_deg": float(np.percentile(np.asarray(direction_steps, dtype=np.float64), 95)) if direction_steps else 0.0,
        "confidence_mean": _safe_mean(confidences),
        **confidence_summary,
        "timeline_rows": rows,
    }


def evaluate_single_source_jobs(
    *,
    method: str,
    method_cfg: dict[str, Any],
    jobs: list[FramewiseEvalJob],
    workers: int | None = None,
) -> list[dict[str, Any]]:
    max_workers = max(1, workers or max(1, (os.cpu_count() or 1) - 1))
    if max_workers == 1:
        return [
            evaluate_single_source_job(method=method, method_cfg=dict(method_cfg), job=job)
            for job in sorted(jobs, key=lambda item: item.job_id)
        ]

    rows: list[dict[str, Any]] = []
    job_payload = [(method, dict(method_cfg), job) for job in jobs]

    def _consume(pool):
        fut_to_job = {
            pool.submit(evaluate_single_source_job, method=job_method, method_cfg=cfg, job=job): job
            for job_method, cfg, job in job_payload
        }
        for future in concurrent.futures.as_completed(fut_to_job):
            rows.append(future.result())

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            _consume(pool)
    except PermissionError:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            _consume(pool)
    return sorted(rows, key=lambda row: (str(row.get("dataset_name")), str(row.get("job_id"))))


def summarize_confidence_pairs(pairs: list[tuple[float, float]]) -> dict[str, float | None]:
    if not pairs:
        return {
            "confidence_high_mae_deg": None,
            "confidence_mid_mae_deg": None,
            "confidence_low_mae_deg": None,
        }
    buckets = {
        "high": [err for conf, err in pairs if conf >= 0.66],
        "mid": [err for conf, err in pairs if 0.33 <= conf < 0.66],
        "low": [err for conf, err in pairs if conf < 0.33],
    }
    return {
        "confidence_high_mae_deg": _safe_mean(buckets["high"]),
        "confidence_mid_mae_deg": _safe_mean(buckets["mid"]),
        "confidence_low_mae_deg": _safe_mean(buckets["low"]),
    }


def summarize_dataset_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[str(row["dataset_name"])].append(row)

    out: dict[str, dict[str, Any]] = {}
    for dataset_name, dataset_rows in groups.items():
        mae_values = [float(row["mae_deg"]) for row in dataset_rows if row.get("mae_deg") is not None]
        acc10_values = [float(row["acc_at_10"]) for row in dataset_rows if row.get("acc_at_10") is not None]
        acc25_values = [float(row["acc_at_25"]) for row in dataset_rows if row.get("acc_at_25") is not None]
        coverage_values = [float(row["coverage_rate"]) for row in dataset_rows if row.get("coverage_rate") is not None]
        accepted_values = [float(row["accepted_rate"]) for row in dataset_rows if row.get("accepted_rate") is not None]
        hold_values = [float(row["hold_rate"]) for row in dataset_rows if row.get("hold_rate") is not None]
        abstain_values = [float(row["abstain_rate"]) for row in dataset_rows if row.get("abstain_rate") is not None]
        runtime_values = [float(row["runtime_ms_mean"]) for row in dataset_rows if row.get("runtime_ms_mean") is not None]
        rtf_values = [float(row["rtf"]) for row in dataset_rows if row.get("rtf") is not None]
        jump_values = [float(row["direction_jump_p95_deg"]) for row in dataset_rows if row.get("direction_jump_p95_deg") is not None]
        confidence_high = [float(row["confidence_high_mae_deg"]) for row in dataset_rows if row.get("confidence_high_mae_deg") is not None]
        confidence_low = [float(row["confidence_low_mae_deg"]) for row in dataset_rows if row.get("confidence_low_mae_deg") is not None]
        domain_type = str(dataset_rows[0].get("domain_type", "simulation"))
        score = dataset_quality_score(
            mae_deg=_safe_mean(mae_values),
            acc10=_safe_mean(acc10_values),
            acc25=_safe_mean(acc25_values),
            coverage_rate=_safe_mean(coverage_values),
        )
        feasible_under_rtf = bool(rtf_values) and all(float(v) < 0.3 for v in rtf_values)
        out[dataset_name] = {
            "dataset_name": dataset_name,
            "domain_type": domain_type,
            "n_jobs": len(dataset_rows),
            "mae_deg_mean": _safe_mean(mae_values),
            "acc_at_10_mean": _safe_mean(acc10_values),
            "acc_at_25_mean": _safe_mean(acc25_values),
            "coverage_rate_mean": _safe_mean(coverage_values),
            "accepted_rate_mean": _safe_mean(accepted_values),
            "hold_rate_mean": _safe_mean(hold_values),
            "abstain_rate_mean": _safe_mean(abstain_values),
            "runtime_ms_mean": _safe_mean(runtime_values),
            "rtf_mean": _safe_mean(rtf_values),
            "direction_jump_p95_deg_mean": _safe_mean(jump_values),
            "confidence_high_mae_deg_mean": _safe_mean(confidence_high),
            "confidence_low_mae_deg_mean": _safe_mean(confidence_low),
            "dataset_score": score,
            "feasible_under_rtf": feasible_under_rtf,
        }
    return out


def dataset_quality_score(
    *,
    mae_deg: float | None,
    acc10: float | None,
    acc25: float | None,
    coverage_rate: float | None,
) -> float:
    mae_term = 1.0 / (1.0 + max(0.0, float(mae_deg if mae_deg is not None else 180.0)))
    acc10_term = _clamp01(acc10 if acc10 is not None else 0.0)
    acc25_term = _clamp01(acc25 if acc25 is not None else 0.0)
    coverage_term = _clamp01(coverage_rate if coverage_rate is not None else 0.0)
    return (0.45 * acc25_term) + (0.25 * acc10_term) + (0.20 * mae_term) + (0.10 * coverage_term)


def aggregate_mixed_dataset_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    by_dataset = summarize_dataset_rows(rows)
    if not by_dataset:
        raise ValueError("No successful rows to aggregate")
    weights = dataset_weights or DEFAULT_DATASET_WEIGHTS
    weighted_score = 0.0
    weighted_total = 0.0
    feasible = True
    for dataset_name, summary in by_dataset.items():
        weight = float(weights.get(dataset_name, 0.0))
        weighted_score += weight * float(summary["dataset_score"])
        weighted_total += weight
        feasible = feasible and bool(summary["feasible_under_rtf"])
    deployable_score = weighted_score / weighted_total if weighted_total > 0 else 0.0
    if not feasible:
        deployable_score -= 10.0
    rtf_values = [float(row["rtf"]) for row in rows if row.get("rtf") is not None]
    real_mae_values = [
        float(summary["mae_deg_mean"])
        for summary in by_dataset.values()
        if summary.get("domain_type") == "real" and summary.get("mae_deg_mean") is not None
    ]
    failure_reasons = infer_failure_reasons(rows, by_dataset)
    return {
        "n_jobs": len([row for row in rows if row.get("status") == "ok"]),
        "dataset_summaries": by_dataset,
        "weighted_dataset_score": deployable_score,
        "feasible_under_rtf": feasible,
        "rtf_mean": _safe_mean(rtf_values),
        "real_mae_deg_mean": _safe_mean(real_mae_values),
        "failure_reasons": failure_reasons,
    }


def infer_failure_reasons(rows: list[dict[str, Any]], by_dataset: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    rtf_values = [float(row["rtf"]) for row in rows if row.get("rtf") is not None]
    if rtf_values and max(rtf_values) >= 0.3:
        reasons.append("latency_budget_violation")
    abstain_values = [float(row["abstain_rate"]) for row in rows if row.get("abstain_rate") is not None]
    if abstain_values and _safe_mean(abstain_values) is not None and float(_safe_mean(abstain_values)) > 0.4:
        reasons.append("over_abstention")
    hold_values = [float(row["hold_rate"]) for row in rows if row.get("hold_rate") is not None]
    if hold_values and _safe_mean(hold_values) is not None and float(_safe_mean(hold_values)) > 0.35:
        reasons.append("excessive_hold")
    jump_values = [float(row["direction_jump_p95_deg"]) for row in rows if row.get("direction_jump_p95_deg") is not None]
    if jump_values and max(jump_values) > 45.0:
        reasons.append("unstable_peak_hopping")
    real_scores = [float(summary["dataset_score"]) for summary in by_dataset.values() if summary.get("domain_type") == "real"]
    sim_scores = [float(summary["dataset_score"]) for summary in by_dataset.values() if summary.get("domain_type") == "simulation"]
    if real_scores and sim_scores and (float(np.mean(sim_scores)) - float(np.mean(real_scores))) > 0.20:
        reasons.append("simulation_only_overfit")
    if not reasons:
        reasons.append("none")
    return reasons


def export_framewise_timelines(path: Path, rows: list[dict[str, Any]]) -> None:
    export_rows: list[dict[str, Any]] = []
    for row in rows:
        for timeline_row in row.get("timeline_rows", []):
            export_rows.append(
                {
                    "job_id": row["job_id"],
                    "dataset_name": row["dataset_name"],
                    "method": row["method"],
                    "window_ms": row["window_ms"],
                    "hop_ms": row["hop_ms"],
                    **timeline_row,
                }
            )
    write_csv(path, export_rows)


def export_job_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = [{k: v for k, v in row.items() if k != "timeline_rows"} for row in rows]
    write_csv(path, payload)


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
