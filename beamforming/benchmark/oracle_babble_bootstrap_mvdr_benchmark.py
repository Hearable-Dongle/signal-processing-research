from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from beamforming.benchmark.metrics import compute_metric_bundle
from beamforming.benchmark.oracle_xvf3800_enhancement_sweep import (
    DEFAULT_PROFILE,
    OracleFrameState,
    _build_clean_reference,
    _build_oracle_frame_states,
    _load_scene_metadata,
    _oracle_srp_override_provider,
    _oracle_target_activity_override_provider,
    _speaker_source_index_map,
    _stage_scene,
)
from mic_array_forwarder.models import SessionStartRequest
from realtime_pipeline.session_runtime import run_offline_session_pipeline
from simulation.create_testing_specific_angles_babble_bootstrap import (
    DEFAULT_ASSET_ROOT,
    DEFAULT_BABBLE_COUNT,
    DEFAULT_BABBLE_GAIN_MAX,
    DEFAULT_BABBLE_GAIN_MIN,
    DEFAULT_BOOTSTRAP_SEC,
    DEFAULT_CONFIG_ROOT,
    DEFAULT_DURATION_SEC,
    DEFAULT_WHAM_GAIN,
    generate_testing_specific_angles_babble_bootstrap_dataset,
)
from simulation.mic_array_profiles import SUPPORTED_MIC_ARRAY_PROFILES
from simulation.simulation_config import SimulationConfig


DEFAULT_SCENES_ROOT = Path("simulation/simulations/configs/testing_specific_angles_babble_bootstrap")
DEFAULT_ASSETS_ROOT = Path("simulation/simulations/assets/testing_specific_angles_babble_bootstrap")
DEFAULT_OUT_ROOT = Path("beamforming/benchmark/oracle_babble_bootstrap_mvdr")
DEFAULT_METHODS = [
    "delay_sum",
    "mvdr_fd_bootstrap_oracle_activity",
    "mvdr_fd_bootstrap_estimated_activity",
]
FAST_FRAME_MS = 40
DEFAULT_FD_ANALYSIS_WINDOW_MS = 80.0
DEFAULT_FD_COV_EMA_ALPHA = 0.08
DEFAULT_FD_DIAG_LOAD = 1e-3
DEFAULT_ACTIVE_UPDATE_SCALE = 0.20
DEFAULT_INACTIVE_UPDATE_SCALE = 1.0


@dataclass(frozen=True)
class MethodSpec:
    method_key: str
    beamforming_mode: str
    target_activity_mode: str | None


@dataclass(frozen=True)
class JobResult:
    row: dict[str, object]
    run_summary: dict[str, object]


METHOD_SPECS: dict[str, MethodSpec] = {
    "delay_sum": MethodSpec("delay_sum", "delay_sum", None),
    "mvdr_fd_bootstrap_oracle_activity": MethodSpec("mvdr_fd_bootstrap_oracle_activity", "mvdr_fd", "oracle_target_activity"),
    "mvdr_fd_bootstrap_estimated_activity": MethodSpec("mvdr_fd_bootstrap_estimated_activity", "mvdr_fd", "estimated_target_activity"),
    "mvdr_fd_bootstrap_estimated_activity_silero": MethodSpec("mvdr_fd_bootstrap_estimated_activity_silero", "mvdr_fd", "estimated_target_activity"),
}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_dict(bundle) -> dict[str, float]:
    return {
        "snr_db_raw": float(bundle.snr_db_raw),
        "snr_db_processed": float(bundle.snr_db_processed),
        "delta_snr_db": float(bundle.delta_snr_db),
        "sii_raw": float(bundle.sii_raw),
        "sii_processed": float(bundle.sii_processed),
        "delta_sii": float(bundle.delta_sii),
        "si_sdr_db_raw": float(bundle.si_sdr_db_raw),
        "si_sdr_db_processed": float(bundle.si_sdr_db_processed),
        "delta_si_sdr_db": float(bundle.delta_si_sdr_db),
    }


def _masked_metric_dict(
    *,
    clean_ref: np.ndarray,
    raw_audio: np.ndarray,
    processed_audio: np.ndarray,
    sample_rate: int,
    mask: np.ndarray,
) -> dict[str, float]:
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    if mask_arr.size == 0 or int(np.sum(mask_arr)) < max(256, sample_rate // 8):
        return {key: float("nan") for key in _metric_dict(compute_metric_bundle(clean_ref=np.zeros(512), raw_audio=np.zeros(512), processed_audio=np.zeros(512), sample_rate=sample_rate)).keys()}
    bundle = compute_metric_bundle(
        clean_ref=np.asarray(clean_ref, dtype=np.float64)[mask_arr],
        raw_audio=np.asarray(raw_audio, dtype=np.float64)[mask_arr],
        processed_audio=np.asarray(processed_audio, dtype=np.float64)[mask_arr],
        sample_rate=int(sample_rate),
    )
    return _metric_dict(bundle)


def _noise_reduction_metrics(raw_audio: np.ndarray, processed_audio: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    if mask_arr.size == 0 or not np.any(mask_arr):
        return {
            f"{prefix}_rms_raw_dbfs": float("nan"),
            f"{prefix}_rms_processed_dbfs": float("nan"),
            f"{prefix}_noise_reduction_db": float("nan"),
        }
    raw = np.asarray(raw_audio, dtype=np.float64)[mask_arr]
    proc = np.asarray(processed_audio, dtype=np.float64)[mask_arr]
    raw_rms = float(np.sqrt(np.mean(raw**2) + 1e-12))
    proc_rms = float(np.sqrt(np.mean(proc**2) + 1e-12))
    return {
        f"{prefix}_rms_raw_dbfs": float(20.0 * np.log10(max(raw_rms, 1e-12))),
        f"{prefix}_rms_processed_dbfs": float(20.0 * np.log10(max(proc_rms, 1e-12))),
        f"{prefix}_noise_reduction_db": float(20.0 * np.log10(max(raw_rms, 1e-12) / max(proc_rms, 1e-12))),
    }


def _summarize_trace(summary: dict[str, object]) -> dict[str, float]:
    srp_trace = list(summary.get("srp_trace", []))
    alphas_active: list[float] = []
    alphas_inactive: list[float] = []
    scores: list[float] = []
    for row in srp_trace:
        debug = dict(row.get("debug", {}))
        target_debug = dict(debug.get("target_activity", {}))
        alpha = target_debug.get("covariance_alpha")
        active = target_debug.get("active")
        score = target_debug.get("score")
        if score is not None:
            scores.append(float(score))
        if alpha is None or active is None:
            continue
        if bool(active):
            alphas_active.append(float(alpha))
        else:
            alphas_inactive.append(float(alpha))
    return {
        "trace_target_active_frames": float(len(alphas_active)),
        "trace_target_inactive_frames": float(len(alphas_inactive)),
        "trace_cov_alpha_active_mean": float(np.mean(alphas_active)) if alphas_active else float("nan"),
        "trace_cov_alpha_inactive_mean": float(np.mean(alphas_inactive)) if alphas_inactive else float("nan"),
        "trace_target_activity_score_mean": float(np.mean(scores)) if scores else float("nan"),
    }


def _trace_activity_errors(summary: dict[str, object], frame_states: list[OracleFrameState]) -> dict[str, float]:
    srp_trace = list(summary.get("srp_trace", []))
    false_active = 0
    false_inactive = 0
    compared = 0
    for idx, row in enumerate(srp_trace):
        if idx >= len(frame_states):
            break
        debug = dict(row.get("debug", {}))
        target_debug = dict(debug.get("target_activity", {}))
        detected = target_debug.get("active")
        if detected is None:
            continue
        compared += 1
        oracle_active = bool(frame_states[idx].target_active)
        if bool(detected) and not oracle_active:
            false_active += 1
        elif oracle_active and not bool(detected):
            false_inactive += 1
    denom = max(compared, 1)
    return {
        "trace_activity_compared_frames": float(compared),
        "trace_false_active_rate": float(false_active / denom),
        "trace_false_inactive_rate": float(false_inactive / denom),
    }


def _mean_or_nan(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _aggregate(rows: list[dict[str, object]], group_fields: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in group_fields)].append(row)
    out_rows: list[dict[str, object]] = []
    metric_fields = [
        "delta_snr_db",
        "delta_sii",
        "speech_delta_snr_db",
        "speech_delta_sii",
        "bootstrap_noise_reduction_db",
        "background_only_noise_reduction_db",
        "trace_false_active_rate",
        "trace_false_inactive_rate",
        "rtf",
        "fast_rtf",
        "slow_rtf",
        "trace_cov_alpha_active_mean",
        "trace_cov_alpha_inactive_mean",
    ]
    for key, items in sorted(grouped.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        out = {field: value for field, value in zip(group_fields, key)}
        out["n_runs"] = len(items)
        for metric in metric_fields:
            out[f"{metric}_mean"] = _mean_or_nan([float(item.get(metric, float("nan"))) for item in items])
        out_rows.append(out)
    return out_rows


def _build_session_request(
    *,
    method_spec: MethodSpec,
    sample_rate: int,
    channel_count: int,
    profile: str,
    params: dict[str, object],
) -> SessionStartRequest:
    return SessionStartRequest(
        input_source="simulation",
        channel_count=int(channel_count),
        sample_rate_hz=int(sample_rate),
        monitor_source="processed",
        mic_array_profile=str(profile),
        fast_path={
            "localization_hop_ms": int(FAST_FRAME_MS),
            "localization_window_ms": 160,
            "overlap": 0.2,
            "freq_low_hz": 200,
            "freq_high_hz": 3000,
            "localization_pair_selection_mode": "all",
            "localization_vad_enabled": True,
            "localization_backend": "srp_phat_localization",
            "beamforming_mode": str(method_spec.beamforming_mode),
            "fd_analysis_window_ms": float(params["fd_analysis_window_ms"]),
            "fd_cov_ema_alpha": float(params["fd_cov_ema_alpha"]),
            "fd_diag_load": float(params["fd_diag_load"]),
            "target_activity_rnn_update_mode": method_spec.target_activity_mode,
            "target_activity_low_threshold": float(params["target_activity_low_threshold"]),
            "target_activity_high_threshold": float(params["target_activity_high_threshold"]),
            "target_activity_enter_frames": int(params["target_activity_enter_frames"]),
            "target_activity_exit_frames": int(params["target_activity_exit_frames"]),
            "fd_cov_update_scale_target_active": float(params["fd_cov_update_scale_target_active"]),
            "fd_cov_update_scale_target_inactive": float(params["fd_cov_update_scale_target_inactive"]),
            "target_activity_detector_mode": str(params["target_activity_detector_mode"]),
            "target_activity_detector_backend": str(params["target_activity_detector_backend"]),
            "target_activity_blocker_offset_deg": float(params["target_activity_blocker_offset_deg"]),
            "target_activity_bootstrap_only_calibration": bool(params["target_activity_bootstrap_only_calibration"]),
            "target_activity_ratio_floor_db": float(params["target_activity_ratio_floor_db"]),
            "target_activity_ratio_active_db": float(params["target_activity_ratio_active_db"]),
            "target_activity_target_rms_floor_scale": float(params["target_activity_target_rms_floor_scale"]),
            "target_activity_blocker_rms_floor_scale": float(params["target_activity_blocker_rms_floor_scale"]),
            "target_activity_speech_weight": float(params["target_activity_speech_weight"]),
            "target_activity_ratio_weight": float(params["target_activity_ratio_weight"]),
            "target_activity_blocker_weight": float(params["target_activity_blocker_weight"]),
            "target_activity_vad_mode": int(params["target_activity_vad_mode"]),
            "target_activity_vad_hangover_frames": int(params["target_activity_vad_hangover_frames"]),
            "target_activity_noise_floor_rise_alpha": float(params["target_activity_noise_floor_rise_alpha"]),
            "target_activity_noise_floor_fall_alpha": float(params["target_activity_noise_floor_fall_alpha"]),
            "target_activity_noise_floor_margin_scale": float(params["target_activity_noise_floor_margin_scale"]),
            "target_activity_rms_scale": float(params["target_activity_rms_scale"]),
            "target_activity_score_exponent": float(params["target_activity_score_exponent"]),
            "postfilter_enabled": False,
            "output_normalization_enabled": False,
            "output_allow_amplification": False,
            "own_voice_suppression_mode": "off",
        },
        slow_path={
            "enabled": False,
            "tracking_mode": "doa_centroid_v1",
            "speaker_match_window_deg": 25.0,
            "centroid_association_mode": "hard_window",
            "centroid_association_sigma_deg": 10.0,
            "centroid_association_min_score": 0.15,
            "slow_chunk_ms": 200,
        },
        max_speakers_hint=max(1, int(channel_count)),
        separation_mode="mock",
        processing_mode="beamform_from_ground_truth",
    )


def _build_babble_bootstrap_frame_states(
    *,
    active_speaker_ids: np.ndarray,
    active_doa_deg: np.ndarray,
    sample_rate: int,
    fast_frame_ms: int,
    bootstrap_window_sec: tuple[float, float],
    bootstrap_reference_doa_deg: float,
) -> list[OracleFrameState]:
    states = _build_oracle_frame_states(
        active_speaker_ids=active_speaker_ids,
        active_doa_deg=active_doa_deg,
        sample_rate=sample_rate,
        fast_frame_ms=fast_frame_ms,
    )
    start_sec = float(bootstrap_window_sec[0])
    end_sec = float(bootstrap_window_sec[1])
    for idx, state in enumerate(states):
        t_sec = float(state.timestamp_ms) / 1000.0
        if start_sec <= t_sec < end_sec:
            states[idx] = OracleFrameState(
                frame_index=state.frame_index,
                timestamp_ms=state.timestamp_ms,
                target_speaker_id=None,
                target_doa_deg=float(bootstrap_reference_doa_deg),
                target_activity_score=0.0,
                target_active=False,
                null_user_speaker_id=None,
                null_user_doa_deg=None,
                force_suppression_active=False,
                null_candidate=False,
                null_fallback=False,
                peaks_deg=(float(bootstrap_reference_doa_deg),),
                peak_scores=(1.0,),
            )
    return states


def _bootstrap_mask(n_samples: int, sample_rate: int, window_sec: tuple[float, float]) -> np.ndarray:
    mask = np.zeros(int(n_samples), dtype=bool)
    start = max(0, int(round(float(window_sec[0]) * sample_rate)))
    end = min(int(n_samples), int(round(float(window_sec[1]) * sample_rate)))
    if end > start:
        mask[start:end] = True
    return mask


def _run_job(
    *,
    scene_path: str,
    assets_root: str,
    out_root: str,
    profile: str,
    method: str,
    params: dict[str, object],
) -> JobResult:
    method_spec = METHOD_SPECS[str(method)]
    scene_cfg_path = Path(scene_path)
    scene_id = scene_cfg_path.stem
    root = Path(out_root)
    staged_scene, staged_metadata = _stage_scene(
        scene_cfg_path,
        root / "_staged_scenes",
        profile,
        Path(assets_root),
        1.0,
        stage_key=method,
    )
    metadata_payload = json.loads(staged_metadata.read_text(encoding="utf-8")) if staged_metadata.exists() else {}
    scene_meta = _load_scene_metadata(scene_id, staged_metadata)
    sim_cfg = SimulationConfig.from_file(staged_scene)

    from simulation.simulator import run_simulation

    mic_audio, mic_pos, source_signals = run_simulation(sim_cfg)
    sample_rate = int(sim_cfg.audio.fs)
    n_samples = int(mic_audio.shape[0])
    speech_rows = list(metadata_payload.get("assets", {}).get("speech", []))
    active_speaker_ids = np.full(n_samples, -1, dtype=np.int32)
    active_doa_deg = np.full(n_samples, np.nan, dtype=np.float64)
    for row in speech_rows:
        window = row.get("active_window_sec")
        if not isinstance(window, list) or len(window) < 2:
            continue
        start = max(0, int(round(float(window[0]) * sample_rate)))
        end = min(n_samples, int(round(float(window[1]) * sample_rate)))
        if end <= start:
            continue
        speaker_id = int(row.get("speaker_id", -1))
        doa_deg = float(row.get("angle_deg", math.nan))
        active_speaker_ids[start:end] = speaker_id
        active_doa_deg[start:end] = doa_deg

    speech_mask = np.asarray(active_speaker_ids >= 0, dtype=bool)
    bootstrap_window = tuple(float(v) for v in metadata_payload.get("bootstrap_noise_only_window_sec", [0.0, 0.0])[:2])
    bootstrap_reference_doa_deg = float(metadata_payload.get("bootstrap_reference_doa_deg", scene_meta["main_angle_deg"] or 0.0))
    bootstrap_mask = _bootstrap_mask(n_samples, sample_rate, bootstrap_window)
    background_only_mask = np.logical_not(speech_mask)
    speaker_to_source_idx = _speaker_source_index_map(sim_cfg, metadata_payload)
    clean_ref = _build_clean_reference(source_signals, speaker_to_source_idx, active_speaker_ids)
    raw_mix = np.mean(np.asarray(mic_audio, dtype=np.float64), axis=1).astype(np.float32, copy=False)

    frame_states = _build_babble_bootstrap_frame_states(
        active_speaker_ids=active_speaker_ids,
        active_doa_deg=active_doa_deg,
        sample_rate=sample_rate,
        fast_frame_ms=FAST_FRAME_MS,
        bootstrap_window_sec=bootstrap_window,
        bootstrap_reference_doa_deg=bootstrap_reference_doa_deg,
    )
    req = _build_session_request(
        method_spec=method_spec,
        sample_rate=sample_rate,
        channel_count=int(mic_audio.shape[1]),
        profile=profile,
        params=params,
    )

    run_dir = root / "runs" / scene_id / method_spec.method_key
    run_dir.mkdir(parents=True, exist_ok=True)
    sf.write(run_dir / "raw_mix_mean.wav", raw_mix, sample_rate)
    sf.write(run_dir / "clean_active_target.wav", clean_ref.astype(np.float32), sample_rate)

    t0 = perf_counter()
    runtime_summary = run_offline_session_pipeline(
        req=req,
        mic_audio=np.asarray(mic_audio, dtype=np.float32),
        mic_geometry_xyz=np.asarray(mic_pos, dtype=np.float64),
        out_dir=run_dir,
        capture_trace=True,
        srp_override_provider=_oracle_srp_override_provider(frame_states),
        target_activity_override_provider=(
            _oracle_target_activity_override_provider(frame_states)
            if method_spec.target_activity_mode == "oracle_target_activity"
            else None
        ),
    )
    elapsed_s = max(perf_counter() - t0, 1e-9)
    processed, proc_sr = sf.read(run_dir / "enhanced_fast_path.wav", dtype="float32", always_2d=False)
    processed_audio = np.asarray(processed, dtype=np.float32).reshape(-1)
    if int(proc_sr) != int(sample_rate):
        raise ValueError(f"Unexpected sample rate {proc_sr}, expected {sample_rate}")
    sf.write(run_dir / "enhanced.wav", processed_audio, sample_rate)

    overall = compute_metric_bundle(
        clean_ref=np.asarray(clean_ref, dtype=np.float64),
        raw_audio=np.asarray(raw_mix, dtype=np.float64),
        processed_audio=np.asarray(processed_audio, dtype=np.float64),
        sample_rate=sample_rate,
    )
    speech_metrics = _masked_metric_dict(
        clean_ref=clean_ref,
        raw_audio=raw_mix,
        processed_audio=processed_audio,
        sample_rate=sample_rate,
        mask=speech_mask,
    )
    bootstrap_metrics = _noise_reduction_metrics(raw_mix, processed_audio, bootstrap_mask, "bootstrap")
    background_only_metrics = _noise_reduction_metrics(raw_mix, processed_audio, background_only_mask, "background_only")
    trace_metrics = _summarize_trace(runtime_summary)
    trace_error_metrics = _trace_activity_errors(runtime_summary, frame_states)
    duration_s = float(len(processed_audio) / max(sample_rate, 1))

    row: dict[str, object] = {
        "scene": scene_id,
        "method": method_spec.method_key,
        "beamforming_mode_runtime": str(runtime_summary.get("beamforming_mode", "")),
        "target_activity_mode": "" if method_spec.target_activity_mode is None else str(method_spec.target_activity_mode),
        "target_activity_detector_backend": str(params["target_activity_detector_backend"]),
        "main_angle_deg": scene_meta["main_angle_deg"],
        "secondary_angle_deg": scene_meta["secondary_angle_deg"],
        "scene_layout_family": scene_meta["scene_layout_family"],
        "bootstrap_sec": float(bootstrap_window[1] - bootstrap_window[0]),
        "background_babble_count": int(metadata_payload.get("background_babble_count", 0)),
        "background_wham_gain": float(metadata_payload.get("background_wham_gain", 0.0)),
        "speech_frame_fraction": float(np.mean(speech_mask.astype(np.float64))),
        "bootstrap_frame_fraction": float(np.mean(bootstrap_mask.astype(np.float64))),
        "background_only_frame_fraction": float(np.mean(background_only_mask.astype(np.float64))),
        "rtf": float(elapsed_s / max(duration_s, 1e-9)),
        "fast_rtf": float(runtime_summary.get("fast_rtf", float("nan"))),
        "slow_rtf": float(runtime_summary.get("slow_rtf", float("nan"))),
        **_metric_dict(overall),
        "speech_delta_snr_db": float(speech_metrics["delta_snr_db"]),
        "speech_delta_sii": float(speech_metrics["delta_sii"]),
        "speech_delta_si_sdr_db": float(speech_metrics["delta_si_sdr_db"]),
        **bootstrap_metrics,
        **background_only_metrics,
        **trace_metrics,
        **trace_error_metrics,
        "run_dir": str(run_dir.resolve()),
    }

    run_summary: dict[str, object] = {
        "scene": scene_id,
        "method": method_spec.method_key,
        "profile": profile,
        "staged_scene": str(staged_scene.resolve()),
        "scenario_metadata": str(staged_metadata.resolve()) if staged_metadata.exists() else "",
        "run_dir": str(run_dir.resolve()),
        "runtime_summary": runtime_summary,
        "benchmark_params": dict(params),
        "metrics": {
            "overall": _metric_dict(overall),
            "speech_only": speech_metrics,
            "bootstrap_only": bootstrap_metrics,
            "background_only": background_only_metrics,
            "trace": trace_metrics,
            "trace_errors": trace_error_metrics,
        },
    }
    _write_json(run_dir / "summary.json", run_summary)
    return JobResult(row=row, run_summary=run_summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GT-DOA MVDR on babble-bootstrap scenes with aggressive Rnn tuning.")
    parser.add_argument("--scenes-root", default=str(DEFAULT_SCENES_ROOT))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=list(SUPPORTED_MIC_ARRAY_PROFILES))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--regenerate-scenes", action="store_true")
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument("--bootstrap-noise-only-sec", type=float, default=DEFAULT_BOOTSTRAP_SEC)
    parser.add_argument("--background-babble-count", type=int, default=DEFAULT_BABBLE_COUNT)
    parser.add_argument("--background-babble-gain-min", type=float, default=DEFAULT_BABBLE_GAIN_MIN)
    parser.add_argument("--background-babble-gain-max", type=float, default=DEFAULT_BABBLE_GAIN_MAX)
    parser.add_argument("--background-wham-gain", type=float, default=DEFAULT_WHAM_GAIN)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--fd-analysis-window-ms", type=float, default=DEFAULT_FD_ANALYSIS_WINDOW_MS)
    parser.add_argument("--fd-cov-ema-alpha", type=float, default=DEFAULT_FD_COV_EMA_ALPHA)
    parser.add_argument("--fd-diag-load", type=float, default=DEFAULT_FD_DIAG_LOAD)
    parser.add_argument("--target-activity-low-threshold", type=float, default=0.22)
    parser.add_argument("--target-activity-high-threshold", type=float, default=0.40)
    parser.add_argument("--target-activity-enter-frames", type=int, default=2)
    parser.add_argument("--target-activity-exit-frames", type=int, default=3)
    parser.add_argument("--fd-cov-update-scale-target-active", type=float, default=DEFAULT_ACTIVE_UPDATE_SCALE)
    parser.add_argument("--fd-cov-update-scale-target-inactive", type=float, default=DEFAULT_INACTIVE_UPDATE_SCALE)
    parser.add_argument("--target-activity-detector-mode", default="target_blocker_calibrated")
    parser.add_argument("--target-activity-detector-backend", default="webrtc_fused", choices=["webrtc_fused", "silero_fused"])
    parser.add_argument("--target-activity-blocker-offset-deg", type=float, default=90.0)
    parser.add_argument("--target-activity-bootstrap-only-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-activity-ratio-floor-db", type=float, default=0.0)
    parser.add_argument("--target-activity-ratio-active-db", type=float, default=3.0)
    parser.add_argument("--target-activity-target-rms-floor-scale", type=float, default=1.6)
    parser.add_argument("--target-activity-blocker-rms-floor-scale", type=float, default=1.1)
    parser.add_argument("--target-activity-speech-weight", type=float, default=0.55)
    parser.add_argument("--target-activity-ratio-weight", type=float, default=0.30)
    parser.add_argument("--target-activity-blocker-weight", type=float, default=0.15)
    parser.add_argument("--target-activity-vad-mode", type=int, default=2)
    parser.add_argument("--target-activity-vad-hangover-frames", type=int, default=2)
    parser.add_argument("--target-activity-noise-floor-rise-alpha", type=float, default=0.01)
    parser.add_argument("--target-activity-noise-floor-fall-alpha", type=float, default=0.10)
    parser.add_argument("--target-activity-noise-floor-margin-scale", type=float, default=1.25)
    parser.add_argument("--target-activity-rms-scale", type=float, default=4.0)
    parser.add_argument("--target-activity-score-exponent", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenes_root = Path(args.scenes_root)
    assets_root = Path(args.assets_root)
    if args.regenerate_scenes or not list(scenes_root.glob("*.json")):
        generate_testing_specific_angles_babble_bootstrap_dataset(
            config_root=scenes_root,
            asset_root=assets_root,
            duration_sec=float(args.duration_sec),
            bootstrap_noise_only_sec=float(args.bootstrap_noise_only_sec),
            background_babble_count=int(args.background_babble_count),
            background_babble_gain_min=float(args.background_babble_gain_min),
            background_babble_gain_max=float(args.background_babble_gain_max),
            background_wham_gain=float(args.background_wham_gain),
            manifest_path=args.manifest_path,
        )

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    scenes = sorted(scenes_root.glob("*.json"))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    if not scenes:
        raise FileNotFoundError(f"No scenes found under {args.scenes_root}")

    params = {
        "fd_analysis_window_ms": float(args.fd_analysis_window_ms),
        "fd_cov_ema_alpha": float(args.fd_cov_ema_alpha),
        "fd_diag_load": float(args.fd_diag_load),
        "target_activity_low_threshold": float(args.target_activity_low_threshold),
        "target_activity_high_threshold": float(args.target_activity_high_threshold),
        "target_activity_enter_frames": int(args.target_activity_enter_frames),
        "target_activity_exit_frames": int(args.target_activity_exit_frames),
        "fd_cov_update_scale_target_active": float(args.fd_cov_update_scale_target_active),
        "fd_cov_update_scale_target_inactive": float(args.fd_cov_update_scale_target_inactive),
        "target_activity_detector_mode": str(args.target_activity_detector_mode),
        "target_activity_detector_backend": str(args.target_activity_detector_backend),
        "target_activity_blocker_offset_deg": float(args.target_activity_blocker_offset_deg),
        "target_activity_bootstrap_only_calibration": bool(args.target_activity_bootstrap_only_calibration),
        "target_activity_ratio_floor_db": float(args.target_activity_ratio_floor_db),
        "target_activity_ratio_active_db": float(args.target_activity_ratio_active_db),
        "target_activity_target_rms_floor_scale": float(args.target_activity_target_rms_floor_scale),
        "target_activity_blocker_rms_floor_scale": float(args.target_activity_blocker_rms_floor_scale),
        "target_activity_speech_weight": float(args.target_activity_speech_weight),
        "target_activity_ratio_weight": float(args.target_activity_ratio_weight),
        "target_activity_blocker_weight": float(args.target_activity_blocker_weight),
        "target_activity_vad_mode": int(args.target_activity_vad_mode),
        "target_activity_vad_hangover_frames": int(args.target_activity_vad_hangover_frames),
        "target_activity_noise_floor_rise_alpha": float(args.target_activity_noise_floor_rise_alpha),
        "target_activity_noise_floor_fall_alpha": float(args.target_activity_noise_floor_fall_alpha),
        "target_activity_noise_floor_margin_scale": float(args.target_activity_noise_floor_margin_scale),
        "target_activity_rms_scale": float(args.target_activity_rms_scale),
        "target_activity_score_exponent": float(args.target_activity_score_exponent),
        "bootstrap_noise_only_sec": float(args.bootstrap_noise_only_sec),
        "background_babble_count": int(args.background_babble_count),
        "background_babble_gain_min": float(args.background_babble_gain_min),
        "background_babble_gain_max": float(args.background_babble_gain_max),
        "background_wham_gain": float(args.background_wham_gain),
    }
    jobs = [
        {
            "scene_path": str(scene.resolve()),
            "assets_root": str(assets_root.resolve()),
            "out_root": str(out_root.resolve()),
            "profile": str(args.profile),
            "method": str(method),
            "params": {
                **params,
                "target_activity_detector_backend": (
                    "silero_fused"
                    if str(method) == "mvdr_fd_bootstrap_estimated_activity_silero"
                    else str(params["target_activity_detector_backend"])
                ),
            },
        }
        for scene in scenes
        for method in list(args.methods)
    ]

    results: list[JobResult] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_map = {pool.submit(_run_job, **job): (Path(job["scene_path"]).stem, job["method"]) for job in jobs}
        for future in as_completed(future_map):
            scene_id, method = future_map[future]
            print(f"completed {scene_id} :: {method}")
            results.append(future.result())

    scene_rows = [result.row for result in sorted(results, key=lambda item: (str(item.row["scene"]), str(item.row["method"])))]
    _write_csv(out_root / "summary_rows.csv", scene_rows)
    _write_json(out_root / "summary_rows.json", {"rows": scene_rows})
    summary_by_method = _aggregate(scene_rows, ["method"])
    summary_by_layout = _aggregate(scene_rows, ["method", "scene_layout_family"])
    _write_csv(out_root / "summary_by_method.csv", summary_by_method)
    _write_csv(out_root / "summary_by_scene_layout.csv", summary_by_layout)
    _write_json(
        out_root / "summary.json",
        {
            "run_id": run_id,
            "profile": str(args.profile),
            "scenes_root": str(scenes_root.resolve()),
            "assets_root": str(assets_root.resolve()),
            "methods": list(args.methods),
            "n_scenes": len(scenes),
            "benchmark_params": params,
            "summary_by_method": summary_by_method,
            "summary_by_scene_layout": summary_by_layout,
            "runs": [result.run_summary for result in results],
        },
    )
    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_root.resolve(), target_is_directory=True)
    print(json.dumps({"out_root": str(out_root.resolve()), "n_runs": len(results), "n_scenes": len(scenes)}, indent=2))


if __name__ == "__main__":
    main()
