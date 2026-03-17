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
from simulation.create_testing_specific_angles_near_target_far_diffuse import (
    DEFAULT_ASSET_ROOT as DEFAULT_NEAR_DIFFUSE_ASSET_ROOT,
    DEFAULT_BABBLE_COUNT as DEFAULT_NEAR_DIFFUSE_BABBLE_COUNT,
    DEFAULT_BABBLE_GAIN_MAX as DEFAULT_NEAR_DIFFUSE_BABBLE_GAIN_MAX,
    DEFAULT_BABBLE_GAIN_MIN as DEFAULT_NEAR_DIFFUSE_BABBLE_GAIN_MIN,
    DEFAULT_BOOTSTRAP_SEC as DEFAULT_NEAR_DIFFUSE_BOOTSTRAP_SEC,
    DEFAULT_CONFIG_ROOT as DEFAULT_NEAR_DIFFUSE_CONFIG_ROOT,
    DEFAULT_DURATION_SEC as DEFAULT_NEAR_DIFFUSE_DURATION_SEC,
    DEFAULT_WHAM_GAIN as DEFAULT_NEAR_DIFFUSE_WHAM_GAIN,
    generate_testing_specific_angles_near_target_far_diffuse_dataset,
)
from simulation.create_testing_specific_angles_zone_overlap import (
    DEFAULT_ASSET_ROOT as DEFAULT_ZONE_OVERLAP_ASSET_ROOT,
    DEFAULT_CONFIG_ROOT as DEFAULT_ZONE_OVERLAP_CONFIG_ROOT,
    DEFAULT_DURATION_SEC as DEFAULT_ZONE_OVERLAP_DURATION_SEC,
    DEFAULT_ZONE_WIDTH_DEG,
    generate_testing_specific_angles_zone_overlap_dataset,
)
from simulation.mic_array_profiles import SUPPORTED_MIC_ARRAY_PROFILES
from simulation.simulation_config import SimulationConfig


DEFAULT_SCENES_ROOT = Path("simulation/simulations/configs/testing_specific_angles_babble_bootstrap")
DEFAULT_ASSETS_ROOT = Path("simulation/simulations/assets/testing_specific_angles_babble_bootstrap")
DEFAULT_OUT_ROOT = Path("beamforming/benchmark/oracle_babble_bootstrap_mvdr")
DEFAULT_METHODS = [
    "mvdr_fd_silero_no_pf",
    "mvdr_fd_silero_pf_wiener",
    "mvdr_fd_silero_pf_rnnoise",
    "mvdr_fd_silero_pf_coherence_wiener",
    "mvdr_fd_silero_pf_wiener_then_rnnoise",
]
FAST_FRAME_MS = 40
DEFAULT_MVDR_HOP_MS = 40
DEFAULT_FD_ANALYSIS_WINDOW_MS = 80.0
DEFAULT_TARGET_ACTIVITY_UPDATE_EVERY_N_FAST_FRAMES = 1
DEFAULT_FD_COV_EMA_ALPHA = 0.08
DEFAULT_FD_DIAG_LOAD = 1e-3
DEFAULT_ACTIVE_UPDATE_SCALE = 0.20
DEFAULT_INACTIVE_UPDATE_SCALE = 1.0
SCENE_FAMILY_DEFAULTS: dict[str, dict[str, object]] = {
    "babble_bootstrap": {
        "scenes_root": DEFAULT_SCENES_ROOT,
        "assets_root": DEFAULT_ASSETS_ROOT,
        "duration_sec": DEFAULT_DURATION_SEC,
        "bootstrap_noise_only_sec": DEFAULT_BOOTSTRAP_SEC,
        "background_babble_count": DEFAULT_BABBLE_COUNT,
        "background_babble_gain_min": DEFAULT_BABBLE_GAIN_MIN,
        "background_babble_gain_max": DEFAULT_BABBLE_GAIN_MAX,
        "background_wham_gain": DEFAULT_WHAM_GAIN,
        "generator": generate_testing_specific_angles_babble_bootstrap_dataset,
    },
    "near_target_far_diffuse": {
        "scenes_root": DEFAULT_NEAR_DIFFUSE_CONFIG_ROOT,
        "assets_root": DEFAULT_NEAR_DIFFUSE_ASSET_ROOT,
        "duration_sec": DEFAULT_NEAR_DIFFUSE_DURATION_SEC,
        "bootstrap_noise_only_sec": DEFAULT_NEAR_DIFFUSE_BOOTSTRAP_SEC,
        "background_babble_count": DEFAULT_NEAR_DIFFUSE_BABBLE_COUNT,
        "background_babble_gain_min": DEFAULT_NEAR_DIFFUSE_BABBLE_GAIN_MIN,
        "background_babble_gain_max": DEFAULT_NEAR_DIFFUSE_BABBLE_GAIN_MAX,
        "background_wham_gain": DEFAULT_NEAR_DIFFUSE_WHAM_GAIN,
        "generator": generate_testing_specific_angles_near_target_far_diffuse_dataset,
    },
    "zone_overlap": {
        "scenes_root": DEFAULT_ZONE_OVERLAP_CONFIG_ROOT,
        "assets_root": DEFAULT_ZONE_OVERLAP_ASSET_ROOT,
        "duration_sec": DEFAULT_ZONE_OVERLAP_DURATION_SEC,
        "bootstrap_noise_only_sec": DEFAULT_BOOTSTRAP_SEC,
        "background_babble_count": DEFAULT_BABBLE_COUNT,
        "background_babble_gain_min": DEFAULT_BABBLE_GAIN_MIN,
        "background_babble_gain_max": DEFAULT_BABBLE_GAIN_MAX,
        "background_wham_gain": DEFAULT_WHAM_GAIN,
        "target_zone_width_deg": DEFAULT_ZONE_WIDTH_DEG,
        "generator": generate_testing_specific_angles_zone_overlap_dataset,
    },
}

# Sensitivity-tuned detector presets from
# `beamforming/benchmark/run_optuna_babble_bootstrap_mvdr.py`.
# Artifacts:
# - `beamforming/benchmark/_sens_tune_webrtc/best_params.json`
# - `beamforming/benchmark/_sens_tune_silero/best_params.json`
DEFAULT_TARGET_ACTIVITY_PRESETS: dict[str, dict[str, object]] = {
    "webrtc_fused": {
        "fd_cov_ema_alpha": 0.12104487978324685,
        "fd_diag_load": 0.0025330746540014474,
        "target_activity_low_threshold": 0.25508542011761026,
        "target_activity_high_threshold": 0.3457091527983343,
        "target_activity_enter_frames": 1,
        "target_activity_exit_frames": 8,
        "fd_cov_update_scale_target_active": 0.4650796940166687,
        "fd_cov_update_scale_target_inactive": 1.4455490474077703,
        "target_activity_detector_mode": "target_blocker_calibrated",
        "target_activity_detector_backend": "webrtc_fused",
        "target_activity_blocker_offset_deg": 120.0,
        "target_activity_bootstrap_only_calibration": True,
        "target_activity_ratio_floor_db": 1.587399872866511,
        "target_activity_ratio_active_db": 7.414056762673376,
        "target_activity_target_rms_floor_scale": 1.2654775061557584,
        "target_activity_blocker_rms_floor_scale": 1.0743760073868034,
        "target_activity_speech_weight": 0.22939773779184974,
        "target_activity_ratio_weight": 0.2614647149961218,
        "target_activity_blocker_weight": 0.1360370513913187,
        "target_activity_vad_mode": 1,
        "target_activity_vad_hangover_frames": 5,
        "target_activity_noise_floor_rise_alpha": 0.0047745636119070406,
        "target_activity_noise_floor_fall_alpha": 0.07742428232497138,
        "target_activity_noise_floor_margin_scale": 1.8140441247373726,
        "target_activity_rms_scale": 1.9864695748233385,
        "target_activity_score_exponent": 0.7719772826786356,
    },
    "silero_fused": {
        "fd_cov_ema_alpha": 0.2965906035161345,
        "fd_diag_load": 0.012141307774357374,
        "target_activity_low_threshold": 0.10544774305969414,
        "target_activity_high_threshold": 0.6508335197763335,
        "target_activity_enter_frames": 1,
        "target_activity_exit_frames": 7,
        "fd_cov_update_scale_target_active": 0.4241144063085703,
        "fd_cov_update_scale_target_inactive": 1.2561064512368887,
        "target_activity_detector_mode": "target_blocker_calibrated",
        "target_activity_detector_backend": "silero_fused",
        "target_activity_blocker_offset_deg": 120.0,
        "target_activity_bootstrap_only_calibration": True,
        "target_activity_ratio_floor_db": -1.5557320895954578,
        "target_activity_ratio_active_db": 3.1884929640820445,
        "target_activity_target_rms_floor_scale": 1.3476071785753891,
        "target_activity_blocker_rms_floor_scale": 2.008344796225831,
        "target_activity_speech_weight": 0.6051437824379127,
        "target_activity_ratio_weight": 0.26508371615422194,
        "target_activity_blocker_weight": 0.02224542260010827,
        "target_activity_vad_mode": 1,
        "target_activity_vad_hangover_frames": 2,
        "target_activity_noise_floor_rise_alpha": 0.024462802690520202,
        "target_activity_noise_floor_fall_alpha": 0.16301379312525116,
        "target_activity_noise_floor_margin_scale": 2.33081911386449,
        "target_activity_rms_scale": 4.305504476133645,
        "target_activity_score_exponent": 0.15763482134447154,
    },
}

DEFAULT_ORACLE_NOISE_PRESETS: dict[str, dict[str, object]] = {
    "oracle_noise_mild": {
        "fd_noise_covariance_mode": "oracle_non_target_residual",
        "fd_cov_ema_alpha": 0.18,
        "fd_diag_load": 0.006,
        "fd_cov_update_scale_target_active": 1.0,
        "fd_cov_update_scale_target_inactive": 1.0,
        "target_activity_detector_backend": "silero_fused",
    },
    "oracle_noise_medium": {
        "fd_noise_covariance_mode": "oracle_non_target_residual",
        "fd_cov_ema_alpha": 0.28,
        "fd_diag_load": 0.003,
        "fd_cov_update_scale_target_active": 1.0,
        "fd_cov_update_scale_target_inactive": 1.0,
        "target_activity_detector_backend": "silero_fused",
    },
    "oracle_noise_hard": {
        "fd_noise_covariance_mode": "oracle_non_target_residual",
        "fd_cov_ema_alpha": 0.42,
        "fd_diag_load": 0.0012,
        "fd_cov_update_scale_target_active": 1.0,
        "fd_cov_update_scale_target_inactive": 1.0,
        "target_activity_detector_backend": "silero_fused",
    },
}

DEFAULT_POSTFILTER_PRESETS: dict[str, dict[str, object]] = {
    "no_pf": {
        "postfilter_enabled": False,
        "postfilter_method": "off",
        "postfilter_noise_ema_alpha": 0.08,
        "postfilter_speech_ema_alpha": 0.12,
        "postfilter_gain_floor": 0.22,
        "postfilter_gain_ema_alpha": 0.2,
        "postfilter_dd_alpha": 0.92,
        "postfilter_noise_update_speech_scale": 0.2,
        "postfilter_freq_smoothing_bins": 2,
        "postfilter_gain_max_step_db": 2.5,
        "rnnoise_wet_mix": 1.0,
        "rnnoise_input_gain_db": 0.0,
        "coherence_wiener_gain_floor": 0.12,
        "coherence_wiener_coherence_exponent": 1.5,
        "coherence_wiener_temporal_alpha": 0.65,
    },
    "pf_mild": {
        "postfilter_enabled": True,
        "postfilter_method": "wiener_dd",
        "postfilter_noise_ema_alpha": 0.06,
        "postfilter_speech_ema_alpha": 0.10,
        "postfilter_gain_floor": 0.34,
        "postfilter_gain_ema_alpha": 0.16,
        "postfilter_dd_alpha": 0.94,
        "postfilter_noise_update_speech_scale": 0.12,
        "postfilter_freq_smoothing_bins": 1,
        "postfilter_gain_max_step_db": 1.8,
        "rnnoise_wet_mix": 1.0,
        "rnnoise_input_gain_db": 0.0,
        "coherence_wiener_gain_floor": 0.12,
        "coherence_wiener_coherence_exponent": 1.5,
        "coherence_wiener_temporal_alpha": 0.65,
    },
    "pf_medium": {
        "postfilter_enabled": True,
        "postfilter_method": "wiener_dd",
        "postfilter_noise_ema_alpha": 0.08,
        "postfilter_speech_ema_alpha": 0.12,
        "postfilter_gain_floor": 0.26,
        "postfilter_gain_ema_alpha": 0.20,
        "postfilter_dd_alpha": 0.92,
        "postfilter_noise_update_speech_scale": 0.15,
        "postfilter_freq_smoothing_bins": 2,
        "postfilter_gain_max_step_db": 2.2,
        "rnnoise_wet_mix": 1.0,
        "rnnoise_input_gain_db": 0.0,
        "coherence_wiener_gain_floor": 0.12,
        "coherence_wiener_coherence_exponent": 1.5,
        "coherence_wiener_temporal_alpha": 0.65,
    },
    "pf_rnnoise": {
        "postfilter_enabled": True,
        "postfilter_method": "rnnoise",
        "postfilter_noise_ema_alpha": 0.08,
        "postfilter_speech_ema_alpha": 0.12,
        "postfilter_gain_floor": 0.22,
        "postfilter_gain_ema_alpha": 0.2,
        "postfilter_dd_alpha": 0.92,
        "postfilter_noise_update_speech_scale": 0.2,
        "postfilter_freq_smoothing_bins": 2,
        "postfilter_gain_max_step_db": 2.5,
        "rnnoise_wet_mix": 1.0,
        "rnnoise_input_gain_db": 0.0,
    },
    "pf_coherence_wiener": {
        "postfilter_enabled": True,
        "postfilter_method": "coherence_wiener",
        "postfilter_noise_ema_alpha": 0.08,
        "postfilter_speech_ema_alpha": 0.12,
        "postfilter_gain_floor": 0.22,
        "postfilter_gain_ema_alpha": 0.2,
        "postfilter_dd_alpha": 0.92,
        "postfilter_noise_update_speech_scale": 0.2,
        "postfilter_freq_smoothing_bins": 2,
        "postfilter_gain_max_step_db": 2.5,
        "coherence_wiener_gain_floor": 0.12,
        "coherence_wiener_coherence_exponent": 1.5,
        "coherence_wiener_temporal_alpha": 0.65,
    },
    "pf_wiener_then_rnnoise": {
        "postfilter_enabled": True,
        "postfilter_method": "wiener_then_rnnoise",
        "postfilter_noise_ema_alpha": 0.06,
        "postfilter_speech_ema_alpha": 0.10,
        "postfilter_gain_floor": 0.34,
        "postfilter_gain_ema_alpha": 0.16,
        "postfilter_dd_alpha": 0.94,
        "postfilter_noise_update_speech_scale": 0.12,
        "postfilter_freq_smoothing_bins": 1,
        "postfilter_gain_max_step_db": 1.8,
        "rnnoise_wet_mix": 1.0,
        "rnnoise_input_gain_db": 0.0,
    },
}


@dataclass(frozen=True)
class MethodSpec:
    method_key: str
    beamforming_mode: str
    target_activity_mode: str | None
    noise_covariance_mode: str = "estimated_target_subtractive"
    aggression_level: str = "baseline"
    postfilter_preset: str = "no_pf"
    focus_zone_target: bool = False
    slow_path_enabled: bool = False
    multi_target_tracked: bool = False


@dataclass(frozen=True)
class JobResult:
    row: dict[str, object]
    run_summary: dict[str, object]


METHOD_SPECS: dict[str, MethodSpec] = {
    "delay_sum": MethodSpec("delay_sum", "delay_sum", None),
    "mvdr_fd_bootstrap_oracle_activity": MethodSpec("mvdr_fd_bootstrap_oracle_activity", "mvdr_fd", "oracle_target_activity"),
    "mvdr_fd_bootstrap_estimated_activity": MethodSpec("mvdr_fd_bootstrap_estimated_activity", "mvdr_fd", "estimated_target_activity"),
    "mvdr_fd_bootstrap_estimated_activity_silero": MethodSpec("mvdr_fd_bootstrap_estimated_activity_silero", "mvdr_fd", "estimated_target_activity"),
    "mvdr_fd_bootstrap_oracle_noise_mild": MethodSpec("mvdr_fd_bootstrap_oracle_noise_mild", "mvdr_fd", "oracle_target_activity", "oracle_non_target_residual", "mild"),
    "mvdr_fd_bootstrap_oracle_noise_medium": MethodSpec("mvdr_fd_bootstrap_oracle_noise_medium", "mvdr_fd", "oracle_target_activity", "oracle_non_target_residual", "medium"),
    "mvdr_fd_bootstrap_oracle_noise_hard": MethodSpec("mvdr_fd_bootstrap_oracle_noise_hard", "mvdr_fd", "oracle_target_activity", "oracle_non_target_residual", "hard"),
    "mvdr_fd_silero_no_pf": MethodSpec("mvdr_fd_silero_no_pf", "mvdr_fd", "estimated_target_activity", postfilter_preset="no_pf"),
    "mvdr_fd_silero_pf_wiener": MethodSpec("mvdr_fd_silero_pf_wiener", "mvdr_fd", "estimated_target_activity", postfilter_preset="pf_mild"),
    "mvdr_fd_silero_pf_rnnoise": MethodSpec("mvdr_fd_silero_pf_rnnoise", "mvdr_fd", "estimated_target_activity", postfilter_preset="pf_rnnoise"),
    "mvdr_fd_silero_pf_coherence_wiener": MethodSpec("mvdr_fd_silero_pf_coherence_wiener", "mvdr_fd", "estimated_target_activity", postfilter_preset="pf_coherence_wiener"),
    "mvdr_fd_silero_pf_wiener_then_rnnoise": MethodSpec("mvdr_fd_silero_pf_wiener_then_rnnoise", "mvdr_fd", "estimated_target_activity", postfilter_preset="pf_wiener_then_rnnoise"),
    "delay_sum_zone_target": MethodSpec("delay_sum_zone_target", "delay_sum", None, focus_zone_target=True),
    "mvdr_fd_zone_target_oracle_activity": MethodSpec("mvdr_fd_zone_target_oracle_activity", "mvdr_fd", "oracle_target_activity", focus_zone_target=True),
    "mvdr_fd_zone_target_estimated_activity": MethodSpec("mvdr_fd_zone_target_estimated_activity", "mvdr_fd", "estimated_target_activity", focus_zone_target=True),
    "lcmv_top2_tracked_oracle": MethodSpec("lcmv_top2_tracked_oracle", "lcmv_top2_tracked", "oracle_target_activity", multi_target_tracked=True),
    "lcmv_top2_tracked_estimated": MethodSpec("lcmv_top2_tracked_estimated", "lcmv_top2_tracked", "estimated_target_activity", slow_path_enabled=True, multi_target_tracked=True),
}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _detector_param_overrides(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    detector_fields = [
        "fd_cov_ema_alpha",
        "fd_diag_load",
        "fd_noise_covariance_mode",
        "mvdr_hop_ms",
        "target_activity_low_threshold",
        "target_activity_high_threshold",
        "target_activity_enter_frames",
        "target_activity_exit_frames",
        "fd_cov_update_scale_target_active",
        "fd_cov_update_scale_target_inactive",
        "target_activity_detector_mode",
        "target_activity_detector_backend",
        "target_activity_update_every_n_fast_frames",
        "target_activity_blocker_offset_deg",
        "target_activity_bootstrap_only_calibration",
        "target_activity_ratio_floor_db",
        "target_activity_ratio_active_db",
        "target_activity_target_rms_floor_scale",
        "target_activity_blocker_rms_floor_scale",
        "target_activity_speech_weight",
        "target_activity_ratio_weight",
        "target_activity_blocker_weight",
        "target_activity_vad_mode",
        "target_activity_vad_hangover_frames",
        "target_activity_noise_floor_rise_alpha",
        "target_activity_noise_floor_fall_alpha",
        "target_activity_noise_floor_margin_scale",
        "target_activity_rms_scale",
        "target_activity_score_exponent",
        "postfilter_enabled",
        "postfilter_method",
        "postfilter_noise_ema_alpha",
        "postfilter_speech_ema_alpha",
        "postfilter_gain_floor",
        "postfilter_gain_ema_alpha",
        "postfilter_dd_alpha",
        "postfilter_noise_update_speech_scale",
        "postfilter_freq_smoothing_bins",
        "postfilter_gain_max_step_db",
        "rnnoise_wet_mix",
        "rnnoise_input_gain_db",
        "coherence_wiener_gain_floor",
        "coherence_wiener_coherence_exponent",
        "coherence_wiener_temporal_alpha",
        "split_runtime_mode",
        "postfilter_queue_max_frames",
        "postfilter_queue_drop_oldest",
        "focus_direction_match_window_deg",
        "focus_target_hold_frames",
    ]
    for field in detector_fields:
        value = getattr(args, field, None)
        if value is not None:
            out[field] = value
    return out


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
        "beamforming_rtf",
        "postfilter_rtf",
        "pipeline_rtf",
        "interstage_queue_wait_p95_ms",
        "interstage_queue_depth_max",
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
    slow_path_enabled = bool(method_spec.slow_path_enabled)
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
            "localization_vad_enabled": False,
            "localization_backend": str(params["localization_backend"]),
            "beamforming_mode": str(method_spec.beamforming_mode),
            "mvdr_hop_ms": None if method_spec.beamforming_mode != "mvdr_fd" else int(params["mvdr_hop_ms"]),
            "fd_analysis_window_ms": float(params["fd_analysis_window_ms"]),
            "fd_cov_ema_alpha": float(params["fd_cov_ema_alpha"]),
            "fd_diag_load": float(params["fd_diag_load"]),
            "fd_noise_covariance_mode": str(params["fd_noise_covariance_mode"]),
            "target_activity_rnn_update_mode": method_spec.target_activity_mode,
            "target_activity_low_threshold": float(params["target_activity_low_threshold"]),
            "target_activity_high_threshold": float(params["target_activity_high_threshold"]),
            "target_activity_enter_frames": int(params["target_activity_enter_frames"]),
            "target_activity_exit_frames": int(params["target_activity_exit_frames"]),
            "fd_cov_update_scale_target_active": float(params["fd_cov_update_scale_target_active"]),
            "fd_cov_update_scale_target_inactive": float(params["fd_cov_update_scale_target_inactive"]),
            "target_activity_detector_mode": str(params["target_activity_detector_mode"]),
            "target_activity_detector_backend": str(params["target_activity_detector_backend"]),
            "target_activity_update_every_n_fast_frames": int(params["target_activity_update_every_n_fast_frames"]),
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
            "split_runtime_mode": str(params["split_runtime_mode"]),
            "postfilter_queue_max_frames": int(params["postfilter_queue_max_frames"]),
            "postfilter_queue_drop_oldest": bool(params["postfilter_queue_drop_oldest"]),
            "postfilter_enabled": bool(params["postfilter_enabled"]),
            "postfilter_method": str(params["postfilter_method"]),
            "postfilter_noise_ema_alpha": float(params["postfilter_noise_ema_alpha"]),
            "postfilter_speech_ema_alpha": float(params["postfilter_speech_ema_alpha"]),
            "postfilter_gain_floor": float(params["postfilter_gain_floor"]),
            "postfilter_gain_ema_alpha": float(params["postfilter_gain_ema_alpha"]),
            "postfilter_dd_alpha": float(params["postfilter_dd_alpha"]),
            "postfilter_noise_update_speech_scale": float(params["postfilter_noise_update_speech_scale"]),
            "postfilter_freq_smoothing_bins": int(params["postfilter_freq_smoothing_bins"]),
            "postfilter_gain_max_step_db": float(params["postfilter_gain_max_step_db"]),
            "rnnoise_wet_mix": float(params["rnnoise_wet_mix"]),
            "rnnoise_input_gain_db": float(params["rnnoise_input_gain_db"]),
            "coherence_wiener_gain_floor": float(params["coherence_wiener_gain_floor"]),
            "coherence_wiener_coherence_exponent": float(params["coherence_wiener_coherence_exponent"]),
            "coherence_wiener_temporal_alpha": float(params["coherence_wiener_temporal_alpha"]),
            "focus_direction_match_window_deg": float(params["focus_direction_match_window_deg"]),
            "focus_target_hold_frames": int(params["focus_target_hold_frames"]),
            "multi_target_max_speakers": int(params["multi_target_max_speakers"]),
            "multi_target_hold_frames": int(params["multi_target_hold_frames"]),
            "multi_target_min_confidence": float(params["multi_target_min_confidence"]),
            "multi_target_min_activity": float(params["multi_target_min_activity"]),
            "output_normalization_enabled": False,
            "output_allow_amplification": False,
            "own_voice_suppression_mode": "off",
        },
        slow_path={
            "enabled": bool(slow_path_enabled),
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


def _build_target_only_schedule(
    metadata: dict[str, object],
    *,
    target_speaker_id: int,
    sample_rate: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    active_speaker_ids = np.full(n_samples, -1, dtype=np.int32)
    active_doa_deg = np.full(n_samples, np.nan, dtype=np.float64)
    for row in list(metadata.get("assets", {}).get("speech", [])):
        if int(row.get("speaker_id", -1)) != int(target_speaker_id):
            continue
        window = row.get("active_window_sec")
        if not isinstance(window, list) or len(window) < 2:
            continue
        start = max(0, int(round(float(window[0]) * sample_rate)))
        end = min(n_samples, int(round(float(window[1]) * sample_rate)))
        if end <= start:
            continue
        active_speaker_ids[start:end] = int(target_speaker_id)
        active_doa_deg[start:end] = float(row.get("angle_deg", math.nan))
    return active_speaker_ids, active_doa_deg


def _build_multi_target_activity_mask(
    metadata: dict[str, object],
    *,
    target_speaker_ids: set[int],
    sample_rate: int,
    n_samples: int,
) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for row in list(metadata.get("assets", {}).get("speech", [])):
        sid = int(row.get("speaker_id", -1))
        if sid not in target_speaker_ids:
            continue
        window = row.get("active_window_sec")
        if not isinstance(window, list) or len(window) < 2:
            continue
        start = max(0, int(round(float(window[0]) * sample_rate)))
        end = min(n_samples, int(round(float(window[1]) * sample_rate)))
        if end > start:
            mask[start:end] = True
    return mask


def _build_clean_reference_for_target(
    source_signals: list[np.ndarray],
    speaker_to_source_idx: dict[int, int],
    active_speaker_ids: np.ndarray,
    target_speaker_ids: set[int],
) -> np.ndarray:
    n_samples = int(active_speaker_ids.shape[0])
    clean_ref = np.zeros(n_samples, dtype=np.float64)
    for speaker_id in target_speaker_ids:
        source_idx = speaker_to_source_idx.get(int(speaker_id))
        if source_idx is None:
            continue
        mask = active_speaker_ids == int(speaker_id)
        if not np.any(mask):
            continue
        source = np.asarray(source_signals[int(source_idx)], dtype=np.float64).reshape(-1)
        if source.shape[0] < n_samples:
            source = np.pad(source, (0, n_samples - source.shape[0]))
        clean_ref[mask] = source[:n_samples][mask]
    return clean_ref


def _build_clean_reference_sum_for_targets(
    metadata: dict[str, object],
    *,
    source_signals: list[np.ndarray],
    speaker_to_source_idx: dict[int, int],
    target_speaker_ids: set[int],
    sample_rate: int,
    n_samples: int,
) -> np.ndarray:
    clean_ref = np.zeros(n_samples, dtype=np.float64)
    for row in list(metadata.get("assets", {}).get("speech", [])):
        sid = int(row.get("speaker_id", -1))
        if sid not in target_speaker_ids:
            continue
        source_idx = speaker_to_source_idx.get(sid)
        if source_idx is None:
            continue
        window = row.get("active_window_sec")
        if not isinstance(window, list) or len(window) < 2:
            continue
        start = max(0, int(round(float(window[0]) * sample_rate)))
        end = min(n_samples, int(round(float(window[1]) * sample_rate)))
        if end <= start:
            continue
        source = np.asarray(source_signals[int(source_idx)], dtype=np.float64).reshape(-1)
        if source.shape[0] < n_samples:
            source = np.pad(source, (0, n_samples - source.shape[0]))
        clean_ref[start:end] += source[start:end]
    return clean_ref


def _build_oracle_target_mic_audio_for_target(
    *,
    source_mic_audio: list[np.ndarray],
    speaker_to_source_idx: dict[int, int],
    active_speaker_ids: np.ndarray,
    target_speaker_ids: set[int],
    n_samples: int,
    n_mics: int,
) -> np.ndarray:
    oracle_target = np.zeros((int(n_samples), int(n_mics)), dtype=np.float32)
    for speaker_id in target_speaker_ids:
        source_idx = speaker_to_source_idx.get(int(speaker_id))
        if source_idx is None:
            continue
        mask = np.asarray(active_speaker_ids == int(speaker_id), dtype=bool).reshape(-1)
        if not np.any(mask):
            continue
        source_mic = np.asarray(source_mic_audio[int(source_idx)], dtype=np.float32)
        if source_mic.shape[0] < n_samples:
            source_mic = np.pad(source_mic, ((0, n_samples - source_mic.shape[0]), (0, 0)))
        oracle_target[mask, :] += source_mic[:n_samples][mask, :]
    return oracle_target.astype(np.float32, copy=False)


def _build_oracle_non_target_mic_audio_for_target(
    *,
    mic_audio: np.ndarray,
    source_mic_audio: list[np.ndarray],
    speaker_to_source_idx: dict[int, int],
    active_speaker_ids: np.ndarray,
    target_speaker_ids: set[int],
    n_mics: int,
) -> np.ndarray:
    oracle_target = _build_oracle_target_mic_audio_for_target(
        source_mic_audio=source_mic_audio,
        speaker_to_source_idx=speaker_to_source_idx,
        active_speaker_ids=active_speaker_ids,
        target_speaker_ids=target_speaker_ids,
        n_samples=int(active_speaker_ids.shape[0]),
        n_mics=int(n_mics),
    )
    return (np.asarray(mic_audio, dtype=np.float32) - oracle_target).astype(np.float32, copy=False)


def _build_pair_activity_frame_states(
    *,
    metadata: dict[str, object],
    sample_rate: int,
    n_samples: int,
    fast_frame_ms: int,
    target_speaker_ids: set[int],
    bootstrap_window_sec: tuple[float, float],
    bootstrap_reference_doa_deg: float,
) -> list[OracleFrameState]:
    frame_samples = max(1, int(round(sample_rate * (float(fast_frame_ms) / 1000.0))))
    n_frames = int(math.ceil(n_samples / frame_samples))
    speech_rows = [row for row in list(metadata.get("assets", {}).get("speech", [])) if int(row.get("speaker_id", -1)) in target_speaker_ids]
    states: list[OracleFrameState] = []
    for frame_idx in range(n_frames):
        start = frame_idx * frame_samples
        start_sec = float(start) / max(sample_rate, 1)
        active_ids: list[int] = []
        active_peaks: list[float] = []
        for row in speech_rows:
            window = row.get("active_window_sec")
            if not isinstance(window, list) or len(window) < 2:
                continue
            if float(window[0]) <= start_sec < float(window[1]):
                active_ids.append(int(row.get("speaker_id", -1)))
                active_peaks.append(float(row.get("angle_deg", math.nan)))
        if float(bootstrap_window_sec[0]) <= start_sec < float(bootstrap_window_sec[1]):
            peaks = [float(bootstrap_reference_doa_deg)]
            scores = [1.0]
        else:
            peaks = [float(v) for v in active_peaks]
            scores = [1.0 for _ in active_peaks]
        states.append(
            OracleFrameState(
                frame_index=frame_idx,
                timestamp_ms=float(frame_idx * fast_frame_ms),
                target_speaker_id=(int(active_ids[0]) if active_ids else None),
                target_doa_deg=(float(active_peaks[0]) if active_peaks else None),
                target_activity_score=1.0 if active_ids else 0.0,
                target_active=bool(active_ids),
                null_user_speaker_id=None,
                null_user_doa_deg=None,
                force_suppression_active=False,
                null_candidate=False,
                null_fallback=False,
                peaks_deg=tuple(peaks),
                peak_scores=tuple(scores),
            )
        )
    return states


def _build_zone_overlap_frame_states(
    *,
    metadata: dict[str, object],
    sample_rate: int,
    n_samples: int,
    fast_frame_ms: int,
    target_speaker_id: int,
    bootstrap_window_sec: tuple[float, float],
    bootstrap_reference_doa_deg: float,
) -> list[OracleFrameState]:
    frame_samples = max(1, int(round(sample_rate * (float(fast_frame_ms) / 1000.0))))
    n_frames = int(math.ceil(n_samples / frame_samples))
    speech_rows = list(metadata.get("assets", {}).get("speech", []))
    target_row = next((row for row in speech_rows if int(row.get("speaker_id", -1)) == int(target_speaker_id)), None)
    target_doa = float(target_row.get("angle_deg", bootstrap_reference_doa_deg)) if target_row is not None else float(bootstrap_reference_doa_deg)
    states: list[OracleFrameState] = []
    for frame_idx in range(n_frames):
        start = frame_idx * frame_samples
        start_sec = float(start) / max(sample_rate, 1)
        active_ids: list[int] = []
        active_peaks: list[float] = []
        for row in speech_rows:
            window = row.get("active_window_sec")
            if not isinstance(window, list) or len(window) < 2:
                continue
            if float(window[0]) <= start_sec < float(window[1]):
                active_ids.append(int(row.get("speaker_id", -1)))
                active_peaks.append(float(row.get("angle_deg", math.nan)))
        target_active = int(target_speaker_id) in active_ids
        peaks: list[float] = []
        scores: list[float] = []
        if float(bootstrap_window_sec[0]) <= start_sec < float(bootstrap_window_sec[1]):
            peaks = [float(bootstrap_reference_doa_deg)]
            scores = [1.0]
        else:
            for speaker_id, doa_deg in zip(active_ids, active_peaks, strict=True):
                peaks.append(float(doa_deg))
                scores.append(1.0 if int(speaker_id) == int(target_speaker_id) else 0.95)
        states.append(
            OracleFrameState(
                frame_index=frame_idx,
                timestamp_ms=float(frame_idx * fast_frame_ms),
                target_speaker_id=int(target_speaker_id) if target_active else None,
                target_doa_deg=float(target_doa) if target_active else None,
                target_activity_score=1.0 if target_active else 0.0,
                target_active=bool(target_active),
                null_user_speaker_id=None,
                null_user_doa_deg=None,
                force_suppression_active=False,
                null_candidate=False,
                null_fallback=False,
                peaks_deg=tuple(float(v) for v in peaks),
                peak_scores=tuple(float(v) for v in scores),
            )
        )
    return states


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


def _build_oracle_non_target_mic_audio(
    *,
    mic_audio: np.ndarray,
    source_mic_audio: list[np.ndarray],
    speaker_to_source_idx: dict[int, int],
    active_speaker_ids: np.ndarray,
) -> np.ndarray:
    oracle_target = np.zeros_like(np.asarray(mic_audio, dtype=np.float32))
    n_samples = int(active_speaker_ids.shape[0])
    for speaker_id, source_idx in speaker_to_source_idx.items():
        mask = np.asarray(active_speaker_ids == int(speaker_id), dtype=bool).reshape(-1)
        if not np.any(mask):
            continue
        source_mic = np.asarray(source_mic_audio[int(source_idx)], dtype=np.float32)
        if source_mic.shape[0] < n_samples:
            source_mic = np.pad(source_mic, ((0, n_samples - source_mic.shape[0]), (0, 0)))
        oracle_target[mask, :] += source_mic[:n_samples][mask, :]
    return (np.asarray(mic_audio, dtype=np.float32) - oracle_target).astype(np.float32, copy=False)


def _build_oracle_target_mic_audio(
    *,
    source_mic_audio: list[np.ndarray],
    speaker_to_source_idx: dict[int, int],
    active_speaker_ids: np.ndarray,
    n_samples: int,
    n_mics: int,
) -> np.ndarray:
    oracle_target = np.zeros((int(n_samples), int(n_mics)), dtype=np.float32)
    for speaker_id, source_idx in speaker_to_source_idx.items():
        mask = np.asarray(active_speaker_ids == int(speaker_id), dtype=bool).reshape(-1)
        if not np.any(mask):
            continue
        source_mic = np.asarray(source_mic_audio[int(source_idx)], dtype=np.float32)
        if source_mic.shape[0] < n_samples:
            source_mic = np.pad(source_mic, ((0, n_samples - source_mic.shape[0]), (0, 0)))
        oracle_target[mask, :] += source_mic[:n_samples][mask, :]
    return oracle_target.astype(np.float32, copy=False)


def _oracle_noise_frame_provider(oracle_noise_audio: np.ndarray, frame_samples: int):
    audio = np.asarray(oracle_noise_audio, dtype=np.float32)

    def _provider(frame_index: int, _timestamp_ms: float) -> np.ndarray | None:
        start = int(frame_index) * int(frame_samples)
        if start >= audio.shape[0]:
            return None
        end = min(audio.shape[0], start + int(frame_samples))
        frame = audio[start:end, :]
        if frame.shape[0] < int(frame_samples):
            frame = np.pad(frame, ((0, int(frame_samples) - frame.shape[0]), (0, 0)))
        return frame.astype(np.float32, copy=False)

    return _provider


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

    from simulation.simulator import run_simulation_with_source_contributions

    mic_audio, mic_pos, source_signals, source_mic_audio = run_simulation_with_source_contributions(sim_cfg)
    sample_rate = int(sim_cfg.audio.fs)
    n_samples = int(mic_audio.shape[0])
    speech_rows = list(metadata_payload.get("assets", {}).get("speech", []))
    target_speaker_id = int(metadata_payload.get("zone_target_speaker_id", 0))
    multi_target_speaker_ids = {
        int(row.get("speaker_id", -1))
        for row in speech_rows
        if int(row.get("speaker_id", -1)) >= 0
    } if method_spec.multi_target_tracked else {int(target_speaker_id)}
    if method_spec.focus_zone_target:
        active_speaker_ids, active_doa_deg = _build_target_only_schedule(
            metadata_payload,
            target_speaker_id=target_speaker_id,
            sample_rate=sample_rate,
            n_samples=n_samples,
        )
    elif method_spec.multi_target_tracked:
        active_speaker_ids = np.full(n_samples, -1, dtype=np.int32)
        active_doa_deg = np.full(n_samples, np.nan, dtype=np.float64)
    else:
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

    speech_mask = (
        _build_multi_target_activity_mask(
            metadata_payload,
            target_speaker_ids=multi_target_speaker_ids,
            sample_rate=sample_rate,
            n_samples=n_samples,
        )
        if method_spec.multi_target_tracked
        else np.asarray(active_speaker_ids >= 0, dtype=bool)
    )
    bootstrap_window = tuple(float(v) for v in metadata_payload.get("bootstrap_noise_only_window_sec", [0.0, 0.0])[:2])
    bootstrap_reference_doa_deg = float(metadata_payload.get("bootstrap_reference_doa_deg", scene_meta["main_angle_deg"] or 0.0))
    bootstrap_mask = _bootstrap_mask(n_samples, sample_rate, bootstrap_window)
    background_only_mask = np.logical_not(speech_mask)
    speaker_to_source_idx = _speaker_source_index_map(sim_cfg, metadata_payload)
    if method_spec.multi_target_tracked:
        clean_ref_dry = _build_clean_reference_sum_for_targets(
            metadata_payload,
            source_signals=source_signals,
            speaker_to_source_idx=speaker_to_source_idx,
            target_speaker_ids=multi_target_speaker_ids,
            sample_rate=sample_rate,
            n_samples=n_samples,
        )
    elif method_spec.focus_zone_target:
        clean_ref_dry = _build_clean_reference_for_target(
            source_signals,
            speaker_to_source_idx,
            active_speaker_ids,
            {int(target_speaker_id)},
        )
    else:
        clean_ref_dry = _build_clean_reference(source_signals, speaker_to_source_idx, active_speaker_ids)
    raw_mix = np.mean(np.asarray(mic_audio, dtype=np.float64), axis=1).astype(np.float32, copy=False)
    if method_spec.multi_target_tracked:
        oracle_target_mic_audio = np.zeros((int(n_samples), int(mic_audio.shape[1])), dtype=np.float32)
        for sid in multi_target_speaker_ids:
            sid_active_ids, _sid_doa = _build_target_only_schedule(
                metadata_payload,
                target_speaker_id=int(sid),
                sample_rate=sample_rate,
                n_samples=n_samples,
            )
            oracle_target_mic_audio += _build_oracle_target_mic_audio_for_target(
                source_mic_audio=source_mic_audio,
                speaker_to_source_idx=speaker_to_source_idx,
                active_speaker_ids=sid_active_ids,
                target_speaker_ids={int(sid)},
                n_samples=n_samples,
                n_mics=int(mic_audio.shape[1]),
            )
    elif method_spec.focus_zone_target:
        oracle_target_mic_audio = _build_oracle_target_mic_audio_for_target(
            source_mic_audio=source_mic_audio,
            speaker_to_source_idx=speaker_to_source_idx,
            active_speaker_ids=active_speaker_ids,
            target_speaker_ids={int(target_speaker_id)},
            n_samples=n_samples,
            n_mics=int(mic_audio.shape[1]),
        )
    else:
        oracle_target_mic_audio = _build_oracle_target_mic_audio(
            source_mic_audio=source_mic_audio,
            speaker_to_source_idx=speaker_to_source_idx,
            active_speaker_ids=active_speaker_ids,
            n_samples=n_samples,
            n_mics=int(mic_audio.shape[1]),
        )
    oracle_noise_audio = None
    if method_spec.noise_covariance_mode == "oracle_non_target_residual":
        if method_spec.multi_target_tracked:
            oracle_noise_audio = (np.asarray(mic_audio, dtype=np.float32) - oracle_target_mic_audio).astype(np.float32, copy=False)
        elif method_spec.focus_zone_target:
            oracle_noise_audio = _build_oracle_non_target_mic_audio_for_target(
                mic_audio=np.asarray(mic_audio, dtype=np.float32),
                source_mic_audio=source_mic_audio,
                speaker_to_source_idx=speaker_to_source_idx,
                active_speaker_ids=active_speaker_ids,
                target_speaker_ids={int(target_speaker_id)},
                n_mics=int(mic_audio.shape[1]),
            )
        else:
            oracle_noise_audio = _build_oracle_non_target_mic_audio(
                mic_audio=np.asarray(mic_audio, dtype=np.float32),
                source_mic_audio=source_mic_audio,
                speaker_to_source_idx=speaker_to_source_idx,
                active_speaker_ids=active_speaker_ids,
            )

    if method_spec.multi_target_tracked:
        frame_states = _build_pair_activity_frame_states(
            metadata=metadata_payload,
            sample_rate=sample_rate,
            n_samples=n_samples,
            fast_frame_ms=FAST_FRAME_MS,
            target_speaker_ids=multi_target_speaker_ids,
            bootstrap_window_sec=bootstrap_window,
            bootstrap_reference_doa_deg=bootstrap_reference_doa_deg,
        )
    elif method_spec.focus_zone_target:
        frame_states = _build_zone_overlap_frame_states(
            metadata=metadata_payload,
            sample_rate=sample_rate,
            n_samples=n_samples,
            fast_frame_ms=FAST_FRAME_MS,
            target_speaker_id=target_speaker_id,
            bootstrap_window_sec=bootstrap_window,
            bootstrap_reference_doa_deg=bootstrap_reference_doa_deg,
        )
    else:
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
    sf.write(run_dir / "clean_active_target_dry.wav", clean_ref_dry.astype(np.float32), sample_rate)
    sf.write(run_dir / "oracle_target_mic_mean.wav", np.mean(np.asarray(oracle_target_mic_audio, dtype=np.float64), axis=1).astype(np.float32, copy=False), sample_rate)
    if oracle_noise_audio is not None:
        sf.write(
            run_dir / "oracle_non_target_noise.wav",
            np.mean(np.asarray(oracle_noise_audio, dtype=np.float64), axis=1).astype(np.float32, copy=False),
            sample_rate,
        )

    t0 = perf_counter()
    runtime_summary = run_offline_session_pipeline(
        req=req,
        mic_audio=np.asarray(mic_audio, dtype=np.float32),
        mic_geometry_xyz=np.asarray(mic_pos, dtype=np.float64),
        out_dir=run_dir,
        capture_trace=True,
        srp_override_provider=(
            None
            if (method_spec.multi_target_tracked and method_spec.target_activity_mode == "estimated_target_activity")
            else _oracle_srp_override_provider(frame_states)
        ),
        initial_focus_direction_deg=(
            float(metadata_payload.get("target_zone_center_deg"))
            if method_spec.focus_zone_target and metadata_payload.get("target_zone_center_deg") is not None
            else None
        ),
        target_activity_override_provider=(
            _oracle_target_activity_override_provider(frame_states)
            if method_spec.target_activity_mode == "oracle_target_activity"
            else None
        ),
        oracle_noise_frame_provider=(
            None
            if oracle_noise_audio is None
            else _oracle_noise_frame_provider(oracle_noise_audio, max(1, int(round(sample_rate * (FAST_FRAME_MS / 1000.0)))))
        ),
    )
    reference_dir = run_dir / "_oracle_target_reference"
    reference_runtime_summary = run_offline_session_pipeline(
        req=req,
        mic_audio=np.asarray(oracle_target_mic_audio, dtype=np.float32),
        mic_geometry_xyz=np.asarray(mic_pos, dtype=np.float64),
        out_dir=reference_dir,
        capture_trace=False,
        srp_override_provider=_oracle_srp_override_provider(frame_states),
        initial_focus_direction_deg=(
            float(metadata_payload.get("target_zone_center_deg"))
            if method_spec.focus_zone_target and metadata_payload.get("target_zone_center_deg") is not None
            else None
        ),
        target_activity_override_provider=(
            _oracle_target_activity_override_provider(frame_states)
            if method_spec.target_activity_mode == "oracle_target_activity"
            else None
        ),
        oracle_noise_frame_provider=(
            None
            if oracle_noise_audio is None
            else _oracle_noise_frame_provider(oracle_noise_audio, max(1, int(round(sample_rate * (FAST_FRAME_MS / 1000.0)))))
        ),
    )
    elapsed_s = max(perf_counter() - t0, 1e-9)
    processed, proc_sr = sf.read(run_dir / "enhanced_fast_path.wav", dtype="float32", always_2d=False)
    processed_audio = np.asarray(processed, dtype=np.float32).reshape(-1)
    oracle_ref_audio, oracle_ref_sr = sf.read(reference_dir / "enhanced_fast_path.wav", dtype="float32", always_2d=False)
    oracle_ref_audio = np.asarray(oracle_ref_audio, dtype=np.float32).reshape(-1)
    if int(proc_sr) != int(sample_rate):
        raise ValueError(f"Unexpected sample rate {proc_sr}, expected {sample_rate}")
    if int(oracle_ref_sr) != int(sample_rate):
        raise ValueError(f"Unexpected oracle reference sample rate {oracle_ref_sr}, expected {sample_rate}")
    sf.write(run_dir / "enhanced.wav", processed_audio, sample_rate)
    sf.write(run_dir / "clean_active_target.wav", oracle_ref_audio, sample_rate)

    overall = compute_metric_bundle(
        clean_ref=np.asarray(oracle_ref_audio, dtype=np.float64),
        raw_audio=np.asarray(raw_mix, dtype=np.float64),
        processed_audio=np.asarray(processed_audio, dtype=np.float64),
        sample_rate=sample_rate,
    )
    speech_metrics = _masked_metric_dict(
        clean_ref=oracle_ref_audio,
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
        "fd_noise_covariance_mode": str(params["fd_noise_covariance_mode"]),
        "aggression_level": str(method_spec.aggression_level),
        "postfilter_preset": str(method_spec.postfilter_preset),
        "focus_zone_target": bool(method_spec.focus_zone_target),
        "postfilter_enabled": bool(params["postfilter_enabled"]),
        "postfilter_method": str(params["postfilter_method"]),
        "target_activity_detector_backend": str(params["target_activity_detector_backend"]),
        "split_runtime_mode": str(params["split_runtime_mode"]),
        "mvdr_hop_ms": int(params["mvdr_hop_ms"]),
        "target_activity_update_every_n_fast_frames": int(params["target_activity_update_every_n_fast_frames"]),
        "main_angle_deg": scene_meta["main_angle_deg"],
        "secondary_angle_deg": scene_meta["secondary_angle_deg"],
        "target_zone_center_deg": float(metadata_payload.get("target_zone_center_deg", float("nan"))),
        "target_zone_width_deg": float(metadata_payload.get("target_zone_width_deg", float("nan"))),
        "zone_target_speaker_id": int(target_speaker_id),
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
        "beamforming_rtf": float(runtime_summary.get("beamforming_rtf", float("nan"))),
        "postfilter_rtf": float(runtime_summary.get("postfilter_rtf", float("nan"))),
        "pipeline_rtf": float(runtime_summary.get("pipeline_rtf", float("nan"))),
        "interstage_queue_wait_p95_ms": float(runtime_summary.get("interstage_queue_wait_p95_ms", float("nan"))),
        "interstage_queue_depth_max": float(runtime_summary.get("interstage_queue_depth_max", float("nan"))),
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
        "noise_covariance_mode": str(method_spec.noise_covariance_mode),
        "aggression_level": str(method_spec.aggression_level),
        "reference_runtime_summary": reference_runtime_summary,
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
    parser.add_argument("--scene-family", default="babble_bootstrap", choices=sorted(SCENE_FAMILY_DEFAULTS.keys()))
    parser.add_argument("--scenes-root", default=None)
    parser.add_argument("--assets-root", default=None)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=list(SUPPORTED_MIC_ARRAY_PROFILES))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--regenerate-scenes", action="store_true")
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--bootstrap-noise-only-sec", type=float, default=None)
    parser.add_argument("--background-babble-count", type=int, default=None)
    parser.add_argument("--background-babble-gain-min", type=float, default=None)
    parser.add_argument("--background-babble-gain-max", type=float, default=None)
    parser.add_argument("--background-wham-gain", type=float, default=None)
    parser.add_argument("--target-zone-width-deg", type=float, default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--mvdr-hop-ms", type=int, default=DEFAULT_MVDR_HOP_MS)
    parser.add_argument("--fd-analysis-window-ms", type=float, default=DEFAULT_FD_ANALYSIS_WINDOW_MS)
    parser.add_argument(
        "--localization-backend",
        default="srp_phat_localization",
        choices=["srp_phat_localization", "capon_1src", "capon_multisrc", "capon_mvdr_refine_1src", "music_1src"],
    )
    parser.add_argument("--split-runtime-mode", default="monolithic", choices=["monolithic", "pipelined", "beamforming_only", "postfilter_only"])
    parser.add_argument("--postfilter-queue-max-frames", type=int, default=4)
    parser.add_argument("--postfilter-queue-drop-oldest", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fd-cov-ema-alpha", type=float, default=None)
    parser.add_argument("--fd-diag-load", type=float, default=None)
    parser.add_argument("--fd-noise-covariance-mode", default=None, choices=["estimated_target_subtractive", "oracle_non_target_residual"])
    parser.add_argument("--target-activity-low-threshold", type=float, default=None)
    parser.add_argument("--target-activity-high-threshold", type=float, default=None)
    parser.add_argument("--target-activity-enter-frames", type=int, default=None)
    parser.add_argument("--target-activity-exit-frames", type=int, default=None)
    parser.add_argument("--fd-cov-update-scale-target-active", type=float, default=None)
    parser.add_argument("--fd-cov-update-scale-target-inactive", type=float, default=None)
    parser.add_argument("--target-activity-detector-mode", default=None)
    parser.add_argument("--target-activity-detector-backend", default="webrtc_fused", choices=["webrtc_fused", "silero_fused"])
    parser.add_argument("--target-activity-update-every-n-fast-frames", type=int, default=DEFAULT_TARGET_ACTIVITY_UPDATE_EVERY_N_FAST_FRAMES)
    parser.add_argument("--target-activity-blocker-offset-deg", type=float, default=None)
    parser.add_argument("--target-activity-bootstrap-only-calibration", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--target-activity-ratio-floor-db", type=float, default=None)
    parser.add_argument("--target-activity-ratio-active-db", type=float, default=None)
    parser.add_argument("--target-activity-target-rms-floor-scale", type=float, default=None)
    parser.add_argument("--target-activity-blocker-rms-floor-scale", type=float, default=None)
    parser.add_argument("--target-activity-speech-weight", type=float, default=None)
    parser.add_argument("--target-activity-ratio-weight", type=float, default=None)
    parser.add_argument("--target-activity-blocker-weight", type=float, default=None)
    parser.add_argument("--target-activity-vad-mode", type=int, default=None)
    parser.add_argument("--target-activity-vad-hangover-frames", type=int, default=None)
    parser.add_argument("--target-activity-noise-floor-rise-alpha", type=float, default=None)
    parser.add_argument("--target-activity-noise-floor-fall-alpha", type=float, default=None)
    parser.add_argument("--target-activity-noise-floor-margin-scale", type=float, default=None)
    parser.add_argument("--target-activity-rms-scale", type=float, default=None)
    parser.add_argument("--target-activity-score-exponent", type=float, default=None)
    parser.add_argument("--postfilter-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--postfilter-noise-ema-alpha", type=float, default=None)
    parser.add_argument("--postfilter-speech-ema-alpha", type=float, default=None)
    parser.add_argument("--postfilter-gain-floor", type=float, default=None)
    parser.add_argument("--postfilter-gain-ema-alpha", type=float, default=None)
    parser.add_argument("--postfilter-dd-alpha", type=float, default=None)
    parser.add_argument("--postfilter-noise-update-speech-scale", type=float, default=None)
    parser.add_argument("--postfilter-freq-smoothing-bins", type=int, default=None)
    parser.add_argument("--postfilter-gain-max-step-db", type=float, default=None)
    parser.add_argument("--rnnoise-wet-mix", type=float, default=None)
    parser.add_argument("--rnnoise-input-gain-db", type=float, default=None)
    parser.add_argument("--coherence-wiener-gain-floor", type=float, default=None)
    parser.add_argument("--coherence-wiener-coherence-exponent", type=float, default=None)
    parser.add_argument("--coherence-wiener-temporal-alpha", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    family_defaults = SCENE_FAMILY_DEFAULTS[str(args.scene_family)]
    scenes_root = Path(str(args.scenes_root) if args.scenes_root is not None else family_defaults["scenes_root"])
    assets_root = Path(str(args.assets_root) if args.assets_root is not None else family_defaults["assets_root"])
    duration_sec = float(args.duration_sec) if args.duration_sec is not None else float(family_defaults["duration_sec"])
    bootstrap_noise_only_sec = (
        float(args.bootstrap_noise_only_sec)
        if args.bootstrap_noise_only_sec is not None
        else float(family_defaults["bootstrap_noise_only_sec"])
    )
    background_babble_count = (
        int(args.background_babble_count)
        if args.background_babble_count is not None
        else int(family_defaults["background_babble_count"])
    )
    background_babble_gain_min = (
        float(args.background_babble_gain_min)
        if args.background_babble_gain_min is not None
        else float(family_defaults["background_babble_gain_min"])
    )
    background_babble_gain_max = (
        float(args.background_babble_gain_max)
        if args.background_babble_gain_max is not None
        else float(family_defaults["background_babble_gain_max"])
    )
    background_wham_gain = (
        float(args.background_wham_gain)
        if args.background_wham_gain is not None
        else float(family_defaults["background_wham_gain"])
    )
    target_zone_width_deg = (
        float(args.target_zone_width_deg)
        if args.target_zone_width_deg is not None
        else float(family_defaults.get("target_zone_width_deg", DEFAULT_ZONE_WIDTH_DEG))
    )
    if args.regenerate_scenes or not list(scenes_root.glob("*.json")):
        generator_kwargs = dict(
            config_root=scenes_root,
            asset_root=assets_root,
            duration_sec=duration_sec,
            bootstrap_noise_only_sec=bootstrap_noise_only_sec,
            background_babble_count=background_babble_count,
            background_babble_gain_min=background_babble_gain_min,
            background_babble_gain_max=background_babble_gain_max,
            background_wham_gain=background_wham_gain,
            manifest_path=args.manifest_path,
        )
        if str(args.scene_family) == "zone_overlap":
            generator_kwargs["target_zone_width_deg"] = target_zone_width_deg
        family_defaults["generator"](**generator_kwargs)

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    scenes = sorted(scenes_root.glob("*.json"))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    if not scenes:
        raise FileNotFoundError(f"No scenes found under {args.scenes_root}")

    detector_overrides = _detector_param_overrides(args)
    detector_backend = str(args.target_activity_detector_backend)
    if detector_backend not in DEFAULT_TARGET_ACTIVITY_PRESETS:
        raise ValueError(f"Unsupported detector backend for defaults: {detector_backend}")
    base_detector_params = dict(DEFAULT_TARGET_ACTIVITY_PRESETS[detector_backend])
    params = {
        "mvdr_hop_ms": int(args.mvdr_hop_ms),
        "fd_analysis_window_ms": float(args.fd_analysis_window_ms),
        "localization_backend": str(args.localization_backend),
        "split_runtime_mode": str(args.split_runtime_mode),
        "postfilter_queue_max_frames": int(args.postfilter_queue_max_frames),
        "postfilter_queue_drop_oldest": bool(args.postfilter_queue_drop_oldest),
        "fd_noise_covariance_mode": "estimated_target_subtractive",
        **base_detector_params,
        **DEFAULT_POSTFILTER_PRESETS["no_pf"],
        **detector_overrides,
        "scene_family": str(args.scene_family),
        "bootstrap_noise_only_sec": bootstrap_noise_only_sec,
        "background_babble_count": background_babble_count,
        "background_babble_gain_min": background_babble_gain_min,
        "background_babble_gain_max": background_babble_gain_max,
        "background_wham_gain": background_wham_gain,
        "focus_direction_match_window_deg": target_zone_width_deg,
        "focus_target_hold_frames": 8,
        "multi_target_max_speakers": 2,
        "multi_target_hold_frames": 12,
        "multi_target_min_confidence": 0.2,
        "multi_target_min_activity": 0.15,
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
                **(
                    DEFAULT_POSTFILTER_PRESETS.get(METHOD_SPECS[str(method)].postfilter_preset, {})
                    if str(method) in METHOD_SPECS
                    else {}
                ),
                **(
                    DEFAULT_TARGET_ACTIVITY_PRESETS["silero_fused"]
                    if str(method) in {
                        "mvdr_fd_bootstrap_estimated_activity_silero",
                        "mvdr_fd_silero_no_pf",
                        "mvdr_fd_silero_pf_mild",
                        "mvdr_fd_silero_pf_medium",
                        "mvdr_fd_zone_target_estimated_activity",
                        "lcmv_top2_tracked_estimated",
                    }
                    else {}
                ),
                **(
                    DEFAULT_ORACLE_NOISE_PRESETS[str(method).replace("mvdr_fd_bootstrap_", "")]
                    if str(method).startswith("mvdr_fd_bootstrap_oracle_noise_")
                    else {}
                ),
                **detector_overrides,
                "target_activity_detector_backend": (
                    "silero_fused"
                    if str(method) in {"mvdr_fd_bootstrap_estimated_activity_silero", "mvdr_fd_zone_target_estimated_activity", "lcmv_top2_tracked_estimated"} or str(method).startswith("mvdr_fd_bootstrap_oracle_noise_")
                    else str(params["target_activity_detector_backend"])
                ),
                "fd_noise_covariance_mode": (
                    "oracle_non_target_residual"
                    if str(method).startswith("mvdr_fd_bootstrap_oracle_noise_")
                    else str(params["fd_noise_covariance_mode"])
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
