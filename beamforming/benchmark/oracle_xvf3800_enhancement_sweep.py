from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mic_array_forwarder.models import SessionStartRequest
from realtime_pipeline.contracts import SRPPeakSnapshot
from realtime_pipeline.session_runtime import run_offline_session_pipeline
from simulation.mic_array_profiles import SUPPORTED_MIC_ARRAY_PROFILES, mic_positions_xyz
from simulation.simulation_config import MicrophoneArray, SimulationConfig


DEFAULT_SCENES_ROOT = Path("simulation/simulations/configs/testing_specific_angles")
DEFAULT_ASSETS_ROOT = Path("simulation/simulations/assets/testing_specific_angles")
DEFAULT_OUT_ROOT = Path("beamforming/benchmark/oracle_xvf3800_enhancement")
DEFAULT_PROFILE = "respeaker_xvf3800_0650"
DEFAULT_METHODS = [
    "mvdr_fd",
    "delay_sum",
    "lcmv_target_only",
    "lcmv_null",
    "mvdr_fd_rnnoise",
    "mvdr_fd_coherence_wiener",
]
DEFAULT_NOISE_GAIN_SCALES = [0.2, 0.4, 0.6, 0.8, 1.0]
FAST_FRAME_MS = 10
FD_ANALYSIS_WINDOW_MS = 40.0
NULL_USER_SPEAKER_ID = 0
NULL_CONFLICT_DEG = 30.0


@dataclass(frozen=True)
class OracleFrameState:
    frame_index: int
    timestamp_ms: float
    target_speaker_id: int | None
    target_doa_deg: float | None
    target_activity_score: float
    target_active: bool
    null_user_speaker_id: int | None
    null_user_doa_deg: float | None
    force_suppression_active: bool
    null_candidate: bool
    null_fallback: bool
    peaks_deg: tuple[float, ...]
    peak_scores: tuple[float, ...]


@dataclass(frozen=True)
class JobResult:
    row: dict[str, object]
    run_summary: dict[str, object]


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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _slug(value: str) -> str:
    return str(value).strip().lower().replace("/", "_").replace(" ", "_")


def _normalize_deg(value: float) -> float:
    return float(value % 360.0)


def _angular_error_deg(a_deg: float, b_deg: float) -> float:
    return float(abs((float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0))


def _scale_noise_gains(sim_cfg: SimulationConfig, noise_gain_scale: float) -> SimulationConfig:
    if abs(float(noise_gain_scale) - 1.0) < 1e-12:
        return sim_cfg
    for source in sim_cfg.audio.sources:
        if str(getattr(source, "classification", "")).strip().lower() == "noise":
            source.gain = float(source.gain) * float(noise_gain_scale)
    return sim_cfg


def _stage_scene(
    scene_path: Path,
    staging_root: Path,
    profile: str,
    assets_root: Path,
    noise_gain_scale: float,
    stage_key: str,
) -> tuple[Path, Path]:
    stage_dir = staging_root / scene_path.stem / f"noise_{str(noise_gain_scale).replace('.', 'p')}" / _slug(stage_key)
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_scene = stage_dir / scene_path.name
    sim_cfg = SimulationConfig.from_file(scene_path)
    sim_cfg = _scale_noise_gains(sim_cfg, float(noise_gain_scale))
    rel_positions = mic_positions_xyz(profile)
    sim_cfg.microphone_array = MicrophoneArray(
        mic_center=list(sim_cfg.microphone_array.mic_center),
        mic_radius=float(np.max(np.linalg.norm(rel_positions[:, :2], axis=1))),
        mic_count=int(rel_positions.shape[0]),
        mic_positions=rel_positions.tolist(),
    )
    sim_cfg.to_file(staged_scene)

    asset_dir = assets_root / scene_path.stem
    metadata_path = stage_dir / "scenario_metadata.json"
    for filename in ("scenario_metadata.json", "metrics_summary.json", "frame_ground_truth.csv"):
        src = asset_dir / filename
        if src.exists():
            shutil.copy2(src, stage_dir / filename)
    return staged_scene, metadata_path


def _infer_noise_layout_type(noise_angles: list[float]) -> str:
    if len(noise_angles) <= 1:
        return "single"
    sorted_angles = sorted(_normalize_deg(v) for v in noise_angles)
    if len(sorted_angles) == 2:
        diff = abs(((sorted_angles[1] - sorted_angles[0]) + 180.0) % 360.0 - 180.0)
        if abs(diff - 180.0) <= 1.0:
            return "opposite_pair"
    return f"custom_{len(sorted_angles)}"


def _load_scene_metadata(scene_id: str, metadata_path: Path) -> dict[str, object]:
    if not metadata_path.exists():
        return {
            "scene_id": scene_id,
            "scene_layout_family": scene_id,
            "main_angle_deg": None,
            "secondary_angle_deg": None,
            "noise_layout_type": None,
            "noise_angles_deg": [],
        }
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    speech_rows = list(payload.get("assets", {}).get("speech", []))
    noise_rows = list(payload.get("assets", {}).get("noise", []))
    speech_angles = [float(row["angle_deg"]) for row in speech_rows if row.get("angle_deg") is not None]
    noise_angles = [float(row["angle_deg"]) for row in noise_rows if row.get("angle_deg") is not None]
    main_angle = payload.get("main_angle_deg")
    secondary_angle = payload.get("secondary_angle_deg")
    if main_angle is None and speech_angles:
        main_angle = float(speech_angles[0])
    if secondary_angle is None and len(speech_angles) > 1:
        secondary_angle = float(speech_angles[1])
    noise_layout_type = payload.get("noise_layout_type")
    if noise_layout_type is None and noise_angles:
        noise_layout_type = _infer_noise_layout_type(noise_angles)
    family_bits = [scene_id]
    if noise_layout_type:
        family_bits.append(str(noise_layout_type))
    if main_angle is not None and secondary_angle is not None:
        family_bits.append(f"{int(round(float(main_angle)))}_{int(round(float(secondary_angle)))}")
    return {
        "scene_id": scene_id,
        "scene_layout_family": "__".join(family_bits),
        "main_angle_deg": None if main_angle is None else float(main_angle),
        "secondary_angle_deg": None if secondary_angle is None else float(secondary_angle),
        "noise_layout_type": noise_layout_type,
        "noise_angles_deg": [float(v) for v in noise_angles],
    }


def _speaker_source_index_map(sim_cfg: SimulationConfig, metadata: dict[str, object]) -> dict[int, int]:
    speech_source_indices = [
        idx
        for idx, source in enumerate(sim_cfg.audio.sources)
        if str(getattr(source, "classification", "")).strip().lower() == "speech"
    ]
    speech_rows = sorted(
        list(metadata.get("assets", {}).get("speech", [])),
        key=lambda row: (
            float(row.get("active_window_sec", [0.0, 0.0])[0]) if row.get("active_window_sec") else 0.0,
            int(row.get("speaker_id", 0)),
        ),
    )
    mapping: dict[int, int] = {}
    for source_idx, row in zip(speech_source_indices, speech_rows):
        mapping[int(row.get("speaker_id", len(mapping)))] = int(source_idx)
    return mapping


def _speaker_doa_map(metadata: dict[str, object]) -> dict[int, float]:
    doa_by_speaker: dict[int, float] = {}
    for row in metadata.get("assets", {}).get("speech", []):
        if row.get("speaker_id") is None or row.get("angle_deg") is None:
            continue
        doa_by_speaker[int(row["speaker_id"])] = float(row["angle_deg"])
    return doa_by_speaker


def _build_active_target_schedule(
    metadata: dict[str, object],
    *,
    sample_rate: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    active_speaker_ids = np.full(n_samples, -1, dtype=np.int32)
    active_doa_deg = np.full(n_samples, np.nan, dtype=np.float64)
    speech_rows = sorted(
        list(metadata.get("assets", {}).get("speech", [])),
        key=lambda row: (
            float(row.get("active_window_sec", [0.0, 0.0])[0]) if row.get("active_window_sec") else 0.0,
            int(row.get("speaker_id", 0)),
        ),
    )
    events: list[tuple[float, float, int, float]] = []
    for row in speech_rows:
        window = row.get("active_window_sec")
        if not isinstance(window, list) or len(window) < 2:
            continue
        start_sec = max(0.0, float(window[0]))
        end_sec = max(start_sec, float(window[1]))
        doa_deg = float(row.get("angle_deg", math.nan))
        events.append((start_sec, end_sec, int(row.get("speaker_id", -1)), doa_deg))
    for start_sec, end_sec, speaker_id, doa_deg in events:
        start = int(round(start_sec * sample_rate))
        end = min(n_samples, int(round(end_sec * sample_rate)))
        if end <= start:
            continue
        active_speaker_ids[start:end] = int(speaker_id)
        active_doa_deg[start:end] = float(doa_deg)
    return active_speaker_ids, active_doa_deg


def _build_clean_reference(
    source_signals: list[np.ndarray],
    speaker_to_source_idx: dict[int, int],
    active_speaker_ids: np.ndarray,
) -> np.ndarray:
    n_samples = int(active_speaker_ids.shape[0])
    clean_ref = np.zeros(n_samples, dtype=np.float64)
    for speaker_id, source_idx in speaker_to_source_idx.items():
        mask = active_speaker_ids == int(speaker_id)
        if not np.any(mask):
            continue
        source = np.asarray(source_signals[int(source_idx)], dtype=np.float64).reshape(-1)
        if source.shape[0] < n_samples:
            source = np.pad(source, (0, n_samples - source.shape[0]))
        clean_ref[mask] = source[:n_samples][mask]
    return clean_ref


def _sample_last_finite(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(finite[-1])


def _sample_last_speaker_id(values: np.ndarray) -> int | None:
    arr = np.asarray(values, dtype=np.int32)
    valid = arr[arr >= 0]
    if valid.size == 0:
        return None
    return int(valid[-1])


def _build_oracle_frame_states(
    *,
    active_speaker_ids: np.ndarray,
    active_doa_deg: np.ndarray,
    sample_rate: int,
    fast_frame_ms: int,
    null_user_speaker_id: int | None = None,
    null_user_doa_deg: float | None = None,
    null_conflict_deg: float = NULL_CONFLICT_DEG,
) -> list[OracleFrameState]:
    frame_samples = max(1, int(round(sample_rate * (float(fast_frame_ms) / 1000.0))))
    n_frames = int(math.ceil(active_speaker_ids.shape[0] / frame_samples))
    states: list[OracleFrameState] = []
    for frame_idx in range(n_frames):
        start = frame_idx * frame_samples
        end = min(active_speaker_ids.shape[0], start + frame_samples)
        target_activity_score = float(np.mean(active_speaker_ids[start:end] >= 0)) if end > start else 0.0
        target_speaker_id = _sample_last_speaker_id(active_speaker_ids[start:end])
        target_doa_deg = _sample_last_finite(active_doa_deg[start:end])
        null_candidate = False
        null_fallback = False
        force_suppression_active = False
        peaks: list[float] = []
        scores: list[float] = []
        if target_doa_deg is not None:
            peaks.append(float(target_doa_deg))
            scores.append(1.0)
        if (
            null_user_speaker_id is not None
            and null_user_doa_deg is not None
            and target_doa_deg is not None
            and target_speaker_id is not None
        ):
            if int(target_speaker_id) == int(null_user_speaker_id):
                null_fallback = True
                force_suppression_active = False
            else:
                null_candidate = True
                if _angular_error_deg(float(target_doa_deg), float(null_user_doa_deg)) >= float(null_conflict_deg):
                    peaks.append(float(null_user_doa_deg))
                    scores.append(0.99)
                    force_suppression_active = True
                else:
                    null_fallback = True
        states.append(
            OracleFrameState(
                frame_index=frame_idx,
                timestamp_ms=float(frame_idx * fast_frame_ms),
                target_speaker_id=target_speaker_id,
                target_doa_deg=target_doa_deg,
                target_activity_score=float(target_activity_score),
                target_active=bool(target_activity_score >= 0.5),
                null_user_speaker_id=null_user_speaker_id,
                null_user_doa_deg=null_user_doa_deg,
                force_suppression_active=bool(force_suppression_active),
                null_candidate=bool(null_candidate),
                null_fallback=bool(null_fallback),
                peaks_deg=tuple(float(v) for v in peaks),
                peak_scores=tuple(float(v) for v in scores),
            )
        )
    return states


def _oracle_srp_override_provider(frame_states: list[OracleFrameState]):
    snapshots = [
        SRPPeakSnapshot(
            timestamp_ms=float(state.timestamp_ms),
            peaks_deg=tuple(state.peaks_deg),
            peak_scores=tuple(state.peak_scores),
            raw_peaks_deg=tuple(state.peaks_deg),
            raw_peak_scores=tuple(state.peak_scores),
            debug={
                "oracle_override": True,
                "oracle_target_speaker_id": state.target_speaker_id,
                "oracle_target_doa_deg": state.target_doa_deg,
                "oracle_target_activity_score": float(state.target_activity_score),
                "oracle_target_active": bool(state.target_active),
                "oracle_null_user_speaker_id": state.null_user_speaker_id,
                "oracle_null_user_doa_deg": state.null_user_doa_deg,
                "force_suppression_active": bool(state.force_suppression_active),
                "oracle_null_candidate": bool(state.null_candidate),
                "oracle_null_fallback": bool(state.null_fallback),
            },
        )
        for state in frame_states
    ]

    def _provider(frame_index: int, _timestamp_ms: float) -> SRPPeakSnapshot | None:
        if 0 <= int(frame_index) < len(snapshots):
            return snapshots[int(frame_index)]
        return None

    return _provider


def _oracle_target_activity_override_provider(frame_states: list[OracleFrameState]):
    target_scores = [float(state.target_activity_score) for state in frame_states]

    def _provider(frame_index: int, _timestamp_ms: float) -> float | None:
        if 0 <= int(frame_index) < len(target_scores):
            return float(target_scores[int(frame_index)])
        return None

    return _provider


def _steering_vector(doa_deg: float, freqs_hz: np.ndarray, mic_geometry_xyz: np.ndarray) -> np.ndarray:
    mic_pos = np.asarray(mic_geometry_xyz, dtype=np.float64)
    if mic_pos.shape[0] == 3:
        mic_pos = mic_pos.T
    if mic_pos.shape[1] == 2:
        mic_pos = np.hstack([mic_pos, np.zeros((mic_pos.shape[0], 1), dtype=np.float64)])
    az = np.deg2rad(float(doa_deg))
    direction = np.array([np.cos(az), np.sin(az), 0.0], dtype=np.float64)
    tau = (mic_pos @ direction) / 343.0
    tau = tau - float(np.mean(tau))
    phase = -2j * np.pi * freqs_hz[:, None] * tau[None, :]
    return np.exp(phase)


def _coherence_wiener_postfilter(
    mic_audio: np.ndarray,
    active_doa_deg: np.ndarray,
    *,
    sample_rate: int,
    mic_geometry_xyz: np.ndarray,
    n_fft: int = 512,
    hop: int = 128,
    gain_floor: float = 0.12,
    coherence_exponent: float = 1.5,
    temporal_alpha: float = 0.65,
) -> np.ndarray:
    x = np.asarray(mic_audio, dtype=np.float64)
    noverlap = max(0, n_fft - hop)
    chan_specs = []
    f_hz = None
    t_s = None
    for idx in range(x.shape[1]):
        f_hz, t_s, zxx = stft(
            x[:, idx],
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=noverlap,
            boundary="zeros",
            padded=True,
        )
        chan_specs.append(zxx)
    X = np.stack(chan_specs, axis=-1)
    assert f_hz is not None and t_s is not None
    frame_idx = np.clip(np.round(t_s * sample_rate).astype(int), 0, len(active_doa_deg) - 1)
    doa_per_frame = active_doa_deg[frame_idx]
    valid_doa = doa_per_frame[np.isfinite(doa_per_frame)]
    fallback_doa = float(valid_doa[0]) if valid_doa.size else 0.0
    y_out = np.zeros((X.shape[0], X.shape[1]), dtype=np.complex128)
    prev_gain = np.ones(X.shape[0], dtype=np.float64)
    for t_idx in range(X.shape[1]):
        doa = float(doa_per_frame[t_idx]) if np.isfinite(doa_per_frame[t_idx]) else fallback_doa
        steering = _steering_vector(doa, f_hz, mic_geometry_xyz)
        aligned = X[:, t_idx, :] * np.conj(steering)
        y_ds = np.mean(aligned, axis=1)
        auto_psd = np.mean(np.abs(aligned) ** 2, axis=1)
        pair_terms = []
        for i in range(aligned.shape[1]):
            for j in range(i + 1, aligned.shape[1]):
                denom = np.sqrt(np.abs(aligned[:, i]) ** 2 * np.abs(aligned[:, j]) ** 2) + 1e-12
                pair_terms.append(np.abs(aligned[:, i] * np.conj(aligned[:, j])) / denom)
        coherence = np.mean(pair_terms, axis=0) if pair_terms else np.ones_like(auto_psd)
        coherence = np.clip(coherence, 0.0, 1.0)
        noise_psd = np.maximum((1.0 - coherence) * auto_psd, 1e-10)
        speech_psd = np.maximum(np.abs(y_ds) ** 2 - noise_psd, 1e-10)
        wiener = speech_psd / (speech_psd + noise_psd + 1e-10)
        gain = gain_floor + (1.0 - gain_floor) * ((coherence ** coherence_exponent) * wiener)
        gain = (temporal_alpha * prev_gain) + ((1.0 - temporal_alpha) * gain)
        prev_gain = gain
        y_out[:, t_idx] = y_ds * gain
    from scipy.signal import istft

    _, y = istft(
        y_out,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    if y.shape[0] < x.shape[0]:
        y = np.pad(y, (0, x.shape[0] - y.shape[0]))
    return np.asarray(y[: x.shape[0]], dtype=np.float32)


class _RNNoiseWrapper:
    def __init__(self) -> None:
        self._lib = None
        self._state = None
        self._py_backend = None
        self.backend_name = ""
        self.error = ""
        self._load()

    def _load(self) -> None:
        candidates = [
            os.environ.get("RNNOISE_LIB"),
            "librnnoise.so",
            "librnnoise.dylib",
            "rnnoise.dll",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                lib = ctypes.cdll.LoadLibrary(candidate)
                lib.rnnoise_create.restype = ctypes.c_void_p
                lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
                lib.rnnoise_process_frame.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                ]
                state = lib.rnnoise_create(None)
                if not state:
                    continue
                self._lib = lib
                self._state = state
                self.backend_name = str(candidate)
                return
            except OSError:
                continue
            except Exception as exc:  # pragma: no cover
                self.error = str(exc)
        try:
            from pyrnnoise import RNNoise as PyRNNoise

            self._py_backend = PyRNNoise(48000)
            self.backend_name = "pyrnnoise"
            self.error = ""
        except Exception as exc:  # pragma: no cover
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return (self._lib is not None and self._state is not None) or (self._py_backend is not None)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.available:
            raise RuntimeError(self.error or "RNNoise unavailable")
        target_sr = 48000
        x = np.asarray(audio, dtype=np.float32)
        if sample_rate != target_sr:
            x_48k = resample_poly(x, target_sr, sample_rate).astype(np.float32, copy=False)
        else:
            x_48k = x
        frame_len = 480
        if x_48k.shape[0] % frame_len:
            x_48k = np.pad(x_48k, (0, frame_len - (x_48k.shape[0] % frame_len)))
        if self._py_backend is not None:
            parts = []
            for _vad, den in self._py_backend.denoise_chunk(np.asarray(x_48k, dtype=np.float32)):
                den_arr = np.asarray(den, dtype=np.float32).reshape(-1)
                if den_arr.dtype.kind in {"i", "u"} or np.max(np.abs(den_arr)) > 1.5:
                    den_arr = den_arr / 32768.0
                parts.append(den_arr)
            out = np.concatenate(parts).astype(np.float32, copy=False)
        else:
            out = np.zeros_like(x_48k, dtype=np.float32)
            for start in range(0, x_48k.shape[0], frame_len):
                frame = np.ascontiguousarray(x_48k[start : start + frame_len], dtype=np.float32)
                out_frame = np.zeros(frame_len, dtype=np.float32)
                self._lib.rnnoise_process_frame(
                    ctypes.c_void_p(self._state),
                    out_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )
                out[start : start + frame_len] = out_frame
        out = out[: x_48k.shape[0]]
        if sample_rate != target_sr:
            out = resample_poly(out, sample_rate, target_sr).astype(np.float32, copy=False)
        if out.shape[0] < audio.shape[0]:
            out = np.pad(out, (0, audio.shape[0] - out.shape[0]))
        return np.asarray(out[: audio.shape[0]], dtype=np.float32)

    def close(self) -> None:
        if self._lib is not None and self._state is not None:
            try:
                self._lib.rnnoise_destroy(ctypes.c_void_p(self._state))
            except Exception:
                pass
            self._state = None


def _build_metric_bundle(
    *,
    clean_ref: np.ndarray,
    raw_audio: np.ndarray,
    processed_audio: np.ndarray,
    sample_rate: int,
):
    from beamforming.benchmark.metrics import compute_metric_bundle

    return compute_metric_bundle(
        clean_ref=np.asarray(clean_ref, dtype=np.float64),
        raw_audio=np.asarray(raw_audio, dtype=np.float64),
        processed_audio=np.asarray(processed_audio, dtype=np.float64),
        sample_rate=sample_rate,
    )


def _shared_method_name(method: str) -> str:
    if method in {"mvdr_fd", "delay_sum", "lcmv_target_only", "lcmv_null"}:
        return method
    if method in {"mvdr_fd_rnnoise", "mvdr_fd_coherence_wiener"}:
        return "mvdr_fd"
    raise ValueError(f"Unsupported method: {method}")


def _build_session_request(
    *,
    method: str,
    sample_rate: int,
    channel_count: int,
    profile: str,
    null_user_doa_deg: float | None,
    target_activity_rnn_update_mode: str | None,
) -> SessionStartRequest:
    shared_method = _shared_method_name(method)
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
            "own_voice_suppression_mode": "lcmv_null_hysteresis" if shared_method == "lcmv_null" else "off",
            "suppressed_user_voice_doa_deg": None if shared_method != "lcmv_null" else null_user_doa_deg,
            "suppressed_user_match_window_deg": 33.0,
            "suppressed_user_null_on_frames": 1,
            "suppressed_user_null_off_frames": 1,
            "suppressed_user_gate_attenuation_db": 18.0,
            "suppressed_user_target_conflict_deg": float(NULL_CONFLICT_DEG),
            "assume_single_speaker": False,
            "localization_backend": "srp_phat_localization",
            "beamforming_mode": "mvdr_fd" if shared_method in {"mvdr_fd", "lcmv_target_only", "lcmv_null"} else "delay_sum",
            "fd_analysis_window_ms": float(FD_ANALYSIS_WINDOW_MS),
            "target_activity_rnn_update_mode": target_activity_rnn_update_mode,
            "postfilter_enabled": method not in {"mvdr_fd", "delay_sum", "lcmv_target_only", "lcmv_null"},
            "output_normalization_enabled": method not in {"mvdr_fd", "delay_sum", "lcmv_target_only", "lcmv_null"},
            "output_allow_amplification": False,
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


def _run_shared_oracle_method(
    *,
    method: str,
    mic_audio: np.ndarray,
    mic_geometry_xyz: np.ndarray,
    sample_rate: int,
    profile: str,
    out_dir: Path,
    active_speaker_ids: np.ndarray,
    active_doa_deg: np.ndarray,
    null_user_doa_deg: float | None,
    capture_trace: bool = True,
) -> tuple[np.ndarray, dict[str, object], list[OracleFrameState]]:
    is_lcmv = _shared_method_name(method) == "lcmv_null"
    frame_states = _build_oracle_frame_states(
        active_speaker_ids=active_speaker_ids,
        active_doa_deg=active_doa_deg,
        sample_rate=sample_rate,
        fast_frame_ms=FAST_FRAME_MS,
        null_user_speaker_id=(NULL_USER_SPEAKER_ID if is_lcmv else None),
        null_user_doa_deg=(null_user_doa_deg if is_lcmv else None),
        null_conflict_deg=float(NULL_CONFLICT_DEG),
    )
    req = _build_session_request(
        method=method,
        sample_rate=sample_rate,
        channel_count=int(mic_audio.shape[1]),
        profile=profile,
        null_user_doa_deg=null_user_doa_deg,
        target_activity_rnn_update_mode=("oracle_target_activity" if _shared_method_name(method) in {"mvdr_fd", "lcmv_target_only", "lcmv_null"} else None),
    )
    summary = run_offline_session_pipeline(
        req=req,
        mic_audio=np.asarray(mic_audio, dtype=np.float32),
        mic_geometry_xyz=np.asarray(mic_geometry_xyz, dtype=np.float64),
        out_dir=out_dir,
        capture_trace=bool(capture_trace),
        srp_override_provider=_oracle_srp_override_provider(frame_states),
        target_activity_override_provider=_oracle_target_activity_override_provider(frame_states),
    )
    processed, proc_sr = sf.read(out_dir / "enhanced_fast_path.wav", dtype="float32", always_2d=False)
    processed = np.asarray(processed, dtype=np.float32).reshape(-1)
    if int(proc_sr) != int(sample_rate):
        raise ValueError(f"Unexpected sample rate {proc_sr}, expected {sample_rate}")
    return processed, summary, frame_states


def _postprocess_variant(
    *,
    method: str,
    base_audio: np.ndarray,
    mic_audio: np.ndarray,
    active_doa_deg: np.ndarray,
    sample_rate: int,
    mic_geometry_xyz: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, object]]:
    meta: dict[str, object] = {"status": "ok", "postfilter_name": ""}
    if method == "mvdr_fd_rnnoise":
        rnnoise = _RNNoiseWrapper()
        try:
            if not rnnoise.available:
                meta["status"] = "unavailable"
                meta["rnnoise_backend"] = ""
                meta["rnnoise_error"] = rnnoise.error
                return None, meta
            processed = rnnoise.process(base_audio, sample_rate)
            meta["rnnoise_backend"] = rnnoise.backend_name
            meta["postfilter_name"] = "rnnoise"
            return processed, meta
        finally:
            rnnoise.close()
    if method == "mvdr_fd_coherence_wiener":
        processed = _coherence_wiener_postfilter(
            mic_audio=np.asarray(mic_audio, dtype=np.float32),
            active_doa_deg=np.asarray(active_doa_deg, dtype=np.float64),
            sample_rate=sample_rate,
            mic_geometry_xyz=np.asarray(mic_geometry_xyz, dtype=np.float64),
        )
        meta["postfilter_name"] = "coherence_wiener"
        return processed, meta
    return np.asarray(base_audio, dtype=np.float32), meta


def _compare_text(summary_by_method: list[dict[str, object]], baseline_method: str = "delay_sum") -> str:
    baseline = next((row for row in summary_by_method if row["method"] == baseline_method), None)
    if baseline is None:
        return "No baseline summary available.\n"
    baseline_snr = float(baseline.get("delta_snr_db_mean", math.nan))
    baseline_sii = float(baseline.get("delta_sii_mean", math.nan))
    lines = []
    for row in summary_by_method:
        method = str(row["method"])
        if method == baseline_method:
            lines.append(f"{method}: baseline")
            continue
        snr = float(row.get("delta_snr_db_mean", math.nan))
        sii = float(row.get("delta_sii_mean", math.nan))
        snr_cmp = "helps" if snr > baseline_snr else "hurts"
        sii_cmp = "helps" if sii > baseline_sii else "hurts"
        lines.append(f"{method}: {snr_cmp} delta_snr_db, {sii_cmp} delta_sii relative to {baseline_method}")
    return "\n".join(lines) + "\n"


def _mean_or_nan(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _aggregate(rows: list[dict[str, object]], group_fields: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("status", "ok")) != "ok":
            continue
        key = tuple(row.get(field) for field in group_fields)
        grouped[key].append(row)
    out_rows: list[dict[str, object]] = []
    for key, items in sorted(grouped.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        out_row = {field: value for field, value in zip(group_fields, key)}
        out_row.update(
            {
                "n_runs": len(items),
                "delta_snr_db_mean": _mean_or_nan([float(item["delta_snr_db"]) for item in items]),
                "delta_sii_mean": _mean_or_nan([float(item["delta_sii"]) for item in items]),
                "delta_si_sdr_db_mean": _mean_or_nan([float(item["delta_si_sdr_db"]) for item in items]),
                "snr_db_processed_mean": _mean_or_nan([float(item["snr_db_processed"]) for item in items]),
                "sii_processed_mean": _mean_or_nan([float(item["sii_processed"]) for item in items]),
                "si_sdr_db_processed_mean": _mean_or_nan([float(item["si_sdr_db_processed"]) for item in items]),
                "rtf_mean": _mean_or_nan([float(item["rtf"]) for item in items]),
                "null_applied_fraction_mean": _mean_or_nan([float(item["null_applied_fraction"]) for item in items]),
                "null_fallback_fraction_mean": _mean_or_nan([float(item["null_fallback_fraction"]) for item in items]),
            }
        )
        out_rows.append(out_row)
    return out_rows


def _plot_noise_sweep(summary_rows: list[dict[str, object]], out_path: Path, metric_key: str, title: str) -> None:
    if not summary_rows:
        return
    plt.figure(figsize=(10, 4))
    methods = sorted({str(row["method"]) for row in summary_rows})
    for method in methods:
        method_rows = sorted(
            [row for row in summary_rows if str(row["method"]) == method],
            key=lambda row: float(row["noise_gain_scale"]),
        )
        xs = [float(row["noise_gain_scale"]) for row in method_rows]
        ys = [float(row[metric_key]) for row in method_rows]
        plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("noise_gain_scale")
    plt.ylabel(metric_key)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_scene_layout(summary_rows: list[dict[str, object]], out_path: Path, metric_key: str, title: str) -> None:
    if not summary_rows:
        return
    labels = [f"{row['method']}\n{row['scene_layout_family']}" for row in summary_rows]
    vals = [float(row[metric_key]) for row in summary_rows]
    plt.figure(figsize=(max(10, len(labels) * 0.7), 4.5))
    plt.bar(np.arange(len(labels)), vals)
    plt.xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel(metric_key)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _run_job(
    *,
    scene_path: str,
    assets_root: str,
    out_root: str,
    profile: str,
    noise_gain_scale: float,
    method: str,
) -> JobResult:
    scene_cfg_path = Path(scene_path)
    root = Path(out_root)
    scene_id = scene_cfg_path.stem
    staged_scene, staged_metadata = _stage_scene(
        scene_cfg_path,
        root / "_staged_scenes",
        profile,
        Path(assets_root),
        float(noise_gain_scale),
        stage_key=method,
    )
    sim_cfg = SimulationConfig.from_file(staged_scene)
    metadata_payload = json.loads(staged_metadata.read_text(encoding="utf-8")) if staged_metadata.exists() else {}
    scene_meta = _load_scene_metadata(scene_id, staged_metadata)

    from simulation.simulator import run_simulation

    mic_audio, mic_pos, source_signals = run_simulation(sim_cfg)
    sample_rate = int(sim_cfg.audio.fs)
    active_speaker_ids, active_doa_deg = _build_active_target_schedule(
        metadata_payload,
        sample_rate=sample_rate,
        n_samples=int(mic_audio.shape[0]),
    )
    speaker_to_source_idx = _speaker_source_index_map(sim_cfg, metadata_payload)
    doa_by_speaker = _speaker_doa_map(metadata_payload)
    null_user_doa_deg = doa_by_speaker.get(int(NULL_USER_SPEAKER_ID))
    clean_ref = _build_clean_reference(source_signals, speaker_to_source_idx, active_speaker_ids)
    raw_mix = np.mean(np.asarray(mic_audio, dtype=np.float64), axis=1).astype(np.float32, copy=False)

    run_dir = root / "runs" / scene_id / f"noise_{str(noise_gain_scale).replace('.', 'p')}" / _slug(method)
    run_dir.mkdir(parents=True, exist_ok=True)
    sf.write(run_dir / "raw_mix_mean.wav", raw_mix, sample_rate)
    sf.write(run_dir / "clean_active_target.wav", clean_ref.astype(np.float32), sample_rate)

    t0 = perf_counter()
    shared_audio, shared_summary, frame_states = _run_shared_oracle_method(
        method=method,
        mic_audio=np.asarray(mic_audio, dtype=np.float32),
        mic_geometry_xyz=np.asarray(mic_pos, dtype=np.float64),
        sample_rate=sample_rate,
        profile=profile,
        out_dir=run_dir,
        active_speaker_ids=active_speaker_ids,
        active_doa_deg=active_doa_deg,
        null_user_doa_deg=null_user_doa_deg,
    )
    processed_audio, post_meta = _postprocess_variant(
        method=method,
        base_audio=shared_audio,
        mic_audio=np.asarray(mic_audio, dtype=np.float32),
        active_doa_deg=active_doa_deg,
        sample_rate=sample_rate,
        mic_geometry_xyz=np.asarray(mic_pos, dtype=np.float64),
    )
    elapsed_s = max(perf_counter() - t0, 1e-9)

    row: dict[str, object] = {
        "scene": scene_id,
        "method": method,
        "status": str(post_meta.get("status", "ok")),
        "sample_rate_hz": sample_rate,
        "noise_gain_scale": float(noise_gain_scale),
        "main_angle_deg": scene_meta["main_angle_deg"],
        "secondary_angle_deg": scene_meta["secondary_angle_deg"],
        "noise_layout_type": scene_meta["noise_layout_type"],
        "noise_angles_deg": json.dumps(scene_meta["noise_angles_deg"]),
        "scene_layout_family": scene_meta["scene_layout_family"],
        "null_user_speaker_id": NULL_USER_SPEAKER_ID if null_user_doa_deg is not None else "",
        "null_user_doa_deg": "" if null_user_doa_deg is None else float(null_user_doa_deg),
        "null_applied_fraction": 0.0,
        "null_fallback_fraction": 0.0,
        "rtf": float("nan"),
        "snr_db_raw": float("nan"),
        "snr_db_processed": float("nan"),
        "delta_snr_db": float("nan"),
        "sii_raw": float("nan"),
        "sii_processed": float("nan"),
        "delta_sii": float("nan"),
        "si_sdr_db_raw": float("nan"),
        "si_sdr_db_processed": float("nan"),
        "delta_si_sdr_db": float("nan"),
        "rnnoise_backend": str(post_meta.get("rnnoise_backend", "")),
        "rnnoise_error": str(post_meta.get("rnnoise_error", "")),
        "postfilter_name": str(post_meta.get("postfilter_name", "")),
        "beamforming_mode_runtime": str(shared_summary.get("beamforming_mode", "")),
        "run_dir": str(run_dir.resolve()),
    }

    if method == "lcmv_null":
        target_frames = [state for state in frame_states if state.target_doa_deg is not None]
        row["null_applied_fraction"] = (
            float(sum(1 for state in target_frames if state.force_suppression_active) / len(target_frames))
            if target_frames
            else 0.0
        )
        row["null_fallback_fraction"] = (
            float(sum(1 for state in target_frames if state.null_fallback) / len(target_frames))
            if target_frames
            else 0.0
        )

    run_summary: dict[str, object] = {
        "scene": scene_id,
        "method": method,
        "noise_gain_scale": float(noise_gain_scale),
        "profile": profile,
        "status": row["status"],
        "staged_scene": str(staged_scene.resolve()),
        "scenario_metadata": str(staged_metadata.resolve()) if staged_metadata.exists() else "",
        "run_dir": str(run_dir.resolve()),
        "scene_metadata": scene_meta,
        "shared_summary": shared_summary,
        "postprocess_metadata": post_meta,
        "null_user_speaker_id": row["null_user_speaker_id"],
        "null_user_doa_deg": row["null_user_doa_deg"],
        "null_applied_fraction": row["null_applied_fraction"],
        "null_fallback_fraction": row["null_fallback_fraction"],
    }

    if processed_audio is not None:
        processed_audio = np.asarray(processed_audio, dtype=np.float32).reshape(-1)
        sf.write(run_dir / "enhanced.wav", processed_audio, sample_rate)
        if method in {"mvdr_fd_rnnoise", "mvdr_fd_coherence_wiener"}:
            sf.write(run_dir / "enhanced_fast_path_shared.wav", shared_audio.astype(np.float32), sample_rate)
            sf.write(run_dir / "enhanced_fast_path.wav", processed_audio, sample_rate)
        bundle = _build_metric_bundle(
            clean_ref=clean_ref,
            raw_audio=raw_mix,
            processed_audio=processed_audio,
            sample_rate=sample_rate,
        )
        duration_s = float(len(processed_audio) / max(sample_rate, 1))
        row.update(
            {
                "rtf": float(elapsed_s / max(duration_s, 1e-9)),
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
        )
        run_summary["metrics"] = {
            "snr_db_raw": row["snr_db_raw"],
            "snr_db_processed": row["snr_db_processed"],
            "delta_snr_db": row["delta_snr_db"],
            "sii_raw": row["sii_raw"],
            "sii_processed": row["sii_processed"],
            "delta_sii": row["delta_sii"],
            "si_sdr_db_raw": row["si_sdr_db_raw"],
            "si_sdr_db_processed": row["si_sdr_db_processed"],
            "delta_si_sdr_db": row["delta_si_sdr_db"],
            "rtf": row["rtf"],
        }
    _write_json(run_dir / "summary.json", run_summary)
    return JobResult(row=row, run_summary=run_summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle-steered synthetic enhancement sweep aligned to shared beamformer runtime.")
    parser.add_argument("--scenes-root", default=str(DEFAULT_SCENES_ROOT))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=list(SUPPORTED_MIC_ARRAY_PROFILES))
    parser.add_argument("--noise-gain-scales", nargs="+", type=float, default=list(DEFAULT_NOISE_GAIN_SCALES))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    scenes = sorted(Path(args.scenes_root).glob("*.json"))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    if not scenes:
        raise FileNotFoundError(f"No scenes found under {args.scenes_root}")

    jobs = [
        {
            "scene_path": str(scene.resolve()),
            "assets_root": str(Path(args.assets_root).resolve()),
            "out_root": str(out_root.resolve()),
            "profile": str(args.profile),
            "noise_gain_scale": float(scale),
            "method": str(method),
        }
        for scene in scenes
        for scale in list(args.noise_gain_scales)
        for method in list(args.methods)
    ]

    results: list[JobResult] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_map = {pool.submit(_run_job, **job): (Path(job["scene_path"]).stem, job["method"], job["noise_gain_scale"]) for job in jobs}
        for future in as_completed(future_map):
            scene_id, method, scale = future_map[future]
            print(f"completed {scene_id} :: {method} :: noise={scale}")
            results.append(future.result())

    scene_rows = [result.row for result in sorted(results, key=lambda item: (str(item.row["scene"]), float(item.row["noise_gain_scale"]), str(item.row["method"])))]
    _write_csv(out_root / "scene_metrics.csv", scene_rows)

    summary_by_method = _aggregate(scene_rows, ["method"])
    summary_by_noise = _aggregate(scene_rows, ["method", "noise_gain_scale"])
    summary_by_layout = _aggregate(scene_rows, ["method", "scene_layout_family", "noise_layout_type"])
    _write_csv(out_root / "summary_by_method.csv", summary_by_method)
    _write_csv(out_root / "summary_by_noise_scale.csv", summary_by_noise)
    _write_csv(out_root / "summary_by_scene_layout.csv", summary_by_layout)

    _plot_noise_sweep(summary_by_noise, out_root / "plots" / "delta_snr_by_noise.png", "delta_snr_db_mean", "Delta SNR vs noise scale")
    _plot_noise_sweep(summary_by_noise, out_root / "plots" / "delta_sii_by_noise.png", "delta_sii_mean", "Delta SII vs noise scale")
    _plot_scene_layout(summary_by_layout, out_root / "plots" / "delta_snr_by_scene_layout.png", "delta_snr_db_mean", "Delta SNR by scene/layout")
    _plot_scene_layout(summary_by_layout, out_root / "plots" / "delta_sii_by_scene_layout.png", "delta_sii_mean", "Delta SII by scene/layout")

    summary_text = _compare_text(summary_by_method)
    (out_root / "summary.txt").write_text(summary_text, encoding="utf-8")
    payload = {
        "run_id": run_id,
        "profile": str(args.profile),
        "scenes_root": str(Path(args.scenes_root).resolve()),
        "assets_root": str(Path(args.assets_root).resolve()),
        "methods": list(args.methods),
        "noise_gain_scales": [float(v) for v in args.noise_gain_scales],
        "n_scenes": len(scenes),
        "summary_by_method": summary_by_method,
        "summary_by_noise_scale": summary_by_noise,
        "summary_by_scene_layout": summary_by_layout,
        "comparison_summary": summary_text.strip().splitlines(),
        "runs": [result.run_summary for result in results],
    }
    _write_json(out_root / "summary.json", payload)
    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_root.resolve(), target_is_directory=True)
    print(json.dumps({"out_root": str(out_root.resolve()), "n_runs": len(results), "n_scenes": len(scenes)}, indent=2))


if __name__ == "__main__":
    main()
