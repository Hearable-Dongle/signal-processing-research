from __future__ import annotations

import queue
import threading
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfilt, sosfiltfilt, tf2sos
from scipy.special import exp1

try:
    from pyrnnoise import RNNoise as PyRNNoise  # type: ignore
except ImportError:  # pragma: no cover
    PyRNNoise = None

from .contracts import FastPathAudioPacket, NoiseModelUpdateSnapshot, PipelineConfig, SRPPeakSnapshot, SpeakerGainDirection
from .localization_strategies.vad_utils import SileroVADGate, WebRTCVADGate
from .shared_state import SharedPipelineState, Timer
from .srp_tracker import SRPPeakTracker


FrameSource = Callable[[], np.ndarray | None]
FrameSink = Callable[[np.ndarray], None]


def _shift_with_zeros(x: np.ndarray, shift: int) -> np.ndarray:
    y = np.zeros_like(x)
    if shift == 0:
        y[:] = x
    elif shift > 0:
        y[shift:] = x[:-shift]
    else:
        y[:shift] = x[-shift:]
    return y


def _fractional_delay_shift(x: np.ndarray, delay_samples: float) -> np.ndarray:
    n = x.shape[0]
    t = np.arange(n, dtype=np.float64)
    src_t = t - float(delay_samples)
    return np.interp(src_t, t, np.asarray(x, dtype=np.float64), left=0.0, right=0.0)


def _center_mic_geometry_xyz(mic_geometry_xyz: np.ndarray) -> np.ndarray:
    mic_pos = np.asarray(mic_geometry_xyz, dtype=np.float64)
    if mic_pos.ndim != 2:
        raise ValueError("mic geometry must be 2D")
    if mic_pos.shape[0] == 3:
        centered = mic_pos - np.mean(mic_pos, axis=1, keepdims=True)
        return centered.astype(np.float64, copy=False)
    if mic_pos.shape[1] == 3:
        centered = mic_pos - np.mean(mic_pos, axis=0, keepdims=True)
        return centered.astype(np.float64, copy=False)
    raise ValueError("mic geometry must have 3 rows or 3 columns")


def _norm_deg(v: float) -> float:
    return float(v % 360.0)


def _wrap_to_180(v: float) -> float:
    a = (float(v) + 180.0) % 360.0 - 180.0
    return float(a)


def _angular_dist_deg(a: float, b: float) -> float:
    return abs(_wrap_to_180(float(a) - float(b)))


def _step_limited_angle(prev_deg: float, next_deg: float, max_step_deg: float) -> float:
    delta = _wrap_to_180(next_deg - prev_deg)
    step = float(np.clip(delta, -max_step_deg, max_step_deg))
    return _norm_deg(prev_deg + step)


def _ema_angle(prev_deg: float, new_deg: float, alpha: float) -> float:
    p = np.deg2rad(float(prev_deg))
    n = np.deg2rad(float(new_deg))
    pv = np.array([np.cos(p), np.sin(p)], dtype=np.float64)
    nv = np.array([np.cos(n), np.sin(n)], dtype=np.float64)
    v = (1.0 - alpha) * pv + alpha * nv
    if float(np.linalg.norm(v)) < 1e-12:
        return _norm_deg(new_deg)
    return _norm_deg(np.degrees(np.arctan2(v[1], v[0])))


def _frame_speech_activity(frame_mc: np.ndarray) -> float:
    mono = np.mean(np.asarray(frame_mc, dtype=np.float64), axis=1)
    rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
    return float(np.clip((rms - 0.005) / 0.03, 0.0, 1.0))


def _rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12))
    return float(20.0 * np.log10(max(rms, 1e-12)))


def _safe_condition_number(x: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(np.asarray(x, dtype=np.complex128)))
    except np.linalg.LinAlgError:
        return float("inf")


def _inactive_noise_model_update() -> dict:
    return {"active": False, "sources": (), "reasons": (), "debug": {}}


def _normalize_noise_model_update(info: dict | None) -> dict:
    if not info:
        return _inactive_noise_model_update()
    return {
        "active": bool(info.get("active", False)),
        "sources": tuple(str(v) for v in info.get("sources", ())),
        "reasons": tuple(str(v) for v in info.get("reasons", ())),
        "debug": dict(info.get("debug", {})),
    }


def _merge_noise_model_updates(*infos: dict | None) -> dict:
    active = False
    sources: list[str] = []
    reasons: list[str] = []
    debug: dict[str, object] = {}
    for raw in infos:
        info = _normalize_noise_model_update(raw)
        active = active or bool(info["active"])
        for source in info["sources"]:
            if source not in sources:
                sources.append(source)
        for reason in info["reasons"]:
            if reason not in reasons:
                reasons.append(reason)
        if info["debug"]:
            for key, value in info["debug"].items():
                debug[key] = value
    return {
        "active": bool(active),
        "sources": tuple(sources),
        "reasons": tuple(reasons),
        "debug": debug,
    }


def delay_and_sum_frame(
    frame_mc: np.ndarray,
    doa_deg: float,
    mic_geometry_xyz: np.ndarray,
    fs: int,
    sound_speed_m_s: float,
) -> np.ndarray:
    frame = np.asarray(frame_mc, dtype=np.float64)
    if frame.ndim != 2:
        raise ValueError("frame_mc must be shape (samples, n_mics)")

    mic_pos = np.asarray(mic_geometry_xyz, dtype=float)
    if mic_pos.shape[0] == 3:
        mic_pos = mic_pos.T
    if mic_pos.shape[1] == 2:
        mic_pos = np.hstack([mic_pos, np.zeros((mic_pos.shape[0], 1), dtype=float)])

    az = np.deg2rad(float(doa_deg))
    # Scene DOAs are source positions relative to the array center; propagation arrives from the opposite direction.
    direction = np.array([-np.cos(az), -np.sin(az), 0.0], dtype=float)
    tau = (mic_pos @ direction) / float(sound_speed_m_s)
    tau = tau - float(np.mean(tau))
    delays = tau * float(fs)

    aligned = np.zeros(frame.shape[0], dtype=np.float64)
    for m in range(frame.shape[1]):
        aligned += _fractional_delay_shift(frame[:, m], float(delays[m]))

    return (aligned / max(1, frame.shape[1])).astype(np.float32, copy=False)


def _steering_vector_f_domain(
    doa_deg: float,
    n_fft: int,
    fs: int,
    mic_geometry_xyz: np.ndarray,
    sound_speed_m_s: float,
) -> np.ndarray:
    mic_pos = np.asarray(mic_geometry_xyz, dtype=np.float64)
    if mic_pos.shape[0] == 3:
        mic_pos = mic_pos.T
    if mic_pos.shape[1] == 2:
        mic_pos = np.hstack([mic_pos, np.zeros((mic_pos.shape[0], 1), dtype=np.float64)])

    az = np.deg2rad(float(doa_deg))
    # Scene DOAs are source positions relative to the array center; propagation arrives from the opposite direction.
    direction = np.array([-np.cos(az), -np.sin(az), 0.0], dtype=np.float64)
    tau = (mic_pos @ direction) / float(sound_speed_m_s)
    tau = tau - float(np.mean(tau))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    phase = -2j * np.pi * freqs[:, None] * tau[None, :]
    return np.exp(phase).astype(np.complex128)


class _FDBufferedBeamformer:
    def __init__(self, n_mics: int, frame_samples: int, cfg: PipelineConfig, mic_geometry_xyz: np.ndarray):
        self.n_mics = int(n_mics)
        self.hop = int(frame_samples)
        analysis_samples = int(round(float(cfg.sample_rate_hz) * (float(cfg.fd_analysis_window_ms) / 1000.0)))
        analysis_samples = max(self.hop, analysis_samples)
        if analysis_samples % self.hop != 0:
            analysis_samples = int(np.ceil(analysis_samples / self.hop) * self.hop)
        self.n = int(self.hop)
        self.analysis_n = int(analysis_samples)
        self.cfg = cfg
        self.mic_geometry_xyz = np.asarray(mic_geometry_xyz, dtype=np.float64)
        self.rnn_mvdr_noise: np.ndarray | None = None
        self.rxx_mvdr: np.ndarray | None = None
        self.target_psd_mvdr: np.ndarray | None = None
        self.rnn_gsc: np.ndarray | None = None
        self._win = np.sqrt(np.hanning(max(4, self.analysis_n))).astype(np.float64)
        self._prev_mc = np.zeros((max(0, self.analysis_n - self.hop), self.n_mics), dtype=np.float64)
        self._prev_noise_mc = np.zeros((max(0, self.analysis_n - self.hop), self.n_mics), dtype=np.float64)
        self._ola_tail_mvdr = np.zeros(self.analysis_n, dtype=np.float64)
        self._ola_tail_gsc = np.zeros(self.analysis_n, dtype=np.float64)
        self._ola_norm_mvdr = np.zeros(self.analysis_n, dtype=np.float64)
        self._ola_norm_gsc = np.zeros(self.analysis_n, dtype=np.float64)
        mvdr_hop_ms = getattr(cfg, "mvdr_hop_ms", None)
        if mvdr_hop_ms is None:
            self.solve_interval_frames = 1
        else:
            self.solve_interval_frames = max(1, int(np.ceil(float(mvdr_hop_ms) / max(float(cfg.fast_frame_ms), 1.0))))
        self._mvdr_frame_counter = 0
        self._cached_mvdr_weights: np.ndarray | None = None
        self._cached_lcmv_weights: np.ndarray | None = None
        self._cached_steering: np.ndarray | None = None
        self._cached_target_doa_deg: float | None = None
        self._cached_secondary_target_doa_deg: float | None = None
        self._cached_null_doa_deg: float | None = None
        self._cached_last_target_active: bool | None = None
        self._cached_last_solved_rnn: np.ndarray | None = None
        self._last_weights_reused: bool = False
        self._beamformer_refresh_requests: int = 0
        self._beamformer_refresh_executed: int = 0
        self._beamformer_refresh_skipped_clean: int = 0
        self._beamformer_weights_reused_frames: int = 0
        self._beamformer_last_dirty_stat: float = 0.0
        self._beamformer_dirty_stats: list[float] = []
        self._last_noise_model_update: dict = _inactive_noise_model_update()
        self._frozen_bootstrap_frames = max(1, int(np.ceil(1000.0 / max(float(cfg.fast_frame_ms), 1.0))))
        self._beamformer_snapshot_targets = tuple(
            int(v) for v in getattr(cfg, "beamformer_snapshot_frame_indices", ()) if int(v) > 0
        )
        self._beamformer_snapshot_target_set = set(self._beamformer_snapshot_targets)
        self._beamformer_snapshot_trace: list[dict] = []

    def get_last_noise_model_update(self) -> dict:
        return _normalize_noise_model_update(self._last_noise_model_update)

    def get_beamformer_snapshot_trace(self) -> list[dict]:
        return [dict(item) for item in self._beamformer_snapshot_trace]

    def get_beamformer_runtime_stats(self) -> dict[str, float | int]:
        dirty = np.asarray(self._beamformer_dirty_stats, dtype=np.float64)
        return {
            "beamformer_refresh_requests": int(self._beamformer_refresh_requests),
            "beamformer_refresh_executed": int(self._beamformer_refresh_executed),
            "beamformer_refresh_skipped_clean": int(self._beamformer_refresh_skipped_clean),
            "beamformer_weights_reused_frames": int(self._beamformer_weights_reused_frames),
            "beamformer_dirty_stat_last": float(self._beamformer_last_dirty_stat),
            "beamformer_dirty_stat_mean": float(np.mean(dirty)) if dirty.size else 0.0,
            "beamformer_dirty_stat_p95": float(np.percentile(dirty, 95.0)) if dirty.size else 0.0,
        }

    def get_output_noise_psd_for_fft(self, target_n_fft: int) -> np.ndarray | None:
        if self.rnn_mvdr_noise is None:
            return None
        weights = self._cached_lcmv_weights if self._cached_lcmv_weights is not None else self._cached_mvdr_weights
        if weights is None:
            return None
        target_n_fft = int(max(2, target_n_fft))
        src_freq = np.fft.rfftfreq(self.analysis_n, d=1.0 / float(self.cfg.sample_rate_hz))
        dst_freq = np.fft.rfftfreq(target_n_fft, d=1.0 / float(self.cfg.sample_rate_hz))
        phi = np.zeros((weights.shape[0],), dtype=np.float64)
        for f in range(weights.shape[0]):
            w = weights[f].reshape(-1, 1)
            r = np.asarray(self.rnn_mvdr_noise[f], dtype=np.complex128)
            phi[f] = max(float(np.real((w.conj().T @ r @ w)[0, 0])), 1e-12)
        if phi.shape[0] == dst_freq.shape[0]:
            return phi.astype(np.float32, copy=False)
        return np.asarray(np.interp(dst_freq, src_freq, phi, left=phi[0], right=phi[-1]), dtype=np.float32)

    def _maybe_record_beamformer_snapshot(
        self,
        *,
        beamforming_mode: str,
        target_doa_deg: float,
        weights: np.ndarray,
        null_doa_deg: float | None = None,
        secondary_doa_deg: float | None = None,
        target_band_width_deg: float | None = None,
    ) -> None:
        frame_idx = int(self._mvdr_frame_counter)
        if frame_idx not in self._beamformer_snapshot_target_set:
            return
        if any(int(item.get("frame_index", -1)) == frame_idx for item in self._beamformer_snapshot_trace):
            return
        w = np.asarray(weights, dtype=np.complex128)
        self._beamformer_snapshot_trace.append(
            {
                "frame_index": frame_idx,
                "beamforming_mode": str(beamforming_mode),
                "target_doa_deg": float(target_doa_deg),
                "null_doa_deg": None if null_doa_deg is None else float(null_doa_deg),
                "secondary_doa_deg": None if secondary_doa_deg is None else float(secondary_doa_deg),
                "target_band_width_deg": None if target_band_width_deg is None else float(target_band_width_deg),
                "weights_real": np.asarray(np.real(w), dtype=np.float32).tolist(),
                "weights_imag": np.asarray(np.imag(w), dtype=np.float32).tolist(),
            }
        )

    def _refresh_reason(
        self,
        *,
        doa_deg: float,
        target_active: bool,
        null_doa_deg: float | None = None,
        secondary_doa_deg: float | None = None,
        requires_lcmv_weights: bool = False,
    ) -> str | None:
        if requires_lcmv_weights:
            if self._cached_lcmv_weights is None:
                return "uninitialized"
        elif self._cached_mvdr_weights is None:
            return "uninitialized"
        if self._cached_mvdr_weights is None and self._cached_lcmv_weights is None:
            return "uninitialized"
        doa_tolerance_deg = float(max(getattr(self.cfg, "beamformer_doa_refresh_tolerance_deg", 5.0), 0.1))
        if self._cached_target_doa_deg is None or _angular_dist_deg(self._cached_target_doa_deg, doa_deg) >= doa_tolerance_deg:
            return "target_doa_changed"
        if secondary_doa_deg is not None:
            if self._cached_secondary_target_doa_deg is None:
                return "secondary_doa_changed"
            if _angular_dist_deg(self._cached_secondary_target_doa_deg, secondary_doa_deg) >= doa_tolerance_deg:
                return "secondary_doa_changed"
        if self._cached_last_target_active is None or bool(self._cached_last_target_active) != bool(target_active):
            return "target_activity_flip"
        if null_doa_deg is not None:
            if self._cached_lcmv_weights is None:
                return "null_doa_changed"
            if self._cached_null_doa_deg is None or _angular_dist_deg(self._cached_null_doa_deg, null_doa_deg) >= doa_tolerance_deg:
                return "null_doa_changed"
        if self._mvdr_frame_counter % max(self.solve_interval_frames, 1) == 0:
            return "hop"
        return None

    def _rnn_dirty_stat(self, candidate_rnn: np.ndarray | None) -> float:
        if candidate_rnn is None or self._cached_last_solved_rnn is None:
            stat = float("inf")
            self._beamformer_last_dirty_stat = stat
            return stat
        eps = float(max(getattr(self.cfg, "beamformer_rnn_dirty_eps", 1e-8), 1e-12))
        prev = np.asarray(self._cached_last_solved_rnn, dtype=np.complex128)
        cur = np.asarray(candidate_rnn, dtype=np.complex128)
        if prev.shape != cur.shape:
            stat = float("inf")
            self._beamformer_last_dirty_stat = stat
            return stat
        deltas: list[float] = []
        for f in range(cur.shape[0]):
            num = float(np.linalg.norm((cur[f] - prev[f]).reshape(-1), ord=2))
            den = max(float(np.linalg.norm(prev[f].reshape(-1), ord=2)), eps)
            deltas.append(num / den)
        values = np.asarray(deltas, dtype=np.float64)
        mode = str(getattr(self.cfg, "beamformer_rnn_dirty_stat", "max")).strip().lower()
        stat = float(np.mean(values)) if mode == "mean" else float(np.max(values))
        self._beamformer_last_dirty_stat = stat
        self._beamformer_dirty_stats.append(stat)
        return stat

    def _should_execute_refresh(
        self,
        *,
        refresh_reason: str | None,
        candidate_rnn: np.ndarray | None,
    ) -> tuple[bool, str | None]:
        if refresh_reason is None:
            return False, None
        self._beamformer_refresh_requests += 1
        if refresh_reason != "hop":
            return True, refresh_reason
        if not bool(getattr(self.cfg, "beamformer_rnn_skip_refresh_when_clean", False)):
            return True, refresh_reason
        threshold = float(max(getattr(self.cfg, "beamformer_rnn_dirty_threshold", 0.0), 0.0))
        if threshold <= 0.0:
            return True, refresh_reason
        dirty = self._rnn_dirty_stat(candidate_rnn)
        if np.isfinite(dirty) and dirty <= threshold:
            self._beamformer_refresh_skipped_clean += 1
            return False, "hop_clean_skip"
        return True, refresh_reason

    def _solve_mvdr_weights(self, steering: np.ndarray) -> np.ndarray:
        f_bins = steering.shape[0]
        weights = np.zeros((f_bins, self.n_mics), dtype=np.complex128)
        for f in range(f_bins):
            r = self._loaded_noise_covariance(self.rnn_mvdr_noise[f])
            af = steering[f].reshape(-1, 1)
            try:
                rinv_a = np.linalg.solve(r, af)
            except np.linalg.LinAlgError:
                rinv_a = np.linalg.pinv(r) @ af
            denom = (af.conj().T @ rinv_a)[0, 0]
            wf = rinv_a / (denom + 1e-10)
            weights[f, :] = wf.reshape(-1)
        return weights

    def _solve_lcmv_constraint_weights(self, constraints: np.ndarray, response: np.ndarray) -> np.ndarray:
        f_bins = constraints.shape[0]
        weights = np.zeros((f_bins, self.n_mics), dtype=np.complex128)
        for f in range(f_bins):
            r = self._loaded_noise_covariance(self.rnn_mvdr_noise[f])
            try:
                r_inv_constraint = np.linalg.solve(r, constraints[f])
            except np.linalg.LinAlgError:
                r_inv_constraint = np.linalg.pinv(r) @ constraints[f]
            gram = constraints[f].conj().T @ r_inv_constraint
            try:
                mid = np.linalg.solve(gram + 1e-8 * np.eye(gram.shape[0], dtype=np.complex128), response)
            except np.linalg.LinAlgError:
                mid = np.linalg.pinv(gram + 1e-8 * np.eye(gram.shape[0], dtype=np.complex128)) @ response
            weights[f, :] = (r_inv_constraint @ mid).reshape(-1)
        return weights

    def _solve_lcmv_weights(self, a_target: np.ndarray, a_null: np.ndarray) -> np.ndarray:
        constraints = np.stack([a_target, a_null], axis=2)
        response = np.asarray([[1.0 + 0.0j], [0.0 + 0.0j]], dtype=np.complex128)
        return self._solve_lcmv_constraint_weights(constraints, response)

    def _solve_lcmv_top2_weights(self, a_primary: np.ndarray, a_secondary: np.ndarray) -> np.ndarray:
        constraints = np.stack([a_primary, a_secondary], axis=2)
        response = np.asarray([[1.0 + 0.0j], [1.0 + 0.0j]], dtype=np.complex128)
        return self._solve_lcmv_constraint_weights(constraints, response)

    def _solve_lcmv_target_band_weights(self, steerings: list[np.ndarray]) -> np.ndarray:
        max_freq_hz = float(max(0.0, getattr(self.cfg, "robust_target_band_max_freq_hz", 0.0)))
        freq_axis = np.fft.rfftfreq(self.analysis_n, d=1.0 / float(self.cfg.sample_rate_hz))
        constraints = np.stack(steerings, axis=2)
        f_bins = constraints.shape[0]
        weights = np.zeros((f_bins, self.n_mics), dtype=np.complex128)
        sparse_enabled = bool(getattr(self.cfg, "beamformer_sparse_solve_enabled", False))
        sparse_stride = max(1, int(getattr(self.cfg, "beamformer_sparse_solve_stride", 1)))
        sparse_min_freq_hz = float(max(0.0, getattr(self.cfg, "beamformer_sparse_solve_min_freq_hz", 200.0)))
        condition_limit = float(max(1.0, getattr(self.cfg, "robust_target_band_condition_limit", 1e3)))
        self._last_target_band_conditioning = {
            "full_band_frames": 0,
            "edge_only_frames": 0,
            "center_only_frames": 0,
            "high_freq_fallback_frames": 0,
            "max_selected_cond": 0.0,
        }
        if (not sparse_enabled) or sparse_stride <= 1:
            solve_bins = list(range(f_bins))
            interp_bins: list[int] = []
        else:
            solve_bins = []
            interp_bins = []
            for f in range(f_bins):
                freq_hz = float(freq_axis[f])
                if max_freq_hz > 0.0 and freq_hz > max_freq_hz:
                    continue
                if freq_hz < sparse_min_freq_hz:
                    solve_bins.append(f)
                    continue
                interp_bins.append(f)
            sampled_interp = interp_bins[::sparse_stride]
            if interp_bins and interp_bins[-1] not in sampled_interp:
                sampled_interp.append(interp_bins[-1])
            solve_bins.extend(sampled_interp)
            solve_bins = sorted(set(solve_bins))
        solved_mask = np.zeros((f_bins,), dtype=bool)
        center_steering = constraints[:, :, constraints.shape[2] // 2]

        for f in solve_bins:
            full_constraints = constraints[f]
            full_response = np.ones((full_constraints.shape[1], 1), dtype=np.complex128)
            center_idx = full_constraints.shape[1] // 2
            center_constraints = full_constraints[:, [center_idx]]
            center_response = np.ones((1, 1), dtype=np.complex128)
            if max_freq_hz > 0.0 and float(freq_axis[f]) > max_freq_hz:
                af = center_constraints[:, 0].reshape(-1, 1)
                denom = (af.conj().T @ af)[0, 0]
                wf = af / (denom + 1e-10)
                weights[f, :] = wf.reshape(-1)
                solved_mask[f] = True
                self._last_target_band_conditioning["high_freq_fallback_frames"] += 1
                continue
            else:
                candidates: list[tuple[str, np.ndarray, np.ndarray]] = [("full_band", full_constraints, full_response)]
                if full_constraints.shape[1] >= 3:
                    candidates.append(("center_only", center_constraints, center_response))
                selected_label, selected_constraints, selected_response = candidates[-1]
                selected_cond = float("inf")
                for label, candidate_constraints, candidate_response in candidates:
                    gram = candidate_constraints.conj().T @ candidate_constraints
                    cond = _safe_condition_number(gram)
                    if np.isfinite(cond) and cond <= condition_limit:
                        selected_label = label
                        selected_constraints = candidate_constraints
                        selected_response = candidate_response
                        selected_cond = cond
                        break
                    if label == candidates[-1][0]:
                        selected_label = label
                        selected_constraints = candidate_constraints
                        selected_response = candidate_response
                        selected_cond = cond
            r = self._loaded_noise_covariance(self.rnn_mvdr_noise[f])
            try:
                r_inv_constraint = np.linalg.solve(r, selected_constraints)
            except np.linalg.LinAlgError:
                r_inv_constraint = np.linalg.pinv(r) @ selected_constraints
            gram = selected_constraints.conj().T @ r_inv_constraint
            try:
                mid = np.linalg.solve(gram + 1e-8 * np.eye(gram.shape[0], dtype=np.complex128), selected_response)
            except np.linalg.LinAlgError:
                mid = np.linalg.pinv(gram + 1e-8 * np.eye(gram.shape[0], dtype=np.complex128)) @ selected_response
            weights[f, :] = (r_inv_constraint @ mid).reshape(-1)
            weights[f, :] = self._renormalize_weights_to_target(weights[f, :], center_steering[f])
            solved_mask[f] = True
            if selected_label == "full_band":
                self._last_target_band_conditioning["full_band_frames"] += 1
            elif selected_label == "high_freq_center_only":
                self._last_target_band_conditioning["high_freq_fallback_frames"] += 1
            else:
                self._last_target_band_conditioning["center_only_frames"] += 1
            self._last_target_band_conditioning["max_selected_cond"] = max(
                float(self._last_target_band_conditioning["max_selected_cond"]),
                float(selected_cond if np.isfinite(selected_cond) else 0.0),
            )
        if sparse_enabled and sparse_stride > 1 and np.any(solved_mask):
            weights = self._interpolate_sparse_lcmv_weights(
                weights=weights,
                solved_mask=solved_mask,
                center_steering=center_steering,
                freq_axis=freq_axis,
                max_freq_hz=max_freq_hz,
                min_freq_hz=sparse_min_freq_hz,
            )
        return weights

    def _renormalize_weights_to_target(self, weights: np.ndarray, target_steering: np.ndarray) -> np.ndarray:
        wf = np.asarray(weights, dtype=np.complex128).reshape(-1)
        af = np.asarray(target_steering, dtype=np.complex128).reshape(-1)
        denom = np.vdot(af, wf)
        if not np.isfinite(np.real(denom)) or abs(denom) < 1e-10:
            denom = np.vdot(af, af) + 1e-10
            return (af / denom).reshape(-1)
        return (wf / denom).reshape(-1)

    def _interpolate_sparse_lcmv_weights(
        self,
        *,
        weights: np.ndarray,
        solved_mask: np.ndarray,
        center_steering: np.ndarray,
        freq_axis: np.ndarray,
        max_freq_hz: float,
        min_freq_hz: float,
    ) -> np.ndarray:
        interp_mode = str(getattr(self.cfg, "beamformer_sparse_solve_interp", "linear_complex")).strip().lower()
        if interp_mode != "linear_complex":
            return weights
        out = np.asarray(weights, dtype=np.complex128).copy()
        solved_idx = np.flatnonzero(solved_mask)
        if solved_idx.size < 2:
            return out
        upper_hz = max_freq_hz if max_freq_hz > 0.0 else float(freq_axis[-1])
        for f in range(out.shape[0]):
            freq_hz = float(freq_axis[f])
            if solved_mask[f]:
                continue
            if freq_hz < min_freq_hz:
                continue
            if freq_hz > upper_hz:
                continue
            for m in range(self.n_mics):
                out[f, m] = np.interp(freq_hz, freq_axis[solved_idx], np.real(out[solved_idx, m])) + (
                    1j * np.interp(freq_hz, freq_axis[solved_idx], np.imag(out[solved_idx, m]))
                )
            out[f, :] = self._renormalize_weights_to_target(out[f, :], center_steering[f])
        return out

    def _apply_lcmv_weight_smoothing(self, new_weights: np.ndarray, target_steering: np.ndarray) -> np.ndarray:
        if not bool(getattr(self.cfg, "beamformer_weight_reuse_enabled", True)):
            return new_weights
        alpha = float(np.clip(getattr(self.cfg, "beamformer_weight_smoothing_alpha", 1.0), 0.0, 1.0))
        if alpha >= 0.999 or self._cached_lcmv_weights is None or self._cached_lcmv_weights.shape != new_weights.shape:
            return new_weights
        blended = ((1.0 - alpha) * np.asarray(self._cached_lcmv_weights, dtype=np.complex128)) + (alpha * np.asarray(new_weights, dtype=np.complex128))
        for f in range(blended.shape[0]):
            blended[f, :] = self._renormalize_weights_to_target(blended[f, :], target_steering[f])
        return blended

    def _loaded_noise_covariance(self, rnn_bin: np.ndarray) -> np.ndarray:
        r = np.asarray(rnn_bin, dtype=np.complex128)
        mics = r.shape[0]
        eye = np.eye(mics, dtype=np.complex128)
        blend_alpha = float(np.clip(getattr(self.cfg, "fd_identity_blend_alpha", 0.0), 0.0, 1.0))
        trace_scale = float(np.real(np.trace(r))) / float(max(1, mics))
        if blend_alpha > 0.0:
            isotropic = max(trace_scale, 0.0) * eye
            r = ((1.0 - blend_alpha) * r) + (blend_alpha * isotropic)
        static_load = float(max(getattr(self.cfg, "fd_diag_load", 0.0), 0.0))
        trace_factor = float(max(getattr(self.cfg, "fd_trace_diagonal_loading_factor", 0.0), 0.0))
        trace_scale = float(np.real(np.trace(r))) / float(max(1, mics))
        adaptive_load = trace_factor * max(trace_scale, 0.0)
        total_load = static_load + adaptive_load + 1e-8
        return r + (total_load * eye)

    def _update_covariance(self, rnn_prev: np.ndarray | None, x_fft: np.ndarray, cov_alpha: float) -> np.ndarray:
        # x_fft: (F, M)
        f_bins, mics = x_fft.shape
        inst = np.einsum("fm,fn->fmn", x_fft, x_fft.conj())
        if rnn_prev is None:
            rnn = inst
        else:
            a = float(np.clip(cov_alpha, 0.0, 1.0))
            rnn = (1.0 - a) * rnn_prev + a * inst
        diag = float(max(self.cfg.fd_diag_load, 1e-9))
        rnn += diag * np.eye(mics, dtype=np.complex128)[None, :, :]
        return rnn

    def _noise_covariance_mode(self) -> str:
        return str(getattr(self.cfg, "fd_noise_covariance_mode", "estimated_target_subtractive")).strip().lower()

    def _noise_covariance_is_frozen(self) -> bool:
        return self._noise_covariance_mode() == "estimated_target_subtractive_frozen"

    def _noise_covariance_frozen_bootstrap_active(self) -> bool:
        return bool(self._noise_covariance_is_frozen() and self._mvdr_frame_counter <= self._frozen_bootstrap_frames)

    def _update_noise_covariance(
        self,
        rnn_prev: np.ndarray | None,
        x_fft: np.ndarray,
        steering: np.ndarray,
        cov_alpha: float,
        *,
        target_active: bool,
        oracle_noise_fft: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._noise_covariance_is_frozen() and rnn_prev is not None and not self._noise_covariance_frozen_bootstrap_active():
            return np.asarray(rnn_prev, dtype=np.complex128)
        if self._noise_covariance_mode() == "oracle_non_target_residual":
            if oracle_noise_fft is None:
                raise ValueError("oracle_non_target_residual mode requires oracle noise FFT observations.")
            return self._update_covariance(rnn_prev, oracle_noise_fft, cov_alpha)
        if self._noise_covariance_frozen_bootstrap_active():
            return self._update_covariance(rnn_prev, x_fft, cov_alpha)
        # When the target is inactive, the mixture is a direct observation of the noise field and can
        # refresh Rnn without any target subtraction. Otherwise estimate interference/noise covariance
        # by subtracting the steered target covariance from the full mixture covariance.
        f_bins, mics = x_fft.shape
        inst_rxx = np.einsum("fm,fn->fmn", x_fft, x_fft.conj())
        a = float(np.clip(cov_alpha, 0.0, 1.0))
        if self.rxx_mvdr is None:
            self.rxx_mvdr = inst_rxx
        else:
            self.rxx_mvdr = (1.0 - a) * self.rxx_mvdr + a * inst_rxx
        if not bool(target_active):
            return self._update_covariance(rnn_prev, x_fft, a)

        a_norm_sq = np.sum(np.abs(steering) ** 2, axis=1) + 1e-10
        target_ref = np.sum(np.conj(steering) * x_fft, axis=1) / a_norm_sq
        inst_target_psd = np.abs(target_ref) ** 2
        if rnn_prev is not None:
            noise_ref_psd = np.real(np.einsum("fm,fmn,fn->f", np.conj(steering), rnn_prev, steering)) / (a_norm_sq**2)
            inst_target_psd = np.maximum(inst_target_psd - np.maximum(noise_ref_psd, 0.0), 0.0)
        if self.target_psd_mvdr is None:
            self.target_psd_mvdr = inst_target_psd
        else:
            self.target_psd_mvdr = (1.0 - a) * self.target_psd_mvdr + a * inst_target_psd

        target_cov = self.target_psd_mvdr[:, None, None] * np.einsum("fm,fn->fmn", steering, steering.conj())
        residual_cov = np.einsum("fm,fn->fmn", x_fft - (steering * target_ref[:, None]), np.conj(x_fft - (steering * target_ref[:, None])))
        if rnn_prev is None:
            smoothed_residual_cov = residual_cov
        else:
            smoothed_residual_cov = (1.0 - a) * rnn_prev + a * residual_cov
        subtractive_rnn = self.rxx_mvdr - target_cov
        rnn = (0.75 * subtractive_rnn) + (0.25 * smoothed_residual_cov)
        rnn = 0.5 * (rnn + np.conjugate(np.swapaxes(rnn, 1, 2)))
        diag = float(max(self.cfg.fd_diag_load, 1e-9))
        eye = np.eye(mics, dtype=np.complex128)
        for f in range(f_bins):
            eigvals, eigvecs = np.linalg.eigh(rnn[f])
            avg_power = float(np.real(np.trace(self.rxx_mvdr[f]))) / max(mics, 1)
            floor = max(diag, 1e-4 * avg_power)
            eigvals = np.maximum(np.real(eigvals), floor)
            rnn[f] = (eigvecs @ np.diag(eigvals.astype(np.complex128)) @ eigvecs.conj().T) + (diag * eye)
        return rnn

    def _update_noise_covariance_multi(
        self,
        rnn_prev: np.ndarray | None,
        x_fft: np.ndarray,
        steerings: list[np.ndarray],
        cov_alpha: float,
        *,
        target_active: bool,
        oracle_noise_fft: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._noise_covariance_is_frozen() and rnn_prev is not None and not self._noise_covariance_frozen_bootstrap_active():
            return np.asarray(rnn_prev, dtype=np.complex128)
        if not steerings:
            raise ValueError("multi-target covariance update requires at least one steering vector")
        if self._noise_covariance_mode() == "oracle_non_target_residual":
            if oracle_noise_fft is None:
                raise ValueError("oracle_non_target_residual mode requires oracle noise FFT observations.")
            return self._update_covariance(rnn_prev, oracle_noise_fft, cov_alpha)
        if self._noise_covariance_frozen_bootstrap_active():
            return self._update_covariance(rnn_prev, x_fft, cov_alpha)
        if len(steerings) == 1 or not bool(target_active):
            return self._update_noise_covariance(
                rnn_prev,
                x_fft,
                steerings[0],
                cov_alpha,
                target_active=target_active,
                oracle_noise_fft=oracle_noise_fft,
            )
        f_bins, mics = x_fft.shape
        inst_rxx = np.einsum("fm,fn->fmn", x_fft, x_fft.conj())
        a = float(np.clip(cov_alpha, 0.0, 1.0))
        if self.rxx_mvdr is None:
            self.rxx_mvdr = inst_rxx
        else:
            self.rxx_mvdr = (1.0 - a) * self.rxx_mvdr + a * inst_rxx
        target_cov = np.zeros_like(inst_rxx)
        residual = np.asarray(x_fft, dtype=np.complex128).copy()
        for steering in steerings:
            a_norm_sq = np.sum(np.abs(steering) ** 2, axis=1) + 1e-10
            target_ref = np.sum(np.conj(steering) * residual, axis=1) / a_norm_sq
            inst_target_psd = np.abs(target_ref) ** 2
            target_cov += inst_target_psd[:, None, None] * np.einsum("fm,fn->fmn", steering, steering.conj())
            residual = residual - (steering * target_ref[:, None])
        residual_cov = np.einsum("fm,fn->fmn", residual, np.conj(residual))
        if rnn_prev is None:
            smoothed_residual_cov = residual_cov
        else:
            smoothed_residual_cov = (1.0 - a) * rnn_prev + a * residual_cov
        subtractive_rnn = self.rxx_mvdr - target_cov
        rnn = (0.75 * subtractive_rnn) + (0.25 * smoothed_residual_cov)
        rnn = 0.5 * (rnn + np.conjugate(np.swapaxes(rnn, 1, 2)))
        diag = float(max(self.cfg.fd_diag_load, 1e-9))
        eye = np.eye(mics, dtype=np.complex128)
        for f in range(f_bins):
            eigvals, eigvecs = np.linalg.eigh(rnn[f])
            avg_power = float(np.real(np.trace(self.rxx_mvdr[f]))) / max(mics, 1)
            floor = max(diag, 1e-4 * avg_power)
            eigvals = np.maximum(np.real(eigvals), floor)
            rnn[f] = (eigvecs @ np.diag(eigvals.astype(np.complex128)) @ eigvecs.conj().T) + (diag * eye)
        return rnn

    def _analysis_block_with_state(
        self,
        frame_mc: np.ndarray,
        prev_mc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(frame_mc, dtype=np.float64)
        block = np.concatenate([np.asarray(prev_mc, dtype=np.float64), x], axis=0)
        if block.shape[0] != self.analysis_n:
            raise ValueError(f"analysis block size mismatch: got {block.shape[0]}, expected {self.analysis_n}")
        next_prev = np.zeros((max(0, self.analysis_n - self.hop), self.n_mics), dtype=np.float64)
        if self.analysis_n > self.hop:
            next_prev = block[-(self.analysis_n - self.hop) :].copy()
        xw = block * self._win[:, None]
        return np.fft.rfft(xw, axis=0), next_prev  # (F, M)

    def _analysis_block(self, frame_mc: np.ndarray) -> np.ndarray:
        x_fft, next_prev = self._analysis_block_with_state(frame_mc, self._prev_mc)
        self._prev_mc = next_prev
        return x_fft

    def _synthesis_block(
        self,
        y_fft: np.ndarray,
        ola_buffer: np.ndarray,
        norm_buffer: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_block = np.fft.irfft(y_fft, n=self.analysis_n).real
        yw = y_block * self._win
        win_sq = self._win**2
        buffer = np.asarray(ola_buffer, dtype=np.float64).copy()
        norm = np.asarray(norm_buffer, dtype=np.float64).copy()
        buffer[: self.analysis_n] += yw
        norm[: self.analysis_n] += win_sq
        out = buffer[: self.hop] / np.maximum(norm[: self.hop], 1e-8)
        buffer[:-self.hop] = buffer[self.hop :]
        buffer[-self.hop :] = 0.0
        norm[:-self.hop] = norm[self.hop :]
        norm[-self.hop :] = 0.0
        return out.astype(np.float32, copy=False), buffer, norm

    def _cov_alpha_from_activity(self, speech_activity: float) -> float:
        base = float(np.clip(self.cfg.fd_cov_ema_alpha, 0.0, 1.0))
        scale = float(np.clip(self.cfg.fd_speech_cov_update_scale, 0.0, 1.0))
        if speech_activity >= 0.5:
            return base * scale
        return base

    def _noise_update_reason(self, *, target_active: bool, multi_target: bool = False) -> str:
        if self._noise_covariance_mode() == "oracle_non_target_residual":
            return "oracle_non_target_residual"
        if self._noise_covariance_is_frozen() and self._noise_covariance_frozen_bootstrap_active():
            return "estimated_target_subtractive_frozen_bootstrap"
        if self._noise_covariance_is_frozen():
            return "estimated_target_subtractive_frozen_hold"
        if bool(target_active):
            return "estimated_multitarget_subtractive" if multi_target else "estimated_target_subtractive"
        return "target_inactive_direct_mix"

    def mvdr(
        self,
        frame_mc: np.ndarray,
        doa_deg: float,
        covariance_alpha: float | None = None,
        *,
        target_active: bool = True,
        oracle_noise_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        x = np.asarray(frame_mc, dtype=np.float64)
        x_fft = self._analysis_block(x)  # (F, M)
        self._mvdr_frame_counter += 1
        oracle_noise_fft = None
        if oracle_noise_frame is not None:
            oracle_noise_fft, next_prev_noise = self._analysis_block_with_state(oracle_noise_frame, self._prev_noise_mc)
            self._prev_noise_mc = next_prev_noise
        refresh_reason = self._refresh_reason(doa_deg=float(doa_deg), target_active=bool(target_active))
        a = self._cached_steering if (refresh_reason is None and self._cached_steering is not None) else _steering_vector_f_domain(
            doa_deg=doa_deg,
            n_fft=self.analysis_n,
            fs=self.cfg.sample_rate_hz,
            mic_geometry_xyz=self.mic_geometry_xyz,
            sound_speed_m_s=self.cfg.sound_speed_m_s,
        )  # (F, M)
        candidate_rnn = None
        if refresh_reason is not None:
            candidate_rnn = self._update_noise_covariance(
                self.rnn_mvdr_noise,
                x_fft,
                a,
                float(self.cfg.fd_cov_ema_alpha if covariance_alpha is None else covariance_alpha),
                target_active=bool(target_active),
                oracle_noise_fft=oracle_noise_fft,
            )
            self.rnn_mvdr_noise = candidate_rnn
        refresh, refresh_reason = self._should_execute_refresh(refresh_reason=refresh_reason, candidate_rnn=candidate_rnn)
        self._last_weights_reused = bool(not refresh)
        if not refresh:
            self._beamformer_weights_reused_frames += 1
        if refresh:
            self._cached_mvdr_weights = self._solve_mvdr_weights(a)
            self._cached_last_solved_rnn = None if self.rnn_mvdr_noise is None else np.asarray(self.rnn_mvdr_noise, dtype=np.complex128).copy()
            self._beamformer_refresh_executed += 1
            self._maybe_record_beamformer_snapshot(
                beamforming_mode="mvdr_fd",
                target_doa_deg=float(doa_deg),
                weights=self._cached_mvdr_weights,
            )
            self._cached_steering = np.asarray(a, dtype=np.complex128)
            self._cached_target_doa_deg = float(doa_deg)
            self._cached_null_doa_deg = None
            self._cached_last_target_active = bool(target_active)
            self._last_noise_model_update = {
                "active": True,
                "sources": ("beamformer_rnn",),
                "reasons": (self._noise_update_reason(target_active=bool(target_active)),),
                "debug": {
                    "beamforming_mode": "mvdr_fd",
                    "covariance_alpha": float(self.cfg.fd_cov_ema_alpha if covariance_alpha is None else covariance_alpha),
                    "target_active": bool(target_active),
                    "refresh_reason": str(refresh_reason),
                    "rnn_dirty_stat": float(self._beamformer_last_dirty_stat),
                },
            }
        elif candidate_rnn is not None:
            self._last_noise_model_update = {
                "active": True,
                "sources": ("beamformer_rnn",),
                "reasons": (self._noise_update_reason(target_active=bool(target_active)), str(refresh_reason)),
                "debug": {
                    "beamforming_mode": "mvdr_fd",
                    "covariance_alpha": float(self.cfg.fd_cov_ema_alpha if covariance_alpha is None else covariance_alpha),
                    "target_active": bool(target_active),
                    "refresh_reason": str(refresh_reason),
                    "rnn_dirty_stat": float(self._beamformer_last_dirty_stat),
                },
            }
        else:
            self._last_noise_model_update = _inactive_noise_model_update()
        if self._cached_mvdr_weights is None:
            raise RuntimeError("MVDR weights were not initialized.")
        y_fft = np.einsum("fm,fm->f", np.conj(self._cached_mvdr_weights), x_fft)

        y, self._ola_tail_mvdr, self._ola_norm_mvdr = self._synthesis_block(y_fft, self._ola_tail_mvdr, self._ola_norm_mvdr)
        return y

    def lcmv_null(
        self,
        frame_mc: np.ndarray,
        target_doa_deg: float,
        null_doa_deg: float,
        covariance_alpha: float | None = None,
        *,
        target_active: bool = True,
        oracle_noise_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        x = np.asarray(frame_mc, dtype=np.float64)
        x_fft = self._analysis_block(x)
        self._mvdr_frame_counter += 1
        oracle_noise_fft = None
        if oracle_noise_frame is not None:
            oracle_noise_fft, next_prev_noise = self._analysis_block_with_state(oracle_noise_frame, self._prev_noise_mc)
            self._prev_noise_mc = next_prev_noise
        refresh_reason = self._refresh_reason(
            doa_deg=float(target_doa_deg),
            target_active=bool(target_active),
            null_doa_deg=float(null_doa_deg),
            requires_lcmv_weights=True,
        )
        a_target = self._cached_steering if (refresh_reason is None and self._cached_steering is not None) else _steering_vector_f_domain(
            doa_deg=target_doa_deg,
            n_fft=self.analysis_n,
            fs=self.cfg.sample_rate_hz,
            mic_geometry_xyz=self.mic_geometry_xyz,
            sound_speed_m_s=self.cfg.sound_speed_m_s,
        )
        a_null = None if (refresh_reason is None and self._cached_null_doa_deg is not None and self._cached_lcmv_weights is not None) else _steering_vector_f_domain(
            doa_deg=null_doa_deg,
            n_fft=self.analysis_n,
            fs=self.cfg.sample_rate_hz,
            mic_geometry_xyz=self.mic_geometry_xyz,
            sound_speed_m_s=self.cfg.sound_speed_m_s,
        )
        candidate_rnn = None
        if refresh_reason is not None:
            candidate_rnn = self._update_noise_covariance(
                self.rnn_mvdr_noise,
                x_fft,
                a_target,
                float(self.cfg.fd_cov_ema_alpha if covariance_alpha is None else covariance_alpha),
                target_active=bool(target_active),
                oracle_noise_fft=oracle_noise_fft,
            )
            self.rnn_mvdr_noise = candidate_rnn
        refresh, refresh_reason = self._should_execute_refresh(refresh_reason=refresh_reason, candidate_rnn=candidate_rnn)
        self._last_weights_reused = bool(not refresh)
        if not refresh:
            self._beamformer_weights_reused_frames += 1
        if refresh:
            if a_null is None:
                a_null = _steering_vector_f_domain(
                    doa_deg=null_doa_deg,
                    n_fft=self.analysis_n,
                    fs=self.cfg.sample_rate_hz,
                    mic_geometry_xyz=self.mic_geometry_xyz,
                    sound_speed_m_s=self.cfg.sound_speed_m_s,
                )
            self._cached_lcmv_weights = self._solve_lcmv_weights(a_target, a_null)
            self._cached_last_solved_rnn = None if self.rnn_mvdr_noise is None else np.asarray(self.rnn_mvdr_noise, dtype=np.complex128).copy()
            self._beamformer_refresh_executed += 1
            self._maybe_record_beamformer_snapshot(
                beamforming_mode="lcmv_null",
                target_doa_deg=float(target_doa_deg),
                null_doa_deg=float(null_doa_deg),
                weights=self._cached_lcmv_weights,
            )
            self._cached_steering = np.asarray(a_target, dtype=np.complex128)
            self._cached_target_doa_deg = float(target_doa_deg)
            self._cached_null_doa_deg = float(null_doa_deg)
            self._cached_last_target_active = bool(target_active)
            self._last_noise_model_update = {
                "active": True,
                "sources": ("beamformer_rnn",),
                "reasons": (self._noise_update_reason(target_active=bool(target_active)), "lcmv_active"),
                "debug": {
                    "beamforming_mode": "lcmv_null",
                    "covariance_alpha": float(self.cfg.fd_cov_ema_alpha if covariance_alpha is None else covariance_alpha),
                    "target_active": bool(target_active),
                    "null_doa_deg": float(null_doa_deg),
                    "refresh_reason": str(refresh_reason),
                    "rnn_dirty_stat": float(self._beamformer_last_dirty_stat),
                },
            }
        elif candidate_rnn is not None:
            self._last_noise_model_update = {
                "active": True,
                "sources": ("beamformer_rnn",),
                "reasons": (self._noise_update_reason(target_active=bool(target_active)), "lcmv_active", str(refresh_reason)),
                "debug": {
                    "beamforming_mode": "lcmv_null",
                    "covariance_alpha": float(self.cfg.fd_cov_ema_alpha if covariance_alpha is None else covariance_alpha),
                    "target_active": bool(target_active),
                    "null_doa_deg": float(null_doa_deg),
                    "refresh_reason": str(refresh_reason),
                    "rnn_dirty_stat": float(self._beamformer_last_dirty_stat),
                },
            }
        else:
            self._last_noise_model_update = _inactive_noise_model_update()
        if self._cached_lcmv_weights is None:
            raise RuntimeError("LCMV weights were not initialized.")
        y_fft = np.einsum("fm,fm->f", np.conj(self._cached_lcmv_weights), x_fft)

        y, self._ola_tail_mvdr, self._ola_norm_mvdr = self._synthesis_block(y_fft, self._ola_tail_mvdr, self._ola_norm_mvdr)
        return y

    def lcmv_top2(
        self,
        frame_mc: np.ndarray,
        primary_doa_deg: float,
        secondary_doa_deg: float,
        covariance_alpha: float | None = None,
        *,
        target_active: bool = True,
        oracle_noise_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        x = np.asarray(frame_mc, dtype=np.float64)
        x_fft = self._analysis_block(x)
        self._mvdr_frame_counter += 1
        oracle_noise_fft = None
        if oracle_noise_frame is not None:
            oracle_noise_fft, next_prev_noise = self._analysis_block_with_state(oracle_noise_frame, self._prev_noise_mc)
            self._prev_noise_mc = next_prev_noise
        refresh_reason = self._refresh_reason(
            doa_deg=float(primary_doa_deg),
            target_active=bool(target_active),
            secondary_doa_deg=float(secondary_doa_deg),
            requires_lcmv_weights=True,
        )
        a_primary = self._cached_steering if (refresh_reason is None and self._cached_steering is not None) else _steering_vector_f_domain(
            doa_deg=primary_doa_deg,
            n_fft=self.analysis_n,
            fs=self.cfg.sample_rate_hz,
            mic_geometry_xyz=self.mic_geometry_xyz,
            sound_speed_m_s=self.cfg.sound_speed_m_s,
        )
        a_secondary = _steering_vector_f_domain(
            doa_deg=secondary_doa_deg,
            n_fft=self.analysis_n,
            fs=self.cfg.sample_rate_hz,
            mic_geometry_xyz=self.mic_geometry_xyz,
            sound_speed_m_s=self.cfg.sound_speed_m_s,
        )
        candidate_rnn = None
        if refresh_reason is not None:
            candidate_rnn = self._update_noise_covariance_multi(
                self.rnn_mvdr_noise,
                x_fft,
                [a_primary, a_secondary],
                float(self.cfg.fd_cov_ema_alpha if covariance_alpha is None else covariance_alpha),
                target_active=bool(target_active),
                oracle_noise_fft=oracle_noise_fft,
            )
            self.rnn_mvdr_noise = candidate_rnn
        refresh, refresh_reason = self._should_execute_refresh(refresh_reason=refresh_reason, candidate_rnn=candidate_rnn)
        self._last_weights_reused = bool(not refresh)
        if not refresh:
            self._beamformer_weights_reused_frames += 1
        if refresh:
            self._cached_lcmv_weights = self._solve_lcmv_top2_weights(a_primary, a_secondary)
            self._cached_last_solved_rnn = None if self.rnn_mvdr_noise is None else np.asarray(self.rnn_mvdr_noise, dtype=np.complex128).copy()
            self._beamformer_refresh_executed += 1
            self._maybe_record_beamformer_snapshot(
                beamforming_mode="lcmv_top2_tracked",
                target_doa_deg=float(primary_doa_deg),
                secondary_doa_deg=float(secondary_doa_deg),
                weights=self._cached_lcmv_weights,
            )
            self._cached_steering = np.asarray(a_primary, dtype=np.complex128)
            self._cached_target_doa_deg = float(primary_doa_deg)
            self._cached_secondary_target_doa_deg = float(secondary_doa_deg)
            self._cached_null_doa_deg = None
            self._cached_last_target_active = bool(target_active)
            self._last_noise_model_update = {
                "active": True,
                "sources": ("beamformer_rnn",),
                "reasons": (self._noise_update_reason(target_active=bool(target_active), multi_target=True), "lcmv_top2_active"),
                "debug": {
                    "beamforming_mode": "lcmv_top2_tracked",
                    "covariance_alpha": float(self.cfg.fd_cov_ema_alpha if covariance_alpha is None else covariance_alpha),
                    "target_active": bool(target_active),
                    "primary_doa_deg": float(primary_doa_deg),
                    "secondary_doa_deg": float(secondary_doa_deg),
                    "refresh_reason": str(refresh_reason),
                    "rnn_dirty_stat": float(self._beamformer_last_dirty_stat),
                },
            }
        elif candidate_rnn is not None:
            self._last_noise_model_update = {
                "active": True,
                "sources": ("beamformer_rnn",),
                "reasons": (self._noise_update_reason(target_active=bool(target_active), multi_target=True), "lcmv_top2_active", str(refresh_reason)),
                "debug": {
                    "beamforming_mode": "lcmv_top2_tracked",
                    "covariance_alpha": float(self.cfg.fd_cov_ema_alpha if covariance_alpha is None else covariance_alpha),
                    "target_active": bool(target_active),
                    "primary_doa_deg": float(primary_doa_deg),
                    "secondary_doa_deg": float(secondary_doa_deg),
                    "refresh_reason": str(refresh_reason),
                    "rnn_dirty_stat": float(self._beamformer_last_dirty_stat),
                },
            }
        else:
            self._last_noise_model_update = _inactive_noise_model_update()
        if self._cached_lcmv_weights is None:
            raise RuntimeError("LCMV top2 weights were not initialized.")
        y_fft = np.einsum("fm,fm->f", np.conj(self._cached_lcmv_weights), x_fft)
        y, self._ola_tail_mvdr, self._ola_norm_mvdr = self._synthesis_block(y_fft, self._ola_tail_mvdr, self._ola_norm_mvdr)
        return y

    def lcmv_target_band(
        self,
        frame_mc: np.ndarray,
        target_doa_deg: float,
        covariance_alpha: float | None = None,
        *,
        target_active: bool = True,
        oracle_noise_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        x = np.asarray(frame_mc, dtype=np.float64)
        x_fft = self._analysis_block(x)
        self._mvdr_frame_counter += 1
        oracle_noise_fft = None
        if oracle_noise_frame is not None:
            oracle_noise_fft, next_prev_noise = self._analysis_block_with_state(oracle_noise_frame, self._prev_noise_mc)
            self._prev_noise_mc = next_prev_noise
        refresh_reason = self._refresh_reason(
            doa_deg=float(target_doa_deg),
            target_active=bool(target_active),
            requires_lcmv_weights=True,
        )
        band_width_deg = float(max(0.0, getattr(self.cfg, "robust_target_band_width_deg", 10.0)))
        target_steering = self._cached_steering if (refresh_reason is None and self._cached_steering is not None) else _steering_vector_f_domain(
            doa_deg=target_doa_deg,
            n_fft=self.analysis_n,
            fs=self.cfg.sample_rate_hz,
            mic_geometry_xyz=self.mic_geometry_xyz,
            sound_speed_m_s=self.cfg.sound_speed_m_s,
        )
        candidate_rnn = None
        if refresh_reason is not None:
            candidate_rnn = self._update_noise_covariance(
                self.rnn_mvdr_noise,
                x_fft,
                target_steering,
                float(self.cfg.fd_cov_ema_alpha if covariance_alpha is None else covariance_alpha),
                target_active=bool(target_active),
                oracle_noise_fft=oracle_noise_fft,
            )
            self.rnn_mvdr_noise = candidate_rnn
        refresh, refresh_reason = self._should_execute_refresh(refresh_reason=refresh_reason, candidate_rnn=candidate_rnn)
        self._last_weights_reused = bool(not refresh)
        if not refresh:
            self._beamformer_weights_reused_frames += 1
        if refresh:
            if band_width_deg <= 0.0:
                solved_weights = self._solve_lcmv_target_band_weights([target_steering])
            else:
                offsets = (-band_width_deg, 0.0, band_width_deg)
                steerings = [
                    _steering_vector_f_domain(
                        doa_deg=float(target_doa_deg + offset),
                        n_fft=self.analysis_n,
                        fs=self.cfg.sample_rate_hz,
                        mic_geometry_xyz=self.mic_geometry_xyz,
                        sound_speed_m_s=self.cfg.sound_speed_m_s,
                    )
                    for offset in offsets
                ]
                solved_weights = self._solve_lcmv_target_band_weights(steerings)
            self._cached_lcmv_weights = self._apply_lcmv_weight_smoothing(solved_weights, target_steering)
            self._cached_last_solved_rnn = None if self.rnn_mvdr_noise is None else np.asarray(self.rnn_mvdr_noise, dtype=np.complex128).copy()
            self._beamformer_refresh_executed += 1
            self._maybe_record_beamformer_snapshot(
                beamforming_mode="lcmv_target_band",
                target_doa_deg=float(target_doa_deg),
                target_band_width_deg=float(band_width_deg),
                weights=self._cached_lcmv_weights,
            )
            self._cached_steering = np.asarray(target_steering, dtype=np.complex128)
            self._cached_target_doa_deg = float(target_doa_deg)
            self._cached_null_doa_deg = None
            self._cached_secondary_target_doa_deg = None
            self._cached_last_target_active = bool(target_active)
            self._last_noise_model_update = {
                "active": True,
                "sources": ("beamformer_rnn",),
                "reasons": (self._noise_update_reason(target_active=bool(target_active)), "lcmv_target_band_active"),
                "debug": {
                    "beamforming_mode": "lcmv_target_band",
                    "covariance_alpha": float(self.cfg.fd_cov_ema_alpha if covariance_alpha is None else covariance_alpha),
                    "target_active": bool(target_active),
                    "target_doa_deg": float(target_doa_deg),
                    "target_band_width_deg": float(band_width_deg),
                    "refresh_reason": str(refresh_reason),
                    "rnn_dirty_stat": float(self._beamformer_last_dirty_stat),
                },
            }
        elif candidate_rnn is not None:
            self._last_noise_model_update = {
                "active": True,
                "sources": ("beamformer_rnn",),
                "reasons": (self._noise_update_reason(target_active=bool(target_active)), "lcmv_target_band_active", str(refresh_reason)),
                "debug": {
                    "beamforming_mode": "lcmv_target_band",
                    "covariance_alpha": float(self.cfg.fd_cov_ema_alpha if covariance_alpha is None else covariance_alpha),
                    "target_active": bool(target_active),
                    "target_doa_deg": float(target_doa_deg),
                    "target_band_width_deg": float(band_width_deg),
                    "refresh_reason": str(refresh_reason),
                    "rnn_dirty_stat": float(self._beamformer_last_dirty_stat),
                },
            }
        else:
            self._last_noise_model_update = _inactive_noise_model_update()
        if self._cached_lcmv_weights is None:
            raise RuntimeError("LCMV target-band weights were not initialized.")
        y_fft = np.einsum("fm,fm->f", np.conj(self._cached_lcmv_weights), x_fft)
        y, self._ola_tail_mvdr, self._ola_norm_mvdr = self._synthesis_block(y_fft, self._ola_tail_mvdr, self._ola_norm_mvdr)
        return y

    def gsc(self, frame_mc: np.ndarray, doa_deg: float, speech_activity: float = 0.0) -> np.ndarray:
        x = np.asarray(frame_mc, dtype=np.float64)
        x_fft = self._analysis_block(x)  # (F, M)
        self.rnn_gsc = self._update_covariance(self.rnn_gsc, x_fft, self._cov_alpha_from_activity(speech_activity))
        self._last_noise_model_update = {
            "active": True,
            "sources": ("beamformer_rnn",),
            "reasons": ("gsc_adaptive_noise_covariance",),
            "debug": {
                "beamforming_mode": "gsc_fd",
                "speech_activity": float(speech_activity),
                "covariance_alpha": float(self._cov_alpha_from_activity(speech_activity)),
            },
        }
        a = _steering_vector_f_domain(
            doa_deg=doa_deg,
            n_fft=self.analysis_n,
            fs=self.cfg.sample_rate_hz,
            mic_geometry_xyz=self.mic_geometry_xyz,
            sound_speed_m_s=self.cfg.sound_speed_m_s,
        )  # (F, M)

        f_bins = x_fft.shape[0]
        y_fft = np.zeros(f_bins, dtype=np.complex128)
        eye = np.eye(self.n_mics, dtype=np.complex128)
        for f in range(f_bins):
            af = a[f].reshape(-1, 1)
            af = af / (np.linalg.norm(af) + 1e-10)
            q, _ = np.linalg.qr(af, mode="complete")
            b = q[:, 1:]

            wq = af / (af.conj().T @ af + 1e-10)
            if b.shape[1] == 0:
                wf = wq
            else:
                r = self.rnn_gsc[f] + (1e-8 * eye)
                denom = b.conj().T @ r @ b
                num = b.conj().T @ r @ wq
                wa = np.linalg.pinv(denom + 1e-6 * np.eye(denom.shape[0])) @ num
                wf = wq - b @ wa
            y_fft[f] = (wf.conj().T @ x_fft[f].reshape(-1, 1))[0, 0]

        y, self._ola_tail_gsc, self._ola_norm_gsc = self._synthesis_block(y_fft, self._ola_tail_gsc, self._ola_norm_gsc)
        return y


class _PostFilterState:
    def __init__(self, frame_samples: int, cfg: PipelineConfig, *, estimator: str = "wiener_dd"):
        self.n = int(frame_samples)
        self.fft_n = max(4, 2 * self.n)
        self.cfg = cfg
        self.estimator = str(estimator).strip().lower()
        self._win = np.sqrt(np.hanning(self.fft_n)).astype(np.float64)
        self._prev_in = np.zeros(self.n, dtype=np.float64)
        self._ola_tail = np.zeros(self.n, dtype=np.float64)
        self.noise_psd: np.ndarray | None = None
        self.speech_psd: np.ndarray | None = None
        self.gain_prev: np.ndarray | None = None
        self.post_snr_prev: np.ndarray | None = None
        self._noise_bootstrap_frames_seen = 0
        self._noise_bootstrap_target_frames = 5
        self.last_noise_model_update: dict = _inactive_noise_model_update()

    def _smooth_gain_frequency(self, gain: np.ndarray) -> np.ndarray:
        radius = max(0, int(self.cfg.postfilter_freq_smoothing_bins))
        if radius <= 0:
            return gain
        width = (2 * radius) + 1
        positions = np.arange(width, dtype=np.float64) - float(radius)
        kernel = np.exp(-0.5 * (positions / max(float(radius), 1.0)) ** 2)
        kernel /= np.sum(kernel)
        return np.convolve(gain, kernel, mode="same")

    def process(
        self,
        frame: np.ndarray,
        speech_activity: float,
        *,
        target_activity_active: bool = False,
        external_noise_psd: np.ndarray | None = None,
    ) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float64).reshape(-1)
        block = np.concatenate([self._prev_in, x], axis=0)
        self._prev_in = x.copy()
        xw = block * self._win
        x_fft = np.fft.rfft(xw, n=self.fft_n)
        psd = np.abs(x_fft) ** 2

        external_noise = None if external_noise_psd is None else np.asarray(external_noise_psd, dtype=np.float64).reshape(-1)
        use_external_noise = bool(
            str(getattr(self.cfg, "postfilter_noise_source", "tracked_mono")).strip().lower() == "beamformer_rnn_output"
            and external_noise is not None
            and external_noise.shape == psd.shape
            and np.all(np.isfinite(external_noise))
        )
        if self.noise_psd is None:
            self.noise_psd = psd.copy() if not use_external_noise else np.maximum(external_noise, 1e-12)
        if self.speech_psd is None:
            self.speech_psd = np.maximum(psd * 0.25, 1e-12)
        if self.gain_prev is None:
            self.gain_prev = np.ones_like(psd, dtype=np.float64)
        if self.post_snr_prev is None:
            self.post_snr_prev = np.maximum(psd / np.maximum(self.noise_psd, 1e-12) - 1.0, 0.0)

        oversub_alpha = float(max(self.cfg.postfilter_oversubtraction_alpha, 0.0))
        spectral_floor_beta = float(np.clip(self.cfg.postfilter_spectral_floor_beta, 1e-6, 1.0))
        n_alpha = float(np.clip(self.cfg.postfilter_noise_ema_alpha, 0.0, 1.0))
        s_alpha = float(np.clip(self.cfg.postfilter_speech_ema_alpha, 0.0, 1.0))
        g_alpha = float(np.clip(self.cfg.postfilter_gain_ema_alpha, 0.0, 1.0))
        dd_alpha = float(np.clip(self.cfg.postfilter_dd_alpha, 0.0, 0.999))
        floor = float(np.clip(self.cfg.postfilter_gain_floor, 0.05, 1.0))
        gain_max_step_db = float(max(0.0, self.cfg.postfilter_gain_max_step_db))

        output_rms = float(np.sqrt(np.mean(x**2) + 1e-12))
        output_activity = float(np.clip((output_rms - 0.003) / 0.02, 0.0, 1.0))
        posterior_snr_hint = float(np.clip(np.mean(psd / np.maximum(self.noise_psd, 1e-12) - 1.0) / 6.0, 0.0, 1.0))
        speech_presence = float(np.clip(max(float(speech_activity), output_activity, posterior_snr_hint), 0.0, 1.0))
        allow_noise_update = (not bool(target_activity_active)) and (float(speech_activity) < 0.2) and (not use_external_noise)
        noise_alpha_eff = 0.0
        update_reason = "wiener_noise_psd_hold_speech"
        if use_external_noise:
            self.noise_psd = np.maximum(external_noise, 1e-12)
            update_reason = "beamformer_rnn_output_psd"
        elif allow_noise_update:
            if self._noise_bootstrap_frames_seen < self._noise_bootstrap_target_frames:
                if self._noise_bootstrap_frames_seen == 0:
                    self.noise_psd = psd.copy()
                else:
                    self.noise_psd = np.minimum(self.noise_psd, psd)
                self._noise_bootstrap_frames_seen += 1
                update_reason = "wiener_noise_psd_bootstrap"
            else:
                noise_alpha_eff = n_alpha
                noise_observation = np.minimum(psd, self.noise_psd * (1.0 + (2.5 * noise_alpha_eff)) + 1e-12)
                self.noise_psd = (1.0 - noise_alpha_eff) * self.noise_psd + noise_alpha_eff * noise_observation
                update_reason = "wiener_noise_psd_ema"
        self.last_noise_model_update = {
            "active": bool(allow_noise_update),
            "sources": ("postfilter_noise",),
            "reasons": (update_reason,),
            "debug": {
                "speech_presence": float(speech_presence),
                "noise_alpha_eff": float(noise_alpha_eff),
                "speech_activity": float(speech_activity),
                "target_activity_active": bool(target_activity_active),
                "noise_update_applied": bool(allow_noise_update),
                "noise_source": "beamformer_rnn_output" if use_external_noise else "tracked_mono",
                "external_noise_psd_valid": bool(use_external_noise),
                "noise_bootstrap_frames_seen": int(self._noise_bootstrap_frames_seen),
                "noise_bootstrap_target_frames": int(self._noise_bootstrap_target_frames),
                "noise_bootstrap_complete": bool(self._noise_bootstrap_frames_seen >= self._noise_bootstrap_target_frames),
            },
        }

        effective_noise_psd = np.maximum(oversub_alpha * self.noise_psd, 1e-12)
        post_snr = np.maximum(psd / effective_noise_psd - 1.0, 0.0)
        dd_prior = dd_alpha * (self.gain_prev**2) * self.post_snr_prev + (1.0 - dd_alpha) * post_snr
        speech_observation = np.maximum(psd - effective_noise_psd, spectral_floor_beta * self.noise_psd)
        self.speech_psd = (1.0 - s_alpha) * self.speech_psd + s_alpha * speech_observation
        prior_snr = np.maximum(dd_prior, self.speech_psd / effective_noise_psd)
        if self.estimator == "log_mmse":
            gamma = np.maximum(psd / effective_noise_psd, 1.0)
            nu = np.clip((prior_snr / (1.0 + prior_snr)) * gamma, 1e-8, 1e6)
            gain = (prior_snr / (1.0 + prior_snr)) * np.exp(0.5 * exp1(nu))
        else:
            gain = prior_snr / (1.0 + prior_snr)
        gain = np.maximum(gain, max(floor, spectral_floor_beta))
        gain = self._smooth_gain_frequency(gain)
        if gain_max_step_db > 0.0:
            max_ratio = float(10.0 ** (gain_max_step_db / 20.0))
            gain = np.clip(gain, self.gain_prev / max_ratio, self.gain_prev * max_ratio)
        gain = (1.0 - g_alpha) * self.gain_prev + g_alpha * gain
        self.gain_prev = gain
        self.post_snr_prev = post_snr

        y_fft = x_fft * gain
        y_block = np.fft.irfft(y_fft, n=self.fft_n).real
        yw = y_block * self._win
        out = self._ola_tail + yw[: self.n]
        self._ola_tail = yw[self.n :].copy()
        return out.astype(np.float32, copy=False)


class _RNNoisePostFilter:
    def __init__(self, cfg: PipelineConfig):
        if PyRNNoise is None:
            raise RuntimeError("RNNoise postfilter requested but pyrnnoise is unavailable.")
        self.cfg = cfg
        self._backend_sample_rate_hz = 48000
        self._input_sample_rate_hz = int(cfg.sample_rate_hz)
        self._backend = PyRNNoise(self._backend_sample_rate_hz)
        self._frame_size = 480
        self._pending_in = np.zeros((0,), dtype=np.float32)
        self._pending_out = np.zeros((0,), dtype=np.float32)
        self._residual_ema_state = np.zeros((0,), dtype=np.float32)
        self._backend.channels = 1
        self._backend.dtype = np.int16
        cutoff_hz = float(max(getattr(cfg, "rnnoise_output_lowpass_cutoff_hz", 0.0), 0.0))
        self._output_lowpass_cutoff_hz = cutoff_hz
        self._output_lowpass_sos = (
            None
            if cutoff_hz <= 0.0 or cutoff_hz >= 0.5 * float(self._input_sample_rate_hz)
            else butter(6, cutoff_hz, btype="lowpass", fs=float(self._input_sample_rate_hz), output="sos")
        )
        notch_freq_hz = float(max(getattr(cfg, "rnnoise_output_notch_freq_hz", 0.0), 0.0))
        notch_q = float(max(getattr(cfg, "rnnoise_output_notch_q", 0.0), 0.0))
        if 0.0 < notch_freq_hz < 0.5 * float(self._input_sample_rate_hz) and notch_q > 0.0:
            b_notch, a_notch = iirnotch(notch_freq_hz, notch_q, fs=float(self._input_sample_rate_hz))
            self._output_notch_sos = tf2sos(b_notch, a_notch)
            self._output_notch_zi = np.zeros((self._output_notch_sos.shape[0], 2), dtype=np.float32)
        else:
            self._output_notch_sos = None
            self._output_notch_zi = None

    def process(self, frame: np.ndarray, speech_activity: float = 0.0) -> np.ndarray:
        del speech_activity
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        gain = float(10.0 ** (float(self.cfg.rnnoise_input_gain_db) / 20.0))
        x_in = (x * gain).astype(np.float32, copy=False)
        if self._input_sample_rate_hz != self._backend_sample_rate_hz:
            x_in = np.asarray(
                resample_poly(x_in, up=self._backend_sample_rate_hz, down=self._input_sample_rate_hz),
                dtype=np.float32,
            )
        expected_backend_samples = int(x_in.shape[0])
        self._pending_in = np.concatenate([self._pending_in, x_in], axis=0)
        parts: list[np.ndarray] = []
        while self._pending_in.shape[0] >= self._frame_size:
            chunk = self._pending_in[: self._frame_size]
            self._pending_in = self._pending_in[self._frame_size :]
            chunk_i16 = np.clip(np.round(chunk * 32768.0), -32768.0, 32767.0).astype(np.int16, copy=False)
            _vad, den = self._backend.denoise_frame(np.atleast_2d(chunk_i16), partial=False)
            den_arr = np.asarray(den, dtype=np.float32).reshape(-1)
            if den_arr.dtype.kind in {"i", "u"} or float(np.max(np.abs(den_arr))) > 1.5:
                den_arr = den_arr / 32768.0
            parts.append(den_arr)
        if parts:
            self._pending_out = np.concatenate([self._pending_out, np.concatenate(parts, axis=0)], axis=0)
        if self._pending_out.shape[0] < expected_backend_samples:
            out = np.pad(self._pending_out, (0, expected_backend_samples - self._pending_out.shape[0]))
            self._pending_out = np.zeros((0,), dtype=np.float32)
        else:
            out = self._pending_out[: expected_backend_samples]
            self._pending_out = self._pending_out[expected_backend_samples :]
        if self._input_sample_rate_hz != self._backend_sample_rate_hz:
            out = np.asarray(
                resample_poly(out, up=self._input_sample_rate_hz, down=self._backend_sample_rate_hz),
                dtype=np.float32,
            )
            if out.shape[0] > x.shape[0]:
                out = out[: x.shape[0]]
            elif out.shape[0] < x.shape[0]:
                out = np.pad(out, (0, x.shape[0] - out.shape[0]))
        if self._residual_ema_state.shape[0] != x.shape[0]:
            self._residual_ema_state = np.zeros((x.shape[0],), dtype=np.float32)
        residual = np.asarray(out - x, dtype=np.float32)
        residual_ema_enabled = bool(getattr(self.cfg, "rnnoise_residual_ema_enabled", False))
        residual_ema_alpha = float(np.clip(getattr(self.cfg, "rnnoise_residual_ema_alpha", 0.0), 0.0, 0.999))
        if residual_ema_enabled and residual_ema_alpha > 0.0:
            self._residual_ema_state = (
                (residual_ema_alpha * self._residual_ema_state)
                + ((1.0 - residual_ema_alpha) * residual)
            ).astype(np.float32, copy=False)
            out = np.asarray(x + self._residual_ema_state, dtype=np.float32)
        else:
            self._residual_ema_state = residual.astype(np.float32, copy=False)
        wet = float(np.clip(self.cfg.rnnoise_wet_mix, 0.0, 1.0))
        mixed = ((wet * out) + ((1.0 - wet) * x)).astype(np.float32, copy=False)
        if self._output_lowpass_sos is not None and mixed.shape[0] > 8:
            mixed = np.asarray(sosfiltfilt(self._output_lowpass_sos, mixed), dtype=np.float32)
        if self._output_notch_sos is not None and self._output_notch_zi is not None and mixed.shape[0] > 0:
            mixed, self._output_notch_zi = sosfilt(self._output_notch_sos, mixed, zi=self._output_notch_zi)
            mixed = np.asarray(mixed, dtype=np.float32)
        return mixed.astype(np.float32, copy=False)


class _CoherenceWienerPostFilter:
    def __init__(self, cfg: PipelineConfig, mic_geometry_xyz: np.ndarray):
        self.cfg = cfg
        self.mic_geometry_xyz = np.asarray(mic_geometry_xyz, dtype=np.float64)
        self._prev_gain: np.ndarray | None = None

    def process(self, frame_mc: np.ndarray, beamformed: np.ndarray, doa_deg: float) -> np.ndarray:
        x_mc = np.asarray(frame_mc, dtype=np.float64)
        y = np.asarray(beamformed, dtype=np.float64).reshape(-1)
        n = int(y.shape[0])
        win = np.sqrt(np.hanning(max(4, n))).astype(np.float64)
        x_fft = np.fft.rfft(x_mc * win[:, None], axis=0)
        y_fft = np.fft.rfft(y * win)
        steering = _steering_vector_f_domain(
            doa_deg=float(doa_deg),
            n_fft=n,
            fs=self.cfg.sample_rate_hz,
            mic_geometry_xyz=self.mic_geometry_xyz,
            sound_speed_m_s=self.cfg.sound_speed_m_s,
        )
        aligned = x_fft * np.conj(steering)
        auto_psd = np.mean(np.abs(aligned) ** 2, axis=1)
        pair_terms = []
        for i in range(aligned.shape[1]):
            for j in range(i + 1, aligned.shape[1]):
                denom = np.sqrt((np.abs(aligned[:, i]) ** 2) * (np.abs(aligned[:, j]) ** 2)) + 1e-12
                pair_terms.append(np.abs(aligned[:, i] * np.conj(aligned[:, j])) / denom)
        coherence = np.mean(pair_terms, axis=0) if pair_terms else np.ones_like(auto_psd)
        coherence = np.clip(coherence, 0.0, 1.0)
        noise_psd = np.maximum((1.0 - coherence) * auto_psd, 1e-10)
        speech_psd = np.maximum(np.abs(y_fft) ** 2 - noise_psd, 1e-10)
        wiener = speech_psd / (speech_psd + noise_psd + 1e-10)
        floor = float(np.clip(self.cfg.coherence_wiener_gain_floor, 0.05, 1.0))
        exponent = float(max(self.cfg.coherence_wiener_coherence_exponent, 0.1))
        temporal_alpha = float(np.clip(self.cfg.coherence_wiener_temporal_alpha, 0.0, 0.999))
        gain = floor + (1.0 - floor) * ((coherence**exponent) * wiener)
        if self._prev_gain is None:
            self._prev_gain = np.ones_like(gain)
        gain = (temporal_alpha * self._prev_gain) + ((1.0 - temporal_alpha) * gain)
        self._prev_gain = gain
        out = np.fft.irfft(y_fft * gain, n=n).real
        return out.astype(np.float32, copy=False)


class _VoiceBandpassPostFilter:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def process(self, frame: np.ndarray, speech_activity: float = 0.0) -> np.ndarray:
        del speech_activity
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        n = int(x.shape[0])
        if n <= 1:
            return x.astype(np.float32, copy=False)
        x_fft = np.fft.rfft(x, n=n)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(self.cfg.sample_rate_hz))
        # Cheap speech-focused band limit for quick real-data sanity checks.
        mask = (freqs >= 180.0) & (freqs <= 3600.0)
        y = np.fft.irfft(x_fft * mask.astype(np.float32), n=n).real
        return y.astype(np.float32, copy=False)


class _HybridPostFilter:
    def __init__(self, first, second):
        self._first = first
        self._second = second

    def process(self, frame: np.ndarray, speech_activity: float = 0.0) -> np.ndarray:
        first = self._first.process(frame, speech_activity=speech_activity)
        return self._second.process(first, speech_activity=speech_activity)


class _PostFilterRouter:
    def __init__(self, frame_samples: int, cfg: PipelineConfig, mic_geometry_xyz: np.ndarray):
        self.cfg = cfg
        self.method = str(getattr(cfg, "postfilter_method", "off")).strip().lower()
        self._wiener = _PostFilterState(frame_samples=frame_samples, cfg=cfg, estimator="wiener_dd")
        self._log_mmse = _PostFilterState(frame_samples=frame_samples, cfg=cfg, estimator="log_mmse")
        self._rnnoise = None
        self._coherence = None
        self._voice_bandpass = None
        if self.method in {"rnnoise", "wiener_then_rnnoise", "rnnoise_then_voice_bandpass"}:
            self._rnnoise = _RNNoisePostFilter(cfg)
        if self.method == "coherence_wiener":
            self._coherence = _CoherenceWienerPostFilter(cfg, mic_geometry_xyz)
        if self.method in {"voice_bandpass", "wiener_then_voice_bandpass", "rnnoise_then_voice_bandpass"}:
            self._voice_bandpass = _VoiceBandpassPostFilter(cfg)
        if self.method == "wiener_then_rnnoise":
            self._hybrid = _HybridPostFilter(self._wiener, self._rnnoise)
        elif self.method == "wiener_then_voice_bandpass":
            self._hybrid = _HybridPostFilter(self._wiener, self._voice_bandpass)
        elif self.method == "rnnoise_then_voice_bandpass":
            self._hybrid = _HybridPostFilter(self._rnnoise, self._voice_bandpass)
        else:
            self._hybrid = None
        self._last_noise_model_update: dict = _inactive_noise_model_update()

    def process_with_stages(
        self,
        beamformed: np.ndarray,
        *,
        speech_activity: float,
        target_activity_active: bool = False,
        frame_mc: np.ndarray | None = None,
        doa_deg: float | None = None,
        external_noise_psd: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        stages: dict[str, np.ndarray] = {}
        if not bool(self.cfg.postfilter_enabled) or self.method == "off":
            self._last_noise_model_update = _inactive_noise_model_update()
            out = np.asarray(beamformed, dtype=np.float32)
            stages["postfilter_output"] = out
            return out, stages
        if self.method == "wiener_dd":
            out = self._wiener.process(
                beamformed,
                speech_activity=speech_activity,
                target_activity_active=target_activity_active,
                external_noise_psd=external_noise_psd,
            )
            self._last_noise_model_update = _normalize_noise_model_update(self._wiener.last_noise_model_update)
            stages["post_wiener"] = np.asarray(out, dtype=np.float32)
            stages["postfilter_output"] = np.asarray(out, dtype=np.float32)
            return out, stages
        if self.method == "log_mmse":
            out = self._log_mmse.process(
                beamformed,
                speech_activity=speech_activity,
                target_activity_active=target_activity_active,
                external_noise_psd=external_noise_psd,
            )
            self._last_noise_model_update = _normalize_noise_model_update(self._log_mmse.last_noise_model_update)
            stages["post_wiener"] = np.asarray(out, dtype=np.float32)
            stages["postfilter_output"] = np.asarray(out, dtype=np.float32)
            return out, stages
        if self.method == "rnnoise":
            if self._rnnoise is None:
                raise RuntimeError("RNNoise postfilter not initialized.")
            self._last_noise_model_update = _inactive_noise_model_update()
            out = self._rnnoise.process(beamformed, speech_activity=speech_activity)
            stages["post_rnnoise"] = np.asarray(out, dtype=np.float32)
            stages["postfilter_output"] = np.asarray(out, dtype=np.float32)
            return out, stages
        if self.method == "coherence_wiener":
            if self._coherence is None or frame_mc is None or doa_deg is None:
                raise RuntimeError("coherence_wiener postfilter requires frame_mc and doa_deg.")
            self._last_noise_model_update = _inactive_noise_model_update()
            out = self._coherence.process(frame_mc, beamformed, doa_deg)
            stages["postfilter_output"] = np.asarray(out, dtype=np.float32)
            return out, stages
        if self.method == "voice_bandpass":
            if self._voice_bandpass is None:
                raise RuntimeError("voice_bandpass postfilter not initialized.")
            self._last_noise_model_update = _inactive_noise_model_update()
            out = self._voice_bandpass.process(beamformed, speech_activity=speech_activity)
            stages["post_bandpass"] = np.asarray(out, dtype=np.float32)
            stages["postfilter_output"] = np.asarray(out, dtype=np.float32)
            return out, stages
        if self.method == "wiener_then_voice_bandpass":
            if self._wiener is None or self._voice_bandpass is None:
                raise RuntimeError("Wiener->voice_bandpass postfilter not initialized.")
            post_wiener = self._wiener.process(
                beamformed,
                speech_activity=speech_activity,
                target_activity_active=target_activity_active,
                external_noise_psd=external_noise_psd,
            )
            out = self._voice_bandpass.process(post_wiener, speech_activity=speech_activity)
            self._last_noise_model_update = _normalize_noise_model_update(self._wiener.last_noise_model_update)
            stages["post_wiener"] = np.asarray(post_wiener, dtype=np.float32)
            stages["post_bandpass"] = np.asarray(out, dtype=np.float32)
            stages["postfilter_output"] = np.asarray(out, dtype=np.float32)
            return out, stages
        if self.method == "rnnoise_then_voice_bandpass":
            if self._rnnoise is None or self._voice_bandpass is None:
                raise RuntimeError("RNNoise->voice_bandpass postfilter not initialized.")
            post_rnnoise = self._rnnoise.process(beamformed, speech_activity=speech_activity)
            out = self._voice_bandpass.process(post_rnnoise, speech_activity=speech_activity)
            self._last_noise_model_update = _inactive_noise_model_update()
            stages["post_rnnoise"] = np.asarray(post_rnnoise, dtype=np.float32)
            stages["post_bandpass"] = np.asarray(out, dtype=np.float32)
            stages["postfilter_output"] = np.asarray(out, dtype=np.float32)
            return out, stages
        if self.method == "wiener_then_rnnoise":
            if self._wiener is None or self._rnnoise is None:
                raise RuntimeError("Hybrid postfilter not initialized.")
            post_wiener = self._wiener.process(
                beamformed,
                speech_activity=speech_activity,
                target_activity_active=target_activity_active,
                external_noise_psd=external_noise_psd,
            )
            out = self._rnnoise.process(post_wiener, speech_activity=speech_activity)
            self._last_noise_model_update = _normalize_noise_model_update(self._wiener.last_noise_model_update)
            stages["post_wiener"] = np.asarray(post_wiener, dtype=np.float32)
            stages["post_rnnoise"] = np.asarray(out, dtype=np.float32)
            stages["postfilter_output"] = np.asarray(out, dtype=np.float32)
            return out, stages
        self._last_noise_model_update = _inactive_noise_model_update()
        out = np.asarray(beamformed, dtype=np.float32)
        stages["postfilter_output"] = out
        return out, stages

    def process(
        self,
        beamformed: np.ndarray,
        *,
        speech_activity: float,
        target_activity_active: bool = False,
        frame_mc: np.ndarray | None = None,
        doa_deg: float | None = None,
        external_noise_psd: np.ndarray | None = None,
    ) -> np.ndarray:
        out, _stages = self.process_with_stages(
            beamformed,
            speech_activity=speech_activity,
            target_activity_active=target_activity_active,
            frame_mc=frame_mc,
            doa_deg=doa_deg,
            external_noise_psd=external_noise_psd,
        )
        return out

    def get_last_noise_model_update(self) -> dict:
        return _normalize_noise_model_update(self._last_noise_model_update)


class _PostFilterStageCore:
    def __init__(
        self,
        *,
        cfg: PipelineConfig,
        frame_samples: int,
        mic_geometry_xyz: np.ndarray,
        frame_sink: FrameSink,
        shared_state: SharedPipelineState,
    ):
        self._cfg = cfg
        self._sink = frame_sink
        self._state = shared_state
        self._rms_gain_ema = 1.0
        self._postfilter = _PostFilterRouter(
            frame_samples=frame_samples,
            cfg=cfg,
            mic_geometry_xyz=mic_geometry_xyz,
        )

    def process_packet(self, packet: FastPathAudioPacket) -> tuple[float, float, float]:
        t0 = perf_counter()
        beamformed = np.asarray(packet.beamformed_audio, dtype=np.float32).reshape(-1)
        input_source = str(getattr(self._cfg, "postfilter_input_source", "beamformed_mono")).strip().lower()
        if input_source == "raw_mix_mono" and packet.frame_mc is not None:
            out = np.mean(np.asarray(packet.frame_mc, dtype=np.float32), axis=1).astype(np.float32, copy=False)
        else:
            out = beamformed
        postfilter_stages: dict[str, np.ndarray] = {"post_beamforming": beamformed.copy()}
        if bool(self._cfg.postfilter_enabled):
            out, postfilter_stages = self._postfilter.process_with_stages(
                out,
                speech_activity=float(packet.speech_activity),
                target_activity_active=bool(packet.target_activity_state),
                frame_mc=None if packet.frame_mc is None else np.asarray(packet.frame_mc, dtype=np.float32),
                doa_deg=None if packet.target_doa_deg is None else float(packet.target_doa_deg),
                external_noise_psd=(
                    None
                    if packet.beamformer_output_noise_psd is None
                    else np.asarray(packet.beamformer_output_noise_psd, dtype=np.float32).reshape(-1)
                ),
            )
            postfilter_stages["post_beamforming"] = beamformed.copy()
        postfilter_noise_update = self._postfilter.get_last_noise_model_update()
        packet.postfilter_wiener_audio = (
            None
            if postfilter_stages.get("post_wiener") is None
            else np.asarray(postfilter_stages["post_wiener"], dtype=np.float32).reshape(-1).copy()
        )
        packet.postfilter_rnnoise_audio = (
            None
            if postfilter_stages.get("post_rnnoise") is None
            else np.asarray(postfilter_stages["post_rnnoise"], dtype=np.float32).reshape(-1).copy()
        )
        packet.postfilter_bandpass_audio = (
            None
            if postfilter_stages.get("post_bandpass") is None
            else np.asarray(postfilter_stages["post_bandpass"], dtype=np.float32).reshape(-1).copy()
        )
        packet.postfilter_output_audio = np.asarray(postfilter_stages.get("postfilter_output", out), dtype=np.float32).reshape(-1).copy()
        combined_noise_update = _merge_noise_model_updates(
            {
                "active": bool(packet.noise_model_update_active),
                "sources": packet.noise_model_update_sources,
                "reasons": packet.noise_model_update_reasons,
                "debug": {} if packet.noise_model_update_debug is None else packet.noise_model_update_debug,
            },
            postfilter_noise_update,
        )
        packet.noise_model_update_active = bool(combined_noise_update["active"])
        packet.noise_model_update_sources = tuple(combined_noise_update["sources"])
        packet.noise_model_update_reasons = tuple(combined_noise_update["reasons"])
        packet.noise_model_update_debug = dict(combined_noise_update["debug"])
        postfilter_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        out, self._rms_gain_ema = _apply_output_safety(out, self._cfg, self._rms_gain_ema)
        safety_ms = (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        self._state.publish_noise_model_update_snapshot(
            NoiseModelUpdateSnapshot(
                timestamp_ms=(1000.0 * float(packet.start_sample) / max(int(packet.sample_rate_hz), 1)),
                active=bool(packet.noise_model_update_active),
                sources=tuple(packet.noise_model_update_sources),
                reasons=tuple(packet.noise_model_update_reasons),
                debug=None if packet.noise_model_update_debug is None else dict(packet.noise_model_update_debug),
            )
        )
        self._sink(out)
        sink_ms = (perf_counter() - t0) * 1000.0
        return float(postfilter_ms), float(safety_ms), float(sink_ms)


def _soft_clip(x: np.ndarray, drive: float) -> np.ndarray:
    d = max(float(drive), 1e-6)
    y = np.tanh(d * np.asarray(x, dtype=np.float64))
    return y.astype(np.float32, copy=False)


def _apply_output_safety(out: np.ndarray, cfg: PipelineConfig, rms_gain_ema: float) -> tuple[np.ndarray, float]:
    y = np.asarray(out, dtype=np.float32)
    next_rms_gain = float(rms_gain_ema)
    if cfg.output_normalization_enabled and cfg.output_target_rms is not None:
        cur_rms = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2) + 1e-12))
        target_rms = max(float(cfg.output_target_rms), 1e-6)
        desired_gain = target_rms / max(cur_rms, 1e-6)
        max_gain = float(10.0 ** (float(cfg.output_rms_max_gain_db) / 20.0))
        if cfg.output_allow_amplification:
            desired_gain = float(np.clip(desired_gain, 1.0 / max_gain, max_gain))
        else:
            # Attenuation-only normalization: reduce noise/pumping but avoid speaker amplification.
            desired_gain = float(np.clip(desired_gain, 1.0 / max_gain, 1.0))
        alpha = float(np.clip(cfg.output_rms_ema_alpha, 0.0, 1.0))
        next_rms_gain = (1.0 - alpha) * next_rms_gain + alpha * desired_gain
        y = y * float(next_rms_gain)

    if cfg.output_soft_clip_enabled:
        y = _soft_clip(y, drive=cfg.output_soft_clip_drive)
    return y, next_rms_gain


class PostFilterWorker(threading.Thread):
    def __init__(
        self,
        *,
        config: PipelineConfig,
        shared_state: SharedPipelineState,
        frame_sink: FrameSink,
        stop_event: threading.Event,
        mic_geometry_xyz: np.ndarray,
        packet_queue: "queue.Queue[FastPathAudioPacket | None] | None" = None,
        packet_source: Callable[[], FastPathAudioPacket | None] | None = None,
    ):
        super().__init__(name="PostFilterWorker", daemon=True)
        self._cfg = config
        self._state = shared_state
        self._stop = stop_event
        self._packet_queue = packet_queue
        self._packet_source = packet_source
        frame_samples = max(1, int(config.sample_rate_hz * config.fast_frame_ms / 1000))
        self._core = _PostFilterStageCore(
            cfg=config,
            frame_samples=frame_samples,
            mic_geometry_xyz=_center_mic_geometry_xyz(mic_geometry_xyz),
            frame_sink=frame_sink,
            shared_state=shared_state,
        )

    def _next_packet(self) -> FastPathAudioPacket | None:
        if self._packet_source is not None:
            return self._packet_source()
        if self._packet_queue is None:
            return None
        while not self._stop.is_set():
            try:
                return self._packet_queue.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    def run(self) -> None:
        while not self._stop.is_set():
            packet = self._next_packet()
            if packet is None:
                break
            queue_wait_ms = 0.0
            if packet.queue_enqueue_t is not None:
                queue_wait_ms = max(0.0, (perf_counter() - float(packet.queue_enqueue_t)) * 1000.0)
            packet.queue_wait_ms = float(queue_wait_ms)
            packet.postfilter_start_t = perf_counter()
            with Timer() as t:
                _postfilter_ms, safety_ms, sink_ms = self._core.process_packet(packet)
            packet.postfilter_end_t = perf_counter()
            self._state.record_postfilter_stage(t.elapsed_ms)
            queue_depth = 0 if self._packet_queue is None else int(self._packet_queue.qsize())
            end_to_end_latency_ms = max(0.0, (packet.postfilter_end_t - float(packet.capture_t_monotonic)) * 1000.0)
            self._state.record_pipeline_latency(
                queue_wait_ms=float(queue_wait_ms),
                end_to_end_latency_ms=float(end_to_end_latency_ms),
                queue_depth=int(queue_depth),
            )
            self._state.incr_fast_stage_times(
                srp_ms=0.0,
                beamform_ms=0.0,
                safety_ms=float(safety_ms),
                sink_ms=float(sink_ms),
                enqueue_ms=0.0,
            )


class FastPathWorker(threading.Thread):
    def __init__(
        self,
        *,
        config: PipelineConfig,
        shared_state: SharedPipelineState,
        frame_source: FrameSource,
        frame_sink: FrameSink,
        slow_queue: "queue.Queue[np.ndarray | None]",
        mic_geometry_xyz: np.ndarray,
        stop_event: threading.Event,
        srp_override_provider: Callable[[int, float], SRPPeakSnapshot | None] | None = None,
        target_activity_override_provider: Callable[[int, float], float | None] | None = None,
        oracle_noise_frame_provider: Callable[[int, float], np.ndarray | None] | None = None,
        postfilter_queue: "queue.Queue[FastPathAudioPacket | None] | None" = None,
        frame_packet_sink: Callable[[FastPathAudioPacket], None] | None = None,
    ):
        super().__init__(name="FastPathWorker", daemon=True)
        self._cfg = config
        self._state = shared_state
        self._source = frame_source
        self._sink = frame_sink
        self._slow_queue = slow_queue
        self._postfilter_queue = postfilter_queue
        self._frame_packet_sink = frame_packet_sink
        self._mic_geometry_xyz = _center_mic_geometry_xyz(mic_geometry_xyz)
        self._stop = stop_event
        self._split_runtime_mode = str(getattr(config, "split_runtime_mode", "monolithic")).strip().lower()
        dominant_lock_mode = str(config.tracking_mode).strip().lower() == "dominant_lock_v1"
        single_source_backend = (
            config.localization_backend in {
                "capon_1src",
                "capon_mvdr_refine_1src",
                "music_1src",
            }
            or bool(config.single_source_mode_enabled)
            or dominant_lock_mode
        )
        tracker_window_ms = config.single_source_window_ms if single_source_backend else config.localization_window_ms
        tracker_grid = config.single_source_grid_size if single_source_backend else config.localization_grid_size
        tracker_freq = (
            (config.single_source_freq_min_hz, config.single_source_freq_max_hz)
            if single_source_backend
            else (config.srp_freq_min_hz, config.srp_freq_max_hz)
        )
        tracker_max_sources = 1 if (single_source_backend or bool(config.assume_single_speaker)) else config.srp_max_sources
        self._tracker = SRPPeakTracker(
            mic_pos=self._mic_geometry_xyz if self._mic_geometry_xyz.shape[0] == 3 else self._mic_geometry_xyz.T,
            fs=config.sample_rate_hz,
            window_ms=tracker_window_ms,
            nfft=config.srp_nfft,
            overlap=config.srp_overlap,
            freq_range=tracker_freq,
            max_sources=tracker_max_sources,
            prior_enabled=config.srp_prior_enabled,
            min_score=config.srp_peak_min_score,
            ema_alpha=config.srp_peak_ema_alpha,
            hysteresis_margin=config.srp_peak_hysteresis_margin,
            match_tolerance_deg=config.srp_peak_match_tolerance_deg,
            hold_frames=config.srp_peak_hold_frames,
            max_step_deg=config.srp_peak_max_step_deg,
            score_decay=config.srp_peak_score_decay,
            backend=config.localization_backend,
            grid_size=tracker_grid,
            min_peak_separation_deg=config.localization_min_peak_separation_deg,
            small_aperture_bias=config.localization_small_aperture_bias,
            sound_speed_m_s=config.sound_speed_m_s,
            tracking_mode=config.tracking_mode,
            pair_selection_mode=str(config.localization_pair_selection_mode),
            vad_enabled=bool(config.localization_vad_enabled),
            capon_spectrum_ema_alpha=float(config.capon_spectrum_ema_alpha),
            capon_peak_min_sharpness=float(config.capon_peak_min_sharpness),
            capon_peak_min_margin=float(config.capon_peak_min_margin),
            capon_hold_frames=int(config.capon_hold_frames),
            max_tracks=config.localization_max_tracks,
            max_assoc_distance_deg=config.localization_max_assoc_distance_deg,
            track_hold_frames=config.localization_track_hold_frames,
            track_kill_frames=config.localization_track_kill_frames,
            new_track_min_confidence=config.localization_new_track_min_confidence,
            track_confidence_decay=config.localization_track_confidence_decay,
            velocity_alpha=config.localization_velocity_alpha,
            angle_alpha=config.localization_angle_alpha,
            min_relative_peak_score=config.localization_min_relative_peak_score,
            min_peak_contrast=config.localization_min_peak_contrast,
            single_source_motion_filter_enabled=config.single_source_motion_filter_enabled,
            dominant_lock_acquire_min_score=config.dominant_lock_acquire_min_score,
            dominant_lock_acquire_confirm_frames=config.dominant_lock_acquire_confirm_frames,
            dominant_lock_stay_radius_deg=config.dominant_lock_stay_radius_deg,
            dominant_lock_update_alpha=config.dominant_lock_update_alpha,
            dominant_lock_max_step_deg=config.dominant_lock_max_step_deg,
            dominant_lock_hold_missing_frames=config.dominant_lock_hold_missing_frames,
            dominant_lock_unlock_after_missing_frames=config.dominant_lock_unlock_after_missing_frames,
            dominant_lock_challenger_min_score=config.dominant_lock_challenger_min_score,
            dominant_lock_challenger_margin=config.dominant_lock_challenger_margin,
            dominant_lock_challenger_consistency_deg=config.dominant_lock_challenger_consistency_deg,
            dominant_lock_switch_confirm_frames=config.dominant_lock_switch_confirm_frames,
            dominant_lock_switch_min_confidence=config.dominant_lock_switch_min_confidence,
        )
        self._frame_idx = 0
        self._rms_gain_ema = 1.0
        self._smoothed_doa_by_speaker: dict[int, float] = {}
        self._smoothed_gain_by_speaker: dict[int, float] = {}
        self._single_active_localized_doa_deg: float | None = None
        self._single_active_localized_score: float = 0.0
        self._single_active_missing_frames: int = 0
        frame_samples = max(1, int(config.sample_rate_hz * config.fast_frame_ms / 1000))
        self._fd = _FDBufferedBeamformer(
            n_mics=self._mic_geometry_xyz.shape[1] if self._mic_geometry_xyz.shape[0] == 3 else self._mic_geometry_xyz.shape[0],
            frame_samples=frame_samples,
            cfg=config,
            mic_geometry_xyz=self._mic_geometry_xyz,
        )
        self._postfilter_stage = _PostFilterStageCore(
            cfg=config,
            frame_samples=frame_samples,
            mic_geometry_xyz=self._mic_geometry_xyz,
            frame_sink=self._sink,
            shared_state=self._state,
        )
        self._srp_override_provider = srp_override_provider
        self._target_activity_override_provider = target_activity_override_provider
        self._oracle_noise_frame_provider = oracle_noise_frame_provider
        self._own_voice_active = False
        self._own_voice_on_count = 0
        self._own_voice_off_count = 0
        self._target_activity_state = False
        self._target_activity_on_count = 0
        self._target_activity_off_count = 0
        self._target_activity_noise_floor = 1e-3
        self._target_activity_target_floor = 1e-3
        self._target_activity_blocker_floor = 1e-3
        self._target_activity_ratio_baseline_db = 0.0
        self._target_activity_calibration_frames = 0
        self._target_activity_bootstrap_complete = False
        self._target_activity_last_debug: dict[str, float | str | bool] = {}
        self._target_activity_last_score = 0.0
        self._target_activity_last_covariance_alpha = float(config.fd_cov_ema_alpha)
        self._target_activity_update_every_n_fast_frames = max(
            1,
            int(getattr(config, "target_activity_update_every_n_fast_frames", 1)),
        )
        self._last_target_doa_deg: float | None = None
        self._last_target_speaker_id: int | None = None
        self._last_multi_target_speaker_ids: tuple[int, ...] = ()
        self._last_multi_target_doas_deg: tuple[float, ...] = ()
        self._multi_target_hold_remaining_frames: int = 0
        self._focus_hold_remaining_frames: int = 0
        self._delay_sum_state_by_key: dict[str, dict[str, float | int | None]] = {}
        self._beamformer_snapshot_targets = tuple(
            int(v) for v in getattr(config, "beamformer_snapshot_frame_indices", ()) if int(v) > 0
        )
        self._beamformer_snapshot_target_set = set(self._beamformer_snapshot_targets)
        self._delay_sum_snapshot_trace: list[dict] = []
        backend = str(getattr(config, "target_activity_detector_backend", "webrtc_fused")).strip().lower()
        if backend == "silero_fused":
            self._target_activity_vad = SileroVADGate(
                sample_rate_hz=int(config.sample_rate_hz),
                frame_ms=max(10, int(config.fast_frame_ms)),
                hangover_frames=max(0, int(config.target_activity_vad_hangover_frames)),
            )
        else:
            self._target_activity_vad = WebRTCVADGate(
                sample_rate_hz=int(config.sample_rate_hz),
                mode=max(0, min(3, int(config.target_activity_vad_mode))),
                frame_ms=max(10, int(config.fast_frame_ms)),
                hangover_frames=max(0, int(config.target_activity_vad_hangover_frames)),
            )

        mode = self._target_activity_mode()
        beamforming_mode = str(self._cfg.beamforming_mode).strip().lower()
        if beamforming_mode in {"mvdr_fd", "lcmv_top2_tracked", "lcmv_target_band"} and mode is None:
            raise ValueError("Covariance beamforming requires target_activity_rnn_update_mode to be configured.")
        if mode == "oracle_target_activity" and self._target_activity_override_provider is None:
            raise ValueError("oracle_target_activity mode requires a target activity override provider.")
        if (
            beamforming_mode in {"mvdr_fd", "lcmv_top2_tracked", "lcmv_target_band"}
            and str(getattr(self._cfg, "fd_noise_covariance_mode", "estimated_target_subtractive")).strip().lower() == "oracle_non_target_residual"
            and self._oracle_noise_frame_provider is None
        ):
            raise ValueError("oracle_non_target_residual mode requires an oracle noise frame provider.")

    def get_beamformer_snapshot_trace(self) -> list[dict]:
        mode = str(self._cfg.beamforming_mode).strip().lower()
        if mode in {"delay_sum", "delay_sum_subtractive"}:
            return [dict(item) for item in self._delay_sum_snapshot_trace]
        return self._fd.get_beamformer_snapshot_trace()

    def get_beamformer_runtime_stats(self) -> dict[str, float | int]:
        return self._fd.get_beamformer_runtime_stats()

    def _maybe_record_delay_sum_snapshot(self, *, state_key: str, doa_deg: float) -> None:
        del state_key
        frame_idx = int(self._frame_idx) + 1
        if frame_idx not in self._beamformer_snapshot_target_set:
            return
        if any(int(item.get("frame_index", -1)) == frame_idx for item in self._delay_sum_snapshot_trace):
            return
        weights = (
            np.conj(
                _steering_vector_f_domain(
                    doa_deg=float(doa_deg),
                    n_fft=int(self._fd.analysis_n),
                    fs=int(self._cfg.sample_rate_hz),
                    mic_geometry_xyz=self._mic_geometry_xyz,
                    sound_speed_m_s=float(self._cfg.sound_speed_m_s),
                )
            )
            / max(1, int(self._fd.n_mics))
        )
        self._delay_sum_snapshot_trace.append(
            {
                "frame_index": frame_idx,
                "target_doa_deg": float(doa_deg),
                "weights_real": np.asarray(np.real(weights), dtype=np.float64).tolist(),
                "weights_imag": np.asarray(np.imag(weights), dtype=np.float64).tolist(),
            }
        )

    def _delay_sum_candidate_doa(self, *, state: dict[str, float | int | None], doa_deg: float) -> float:
        candidate = _norm_deg(float(doa_deg))
        if not bool(getattr(self._cfg, "delay_sum_use_smoothed_doa", True)):
            state["smoothed_doa_deg"] = candidate
            return candidate
        prev_smoothed = state.get("smoothed_doa_deg")
        if prev_smoothed is None:
            state["smoothed_doa_deg"] = candidate
            return candidate
        limited = _step_limited_angle(
            float(prev_smoothed),
            candidate,
            float(max(0.1, getattr(self._cfg, "doa_max_step_deg_per_frame", 10.0))),
        )
        smoothed = _ema_angle(
            float(prev_smoothed),
            limited,
            float(np.clip(getattr(self._cfg, "doa_ema_alpha", 0.2), 0.0, 1.0)),
        )
        state["smoothed_doa_deg"] = smoothed
        return float(smoothed)

    def _delay_sum_with_state(self, x: np.ndarray, *, doa_deg: float, state_key: str) -> np.ndarray:
        state = self._delay_sum_state_by_key.setdefault(
            str(state_key),
            {
                "applied_doa_deg": None,
                "smoothed_doa_deg": None,
                "transition_from_doa_deg": None,
                "transition_to_doa_deg": None,
                "transition_frame_idx": 0,
                "transition_total_frames": 0,
            },
        )
        candidate_doa = self._delay_sum_candidate_doa(state=state, doa_deg=float(doa_deg))
        applied_doa = state.get("applied_doa_deg")
        min_delta = float(max(0.0, getattr(self._cfg, "delay_sum_update_min_delta_deg", 0.0)))
        crossfade_frames = max(1, int(getattr(self._cfg, "delay_sum_crossfade_frames", 1)))
        if applied_doa is None:
            state["applied_doa_deg"] = candidate_doa
            self._maybe_record_delay_sum_snapshot(state_key=str(state_key), doa_deg=candidate_doa)
            return delay_and_sum_frame(
                x,
                doa_deg=float(candidate_doa),
                mic_geometry_xyz=self._mic_geometry_xyz,
                fs=self._cfg.sample_rate_hz,
                sound_speed_m_s=self._cfg.sound_speed_m_s,
            )

        if _angular_dist_deg(float(applied_doa), float(candidate_doa)) < min_delta:
            candidate_doa = float(applied_doa)

        transition_total_frames = int(state.get("transition_total_frames", 0) or 0)
        transition_frame_idx = int(state.get("transition_frame_idx", 0) or 0)
        transition_from_doa = state.get("transition_from_doa_deg")
        transition_to_doa = state.get("transition_to_doa_deg")

        if candidate_doa != float(applied_doa):
            if transition_total_frames <= 0 or transition_to_doa is None or _angular_dist_deg(float(transition_to_doa), float(candidate_doa)) >= min_delta:
                state["transition_from_doa_deg"] = float(applied_doa)
                state["transition_to_doa_deg"] = float(candidate_doa)
                state["transition_frame_idx"] = 0
                state["transition_total_frames"] = crossfade_frames
                transition_total_frames = crossfade_frames
                transition_frame_idx = 0
                transition_from_doa = float(applied_doa)
                transition_to_doa = float(candidate_doa)

        if transition_total_frames > 0 and transition_from_doa is not None and transition_to_doa is not None:
            old_out = delay_and_sum_frame(
                x,
                doa_deg=float(transition_from_doa),
                mic_geometry_xyz=self._mic_geometry_xyz,
                fs=self._cfg.sample_rate_hz,
                sound_speed_m_s=self._cfg.sound_speed_m_s,
            )
            new_out = delay_and_sum_frame(
                x,
                doa_deg=float(transition_to_doa),
                mic_geometry_xyz=self._mic_geometry_xyz,
                fs=self._cfg.sample_rate_hz,
                sound_speed_m_s=self._cfg.sound_speed_m_s,
            )
            sample_progress = np.linspace(
                float(transition_frame_idx) / float(transition_total_frames),
                float(transition_frame_idx + 1) / float(transition_total_frames),
                num=int(old_out.shape[0]),
                endpoint=True,
                dtype=np.float32,
            )
            out = ((1.0 - sample_progress) * old_out.astype(np.float32) + sample_progress * new_out.astype(np.float32)).astype(np.float32)
            transition_frame_idx += 1
            if transition_frame_idx >= transition_total_frames:
                state["applied_doa_deg"] = float(transition_to_doa)
                state["transition_from_doa_deg"] = None
                state["transition_to_doa_deg"] = None
                state["transition_frame_idx"] = 0
                state["transition_total_frames"] = 0
                self._maybe_record_delay_sum_snapshot(state_key=str(state_key), doa_deg=float(transition_to_doa))
            else:
                state["transition_frame_idx"] = transition_frame_idx
            return out

        self._maybe_record_delay_sum_snapshot(state_key=str(state_key), doa_deg=float(candidate_doa))
        return delay_and_sum_frame(
            x,
            doa_deg=float(candidate_doa),
            mic_geometry_xyz=self._mic_geometry_xyz,
            fs=self._cfg.sample_rate_hz,
            sound_speed_m_s=self._cfg.sound_speed_m_s,
        )

    def _delay_sum_subtractive_interferer_doa(self) -> float | None:
        if bool(getattr(self._cfg, "delay_sum_subtractive_use_suppressed_user_doa", True)):
            suppressed = getattr(self._cfg, "suppressed_user_voice_doa_deg", None)
            if suppressed is not None:
                return float(suppressed)
        explicit = getattr(self._cfg, "delay_sum_subtractive_interferer_doa_deg", None)
        if explicit is None:
            return None
        return float(explicit)

    def _delay_sum_subtractive(self, x: np.ndarray, *, target_doa_deg: float) -> np.ndarray:
        target = self._delay_sum_with_state(x, doa_deg=float(target_doa_deg), state_key="subtractive_target")
        interferer_doa = self._delay_sum_subtractive_interferer_doa()
        if interferer_doa is None:
            return target.astype(np.float32, copy=False)
        interferer = self._delay_sum_with_state(x, doa_deg=float(interferer_doa), state_key="subtractive_interferer")
        alpha = float(max(0.0, getattr(self._cfg, "delay_sum_subtractive_alpha", 0.5)))
        out = np.asarray(target, dtype=np.float32) - (alpha * np.asarray(interferer, dtype=np.float32))
        if bool(getattr(self._cfg, "delay_sum_subtractive_output_clip_guard", True)):
            peak = float(np.max(np.abs(out))) if out.size else 0.0
            if peak > 1.0:
                out = out / peak
        frame_idx = int(self._frame_idx) + 1
        if frame_idx in self._beamformer_snapshot_target_set:
            if not any(int(item.get("frame_index", -1)) == frame_idx for item in self._delay_sum_snapshot_trace):
                self._maybe_record_delay_sum_snapshot(state_key="subtractive_target", doa_deg=float(target_doa_deg))
        return np.asarray(out, dtype=np.float32)

    def _suppression_mode(self) -> str:
        return str(self._cfg.own_voice_suppression_mode).strip().lower()

    def _target_activity_mode(self) -> str | None:
        value = getattr(self._cfg, "target_activity_rnn_update_mode", None)
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

    def _target_activity_detector_backend(self) -> str:
        return str(getattr(self._cfg, "target_activity_detector_backend", "webrtc_fused")).strip().lower()

    def _target_activity_detector_mode_name(self) -> str:
        return str(getattr(self._cfg, "target_activity_detector_mode", "target_blocker_calibrated")).strip().lower()

    def _target_activity_blocker_doa(self, doa_deg: float) -> float:
        offset = float(getattr(self._cfg, "target_activity_blocker_offset_deg", 90.0))
        candidate = _norm_deg(float(doa_deg) + offset)
        user_doa = self._suppressed_user_doa()
        if user_doa is not None and _angular_dist_deg(candidate, user_doa) <= float(self._cfg.suppressed_user_match_window_deg):
            candidate = _norm_deg(float(doa_deg) - offset)
        return float(candidate)

    def _target_activity_beams(self, frame_mc: np.ndarray, doa_deg: float) -> tuple[np.ndarray, np.ndarray, float]:
        target_ref = delay_and_sum_frame(
            frame_mc,
            doa_deg=float(doa_deg),
            mic_geometry_xyz=self._mic_geometry_xyz,
            fs=self._cfg.sample_rate_hz,
            sound_speed_m_s=self._cfg.sound_speed_m_s,
        )
        blocker_doa = self._target_activity_blocker_doa(doa_deg)
        blocker_ref = delay_and_sum_frame(
            frame_mc,
            doa_deg=float(blocker_doa),
            mic_geometry_xyz=self._mic_geometry_xyz,
            fs=self._cfg.sample_rate_hz,
            sound_speed_m_s=self._cfg.sound_speed_m_s,
        )
        return target_ref, blocker_ref, float(blocker_doa)

    def _update_target_activity_calibration(self, *, target_rms: float, blocker_rms: float, ratio_db: float) -> None:
        rise_alpha = float(np.clip(self._cfg.target_activity_noise_floor_rise_alpha, 0.0, 1.0))
        fall_alpha = float(np.clip(self._cfg.target_activity_noise_floor_fall_alpha, 0.0, 1.0))
        floor_margin = float(max(self._cfg.target_activity_noise_floor_margin_scale, 1.0))
        target_obs = float(target_rms)
        blocker_obs = float(blocker_rms)
        target_alpha = rise_alpha if target_obs > self._target_activity_target_floor else fall_alpha
        blocker_alpha = rise_alpha if blocker_obs > self._target_activity_blocker_floor else fall_alpha
        self._target_activity_target_floor = ((1.0 - target_alpha) * self._target_activity_target_floor) + (target_alpha * target_obs)
        self._target_activity_blocker_floor = ((1.0 - blocker_alpha) * self._target_activity_blocker_floor) + (blocker_alpha * blocker_obs)
        ratio_alpha = float(np.clip(max(rise_alpha, 0.02), 0.0, 1.0))
        self._target_activity_ratio_baseline_db = ((1.0 - ratio_alpha) * self._target_activity_ratio_baseline_db) + (ratio_alpha * float(ratio_db))
        self._target_activity_noise_floor = float(self._target_activity_target_floor)
        self._target_activity_calibration_frames += 1

    def _estimate_target_activity(self, frame_mc: np.ndarray, doa_deg: float) -> float:
        detector_mode = self._target_activity_detector_mode_name()
        if detector_mode == "localization_peak_confidence":
            return self._estimate_target_activity_from_localization(doa_deg)
        target_ref, blocker_ref, blocker_doa = self._target_activity_beams(frame_mc, doa_deg)
        target_rms = float(np.sqrt(np.mean(np.asarray(target_ref, dtype=np.float64) ** 2) + 1e-12))
        blocker_rms = float(np.sqrt(np.mean(np.asarray(blocker_ref, dtype=np.float64) ** 2) + 1e-12))
        ratio_db = float(20.0 * np.log10(max(target_rms, 1e-12) / max(blocker_rms, 1e-12)))
        vad = self._target_activity_vad.process(np.asarray(target_ref, dtype=np.float32))
        allow_calibration = bool(getattr(self._cfg, "target_activity_bootstrap_only_calibration", True))
        should_calibrate = (not self._target_activity_bootstrap_complete) if allow_calibration else (not bool(vad.raw_active))
        if should_calibrate:
            self._update_target_activity_calibration(target_rms=target_rms, blocker_rms=blocker_rms, ratio_db=ratio_db)
        target_scale = float(max(getattr(self._cfg, "target_activity_target_rms_floor_scale", 1.8), 1e-5))
        blocker_scale = float(max(getattr(self._cfg, "target_activity_blocker_rms_floor_scale", 1.1), 1e-5))
        rms_scale = float(max(self._cfg.target_activity_rms_scale, 1e-5))
        target_hint = np.clip((target_rms - (target_scale * self._target_activity_target_floor)) / max(rms_scale * self._target_activity_target_floor, 1e-5), 0.0, 1.0)
        blocker_penalty = np.clip((blocker_rms - (blocker_scale * self._target_activity_blocker_floor)) / max(blocker_scale * self._target_activity_blocker_floor, 1e-5), 0.0, 1.0)
        ratio_floor_db = float(getattr(self._cfg, "target_activity_ratio_floor_db", 0.0))
        ratio_active_db = float(max(getattr(self._cfg, "target_activity_ratio_active_db", 4.0), ratio_floor_db + 1e-3))
        ratio_hint = np.clip((ratio_db - self._target_activity_ratio_baseline_db - ratio_floor_db) / max(ratio_active_db - ratio_floor_db, 1e-5), 0.0, 1.0)
        score_power = float(np.clip(self._cfg.target_activity_score_exponent, 0.0, 1.0))
        speech_feature = np.power(np.clip(float(vad.speech_score), 0.0, 1.0), score_power) * np.power(np.clip(float(target_hint), 0.0, 1.0), 1.0 - score_power)
        blocker_suppression = np.clip(1.0 - float(blocker_penalty), 0.0, 1.0)
        speech_w = float(max(getattr(self._cfg, "target_activity_speech_weight", 0.55), 0.0))
        ratio_w = float(max(getattr(self._cfg, "target_activity_ratio_weight", 0.30), 0.0))
        blocker_w = float(max(getattr(self._cfg, "target_activity_blocker_weight", 0.15), 0.0))
        weight_sum = max(speech_w + ratio_w + blocker_w, 1e-6)
        score = (
            ((speech_w / weight_sum) * float(speech_feature))
            + ((ratio_w / weight_sum) * float(ratio_hint))
            + ((blocker_w / weight_sum) * float(blocker_suppression))
        )
        self._target_activity_last_debug = {
            "backend": self._target_activity_detector_backend(),
            "vad_source": str(vad.source),
            "detector_skipped": False,
            "detector_update_every_n_fast_frames": float(self._target_activity_update_every_n_fast_frames),
            "target_rms_dbfs": _rms_db(target_ref),
            "blocker_rms_dbfs": _rms_db(blocker_ref),
            "target_floor_dbfs": float(20.0 * np.log10(max(self._target_activity_target_floor, 1e-12))),
            "blocker_floor_dbfs": float(20.0 * np.log10(max(self._target_activity_blocker_floor, 1e-12))),
            "ratio_db": float(ratio_db),
            "ratio_baseline_db": float(self._target_activity_ratio_baseline_db),
            "ratio_hint": float(ratio_hint),
            "target_hint": float(target_hint),
            "blocker_penalty": float(blocker_penalty),
            "speech_weight": float(speech_w / weight_sum),
            "ratio_weight": float(ratio_w / weight_sum),
            "blocker_weight": float(blocker_w / weight_sum),
            "blocker_doa_deg": float(blocker_doa),
            "calibration_frames": float(self._target_activity_calibration_frames),
            "bootstrap_complete": bool(self._target_activity_bootstrap_complete),
        }
        return float(np.clip(score, 0.0, 1.0))

    def _estimate_target_activity_from_localization(self, doa_deg: float) -> float:
        snapshot = self._state.get_srp_snapshot()
        peaks = tuple(float(v) for v in (snapshot.raw_peaks_deg or snapshot.peaks_deg or ()))
        scores_raw = snapshot.raw_peak_scores if snapshot.raw_peak_scores is not None else snapshot.peak_scores
        scores = tuple(float(v) for v in (scores_raw or ()))
        if not peaks:
            self._target_activity_last_debug = {
                "backend": "localization_peak_confidence",
                "detector_skipped": False,
                "detector_update_every_n_fast_frames": float(self._target_activity_update_every_n_fast_frames),
                "target_doa_deg": float(doa_deg),
                "matched_peak": False,
                "peak_count": 0.0,
            }
            return 0.0

        best_idx = min(range(len(peaks)), key=lambda idx: _angular_dist_deg(float(doa_deg), float(peaks[idx])))
        best_peak_deg = float(peaks[best_idx])
        best_dist_deg = float(_angular_dist_deg(float(doa_deg), best_peak_deg))
        match_window_deg = float(max(getattr(self._cfg, "focus_direction_match_window_deg", 30.0), 1.0))
        matched = bool(best_dist_deg <= match_window_deg)
        peak_score = float(scores[best_idx]) if best_idx < len(scores) else 0.0
        distance_weight = float(np.clip(1.0 - (best_dist_deg / match_window_deg), 0.0, 1.0))
        score = float(np.clip(peak_score * distance_weight, 0.0, 1.0)) if matched else 0.0
        debug = dict(snapshot.debug or {})
        self._target_activity_last_debug = {
            "backend": "localization_peak_confidence",
            "localization_backend": str(debug.get("backend", "")),
            "detector_skipped": False,
            "detector_update_every_n_fast_frames": float(self._target_activity_update_every_n_fast_frames),
            "target_doa_deg": float(doa_deg),
            "matched_peak": bool(matched),
            "peak_count": float(len(peaks)),
            "peak_doa_deg": float(best_peak_deg),
            "peak_score": float(peak_score),
            "peak_distance_deg": float(best_dist_deg),
            "distance_weight": float(distance_weight),
        }
        for key in ("capon_confidence", "capon_peak_sharpness", "capon_peak_margin", "window_speech_active"):
            if key in debug:
                self._target_activity_last_debug[key] = debug[key]
        return score

    def _update_target_activity_hysteresis(self, score: float) -> bool:
        high = float(np.clip(self._cfg.target_activity_high_threshold, 0.0, 1.0))
        low = float(np.clip(self._cfg.target_activity_low_threshold, 0.0, high))
        if float(score) >= high:
            self._target_activity_on_count += 1
            self._target_activity_off_count = 0
            if not self._target_activity_state and self._target_activity_on_count >= max(1, int(self._cfg.target_activity_enter_frames)):
                self._target_activity_state = True
        elif float(score) <= low:
            self._target_activity_off_count += 1
            self._target_activity_on_count = 0
            if self._target_activity_state and self._target_activity_off_count >= max(1, int(self._cfg.target_activity_exit_frames)):
                self._target_activity_state = False
        else:
            self._target_activity_on_count = 0
            self._target_activity_off_count = 0
        return bool(self._target_activity_state)

    def _resolve_target_activity(self, frame_mc: np.ndarray, target_doa_deg: float | None, now_ms: float) -> tuple[float, bool, float]:
        mode = self._target_activity_mode()
        if mode is None:
            return 0.0, False, float(self._cfg.fd_cov_ema_alpha)
        if mode == "oracle_target_activity":
            score = 0.0
            if self._target_activity_override_provider is not None:
                raw = self._target_activity_override_provider(self._frame_idx, now_ms)
                score = 0.0 if raw is None else float(np.clip(raw, 0.0, 1.0))
        else:
            estimate_doa = target_doa_deg if target_doa_deg is not None else self._last_target_doa_deg
            detector_should_update = (self._frame_idx % max(self._target_activity_update_every_n_fast_frames, 1)) == 0
            if estimate_doa is None:
                score = 0.0
                detector_should_update = False
            elif detector_should_update:
                score = self._estimate_target_activity(frame_mc, float(estimate_doa))
                self._target_activity_last_score = float(score)
            else:
                score = float(self._target_activity_last_score)
                self._target_activity_last_debug = {
                    **dict(self._target_activity_last_debug),
                    "detector_skipped": True,
                    "detector_update_every_n_fast_frames": float(self._target_activity_update_every_n_fast_frames),
                }
        if target_doa_deg is not None:
            self._last_target_doa_deg = float(target_doa_deg)
        if (
            mode == "estimated_target_activity"
            and self._target_activity_detector_mode_name() == "target_blocker_calibrated"
            and not self._target_activity_bootstrap_complete
            and self._target_activity_calibration_frames < 12
        ):
            score = min(float(score), 0.8 * float(np.clip(self._cfg.target_activity_low_threshold, 0.0, 1.0)))
        is_active = self._update_target_activity_hysteresis(score)
        if bool(is_active):
            self._target_activity_bootstrap_complete = True
        base_alpha = float(np.clip(self._cfg.fd_cov_ema_alpha, 0.0, 1.0))
        scale = (
            float(np.clip(self._cfg.fd_cov_update_scale_target_active, 0.0, 1.0))
            if is_active
            else float(np.clip(self._cfg.fd_cov_update_scale_target_inactive, 0.0, 1.0))
        )
        covariance_alpha = float(base_alpha * scale)
        self._target_activity_last_score = float(score)
        self._target_activity_last_covariance_alpha = covariance_alpha
        return float(score), bool(is_active), covariance_alpha

    def _resolve_multi_target_activity(self, frame_mc: np.ndarray, target_doas_deg: list[float], now_ms: float) -> tuple[float, bool, float]:
        if not target_doas_deg:
            return self._resolve_target_activity(frame_mc, None, now_ms)
        scores: list[float] = []
        prev_last = self._last_target_doa_deg
        prev_debug = dict(self._target_activity_last_debug)
        prev_state = bool(self._target_activity_state)
        prev_on = int(self._target_activity_on_count)
        prev_off = int(self._target_activity_off_count)
        for doa in target_doas_deg:
            score, _active_unused, _cov_unused = self._resolve_target_activity(frame_mc, float(doa), now_ms)
            scores.append(float(score))
            self._target_activity_state = prev_state
            self._target_activity_on_count = prev_on
            self._target_activity_off_count = prev_off
        self._last_target_doa_deg = prev_last
        best_score = float(max(scores, default=0.0))
        is_active = self._update_target_activity_hysteresis(best_score)
        if bool(is_active):
            self._target_activity_bootstrap_complete = True
        base_alpha = float(np.clip(self._cfg.fd_cov_ema_alpha, 0.0, 1.0))
        scale = (
            float(np.clip(self._cfg.fd_cov_update_scale_target_active, 0.0, 1.0))
            if is_active
            else float(np.clip(self._cfg.fd_cov_update_scale_target_inactive, 0.0, 1.0))
        )
        covariance_alpha = float(base_alpha * scale)
        self._target_activity_last_score = best_score
        self._target_activity_last_covariance_alpha = covariance_alpha
        self._target_activity_last_debug = {
            **prev_debug,
            **dict(self._target_activity_last_debug),
            "multi_target": True,
            "candidate_target_count": float(len(target_doas_deg)),
            "candidate_target_scores": [float(v) for v in scores],
        }
        return best_score, bool(is_active), covariance_alpha

    def _suppressed_user_doa(self) -> float | None:
        if self._cfg.suppressed_user_voice_doa_deg is None:
            return None
        return _norm_deg(float(self._cfg.suppressed_user_voice_doa_deg))

    def _match_user_peak(
        self,
        peaks: list[float],
        scores: list[float] | None,
    ) -> tuple[float | None, float, int | None]:
        user_doa = self._suppressed_user_doa()
        if user_doa is None:
            return None, 0.0, None
        best_idx = None
        best_dist = None
        best_score = 0.0
        for idx, peak in enumerate(peaks):
            dist = _angular_dist_deg(float(peak), user_doa)
            if dist > float(self._cfg.suppressed_user_match_window_deg):
                continue
            score = 1.0 if scores is None or idx >= len(scores) else float(scores[idx])
            if best_idx is None or dist < float(best_dist):
                best_idx = int(idx)
                best_dist = float(dist)
                best_score = float(score)
        if best_idx is None:
            return None, 0.0, None
        return float(peaks[best_idx]), float(best_score), int(best_idx)

    def _update_own_voice_hysteresis(self, user_present: bool) -> bool:
        if user_present:
            self._own_voice_on_count += 1
            self._own_voice_off_count = 0
            if not self._own_voice_active and self._own_voice_on_count >= max(1, int(self._cfg.suppressed_user_null_on_frames)):
                self._own_voice_active = True
        else:
            self._own_voice_off_count += 1
            self._own_voice_on_count = 0
            if self._own_voice_active and self._own_voice_off_count >= max(1, int(self._cfg.suppressed_user_null_off_frames)):
                self._own_voice_active = False
        return bool(self._own_voice_active)

    def _pick_non_user_peak(self, peaks: list[float], scores: list[float] | None, matched_user_idx: int | None) -> tuple[float | None, float]:
        best_angle = None
        best_score = -1.0
        for idx, peak in enumerate(peaks):
            if matched_user_idx is not None and int(idx) == int(matched_user_idx):
                continue
            score = 1.0 if scores is None or idx >= len(scores) else float(scores[idx])
            if score > best_score:
                best_angle = float(peak)
                best_score = float(score)
        return best_angle, float(max(best_score, 0.0))

    def _pick_non_user_speaker(self, speaker_map) -> tuple[float | None, float]:
        user_doa = self._suppressed_user_doa()
        best_angle = None
        best_score = -1.0
        for item in speaker_map.values():
            angle = float(item.direction_degrees)
            if user_doa is not None and _angular_dist_deg(angle, user_doa) <= float(self._cfg.suppressed_user_match_window_deg):
                continue
            score = float(max(getattr(item, "activity_confidence", 0.0), getattr(item, "confidence", 0.0), getattr(item, "gain_weight", 0.0)))
            if score > best_score:
                best_angle = angle
                best_score = score
        return best_angle, float(max(best_score, 0.0))

    def _pick_best_smoothed_speaker(self, speaker_map) -> tuple[float | None, float]:
        smoothed_items = self._smooth_speaker_items(speaker_map)
        if not smoothed_items:
            return None, 0.0
        best_sid = None
        best_doa = None
        best_score = -1.0
        for sid_i, doa_sm, gain_sm in smoothed_items:
            item = speaker_map.get(sid_i)
            if item is None:
                continue
            score = float(max(getattr(item, "activity_confidence", 0.0), getattr(item, "confidence", 0.0), gain_sm))
            if score > best_score:
                best_sid = sid_i
                best_doa = doa_sm
                best_score = score
        if best_sid is None or best_doa is None:
            return None, 0.0
        return float(best_doa), float(max(best_score, 0.0))

    def _resolve_single_active_localization(
        self,
        *,
        peaks: list[float],
        scores: list[float] | None,
        matched_user_idx: int | None,
    ) -> tuple[list[float], list[float] | None, dict]:
        hold_frames = max(1, int(getattr(self._cfg, "focus_target_hold_frames", 8)))
        max_step = float(max(0.1, getattr(self._cfg, "doa_max_step_deg_per_frame", 10.0)))
        alpha = float(np.clip(getattr(self._cfg, "doa_ema_alpha", 0.2), 0.0, 1.0))
        best_peak = None
        best_score = 0.0
        for idx, peak in enumerate(peaks):
            if matched_user_idx is not None and int(idx) == int(matched_user_idx):
                continue
            score = 1.0 if scores is None or idx >= len(scores) else float(scores[idx])
            if best_peak is None or score > best_score:
                best_peak = float(peak)
                best_score = float(score)

        used_hold = False
        prev_doa = self._single_active_localized_doa_deg
        if best_peak is None:
            self._single_active_missing_frames += 1
            if prev_doa is not None and self._single_active_missing_frames <= hold_frames:
                best_peak = float(prev_doa)
                best_score = float(self._single_active_localized_score)
                used_hold = True
            else:
                self._single_active_localized_doa_deg = None
                self._single_active_localized_score = 0.0
                return [], None, {
                    "single_active": True,
                    "selected_peak_deg": None,
                    "selected_peak_score": 0.0,
                    "used_hold": False,
                    "missing_frames": int(self._single_active_missing_frames),
                }
        else:
            self._single_active_missing_frames = 0

        if prev_doa is None or used_hold:
            smoothed = float(best_peak)
        else:
            limited = _step_limited_angle(float(prev_doa), float(best_peak), max_step)
            smoothed = _ema_angle(float(prev_doa), float(limited), alpha)
        self._single_active_localized_doa_deg = float(smoothed)
        self._single_active_localized_score = float(best_score)
        return [float(smoothed)], [float(best_score)], {
            "single_active": True,
            "selected_peak_deg": float(best_peak),
            "selected_peak_score": float(best_score),
            "smoothed_peak_deg": float(smoothed),
            "used_hold": bool(used_hold),
            "missing_frames": int(self._single_active_missing_frames),
        }

    def _beamforming_mode(self) -> str:
        return str(self._cfg.beamforming_mode).strip().lower()

    def _select_top_tracked_speakers(self, speaker_map) -> tuple[list[tuple[int, float]], str]:
        user_doa = self._suppressed_user_doa()
        min_conf = float(max(0.0, getattr(self._cfg, "multi_target_min_confidence", 0.2)))
        min_activity = float(max(0.0, getattr(self._cfg, "multi_target_min_activity", 0.15)))
        max_targets = max(1, int(getattr(self._cfg, "multi_target_max_speakers", 2)))
        candidates: list[tuple[tuple[float, float, float, float], int, float]] = []
        for sid, item in speaker_map.items():
            sid_i = int(sid)
            angle = float(item.direction_degrees)
            if user_doa is not None and _angular_dist_deg(angle, user_doa) <= float(self._cfg.suppressed_user_match_window_deg):
                continue
            activity = float(getattr(item, "activity_confidence", 0.0))
            conf = float(getattr(item, "confidence", 0.0))
            gain = float(getattr(item, "gain_weight", 0.0))
            if conf < min_conf and activity < min_activity:
                continue
            continuity = 1.0 if sid_i in self._last_multi_target_speaker_ids else 0.0
            candidates.append(((continuity, activity, conf, gain), sid_i, angle))
        candidates.sort(key=lambda row: row[0], reverse=True)
        selected = [(sid_i, float(angle)) for _key, sid_i, angle in candidates[:max_targets]]
        if selected:
            return selected, "top_tracked"
        if self._multi_target_hold_remaining_frames > 0 and self._last_multi_target_speaker_ids and self._last_multi_target_doas_deg:
            self._multi_target_hold_remaining_frames -= 1
            return list(zip(self._last_multi_target_speaker_ids, self._last_multi_target_doas_deg, strict=True)), "top_tracked_hold"
        self._multi_target_hold_remaining_frames = 0
        return [], "top_tracked_miss"

    def _focused_direction_match_window_deg(self) -> float:
        return float(max(1.0, getattr(self._cfg, "focus_direction_match_window_deg", 30.0)))

    def _focus_control_active(self, focus) -> bool:
        return bool(
            getattr(focus, "focused_speaker_ids", None)
            or getattr(focus, "focused_direction_deg", None) is not None
        )

    def _pick_focused_peak(
        self,
        peaks: list[float],
        scores: list[float] | None,
        matched_user_idx: int | None,
        focus,
    ) -> tuple[float | None, float, str]:
        focus_direction = getattr(focus, "focused_direction_deg", None)
        if focus_direction is None:
            doa, score = self._pick_non_user_peak(peaks, scores, matched_user_idx)
            return doa, score, "fallback_peak"
        window_deg = self._focused_direction_match_window_deg()
        best_angle = None
        best_key = None
        best_score = 0.0
        for idx, peak in enumerate(peaks):
            if matched_user_idx is not None and int(idx) == int(matched_user_idx):
                continue
            dist = _angular_dist_deg(float(peak), float(focus_direction))
            if dist > window_deg:
                continue
            score = 1.0 if scores is None or idx >= len(scores) else float(scores[idx])
            continuity = 1.0 if (self._last_target_doa_deg is not None and _angular_dist_deg(float(peak), float(self._last_target_doa_deg)) <= 8.0) else 0.0
            key = (continuity, -dist, score)
            if best_key is None or key > best_key:
                best_key = key
                best_angle = float(peak)
                best_score = float(score)
        if best_angle is not None:
            return best_angle, best_score, "focus_direction_peak"
        return None, 0.0, "focus_direction_miss"

    def _pick_focused_speaker(self, speaker_map, focus) -> tuple[float | None, float, int | None, str]:
        focused_ids = None if getattr(focus, "focused_speaker_ids", None) is None else {int(v) for v in focus.focused_speaker_ids}
        focus_direction = getattr(focus, "focused_direction_deg", None)
        user_doa = self._suppressed_user_doa()
        best_doa = None
        best_sid = None
        best_score = 0.0
        best_key = None
        for sid, item in speaker_map.items():
            sid_i = int(sid)
            angle = float(item.direction_degrees)
            if user_doa is not None and _angular_dist_deg(angle, user_doa) <= float(self._cfg.suppressed_user_match_window_deg):
                continue
            if focused_ids is not None and sid_i not in focused_ids:
                continue
            dist = 0.0 if focus_direction is None else _angular_dist_deg(angle, float(focus_direction))
            if focus_direction is not None and dist > self._focused_direction_match_window_deg():
                continue
            activity = float(getattr(item, "activity_confidence", 0.0))
            conf = float(getattr(item, "confidence", 0.0))
            gain = float(getattr(item, "gain_weight", 0.0))
            continuity = 1.0 if (self._last_target_speaker_id is not None and sid_i == int(self._last_target_speaker_id)) else 0.0
            key = (continuity, activity, conf, gain, -dist)
            if best_key is None or key > best_key:
                best_key = key
                best_sid = sid_i
                best_doa = angle
                best_score = float(max(activity, conf, gain))
        if best_sid is None or best_doa is None:
            if focused_ids is not None:
                return None, 0.0, None, "focus_speaker_id_miss"
            if focus_direction is not None:
                return None, 0.0, None, "focus_direction_speaker_miss"
            return None, 0.0, None, "focus_inactive"
        if focused_ids is not None:
            return float(best_doa), float(best_score), int(best_sid), "focus_speaker_id"
        if focus_direction is not None:
            return float(best_doa), float(best_score), int(best_sid), "focus_direction_speaker"
        return float(best_doa), float(best_score), int(best_sid), "fallback_speaker"

    def _apply_target_hold(self, *, selected: bool) -> tuple[float | None, int | None, bool]:
        if selected:
            self._focus_hold_remaining_frames = max(0, int(getattr(self._cfg, "focus_target_hold_frames", 8)))
            return self._last_target_doa_deg, self._last_target_speaker_id, False
        if self._focus_hold_remaining_frames > 0 and self._last_target_doa_deg is not None:
            self._focus_hold_remaining_frames -= 1
            return float(self._last_target_doa_deg), self._last_target_speaker_id, True
        self._focus_hold_remaining_frames = 0
        return None, None, False

    def _suppression_debug(
        self,
        *,
        matched_user_peak_deg: float | None,
        matched_user_score: float,
        suppression_active: bool,
        target_doa_deg: float | None,
        suppression_applied: bool,
        suppression_strategy: str,
        conflict_fallback: bool,
    ) -> dict:
        user_doa = self._suppressed_user_doa()
        return {
            "configured_user_doa_deg": user_doa,
            "matched_user_peak_deg": matched_user_peak_deg,
            "matched_user_score": float(matched_user_score),
            "user_present": bool(matched_user_peak_deg is not None),
            "suppression_active": bool(suppression_active),
            "target_doa_deg": None if target_doa_deg is None else float(target_doa_deg),
            "suppression_applied": bool(suppression_applied),
            "suppression_strategy": str(suppression_strategy),
            "conflict_fallback": bool(conflict_fallback),
            "on_count": int(self._own_voice_on_count),
            "off_count": int(self._own_voice_off_count),
        }

    def _smooth_speaker_items(self, speaker_map) -> list:
        alpha_doa = float(np.clip(self._cfg.doa_ema_alpha, 0.0, 1.0))
        alpha_gain = float(np.clip(self._cfg.gain_ema_alpha, 0.0, 1.0))
        max_step = float(max(0.1, self._cfg.doa_max_step_deg_per_frame))
        smoothed = []
        for sid, item in speaker_map.items():
            sid_i = int(sid)
            doa_raw = _norm_deg(float(item.direction_degrees))
            if sid_i not in self._smoothed_doa_by_speaker:
                doa_sm = doa_raw
            else:
                limited = _step_limited_angle(self._smoothed_doa_by_speaker[sid_i], doa_raw, max_step)
                doa_sm = _ema_angle(self._smoothed_doa_by_speaker[sid_i], limited, alpha_doa)
            self._smoothed_doa_by_speaker[sid_i] = doa_sm

            gain_raw = float(item.gain_weight)
            prev_gain = float(self._smoothed_gain_by_speaker.get(sid_i, gain_raw))
            gain_sm = (1.0 - alpha_gain) * prev_gain + alpha_gain * gain_raw
            self._smoothed_gain_by_speaker[sid_i] = gain_sm
            smoothed.append((sid_i, doa_sm, max(0.0, gain_sm)))
        return smoothed

    def _enqueue_slow(self, frame: np.ndarray) -> None:
        if not bool(self._cfg.slow_path_enabled):
            return
        try:
            self._slow_queue.put_nowait(frame.copy())
        except queue.Full:
            try:
                _ = self._slow_queue.get_nowait()
            except queue.Empty:
                pass
            self._state.incr_dropped_fast_to_slow(1)
            try:
                self._slow_queue.put_nowait(frame.copy())
            except queue.Full:
                self._state.incr_dropped_fast_to_slow(1)

    def _publish_localization_only_speaker_map(
        self,
        *,
        timestamp_ms: float,
        peaks: list[float],
        scores: list[float] | None,
    ) -> None:
        if not peaks:
            self._state.publish_speaker_map({})
            return
        if bool(self._cfg.assume_single_speaker):
            peaks = list(peaks[:1])
            scores = None if scores is None else list(scores[:1])
        peak_scores = scores if scores is not None else [1.0] * len(peaks)
        speaker_map: dict[int, SpeakerGainDirection] = {}
        for idx, doa_deg in enumerate(peaks, start=1):
            score = float(peak_scores[idx - 1]) if idx - 1 < len(peak_scores) else 1.0
            conf = float(np.clip(score, 0.0, 1.0))
            speaker_map[idx] = SpeakerGainDirection(
                speaker_id=idx,
                direction_degrees=float(doa_deg),
                gain_weight=1.0 if idx == 1 else conf,
                confidence=conf,
                active=True,
                activity_confidence=conf,
                updated_at_ms=float(timestamp_ms),
                predicted_direction_deg=float(doa_deg),
            )
        self._state.publish_speaker_map(speaker_map)

    def _beamform_single_direction(
        self,
        x: np.ndarray,
        doa_deg: float,
        speech_activity: float,
        covariance_alpha: float | None = None,
        *,
        target_active: bool = True,
        oracle_noise_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        mode = str(self._cfg.beamforming_mode).strip().lower()
        if mode == "delay_sum":
            return self._delay_sum_with_state(x, doa_deg=doa_deg, state_key="primary")
        if mode == "delay_sum_subtractive":
            return self._delay_sum_subtractive(x, target_doa_deg=doa_deg)
        if mode == "gsc_fd":
            return self._fd.gsc(x, doa_deg=doa_deg, speech_activity=speech_activity)
        if mode == "lcmv_target_band":
            return self._fd.lcmv_target_band(
                x,
                target_doa_deg=doa_deg,
                covariance_alpha=covariance_alpha,
                target_active=bool(target_active),
                oracle_noise_frame=oracle_noise_frame,
            )
        return self._fd.mvdr(
            x,
            doa_deg=doa_deg,
            covariance_alpha=covariance_alpha,
            target_active=bool(target_active),
            oracle_noise_frame=oracle_noise_frame,
        )

    def _beamform_multi_directions(
        self,
        x: np.ndarray,
        target_doas_deg: list[float],
        speech_activity: float,
        covariance_alpha: float | None = None,
        *,
        target_active: bool = True,
        oracle_noise_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        if not target_doas_deg:
            return np.mean(x, axis=1).astype(np.float32, copy=False)
        mode = self._beamforming_mode()
        if len(target_doas_deg) <= 1 or mode != "lcmv_top2_tracked":
            return self._beamform_single_direction(
                x,
                doa_deg=float(target_doas_deg[0]),
                speech_activity=speech_activity,
                covariance_alpha=covariance_alpha,
                target_active=bool(target_active),
                oracle_noise_frame=oracle_noise_frame,
            )
        if mode == "delay_sum":
            aligned = [self._delay_sum_with_state(x, doa_deg=float(doa), state_key=f"target_{idx}") for idx, doa in enumerate(target_doas_deg[:2])]
            return np.mean(np.stack(aligned, axis=0), axis=0).astype(np.float32, copy=False)
        if mode == "delay_sum_subtractive":
            return self._delay_sum_subtractive(x, target_doa_deg=float(target_doas_deg[0]))
        return self._fd.lcmv_top2(
            x,
            primary_doa_deg=float(target_doas_deg[0]),
            secondary_doa_deg=float(target_doas_deg[1]),
            covariance_alpha=covariance_alpha,
            target_active=bool(target_active),
            oracle_noise_frame=oracle_noise_frame,
        )

    def _oracle_noise_frame(self, now_ms: float) -> np.ndarray | None:
        if self._oracle_noise_frame_provider is None:
            return None
        frame = self._oracle_noise_frame_provider(self._frame_idx, now_ms)
        if frame is None:
            return None
        out = np.asarray(frame, dtype=np.float32)
        if out.ndim != 2:
            raise ValueError("oracle noise frame provider must yield shape (samples, n_mics)")
        return out

    def _apply_soft_gate(self, out: np.ndarray) -> np.ndarray:
        attenuation_db = float(max(0.0, self._cfg.suppressed_user_gate_attenuation_db))
        gain = float(10.0 ** (-attenuation_db / 20.0))
        return (np.asarray(out, dtype=np.float32) * gain).astype(np.float32, copy=False)

    def _current_beamformer_noise_model_update(self) -> dict:
        mode = str(self._cfg.beamforming_mode).strip().lower()
        if mode in {"mvdr_fd", "gsc_fd", "lcmv_target_band", "lcmv_top2_tracked"}:
            return self._fd.get_last_noise_model_update()
        return _inactive_noise_model_update()

    def _build_packet(
        self,
        *,
        out: np.ndarray,
        x: np.ndarray,
        target_doa: float | None,
        target_activity_score: float,
        target_activity_active: bool,
        speech_activity: float,
        frame_samples: int,
        capture_t_monotonic: float,
        beamform_start_t: float,
        beamform_end_t: float,
    ) -> FastPathAudioPacket:
        start_sample = int(self._frame_idx * frame_samples)
        end_sample = int(start_sample + frame_samples)
        method = "off"
        if bool(self._cfg.postfilter_enabled):
            method = str(getattr(self._cfg, "postfilter_method", "off"))
        noise_model_update = self._current_beamformer_noise_model_update()
        beamformer_output_noise_psd = self._fd.get_output_noise_psd_for_fft(int(2 * frame_samples))
        return FastPathAudioPacket.create(
            frame_index=int(self._frame_idx),
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate_hz=int(self._cfg.sample_rate_hz),
            frame_samples=int(frame_samples),
            beamformed_audio=np.asarray(out, dtype=np.float32),
            beamformer_output_noise_psd=beamformer_output_noise_psd,
            frame_mc=np.asarray(x, dtype=np.float32),
            target_doa_deg=None if target_doa is None else float(target_doa),
            target_activity_score=float(target_activity_score),
            target_activity_state=bool(target_activity_active),
            speech_activity=float(speech_activity),
            beamforming_mode=str(self._cfg.beamforming_mode),
            postfilter_method=method,
            capture_t_monotonic=float(capture_t_monotonic),
            beamform_start_t=float(beamform_start_t),
            beamform_end_t=float(beamform_end_t),
            weights_reused=bool(getattr(self._fd, "_last_weights_reused", False)),
            noise_model_update_active=bool(noise_model_update["active"]),
            noise_model_update_sources=tuple(noise_model_update["sources"]),
            noise_model_update_reasons=tuple(noise_model_update["reasons"]),
            noise_model_update_debug=dict(noise_model_update["debug"]),
        )

    def _emit_postfilter_packet(self, packet: FastPathAudioPacket) -> float:
        if self._postfilter_queue is None:
            return 0.0
        t0 = perf_counter()
        packet.queue_enqueue_t = perf_counter()
        if not bool(getattr(self._cfg, "postfilter_queue_drop_oldest", False)):
            self._postfilter_queue.put(packet)
            return float((perf_counter() - t0) * 1000.0)
        try:
            self._postfilter_queue.put_nowait(packet)
        except queue.Full:
            try:
                _ = self._postfilter_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                self._state.incr_dropped_interstage(1)
            packet.queue_overflow_dropped = True
            try:
                self._postfilter_queue.put_nowait(packet)
            except queue.Full:
                self._state.incr_dropped_interstage(1)
        return float((perf_counter() - t0) * 1000.0)

    def run(self) -> None:
        frame_samples = max(1, int(self._cfg.sample_rate_hz * self._cfg.fast_frame_ms / 1000))
        try:
            while not self._stop.is_set():
                frame = self._source()
                if frame is None:
                    break
                capture_t_monotonic = perf_counter()

                with Timer() as t:
                    srp_ms = 0.0
                    beamform_ms = 0.0
                    safety_ms = 0.0
                    sink_ms = 0.0
                    enqueue_ms = 0.0

                    x = np.asarray(frame, dtype=np.float32)
                    if x.ndim != 2:
                        raise ValueError("fast-path frame source must yield shape (samples, n_mics)")
                    if x.shape[0] != frame_samples:
                        if x.shape[0] > frame_samples:
                            x = x[:frame_samples, :]
                        else:
                            x = np.pad(x, ((0, frame_samples - x.shape[0]), (0, 0)))

                    now_ms = 1000.0 * (self._frame_idx * frame_samples) / self._cfg.sample_rate_hz
                    oracle_noise_frame = self._oracle_noise_frame(now_ms)
                    t0 = perf_counter()
                    override = None if self._srp_override_provider is None else self._srp_override_provider(self._frame_idx, now_ms)
                    if override is None:
                        peaks, scores, tracker_debug = self._tracker.update(x)
                        matched_user_peak_deg, matched_user_score, matched_user_idx = self._match_user_peak(peaks, scores)
                        if bool(getattr(self._cfg, "single_active", False)):
                            peaks, scores, single_active_debug = self._resolve_single_active_localization(
                                peaks=peaks,
                                scores=scores,
                                matched_user_idx=matched_user_idx,
                            )
                            tracker_debug = dict(tracker_debug)
                            tracker_debug["single_active_localization"] = dict(single_active_debug)
                        suppression_active = self._update_own_voice_hysteresis(matched_user_peak_deg is not None)
                        tracker_debug = dict(tracker_debug)
                        tracker_debug["own_voice_suppression"] = self._suppression_debug(
                            matched_user_peak_deg=matched_user_peak_deg,
                            matched_user_score=matched_user_score,
                            suppression_active=suppression_active,
                            target_doa_deg=None,
                            suppression_applied=False,
                            suppression_strategy="none",
                            conflict_fallback=False,
                        )
                        snapshot = SRPPeakSnapshot(
                            timestamp_ms=now_ms,
                            peaks_deg=tuple(float(v) for v in peaks),
                            peak_scores=None if scores is None else tuple(float(v) for v in scores),
                            raw_peaks_deg=tuple(float(v) for v in tracker_debug.get("raw_peaks_deg", [])),
                            raw_peak_scores=tuple(float(v) for v in tracker_debug.get("raw_peak_scores", [])) if tracker_debug.get("raw_peak_scores") else None,
                            debug=dict(tracker_debug),
                        )
                    else:
                        snapshot = override
                        peaks = list(snapshot.peaks_deg)
                        scores = None if snapshot.peak_scores is None else list(snapshot.peak_scores)
                        matched_user_peak_deg, matched_user_score, matched_user_idx = self._match_user_peak(peaks, scores)
                        suppression_override = None
                        if snapshot.debug is not None and "force_suppression_active" in snapshot.debug:
                            suppression_override = bool(snapshot.debug.get("force_suppression_active"))
                        if suppression_override is None:
                            suppression_active = self._update_own_voice_hysteresis(matched_user_peak_deg is not None)
                        else:
                            suppression_active = bool(suppression_override)
                    suppression_info = dict((snapshot.debug or {}).get("own_voice_suppression", {}))
                    self._state.publish_srp_snapshot(snapshot)
                    if not bool(self._cfg.slow_path_enabled):
                        self._publish_localization_only_speaker_map(
                            timestamp_ms=now_ms,
                            peaks=peaks,
                            scores=scores,
                        )
                    srp_ms += (perf_counter() - t0) * 1000.0

                    speaker_map = self._state.get_speaker_map_snapshot()
                    t0 = perf_counter()
                    ref_mode = str(self._cfg.fast_path_reference_mode).strip().lower()
                    suppression_mode = self._suppression_mode()
                    focus = self._state.get_focus_control_snapshot()
                    focus_active = self._focus_control_active(focus)
                    beamforming_mode = self._beamforming_mode()
                    user_doa = self._suppressed_user_doa()
                    target_doa = None
                    selected_target_speaker_id = None
                    selected_target_speaker_ids: list[int] = []
                    selected_target_doas: list[float] = []
                    target_selection_mode = "unfocused"
                    target_activity_score = 0.0
                    target_activity_active = False
                    target_covariance_alpha = float(self._cfg.fd_cov_ema_alpha)
                    suppression_applied = False
                    suppression_strategy = "none"
                    conflict_fallback = False
                    fallback_beam_doa = self._last_target_doa_deg
                    if ref_mode == "srp_peak" and peaks and beamforming_mode != "lcmv_top2_tracked":
                        speech_activity = _frame_speech_activity(x)
                        if focus_active:
                            target_doa, _target_score, target_selection_mode = self._pick_focused_peak(peaks, scores, matched_user_idx, focus)
                        else:
                            target_doa, _target_score = self._pick_non_user_peak(peaks, scores, matched_user_idx)
                            target_selection_mode = "fallback_peak"
                            if target_doa is None and not suppression_active:
                                target_doa = float(peaks[0])
                                target_selection_mode = "fallback_first_peak"
                        if focus_active and target_doa is None:
                            held_doa, held_sid, used_hold = self._apply_target_hold(selected=False)
                            if used_hold and held_doa is not None:
                                target_doa = float(held_doa)
                                selected_target_speaker_id = held_sid
                                target_selection_mode = "focus_hold"
                        target_activity_score, target_activity_active, target_covariance_alpha = self._resolve_target_activity(
                            x,
                            target_doa_deg=target_doa,
                            now_ms=now_ms,
                        )
                        if target_doa is not None:
                            can_null = (
                                suppression_mode == "lcmv_null_hysteresis"
                                and suppression_active
                                and user_doa is not None
                                and _angular_dist_deg(target_doa, user_doa) >= float(self._cfg.suppressed_user_target_conflict_deg)
                                and str(self._cfg.beamforming_mode).strip().lower() == "mvdr_fd"
                            )
                            if can_null:
                                out = self._fd.lcmv_null(
                                    x,
                                    target_doa_deg=float(target_doa),
                                    null_doa_deg=float(user_doa),
                                    covariance_alpha=target_covariance_alpha,
                                    target_active=bool(target_activity_active),
                                    oracle_noise_frame=oracle_noise_frame,
                                )
                                suppression_applied = True
                                suppression_strategy = "lcmv_null"
                            else:
                                if suppression_active and suppression_mode == "lcmv_null_hysteresis" and user_doa is not None:
                                    conflict_fallback = bool(
                                        _angular_dist_deg(float(target_doa), float(user_doa)) < float(self._cfg.suppressed_user_target_conflict_deg)
                                    )
                                out = self._beamform_single_direction(
                                    x,
                                    doa_deg=float(target_doa),
                                    speech_activity=speech_activity,
                                    covariance_alpha=target_covariance_alpha,
                                    target_active=bool(target_activity_active),
                                    oracle_noise_frame=oracle_noise_frame,
                                )
                        elif fallback_beam_doa is not None and str(self._cfg.beamforming_mode).strip().lower() == "mvdr_fd":
                            out = self._beamform_single_direction(
                                x,
                                doa_deg=float(fallback_beam_doa),
                                speech_activity=speech_activity,
                                covariance_alpha=target_covariance_alpha,
                                target_active=False,
                                oracle_noise_frame=oracle_noise_frame,
                            )
                        else:
                            out = np.mean(x, axis=1).astype(np.float32, copy=False)
                            if suppression_active:
                                conflict_fallback = True
                    elif speaker_map:
                        speech_activity = float(
                            max((float(getattr(v, "activity_confidence", 0.0)) for v in speaker_map.values()), default=0.0)
                        )
                        if beamforming_mode == "lcmv_top2_tracked":
                            selected_pairs, target_selection_mode = self._select_top_tracked_speakers(speaker_map)
                            selected_target_speaker_ids = [int(sid) for sid, _doa in selected_pairs]
                            selected_target_doas = [float(doa) for _sid, doa in selected_pairs]
                            if selected_target_doas:
                                target_doa = float(selected_target_doas[0])
                                selected_target_speaker_id = int(selected_target_speaker_ids[0])
                            target_activity_score, target_activity_active, target_covariance_alpha = self._resolve_multi_target_activity(
                                x,
                                selected_target_doas,
                                now_ms=now_ms,
                            )
                            if len(selected_target_doas) >= 2:
                                out = self._beamform_multi_directions(
                                    x,
                                    target_doas_deg=selected_target_doas[:2],
                                    speech_activity=speech_activity,
                                    covariance_alpha=target_covariance_alpha,
                                    target_active=bool(target_activity_active),
                                    oracle_noise_frame=oracle_noise_frame,
                                )
                            elif len(selected_target_doas) == 1:
                                out = self._beamform_single_direction(
                                    x,
                                    doa_deg=float(selected_target_doas[0]),
                                    speech_activity=speech_activity,
                                    covariance_alpha=target_covariance_alpha,
                                    target_active=bool(target_activity_active),
                                    oracle_noise_frame=oracle_noise_frame,
                                )
                            elif fallback_beam_doa is not None:
                                out = self._beamform_single_direction(
                                    x,
                                    doa_deg=float(fallback_beam_doa),
                                    speech_activity=speech_activity,
                                    covariance_alpha=target_covariance_alpha,
                                    target_active=False,
                                    oracle_noise_frame=oracle_noise_frame,
                                )
                            else:
                                out = np.mean(x, axis=1).astype(np.float32, copy=False)
                        elif focus_active:
                            target_doa, _target_score, selected_target_speaker_id, target_selection_mode = self._pick_focused_speaker(speaker_map, focus)
                        else:
                            target_doa, _target_score = self._pick_non_user_speaker(speaker_map)
                            target_selection_mode = "fallback_speaker"
                            if target_doa is None and not suppression_active:
                                target_doa, _target_score = self._pick_best_smoothed_speaker(speaker_map)
                                target_selection_mode = "fallback_smoothed_speaker"
                        if focus_active and target_doa is None:
                            held_doa, held_sid, used_hold = self._apply_target_hold(selected=False)
                            if used_hold and held_doa is not None:
                                target_doa = float(held_doa)
                                selected_target_speaker_id = held_sid
                                target_selection_mode = "focus_hold"
                            target_activity_score, target_activity_active, target_covariance_alpha = self._resolve_target_activity(
                                x,
                                target_doa_deg=target_doa,
                                now_ms=now_ms,
                            )
                            if target_doa is not None:
                                can_null = (
                                    suppression_mode == "lcmv_null_hysteresis"
                                    and suppression_active
                                    and user_doa is not None
                                    and _angular_dist_deg(target_doa, user_doa) >= float(self._cfg.suppressed_user_target_conflict_deg)
                                    and beamforming_mode == "mvdr_fd"
                                )
                                if can_null:
                                    out = self._fd.lcmv_null(
                                        x,
                                        target_doa_deg=float(target_doa),
                                        null_doa_deg=float(user_doa),
                                        covariance_alpha=target_covariance_alpha,
                                        target_active=bool(target_activity_active),
                                        oracle_noise_frame=oracle_noise_frame,
                                    )
                                    suppression_applied = True
                                    suppression_strategy = "lcmv_null"
                                else:
                                    if suppression_active and suppression_mode == "lcmv_null_hysteresis" and user_doa is not None:
                                        conflict_fallback = bool(
                                            _angular_dist_deg(float(target_doa), float(user_doa)) < float(self._cfg.suppressed_user_target_conflict_deg)
                                        )
                                    out = self._beamform_single_direction(
                                        x,
                                        doa_deg=float(target_doa),
                                        speech_activity=speech_activity,
                                        covariance_alpha=target_covariance_alpha,
                                        target_active=bool(target_activity_active),
                                        oracle_noise_frame=oracle_noise_frame,
                                    )
                            elif fallback_beam_doa is not None and beamforming_mode == "mvdr_fd":
                                out = self._beamform_single_direction(
                                    x,
                                    doa_deg=float(fallback_beam_doa),
                                    speech_activity=speech_activity,
                                    covariance_alpha=target_covariance_alpha,
                                    target_active=False,
                                    oracle_noise_frame=oracle_noise_frame,
                                )
                            else:
                                out = np.mean(x, axis=1).astype(np.float32, copy=False)
                                if suppression_active:
                                    conflict_fallback = True
                    else:
                        speech_activity = _frame_speech_activity(x)
                        target_activity_score, target_activity_active, target_covariance_alpha = self._resolve_target_activity(
                            x,
                            target_doa_deg=None,
                            now_ms=now_ms,
                        )
                        if fallback_beam_doa is not None and str(self._cfg.beamforming_mode).strip().lower() == "mvdr_fd":
                            target_doa = float(fallback_beam_doa)
                            out = self._beamform_single_direction(
                                x,
                                doa_deg=float(fallback_beam_doa),
                                speech_activity=speech_activity,
                                covariance_alpha=target_covariance_alpha,
                                target_active=False,
                                oracle_noise_frame=oracle_noise_frame,
                            )
                        else:
                            out = np.mean(x, axis=1).astype(np.float32, copy=False)
                    if suppression_active and suppression_mode in {"soft_output_gate", "lcmv_null_hysteresis"} and (
                        suppression_mode == "soft_output_gate" or conflict_fallback or target_doa is None
                    ):
                        out = self._apply_soft_gate(out)
                        suppression_applied = True
                        suppression_strategy = "soft_gate"
                    if target_doa is not None:
                        self._last_target_doa_deg = float(target_doa)
                        self._last_target_speaker_id = None if selected_target_speaker_id is None else int(selected_target_speaker_id)
                        if selected_target_speaker_ids:
                            self._last_multi_target_speaker_ids = tuple(int(v) for v in selected_target_speaker_ids)
                            self._last_multi_target_doas_deg = tuple(float(v) for v in selected_target_doas)
                            self._multi_target_hold_remaining_frames = max(0, int(getattr(self._cfg, "multi_target_hold_frames", 12)))
                        if focus_active:
                            self._focus_hold_remaining_frames = max(0, int(getattr(self._cfg, "focus_target_hold_frames", 8)))
                    target_activity_debug = {
                        "mode": self._target_activity_mode(),
                        "score": float(target_activity_score),
                        "active": bool(target_activity_active),
                        "covariance_alpha": float(target_covariance_alpha),
                        "noise_floor_rms": float(self._target_activity_noise_floor),
                        **dict(self._target_activity_last_debug),
                    }
                    if suppression_info or suppression_mode != "off" or self._target_activity_mode() is not None:
                        suppression_info = self._suppression_debug(
                            matched_user_peak_deg=matched_user_peak_deg,
                            matched_user_score=matched_user_score,
                            suppression_active=suppression_active,
                            target_doa_deg=target_doa,
                            suppression_applied=suppression_applied,
                            suppression_strategy=suppression_strategy,
                            conflict_fallback=conflict_fallback,
                        )
                        snapshot = SRPPeakSnapshot(
                            timestamp_ms=float(snapshot.timestamp_ms),
                            peaks_deg=tuple(snapshot.peaks_deg),
                            peak_scores=snapshot.peak_scores,
                            raw_peaks_deg=tuple(snapshot.raw_peaks_deg),
                            raw_peak_scores=snapshot.raw_peak_scores,
                            debug={
                                **(dict(snapshot.debug or {})),
                                "own_voice_suppression": suppression_info,
                                "target_activity": target_activity_debug,
                                "target_selection": {
                                    "focus_active": bool(focus_active),
                                    "focused_direction_deg": None if focus.focused_direction_deg is None else float(focus.focused_direction_deg),
                                    "focused_speaker_ids": None if focus.focused_speaker_ids is None else [int(v) for v in focus.focused_speaker_ids],
                                    "selected_target_speaker_id": None if selected_target_speaker_id is None else int(selected_target_speaker_id),
                                    "selected_target_speaker_ids": [int(v) for v in selected_target_speaker_ids],
                                    "selected_target_doa_deg": None if target_doa is None else float(target_doa),
                                    "selected_target_doas_deg": [float(v) for v in selected_target_doas],
                                    "selection_mode": str(target_selection_mode),
                                    "focus_hold_remaining_frames": int(self._focus_hold_remaining_frames),
                                    "multi_target_hold_remaining_frames": int(self._multi_target_hold_remaining_frames),
                                },
                            },
                        )
                        self._state.publish_srp_snapshot(snapshot)
                    beamform_ms += (perf_counter() - t0) * 1000.0

                    packet = self._build_packet(
                        out=out,
                        x=x,
                        target_doa=target_doa,
                        target_activity_score=target_activity_score,
                        target_activity_active=target_activity_active,
                        speech_activity=speech_activity,
                        frame_samples=frame_samples,
                        capture_t_monotonic=capture_t_monotonic,
                        beamform_start_t=capture_t_monotonic,
                        beamform_end_t=perf_counter(),
                    )

                    t0 = perf_counter()
                    self._enqueue_slow(x)
                    enqueue_ms += (perf_counter() - t0) * 1000.0

                    if self._split_runtime_mode == "pipelined":
                        enqueue_ms += self._emit_postfilter_packet(packet)
                    elif self._split_runtime_mode == "beamforming_only":
                        t0 = perf_counter()
                        self._state.publish_noise_model_update_snapshot(
                            NoiseModelUpdateSnapshot(
                                timestamp_ms=(1000.0 * float(packet.start_sample) / max(int(packet.sample_rate_hz), 1)),
                                active=bool(packet.noise_model_update_active),
                                sources=tuple(packet.noise_model_update_sources),
                                reasons=tuple(packet.noise_model_update_reasons),
                                debug=None if packet.noise_model_update_debug is None else dict(packet.noise_model_update_debug),
                            )
                        )
                        if self._frame_packet_sink is not None:
                            self._frame_packet_sink(packet)
                        self._sink(packet.beamformed_audio)
                        sink_ms += (perf_counter() - t0) * 1000.0
                    else:
                        packet.postfilter_start_t = perf_counter()
                        postfilter_ms, safety_ms_local, sink_ms_local = self._postfilter_stage.process_packet(packet)
                        packet.postfilter_end_t = perf_counter()
                        if self._frame_packet_sink is not None:
                            self._frame_packet_sink(packet)
                        safety_ms += float(safety_ms_local)
                        sink_ms += float(sink_ms_local)
                        self._state.record_postfilter_stage(float(postfilter_ms + safety_ms_local + sink_ms_local))
                        self._state.record_pipeline_latency(
                            queue_wait_ms=0.0,
                            end_to_end_latency_ms=max(0.0, (packet.postfilter_end_t - float(packet.capture_t_monotonic)) * 1000.0),
                            queue_depth=0,
                        )

                self._state.incr_fast_frame(t.elapsed_ms)
                beamforming_stage_ms = float(srp_ms + beamform_ms + enqueue_ms)
                self._state.record_beamforming_stage(beamforming_stage_ms)
                self._state.incr_fast_stage_times(
                    srp_ms=srp_ms,
                    beamform_ms=beamform_ms,
                    safety_ms=safety_ms,
                    sink_ms=sink_ms,
                    enqueue_ms=enqueue_ms,
                )
                self._frame_idx += 1
        finally:
            if self._postfilter_queue is not None:
                try:
                    self._postfilter_queue.put_nowait(None)
                except queue.Full:
                    try:
                        _ = self._postfilter_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        try:
                            self._postfilter_queue.put_nowait(None)
                        except queue.Full:
                            pass
            if bool(self._cfg.slow_path_enabled):
                try:
                    self._slow_queue.put_nowait(None)
                except queue.Full:
                    try:
                        _ = self._slow_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        try:
                            self._slow_queue.put_nowait(None)
                        except queue.Full:
                            pass
