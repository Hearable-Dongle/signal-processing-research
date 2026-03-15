from __future__ import annotations

import queue
import threading
from time import perf_counter
from typing import Callable

import numpy as np

from .contracts import PipelineConfig, SRPPeakSnapshot, SpeakerGainDirection
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
        self.rnn_gsc: np.ndarray | None = None
        self._win = np.sqrt(np.hanning(max(4, self.analysis_n))).astype(np.float64)
        self._prev_mc = np.zeros((max(0, self.analysis_n - self.hop), self.n_mics), dtype=np.float64)
        self._ola_tail_mvdr = np.zeros(self.analysis_n, dtype=np.float64)
        self._ola_tail_gsc = np.zeros(self.analysis_n, dtype=np.float64)

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

    def _update_noise_covariance(
        self,
        rnn_prev: np.ndarray | None,
        x_fft: np.ndarray,
        steering: np.ndarray,
        cov_alpha: float,
    ) -> np.ndarray:
        # Estimate interference/noise covariance by projecting out the target-aligned component.
        a_norm_sq = np.sum(np.abs(steering) ** 2, axis=1, keepdims=True) + 1e-10
        target_ref = np.sum(np.conj(steering) * x_fft, axis=1, keepdims=True) / a_norm_sq
        residual = x_fft - (steering * target_ref)
        return self._update_covariance(rnn_prev, residual, cov_alpha)

    def _analysis_block(self, frame_mc: np.ndarray) -> np.ndarray:
        x = np.asarray(frame_mc, dtype=np.float64)
        block = np.concatenate([self._prev_mc, x], axis=0)
        if block.shape[0] != self.analysis_n:
            raise ValueError(f"analysis block size mismatch: got {block.shape[0]}, expected {self.analysis_n}")
        if self.analysis_n > self.hop:
            self._prev_mc = block[-(self.analysis_n - self.hop) :].copy()
        xw = block * self._win[:, None]
        return np.fft.rfft(xw, axis=0)  # (F, M)

    def _synthesis_block(self, y_fft: np.ndarray, ola_buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_block = np.fft.irfft(y_fft, n=self.analysis_n).real
        yw = y_block * self._win
        buffer = np.asarray(ola_buffer, dtype=np.float64).copy()
        buffer[: self.analysis_n] += yw
        out = buffer[: self.hop].copy()
        buffer[:-self.hop] = buffer[self.hop :]
        buffer[-self.hop :] = 0.0
        return out.astype(np.float32, copy=False), buffer

    def _cov_alpha_from_activity(self, speech_activity: float) -> float:
        base = float(np.clip(self.cfg.fd_cov_ema_alpha, 0.0, 1.0))
        scale = float(np.clip(self.cfg.fd_speech_cov_update_scale, 0.0, 1.0))
        if speech_activity >= 0.5:
            return base * scale
        return base

    def mvdr(self, frame_mc: np.ndarray, doa_deg: float, speech_activity: float = 0.0) -> np.ndarray:
        x = np.asarray(frame_mc, dtype=np.float64)
        x_fft = self._analysis_block(x)  # (F, M)
        a = _steering_vector_f_domain(
            doa_deg=doa_deg,
            n_fft=self.analysis_n,
            fs=self.cfg.sample_rate_hz,
            mic_geometry_xyz=self.mic_geometry_xyz,
            sound_speed_m_s=self.cfg.sound_speed_m_s,
        )  # (F, M)
        self.rnn_mvdr_noise = self._update_noise_covariance(
            self.rnn_mvdr_noise,
            x_fft,
            a,
            self._cov_alpha_from_activity(speech_activity),
        )

        f_bins = x_fft.shape[0]
        y_fft = np.zeros(f_bins, dtype=np.complex128)
        eye = np.eye(self.n_mics, dtype=np.complex128)
        for f in range(f_bins):
            r = self.rnn_mvdr_noise[f] + (1e-8 * eye)
            af = a[f].reshape(-1, 1)
            rinv_a = np.linalg.pinv(r) @ af
            denom = (af.conj().T @ rinv_a)[0, 0]
            wf = rinv_a / (denom + 1e-10)
            y_fft[f] = (wf.conj().T @ x_fft[f].reshape(-1, 1))[0, 0]

        y, self._ola_tail_mvdr = self._synthesis_block(y_fft, self._ola_tail_mvdr)
        return y

    def lcmv_null(self, frame_mc: np.ndarray, target_doa_deg: float, null_doa_deg: float, speech_activity: float = 0.0) -> np.ndarray:
        x = np.asarray(frame_mc, dtype=np.float64)
        x_fft = self._analysis_block(x)
        a_target = _steering_vector_f_domain(
            doa_deg=target_doa_deg,
            n_fft=self.analysis_n,
            fs=self.cfg.sample_rate_hz,
            mic_geometry_xyz=self.mic_geometry_xyz,
            sound_speed_m_s=self.cfg.sound_speed_m_s,
        )
        a_null = _steering_vector_f_domain(
            doa_deg=null_doa_deg,
            n_fft=self.analysis_n,
            fs=self.cfg.sample_rate_hz,
            mic_geometry_xyz=self.mic_geometry_xyz,
            sound_speed_m_s=self.cfg.sound_speed_m_s,
        )
        self.rnn_mvdr_noise = self._update_noise_covariance(
            self.rnn_mvdr_noise,
            x_fft,
            a_target,
            self._cov_alpha_from_activity(speech_activity),
        )

        f_bins = x_fft.shape[0]
        y_fft = np.zeros(f_bins, dtype=np.complex128)
        eye = np.eye(self.n_mics, dtype=np.complex128)
        response = np.asarray([[1.0 + 0.0j], [0.0 + 0.0j]], dtype=np.complex128)
        for f in range(f_bins):
            r = self.rnn_mvdr_noise[f] + (1e-8 * eye)
            constraint = np.stack([a_target[f], a_null[f]], axis=1)
            r_inv = np.linalg.pinv(r)
            gram = constraint.conj().T @ r_inv @ constraint
            weights = r_inv @ constraint @ np.linalg.pinv(gram + 1e-8 * np.eye(gram.shape[0], dtype=np.complex128)) @ response
            y_fft[f] = (weights.conj().T @ x_fft[f].reshape(-1, 1))[0, 0]

        y, self._ola_tail_mvdr = self._synthesis_block(y_fft, self._ola_tail_mvdr)
        return y

    def gsc(self, frame_mc: np.ndarray, doa_deg: float, speech_activity: float = 0.0) -> np.ndarray:
        x = np.asarray(frame_mc, dtype=np.float64)
        x_fft = self._analysis_block(x)  # (F, M)
        self.rnn_gsc = self._update_covariance(self.rnn_gsc, x_fft, self._cov_alpha_from_activity(speech_activity))
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

        y, self._ola_tail_gsc = self._synthesis_block(y_fft, self._ola_tail_gsc)
        return y


class _PostFilterState:
    def __init__(self, frame_samples: int, cfg: PipelineConfig):
        self.n = int(frame_samples)
        self.fft_n = max(4, 2 * self.n)
        self.cfg = cfg
        self._win = np.sqrt(np.hanning(self.fft_n)).astype(np.float64)
        self._prev_in = np.zeros(self.n, dtype=np.float64)
        self._ola_tail = np.zeros(self.n, dtype=np.float64)
        self.noise_psd: np.ndarray | None = None
        self.speech_psd: np.ndarray | None = None
        self.gain_prev: np.ndarray | None = None
        self.post_snr_prev: np.ndarray | None = None

    def _smooth_gain_frequency(self, gain: np.ndarray) -> np.ndarray:
        radius = max(0, int(self.cfg.postfilter_freq_smoothing_bins))
        if radius <= 0:
            return gain
        width = (2 * radius) + 1
        positions = np.arange(width, dtype=np.float64) - float(radius)
        kernel = np.exp(-0.5 * (positions / max(float(radius), 1.0)) ** 2)
        kernel /= np.sum(kernel)
        return np.convolve(gain, kernel, mode="same")

    def process(self, frame: np.ndarray, speech_activity: float) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float64).reshape(-1)
        block = np.concatenate([self._prev_in, x], axis=0)
        self._prev_in = x.copy()
        xw = block * self._win
        x_fft = np.fft.rfft(xw, n=self.fft_n)
        psd = np.abs(x_fft) ** 2

        if self.noise_psd is None:
            self.noise_psd = psd.copy()
        if self.speech_psd is None:
            self.speech_psd = np.maximum(psd * 0.25, 1e-12)
        if self.gain_prev is None:
            self.gain_prev = np.ones_like(psd, dtype=np.float64)
        if self.post_snr_prev is None:
            self.post_snr_prev = np.maximum(psd / np.maximum(self.noise_psd, 1e-12) - 1.0, 0.0)

        n_alpha = float(np.clip(self.cfg.postfilter_noise_ema_alpha, 0.0, 1.0))
        s_alpha = float(np.clip(self.cfg.postfilter_speech_ema_alpha, 0.0, 1.0))
        g_alpha = float(np.clip(self.cfg.postfilter_gain_ema_alpha, 0.0, 1.0))
        dd_alpha = float(np.clip(self.cfg.postfilter_dd_alpha, 0.0, 0.999))
        floor = float(np.clip(self.cfg.postfilter_gain_floor, 0.05, 1.0))
        speech_update_scale = float(np.clip(self.cfg.postfilter_noise_update_speech_scale, 0.0, 1.0))
        gain_max_step_db = float(max(0.0, self.cfg.postfilter_gain_max_step_db))

        output_rms = float(np.sqrt(np.mean(x**2) + 1e-12))
        output_activity = float(np.clip((output_rms - 0.003) / 0.02, 0.0, 1.0))
        posterior_snr_hint = float(np.clip(np.mean(psd / np.maximum(self.noise_psd, 1e-12) - 1.0) / 6.0, 0.0, 1.0))
        speech_presence = float(np.clip(max(float(speech_activity), output_activity, posterior_snr_hint), 0.0, 1.0))
        noise_alpha_eff = n_alpha * ((1.0 - speech_presence) + (speech_presence * speech_update_scale))
        noise_observation = np.minimum(psd, self.noise_psd * (1.0 + (2.5 * noise_alpha_eff)) + 1e-12)
        self.noise_psd = (1.0 - noise_alpha_eff) * self.noise_psd + noise_alpha_eff * noise_observation

        post_snr = np.maximum(psd / np.maximum(self.noise_psd, 1e-12) - 1.0, 0.0)
        dd_prior = dd_alpha * (self.gain_prev**2) * self.post_snr_prev + (1.0 - dd_alpha) * post_snr
        speech_observation = np.maximum(psd - self.noise_psd, 0.0)
        self.speech_psd = (1.0 - s_alpha) * self.speech_psd + s_alpha * speech_observation
        prior_snr = np.maximum(dd_prior, self.speech_psd / np.maximum(self.noise_psd, 1e-12))
        gain = prior_snr / (1.0 + prior_snr)
        gain = np.maximum(gain, floor)
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
    ):
        super().__init__(name="FastPathWorker", daemon=True)
        self._cfg = config
        self._state = shared_state
        self._source = frame_source
        self._sink = frame_sink
        self._slow_queue = slow_queue
        self._mic_geometry_xyz = np.asarray(mic_geometry_xyz, dtype=float)
        self._stop = stop_event
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
        frame_samples = max(1, int(config.sample_rate_hz * config.fast_frame_ms / 1000))
        self._fd = _FDBufferedBeamformer(
            n_mics=self._mic_geometry_xyz.shape[1] if self._mic_geometry_xyz.shape[0] == 3 else self._mic_geometry_xyz.shape[0],
            frame_samples=frame_samples,
            cfg=config,
            mic_geometry_xyz=self._mic_geometry_xyz,
        )
        self._postfilter = _PostFilterState(frame_samples=frame_samples, cfg=config)
        self._srp_override_provider = srp_override_provider
        self._own_voice_active = False
        self._own_voice_on_count = 0
        self._own_voice_off_count = 0

    def _suppression_mode(self) -> str:
        return str(self._cfg.own_voice_suppression_mode).strip().lower()

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

    def _beamform_single_direction(self, x: np.ndarray, doa_deg: float, speech_activity: float) -> np.ndarray:
        mode = str(self._cfg.beamforming_mode).strip().lower()
        if mode == "delay_sum":
            return delay_and_sum_frame(
                x,
                doa_deg=doa_deg,
                mic_geometry_xyz=self._mic_geometry_xyz,
                fs=self._cfg.sample_rate_hz,
                sound_speed_m_s=self._cfg.sound_speed_m_s,
            )
        if mode == "gsc_fd":
            return self._fd.gsc(x, doa_deg=doa_deg, speech_activity=speech_activity)
        return self._fd.mvdr(x, doa_deg=doa_deg, speech_activity=speech_activity)

    def _apply_soft_gate(self, out: np.ndarray) -> np.ndarray:
        attenuation_db = float(max(0.0, self._cfg.suppressed_user_gate_attenuation_db))
        gain = float(10.0 ** (-attenuation_db / 20.0))
        return (np.asarray(out, dtype=np.float32) * gain).astype(np.float32, copy=False)

    def run(self) -> None:
        frame_samples = max(1, int(self._cfg.sample_rate_hz * self._cfg.fast_frame_ms / 1000))
        try:
            while not self._stop.is_set():
                frame = self._source()
                if frame is None:
                    break

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
                    t0 = perf_counter()
                    override = None if self._srp_override_provider is None else self._srp_override_provider(self._frame_idx, now_ms)
                    if override is None:
                        peaks, scores, tracker_debug = self._tracker.update(x)
                        matched_user_peak_deg, matched_user_score, matched_user_idx = self._match_user_peak(peaks, scores)
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
                    user_doa = self._suppressed_user_doa()
                    target_doa = None
                    suppression_applied = False
                    suppression_strategy = "none"
                    conflict_fallback = False
                    if ref_mode == "srp_peak" and peaks:
                        speech_activity = _frame_speech_activity(x)
                        target_doa, _target_score = self._pick_non_user_peak(peaks, scores, matched_user_idx)
                        if target_doa is None and not suppression_active:
                            target_doa = float(peaks[0])
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
                                    speech_activity=speech_activity,
                                )
                                suppression_applied = True
                                suppression_strategy = "lcmv_null"
                            else:
                                if suppression_active and suppression_mode == "lcmv_null_hysteresis" and user_doa is not None:
                                    conflict_fallback = bool(
                                        _angular_dist_deg(float(target_doa), float(user_doa)) < float(self._cfg.suppressed_user_target_conflict_deg)
                                    )
                                out = self._beamform_single_direction(x, doa_deg=float(target_doa), speech_activity=speech_activity)
                        else:
                            out = np.mean(x, axis=1).astype(np.float32, copy=False)
                            if suppression_active:
                                conflict_fallback = True
                        if self._cfg.postfilter_enabled:
                            out = self._postfilter.process(out, speech_activity=speech_activity)
                    elif speaker_map:
                        speech_activity = float(
                            max((float(getattr(v, "activity_confidence", 0.0)) for v in speaker_map.values()), default=0.0)
                        )
                        target_doa, _target_score = self._pick_non_user_speaker(speaker_map)
                        if target_doa is None and not suppression_active:
                            target_doa, _target_score = self._pick_best_smoothed_speaker(speaker_map)
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
                                    speech_activity=speech_activity,
                                )
                                suppression_applied = True
                                suppression_strategy = "lcmv_null"
                            else:
                                if suppression_active and suppression_mode == "lcmv_null_hysteresis" and user_doa is not None:
                                    conflict_fallback = bool(
                                        _angular_dist_deg(float(target_doa), float(user_doa)) < float(self._cfg.suppressed_user_target_conflict_deg)
                                    )
                                out = self._beamform_single_direction(x, doa_deg=float(target_doa), speech_activity=speech_activity)
                        else:
                            out = np.mean(x, axis=1).astype(np.float32, copy=False)
                            if suppression_active:
                                conflict_fallback = True
                        if self._cfg.postfilter_enabled:
                            out = self._postfilter.process(out, speech_activity=speech_activity)
                    else:
                        out = np.mean(x, axis=1).astype(np.float32, copy=False)
                    if suppression_active and suppression_mode in {"soft_output_gate", "lcmv_null_hysteresis"} and (
                        suppression_mode == "soft_output_gate" or conflict_fallback or target_doa is None
                    ):
                        out = self._apply_soft_gate(out)
                        suppression_applied = True
                        suppression_strategy = "soft_gate"
                    if suppression_info or suppression_mode != "off":
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
                            debug={**(dict(snapshot.debug or {})), "own_voice_suppression": suppression_info},
                        )
                        self._state.publish_srp_snapshot(snapshot)
                    beamform_ms += (perf_counter() - t0) * 1000.0

                    t0 = perf_counter()
                    out, self._rms_gain_ema = _apply_output_safety(out, self._cfg, self._rms_gain_ema)
                    safety_ms += (perf_counter() - t0) * 1000.0

                    t0 = perf_counter()
                    self._sink(out)
                    sink_ms += (perf_counter() - t0) * 1000.0
                    t0 = perf_counter()
                    self._enqueue_slow(x)
                    enqueue_ms += (perf_counter() - t0) * 1000.0

                self._state.incr_fast_frame(t.elapsed_ms)
                self._state.incr_fast_stage_times(
                    srp_ms=srp_ms,
                    beamform_ms=beamform_ms,
                    safety_ms=safety_ms,
                    sink_ms=sink_ms,
                    enqueue_ms=enqueue_ms,
                )
                self._frame_idx += 1
        finally:
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
