from __future__ import annotations

import queue
import threading
from time import perf_counter
from typing import Callable

import numpy as np

from .contracts import PipelineConfig, SRPPeakSnapshot
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
    direction = np.array([np.cos(az), np.sin(az), 0.0], dtype=float)
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
    direction = np.array([np.cos(az), np.sin(az), 0.0], dtype=np.float64)
    tau = (mic_pos @ direction) / float(sound_speed_m_s)
    tau = tau - float(np.mean(tau))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    phase = -2j * np.pi * freqs[:, None] * tau[None, :]
    return np.exp(phase).astype(np.complex128)


class _FDBufferedBeamformer:
    def __init__(self, n_mics: int, frame_samples: int, cfg: PipelineConfig, mic_geometry_xyz: np.ndarray):
        self.n_mics = int(n_mics)
        self.n = int(frame_samples)
        self.cfg = cfg
        self.mic_geometry_xyz = np.asarray(mic_geometry_xyz, dtype=np.float64)
        self.rnn_mvdr: np.ndarray | None = None
        self.rnn_gsc: np.ndarray | None = None
        self._win = np.sqrt(np.hanning(max(4, 2 * self.n))).astype(np.float64)
        self._prev_mc = np.zeros((self.n, self.n_mics), dtype=np.float64)
        self._ola_tail_mvdr = np.zeros(self.n, dtype=np.float64)
        self._ola_tail_gsc = np.zeros(self.n, dtype=np.float64)

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

    def _analysis_block(self, frame_mc: np.ndarray) -> np.ndarray:
        x = np.asarray(frame_mc, dtype=np.float64)
        block = np.concatenate([self._prev_mc, x], axis=0)  # (2N, M)
        self._prev_mc = x.copy()
        xw = block * self._win[:, None]
        return np.fft.rfft(xw, axis=0)  # (F, M)

    def _synthesis_block(self, y_fft: np.ndarray, tail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_block = np.fft.irfft(y_fft, n=2 * self.n).real
        yw = y_block * self._win
        out = tail + yw[: self.n]
        next_tail = yw[self.n :].copy()
        return out.astype(np.float32, copy=False), next_tail

    def _cov_alpha_from_activity(self, speech_activity: float) -> float:
        base = float(np.clip(self.cfg.fd_cov_ema_alpha, 0.0, 1.0))
        scale = float(np.clip(self.cfg.fd_speech_cov_update_scale, 0.0, 1.0))
        if speech_activity >= 0.5:
            return base * scale
        return base

    def mvdr(self, frame_mc: np.ndarray, doa_deg: float, speech_activity: float = 0.0) -> np.ndarray:
        x = np.asarray(frame_mc, dtype=np.float64)
        x_fft = self._analysis_block(x)  # (F, M)
        self.rnn_mvdr = self._update_covariance(self.rnn_mvdr, x_fft, self._cov_alpha_from_activity(speech_activity))
        a = _steering_vector_f_domain(
            doa_deg=doa_deg,
            n_fft=2 * self.n,
            fs=self.cfg.sample_rate_hz,
            mic_geometry_xyz=self.mic_geometry_xyz,
            sound_speed_m_s=self.cfg.sound_speed_m_s,
        )  # (F, M)

        f_bins = x_fft.shape[0]
        y_fft = np.zeros(f_bins, dtype=np.complex128)
        eye = np.eye(self.n_mics, dtype=np.complex128)
        for f in range(f_bins):
            r = self.rnn_mvdr[f] + (1e-8 * eye)
            af = a[f].reshape(-1, 1)
            rinv_a = np.linalg.pinv(r) @ af
            denom = (af.conj().T @ rinv_a)[0, 0]
            wf = rinv_a / (denom + 1e-10)
            y_fft[f] = (wf.conj().T @ x_fft[f].reshape(-1, 1))[0, 0]

        y, self._ola_tail_mvdr = self._synthesis_block(y_fft, self._ola_tail_mvdr)
        return y

    def gsc(self, frame_mc: np.ndarray, doa_deg: float, speech_activity: float = 0.0) -> np.ndarray:
        x = np.asarray(frame_mc, dtype=np.float64)
        x_fft = self._analysis_block(x)  # (F, M)
        self.rnn_gsc = self._update_covariance(self.rnn_gsc, x_fft, self._cov_alpha_from_activity(speech_activity))
        a = _steering_vector_f_domain(
            doa_deg=doa_deg,
            n_fft=2 * self.n,
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
        self.cfg = cfg
        self.noise_psd: np.ndarray | None = None
        self.speech_psd: np.ndarray | None = None
        self.gain_prev: np.ndarray | None = None

    def process(self, frame: np.ndarray, speech_activity: float) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float64).reshape(-1)
        x_fft = np.fft.rfft(x, n=self.n)
        psd = np.abs(x_fft) ** 2

        if self.noise_psd is None:
            self.noise_psd = psd.copy()
        if self.speech_psd is None:
            self.speech_psd = psd.copy()
        if self.gain_prev is None:
            self.gain_prev = np.ones_like(psd, dtype=np.float64)

        n_alpha = float(np.clip(self.cfg.postfilter_noise_ema_alpha, 0.0, 1.0))
        s_alpha = float(np.clip(self.cfg.postfilter_speech_ema_alpha, 0.0, 1.0))
        g_alpha = float(np.clip(self.cfg.postfilter_gain_ema_alpha, 0.0, 1.0))
        floor = float(np.clip(self.cfg.postfilter_gain_floor, 0.05, 1.0))

        if speech_activity < 0.5:
            self.noise_psd = (1.0 - n_alpha) * self.noise_psd + n_alpha * psd
        self.speech_psd = (1.0 - s_alpha) * self.speech_psd + s_alpha * psd

        signal_est = np.maximum(self.speech_psd - self.noise_psd, 1e-12)
        gain = signal_est / (signal_est + self.noise_psd + 1e-12)
        gain = np.maximum(gain, floor)
        gain = (1.0 - g_alpha) * self.gain_prev + g_alpha * gain
        self.gain_prev = gain

        y_fft = x_fft * gain
        y = np.fft.irfft(y_fft, n=self.n).real
        return y.astype(np.float32, copy=False)


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
        single_source_backend = config.localization_backend in {"music_1src", "gcc_tdoa_1src"} or bool(config.single_source_mode_enabled)
        tracker_window_ms = config.single_source_window_ms if single_source_backend else config.localization_window_ms
        tracker_grid = config.single_source_grid_size if single_source_backend else config.localization_grid_size
        tracker_freq = (
            (config.single_source_freq_min_hz, config.single_source_freq_max_hz)
            if single_source_backend
            else (config.srp_freq_min_hz, config.srp_freq_max_hz)
        )
        tracker_max_sources = 1 if single_source_backend else config.srp_max_sources
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
                    self._state.publish_srp_snapshot(snapshot)
                    srp_ms += (perf_counter() - t0) * 1000.0

                    speaker_map = self._state.get_speaker_map_snapshot()
                    t0 = perf_counter()
                    ref_mode = str(self._cfg.fast_path_reference_mode).strip().lower()
                    if ref_mode == "srp_peak" and peaks:
                        speech_activity = _frame_speech_activity(x)
                        doa_deg = float(peaks[0])
                        out = self._beamform_single_direction(x, doa_deg=doa_deg, speech_activity=speech_activity)
                        if self._cfg.postfilter_enabled:
                            out = self._postfilter.process(out, speech_activity=speech_activity)
                    elif speaker_map:
                        out = np.zeros(x.shape[0], dtype=np.float32)
                        smoothed_items = self._smooth_speaker_items(speaker_map)
                        speech_activity = float(
                            max((float(getattr(v, "activity_confidence", 0.0)) for v in speaker_map.values()), default=0.0)
                        )
                        for _sid, doa_deg, gain_weight in smoothed_items:
                            bf = self._beamform_single_direction(x, doa_deg=doa_deg, speech_activity=speech_activity)
                            out += float(gain_weight) * bf
                        if self._cfg.postfilter_enabled:
                            out = self._postfilter.process(out, speech_activity=speech_activity)
                    else:
                        out = np.mean(x, axis=1).astype(np.float32, copy=False)
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
            try:
                self._slow_queue.put_nowait(None)
            except queue.Full:
                try:
                    _ = self._slow_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    self._slow_queue.put_nowait(None)
                except queue.Full:
                    return
