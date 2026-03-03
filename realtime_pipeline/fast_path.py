from __future__ import annotations

import queue
import threading
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
    delays = np.rint(tau * fs).astype(int)

    aligned = np.zeros(frame.shape[0], dtype=np.float64)
    for m in range(frame.shape[1]):
        aligned += _shift_with_zeros(frame[:, m], int(delays[m]))

    return (aligned / max(1, frame.shape[1])).astype(np.float32, copy=False)


def _soft_clip(x: np.ndarray, drive: float) -> np.ndarray:
    d = max(float(drive), 1e-6)
    y = np.tanh(d * np.asarray(x, dtype=np.float64))
    return y.astype(np.float32, copy=False)


def _apply_output_safety(out: np.ndarray, cfg: PipelineConfig, rms_gain_ema: float) -> tuple[np.ndarray, float]:
    y = np.asarray(out, dtype=np.float32)
    next_rms_gain = float(rms_gain_ema)
    if cfg.output_target_rms is not None:
        cur_rms = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2) + 1e-12))
        target_rms = max(float(cfg.output_target_rms), 1e-6)
        desired_gain = target_rms / max(cur_rms, 1e-6)
        max_gain = float(10.0 ** (float(cfg.output_rms_max_gain_db) / 20.0))
        desired_gain = float(np.clip(desired_gain, 1.0 / max_gain, max_gain))
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
    ):
        super().__init__(name="FastPathWorker", daemon=True)
        self._cfg = config
        self._state = shared_state
        self._source = frame_source
        self._sink = frame_sink
        self._slow_queue = slow_queue
        self._mic_geometry_xyz = np.asarray(mic_geometry_xyz, dtype=float)
        self._stop = stop_event
        self._tracker = SRPPeakTracker(
            mic_pos=self._mic_geometry_xyz if self._mic_geometry_xyz.shape[0] == 3 else self._mic_geometry_xyz.T,
            fs=config.sample_rate_hz,
            window_ms=config.srp_window_ms,
            nfft=config.srp_nfft,
            overlap=config.srp_overlap,
            freq_range=(config.srp_freq_min_hz, config.srp_freq_max_hz),
            max_sources=config.srp_max_sources,
        )
        self._frame_idx = 0
        self._rms_gain_ema = 1.0

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

    def run(self) -> None:
        frame_samples = max(1, int(self._cfg.sample_rate_hz * self._cfg.fast_frame_ms / 1000))
        try:
            while not self._stop.is_set():
                frame = self._source()
                if frame is None:
                    break

                with Timer() as t:
                    x = np.asarray(frame, dtype=np.float32)
                    if x.ndim != 2:
                        raise ValueError("fast-path frame source must yield shape (samples, n_mics)")
                    if x.shape[0] != frame_samples:
                        if x.shape[0] > frame_samples:
                            x = x[:frame_samples, :]
                        else:
                            x = np.pad(x, ((0, frame_samples - x.shape[0]), (0, 0)))

                    now_ms = 1000.0 * (self._frame_idx * frame_samples) / self._cfg.sample_rate_hz
                    peaks, scores = self._tracker.update(x)
                    self._state.publish_srp_snapshot(
                        SRPPeakSnapshot(
                            timestamp_ms=now_ms,
                            peaks_deg=tuple(float(v) for v in peaks),
                            peak_scores=None if scores is None else tuple(float(v) for v in scores),
                        )
                    )

                    speaker_map = self._state.get_speaker_map_snapshot()
                    if speaker_map:
                        out = np.zeros(x.shape[0], dtype=np.float32)
                        for item in speaker_map.values():
                            bf = delay_and_sum_frame(
                                x,
                                doa_deg=item.direction_degrees,
                                mic_geometry_xyz=self._mic_geometry_xyz,
                                fs=self._cfg.sample_rate_hz,
                                sound_speed_m_s=self._cfg.sound_speed_m_s,
                            )
                            out += float(item.gain_weight) * bf
                    else:
                        out = np.mean(x, axis=1).astype(np.float32, copy=False)

                    out, self._rms_gain_ema = _apply_output_safety(out, self._cfg, self._rms_gain_ema)

                    self._sink(out)
                    self._enqueue_slow(x)

                self._state.incr_fast_frame(t.elapsed_ms)
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
