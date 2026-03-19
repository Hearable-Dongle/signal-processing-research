from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Sequence

import numpy as np
from scipy.signal import resample_poly

from mic_array_forwarder.models import SessionStartRequest
from realtime_pipeline.orchestrator import RealtimeSpeakerPipeline
from realtime_pipeline.session_runtime import build_pipeline_config_from_request, build_separation_backend_for_request
from simulation.mic_array_profiles import mic_positions_xyz

logger = logging.getLogger(__name__)
_SLOW_PROCESS_CHUNK_MS = 5.0
_SLOW_WAIT_MS = 2.0


def _normalize_mic_geometry_xyz(mic_geometry_xyz: np.ndarray) -> np.ndarray:
    geometry = np.asarray(mic_geometry_xyz, dtype=np.float64)
    if geometry.ndim != 2:
        raise ValueError("mic_geometry_xyz must be a 2D array")
    if geometry.shape[0] == 3:
        return geometry
    if geometry.shape[1] == 3:
        return geometry.T
    raise ValueError("mic_geometry_xyz must have shape (3, n_mics) or (n_mics, 3)")


def _mic_geometry_xy_from_xyz(mic_geometry_xyz: np.ndarray) -> np.ndarray:
    xyz = _normalize_mic_geometry_xyz(mic_geometry_xyz)
    return np.asarray(xyz[:2, :].T, dtype=np.float64)


def _mic_count_from_geometry(mic_geometry_xyz: np.ndarray) -> int:
    return int(_normalize_mic_geometry_xyz(mic_geometry_xyz).shape[1])


def _as_int16_channel_array(channels: Sequence[np.ndarray]) -> np.ndarray:
    if not channels:
        raise ValueError("channels must contain at least one channel")

    arrays: list[np.ndarray] = []
    expected_len: int | None = None
    for idx, channel in enumerate(channels):
        arr = np.asarray(channel)
        if arr.ndim != 1:
            raise ValueError(f"channel {idx} must be a 1D array")
        if arr.dtype != np.int16:
            raise TypeError(f"channel {idx} must have dtype int16")
        if expected_len is None:
            expected_len = int(arr.shape[0])
        elif int(arr.shape[0]) != expected_len:
            raise ValueError("all channels in one callback must have the same number of samples")
        arrays.append(arr)

    assert expected_len is not None
    if expected_len == 0:
        return np.zeros((0, len(arrays)), dtype=np.float32)
    return (np.stack(arrays, axis=1).astype(np.float32) / 32768.0).astype(np.float32, copy=False)


def _resample_callback_chunk(audio: np.ndarray, *, input_rate_hz: int, output_rate_hz: int) -> np.ndarray:
    if int(input_rate_hz) <= 0 or int(output_rate_hz) <= 0:
        raise ValueError("sample rates must be positive")
    if int(input_rate_hz) == int(output_rate_hz) or audio.shape[0] == 0:
        return np.asarray(audio, dtype=np.float32)
    return np.asarray(
        resample_poly(np.asarray(audio, dtype=np.float32), up=int(output_rate_hz), down=int(input_rate_hz), axis=0),
        dtype=np.float32,
    )


class RealtimeIntelligibilityAdapter:
    """Chunk-in/chunk-out wrapper for the realtime enhancement stack.

    `process_chunk()` accepts per-channel `int16` arrays sampled at the adapter's
    input rate and returns the processed mono `float32` samples that are ready so far.

    Default processing mirrors the current realtime path the benchmark uses, with
    these requested defaults:
    - localization backend: `capon_1src`
    - beamforming mode: `delay_sum`
    - postfilter: `rnnoise`
    - mic profile: `respeaker_xvf3800_0650`
    - benchmark-style algorithm selection: `speaker_tracking_single_active`
    - localization VAD: `False` to avoid an extra `webrtcvad` runtime dependency
    - localization hop/window: `200 ms` by default

    To change behavior, override constructor arguments such as:
    - `localization_backend="srp_phat_localization"`
    - `beamforming_mode="mvdr_fd"` or `"delay_sum_subtractive"`
    - `postfilter_method="off"`
    - `enable_resample=True, input_sample_rate_hz=48000, processing_sample_rate_hz=16000`

    Notes:
    - At `16 kHz`, `10 ms` is `160` samples and `200 ms` is `3200` samples.
    - Before enough history is buffered to fill the processing window, the adapter
      returns mono passthrough for each callback chunk.
    - After warm-up, the adapter returns enhanced mono output for each callback
      chunk while keeping a moving history of the most recent window of samples.
    - Output is mono `float32`. If the caller needs `int16`, convert with:
      `np.clip(y * 32767.0, -32768, 32767).astype(np.int16)`.
    """

    def __init__(
        self,
        *,
        mic_array_profile: str = "respeaker_xvf3800_0650",
        mic_geometry_xyz: np.ndarray | None = None,
        input_sample_rate_hz: int = 16000,
        processing_sample_rate_hz: int = 16000,
        enable_resample: bool = False,
        localization_backend: str = "capon_1src",
        beamforming_mode: str = "delay_sum",
        postfilter_method: str = "rnnoise",
        postfilter_enabled: bool = True,
        fast_frame_ms: int = 10,
        localization_hop_ms: int = 200,
        localization_window_ms: int = 200,
        localization_grid_size: int = 72,
        localization_vad_enabled: bool = False,
        separation_mode: str = "single_dominant_no_separator",
        algorithm_mode: str = "speaker_tracking_single_active",
        delay_sum_subtractive_alpha: float = 0.5,
        delay_sum_subtractive_interferer_doa_deg: float | None = None,
        delay_sum_subtractive_multi_offset_deg: float = 10.0,
        process_timeout_s: float = 2.0,
    ) -> None:
        self._closed = False
        self._lock = threading.RLock()
        self._frame_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=256)
        self._output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._pending_input = np.zeros((0, 0), dtype=np.float32)
        self._pending_output = np.zeros((0,), dtype=np.float32)
        self._discard_output_samples = 0
        self._warmup_frames_enqueued = 0
        self._history_mc = np.zeros((0, 0), dtype=np.float32)
        self._process_timeout_s = float(process_timeout_s)
        self._input_sample_rate_hz = int(input_sample_rate_hz)
        self._processing_sample_rate_hz = int(processing_sample_rate_hz)
        self._enable_resample = bool(enable_resample)
        self._mic_array_profile = str(mic_array_profile)

        geometry = mic_positions_xyz(self._mic_array_profile) if mic_geometry_xyz is None else mic_geometry_xyz
        self._mic_geometry_xyz = _normalize_mic_geometry_xyz(np.asarray(geometry, dtype=np.float64))
        self._mic_geometry_xy = _mic_geometry_xy_from_xyz(self._mic_geometry_xyz)
        self._channel_count = _mic_count_from_geometry(self._mic_geometry_xyz)

        self.request = SessionStartRequest(
            input_source="respeaker_live",
            channel_count=int(self._channel_count),
            sample_rate_hz=int(self._input_sample_rate_hz),
            monitor_source="processed",
            mic_array_profile=str(self._mic_array_profile),
            fast_path={
                "fast_frame_ms": int(fast_frame_ms),
                "localization_hop_ms": int(localization_hop_ms),
                "localization_window_ms": int(localization_window_ms),
                "localization_grid_size": int(localization_grid_size),
                "localization_vad_enabled": bool(localization_vad_enabled),
                "input_downsample_rate_hz": (
                    int(self._processing_sample_rate_hz)
                    if self._enable_resample and int(self._processing_sample_rate_hz) != int(self._input_sample_rate_hz)
                    else None
                ),
                "localization_backend": str(localization_backend),
                "beamforming_mode": str(beamforming_mode),
                "postfilter_enabled": bool(postfilter_enabled),
                "postfilter_method": str(postfilter_method),
                "target_activity_rnn_update_mode": None,
                "delay_sum_subtractive_alpha": float(delay_sum_subtractive_alpha),
                "delay_sum_subtractive_interferer_doa_deg": (
                    None
                    if delay_sum_subtractive_interferer_doa_deg is None
                    else float(delay_sum_subtractive_interferer_doa_deg)
                ),
                "delay_sum_subtractive_multi_offset_deg": float(delay_sum_subtractive_multi_offset_deg),
            },
            slow_path={
                "enabled": str(algorithm_mode) != "localization_only",
                "single_active": str(algorithm_mode) == "speaker_tracking_single_active",
            },
            separation_mode=str(separation_mode),
            processing_mode="specific_speaker_enhancement",
        )
        self.config = build_pipeline_config_from_request(
            self.request,
            sample_rate_hz=(
                int(self._processing_sample_rate_hz)
                if self.request.input_downsample_rate_hz is not None
                else int(self._input_sample_rate_hz)
            ),
            max_speakers_hint=max(1, int(self._channel_count)),
        )
        self._frame_samples = max(1, int(self.config.sample_rate_hz * self.config.fast_frame_ms / 1000))
        self._window_samples = max(1, int(round(self.config.sample_rate_hz * self.config.localization_window_ms / 1000.0)))

        self._pipeline = RealtimeSpeakerPipeline(
            config=self.config,
            mic_geometry_xyz=self._mic_geometry_xyz,
            mic_geometry_xy=self._mic_geometry_xy,
            frame_iterator=self._frame_iter(),
            frame_sink=self._on_output_chunk,
            separation_backend=build_separation_backend_for_request(self.request, self.config),
        )
        self._pipeline.start()

    @property
    def channel_count(self) -> int:
        return int(self._channel_count)

    @property
    def frame_samples(self) -> int:
        return int(self._frame_samples)

    @property
    def window_samples(self) -> int:
        return int(self._window_samples)

    @property
    def warmup_ready(self) -> bool:
        return bool(self._history_mc.shape[0] >= int(self._window_samples))

    def _frame_iter(self):
        while True:
            frame = self._frame_queue.get()
            if frame is None:
                return
            yield np.asarray(frame, dtype=np.float32)

    def _on_output_chunk(self, frame_mono: np.ndarray) -> None:
        chunk = np.asarray(frame_mono, dtype=np.float32).reshape(-1).copy()
        self._output_queue.put(chunk)
        logger.info(
            "streaming_adapter_timing output_chunk samples=%d output_q=%d",
            int(chunk.shape[0]),
            int(self._output_queue.qsize()),
        )

    def _mono_passthrough(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.mean(np.asarray(audio, dtype=np.float32), axis=1).astype(np.float32, copy=False)

    def _append_history(self, audio: np.ndarray) -> None:
        if audio.shape[0] == 0:
            return
        if self._history_mc.shape[1] == 0:
            self._history_mc = np.asarray(audio, dtype=np.float32)
        else:
            self._history_mc = np.concatenate([self._history_mc, np.asarray(audio, dtype=np.float32)], axis=0)
        if self._history_mc.shape[0] > int(self._window_samples):
            self._history_mc = self._history_mc[-int(self._window_samples) :, :]

    def _drain_pipeline_outputs(self) -> None:
        while True:
            try:
                part = self._output_queue.get_nowait()
            except queue.Empty:
                break
            if self._pending_output.size == 0:
                self._pending_output = np.asarray(part, dtype=np.float32)
            else:
                self._pending_output = np.concatenate([self._pending_output, np.asarray(part, dtype=np.float32)], axis=0)

    def _consume_discarded_outputs(self) -> None:
        self._drain_pipeline_outputs()
        if self._discard_output_samples <= 0:
            return
        available = min(int(self._discard_output_samples), int(self._pending_output.shape[0]))
        if available > 0:
            self._pending_output = self._pending_output[available:]
            self._discard_output_samples -= available

    def _collect_expected_outputs(self, expected_output_samples: int) -> None:
        target = int(max(0, expected_output_samples))
        while int(self._pending_output.shape[0]) < target:
            pending_before = int(self._pending_output.shape[0])
            wait_started = time.perf_counter()
            try:
                part = self._output_queue.get(timeout=self._process_timeout_s)
            except queue.Empty as exc:
                raise TimeoutError("timed out waiting for processed audio from realtime pipeline") from exc
            wait_ms = (time.perf_counter() - wait_started) * 1000.0
            if self._pending_output.size == 0:
                self._pending_output = np.asarray(part, dtype=np.float32)
            else:
                self._pending_output = np.concatenate([self._pending_output, np.asarray(part, dtype=np.float32)], axis=0)
            if wait_ms >= _SLOW_WAIT_MS:
                logger.info(
                    "streaming_adapter_timing wait_output ms=%.2f target=%d pending_before=%d received=%d pending_after=%d output_q=%d",
                    wait_ms,
                    target,
                    pending_before,
                    int(np.asarray(part).shape[0]),
                    int(self._pending_output.shape[0]),
                    int(self._output_queue.qsize()),
                )

    def _return_enhanced_chunk(self, expected_samples: int) -> np.ndarray:
        self._consume_discarded_outputs()
        target = int(expected_samples) + int(max(0, self._discard_output_samples))
        self._collect_expected_outputs(target)
        self._consume_discarded_outputs()
        if int(self._pending_output.shape[0]) < int(expected_samples):
            raise TimeoutError("processed output alignment fell behind the callback input")
        out = np.asarray(self._pending_output[: int(expected_samples)], dtype=np.float32)
        self._pending_output = self._pending_output[int(expected_samples) :]
        return out

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("adapter is closed")

    def _prepare_chunk(self, channels: Sequence[np.ndarray]) -> np.ndarray:
        audio = _as_int16_channel_array(channels)
        if audio.shape[1] != int(self._channel_count):
            raise ValueError(
                f"expected {self._channel_count} channels for profile {self._mic_array_profile}, got {audio.shape[1]}"
            )
        if self._enable_resample:
            audio = _resample_callback_chunk(
                audio,
                input_rate_hz=int(self._input_sample_rate_hz),
                output_rate_hz=int(self._processing_sample_rate_hz),
            )
        return np.asarray(audio, dtype=np.float32)

    def process_chunk(self, channels: Sequence[np.ndarray]) -> np.ndarray:
        with self._lock:
            self._ensure_open()
            process_started = time.perf_counter()
            wait_ms = 0.0
            expected_output_samples = 0
            frames_enqueued = 0
            prepare_started = time.perf_counter()
            audio = self._prepare_chunk(channels)
            prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
            callback_samples = int(audio.shape[0])
            passthrough = self._mono_passthrough(audio)
            self._append_history(audio)
            warmup_active = not self.warmup_ready
            if self._pending_input.shape[1] == 0:
                self._pending_input = audio
            elif audio.shape[0] > 0:
                self._pending_input = np.concatenate([self._pending_input, audio], axis=0)

            frames_to_process: list[np.ndarray] = []
            while self._pending_input.shape[0] >= int(self._frame_samples):
                frames_to_process.append(self._pending_input[: self._frame_samples, :].copy())
                self._pending_input = self._pending_input[self._frame_samples :, :]
            frames_enqueued = int(len(frames_to_process))

            if not frames_to_process:
                if warmup_active:
                    self._discard_output_samples += int(callback_samples)
                    self._consume_discarded_outputs()
                    total_ms = (time.perf_counter() - process_started) * 1000.0
                    if total_ms >= _SLOW_PROCESS_CHUNK_MS:
                        logger.info(
                            "streaming_adapter_timing process_chunk ms=%.2f prepare_ms=%.2f wait_ms=%.2f callback_samples=%d expected=%d warmup=%s frames=%d pending_input=%d pending_output=%d discard=%d frame_q=%d output_q=%d mode=warmup_no_frames",
                            total_ms,
                            prepare_ms,
                            wait_ms,
                            callback_samples,
                            expected_output_samples,
                            bool(warmup_active),
                            frames_enqueued,
                            int(self._pending_input.shape[0]),
                            int(self._pending_output.shape[0]),
                            int(self._discard_output_samples),
                            int(self._frame_queue.qsize()),
                            int(self._output_queue.qsize()),
                        )
                    return passthrough
                raise ValueError(
                    f"callback must supply at least one full processing frame after warm-up; "
                    f"got {callback_samples} samples, expected {self._frame_samples}"
                )

            for frame in frames_to_process:
                self._frame_queue.put(frame)

            expected_output_samples = int(len(frames_to_process) * self._frame_samples)
            if warmup_active:
                wait_started = time.perf_counter()
                self._warmup_frames_enqueued += int(len(frames_to_process))
                self._discard_output_samples += int(expected_output_samples)
                self._consume_discarded_outputs()
                self._collect_expected_outputs(int(expected_output_samples))
                self._consume_discarded_outputs()
                wait_ms = (time.perf_counter() - wait_started) * 1000.0
                total_ms = (time.perf_counter() - process_started) * 1000.0
                if total_ms >= _SLOW_PROCESS_CHUNK_MS:
                    logger.info(
                        "streaming_adapter_timing process_chunk ms=%.2f prepare_ms=%.2f wait_ms=%.2f callback_samples=%d expected=%d warmup=%s frames=%d pending_input=%d pending_output=%d discard=%d frame_q=%d output_q=%d mode=warmup",
                        total_ms,
                        prepare_ms,
                        wait_ms,
                        callback_samples,
                        expected_output_samples,
                        bool(warmup_active),
                        frames_enqueued,
                        int(self._pending_input.shape[0]),
                        int(self._pending_output.shape[0]),
                        int(self._discard_output_samples),
                        int(self._frame_queue.qsize()),
                        int(self._output_queue.qsize()),
                    )
                return passthrough

            wait_started = time.perf_counter()
            out = self._return_enhanced_chunk(int(callback_samples))
            wait_ms = (time.perf_counter() - wait_started) * 1000.0
            total_ms = (time.perf_counter() - process_started) * 1000.0
            if total_ms >= _SLOW_PROCESS_CHUNK_MS:
                logger.info(
                    "streaming_adapter_timing process_chunk ms=%.2f prepare_ms=%.2f wait_ms=%.2f callback_samples=%d expected=%d warmup=%s frames=%d pending_input=%d pending_output=%d discard=%d frame_q=%d output_q=%d mode=enhanced",
                    total_ms,
                    prepare_ms,
                    wait_ms,
                    callback_samples,
                    expected_output_samples,
                    bool(warmup_active),
                    frames_enqueued,
                    int(self._pending_input.shape[0]),
                    int(self._pending_output.shape[0]),
                    int(self._discard_output_samples),
                    int(self._frame_queue.qsize()),
                    int(self._output_queue.qsize()),
                )
            return out

    def flush(self) -> np.ndarray:
        with self._lock:
            self._ensure_open()
            pending_samples = int(self._pending_input.shape[0])
            if pending_samples <= 0:
                return np.zeros(0, dtype=np.float32)
            if not self.warmup_ready:
                if int(self._warmup_frames_enqueued) > 0:
                    self._pending_input = np.zeros((0, self._channel_count), dtype=np.float32)
                    return np.zeros(0, dtype=np.float32)
                out = self._mono_passthrough(self._pending_input)
                self._pending_input = np.zeros((0, self._channel_count), dtype=np.float32)
                return out
            pad = int(self._frame_samples - pending_samples)
            padded = self._pending_input
            if pad > 0:
                padded = np.pad(padded, ((0, pad), (0, 0)))
            self._pending_input = np.zeros((0, self._channel_count), dtype=np.float32)
            self._frame_queue.put(np.asarray(padded, dtype=np.float32))
            out = self._return_enhanced_chunk(int(self._frame_samples))
            return np.asarray(out[:pending_samples], dtype=np.float32)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pipeline.stop()
            try:
                self._frame_queue.put_nowait(None)
            except queue.Full:
                pass
            self._pipeline.join(timeout=2.0)

    def __enter__(self) -> "RealtimeIntelligibilityAdapter":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()
