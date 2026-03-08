from __future__ import annotations

import math
import queue
import threading
import time
import uuid
from collections import deque
from typing import Any

import numpy as np

from mic_array_forwarder.models import (
    MetricsMessage,
    SCHEMA_VERSION,
    SessionEventMessage,
    SessionStartRequest,
    SessionStatusResponse,
    SpeakerStateItem,
    SpeakerStateMessage,
)
from mic_array_forwarder.session import _wav_bytes_from_mono_float32
from mic_array_forwarder.ws_codec import encode_audio_chunk
from realtime_pipeline.contracts import SpeakerGainDirection
from realtime_pipeline.fast_path import delay_and_sum_frame
from realtime_pipeline.srp_tracker import SRPPeakTracker
from simulation.mic_array_profiles import mic_positions_xyz

def _mic_geometry_from_profile(profile: str) -> np.ndarray:
    return mic_positions_xyz(profile)


def _now_ms() -> float:
    return time.time() * 1000.0


def _normalize_angle_deg(v: float) -> float:
    return float(v % 360.0)


def _angle_error_deg(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))


def _ratio_to_db(ratio: float) -> float:
    return float(20.0 * math.log10(max(float(ratio), 1e-6)))


def _clip01(v: float) -> float:
    return float(np.clip(float(v), 0.0, 1.0))


def _find_input_device(sd: Any, query: str | None, min_channels: int) -> int | None:
    devices = sd.query_devices()
    if query is None or not str(query).strip():
        for idx, dev in enumerate(devices):
            if int(dev.get("max_input_channels", 0)) >= min_channels:
                return idx
        return None

    query_lc = str(query).strip().lower()
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) < min_channels:
            continue
        if query_lc in str(dev.get("name", "")).lower():
            return idx
    return None


def _device_debug_string(sd: Any) -> str:
    parts: list[str] = []
    for idx, dev in enumerate(sd.query_devices()):
        parts.append(f"{idx}: {dev.get('name', 'unknown')} (inputs={dev.get('max_input_channels', 0)})")
    return "; ".join(parts)


class LiveDemoSession:
    def __init__(self, req: SessionStartRequest):
        self.session_id = uuid.uuid4().hex[:12]
        self.req = req
        self.started_at_ms = _now_ms()
        self._status = "starting"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"live-demo-session-{self.session_id}", daemon=True)

        self._selected_speaker_id: int | None = None
        self._speaker_gain_delta_db: dict[int, float] = {}
        self._audio_chunks: deque[tuple[int, bytes]] = deque(maxlen=2048)
        self._audio_seq = 0
        self._audio_frame_idx = 0
        self._speaker_state: dict[str, Any] | None = None
        self._speaker_state_version = 0
        self._metrics_state: dict[str, Any] | None = None
        self._metrics_version = 0
        self._events: deque[tuple[int, dict[str, Any]]] = deque(maxlen=256)
        self._event_seq = 0
        self._raw_mix_parts: deque[np.ndarray] = deque(maxlen=4500)
        self._raw_multichannel_parts: deque[np.ndarray] = deque(maxlen=4500)
        self._raw_mix_sample_rate_hz = int(req.sample_rate_hz)
        self._speaker_tracks: dict[int, SpeakerGainDirection] = {}
        self._next_speaker_id = 1
        self._processing_ms_total = 0.0
        self._processed_frames = 0
        self._last_tracker_debug: dict[str, Any] = {}
        self._last_device_name = ""
        self._monitor_source = str(req.monitor_source)
        self._mic_geometry_xyz = _mic_geometry_from_profile(str(req.mic_array_profile))
        self._channel_map = self._normalize_channel_map(req.channel_map, int(req.channel_count))

    @staticmethod
    def _normalize_channel_map(raw: list[int] | None, channel_count: int) -> list[int] | None:
        if raw is None:
            return None
        try:
            items = [int(v) for v in raw]
        except (TypeError, ValueError):
            return None
        if len(items) != channel_count:
            return None
        if sorted(items) != list(range(channel_count)):
            return None
        return items

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def get_status(self) -> SessionStatusResponse:
        with self._lock:
            return SessionStatusResponse(
                session_id=self.session_id,
                status=self._status,
                started_at_ms=float(self.started_at_ms),
                uptime_ms=float(max(0.0, _now_ms() - self.started_at_ms)),
                selected_speaker_id=self._selected_speaker_id,
                speaker_gain_delta_db={int(k): float(v) for k, v in self._speaker_gain_delta_db.items()},
                last_metrics=self._metrics_state,
            )

    def iter_audio_since(self, seq: int) -> tuple[int, list[bytes]]:
        with self._lock:
            items = [payload for s, payload in self._audio_chunks if s > seq]
            return self._audio_seq, items

    def get_speaker_state_if_new(self, version: int) -> tuple[int, dict[str, Any] | None]:
        with self._lock:
            if self._speaker_state is None or self._speaker_state_version <= version:
                return self._speaker_state_version, None
            return self._speaker_state_version, dict(self._speaker_state)

    def get_metrics_if_new(self, version: int) -> tuple[int, dict[str, Any] | None]:
        with self._lock:
            if self._metrics_state is None or self._metrics_version <= version:
                return self._metrics_version, None
            return self._metrics_version, dict(self._metrics_state)

    def iter_events_since(self, seq: int) -> tuple[int, list[dict[str, Any]]]:
        with self._lock:
            items = [payload for s, payload in self._events if s > seq]
            return self._event_seq, items

    def get_raw_mix_wav_bytes(self) -> bytes:
        with self._lock:
            if not self._raw_mix_parts:
                return b""
            raw = np.concatenate(list(self._raw_mix_parts), axis=0)
            sample_rate_hz = int(self._raw_mix_sample_rate_hz)
        return _wav_bytes_from_mono_float32(raw, sample_rate_hz)

    def get_raw_channel_count(self) -> int:
        return int(self.req.channel_count)

    def get_raw_sample_rate_hz(self) -> int:
        with self._lock:
            return int(self._raw_mix_sample_rate_hz)

    def get_raw_channel_wav_bytes(self, channel_index: int) -> bytes:
        idx = int(channel_index)
        with self._lock:
            if not self._raw_multichannel_parts:
                return b""
            raw = np.concatenate(list(self._raw_multichannel_parts), axis=0)
            sample_rate_hz = int(self._raw_mix_sample_rate_hz)
        if raw.ndim != 2 or idx < 0 or idx >= raw.shape[1]:
            return b""
        return _wav_bytes_from_mono_float32(raw[:, idx], sample_rate_hz)

    def select_speaker(self, speaker_id: int) -> None:
        if self.req.processing_mode != "specific_speaker_enhancement":
            return
        with self._lock:
            sid = int(speaker_id)
            self._selected_speaker_id = sid
            self._speaker_gain_delta_db.setdefault(sid, 0.0)

    def adjust_speaker_gain(self, speaker_id: int, delta_db_step: int) -> float:
        if self.req.processing_mode != "specific_speaker_enhancement":
            return 0.0
        sid = int(speaker_id)
        step = int(delta_db_step)
        with self._lock:
            prev = float(self._speaker_gain_delta_db.get(sid, 0.0))
            curr = float(np.clip(prev + step, -12.0, 12.0))
            self._speaker_gain_delta_db[sid] = curr
        return curr

    def clear_focus(self) -> None:
        with self._lock:
            self._selected_speaker_id = None

    def set_monitor_source(self, source: str) -> None:
        with self._lock:
            self._monitor_source = str(source)

    def _next_audio_timestamp_ms(self) -> float:
        with self._lock:
            ts = float(self._audio_frame_idx * 10)
            self._audio_frame_idx += 1
            return ts

    def _publish_event(self, event: str, detail: str = "") -> None:
        msg = SessionEventMessage(
            schema_version=SCHEMA_VERSION,
            type="session_event",
            event=event,
            detail=detail,
            timestamp_ms=_now_ms(),
        ).model_dump()
        with self._lock:
            self._event_seq += 1
            self._events.append((self._event_seq, msg))

    def _append_raw_audio(self, frame_mc: np.ndarray, frame_mono: np.ndarray) -> None:
        with self._lock:
            self._raw_multichannel_parts.append(frame_mc.copy())
            self._raw_mix_parts.append(frame_mono.copy())

    def _publish_audio_chunk(self, processed: np.ndarray, raw_mixed: np.ndarray) -> None:
        with self._lock:
            source = str(self._monitor_source)
        if source == "raw_mixed":
            frame = raw_mixed
        else:
            frame = processed
        payload = encode_audio_chunk(self._next_audio_timestamp_ms(), frame)
        with self._lock:
            self._audio_seq += 1
            self._audio_chunks.append((self._audio_seq, payload))

    def _match_speaker_id(self, angle_deg: float, matched: set[int]) -> int | None:
        best_id = None
        best_err = None
        now_ms = _now_ms()
        for sid, item in self._speaker_tracks.items():
            if sid in matched:
                continue
            if now_ms - float(item.updated_at_ms) > 1500.0:
                continue
            err = _angle_error_deg(angle_deg, float(item.direction_degrees))
            if err > 25.0:
                continue
            if best_err is None or err < best_err:
                best_err = err
                best_id = sid
        return best_id

    def _build_speaker_items(self, peaks: list[float], scores: list[float] | None) -> list[SpeakerStateItem]:
        scores_eff = list(scores) if scores is not None else [1.0] * len(peaks)
        now_ms = _now_ms()
        next_tracks: dict[int, SpeakerGainDirection] = {}
        matched: set[int] = set()

        for angle_deg, score in zip(peaks, scores_eff):
            sid = self._match_speaker_id(angle_deg, matched)
            if sid is None:
                sid = self._next_speaker_id
                self._next_speaker_id += 1
            matched.add(sid)
            conf = _clip01(score)
            next_tracks[sid] = SpeakerGainDirection(
                speaker_id=int(sid),
                direction_degrees=float(_normalize_angle_deg(angle_deg)),
                gain_weight=float(max(0.2, conf)),
                confidence=conf,
                active=bool(conf >= 0.1),
                activity_confidence=conf,
                updated_at_ms=now_ms,
            )

        for sid, item in self._speaker_tracks.items():
            if sid in next_tracks:
                continue
            age_ms = now_ms - float(item.updated_at_ms)
            if age_ms > 1200.0:
                continue
            conf = _clip01(float(item.confidence) * 0.85)
            next_tracks[sid] = SpeakerGainDirection(
                speaker_id=int(item.speaker_id),
                direction_degrees=float(item.direction_degrees),
                gain_weight=float(max(0.1, conf)),
                confidence=conf,
                active=False,
                activity_confidence=float(conf),
                updated_at_ms=float(item.updated_at_ms),
            )

        self._speaker_tracks = next_tracks
        items = [
            SpeakerStateItem(
                speaker_id=int(item.speaker_id),
                direction_degrees=float(item.direction_degrees),
                confidence=float(item.confidence),
                active=bool(item.active),
                activity_confidence=float(item.activity_confidence),
                gain_weight=float(item.gain_weight),
            )
            for item in sorted(
                next_tracks.values(),
                key=lambda item: (-float(item.active), -float(item.confidence), int(item.speaker_id)),
            )
        ]

        msg = SpeakerStateMessage(timestamp_ms=now_ms, speakers=items, ground_truth=[]).model_dump()
        with self._lock:
            self._speaker_state = msg
            self._speaker_state_version += 1
        return items

    def _publish_metrics(self, queue_size: int) -> None:
        fast_avg_ms = self._processing_ms_total / self._processed_frames if self._processed_frames else 0.0
        msg = MetricsMessage(
            timestamp_ms=_now_ms(),
            fast_rtf=float(fast_avg_ms / 10.0),
            slow_rtf=0.0,
            fast_stage_avg_ms={
                "capture_queue": float(queue_size),
                "frame_process": float(fast_avg_ms),
            },
            slow_stage_avg_ms={},
            startup_lock_ms=0.0,
            reacquire_catchup_ms_median=0.0,
            nearest_change_catchup_ms_median=0.0,
        ).model_dump()
        msg["device_name"] = self._last_device_name
        msg["input_source"] = self.req.input_source
        msg["channel_count"] = int(self.req.channel_count)
        msg["sample_rate_hz"] = int(self.req.sample_rate_hz)
        with self._lock:
            msg["monitor_source"] = str(self._monitor_source)
            msg["mic_array_profile"] = str(self.req.mic_array_profile)
            msg["channel_map"] = list(self._channel_map) if self._channel_map is not None else None
        msg["tracker_debug"] = dict(self._last_tracker_debug)
        with self._lock:
            self._metrics_state = msg
            self._metrics_version += 1

    def _pick_focus_direction(self, items: list[SpeakerStateItem]) -> float | None:
        if not items:
            return None
        with self._lock:
            selected = self._selected_speaker_id
            gains = dict(self._speaker_gain_delta_db)
        if self.req.processing_mode == "specific_speaker_enhancement" and selected is not None:
            for item in items:
                if int(item.speaker_id) == int(selected):
                    return float(item.direction_degrees)
            return None
        if self.req.processing_mode in {"localize_and_beamform", "beamform_from_ground_truth"}:
            best = max(items, key=lambda item: (float(item.activity_confidence), float(item.confidence)))
            return float(best.direction_degrees)
        if selected is not None and gains.get(selected):
            for item in items:
                if int(item.speaker_id) == int(selected):
                    return float(item.direction_degrees)
        return None

    def _render_output(self, frame_mc: np.ndarray, items: list[SpeakerStateItem]) -> np.ndarray:
        frame = np.asarray(frame_mc, dtype=np.float32)
        focus_direction = self._pick_focus_direction(items)
        if focus_direction is None:
            return np.mean(frame, axis=1).astype(np.float32, copy=False)

        mono = delay_and_sum_frame(
            frame,
            doa_deg=float(focus_direction),
            mic_geometry_xyz=self._mic_geometry_xyz,
            fs=int(self.req.sample_rate_hz),
            sound_speed_m_s=343.0,
        )
        if self.req.processing_mode == "specific_speaker_enhancement":
            with self._lock:
                selected = self._selected_speaker_id
                delta_db = 0.0 if selected is None else float(self._speaker_gain_delta_db.get(selected, 0.0))
            boost_db = max(0.0, _ratio_to_db(self.req.focus_ratio) + delta_db)
            mono = mono * float(10.0 ** (boost_db / 20.0))
        return np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)

    def _run(self) -> None:
        try:
            try:
                import sounddevice as sd
            except ImportError as exc:  # pragma: no cover - environment-dependent path
                raise RuntimeError("sounddevice is required for respeaker_live sessions") from exc

            audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
            sample_rate_hz = int(self.req.sample_rate_hz)
            channel_count = int(self.req.channel_count)
            frame_samples = max(1, sample_rate_hz // 100)

            device_idx = _find_input_device(sd, self.req.audio_device_query, channel_count)
            if device_idx is None:
                raise RuntimeError(
                    "No matching input device found for live session. "
                    f"Requested query={self.req.audio_device_query!r}. Available devices: {_device_debug_string(sd)}"
                )

            device_info = sd.query_devices(device_idx)
            self._last_device_name = str(device_info.get("name", f"device-{device_idx}"))

            tracker = SRPPeakTracker(
                mic_pos=self._mic_geometry_xyz.T,
                fs=sample_rate_hz,
                window_ms=160,
                nfft=512,
                overlap=0.5,
                freq_range=(200, 3000),
                max_sources=max(1, min(int(self.req.max_speakers_hint), channel_count)),
                prior_enabled=True,
                min_score=0.03,
                ema_alpha=0.35,
                hysteresis_margin=0.05,
                match_tolerance_deg=20.0,
                hold_frames=6,
                max_step_deg=12.0,
                score_decay=0.9,
                backend=str(self.req.localization_backend),
                grid_size=72,
                min_peak_separation_deg=15.0,
                small_aperture_bias=True,
                tracking_mode=str(self.req.tracking_mode),
                max_tracks=max(1, min(int(self.req.max_speakers_hint), 3)),
            )

            def callback(indata: np.ndarray, _frames: int, _time_info: Any, status: Any) -> None:
                if status:
                    self._last_tracker_debug["stream_status"] = str(status)
                frame = np.asarray(indata, dtype=np.float32)
                try:
                    audio_q.put_nowait(frame.copy())
                except queue.Full:
                    try:
                        audio_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        audio_q.put_nowait(frame.copy())
                    except queue.Full:
                        pass

            with sd.InputStream(
                device=device_idx,
                channels=channel_count,
                samplerate=sample_rate_hz,
                blocksize=frame_samples,
                dtype="float32",
                callback=callback,
            ):
                with self._lock:
                    self._status = "running"
                self._publish_event("started", detail=f"live capture on {self._last_device_name}")

                next_metrics_push = time.perf_counter() + 1.0
                while not self._stop.is_set():
                    try:
                        frame_mc = audio_q.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    t0 = time.perf_counter()
                    if frame_mc.ndim != 2 or frame_mc.shape[1] != channel_count:
                        continue
                    if self._channel_map is not None:
                        frame_mc = frame_mc[:, self._channel_map]
                    raw_mono = np.mean(frame_mc, axis=1).astype(np.float32, copy=False)
                    self._append_raw_audio(frame_mc, raw_mono)
                    peaks, scores, tracker_debug = tracker.update(frame_mc)
                    rms_by_channel = np.sqrt(np.mean(np.square(frame_mc), axis=0)).astype(np.float32, copy=False)
                    debug = dict(tracker_debug)
                    debug["last_frame_shape"] = [int(frame_mc.shape[0]), int(frame_mc.shape[1])]
                    debug["rms_by_channel"] = [float(v) for v in rms_by_channel]
                    self._last_tracker_debug = debug
                    items = self._build_speaker_items(peaks, scores)
                    output = self._render_output(frame_mc, items)
                    self._publish_audio_chunk(output, raw_mono)
                    self._processing_ms_total += (time.perf_counter() - t0) * 1000.0
                    self._processed_frames += 1

                    now = time.perf_counter()
                    if now >= next_metrics_push:
                        self._publish_metrics(audio_q.qsize())
                        next_metrics_push = now + 1.0

                self._publish_metrics(audio_q.qsize())
                with self._lock:
                    self._status = "stopped"
                self._publish_event("stopped")
        except Exception as exc:  # pragma: no cover - defensive reporting path
            with self._lock:
                self._status = "error"
            self._publish_event("error", detail=str(exc))

    def _write_summary(self) -> None:
        return
