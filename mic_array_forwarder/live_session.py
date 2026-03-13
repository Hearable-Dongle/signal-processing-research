from __future__ import annotations

import math
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
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
from realtime_pipeline.catchup_metrics import compute_catchup_metrics
from realtime_pipeline.orchestrator import RealtimeSpeakerPipeline
from realtime_pipeline.session_runtime import (
    build_pipeline_config_from_request,
    build_separation_backend_for_request,
    public_speaker_rows,
)
from mic_array_forwarder.session import _wav_bytes_from_mono_float32
from mic_array_forwarder.tools.channel_plot_utils import default_channel_labels, render_multichannel_plot_png_bytes
from mic_array_forwarder.ws_codec import encode_audio_chunk
from simulation.mic_array_profiles import mic_positions_xyz

def _mic_geometry_from_profile(profile: str) -> np.ndarray:
    return mic_positions_xyz(profile)


def _now_ms() -> float:
    return time.time() * 1000.0

def _ratio_to_db(ratio: float) -> float:
    return float(20.0 * math.log10(max(float(ratio), 1e-6)))


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
        self._processed_audio_parts: deque[np.ndarray] = deque(maxlen=4500)
        self._raw_mix_sample_rate_hz = int(req.sample_rate_hz)
        self._pipeline: RealtimeSpeakerPipeline | None = None
        self._raw_frame_q: deque[np.ndarray] = deque(maxlen=256)
        self._observations: list[dict[str, Any]] = []
        self._obs_idx = 0
        self._last_device_name = ""
        self._monitor_source = str(req.monitor_source)
        self._mic_geometry_xyz = _mic_geometry_from_profile(str(req.mic_array_profile))
        self._capture_channel_count, self._channel_map = self._resolve_capture_layout(req)
        self._audio_q: queue.Queue[np.ndarray | None] | None = None

    @staticmethod
    def _resolve_capture_layout(req: SessionStartRequest) -> tuple[int, list[int] | None]:
        if req.channel_map is None:
            if str(req.mic_array_profile) == "respeaker_xvf3800_0650":
                return 6, [2, 3, 4, 5]
            return int(req.channel_count), None
        try:
            items = [int(v) for v in req.channel_map]
        except (TypeError, ValueError):
            if str(req.mic_array_profile) == "respeaker_xvf3800_0650":
                return 6, [2, 3, 4, 5]
            return int(req.channel_count), None
        if not items:
            return int(req.channel_count), None
        if len(set(items)) != len(items) or min(items) < 0:
            if str(req.mic_array_profile) == "respeaker_xvf3800_0650":
                return 6, [2, 3, 4, 5]
            return int(req.channel_count), None
        return max(int(req.channel_count), max(items) + 1), items

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        pipe = self._pipeline
        if pipe is not None:
            pipe.stop()
        if self._audio_q is not None:
            try:
                self._audio_q.put_nowait(None)
            except queue.Full:
                pass

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

    def get_processed_wav_bytes(self) -> bytes:
        with self._lock:
            if not self._processed_audio_parts:
                return b""
            raw = np.concatenate(list(self._processed_audio_parts), axis=0)
            sample_rate_hz = int(self._raw_mix_sample_rate_hz)
        return _wav_bytes_from_mono_float32(raw, sample_rate_hz)

    def get_raw_channel_count(self) -> int:
        if self._channel_map is not None:
            return int(len(self._channel_map))
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

    def get_raw_channel_plot_png_bytes(self, subtitle: str = "") -> bytes:
        with self._lock:
            if not self._raw_multichannel_parts:
                return b""
            raw = np.concatenate(list(self._raw_multichannel_parts), axis=0)
            sample_rate_hz = int(self._raw_mix_sample_rate_hz)
        if raw.ndim != 2 or raw.shape[1] <= 0:
            return b""
        return render_multichannel_plot_png_bytes(
            data=raw,
            sample_rate_hz=sample_rate_hz,
            channel_labels=default_channel_labels(raw.shape[1]),
            title="Raw mic channels",
            subtitle=subtitle or f"{self.session_id} · {self._last_device_name or self.req.audio_device_query or 'live capture'}",
        )

    def select_speaker(self, speaker_id: int) -> None:
        if self.req.processing_mode != "specific_speaker_enhancement":
            return
        with self._lock:
            sid = int(speaker_id)
            self._selected_speaker_id = sid
            self._speaker_gain_delta_db.setdefault(sid, 0.0)
        self._apply_focus_control()

    def adjust_speaker_gain(self, speaker_id: int, delta_db_step: int) -> float:
        if self.req.processing_mode != "specific_speaker_enhancement":
            return 0.0
        sid = int(speaker_id)
        step = int(delta_db_step)
        with self._lock:
            prev = float(self._speaker_gain_delta_db.get(sid, 0.0))
            curr = float(np.clip(prev + step, -12.0, 12.0))
            self._speaker_gain_delta_db[sid] = curr
        self._apply_focus_control()
        return curr

    def clear_focus(self) -> None:
        with self._lock:
            self._selected_speaker_id = None
        self._apply_focus_control()

    def set_monitor_source(self, source: str) -> None:
        with self._lock:
            self._monitor_source = str(source)

    def _next_audio_timestamp_ms(self) -> float:
        with self._lock:
            hop_ms = max(10, int(self.req.localization_hop_ms))
            ts = float(self._audio_frame_idx * hop_ms)
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

    def _on_audio_chunk(self, frame_mono: np.ndarray) -> None:
        with self._lock:
            source = str(self._monitor_source)
            raw_frame = self._raw_frame_q.popleft() if self._raw_frame_q else None
            self._processed_audio_parts.append(np.asarray(frame_mono, dtype=np.float32, copy=True))
        frame = raw_frame if source == "raw_mixed" and raw_frame is not None else frame_mono
        payload = encode_audio_chunk(self._next_audio_timestamp_ms(), frame)
        with self._lock:
            self._audio_seq += 1
            self._audio_chunks.append((self._audio_seq, payload))

    def _nearest_speaker_id(self, speakers: list[SpeakerStateItem]) -> int | None:
        if not speakers:
            return None
        with self._lock:
            selected = self._selected_speaker_id
        if selected is not None:
            for item in speakers:
                if int(item.speaker_id) == int(selected):
                    return int(selected)
        best = max(speakers, key=lambda s: (float(s.gain_weight), float(s.confidence)))
        return int(best.speaker_id)

    def _best_speaker_direction_deg(self, speakers: list[SpeakerStateItem]) -> float | None:
        if not speakers:
            return None
        best = max(speakers, key=lambda s: (float(s.gain_weight), float(s.confidence)))
        return float(best.direction_degrees) % 360.0

    def _publish_speaker_state(self) -> None:
        pipe = self._pipeline
        if pipe is None:
            return
        rows = public_speaker_rows(pipe.shared_state.get_speaker_map_snapshot(), algorithm_mode=str(self.req.algorithm_mode))
        speakers = [
            SpeakerStateItem(
                speaker_id=int(row["speaker_id"]),
                direction_degrees=float(row["direction_degrees"]),
                confidence=float(row["confidence"]),
                active=bool(row["active"]),
                activity_confidence=float(row["activity_confidence"]),
                gain_weight=float(row["gain_weight"]),
            )
            for row in rows
        ]
        msg = SpeakerStateMessage(timestamp_ms=_now_ms(), speakers=speakers, ground_truth=[]).model_dump()
        if self.req.processing_mode == "localize_and_beamform":
            self._apply_focus_control(speakers=speakers)
        nearest = self._nearest_speaker_id(speakers)
        with self._lock:
            self._speaker_state = msg
            self._speaker_state_version += 1
            self._observations.append(
                {
                    "frame_idx": int(self._obs_idx),
                    "timestamp_ms": float(msg["timestamp_ms"]),
                    "mode": str(self.req.processing_mode),
                    "locked_speaker_id": "" if self._selected_speaker_id is None else int(self._selected_speaker_id),
                    "nearest_speaker_id": "" if nearest is None else int(nearest),
                }
            )
            self._obs_idx += 1

    def _publish_metrics(self, queue_size: int) -> None:
        pipe = self._pipeline
        if pipe is None:
            return
        stats = pipe.stats_snapshot()
        with self._lock:
            observations = list(self._observations)
            monitor_source = str(self._monitor_source)
        catchup = compute_catchup_metrics(observations, stable_frames=3)
        msg = MetricsMessage(
            timestamp_ms=_now_ms(),
            fast_rtf=float(stats.fast_rtf),
            slow_rtf=float(stats.slow_rtf),
            fast_stage_avg_ms={
                "srp": float(stats.fast_srp_avg_ms),
                "beamform": float(stats.fast_beamform_avg_ms),
                "safety": float(stats.fast_safety_avg_ms),
                "sink": float(stats.fast_sink_avg_ms),
                "enqueue": float(stats.fast_enqueue_avg_ms),
                "capture_queue": float(queue_size),
            },
            slow_stage_avg_ms={
                "separation": float(stats.slow_separation_avg_ms),
                "identity": float(stats.slow_identity_avg_ms),
                "direction_assignment": float(stats.slow_direction_avg_ms),
                "publish": float(stats.slow_publish_avg_ms),
            },
            startup_lock_ms=float(catchup["startup_lock_ms"]),
            reacquire_catchup_ms_median=float(catchup["reacquire_catchup_ms_median"]),
            nearest_change_catchup_ms_median=float(catchup["nearest_change_catchup_ms_median"]),
        ).model_dump()
        msg["device_name"] = self._last_device_name
        msg["input_source"] = self.req.input_source
        msg["channel_count"] = int(self.req.channel_count)
        msg["sample_rate_hz"] = int(self.req.sample_rate_hz)
        msg["monitor_source"] = monitor_source
        msg["mic_array_profile"] = str(self.req.mic_array_profile)
        msg["channel_map"] = list(self._channel_map) if self._channel_map is not None else None
        with self._lock:
            self._metrics_state = msg
            self._metrics_version += 1

    def _apply_focus_control(self, speakers: list[SpeakerStateItem] | None = None) -> None:
        pipe = self._pipeline
        if pipe is None:
            return
        if self.req.processing_mode == "localize_and_beamform":
            direction = self._best_speaker_direction_deg(speakers or [])
            pipe.set_focus_control(focused_speaker_ids=None, focused_direction_deg=direction, user_boost_db=0.0)
            return
        with self._lock:
            selected = self._selected_speaker_id
            boost = 0.0 if selected is None else _ratio_to_db(self.req.focus_ratio) + float(self._speaker_gain_delta_db.get(selected, 0.0))
        if selected is None:
            pipe.set_focus_control(focused_speaker_ids=None, focused_direction_deg=None, user_boost_db=0.0)
        else:
            pipe.set_focus_control(focused_speaker_ids=[int(selected)], focused_direction_deg=None, user_boost_db=max(0.0, float(boost)))

    def _is_pipeline_alive(self) -> bool:
        pipe = self._pipeline
        if pipe is None:
            return False
        fast_alive = bool(getattr(pipe, "_fast", None) and pipe._fast.is_alive())
        slow_alive = bool(getattr(pipe, "_slow", None) and pipe._slow.is_alive())
        return bool(fast_alive or slow_alive)

    def _frame_iter(self, audio_q: queue.Queue[np.ndarray | None], capture_channel_count: int) -> Iterator[np.ndarray]:
        while True:
            if self._stop.is_set() and audio_q.empty():
                return
            try:
                frame_mc = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame_mc is None:
                return
            if frame_mc.ndim != 2 or frame_mc.shape[1] != capture_channel_count:
                continue
            if self._channel_map is not None:
                frame_mc = frame_mc[:, self._channel_map]
            raw_mono = np.mean(frame_mc, axis=1).astype(np.float32, copy=False)
            with self._lock:
                self._raw_frame_q.append(raw_mono.copy())
            self._append_raw_audio(frame_mc, raw_mono)
            yield frame_mc.astype(np.float32, copy=False)

    def _run(self) -> None:
        try:
            try:
                import sounddevice as sd
            except ImportError as exc:  # pragma: no cover - environment-dependent path
                raise RuntimeError("sounddevice is required for respeaker_live sessions") from exc

            audio_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=64)
            self._audio_q = audio_q
            sample_rate_hz = int(self.req.sample_rate_hz)
            capture_channel_count = int(self._capture_channel_count)
            output_channel_count = int(len(self._channel_map)) if self._channel_map is not None else int(self.req.channel_count)
            frame_samples = max(1, int(sample_rate_hz * max(10, int(self.req.localization_hop_ms)) / 1000))

            device_idx = _find_input_device(sd, self.req.audio_device_query, capture_channel_count)
            if device_idx is None:
                raise RuntimeError(
                    "No matching input device found for live session. "
                    f"Requested query={self.req.audio_device_query!r}. Available devices: {_device_debug_string(sd)}"
                )

            device_info = sd.query_devices(device_idx)
            self._last_device_name = str(device_info.get("name", f"device-{device_idx}"))

            def callback(indata: np.ndarray, _frames: int, _time_info: Any, status: Any) -> None:
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
                channels=capture_channel_count,
                samplerate=sample_rate_hz,
                blocksize=frame_samples,
                dtype="float32",
                callback=callback,
            ):
                with self._lock:
                    self._status = "running"
                self._publish_event("started", detail=f"live capture on {self._last_device_name}")

                cfg = build_pipeline_config_from_request(
                    self.req,
                    sample_rate_hz=sample_rate_hz,
                    max_speakers_hint=max(int(self.req.max_speakers_hint), output_channel_count, 1),
                )
                mic_geometry_xyz = np.asarray(self._mic_geometry_xyz, dtype=float)
                mic_geometry_xy = mic_geometry_xyz[:, :2] if mic_geometry_xyz.shape[1] >= 2 else mic_geometry_xyz.T[:2, :].T
                self._pipeline = RealtimeSpeakerPipeline(
                    config=cfg,
                    mic_geometry_xyz=mic_geometry_xyz,
                    mic_geometry_xy=np.asarray(mic_geometry_xy, dtype=float),
                    frame_iterator=self._frame_iter(audio_q, capture_channel_count),
                    frame_sink=self._on_audio_chunk,
                    separation_backend=build_separation_backend_for_request(self.req, cfg),
                )
                self._apply_focus_control()
                self._pipeline.start()

                next_metrics_push = time.perf_counter() + 1.0
                next_speaker_push = time.perf_counter()
                speaker_period_s = max(float(cfg.slow_chunk_ms) / 1000.0, 0.05)
                while True:
                    if self._stop.is_set():
                        pipe = self._pipeline
                        if pipe is not None:
                            pipe.stop()
                        try:
                            audio_q.put_nowait(None)
                        except queue.Full:
                            pass
                    now = time.perf_counter()
                    if now >= next_speaker_push:
                        self._publish_speaker_state()
                        next_speaker_push = now + speaker_period_s
                    if now >= next_metrics_push:
                        self._publish_metrics(audio_q.qsize())
                        next_metrics_push = now + 1.0
                    if not self._is_pipeline_alive():
                        break
                    time.sleep(0.02)

                if self._pipeline is not None:
                    self._pipeline.join(timeout=2.0)
                self._publish_speaker_state()
                self._publish_metrics(audio_q.qsize())
                with self._lock:
                    self._status = "stopped"
                self._publish_event("stopped")
        except Exception as exc:  # pragma: no cover - defensive reporting path
            with self._lock:
                self._status = "error"
            self._publish_event("error", detail=str(exc))
        finally:
            self._audio_q = None

    def _write_summary(self) -> None:
        return
