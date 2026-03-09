from __future__ import annotations

import json
import math
import threading
import time
import uuid
import wave
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from realtime_pipeline.catchup_metrics import compute_catchup_metrics
from realtime_pipeline.contracts import PipelineConfig
from realtime_pipeline.orchestrator import RealtimeSpeakerPipeline
from realtime_pipeline.separation_backends import MockSeparationBackend, build_default_backend
from simulation.simulation_config import SimulationConfig, SimulationSource
from simulation.simulator import run_simulation
from simulation.target_policy import iter_target_source_indices

from .models import (
    MetricsMessage,
    SCHEMA_VERSION,
    SessionEventMessage,
    SessionStartRequest,
    SessionStatusResponse,
    SpeakerStateItem,
    SpeakerStateMessage,
)
from .ws_codec import encode_audio_chunk


def _now_ms() -> float:
    return time.time() * 1000.0


def _ratio_to_db(ratio: float) -> float:
    return float(20.0 * math.log10(max(float(ratio), 1e-6)))


def _normalize_angle_deg(v: float) -> float:
    return float(v % 360.0)


def _wav_bytes_from_mono_float32(raw: np.ndarray, sample_rate_hz: int) -> bytes:
    if raw.size == 0:
        return b""
    clipped = np.clip(raw, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16, copy=False)
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _ground_truth_from_scene(scene_cfg: SimulationConfig) -> list[dict[str, float | int]]:
    mic_center = scene_cfg.microphone_array.mic_center
    out: list[dict[str, float | int]] = []
    for idx, src in enumerate(scene_cfg.audio.sources):
        dx = float(src.loc[0] - mic_center[0])
        dy = float(src.loc[1] - mic_center[1])
        doa = _normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))
        out.append({"source_id": int(idx), "direction_degrees": float(doa)})
    return out


def _inject_background_noise_source(
    scene_cfg: SimulationConfig, *, audio_path: str | None, gain: float
) -> None:
    path = "" if audio_path is None else str(audio_path).strip()
    if not path:
        return
    room = scene_cfg.room.dimensions
    # Place background source near a room corner away from mic center for diffuse-like field.
    loc = [max(0.2, float(room[0]) * 0.15), max(0.2, float(room[1]) * 0.85), 1.4]
    scene_cfg.audio.sources.append(
        SimulationSource(
            loc=loc,
            audio_path=path,
            gain=float(np.clip(float(gain), 0.0, 2.0)),
            classification="noise",
        )
    )


class DemoSession:
    def __init__(self, req: SessionStartRequest):
        self.session_id = uuid.uuid4().hex[:12]
        self.req = req
        self.started_at_ms = _now_ms()
        self._status = "starting"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"demo-session-{self.session_id}", daemon=True)
        self._pipeline: RealtimeSpeakerPipeline | None = None

        self._selected_speaker_id: int | None = None
        self._speaker_gain_delta_db: dict[int, float] = {}
        self._gain_history: list[dict[str, Any]] = []
        self._lock_timeline: list[dict[str, Any]] = []

        self._audio_chunks: deque[tuple[int, bytes]] = deque(maxlen=2048)
        self._audio_seq = 0
        self._audio_frame_idx = 0
        self._speaker_state: dict[str, Any] | None = None
        self._speaker_state_version = 0
        self._metrics_state: dict[str, Any] | None = None
        self._metrics_version = 0
        self._events: deque[tuple[int, dict[str, Any]]] = deque(maxlen=256)
        self._event_seq = 0

        self._observations: list[dict[str, Any]] = []
        self._obs_idx = 0
        self._ground_truth_speakers: list[dict[str, float | int]] = []
        self._ground_truth_focus_direction_deg: float | None = None
        self._raw_mix_mono: np.ndarray | None = None
        self._raw_multichannel: np.ndarray | None = None
        self._raw_mix_sample_rate_hz: int = 16000
        self._monitor_source = str(req.monitor_source)
        self._raw_frame_q: deque[np.ndarray] = deque(maxlen=256)

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
            raw = None if self._raw_mix_mono is None else self._raw_mix_mono.copy()
            sample_rate_hz = int(self._raw_mix_sample_rate_hz)
        if raw is None:
            return b""
        return _wav_bytes_from_mono_float32(raw, sample_rate_hz)

    def get_raw_channel_count(self) -> int:
        with self._lock:
            if self._raw_multichannel is None:
                return 0
            return int(self._raw_multichannel.shape[1])

    def get_raw_sample_rate_hz(self) -> int:
        with self._lock:
            return int(self._raw_mix_sample_rate_hz)

    def get_raw_channel_wav_bytes(self, channel_index: int) -> bytes:
        with self._lock:
            raw = None if self._raw_multichannel is None else self._raw_multichannel.copy()
            sample_rate_hz = int(self._raw_mix_sample_rate_hz)
        if raw is None or raw.ndim != 2:
            return b""
        idx = int(channel_index)
        if idx < 0 or idx >= raw.shape[1]:
            return b""
        return _wav_bytes_from_mono_float32(raw[:, idx], sample_rate_hz)

    def select_speaker(self, speaker_id: int) -> None:
        if self.req.processing_mode != "specific_speaker_enhancement":
            return
        with self._lock:
            sid = int(speaker_id)
            self._selected_speaker_id = sid
            self._speaker_gain_delta_db.setdefault(sid, 0.0)
            self._lock_timeline.append({"timestamp_ms": _now_ms(), "event": "selected", "speaker_id": sid})
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
            self._gain_history.append(
                {
                    "timestamp_ms": _now_ms(),
                    "speaker_id": sid,
                    "delta_db_step": step,
                    "delta_db_total": curr,
                }
            )
        self._apply_focus_control()
        return curr

    def clear_focus(self) -> None:
        with self._lock:
            self._selected_speaker_id = None
            self._lock_timeline.append({"timestamp_ms": _now_ms(), "event": "cleared", "speaker_id": ""})
        self._apply_focus_control()

    def set_monitor_source(self, source: str) -> None:
        with self._lock:
            self._monitor_source = str(source)

    def _next_audio_timestamp_ms(self) -> float:
        with self._lock:
            ts = float(self._audio_frame_idx * 10)
            self._audio_frame_idx += 1
            return ts

    def _on_audio_chunk(self, frame_mono: np.ndarray) -> None:
        with self._lock:
            source = str(self._monitor_source)
            raw_frame = self._raw_frame_q.popleft() if self._raw_frame_q else None
        if source == "raw_mixed" and raw_frame is not None:
            frame = raw_frame
        else:
            frame = frame_mono
        payload = encode_audio_chunk(self._next_audio_timestamp_ms(), frame)
        with self._lock:
            self._audio_seq += 1
            self._audio_chunks.append((self._audio_seq, payload))

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
        return _normalize_angle_deg(float(best.direction_degrees))

    def _publish_speaker_state(self) -> None:
        pipe = self._pipeline
        if pipe is None:
            return
        speaker_map = pipe.shared_state.get_speaker_map_snapshot()
        speakers = [
            SpeakerStateItem(
                speaker_id=int(v.speaker_id),
                direction_degrees=float(v.direction_degrees),
                confidence=float(v.confidence),
                active=bool(v.active),
                activity_confidence=float(v.activity_confidence),
                gain_weight=float(v.gain_weight),
            )
            for v in speaker_map.values()
        ]
        with self._lock:
            ground_truth = list(self._ground_truth_speakers)
        msg = SpeakerStateMessage(timestamp_ms=_now_ms(), speakers=speakers, ground_truth=ground_truth).model_dump()

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

    def _publish_metrics(self) -> None:
        pipe = self._pipeline
        if pipe is None:
            return
        stats = pipe.stats_snapshot()
        with self._lock:
            observations = list(self._observations)
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
        with self._lock:
            self._metrics_state = msg
            self._metrics_version += 1

    def _apply_focus_control(self, speakers: list[SpeakerStateItem] | None = None) -> None:
        pipe = self._pipeline
        if pipe is None:
            return

        mode = self.req.processing_mode
        if mode == "localize_and_beamform":
            direction = self._best_speaker_direction_deg(speakers or [])
            pipe.set_focus_control(
                focused_speaker_ids=None,
                focused_direction_deg=direction,
                user_boost_db=0.0,
            )
            return

        if mode == "beamform_from_ground_truth":
            pipe.set_focus_control(
                focused_speaker_ids=None,
                focused_direction_deg=self._ground_truth_focus_direction_deg,
                user_boost_db=0.0,
            )
            return

        with self._lock:
            selected = self._selected_speaker_id
            if selected is None:
                boost = 0.0
            else:
                delta = float(self._speaker_gain_delta_db.get(selected, 0.0))
                boost = _ratio_to_db(self.req.focus_ratio) + delta

        if selected is None:
            pipe.set_focus_control(
                focused_speaker_ids=None,
                focused_direction_deg=None,
                user_boost_db=0.0,
            )
        else:
            pipe.set_focus_control(
                focused_speaker_ids=[int(selected)],
                focused_direction_deg=None,
                user_boost_db=max(0.0, float(boost)),
            )

    def _is_pipeline_alive(self) -> bool:
        pipe = self._pipeline
        if pipe is None:
            return False
        fast_alive = bool(getattr(pipe, "_fast", None) and pipe._fast.is_alive())
        slow_alive = bool(getattr(pipe, "_slow", None) and pipe._slow.is_alive())
        return bool(fast_alive or slow_alive)

    def _write_summary(self) -> None:
        out_dir = Path("realtime_demo/output") / self.session_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            summary = {
                "session_id": self.session_id,
                "status": self._status,
                "request": self.req.model_dump(),
                "started_at_ms": float(self.started_at_ms),
                "ended_at_ms": _now_ms(),
                "gain_adjustment_history": list(self._gain_history),
                "lock_timeline": list(self._lock_timeline),
                "last_metrics": self._metrics_state,
                "schema_version": SCHEMA_VERSION,
            }
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _run(self) -> None:
        try:
            sim_cfg = SimulationConfig.from_file(self.req.scene_config_path)
            _inject_background_noise_source(
                sim_cfg,
                audio_path=self.req.background_noise_audio_path,
                gain=self.req.background_noise_gain,
            )
            mic_audio, mic_pos, _source_signals = run_simulation(sim_cfg)
            with self._lock:
                self._raw_multichannel = mic_audio.astype(np.float32, copy=True)
                self._raw_mix_mono = np.mean(mic_audio, axis=1).astype(np.float32, copy=False)
                self._raw_mix_sample_rate_hz = int(sim_cfg.audio.fs)
                self._ground_truth_speakers = _ground_truth_from_scene(sim_cfg)
                target_source_ids = list(iter_target_source_indices(sim_cfg))
                if target_source_ids:
                    target_source_id = int(target_source_ids[0])
                else:
                    target_source_id = 0
                focus_direction = None
                for item in self._ground_truth_speakers:
                    if int(item["source_id"]) == target_source_id:
                        focus_direction = _normalize_angle_deg(float(item["direction_degrees"]))
                        break
                self._ground_truth_focus_direction_deg = focus_direction

            cfg = PipelineConfig(
                sample_rate_hz=int(sim_cfg.audio.fs),
                fast_frame_ms=10,
                slow_chunk_ms=int(self.req.slow_chunk_ms),
                max_speakers_hint=max(int(self.req.max_speakers_hint), len(list(iter_target_source_indices(sim_cfg))), 1),
                beamforming_mode=str(self.req.beamforming_mode),
                localization_backend=str(self.req.localization_backend),
                tracking_mode=str(self.req.tracking_mode),
                output_normalization_enabled=bool(self.req.output_normalization_enabled),
                output_allow_amplification=bool(self.req.output_allow_amplification),
            )
            frame_samples = max(1, int(cfg.sample_rate_hz * cfg.fast_frame_ms / 1000))

            def frame_iter():
                total = mic_audio.shape[0]
                frame_period_s = float(cfg.fast_frame_ms) / 1000.0
                next_deadline = time.perf_counter()
                for start in range(0, total, frame_samples):
                    if self._stop.is_set():
                        break
                    end = min(total, start + frame_samples)
                    frame = mic_audio[start:end, :]
                    if frame.shape[0] < frame_samples:
                        frame = np.pad(frame, ((0, frame_samples - frame.shape[0]), (0, 0)))
                    raw_mono = np.mean(frame, axis=1).astype(np.float32, copy=False)
                    with self._lock:
                        self._raw_frame_q.append(raw_mono)
                    yield frame.astype(np.float32, copy=False)
                    next_deadline += frame_period_s
                    sleep_s = next_deadline - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)

            if self.req.separation_mode == "mock":
                sep = MockSeparationBackend(n_streams=cfg.max_speakers_hint)
            else:
                sep = build_default_backend(cfg)

            mic_geometry_xyz = np.asarray(mic_pos, dtype=float)
            mic_geometry_xy = mic_geometry_xyz[:2, :].T

            self._pipeline = RealtimeSpeakerPipeline(
                config=cfg,
                mic_geometry_xyz=mic_geometry_xyz,
                mic_geometry_xy=mic_geometry_xy,
                frame_iterator=frame_iter(),
                frame_sink=self._on_audio_chunk,
                separation_backend=sep,
            )

            self._apply_focus_control()
            self._pipeline.start()
            with self._lock:
                self._status = "running"
            self._publish_event("started")

            next_speaker_push = time.perf_counter()
            next_metrics_push = time.perf_counter()
            speaker_period_s = max(float(cfg.slow_chunk_ms) / 1000.0, 0.05)
            while True:
                if self._stop.is_set():
                    pipe = self._pipeline
                    if pipe is not None:
                        pipe.stop()

                now = time.perf_counter()
                if now >= next_speaker_push:
                    self._publish_speaker_state()
                    next_speaker_push = now + speaker_period_s
                if now >= next_metrics_push:
                    self._publish_metrics()
                    next_metrics_push = now + 1.0

                if not self._is_pipeline_alive():
                    break
                time.sleep(0.02)

            if self._pipeline is not None:
                self._pipeline.join(timeout=2.0)
            self._publish_speaker_state()
            self._publish_metrics()

            with self._lock:
                self._status = "stopped"
            self._publish_event("stopped")
            self._write_summary()
        except Exception as exc:  # pragma: no cover - defensive reporting path
            with self._lock:
                self._status = "error"
            self._publish_event("error", detail=str(exc))
            self._write_summary()
