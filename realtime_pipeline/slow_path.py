from __future__ import annotations

import queue
import threading

import numpy as np

from direction_assignment import DirectionAssignmentConfig, DirectionAssignmentEngine, build_direction_assignment_input
from speaker_identity_grouping import IdentityChunkInput, IdentityConfig, SpeakerIdentityGrouper

from .contracts import PipelineConfig, SpeakerGainDirection
from .separation_backends import SeparationBackend
from .shared_state import SharedPipelineState, Timer


class SlowPathWorker(threading.Thread):
    def __init__(
        self,
        *,
        config: PipelineConfig,
        shared_state: SharedPipelineState,
        slow_queue: "queue.Queue[np.ndarray | None]",
        separation_backend: SeparationBackend,
        mic_geometry_xy: np.ndarray,
        stop_event: threading.Event,
    ):
        super().__init__(name="SlowPathWorker", daemon=True)
        self._cfg = config
        self._state = shared_state
        self._slow_queue = slow_queue
        self._sep = separation_backend
        self._stop = stop_event
        self._chunk_id = 0
        self._chunk_samples = max(1, int(config.sample_rate_hz * config.slow_chunk_ms / 1000))
        self._frame_samples = max(1, int(config.sample_rate_hz * config.fast_frame_ms / 1000))

        self._identity = SpeakerIdentityGrouper(
            IdentityConfig(
                sample_rate_hz=config.sample_rate_hz,
                chunk_duration_ms=config.slow_chunk_ms,
                max_speakers=config.max_speakers_hint,
            )
        )
        self._direction = DirectionAssignmentEngine(
            mic_geometry=np.asarray(mic_geometry_xy, dtype=float),
            config=DirectionAssignmentConfig(
                sample_rate=config.sample_rate_hz,
                chunk_ms=config.slow_chunk_ms,
                max_user_boost_db=config.max_user_boost_db,
            ),
        )

    def _process_chunk(self, raw_chunk_mc: np.ndarray) -> None:
        ts_ms = 1000.0 * (self._chunk_id * self._chunk_samples) / self._cfg.sample_rate_hz

        with Timer() as t:
            mono_mix = np.mean(raw_chunk_mc, axis=1).astype(np.float32, copy=False)
            focus = self._state.get_focus_control_snapshot()
            self._direction.set_focus_speakers(None if focus.focused_speaker_ids is None else list(focus.focused_speaker_ids))
            self._direction.set_focus_direction(focus.focused_direction_deg)
            self._direction.set_focus_boost_db(focus.user_boost_db)

            srp = self._state.get_srp_snapshot()
            expected_speakers = len(srp.peaks_deg) if srp.peaks_deg else None
            if expected_speakers is not None:
                expected_speakers = min(max(1, expected_speakers), self._cfg.max_speakers_hint)

            separated = self._sep.separate(mono_mix, expected_speakers=expected_speakers)

            identity_out = self._identity.update(
                IdentityChunkInput(
                    chunk_id=self._chunk_id,
                    timestamp_ms=ts_ms,
                    sample_rate_hz=self._cfg.sample_rate_hz,
                    streams=separated,
                )
            )

            payload, _ = build_direction_assignment_input(
                chunk_id=self._chunk_id,
                timestamp_ms=ts_ms,
                raw_mic_chunk=raw_chunk_mc,
                separated_streams=separated,
                stream_to_speaker=identity_out.stream_to_speaker,
                active_speakers=identity_out.active_speakers,
                srp_doa_peaks_deg=list(srp.peaks_deg),
                srp_peak_scores=None if srp.peak_scores is None else list(srp.peak_scores),
            )
            direction_out = self._direction.update(payload)
            speaker_activity: dict[int, float] = {}
            for stream_idx, speaker_id in identity_out.stream_to_speaker.items():
                if speaker_id is None:
                    continue
                if stream_idx < 0 or stream_idx >= len(separated):
                    continue
                stream = np.asarray(separated[stream_idx], dtype=float).reshape(-1)
                rms = float(np.sqrt(np.mean(stream**2) + 1e-12))
                speaker_activity[int(speaker_id)] = max(speaker_activity.get(int(speaker_id), 0.0), rms)

            gain_by_id = {sid: float(w) for sid, w in zip(direction_out.target_speaker_ids, direction_out.target_weights)}
            speaker_map: dict[int, SpeakerGainDirection] = {}
            for sid, doa in direction_out.speaker_directions_deg.items():
                activity_rms = speaker_activity.get(int(sid), 0.0)
                activity_conf = float(np.clip((activity_rms - 0.005) / 0.03, 0.0, 1.0))
                speaker_map[int(sid)] = SpeakerGainDirection(
                    speaker_id=int(sid),
                    direction_degrees=float(doa),
                    gain_weight=float(gain_by_id.get(int(sid), 1.0)),
                    confidence=float(direction_out.speaker_confidence.get(int(sid), 0.0)),
                    active=bool(activity_conf >= 0.5),
                    activity_confidence=activity_conf,
                    updated_at_ms=ts_ms,
                )

            self._state.publish_speaker_map(speaker_map)

        self._state.incr_slow_chunk(t.elapsed_ms)
        self._chunk_id += 1

    def run(self) -> None:
        frames: list[np.ndarray] = []
        acc_samples = 0

        while not self._stop.is_set():
            try:
                item = self._slow_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            frame = np.asarray(item, dtype=np.float32)
            frames.append(frame)
            acc_samples += frame.shape[0]

            while acc_samples >= self._chunk_samples:
                chunk = np.concatenate(frames, axis=0)
                raw_chunk = chunk[: self._chunk_samples, :]
                residual = chunk[self._chunk_samples :, :]
                frames = [residual] if residual.size else []
                acc_samples = residual.shape[0] if residual.size else 0
                self._process_chunk(raw_chunk)

        if self._cfg.process_partial_chunk and frames:
            chunk = np.concatenate(frames, axis=0)
            if chunk.shape[0] < self._chunk_samples:
                chunk = np.pad(chunk, ((0, self._chunk_samples - chunk.shape[0]), (0, 0)))
            self._process_chunk(chunk[: self._chunk_samples, :])
