from __future__ import annotations

import queue
import threading
from time import perf_counter

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
                backend=config.identity_backend,
                max_speakers=config.max_speakers_hint,
                continuity_bonus=config.identity_continuity_bonus,
                switch_penalty=config.identity_switch_penalty,
                hold_similarity_threshold=config.identity_hold_similarity_threshold,
                carry_forward_chunks=config.identity_carry_forward_chunks,
                confidence_decay=config.identity_confidence_decay,
                speaker_embedding_model=config.identity_speaker_embedding_model,
                speaker_embedding_device=config.identity_speaker_embedding_device,
                speaker_embedding_min_speech_ms=config.identity_speaker_embedding_min_speech_ms,
                speaker_embedding_buffer_ms=config.identity_speaker_embedding_buffer_ms,
                speaker_embedding_update_interval_chunks=config.identity_speaker_embedding_update_interval_chunks,
                speaker_embedding_match_threshold=config.identity_speaker_embedding_match_threshold,
                speaker_embedding_merge_threshold=config.identity_speaker_embedding_merge_threshold,
                speaker_embedding_margin=config.identity_speaker_embedding_margin,
                provisional_speaker_timeout_chunks=config.identity_provisional_speaker_timeout_chunks,
            )
        )
        self._direction = DirectionAssignmentEngine(
            mic_geometry=np.asarray(mic_geometry_xy, dtype=float),
            config=DirectionAssignmentConfig(
                sample_rate=config.sample_rate_hz,
                chunk_ms=config.slow_chunk_ms,
                focus_gain_db=config.direction_focus_gain_db,
                non_focus_attenuation_db=config.direction_non_focus_attenuation_db,
                max_user_boost_db=config.max_user_boost_db,
                transition_penalty_deg=config.direction_transition_penalty_deg,
                min_confidence_for_switch=config.direction_min_confidence_for_switch,
                hold_confidence_decay=config.direction_hold_confidence_decay,
                stale_confidence_decay=config.direction_stale_confidence_decay,
                min_persist_confidence=config.direction_min_persist_confidence,
            ),
        )
        self._published_map: dict[int, SpeakerGainDirection] = {}

    def _process_chunk(self, raw_chunk_mc: np.ndarray) -> None:
        ts_ms = 1000.0 * (self._chunk_id * self._chunk_samples) / self._cfg.sample_rate_hz

        with Timer() as t:
            separation_ms = 0.0
            identity_ms = 0.0
            direction_ms = 0.0
            publish_ms = 0.0

            mono_mix = np.mean(raw_chunk_mc, axis=1).astype(np.float32, copy=False)
            focus = self._state.get_focus_control_snapshot()
            self._direction.set_focus_speakers(None if focus.focused_speaker_ids is None else list(focus.focused_speaker_ids))
            self._direction.set_focus_direction(focus.focused_direction_deg)
            self._direction.set_focus_boost_db(focus.user_boost_db)

            srp = self._state.get_srp_snapshot()
            expected_speakers = len(srp.peaks_deg) if srp.peaks_deg else None
            if expected_speakers is not None:
                expected_speakers = min(max(1, expected_speakers), self._cfg.max_speakers_hint)

            t0 = perf_counter()
            separated = self._sep.separate(mono_mix, expected_speakers=expected_speakers)
            separation_ms += (perf_counter() - t0) * 1000.0

            t0 = perf_counter()
            identity_out = self._identity.update(
                IdentityChunkInput(
                    chunk_id=self._chunk_id,
                    timestamp_ms=ts_ms,
                    sample_rate_hz=self._cfg.sample_rate_hz,
                    streams=separated,
                )
            )
            identity_ms += (perf_counter() - t0) * 1000.0

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
            t0 = perf_counter()
            direction_out = self._direction.update(payload)
            direction_ms += (perf_counter() - t0) * 1000.0
            t0 = perf_counter()
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

            speaker_map = self._merge_with_published_map(speaker_map, ts_ms)
            self._state.publish_speaker_map(speaker_map)
            publish_ms += (perf_counter() - t0) * 1000.0

        self._state.incr_slow_chunk(t.elapsed_ms)
        self._state.incr_slow_stage_times(
            separation_ms=separation_ms,
            identity_ms=identity_ms,
            direction_ms=direction_ms,
            publish_ms=publish_ms,
        )
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

    def _merge_with_published_map(self, speaker_map: dict[int, SpeakerGainDirection], timestamp_ms: float) -> dict[int, SpeakerGainDirection]:
        out: dict[int, SpeakerGainDirection] = {}
        refresh_min = float(self._cfg.speaker_map_min_confidence_for_refresh)
        hold_ms = float(self._cfg.speaker_map_hold_ms)
        conf_decay = float(np.clip(self._cfg.speaker_map_confidence_decay, 0.0, 1.0))
        activity_decay = float(np.clip(self._cfg.speaker_map_activity_decay, 0.0, 1.0))

        for sid, item in speaker_map.items():
            if float(item.confidence) >= refresh_min:
                out[int(sid)] = item
                continue
            prev = self._published_map.get(int(sid))
            if prev is None:
                out[int(sid)] = item
                continue
            age_ms = float(timestamp_ms - prev.updated_at_ms)
            if age_ms > hold_ms:
                out[int(sid)] = item
                continue
            out[int(sid)] = SpeakerGainDirection(
                speaker_id=prev.speaker_id,
                direction_degrees=prev.direction_degrees,
                gain_weight=item.gain_weight,
                confidence=float(np.clip(prev.confidence * conf_decay, 0.0, 1.0)),
                active=bool(prev.active),
                activity_confidence=float(np.clip(prev.activity_confidence * activity_decay, 0.0, 1.0)),
                updated_at_ms=prev.updated_at_ms,
            )

        for sid, prev in self._published_map.items():
            if int(sid) in out:
                continue
            age_ms = float(timestamp_ms - prev.updated_at_ms)
            if age_ms > hold_ms:
                continue
            out[int(sid)] = SpeakerGainDirection(
                speaker_id=prev.speaker_id,
                direction_degrees=prev.direction_degrees,
                gain_weight=prev.gain_weight,
                confidence=float(np.clip(prev.confidence * conf_decay, 0.0, 1.0)),
                active=bool(prev.active),
                activity_confidence=float(np.clip(prev.activity_confidence * activity_decay, 0.0, 1.0)),
                updated_at_ms=prev.updated_at_ms,
            )

        self._published_map = dict(out)
        return out
