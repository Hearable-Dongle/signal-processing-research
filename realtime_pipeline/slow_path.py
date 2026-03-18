from __future__ import annotations

from dataclasses import dataclass
import queue
import threading

import numpy as np

from .contracts import PipelineConfig, SpeakerGainDirection
from .separation_backends import SeparationBackend
from .shared_state import SharedPipelineState, Timer
from .tracking_modes import SUPPORTED_TRACKING_MODE, validate_tracking_mode


def _normalize_angle_deg(v: float) -> float:
    return float(v % 360.0)


def _angular_diff_deg(a: float, b: float) -> float:
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _angular_dist_deg(a: float, b: float) -> float:
    return abs(_angular_diff_deg(a, b))


@dataclass(slots=True)
class _CentroidState:
    speaker_id: int
    direction_deg: float
    confidence: float
    last_seen_ms: float
    last_observed_score: float
    observation_count: int
    velocity_deg_per_s: float = 0.0
    accepted_observations: tuple[tuple[float, float, float], ...] = ()


@dataclass(slots=True)
class _PendingSingleActiveSwitch:
    target_angle_deg: float
    score: float
    count: int
    last_seen_ms: float
    accepted_observations: tuple[tuple[float, float, float], ...] = ()


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
        del separation_backend, mic_geometry_xy
        self._cfg = config
        self._state = shared_state
        self._slow_queue = slow_queue
        self._stop = stop_event
        self._chunk_id = 0
        self._tracking_mode = validate_tracking_mode(str(config.tracking_mode))
        self._centroids: dict[int, _CentroidState] = {}
        self._next_speaker_id = 1
        self._centroid_match_window_deg = float(max(1.0, config.speaker_match_window_deg))
        self._centroid_association_mode = str(config.centroid_association_mode).strip().lower()
        self._centroid_association_sigma_deg = float(max(1.0, config.centroid_association_sigma_deg))
        self._centroid_association_min_score = float(np.clip(config.centroid_association_min_score, 0.0, 1.0))
        self._centroid_ema_alpha = 0.25
        self._centroid_active_hold_ms = 1500.0
        self._centroid_expiry_ms = 30000.0
        self._centroid_new_track_min_score = 0.35
        self._single_active_mode = bool(config.single_active or int(config.max_speakers_hint) <= 1)
        self._single_active_switch_jump_deg = 45.0
        self._single_active_switch_confirm_frames = 3
        self._single_active_switch_match_window_deg = 20.0
        self._single_active_switch_min_score = 0.6
        self._single_active_smoothing_window_ms = 500.0
        self._single_active_min_observation_score = float(
            np.clip(config.single_active_min_observation_score, 0.0, 1.0)
        )
        self._pending_single_active_switch: _PendingSingleActiveSwitch | None = None

    def _score_for_peak(self, idx: int, scores: tuple[float, ...] | None) -> float:
        if scores is None or idx >= len(scores):
            return 1.0
        return float(scores[idx])

    def _update_centroid_angle(self, current_angle_deg: float, observed_angle_deg: float) -> float:
        alpha = float(np.clip(self._centroid_ema_alpha, 0.0, 1.0))
        prev_rad = np.deg2rad(float(current_angle_deg))
        obs_rad = np.deg2rad(float(observed_angle_deg))
        x = ((1.0 - alpha) * float(np.cos(prev_rad))) + (alpha * float(np.cos(obs_rad)))
        y = ((1.0 - alpha) * float(np.sin(prev_rad))) + (alpha * float(np.sin(obs_rad)))
        if abs(x) < 1e-9 and abs(y) < 1e-9:
            return _normalize_angle_deg(observed_angle_deg)
        return _normalize_angle_deg(float(np.rad2deg(np.arctan2(y, x))))

    def _prune_recent_observations(
        self,
        observations: tuple[tuple[float, float, float], ...],
        *,
        ts_ms: float,
        window_ms: float,
    ) -> tuple[tuple[float, float, float], ...]:
        min_ts_ms = float(ts_ms - window_ms)
        return tuple(obs for obs in observations if float(obs[0]) >= min_ts_ms)

    def _append_recent_observation(
        self,
        observations: tuple[tuple[float, float, float], ...],
        *,
        ts_ms: float,
        angle_deg: float,
        score: float,
        window_ms: float,
    ) -> tuple[tuple[float, float, float], ...]:
        pruned = self._prune_recent_observations(observations, ts_ms=ts_ms, window_ms=window_ms)
        return pruned + ((float(ts_ms), float(_normalize_angle_deg(angle_deg)), float(score)),)

    def _weighted_circular_mean_deg(self, observations: tuple[tuple[float, float, float], ...]) -> float:
        if not observations:
            return 0.0
        x = 0.0
        y = 0.0
        for _ts_ms, angle_deg, score in observations:
            weight = float(max(score, 1e-6))
            angle_rad = np.deg2rad(float(angle_deg))
            x += weight * float(np.cos(angle_rad))
            y += weight * float(np.sin(angle_rad))
        if abs(x) < 1e-9 and abs(y) < 1e-9:
            return float(observations[-1][1])
        return _normalize_angle_deg(float(np.rad2deg(np.arctan2(y, x))))

    def _build_speaker_map(self, timestamp_ms: float) -> dict[int, SpeakerGainDirection]:
        speaker_map: dict[int, SpeakerGainDirection] = {}
        keep: dict[int, _CentroidState] = {}
        for sid, centroid in self._centroids.items():
            age_ms = float(timestamp_ms - centroid.last_seen_ms)
            if age_ms > self._centroid_expiry_ms:
                continue
            keep[int(sid)] = centroid
            active = bool(age_ms <= self._centroid_active_hold_ms)
            activity_confidence = float(np.clip(centroid.last_observed_score, 0.0, 1.0)) if active else 0.0
            confidence = float(np.clip(centroid.confidence, 0.0, 1.0))
            speaker_map[int(sid)] = SpeakerGainDirection(
                speaker_id=int(sid),
                direction_degrees=float(centroid.direction_deg),
                gain_weight=float(max(0.2, confidence)),
                confidence=confidence,
                active=active,
                activity_confidence=activity_confidence,
                updated_at_ms=float(centroid.last_seen_ms),
                identity_confidence=0.0,
                identity_maturity="centroid_tracker",
                predicted_direction_deg=float(centroid.direction_deg),
                angular_velocity_deg_per_chunk=float(centroid.velocity_deg_per_s),
                last_separator_stream_index=None,
                anchor_direction_deg=float(centroid.direction_deg),
                anchor_confidence=confidence,
                anchor_locked=False,
                anchor_last_confirmed_ms=float(centroid.last_seen_ms),
            )
        self._centroids = keep
        return speaker_map

    def _association_score(self, observed_angle_deg: float, centroid: _CentroidState) -> float:
        dist = _angular_dist_deg(observed_angle_deg, centroid.direction_deg)
        if dist > self._centroid_match_window_deg:
            return 0.0
        if self._centroid_association_mode == "gaussian":
            sigma = self._centroid_association_sigma_deg
            return float(np.exp(-0.5 * (dist / sigma) ** 2))
        return 1.0 - float(dist / self._centroid_match_window_deg)

    def _current_single_active_centroid_id(self) -> int | None:
        if not self._centroids:
            return None
        return max(
            self._centroids,
            key=lambda sid: (float(self._centroids[int(sid)].last_seen_ms), float(self._centroids[int(sid)].confidence)),
        )

    def _reset_pending_single_active_switch(self) -> None:
        self._pending_single_active_switch = None

    def _update_pending_single_active_switch(self, *, angle_deg: float, score: float, ts_ms: float) -> None:
        pending = self._pending_single_active_switch
        if pending is None:
            self._pending_single_active_switch = _PendingSingleActiveSwitch(
                target_angle_deg=float(_normalize_angle_deg(angle_deg)),
                score=float(score),
                count=1,
                last_seen_ms=float(ts_ms),
                accepted_observations=((float(ts_ms), float(_normalize_angle_deg(angle_deg)), float(score)),),
            )
            return
        if _angular_dist_deg(float(angle_deg), float(pending.target_angle_deg)) <= self._single_active_switch_match_window_deg:
            accepted_observations = self._append_recent_observation(
                pending.accepted_observations,
                ts_ms=ts_ms,
                angle_deg=angle_deg,
                score=score,
                window_ms=self._single_active_smoothing_window_ms,
            )
            merged_angle = self._weighted_circular_mean_deg(accepted_observations)
            self._pending_single_active_switch = _PendingSingleActiveSwitch(
                target_angle_deg=float(merged_angle),
                score=float(max(float(score), float(pending.score))),
                count=int(pending.count) + 1,
                last_seen_ms=float(ts_ms),
                accepted_observations=accepted_observations,
            )
            return
        self._pending_single_active_switch = _PendingSingleActiveSwitch(
            target_angle_deg=float(_normalize_angle_deg(angle_deg)),
            score=float(score),
            count=1,
            last_seen_ms=float(ts_ms),
            accepted_observations=((float(ts_ms), float(_normalize_angle_deg(angle_deg)), float(score)),),
        )

    def _maybe_commit_single_active_switch(self, *, ts_ms: float) -> int | None:
        pending = self._pending_single_active_switch
        if pending is None or int(pending.count) < self._single_active_switch_confirm_frames:
            return None
        current_id = self._current_single_active_centroid_id()
        if current_id is None:
            sid = self._next_speaker_id
            self._next_speaker_id += 1
        else:
            sid = int(current_id)
        self._centroids = {
            int(sid): _CentroidState(
                speaker_id=int(sid),
                direction_deg=float(_normalize_angle_deg(pending.target_angle_deg)),
                confidence=float(np.clip(pending.score, 0.0, 1.0)),
                last_seen_ms=float(ts_ms),
                last_observed_score=float(pending.score),
                observation_count=1,
                velocity_deg_per_s=0.0,
                accepted_observations=tuple(pending.accepted_observations),
            )
        }
        self._reset_pending_single_active_switch()
        return int(sid)

    def _process_centroid_frame(self, frame_mc: np.ndarray) -> None:
        del frame_mc
        srp = self._state.get_srp_snapshot()
        ts_ms = float(srp.timestamp_ms)
        peaks = list(srp.peaks_deg)
        scores = srp.peak_scores
        observations = [
            (float(peaks[idx]), self._score_for_peak(idx, scores))
            for idx in range(min(len(peaks), max(1, int(self._cfg.max_speakers_hint))))
        ]
        observations.sort(key=lambda item: item[1], reverse=True)
        with Timer() as t:
            assigned: set[int] = set()
            for angle_deg, score in observations:
                if self._single_active_mode and float(score) < self._single_active_min_observation_score:
                    continue
                best_id: int | None = None
                best_assoc_score = 0.0
                for sid, centroid in self._centroids.items():
                    if int(sid) in assigned:
                        continue
                    if float(ts_ms - centroid.last_seen_ms) > self._centroid_expiry_ms:
                        continue
                    assoc_score = self._association_score(angle_deg, centroid)
                    if assoc_score < self._centroid_association_min_score:
                        continue
                    if assoc_score > best_assoc_score:
                        best_assoc_score = assoc_score
                        best_id = int(sid)

                if best_id is None:
                    if self._single_active_mode and self._centroids:
                        current_id = self._current_single_active_centroid_id()
                        current = None if current_id is None else self._centroids.get(int(current_id))
                        jump_deg = None if current is None else _angular_dist_deg(angle_deg, current.direction_deg)
                        if (
                            current is not None
                            and jump_deg is not None
                            and jump_deg >= self._single_active_switch_jump_deg
                            and float(score) >= self._single_active_switch_min_score
                        ):
                            self._update_pending_single_active_switch(angle_deg=angle_deg, score=score, ts_ms=ts_ms)
                            switched_id = self._maybe_commit_single_active_switch(ts_ms=ts_ms)
                            if switched_id is not None:
                                assigned.add(int(switched_id))
                            continue
                        self._reset_pending_single_active_switch()
                        continue
                    if float(score) < self._centroid_new_track_min_score:
                        continue
                    sid = self._next_speaker_id
                    self._next_speaker_id += 1
                    self._centroids[int(sid)] = _CentroidState(
                        speaker_id=int(sid),
                        direction_deg=float(_normalize_angle_deg(angle_deg)),
                        confidence=float(np.clip(score, 0.0, 1.0)),
                        last_seen_ms=float(ts_ms),
                        last_observed_score=float(score),
                        observation_count=1,
                        accepted_observations=((float(ts_ms), float(_normalize_angle_deg(angle_deg)), float(score)),),
                    )
                    assigned.add(int(sid))
                    continue

                self._reset_pending_single_active_switch()
                prev = self._centroids[int(best_id)]
                accepted_observations = prev.accepted_observations
                if self._single_active_mode:
                    accepted_observations = self._append_recent_observation(
                        prev.accepted_observations,
                        ts_ms=ts_ms,
                        angle_deg=angle_deg,
                        score=score,
                        window_ms=self._single_active_smoothing_window_ms,
                    )
                    updated_angle = self._weighted_circular_mean_deg(accepted_observations)
                else:
                    updated_angle = self._update_centroid_angle(prev.direction_deg, angle_deg)
                dt_s = max((float(ts_ms) - float(prev.last_seen_ms)) / 1000.0, 1e-3)
                velocity_deg_per_s = _angular_diff_deg(updated_angle, prev.direction_deg) / dt_s
                self._centroids[int(best_id)] = _CentroidState(
                    speaker_id=int(best_id),
                    direction_deg=float(updated_angle),
                    confidence=float(np.clip((0.7 * prev.confidence) + (0.3 * score), 0.0, 1.0)),
                    last_seen_ms=float(ts_ms),
                    last_observed_score=float(score),
                    observation_count=int(prev.observation_count) + 1,
                    velocity_deg_per_s=float(velocity_deg_per_s),
                    accepted_observations=accepted_observations,
                )
                assigned.add(int(best_id))

            speaker_map = self._build_speaker_map(ts_ms)
            self._state.publish_speaker_map(speaker_map)

        self._state.incr_slow_chunk(t.elapsed_ms)
        self._state.incr_slow_stage_times(
            separation_ms=0.0,
            identity_ms=0.0,
            direction_ms=0.0,
            publish_ms=float(t.elapsed_ms),
        )
        self._chunk_id += 1

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._slow_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            if self._tracking_mode == SUPPORTED_TRACKING_MODE:
                self._process_centroid_frame(np.asarray(item, dtype=np.float32))
