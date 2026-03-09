from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from beamforming.localization_bridge import normalize_doa_list

from .localization_backends import _normalize_spectrum, build_localization_backend


def _angular_diff_deg(a: float, b: float) -> float:
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _angular_dist_deg(a: float, b: float) -> float:
    return abs(_angular_diff_deg(a, b))


def _grid_angles_deg(n: int) -> np.ndarray:
    return np.linspace(0.0, 360.0, int(n), endpoint=False, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PeakCandidate:
    angle_deg: float
    score: float
    raw_score: float


@dataclass
class _TrackedPeak:
    track_id: int
    angle_deg: float
    velocity_deg_per_s: float
    confidence: float
    age_frames: int
    missed_frames: int
    active: bool
    last_seen_frame: int


class SRPPeakTracker:
    def __init__(
        self,
        *,
        mic_pos: np.ndarray,
        fs: int,
        window_ms: int,
        nfft: int,
        overlap: float,
        freq_range: tuple[int, int],
        max_sources: int,
        prior_enabled: bool = True,
        min_score: float = 0.05,
        ema_alpha: float = 0.35,
        hysteresis_margin: float = 0.08,
        match_tolerance_deg: float = 20.0,
        hold_frames: int = 8,
        max_step_deg: float = 12.0,
        score_decay: float = 0.9,
        backend: str = "srp_phat_legacy",
        grid_size: int = 72,
        min_peak_separation_deg: float = 15.0,
        small_aperture_bias: bool = True,
        sound_speed_m_s: float = 343.0,
        tracking_mode: str = "multi_peak_v2",
        max_tracks: int | None = None,
        max_assoc_distance_deg: float = 20.0,
        track_hold_frames: int = 5,
        track_kill_frames: int = 9,
        new_track_min_confidence: float = 0.42,
        track_confidence_decay: float = 0.88,
        velocity_alpha: float = 0.35,
        angle_alpha: float = 0.30,
        min_relative_peak_score: float = 0.28,
        min_peak_contrast: float = 0.08,
        single_source_motion_filter_enabled: bool = True,
    ):
        self._fs = int(fs)
        self._window_samples = max(1, int(window_ms * fs / 1000))
        self._frames: deque[np.ndarray] = deque()
        self._total = 0
        self._frame_idx = 0
        self._prior_enabled = bool(prior_enabled)
        self._min_score = float(min_score)
        self._ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self._hysteresis_margin = float(max(0.0, hysteresis_margin))
        self._match_tolerance_deg = float(max(1.0, match_tolerance_deg))
        self._hold_frames = int(max(0, hold_frames))
        self._max_step_deg = float(max(0.1, max_step_deg))
        self._score_decay = float(np.clip(score_decay, 0.0, 1.0))
        self._tracking_mode = str(tracking_mode)
        self._single_source_motion_filter_enabled = bool(single_source_motion_filter_enabled)
        nfft_eff = max(32, min(int(nfft), self._window_samples))
        overlap_eff = float(overlap)
        if int(nfft_eff * overlap_eff) >= nfft_eff:
            overlap_eff = float((nfft_eff - 1) / nfft_eff)
        self._backend_name = str(backend)
        if self._tracking_mode == "multi_peak_v2" and self._backend_name == "weighted_srp_dp":
            if abs(float(max_assoc_distance_deg) - 20.0) < 1e-9:
                max_assoc_distance_deg = 14.0
            if int(track_hold_frames) == 5:
                track_hold_frames = 3
            if int(track_kill_frames) == 9:
                track_kill_frames = 6
            if abs(float(new_track_min_confidence) - 0.42) < 1e-9:
                new_track_min_confidence = 0.55
            if abs(float(velocity_alpha) - 0.35) < 1e-9:
                velocity_alpha = 0.25
            if abs(float(angle_alpha) - 0.30) < 1e-9:
                angle_alpha = 0.20
            if abs(float(min_relative_peak_score) - 0.28) < 1e-9:
                min_relative_peak_score = 0.38
            if abs(float(min_peak_contrast) - 0.08) < 1e-9:
                min_peak_contrast = 0.12
        self._tracks: list[_TrackedPeak] = []
        self._next_track_id = 1
        self._max_tracks = int(max_tracks or max_sources)
        self._max_assoc_distance_deg = float(max_assoc_distance_deg)
        self._track_hold_frames = int(track_hold_frames)
        self._track_kill_frames = int(track_kill_frames)
        self._new_track_min_confidence = float(new_track_min_confidence)
        self._track_confidence_decay = float(track_confidence_decay)
        self._velocity_alpha = float(np.clip(velocity_alpha, 0.0, 1.0))
        self._angle_alpha = float(np.clip(angle_alpha, 0.0, 1.0))
        self._min_relative_peak_score = float(min_relative_peak_score)
        self._min_peak_contrast = float(min_peak_contrast)
        self._min_peak_separation_deg = float(min_peak_separation_deg)
        self._update_interval_s = max(1e-3, float(window_ms) / 1000.0)
        self._backend = build_localization_backend(
            self._backend_name,
            mic_pos=np.asarray(mic_pos, dtype=float),
            fs=fs,
            nfft=nfft_eff,
            overlap=overlap_eff,
            freq_range=freq_range,
            max_sources=max_sources,
            sound_speed_m_s=sound_speed_m_s,
            grid_size=grid_size,
            min_separation_deg=min_peak_separation_deg,
            small_aperture_bias=small_aperture_bias,
        )
        self._max_sources = int(max_sources)
        self._single_source_backends = {"music_1src", "gcc_tdoa_1src"}
        self._single_state_x: np.ndarray | None = None
        self._single_state_p: np.ndarray | None = None

    def _unwrap_measurement(self, meas_deg: float, ref_deg: float) -> float:
        return float(ref_deg + _angular_diff_deg(meas_deg, ref_deg))

    def _single_source_predict_update(self, peaks: list[float], scores: list[float] | None, extra_debug: dict) -> tuple[list[float], list[float] | None, dict]:
        dt = self._update_interval_s
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        H = np.array([[1.0, 0.0]], dtype=np.float64)
        q_angle = 6.0
        q_vel = 40.0
        Q = np.array([[q_angle * dt * dt, 0.0], [0.0, q_vel * dt]], dtype=np.float64)
        if self._single_state_x is None:
            init_angle = float(peaks[0]) if peaks else 0.0
            self._single_state_x = np.array([init_angle, 0.0], dtype=np.float64)
            self._single_state_p = np.diag([100.0, 1000.0]).astype(np.float64)
        assert self._single_state_p is not None
        x_pred = F @ self._single_state_x
        p_pred = F @ self._single_state_p @ F.T + Q
        gate = "predict_only"
        measurement_used = False
        filtered_score = float(scores[0]) if scores else 0.0
        if peaks and self._single_source_motion_filter_enabled:
            meas = self._unwrap_measurement(float(peaks[0]), float(x_pred[0]))
            score = float(scores[0]) if scores else 0.5
            score = float(np.clip(score, 0.0, 1.0))
            R = np.array([[max(4.0, 180.0 * (1.0 - score) + 6.0)]], dtype=np.float64)
            y = np.array([[meas]], dtype=np.float64) - (H @ x_pred.reshape(-1, 1))
            S = H @ p_pred @ H.T + R
            K = p_pred @ H.T @ np.linalg.inv(S)
            x_upd = x_pred.reshape(-1, 1) + K @ y
            p_upd = (np.eye(2) - K @ H) @ p_pred
            self._single_state_x = x_upd.reshape(-1)
            self._single_state_p = p_upd
            measurement_used = True
            gate = "update"
        else:
            self._single_state_x = x_pred
            self._single_state_p = p_pred
            filtered_score = max(0.0, filtered_score * 0.9)
        filtered_angle = float(self._single_state_x[0] % 360.0)
        filtered_vel = float(self._single_state_x[1])
        self._frame_idx += 1
        debug = {
            "backend": self._backend_name,
            "raw_peaks_deg": list(peaks),
            "raw_peak_scores": [] if scores is None else list(scores),
            "tracked_peaks_deg": [filtered_angle] if peaks or measurement_used else [],
            "tracked_peak_scores": [filtered_score] if peaks or measurement_used else [],
            "held_tracks": 0 if measurement_used else 1,
            "single_source_filter_state": {"angle_deg": filtered_angle, "velocity_deg_per_s": filtered_vel},
            "single_source_filter_mode": gate,
            **extra_debug,
        }
        if not peaks and not measurement_used:
            return [], None, debug
        return [filtered_angle], [filtered_score], debug

    def _step_limit(self, prev_deg: float, next_deg: float) -> float:
        delta = _angular_diff_deg(next_deg, prev_deg)
        step = float(np.clip(delta, -self._max_step_deg, self._max_step_deg))
        return float((prev_deg + step) % 360.0)

    def _ema_angle(self, prev_deg: float, new_deg: float) -> float:
        p = np.deg2rad(float(prev_deg))
        n = np.deg2rad(float(new_deg))
        pv = np.array([np.cos(p), np.sin(p)], dtype=np.float64)
        nv = np.array([np.cos(n), np.sin(n)], dtype=np.float64)
        vv = (1.0 - self._ema_alpha) * pv + self._ema_alpha * nv
        if float(np.linalg.norm(vv)) < 1e-12:
            return float(new_deg % 360.0)
        return float(np.degrees(np.arctan2(vv[1], vv[0])) % 360.0)

    def _extract_candidates(self, spectrum: np.ndarray | None) -> tuple[list[PeakCandidate], dict]:
        if spectrum is None:
            return [], {
                "raw_candidate_peaks_deg": [],
                "selected_peaks_deg": [],
                "suppressed_peaks_deg": [],
            }
        spec = np.asarray(spectrum, dtype=np.float64).reshape(-1)
        if spec.size == 0:
            return [], {
                "raw_candidate_peaks_deg": [],
                "selected_peaks_deg": [],
                "suppressed_peaks_deg": [],
            }
        if np.max(spec) > 0:
            spec = spec / np.max(spec)
        spec = np.clip(spec, 0.0, None)
        angles = _grid_angles_deg(spec.size)
        median = float(np.median(spec))
        candidates: list[PeakCandidate] = []
        for idx in range(spec.size):
            prev_v = spec[(idx - 1) % spec.size]
            cur_v = spec[idx]
            next_v = spec[(idx + 1) % spec.size]
            if cur_v > prev_v and cur_v >= next_v:
                contrast = float(cur_v - max(prev_v, next_v))
                if cur_v < self._min_relative_peak_score or contrast < self._min_peak_contrast:
                    continue
                candidates.append(PeakCandidate(angle_deg=float(angles[idx]), score=float(cur_v), raw_score=float(cur_v)))
        candidates.sort(key=lambda item: item.score, reverse=True)
        selected: list[PeakCandidate] = []
        suppressed: list[float] = []
        for cand in candidates:
            if len(selected) >= self._max_tracks:
                suppressed.append(float(cand.angle_deg))
                continue
            if any(_angular_dist_deg(cand.angle_deg, sel.angle_deg) < self._min_peak_separation_deg for sel in selected):
                suppressed.append(float(cand.angle_deg))
                continue
            if len(selected) == 2 and cand.score < max(self._new_track_min_confidence, self._min_relative_peak_score + 0.08):
                suppressed.append(float(cand.angle_deg))
                continue
            selected.append(cand)
        debug = {
            "raw_candidate_peaks_deg": [float(c.angle_deg) for c in candidates],
            "selected_peaks_deg": [float(c.angle_deg) for c in selected],
            "suppressed_peaks_deg": suppressed,
            "peak_threshold_relative": self._min_relative_peak_score,
            "peak_threshold_contrast": self._min_peak_contrast,
            "spectrum_median": median,
        }
        return selected, debug

    def _update_legacy_tracks(self, peaks: list[float], scores: list[float] | None, extra_debug: dict) -> tuple[list[float], list[float] | None, dict]:
        if not self._prior_enabled:
            self._frame_idx += 1
            return peaks, scores, {
                "backend": self._backend_name,
                "raw_peaks_deg": list(peaks),
                "raw_peak_scores": [] if scores is None else list(scores),
                "tracked_peaks_deg": list(peaks),
                "tracked_peak_scores": [] if scores is None else list(scores),
                "held_tracks": 0,
                **extra_debug,
            }

        raw_scores = list(scores) if scores else [1.0] * len(peaks)
        used_tracks: set[int] = set()
        next_tracks: list[_TrackedPeak] = []
        matched_raw: set[int] = set()

        for raw_idx, (angle_deg, score) in enumerate(zip(peaks, raw_scores)):
            if float(score) < self._min_score:
                continue
            best_idx = None
            best_dist = None
            for idx, track in enumerate(self._tracks):
                if idx in used_tracks:
                    continue
                dist = _angular_dist_deg(angle_deg, track.angle_deg)
                if dist > self._match_tolerance_deg:
                    continue
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is None:
                continue
            track = self._tracks[best_idx]
            used_tracks.add(best_idx)
            matched_raw.add(raw_idx)
            angle_next = float(angle_deg)
            if float(score) + self._hysteresis_margin < track.confidence:
                angle_next = track.angle_deg
            limited = self._step_limit(track.angle_deg, angle_next)
            next_tracks.append(
                _TrackedPeak(
                    track_id=track.track_id,
                    angle_deg=self._ema_angle(track.angle_deg, limited),
                    velocity_deg_per_s=0.0,
                    confidence=float((1.0 - self._ema_alpha) * track.confidence + self._ema_alpha * float(score)),
                    age_frames=track.age_frames + 1,
                    missed_frames=0,
                    active=True,
                    last_seen_frame=self._frame_idx,
                )
            )

        held_tracks = 0
        for idx, track in enumerate(self._tracks):
            if idx in used_tracks:
                continue
            age = self._frame_idx - track.last_seen_frame
            if age > self._hold_frames:
                continue
            held_tracks += 1
            next_tracks.append(
                _TrackedPeak(
                    track_id=track.track_id,
                    angle_deg=track.angle_deg,
                    velocity_deg_per_s=track.velocity_deg_per_s,
                    confidence=float(track.confidence * self._score_decay),
                    age_frames=track.age_frames + 1,
                    missed_frames=track.missed_frames + 1,
                    active=False,
                    last_seen_frame=track.last_seen_frame,
                )
            )

        for raw_idx, (angle_deg, score) in enumerate(zip(peaks, raw_scores)):
            if raw_idx in matched_raw or float(score) < self._min_score:
                continue
            next_tracks.append(
                _TrackedPeak(
                    track_id=self._next_track_id,
                    angle_deg=float(angle_deg % 360.0),
                    velocity_deg_per_s=0.0,
                    confidence=float(score),
                    age_frames=1,
                    missed_frames=0,
                    active=True,
                    last_seen_frame=self._frame_idx,
                )
            )
            self._next_track_id += 1

        next_tracks = [track for track in next_tracks if track.confidence >= self._min_score]
        next_tracks.sort(key=lambda item: (-item.confidence, item.angle_deg))
        self._tracks = next_tracks[: self._max_sources]
        self._frame_idx += 1
        tracked_peaks = [float(track.angle_deg) for track in self._tracks]
        tracked_scores = [float(track.confidence) for track in self._tracks] if self._tracks else None
        return tracked_peaks, tracked_scores, {
            "backend": self._backend_name,
            "raw_peaks_deg": list(peaks),
            "raw_peak_scores": list(raw_scores),
            "tracked_peaks_deg": tracked_peaks,
            "tracked_peak_scores": [] if tracked_scores is None else list(tracked_scores),
            "held_tracks": int(held_tracks),
            **extra_debug,
        }

    def _update_multi_peak_tracks(self, candidates: list[PeakCandidate], extra_debug: dict) -> tuple[list[float], list[float] | None, dict]:
        raw_peaks = [float(c.angle_deg) for c in candidates]
        raw_scores = [float(c.score) for c in candidates]
        if not self._tracks:
            spawned = []
            for cand in candidates[: self._max_tracks]:
                if cand.score < self._new_track_min_confidence:
                    continue
                self._tracks.append(
                    _TrackedPeak(
                        track_id=self._next_track_id,
                        angle_deg=float(cand.angle_deg),
                        velocity_deg_per_s=0.0,
                        confidence=float(cand.score),
                        age_frames=1,
                        missed_frames=0,
                        active=True,
                        last_seen_frame=self._frame_idx,
                    )
                )
                spawned.append(int(self._next_track_id))
                self._next_track_id += 1
            self._frame_idx += 1
            self._tracks.sort(key=lambda item: (-item.confidence, item.track_id))
            tracked_peaks = [float(track.angle_deg) for track in self._tracks]
            tracked_scores = [float(track.confidence) for track in self._tracks]
            return tracked_peaks, tracked_scores, {
                "backend": self._backend_name,
                "raw_peaks_deg": raw_peaks,
                "raw_peak_scores": raw_scores,
                "tracked_peaks_deg": tracked_peaks,
                "tracked_peak_scores": tracked_scores,
                "held_tracks": 0,
                "track_assignments": [],
                "spawned_tracks": spawned,
                "retired_tracks": [],
                **extra_debug,
            }

        assignments: list[tuple[int, int]] = []
        used_track_ids: set[int] = set()
        used_cand_ids: set[int] = set()
        if self._tracks and candidates:
            cost = np.full((len(self._tracks), len(candidates)), 1e6, dtype=np.float64)
            for t_idx, track in enumerate(self._tracks):
                predicted = float((track.angle_deg + track.velocity_deg_per_s * self._update_interval_s) % 360.0)
                for c_idx, cand in enumerate(candidates):
                    dist = _angular_dist_deg(predicted, cand.angle_deg)
                    if dist > self._max_assoc_distance_deg:
                        continue
                    cost[t_idx, c_idx] = dist + (5.0 * max(0.0, 0.5 - cand.score))
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] >= 1e5:
                    continue
                assignments.append((int(r), int(c)))
                used_track_ids.add(int(r))
                used_cand_ids.add(int(c))

        next_tracks: list[_TrackedPeak] = []
        held_tracks = 0
        spawned_tracks: list[int] = []
        retired_tracks: list[int] = []
        assignment_debug: list[dict] = []

        for t_idx, c_idx in assignments:
            track = self._tracks[t_idx]
            cand = candidates[c_idx]
            delta = _angular_diff_deg(cand.angle_deg, track.angle_deg)
            velocity = (delta / self._update_interval_s)
            velocity = (1.0 - self._velocity_alpha) * track.velocity_deg_per_s + self._velocity_alpha * velocity
            predicted = float((track.angle_deg + velocity * self._update_interval_s) % 360.0)
            limited = self._step_limit(predicted, cand.angle_deg)
            angle = self._ema_angle(track.angle_deg, (1.0 - self._angle_alpha) * predicted + self._angle_alpha * limited)
            confidence = float(np.clip(max(cand.score, (1.0 - self._ema_alpha) * track.confidence + self._ema_alpha * cand.score), 0.0, 1.0))
            next_tracks.append(
                _TrackedPeak(
                    track_id=track.track_id,
                    angle_deg=float(angle),
                    velocity_deg_per_s=float(velocity),
                    confidence=confidence,
                    age_frames=track.age_frames + 1,
                    missed_frames=0,
                    active=True,
                    last_seen_frame=self._frame_idx,
                )
            )
            assignment_debug.append({"track_id": int(track.track_id), "candidate_angle_deg": float(cand.angle_deg)})

        for t_idx, track in enumerate(self._tracks):
            if t_idx in used_track_ids:
                continue
            missed = int(track.missed_frames + 1)
            if missed > self._track_kill_frames:
                retired_tracks.append(int(track.track_id))
                continue
            held_tracks += 1
            next_tracks.append(
                _TrackedPeak(
                    track_id=track.track_id,
                    angle_deg=float((track.angle_deg + track.velocity_deg_per_s * self._update_interval_s) % 360.0),
                    velocity_deg_per_s=float(track.velocity_deg_per_s * 0.7),
                    confidence=float(track.confidence * self._track_confidence_decay),
                    age_frames=track.age_frames + 1,
                    missed_frames=missed,
                    active=missed <= self._track_hold_frames,
                    last_seen_frame=track.last_seen_frame,
                )
            )

        for c_idx, cand in enumerate(candidates):
            if c_idx in used_cand_ids or cand.score < self._new_track_min_confidence:
                continue
            if any(_angular_dist_deg(cand.angle_deg, tr.angle_deg) < self._min_peak_separation_deg for tr in next_tracks):
                continue
            next_tracks.append(
                _TrackedPeak(
                    track_id=self._next_track_id,
                    angle_deg=float(cand.angle_deg),
                    velocity_deg_per_s=0.0,
                    confidence=float(cand.score),
                    age_frames=1,
                    missed_frames=0,
                    active=True,
                    last_seen_frame=self._frame_idx,
                )
            )
            spawned_tracks.append(int(self._next_track_id))
            self._next_track_id += 1

        next_tracks = [track for track in next_tracks if track.confidence >= self._min_score]
        next_tracks.sort(key=lambda item: (-float(item.active), -item.confidence, item.track_id))
        self._tracks = next_tracks[: self._max_tracks]
        self._frame_idx += 1
        tracked_peaks = [float(track.angle_deg) for track in self._tracks]
        tracked_scores = [float(track.confidence) for track in self._tracks] if self._tracks else None
        return tracked_peaks, tracked_scores, {
            "backend": self._backend_name,
            "raw_peaks_deg": raw_peaks,
            "raw_peak_scores": raw_scores,
            "tracked_peaks_deg": tracked_peaks,
            "tracked_peak_scores": tracked_scores or [],
            "held_tracks": int(held_tracks),
            "track_assignments": assignment_debug,
            "spawned_tracks": spawned_tracks,
            "retired_tracks": retired_tracks,
            **extra_debug,
        }

    def update(self, frame_mc: np.ndarray) -> tuple[list[float], list[float] | None, dict]:
        frame = np.asarray(frame_mc, dtype=float)
        if frame.ndim != 2:
            raise ValueError("frame_mc must be shape (samples, n_mics)")

        frame_t = frame.T
        self._frames.append(frame_t)
        self._total += frame_t.shape[1]

        while self._frames and self._total > self._window_samples:
            left = self._frames[0]
            extra = self._total - self._window_samples
            if left.shape[1] <= extra:
                self._frames.popleft()
                self._total -= left.shape[1]
            else:
                self._frames[0] = left[:, extra:]
                self._total -= extra

        min_needed = max(32, getattr(self._backend, "nfft", 32) if hasattr(self._backend, "nfft") else 32)
        if self._total < min_needed:
            return [], None, {
                "backend": self._backend_name,
                "raw_peaks_deg": [],
                "tracked_peaks_deg": [],
                "held_tracks": 0,
                "track_assignments": [],
                "spawned_tracks": [],
                "retired_tracks": [],
                "tracking_mode": self._tracking_mode,
            }

        audio = np.concatenate(list(self._frames), axis=1)
        backend_result = self._backend.process(audio)
        legacy_peaks = normalize_doa_list(list(backend_result.peaks_deg), max_targets=len(backend_result.peaks_deg) if backend_result.peaks_deg else None)
        legacy_scores = list(backend_result.peak_scores)
        candidates, extractor_debug = self._extract_candidates(backend_result.score_spectrum)
        if not candidates and legacy_peaks:
            candidates = [PeakCandidate(angle_deg=float(a), score=float(s if i < len(legacy_scores) else 1.0), raw_score=float(s if i < len(legacy_scores) else 1.0)) for i, (a, s) in enumerate(zip(legacy_peaks, legacy_scores or [1.0] * len(legacy_peaks)))]
            extractor_debug["selected_peaks_deg"] = [float(c.angle_deg) for c in candidates]
        extra_debug = {**dict(backend_result.debug), **extractor_debug, "tracking_mode": self._tracking_mode}
        if self._backend_name in self._single_source_backends:
            if not legacy_peaks and backend_result.score_spectrum is not None:
                spec = np.asarray(backend_result.score_spectrum, dtype=np.float64)
                idx = _dominant_peak_idx(spec)
                if idx is not None:
                    ang = float(_grid_angles_deg(spec.size)[idx])
                    legacy_peaks = [ang]
                    legacy_scores = [float(_normalize_spectrum(spec)[idx])]
            return self._single_source_predict_update(legacy_peaks[:1], legacy_scores[:1] if legacy_scores else None, extra_debug)
        if self._tracking_mode != "multi_peak_v2":
            return self._update_legacy_tracks(legacy_peaks, legacy_scores, extra_debug)
        return self._update_multi_peak_tracks(candidates, extra_debug)
