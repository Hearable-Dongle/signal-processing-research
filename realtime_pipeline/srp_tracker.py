from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from beamforming.localization_bridge import normalize_doa_list

from .localization_backends import build_localization_backend


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
    ):
        self._fs = int(fs)
        self._window_samples = max(1, int(window_ms * fs / 1000))
        self._hop_samples = self._window_samples
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
        self._tracks: list[_TrackedPeak] = []
        nfft_eff = max(32, min(int(nfft), self._window_samples))
        overlap_eff = float(overlap)
        if int(nfft_eff * overlap_eff) >= nfft_eff:
            overlap_eff = float((nfft_eff - 1) / nfft_eff)
        self._backend_name = str(backend)
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

    @staticmethod
    def _circular_diff_deg(target_deg: float, source_deg: float) -> float:
        return float((float(target_deg) - float(source_deg) + 180.0) % 360.0 - 180.0)

    def _match_track(self, angle_deg: float, used: set[int]) -> int | None:
        best_idx = None
        best_dist = None
        for idx, track in enumerate(self._tracks):
            if idx in used:
                continue
            dist = abs(self._circular_diff_deg(angle_deg, track.angle_deg))
            if dist > self._match_tolerance_deg:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _step_limit(self, prev_deg: float, next_deg: float) -> float:
        delta = self._circular_diff_deg(next_deg, prev_deg)
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

    def update(self, frame_mc: np.ndarray) -> tuple[list[float], list[float] | None, dict]:
        frame = np.asarray(frame_mc, dtype=float)
        if frame.ndim != 2:
            raise ValueError("frame_mc must be shape (samples, n_mics)")

        frame_t = frame.T  # (n_mics, samples)
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
            return [], None, {"backend": self._backend_name, "raw_peaks_deg": [], "tracked_peaks_deg": [], "held_tracks": 0}

        audio = np.concatenate(list(self._frames), axis=1)
        backend_result = self._backend.process(audio)
        peaks = normalize_doa_list(list(backend_result.peaks_deg), max_targets=len(backend_result.peaks_deg) if backend_result.peaks_deg else None)
        scores = list(backend_result.peak_scores)
        if not self._prior_enabled:
            self._frame_idx += 1
            return peaks, scores, {
                "backend": self._backend_name,
                "raw_peaks_deg": list(peaks),
                "raw_peak_scores": list(scores),
                "tracked_peaks_deg": list(peaks),
                "tracked_peak_scores": list(scores),
                "held_tracks": 0,
                **dict(backend_result.debug),
            }

        raw_scores = list(scores) if scores else [1.0] * len(peaks)
        used_tracks: set[int] = set()
        next_tracks: list[_TrackedPeak] = []
        matched_raw: set[int] = set()

        for raw_idx, (angle_deg, score) in enumerate(zip(peaks, raw_scores)):
            if float(score) < self._min_score:
                continue
            track_idx = self._match_track(angle_deg, used_tracks)
            if track_idx is None:
                continue
            track = self._tracks[track_idx]
            used_tracks.add(track_idx)
            matched_raw.add(raw_idx)
            angle_next = float(angle_deg)
            if float(score) + self._hysteresis_margin < track.score:
                angle_next = track.angle_deg
            limited = self._step_limit(track.angle_deg, angle_next)
            next_tracks.append(
                _TrackedPeak(
                    angle_deg=self._ema_angle(track.angle_deg, limited),
                    score=float((1.0 - self._ema_alpha) * track.score + self._ema_alpha * float(score)),
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
                    angle_deg=track.angle_deg,
                    score=float(track.score * self._score_decay),
                    last_seen_frame=track.last_seen_frame,
                )
            )

        for raw_idx, (angle_deg, score) in enumerate(zip(peaks, raw_scores)):
            if raw_idx in matched_raw or float(score) < self._min_score:
                continue
            next_tracks.append(
                _TrackedPeak(
                    angle_deg=float(angle_deg % 360.0),
                    score=float(score),
                    last_seen_frame=self._frame_idx,
                )
            )

        next_tracks = [track for track in next_tracks if track.score >= self._min_score]
        next_tracks.sort(key=lambda item: (-item.score, item.angle_deg))
        self._tracks = next_tracks[: self._max_sources]
        self._frame_idx += 1
        tracked_peaks = [float(track.angle_deg) for track in self._tracks]
        tracked_scores = [float(track.score) for track in self._tracks] if self._tracks else None

        return tracked_peaks, tracked_scores, {
            "backend": self._backend_name,
            "raw_peaks_deg": list(peaks),
            "raw_peak_scores": list(raw_scores),
            "tracked_peaks_deg": tracked_peaks,
            "tracked_peak_scores": [] if tracked_scores is None else list(tracked_scores),
            "held_tracks": int(held_tracks),
            **dict(backend_result.debug),
        }

@dataclass
class _TrackedPeak:
    angle_deg: float
    score: float
    last_seen_frame: int
