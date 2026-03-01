from __future__ import annotations

from collections import deque

import numpy as np

from beamforming.localization_bridge import normalize_doa_list
from localization.algo import SRPPHATLocalization


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
    ):
        self._fs = int(fs)
        self._window_samples = max(1, int(window_ms * fs / 1000))
        self._frames: deque[np.ndarray] = deque()
        self._total = 0
        nfft_eff = max(32, min(int(nfft), self._window_samples))
        overlap_eff = float(overlap)
        if int(nfft_eff * overlap_eff) >= nfft_eff:
            overlap_eff = float((nfft_eff - 1) / nfft_eff)

        self._localizer = SRPPHATLocalization(
            mic_pos=np.asarray(mic_pos, dtype=float),
            fs=fs,
            nfft=nfft_eff,
            overlap=overlap_eff,
            freq_range=freq_range,
            max_sources=max_sources,
        )

    def update(self, frame_mc: np.ndarray) -> tuple[list[float], list[float] | None]:
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

        min_needed = max(2, int(getattr(self._localizer, "nfft", 2)))
        if self._total < min_needed:
            return [], None

        audio = np.concatenate(list(self._frames), axis=1)
        doas_rad, p_theta, _ = self._localizer.process(audio)
        doas_deg = [float(np.degrees(a) % 360.0) for a in doas_rad]
        peaks = normalize_doa_list(doas_deg, max_targets=len(doas_deg) if doas_deg else None)

        scores: list[float] | None = None
        if peaks and p_theta is not None and len(p_theta) > 0:
            bins = np.asarray(p_theta, dtype=float)
            n = bins.shape[0]
            scores = [float(bins[int(round((d % 360.0) / 360.0 * (n - 1)))]) for d in peaks]

        return peaks, scores
