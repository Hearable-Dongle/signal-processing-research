from __future__ import annotations

import numpy as np

from realtime_pipeline.localization_backends import (
    LocalizationBackendBase,
    LocalizationBackendResult,
    _band_weight,
    _pair_delays,
    _speech_frame_mask,
    _stft_roi,
)


FREQ_RANGE_HZ = (300, 2500)
MASK_PERCENTILE = 60.0
MASK_EMA_ALPHA = 0.40


class NeuralMaskGCCPHATBackend(LocalizationBackendBase):
    def __init__(self, **kwargs):
        kwargs["freq_range"] = tuple(FREQ_RANGE_HZ)
        super().__init__(**kwargs)
        self._ema_spectrum: np.ndarray | None = None

    def _speech_mask(self, z_active: np.ndarray) -> np.ndarray:
        energy = np.mean(np.abs(z_active), axis=0)
        coherence = np.mean(np.abs(z_active / np.maximum(np.abs(z_active), 1e-10)), axis=0)
        score = energy * (0.5 + 0.5 * np.clip(coherence, 0.0, 1.0))
        threshold = float(np.percentile(score, MASK_PERCENTILE))
        return score >= threshold

    def _smooth(self, spectrum: np.ndarray) -> np.ndarray:
        spec = np.asarray(spectrum, dtype=np.float64)
        if self._ema_spectrum is None or self._ema_spectrum.shape != spec.shape:
            self._ema_spectrum = spec.copy()
        else:
            self._ema_spectrum = ((1.0 - MASK_EMA_ALPHA) * self._ema_spectrum) + (MASK_EMA_ALPHA * spec)
        return np.asarray(self._ema_spectrum, dtype=np.float64)

    def process(self, audio: np.ndarray) -> LocalizationBackendResult:
        roi = _stft_roi(audio, self.fs, self.nfft, self.overlap, self.freq_range)
        if roi is None:
            return LocalizationBackendResult(
                peaks_deg=[],
                peak_scores=[],
                score_spectrum=None,
                debug={"backend": "neural_mask_gcc_phat", "reason": "empty_roi"},
            )
        freqs, zxx_roi = roi
        active_mask = _speech_frame_mask(zxx_roi)
        if active_mask.size == 0 or not np.any(active_mask):
            active_mask = np.ones(zxx_roi.shape[2], dtype=bool)
        z_active = zxx_roi[:, :, active_mask]
        speech_mask = self._speech_mask(z_active)
        masked = z_active * speech_mask[None, :, :]
        pairs, delays = _pair_delays(self.mic_pos, freqs, self.grid_size, self.sound_speed_m_s)
        band_weight = _band_weight(freqs, self.small_aperture_bias)
        spectrum = np.zeros(self.grid_size, dtype=np.float64)

        for p_idx, (i, j) in enumerate(pairs):
            prod = masked[i] * np.conj(masked[j])
            denom = np.abs(prod)
            if denom.size == 0:
                continue
            norm_prod = prod / np.maximum(denom, 1e-10)
            mean_vec = np.mean(norm_prod, axis=1)
            phase = np.angle(mean_vec)[:, None]
            coherence = np.clip(np.abs(mean_vec), 0.0, 1.0)
            score_pf = np.cos(phase - delays[p_idx]) * (band_weight * coherence)[:, None]
            spectrum += np.sum(np.maximum(score_pf, 0.0), axis=0)

        smoothed = self._smooth(spectrum)
        peaks_deg, peak_scores = self._peaks_from_spectrum(smoothed)
        return LocalizationBackendResult(
            peaks_deg=peaks_deg,
            peak_scores=peak_scores,
            score_spectrum=smoothed,
            debug={
                "backend": "neural_mask_gcc_phat",
                "speech_mask_ratio": float(np.mean(speech_mask)),
                "active_frames": int(np.sum(active_mask)),
            },
        )


def main() -> None:
    from realtime_pipeline.localization_strategies.cli import run_backend_cli

    run_backend_cli("neural_mask_gcc_phat")


if __name__ == "__main__":
    main()
