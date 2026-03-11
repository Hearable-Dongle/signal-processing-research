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
SPECTRUM_EMA_ALPHA = 0.42
NOISE_PERCENTILE = 20.0
SNR_FLOOR_DB = -12.0
SNR_CEIL_DB = 25.0


def _normalize_snr_db(snr_db: float) -> float:
    clipped = float(np.clip(snr_db, SNR_FLOOR_DB, SNR_CEIL_DB))
    return float((clipped - SNR_FLOOR_DB) / max(1e-6, SNR_CEIL_DB - SNR_FLOOR_DB))


class SNRWeightedSRPPHATBackend(LocalizationBackendBase):
    def __init__(self, **kwargs):
        kwargs["freq_range"] = tuple(FREQ_RANGE_HZ)
        super().__init__(**kwargs)
        self._ema_spectrum: np.ndarray | None = None

    def _pair_snr_weight(self, pair_mag: np.ndarray) -> tuple[float, float]:
        frame_energy = np.mean(np.square(pair_mag), axis=0)
        if frame_energy.size == 0:
            return 0.0, SNR_FLOOR_DB
        noise_floor = float(np.percentile(frame_energy, NOISE_PERCENTILE))
        signal_level = float(np.mean(frame_energy))
        snr_db = 10.0 * np.log10(max(signal_level, 1e-9) / max(noise_floor, 1e-9))
        return _normalize_snr_db(snr_db), float(snr_db)

    def _smooth_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        spec = np.asarray(spectrum, dtype=np.float64)
        if self._ema_spectrum is None or self._ema_spectrum.shape != spec.shape:
            self._ema_spectrum = spec.copy()
        else:
            self._ema_spectrum = ((1.0 - SPECTRUM_EMA_ALPHA) * self._ema_spectrum) + (SPECTRUM_EMA_ALPHA * spec)
        return np.asarray(self._ema_spectrum, dtype=np.float64)

    def process(self, audio: np.ndarray) -> LocalizationBackendResult:
        roi = _stft_roi(audio, self.fs, self.nfft, self.overlap, self.freq_range)
        if roi is None:
            return LocalizationBackendResult(
                peaks_deg=[],
                peak_scores=[],
                score_spectrum=None,
                debug={"backend": "snr_weighted_srp_phat", "reason": "empty_roi"},
            )
        freqs, zxx_roi = roi
        active_mask = _speech_frame_mask(zxx_roi)
        if active_mask.size == 0 or not np.any(active_mask):
            active_mask = np.ones(zxx_roi.shape[2], dtype=bool)
        z_active = zxx_roi[:, :, active_mask]
        pairs, delays = _pair_delays(self.mic_pos, freqs, self.grid_size, self.sound_speed_m_s)
        spectrum = np.zeros(self.grid_size, dtype=np.float64)
        band_weight = _band_weight(freqs, self.small_aperture_bias)
        pair_snr_db: list[float] = []
        pair_weights: list[float] = []

        for p_idx, (i, j) in enumerate(pairs):
            prod = z_active[i] * np.conj(z_active[j])
            denom = np.abs(prod)
            if denom.size == 0:
                pair_snr_db.append(float(SNR_FLOOR_DB))
                pair_weights.append(0.0)
                continue
            norm_prod = prod / np.maximum(denom, 1e-10)
            pair_weight, snr_db = self._pair_snr_weight(np.abs(z_active[i]) + np.abs(z_active[j]))
            pair_snr_db.append(float(snr_db))
            pair_weights.append(float(pair_weight))
            mean_vec = np.mean(norm_prod, axis=1)
            phase = np.angle(mean_vec)[:, None]
            coherence = np.clip(np.abs(mean_vec), 0.0, 1.0)
            score_pf = np.cos(phase - delays[p_idx]) * (band_weight * coherence * pair_weight)[:, None]
            spectrum += np.sum(np.maximum(score_pf, 0.0), axis=0)

        raw_spectrum = np.asarray(spectrum, dtype=np.float64)
        smoothed_spectrum = self._smooth_spectrum(raw_spectrum)
        peaks_deg, peak_scores = self._peaks_from_spectrum(smoothed_spectrum)
        return LocalizationBackendResult(
            peaks_deg=peaks_deg,
            peak_scores=peak_scores,
            score_spectrum=smoothed_spectrum,
            debug={
                "backend": "snr_weighted_srp_phat",
                "pair_snr_db": [float(v) for v in pair_snr_db],
                "pair_weights": [float(v) for v in pair_weights],
                "active_frames": int(np.sum(active_mask)),
                "raw_spectrum": [float(v) for v in raw_spectrum.tolist()],
                "smoothed_spectrum": [float(v) for v in smoothed_spectrum.tolist()],
            },
        )


def main() -> None:
    from realtime_pipeline.localization_strategies.cli import run_backend_cli

    run_backend_cli("snr_weighted_srp_phat")


if __name__ == "__main__":
    main()
