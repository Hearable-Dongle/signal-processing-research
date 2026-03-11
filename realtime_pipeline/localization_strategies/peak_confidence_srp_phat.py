from __future__ import annotations

import numpy as np

from realtime_pipeline.localization_backends import LocalizationBackendResult

from .snr_weighted_srp_phat import SNRWeightedSRPPHATBackend


PEAK_SIDELobe_EXCLUSION_DEG = 25.0
PEAK_TO_SIDELOBE_THRESHOLD = 1.18


class PeakConfidenceSRPPHATBackend(SNRWeightedSRPPHATBackend):
    def _peak_to_sidelobe_ratio(self, spectrum: np.ndarray) -> tuple[float, int | None]:
        arr = np.asarray(spectrum, dtype=np.float64).reshape(-1)
        if arr.size == 0 or float(np.max(arr)) <= 0.0:
            return 0.0, None
        peak_idx = int(np.argmax(arr))
        exclusion_bins = max(1, int(round(PEAK_SIDELobe_EXCLUSION_DEG / (360.0 / arr.size))))
        mask = np.ones(arr.size, dtype=bool)
        for offset in range(-exclusion_bins, exclusion_bins + 1):
            mask[(peak_idx + offset) % arr.size] = False
        sidelobe = float(np.max(arr[mask])) if np.any(mask) else 0.0
        ratio = float(arr[peak_idx] / max(1e-6, sidelobe))
        return ratio, peak_idx

    def process(self, audio: np.ndarray) -> LocalizationBackendResult:
        result = super().process(audio)
        if result.score_spectrum is None:
            return result
        ratio, peak_idx = self._peak_to_sidelobe_ratio(result.score_spectrum)
        debug = dict(result.debug)
        debug["backend"] = "peak_confidence_srp_phat"
        debug["peak_to_sidelobe_ratio"] = float(ratio)
        debug["gated"] = bool(ratio < PEAK_TO_SIDELOBE_THRESHOLD)
        if ratio < PEAK_TO_SIDELOBE_THRESHOLD or peak_idx is None:
            return LocalizationBackendResult(
                peaks_deg=[],
                peak_scores=[],
                score_spectrum=result.score_spectrum,
                debug=debug,
            )
        return LocalizationBackendResult(
            peaks_deg=result.peaks_deg,
            peak_scores=result.peak_scores,
            score_spectrum=result.score_spectrum,
            debug=debug,
        )


def main() -> None:
    from realtime_pipeline.localization_strategies.cli import run_backend_cli

    run_backend_cli("peak_confidence_srp_phat")


if __name__ == "__main__":
    main()
