from __future__ import annotations

import numpy as np

from realtime_pipeline.localization_backends import LocalizationBackendBase, LocalizationBackendResult

from .snr_weighted_srp_phat import SNRWeightedSRPPHATBackend


NUM_PARTICLES = 256
PROCESS_NOISE_DEG = 4.0
OBSERVATION_CONCENTRATION = 10.0
RESAMPLE_FRACTION = 0.55


def _wrap_deg(angle_deg: np.ndarray | float) -> np.ndarray | float:
    return np.mod(angle_deg, 360.0)


def _angular_delta_deg(a: np.ndarray, b: float) -> np.ndarray:
    return ((a - float(b) + 180.0) % 360.0) - 180.0


class ParticleFilterTrackerBackend(LocalizationBackendBase):
    def __init__(self, **kwargs):
        kwargs["freq_range"] = tuple((300, 2500))
        super().__init__(**kwargs)
        self._proposal = SNRWeightedSRPPHATBackend(**kwargs)
        self._rng = np.random.default_rng(0)
        self._particles = np.linspace(0.0, 360.0, NUM_PARTICLES, endpoint=False, dtype=np.float64)
        self._weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES, dtype=np.float64)

    def _estimate_spectrum(self, estimate_deg: float) -> np.ndarray:
        angles = np.linspace(0.0, 360.0, int(self.grid_size), endpoint=False, dtype=np.float64)
        delta = ((angles - float(estimate_deg) + 180.0) % 360.0) - 180.0
        spectrum = np.exp(-0.5 * np.square(delta / 12.0))
        return np.asarray(spectrum, dtype=np.float64)

    def _resample(self) -> None:
        cdf = np.cumsum(self._weights)
        step = 1.0 / NUM_PARTICLES
        start = float(self._rng.uniform(0.0, step))
        points = start + step * np.arange(NUM_PARTICLES, dtype=np.float64)
        idx = np.searchsorted(cdf, points, side="left")
        self._particles = np.asarray(self._particles[idx], dtype=np.float64)
        self._weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES, dtype=np.float64)

    def process(self, audio: np.ndarray) -> LocalizationBackendResult:
        proposal = self._proposal.process(audio)
        self._particles = _wrap_deg(self._particles + self._rng.normal(0.0, PROCESS_NOISE_DEG, size=self._particles.shape))
        debug = dict(proposal.debug)
        debug["backend"] = "particle_filter_tracker"
        debug["observation_available"] = bool(proposal.peaks_deg)
        if proposal.peaks_deg:
            obs = float(proposal.peaks_deg[0])
            delta = np.deg2rad(_angular_delta_deg(self._particles, obs))
            likelihood = np.exp(OBSERVATION_CONCENTRATION * np.cos(delta))
            self._weights *= likelihood
            self._weights /= np.sum(self._weights) + 1e-12
            ess = 1.0 / np.sum(np.square(self._weights))
            if ess < (RESAMPLE_FRACTION * NUM_PARTICLES):
                self._resample()
            map_idx = int(np.argmax(self._weights))
            estimate = float(self._particles[map_idx] % 360.0)
            peak_score = float(np.max(self._weights))
        else:
            estimate = float(np.angle(np.mean(np.exp(1j * np.deg2rad(self._particles)))) * 180.0 / np.pi % 360.0)
            peak_score = float(np.max(self._weights))
        debug["particle_mean_deg"] = float(estimate)
        debug["particle_weight_max"] = float(peak_score)
        debug["proposal_peak_deg"] = None if not proposal.peaks_deg else float(proposal.peaks_deg[0])
        score_spectrum = self._estimate_spectrum(estimate)
        return LocalizationBackendResult(
            peaks_deg=[estimate],
            peak_scores=[peak_score],
            score_spectrum=score_spectrum,
            debug=debug,
        )


def main() -> None:
    from realtime_pipeline.localization_strategies.cli import run_backend_cli

    run_backend_cli("particle_filter_tracker")


if __name__ == "__main__":
    main()
