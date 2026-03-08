from __future__ import annotations

import numpy as np

from realtime_pipeline.localization_backends import LocalizationBackendResult
from realtime_pipeline.srp_tracker import SRPPeakTracker
from simulation.mic_array_profiles import mic_positions_xyz


class _StubBackend:
    def __init__(self, spectra: list[np.ndarray]):
        self._spectra = [np.asarray(v, dtype=np.float64) for v in spectra]
        self.nfft = 256
        self._idx = 0

    def process(self, _audio: np.ndarray) -> LocalizationBackendResult:
        spec = self._spectra[min(self._idx, len(self._spectra) - 1)]
        self._idx += 1
        return LocalizationBackendResult(peaks_deg=[], peak_scores=[], score_spectrum=spec, debug={"backend": "stub"})


def _tracker(**overrides: float) -> SRPPeakTracker:
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    tracker = SRPPeakTracker(
        mic_pos=mic_pos,
        fs=16000,
        window_ms=20,
        nfft=64,
        overlap=0.5,
        freq_range=(300, 2500),
        max_sources=3,
        backend="tiny_dp_ipd",
        tracking_mode="multi_peak_v2",
        min_peak_separation_deg=18.0,
        **overrides,
    )
    return tracker


def test_multipeak_extractor_keeps_separated_strong_peaks() -> None:
    tracker = _tracker(min_peak_contrast=0.0, min_relative_peak_score=0.25)
    spec = np.zeros(72, dtype=np.float64)
    spec[8] = 1.0
    spec[28] = 0.82
    spec[30] = 0.3
    spec[50] = 0.74
    candidates, debug = tracker._extract_candidates(spec)
    assert [round(c.angle_deg) for c in candidates] == [40, 140, 250]
    assert 150.0 in [round(v) for v in debug["suppressed_peaks_deg"]]


def test_multipeak_tracker_maintains_two_stable_tracks() -> None:
    tracker = _tracker()
    spectra = []
    for a, b in [(8, 28), (8, 27), (9, 27), (9, 28)]:
        spec = np.zeros(72, dtype=np.float64)
        spec[a] = 1.0
        spec[b] = 0.88
        spectra.append(spec)
    tracker._backend = _StubBackend(spectra)
    frame = np.zeros((320, 4), dtype=np.float32)
    peak_history = []
    for _ in spectra:
        peaks, _scores, debug = tracker.update(frame)
        peak_history.append(peaks)
        assert len(debug["track_assignments"]) <= 2
    assert len(peak_history[-1]) == 2
    assert abs(peak_history[-1][0] - peak_history[-2][0]) < 15.0
    assert abs(peak_history[-1][1] - peak_history[-2][1]) < 15.0


def test_multipeak_tracker_holds_then_retires_tracks() -> None:
    tracker = _tracker()
    tracker._track_hold_frames = 1
    tracker._track_kill_frames = 2
    spectra = []
    spec = np.zeros(72, dtype=np.float64)
    spec[10] = 1.0
    spectra.append(spec)
    spectra.append(np.zeros(72, dtype=np.float64))
    spectra.append(np.zeros(72, dtype=np.float64))
    spectra.append(np.zeros(72, dtype=np.float64))
    tracker._backend = _StubBackend(spectra)
    frame = np.zeros((320, 4), dtype=np.float32)
    _ = tracker.update(frame)
    _peaks2, _scores2, debug2 = tracker.update(frame)
    assert debug2["held_tracks"] >= 1
    _ = tracker.update(frame)
    _peaks4, _scores4, debug4 = tracker.update(frame)
    assert debug4["retired_tracks"]
