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


class _StubSingleSourceBackend:
    def __init__(self, peaks: list[float], scores: list[float]):
        self._peaks = peaks
        self._scores = scores
        self.nfft = 64
        self._idx = 0

    def process(self, _audio: np.ndarray) -> LocalizationBackendResult:
        idx = min(self._idx, len(self._peaks) - 1)
        self._idx += 1
        peak = float(self._peaks[idx])
        score = float(self._scores[idx])
        spec = np.zeros(72, dtype=np.float64)
        spec[int(round(peak / 5.0)) % 72] = score
        return LocalizationBackendResult(peaks_deg=[peak], peak_scores=[score], score_spectrum=spec, debug={"backend": "music_1src"})


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


def _dominant_lock_tracker(**overrides: float) -> SRPPeakTracker:
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    tracker = SRPPeakTracker(
        mic_pos=mic_pos,
        fs=16000,
        window_ms=20,
        nfft=64,
        overlap=0.5,
        freq_range=(300, 2500),
        max_sources=1,
        backend="tiny_dp_ipd",
        tracking_mode="dominant_lock_v1",
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


def test_single_source_filter_smooths_jittery_measurements() -> None:
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    tracker = SRPPeakTracker(
        mic_pos=mic_pos,
        fs=16000,
        window_ms=20,
        nfft=64,
        overlap=0.5,
        freq_range=(300, 2500),
        max_sources=1,
        backend="music_1src",
        tracking_mode="legacy",
        single_source_motion_filter_enabled=True,
    )
    tracker._backend = _StubSingleSourceBackend([40.0, 55.0, 35.0, 50.0], [0.9, 0.9, 0.9, 0.9])
    frame = np.zeros((320, 4), dtype=np.float32)
    raw = [40.0, 55.0, 35.0, 50.0]
    filtered = []
    for _ in raw:
        peaks, _scores, debug = tracker.update(frame)
        filtered.append(peaks[0])
        assert debug["single_source_filter_mode"] == "update"
    raw_span = max(raw) - min(raw)
    filt_span = max(filtered) - min(filtered)
    assert filt_span < raw_span


def test_dominant_lock_acquires_after_consistent_observations() -> None:
    tracker = _dominant_lock_tracker()
    spectra = []
    for idx in [8, 8, 8]:
        spec = np.zeros(72, dtype=np.float64)
        spec[idx] = 1.0
        spectra.append(spec)
    tracker._backend = _StubBackend(spectra)
    frame = np.zeros((320, 4), dtype=np.float32)
    peaks1, _scores1, debug1 = tracker.update(frame)
    peaks2, _scores2, debug2 = tracker.update(frame)
    assert peaks1 == []
    assert debug1["dominant_mode_decision"] == "acquire_pending"
    assert peaks2
    assert debug2["dominant_mode_decision"] == "acquire_lock"


def test_dominant_lock_holds_on_missing_then_unlocks() -> None:
    tracker = _dominant_lock_tracker(
        dominant_lock_hold_missing_frames=1,
        dominant_lock_unlock_after_missing_frames=2,
    )
    spectra = []
    locked = np.zeros(72, dtype=np.float64)
    locked[10] = 1.0
    spectra.extend([locked, locked, np.zeros(72, dtype=np.float64), np.zeros(72, dtype=np.float64), np.zeros(72, dtype=np.float64)])
    tracker._backend = _StubBackend(spectra)
    frame = np.zeros((320, 4), dtype=np.float32)
    tracker.update(frame)
    tracker.update(frame)
    peaks3, _scores3, debug3 = tracker.update(frame)
    peaks4, _scores4, debug4 = tracker.update(frame)
    peaks5, _scores5, debug5 = tracker.update(frame)
    assert peaks3
    assert debug3["dominant_mode_decision"] == "hold_missing"
    assert peaks4 == []
    assert debug4["dominant_mode_decision"] == "hold_missing"
    assert peaks5 == []
    assert debug5["dominant_mode_decision"] == "unlock_missing"


def test_dominant_lock_requires_repeated_challenger_before_switch() -> None:
    tracker = _dominant_lock_tracker(
        dominant_lock_switch_confirm_frames=3,
        dominant_lock_challenger_margin=0.0,
    )
    spectra = []
    for bins in [(8,), (8,), (30, 8), (30, 8), (30, 8)]:
        spec = np.zeros(72, dtype=np.float64)
        if len(bins) == 1:
            spec[bins[0]] = 1.0
        else:
            spec[bins[0]] = 1.0
            spec[bins[1]] = 0.6
        spectra.append(spec)
    tracker._backend = _StubBackend(spectra)
    frame = np.zeros((320, 4), dtype=np.float32)
    tracker.update(frame)
    tracker.update(frame)
    peaks3, _scores3, debug3 = tracker.update(frame)
    peaks4, _scores4, debug4 = tracker.update(frame)
    peaks5, _scores5, debug5 = tracker.update(frame)
    assert round(peaks3[0]) == 40
    assert debug3["dominant_mode_decision"] == "challenger_pending"
    assert round(peaks4[0]) == 40
    assert debug4["dominant_mode_decision"] == "challenger_pending"
    assert round(peaks5[0]) == 150
    assert debug5["dominant_mode_decision"] == "switch_lock"


def test_dominant_lock_handles_wraparound_challenger_consistently() -> None:
    tracker = _dominant_lock_tracker(
        dominant_lock_switch_confirm_frames=2,
        dominant_lock_challenger_margin=0.01,
    )
    spectra = []
    for idx in [71, 71, 1, 1]:
        spec = np.zeros(72, dtype=np.float64)
        spec[idx] = 1.0
        spectra.append(spec)
    tracker._backend = _StubBackend(spectra)
    frame = np.zeros((320, 4), dtype=np.float32)
    tracker.update(frame)
    tracker.update(frame)
    peaks3, _scores3, debug3 = tracker.update(frame)
    peaks4, _scores4, debug4 = tracker.update(frame)
    assert peaks3
    assert debug3["dominant_mode_decision"] == "stay_update"
    assert peaks4
    assert debug4["dominant_mode_decision"] == "stay_update"
