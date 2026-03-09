from __future__ import annotations

import numpy as np

from realtime_pipeline.localization_backends import build_localization_backend
from simulation.mic_array_profiles import mic_positions_xyz


def _simulate_far_field_signal(angle_deg: float, sr: int, duration_s: float, mic_pos_xyz: np.ndarray) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr
    source = (
        0.6 * np.sin(2.0 * np.pi * 700.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 1200.0 * t + 0.4)
        + 0.2 * np.sin(2.0 * np.pi * 1900.0 * t + 1.0)
    )
    source *= (0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t + 0.2))
    source = source.astype(np.float64)
    source /= np.max(np.abs(source)) + 1e-12

    c = 343.0
    theta = np.deg2rad(float(angle_deg))
    direction = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
    tau = (mic_pos_xyz.T @ direction) / c
    tau = tau - np.mean(tau)
    out = np.zeros((mic_pos_xyz.shape[1], n), dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    for m in range(mic_pos_xyz.shape[1]):
        delayed_idx = idx - tau[m] * sr
        out[m] = np.interp(delayed_idx, idx, source, left=0.0, right=0.0)
    rng = np.random.default_rng(42)
    out += 0.01 * rng.standard_normal(out.shape)
    return out.astype(np.float32)


def _angular_error_deg(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))


def test_localization_backends_detect_synthetic_far_field_source() -> None:
    sr = 16000
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    audio = _simulate_far_field_signal(35.0, sr=sr, duration_s=0.32, mic_pos_xyz=mic_pos)

    for backend_name in ["srp_phat_legacy", "weighted_srp_dp", "tiny_dp_ipd", "music_1src", "gcc_tdoa_1src"]:
        backend = build_localization_backend(
            backend_name,
            mic_pos=mic_pos,
            fs=sr,
            nfft=256,
            overlap=0.5,
            freq_range=(300, 2500),
            max_sources=2,
            grid_size=72,
            min_separation_deg=15.0,
            small_aperture_bias=True,
        )
        result = backend.process(audio)
        assert result.debug["backend"] == backend_name
        assert result.peaks_deg
        if backend_name == "srp_phat_legacy":
            continue
        err = min(_angular_error_deg(35.0, peak) for peak in result.peaks_deg)
        flipped_err = min(_angular_error_deg(215.0, peak) for peak in result.peaks_deg)
        limit = 35.0 if backend_name == "gcc_tdoa_1src" else 25.0
        assert min(err, flipped_err) <= limit


def test_tiny_dp_ipd_preserves_two_source_structure() -> None:
    sr = 16000
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    audio_a = _simulate_far_field_signal(40.0, sr=sr, duration_s=0.32, mic_pos_xyz=mic_pos).astype(np.float64)
    audio_b = _simulate_far_field_signal(140.0, sr=sr, duration_s=0.32, mic_pos_xyz=mic_pos).astype(np.float64)
    audio = (0.95 * audio_a) + (0.65 * audio_b)
    audio /= np.max(np.abs(audio)) + 1e-12

    backend = build_localization_backend(
        "tiny_dp_ipd",
        mic_pos=mic_pos,
        fs=sr,
        nfft=256,
        overlap=0.5,
        freq_range=(300, 2500),
        max_sources=2,
        grid_size=72,
        min_separation_deg=15.0,
        small_aperture_bias=True,
    )
    result = backend.process(audio.astype(np.float32))

    assert len(result.debug["dp_spectrum"]) == 72
    assert len(result.debug["residual_spectrum"]) == 72
    assert len(result.debug["fused_spectrum"]) == 72
    residual = np.asarray(result.debug["residual_spectrum"], dtype=np.float64)
    top_bins = np.argsort(residual)[-2:]
    assert float(np.max(residual)) > 0.0
    assert abs(int(top_bins[-1]) - int(top_bins[-2])) >= 1


def test_tiny_dp_ipd_builds_temporal_prior_across_updates() -> None:
    sr = 16000
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    audio = _simulate_far_field_signal(55.0, sr=sr, duration_s=0.32, mic_pos_xyz=mic_pos)
    backend = build_localization_backend(
        "tiny_dp_ipd",
        mic_pos=mic_pos,
        fs=sr,
        nfft=256,
        overlap=0.5,
        freq_range=(300, 2500),
        max_sources=2,
        grid_size=72,
        min_separation_deg=15.0,
        small_aperture_bias=True,
    )

    first = backend.process(audio)
    second = backend.process(audio)

    prior1 = np.asarray(first.debug["temporal_prior_spectrum"], dtype=np.float64)
    prior2 = np.asarray(second.debug["temporal_prior_spectrum"], dtype=np.float64)
    assert np.max(prior1) == 0.0
    assert np.max(prior2) > 0.0
    assert len(second.debug["current_fused_spectrum"]) == 72
    assert len(second.debug["dominant_spectrum"]) == 72


def test_tiny_dp_ipd_reports_secondary_gate_diagnostics() -> None:
    sr = 16000
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    audio = _simulate_far_field_signal(35.0, sr=sr, duration_s=0.32, mic_pos_xyz=mic_pos)
    backend = build_localization_backend(
        "tiny_dp_ipd",
        mic_pos=mic_pos,
        fs=sr,
        nfft=256,
        overlap=0.5,
        freq_range=(300, 2500),
        max_sources=2,
        grid_size=72,
        min_separation_deg=15.0,
        small_aperture_bias=True,
    )
    result = backend.process(audio)

    assert "secondary_gate_passed" in result.debug
    assert "secondary_gate_reasons" in result.debug
    assert "promoted_secondary_gain" in result.debug
    assert isinstance(result.debug["secondary_gate_reasons"], list)


def test_tiny_dp_ipd_requires_consistent_secondary_before_promotion() -> None:
    sr = 16000
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    audio_a = _simulate_far_field_signal(40.0, sr=sr, duration_s=0.32, mic_pos_xyz=mic_pos).astype(np.float64)
    audio_b = _simulate_far_field_signal(140.0, sr=sr, duration_s=0.32, mic_pos_xyz=mic_pos).astype(np.float64)
    audio = (0.95 * audio_a) + (0.65 * audio_b)
    audio /= np.max(np.abs(audio)) + 1e-12

    backend = build_localization_backend(
        "tiny_dp_ipd",
        mic_pos=mic_pos,
        fs=sr,
        nfft=256,
        overlap=0.5,
        freq_range=(300, 2500),
        max_sources=2,
        grid_size=72,
        min_separation_deg=15.0,
        small_aperture_bias=True,
    )

    first = backend.process(audio.astype(np.float32))
    second = backend.process(audio.astype(np.float32))
    third = backend.process(audio.astype(np.float32))

    assert first.debug["secondary_gate_passed"] is False
    assert second.debug["secondary_gate_passed"] is False
    assert third.debug["secondary_gate_passed"] is True
    assert third.debug["secondary_consistency_hits"] >= 2


def test_gcc_tdoa_1src_returns_no_peak_when_pairs_are_unreliable() -> None:
    sr = 16000
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    backend = build_localization_backend(
        "gcc_tdoa_1src",
        mic_pos=mic_pos,
        fs=sr,
        nfft=256,
        overlap=0.5,
        freq_range=(300, 2500),
        max_sources=1,
        grid_size=72,
        min_separation_deg=15.0,
        small_aperture_bias=True,
    )

    audio = np.zeros((mic_pos.shape[1], int(0.08 * sr)), dtype=np.float32)
    result = backend.process(audio)

    assert result.debug["backend"] == "gcc_tdoa_1src"
    assert result.debug["reliable_pair_count"] == 0
    assert result.peaks_deg == []
