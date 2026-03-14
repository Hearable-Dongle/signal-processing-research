from __future__ import annotations

import numpy as np
import pytest

from realtime_pipeline.localization_backends import SUPPORTED_LOCALIZATION_BACKENDS, build_localization_backend
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
    out += 0.01 * np.random.default_rng(42).standard_normal(out.shape)
    return out.astype(np.float32)


def _angular_error_deg(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))


def test_supported_localization_backends_emit_bridge_results() -> None:
    sr = 16000
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    audio = _simulate_far_field_signal(35.0, sr=sr, duration_s=0.32, mic_pos_xyz=mic_pos)

    for backend_name in SUPPORTED_LOCALIZATION_BACKENDS:
        backend = build_localization_backend(
            backend_name,
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
        result = backend.process(audio)
        assert result.debug["backend"] == backend_name
        assert result.debug["localization_source"] == "localization.algo"
        assert result.score_spectrum is None or len(np.asarray(result.score_spectrum).reshape(-1)) > 0
        assert result.peaks_deg
        assert all(np.isfinite(float(v)) for v in result.peaks_deg)
        assert result.score_spectrum is None or len(np.asarray(result.score_spectrum).reshape(-1)) > 0


def test_capon_1src_smoke_localizes_single_source() -> None:
    sr = 16000
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    audio = _simulate_far_field_signal(35.0, sr=sr, duration_s=0.32, mic_pos_xyz=mic_pos)
    backend = build_localization_backend(
        "capon_1src",
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
    result = backend.process(audio)
    assert result.debug["backend"] == "capon_1src"
    assert result.peaks_deg
    assert _angular_error_deg(result.peaks_deg[0], 35.0) <= 30.0


def test_srp_phat_localization_smoke_localizes_single_source_without_180_flip() -> None:
    sr = 16000
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    audio = _simulate_far_field_signal(35.0, sr=sr, duration_s=0.32, mic_pos_xyz=mic_pos)
    backend = build_localization_backend(
        "srp_phat_localization",
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
    result = backend.process(audio)
    assert result.debug["backend"] == "srp_phat_localization"
    assert result.peaks_deg
    assert _angular_error_deg(result.peaks_deg[0], 35.0) <= 30.0


@pytest.mark.parametrize(
    "backend_name",
    [
        "weighted_srp_dp",
        "tiny_dp_ipd",
        "gcc_tdoa_1src",
        "snr_weighted_srp_phat",
        "peak_confidence_srp_phat",
        "particle_filter_tracker",
        "neural_mask_gcc_phat",
        "ipd_regressor",
    ],
)
def test_removed_realtime_backends_fail_with_clear_error(backend_name: str) -> None:
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    with pytest.raises(ValueError, match="delegates localization to localization/"):
        build_localization_backend(
            backend_name,
            mic_pos=mic_pos,
            fs=16000,
            nfft=256,
            overlap=0.5,
            freq_range=(300, 2500),
            max_sources=1,
            grid_size=72,
            min_separation_deg=15.0,
            small_aperture_bias=True,
        )
