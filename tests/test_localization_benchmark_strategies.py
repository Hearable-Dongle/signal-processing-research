from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realtime_pipeline.localization_backends import SUPPORTED_LOCALIZATION_BACKENDS, build_localization_backend
from realtime_pipeline.localization_strategies.ipd_regressor import extract_ipd_features
from simulation.mic_array_profiles import mic_positions_xyz


def _simulate_far_field_signal(angle_deg: float, sr: int, duration_s: float, mic_pos_xyz: np.ndarray) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr
    source = (
        0.6 * np.sin(2.0 * np.pi * 650.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 1100.0 * t + 0.2)
        + 0.2 * np.sin(2.0 * np.pi * 1800.0 * t + 0.8)
    )
    source *= 0.5 + 0.5 * np.sin(2.0 * np.pi * 2.5 * t)
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
    out += 0.005 * np.random.default_rng(42).standard_normal(out.shape)
    return out.astype(np.float32)


def test_supported_localization_backends_emit_compatible_results() -> None:
    sr = 16000
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    audio = _simulate_far_field_signal(30.0, sr=sr, duration_s=0.32, mic_pos_xyz=mic_pos)
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
        assert result.score_spectrum is None or len(np.asarray(result.score_spectrum).reshape(-1)) > 0


def test_removed_backend_names_are_rejected() -> None:
    sr = 16000
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    with np.testing.assert_raises_regex(ValueError, "delegates localization to localization/"):
        build_localization_backend(
            "peak_confidence_srp_phat",
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


def test_ipd_feature_extractor_has_expected_shape() -> None:
    sr = 16000
    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    audio = _simulate_far_field_signal(75.0, sr=sr, duration_s=0.2, mic_pos_xyz=mic_pos)
    feat = extract_ipd_features(audio, fs=sr, nfft=256, overlap=0.5)
    assert feat.ndim == 1
    assert feat.size > 0
    assert np.all(np.isfinite(feat))
