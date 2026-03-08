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

    for backend_name in ["srp_phat_legacy", "weighted_srp_dp", "tiny_dp_ipd"]:
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
        assert min(err, flipped_err) <= 25.0
