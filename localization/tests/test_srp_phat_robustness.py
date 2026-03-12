from __future__ import annotations

import numpy as np

from localization.algo import SRPPHATLocalization
from simulation.mic_array_profiles import mic_positions_xyz


def _fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    idx = np.arange(x.shape[0], dtype=np.float64)
    src_idx = idx - float(delay_samples)
    return np.interp(src_idx, idx, x.astype(np.float64), left=0.0, right=0.0)


def _plane_wave(mic_pos: np.ndarray, signal: np.ndarray, doa_deg: float, fs: int, sound_speed: float = 343.0) -> np.ndarray:
    positions = np.asarray(mic_pos, dtype=np.float64)
    if positions.shape[0] == 3:
        positions = positions.T
    if positions.shape[1] == 2:
        positions = np.hstack([positions, np.zeros((positions.shape[0], 1), dtype=np.float64)])
    direction = np.array(
        [np.cos(np.deg2rad(float(doa_deg))), np.sin(np.deg2rad(float(doa_deg))), 0.0],
        dtype=np.float64,
    )
    tau = (positions @ direction) / float(sound_speed)
    tau = tau - float(np.mean(tau))
    delays = tau * float(fs)
    return np.stack([_fractional_delay(signal, delay) for delay in delays], axis=0).astype(np.float32)


def _angular_error_deg(pred_deg: float, truth_deg: float) -> float:
    return float(abs((float(pred_deg) - float(truth_deg) + 180.0) % 360.0 - 180.0))


def test_snr_gate_rejects_bins_below_threshold() -> None:
    algo = SRPPHATLocalization(
        mic_pos=np.zeros((3, 4), dtype=np.float64),
        snr_gating_enabled=True,
        snr_threshold_db=6.0,
        snr_soft_range_db=6.0,
    )
    weights, snr_db = algo._snr_weights(
        np.array([1.0, 4.0, 16.0], dtype=np.float64),
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )
    assert snr_db[0] <= 0.1
    assert weights[0] == 0.0
    assert 0.0 < weights[1] < 1.0
    assert weights[2] == 1.0


def test_msc_variance_weighting_prefers_transient_bins() -> None:
    algo = SRPPHATLocalization(
        mic_pos=np.zeros((3, 4), dtype=np.float64),
        msc_variance_enabled=True,
        msc_history_frames=4,
        msc_variance_floor=0.002,
    )
    history = [
        np.array([0.95, 0.20], dtype=np.float64),
        np.array([0.95, 0.85], dtype=np.float64),
        np.array([0.95, 0.15], dtype=np.float64),
    ]
    weights = algo._msc_temporal_weights(np.array([0.95, 0.90], dtype=np.float64), history=history)
    assert weights[1] > weights[0]


def test_robust_srp_phat_localizes_single_source_under_directional_noise() -> None:
    fs = 16000
    duration_s = 1.6
    samples = np.arange(int(fs * duration_s), dtype=np.float64) / float(fs)
    speech = (
        0.6 * np.sin(2.0 * np.pi * 1400.0 * samples)
        + 0.3 * np.sin(2.0 * np.pi * 2100.0 * samples)
        + 0.2 * np.sin(2.0 * np.pi * 3200.0 * samples)
    )
    speech *= np.hanning(speech.shape[0])

    rng = np.random.default_rng(7)
    noise = rng.standard_normal(speech.shape[0]).astype(np.float64)
    noise = np.convolve(noise, np.ones(33, dtype=np.float64) / 33.0, mode="same")

    mic_pos = mic_positions_xyz("respeaker_v3_0457").T
    target_deg = 60.0
    interferer_deg = 210.0
    speech_mc = _plane_wave(mic_pos, speech, target_deg, fs)
    noise_mc = _plane_wave(mic_pos, noise, interferer_deg, fs)
    mixture = speech_mc + 0.7 * noise_mc

    algo = SRPPHATLocalization(
        mic_pos=mic_pos,
        fs=fs,
        overlap=0.2,
        freq_range=(1200, 5400),
        max_sources=1,
        vad_enabled=True,
        snr_gating_enabled=True,
        snr_threshold_db=3.0,
        msc_variance_enabled=True,
        hsda_enabled=True,
    )
    pred_rad, _hist, history = algo.process(mixture)
    assert pred_rad
    pred_deg = float(np.degrees(pred_rad[0]) % 360.0)
    assert _angular_error_deg(pred_deg, target_deg) <= 12.0
    assert history
