from __future__ import annotations

import numpy as np

from .config import DirectionAssignmentConfig
from .geometry import normalize_angle_deg


def _pair_geometry(mic_i_xy: np.ndarray, mic_j_xy: np.ndarray) -> tuple[float, float]:
    d_vec = mic_j_xy - mic_i_xy
    baseline = float(np.linalg.norm(d_vec))
    phi_deg = normalize_angle_deg(np.rad2deg(np.arctan2(d_vec[1], d_vec[0])))
    return baseline, phi_deg


def gcc_phat_tdoa(
    sig_i: np.ndarray,
    sig_j: np.ndarray,
    sr: int,
    max_lag_samples: int,
) -> tuple[float, float]:
    """Returns (tdoa_seconds, peak_coherence_score)."""
    x = np.asarray(sig_i, dtype=float).reshape(-1)
    y = np.asarray(sig_j, dtype=float).reshape(-1)

    n = len(x) + len(y) - 1
    n_fft = 1 << int(np.ceil(np.log2(max(2, n))))

    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    cross = X * np.conj(Y)

    denom = np.abs(cross)
    denom[denom < 1e-10] = 1e-10
    phat = cross / denom

    corr = np.fft.irfft(phat, n=n_fft)
    corr = np.concatenate([corr[-(n_fft // 2):], corr[: (n_fft // 2)]])

    center = corr.shape[0] // 2
    lo = max(0, center - max_lag_samples)
    hi = min(corr.shape[0], center + max_lag_samples + 1)
    window = corr[lo:hi]

    if window.size == 0:
        return 0.0, 0.0

    peak_idx = int(np.argmax(np.abs(window)))
    lag_samples = (lo + peak_idx) - center

    peak_val = float(np.abs(window[peak_idx]))
    noise_floor = float(np.mean(np.abs(window))) + 1e-8
    coherence = float(np.clip(peak_val / noise_floor, 0.0, 10.0) / 10.0)

    return float(lag_samples / sr), coherence


def estimate_stream_doa(
    stream_multichannel: np.ndarray,
    mic_geometry: np.ndarray,
    mic_pairs: list[tuple[int, int]],
    cfg: DirectionAssignmentConfig,
) -> tuple[float | None, float, dict]:
    """Estimate per-stream azimuth (deg), confidence [0..1], and debug."""
    if stream_multichannel.ndim != 2:
        raise ValueError("stream_multichannel must have shape (n_mics, samples)")

    xy = np.asarray(mic_geometry, dtype=float)
    if xy.shape[1] > 2:
        xy = xy[:, :2]

    obs_taus: list[float] = []
    obs_phi_deg: list[float] = []
    obs_baseline: list[float] = []
    obs_w: list[float] = []
    rejected_pairs = 0

    for i, j in mic_pairs:
        baseline, phi_deg = _pair_geometry(xy[i], xy[j])
        if baseline <= 1e-6:
            rejected_pairs += 1
            continue

        max_lag = int(cfg.pair_max_lag_scale * (baseline / cfg.sound_speed_m_s) * cfg.sample_rate)
        max_lag = max(1, max_lag)

        tau_ij, coherence = gcc_phat_tdoa(
            stream_multichannel[i],
            stream_multichannel[j],
            sr=cfg.sample_rate,
            max_lag_samples=max_lag,
        )

        if coherence < cfg.min_pair_coherence:
            rejected_pairs += 1
            continue

        # Hard physical validity guard.
        if abs(tau_ij) > (baseline / cfg.sound_speed_m_s) * cfg.pair_max_lag_scale:
            rejected_pairs += 1
            continue

        obs_taus.append(float(tau_ij))
        obs_phi_deg.append(phi_deg)
        obs_baseline.append(baseline)
        obs_w.append(max(1e-6, coherence))

    if not obs_taus:
        return None, 0.0, {"valid_pairs": 0, "rejected_pairs": rejected_pairs}

    taus = np.asarray(obs_taus, dtype=float)
    phis = np.asarray(obs_phi_deg, dtype=float)
    baselines = np.asarray(obs_baseline, dtype=float)
    weights = np.asarray(obs_w, dtype=float)

    def score(theta_deg: np.ndarray) -> np.ndarray:
        theta_rad = np.deg2rad(theta_deg)[:, None]
        phi_rad = np.deg2rad(phis)[None, :]
        pred_tau = (baselines[None, :] / cfg.sound_speed_m_s) * np.cos(theta_rad - phi_rad)
        err = np.abs(pred_tau - taus[None, :])
        return np.sum(weights[None, :] * err, axis=1)

    coarse = np.arange(0.0, 360.0, max(0.25, cfg.doa_grid_step_deg), dtype=float)
    coarse_scores = score(coarse)
    best_coarse_idx = int(np.argmin(coarse_scores))
    best_theta = float(coarse[best_coarse_idx])

    refine = np.arange(
        best_theta - cfg.doa_refine_span_deg,
        best_theta + cfg.doa_refine_span_deg + cfg.doa_refine_step_deg,
        max(0.1, cfg.doa_refine_step_deg),
        dtype=float,
    )
    refine = np.mod(refine, 360.0)
    refine_scores = score(refine)
    best_ref_idx = int(np.argmin(refine_scores))
    doa_deg = normalize_angle_deg(float(refine[best_ref_idx]))

    # Confidence: combine pair coherence and fit residual quality.
    residual = float(refine_scores[best_ref_idx] / (np.sum(weights) + 1e-8))
    residual_norm = residual * cfg.sample_rate  # convert to "samples-like" scale
    fit_conf = float(np.clip(1.0 - (residual_norm / 3.0), 0.0, 1.0))
    pair_conf = float(np.clip(np.mean(weights), 0.0, 1.0))
    conf = 0.6 * pair_conf + 0.4 * fit_conf

    debug = {
        "valid_pairs": int(len(obs_taus)),
        "rejected_pairs": int(rejected_pairs),
        "pair_conf_mean": pair_conf,
        "fit_residual": residual,
    }
    return doa_deg, conf, debug
