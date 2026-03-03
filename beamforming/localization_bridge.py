from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from localization.algo import (
    CSSMLocalization,
    GMDALaplace,
    MUSICLocalization,
    NormMUSICLocalization,
    SRPPHATLocalization,
    SSZLocalization,
    WAVESLocalization,
)
from simulation.simulation_config import SimulationConfig
from simulation.target_policy import iter_target_source_indices


DEFAULT_METHOD_CONFIGS: dict[str, dict] = {
    "SSZ": {"nfft": 512, "overlap": 0.5, "epsilon": 0.1, "d_freq": 8, "freq_range": (200, 3000)},
    "SRP-PHAT": {"nfft": 512, "overlap": 0.5, "freq_range": (200, 3000)},
    "GMDA": {"nfft": 512, "overlap": 0.5, "freq_range": (500, 3000), "power_thresh_percentile": 80, "mdl_beta": 3.0},
    "MUSIC": {"nfft": 512, "overlap": 0.5, "freq_range": (200, 3000), "grid_size": 360},
    "NormMUSIC": {"nfft": 512, "overlap": 0.5, "freq_range": (200, 3000), "grid_size": 360},
    "CSSM": {"nfft": 512, "overlap": 0.5, "freq_range": (200, 3000), "grid_size": 360, "num_iter": 5},
    "WAVES": {"nfft": 512, "overlap": 0.5, "freq_range": (200, 3000), "grid_size": 360, "num_iter": 5},
}


@dataclass
class LocalizationEstimate:
    method: str
    doas_deg: list[float]
    confidence: float
    error: str | None = None


def _normalize_deg(angle_deg: float) -> float:
    return float(angle_deg % 360.0)


def _angular_distance_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return float(diff)


def normalize_doa_list(doas_deg: list[float], merge_threshold_deg: float = 6.0, max_targets: int | None = None) -> list[float]:
    """Normalize angles, merge near-duplicates, and sort for deterministic constraints."""
    if not doas_deg:
        return []

    vals = sorted(_normalize_deg(d) for d in doas_deg)
    merged: list[list[float]] = []

    for d in vals:
        if not merged:
            merged.append([d])
            continue
        cluster_mean = float(np.mean(merged[-1]))
        if _angular_distance_deg(d, cluster_mean) <= merge_threshold_deg:
            merged[-1].append(d)
        else:
            merged.append([d])

    out = [float(np.mean(c)) % 360.0 for c in merged]
    out.sort()
    if max_targets is not None and max_targets > 0:
        out = out[:max_targets]
    return out


def _estimate_confidence(doas_deg: list[float], n_targets_hint: int) -> float:
    if not doas_deg:
        return 0.0
    count = len(doas_deg)
    hint = max(1, int(n_targets_hint))

    count_score = 1.0 / (1.0 + abs(count - hint))
    if count <= 1:
        sep_score = 1.0
    else:
        ordered = sorted(doas_deg)
        gaps = [
            _angular_distance_deg(ordered[i], ordered[(i + 1) % count])
            for i in range(count)
        ]
        min_gap = min(gaps)
        sep_score = min(1.0, min_gap / 20.0)

    return float(0.65 * count_score + 0.35 * sep_score)


def _build_algorithm(method: str, mic_pos_rel: np.ndarray, fs: int, n_targets: int, cfg: dict):
    common = {
        "mic_pos": mic_pos_rel,
        "fs": fs,
        "nfft": cfg.get("nfft", 512),
        "overlap": cfg.get("overlap", 0.5),
        "freq_range": tuple(cfg.get("freq_range", (200, 3000))),
        "max_sources": max(1, n_targets),
    }
    if method == "SSZ":
        return SSZLocalization(**common, epsilon=cfg.get("epsilon", 0.1), d_freq=cfg.get("d_freq", 8))
    if method == "SRP-PHAT":
        return SRPPHATLocalization(**common)
    if method == "GMDA":
        return GMDALaplace(
            **common,
            power_thresh_percentile=cfg.get("power_thresh_percentile", 80),
            mdl_beta=cfg.get("mdl_beta", 3.0),
        )
    if method == "MUSIC":
        return MUSICLocalization(**common, grid_size=cfg.get("grid_size", 360))
    if method == "NormMUSIC":
        return NormMUSICLocalization(**common, grid_size=cfg.get("grid_size", 360))
    if method == "CSSM":
        return CSSMLocalization(**common, grid_size=cfg.get("grid_size", 360), num_iter=cfg.get("num_iter", 5))
    if method == "WAVES":
        return WAVESLocalization(**common, grid_size=cfg.get("grid_size", 360), num_iter=cfg.get("num_iter", 5))
    raise ValueError(f"Unsupported localization method: {method}")


def oracle_target_locations(sim_config: SimulationConfig) -> np.ndarray:
    idxs = list(iter_target_source_indices(sim_config))
    if not idxs:
        return np.zeros((0, 3), dtype=float)
    return np.asarray([sim_config.audio.sources[i].loc for i in idxs], dtype=float)


def oracle_target_doas_deg(sim_config: SimulationConfig) -> list[float]:
    center = sim_config.microphone_array.mic_center
    out: list[float] = []
    for i in iter_target_source_indices(sim_config):
        s = sim_config.audio.sources[i]
        dx = s.loc[0] - center[0]
        dy = s.loc[1] - center[1]
        out.append(_normalize_deg(math.degrees(math.atan2(dy, dx))))
    return out


def estimate_doas_deg(
    method: str,
    mic_signals: np.ndarray,
    mic_pos_rel: np.ndarray,
    fs: int,
    n_targets_hint: int,
    method_cfg: dict | None = None,
) -> list[float]:
    cfg = dict(DEFAULT_METHOD_CONFIGS.get(method, {}))
    if method_cfg:
        cfg.update(method_cfg)

    algo = _build_algorithm(method, mic_pos_rel, fs, n_targets_hint, cfg)
    estimated_doas_rad, _, _ = algo.process(mic_signals)
    doas = [_normalize_deg(math.degrees(a)) for a in estimated_doas_rad]
    return normalize_doa_list(doas, max_targets=max(1, n_targets_hint))


def estimate_doas_with_fallback(
    methods: list[str],
    mic_signals: np.ndarray,
    mic_pos_rel: np.ndarray,
    fs: int,
    n_targets_hint: int,
    method_cfg_map: dict[str, dict] | None = None,
) -> tuple[LocalizationEstimate, list[LocalizationEstimate]]:
    """
    Run localization methods in priority order and return best available estimate.

    Selection policy:
    - Prefer the first method that returns non-empty DOAs.
    - Attach a confidence score for downstream diagnostics.
    - If none return DOAs, return the first result (error or empty).
    """
    if not methods:
        raise ValueError("methods must not be empty")

    cfg_map = method_cfg_map or {}
    all_results: list[LocalizationEstimate] = []

    for method in methods:
        try:
            doas = estimate_doas_deg(
                method=method,
                mic_signals=mic_signals,
                mic_pos_rel=mic_pos_rel,
                fs=fs,
                n_targets_hint=n_targets_hint,
                method_cfg=cfg_map.get(method),
            )
            result = LocalizationEstimate(
                method=method,
                doas_deg=doas,
                confidence=_estimate_confidence(doas, n_targets_hint),
            )
        except Exception as exc:
            result = LocalizationEstimate(method=method, doas_deg=[], confidence=0.0, error=str(exc))

        all_results.append(result)
        if result.doas_deg:
            return result, all_results

    return all_results[0], all_results
