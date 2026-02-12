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
from localization.target_policy import true_target_doas_deg
from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation


@dataclass(frozen=True)
class AlgorithmResult:
    estimated_doas_deg: list[float]
    true_doas_deg: list[float]
    runtime_seconds: float


def _normalize_deg(angle_deg: float) -> float:
    return float(angle_deg % 360.0)


def _build_algorithm(method: str, mic_pos_rel: np.ndarray, fs: int, n_true_targets: int, cfg: dict):
    common = {
        "mic_pos": mic_pos_rel,
        "fs": fs,
        "nfft": cfg.get("nfft", 512),
        "overlap": cfg.get("overlap", 0.5),
        "freq_range": tuple(cfg.get("freq_range", [200, 3000])),
        "max_sources": max(1, n_true_targets),
    }

    if method == "SSZ":
        return SSZLocalization(
            **common,
            epsilon=cfg.get("epsilon", 0.1),
            d_freq=cfg.get("d_freq", 8),
        )
    if method == "SRP-PHAT":
        return SRPPHATLocalization(**common)
    if method == "GMDA":
        return GMDALaplace(
            **common,
            power_thresh_percentile=cfg.get("power_thresh_percentile", 90),
            mdl_beta=cfg.get("mdl_beta", 3.0),
        )
    if method == "MUSIC":
        return MUSICLocalization(
            **common,
            grid_size=cfg.get("grid_size", 360),
        )
    if method == "NormMUSIC":
        return NormMUSICLocalization(
            **common,
            grid_size=cfg.get("grid_size", 360),
        )
    if method == "CSSM":
        return CSSMLocalization(
            **common,
            grid_size=cfg.get("grid_size", 360),
            num_iter=cfg.get("num_iter", 5),
        )
    if method == "WAVES":
        return WAVESLocalization(
            **common,
            grid_size=cfg.get("grid_size", 360),
            num_iter=cfg.get("num_iter", 5),
        )

    raise ValueError(f"Unsupported method: {method}")


def run_method_on_scene(method: str, method_cfg: dict, sim_config: SimulationConfig) -> AlgorithmResult:
    import time

    t0 = time.perf_counter()

    mic_audio, mic_pos_abs, _ = run_simulation(sim_config)
    mic_signals = mic_audio.T

    center = np.array(sim_config.microphone_array.mic_center).reshape(3, 1)
    mic_pos_rel = mic_pos_abs - center

    true_doas = true_target_doas_deg(sim_config)

    algo = _build_algorithm(
        method=method,
        mic_pos_rel=mic_pos_rel,
        fs=sim_config.audio.fs,
        n_true_targets=len(true_doas),
        cfg=method_cfg,
    )
    estimated_doas_rad, _, _ = algo.process(mic_signals)
    estimated_deg = [_normalize_deg(math.degrees(x)) for x in estimated_doas_rad]

    t1 = time.perf_counter()
    return AlgorithmResult(
        estimated_doas_deg=estimated_deg,
        true_doas_deg=true_doas,
        runtime_seconds=t1 - t0,
    )
