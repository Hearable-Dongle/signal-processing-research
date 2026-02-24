from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validation_contracts import SceneConfig


@dataclass
class SceneData:
    scene_id: str
    sample_rate: int
    chunk_samples: int
    mic_geometry_xy: np.ndarray  # (M,2)
    oracle_sources: np.ndarray  # (K,T)
    raw_mic: np.ndarray  # (T,M)
    oracle_doa_deg_by_chunk: np.ndarray  # (C,K)
    separated_streams_by_chunk: list[list[np.ndarray]]  # C x K x (chunk,)
    stream_to_oracle_by_chunk: list[dict[int, int]]  # C entries map stream_idx->oracle_label
    srp_peaks_by_chunk: list[list[float]]
    srp_scores_by_chunk: list[list[float] | None]


def _angle_wrap_deg(a: np.ndarray | float) -> np.ndarray | float:
    return np.mod(a, 360.0)


def _fractional_delay(sig: np.ndarray, delay_samples: float) -> np.ndarray:
    n = sig.shape[0]
    idx = np.arange(n, dtype=float) - float(delay_samples)
    return np.interp(idx, np.arange(n, dtype=float), sig, left=0.0, right=0.0)


def _build_mic_geometry_xy(n_mics: int, spacing_m: float) -> np.ndarray:
    xs = (np.arange(n_mics) - (n_mics - 1) / 2.0) * spacing_m
    ys = np.zeros_like(xs)
    return np.stack([xs, ys], axis=1).astype(float)


def _make_speaker_signal(sr: int, total_samples: int, rng: np.random.Generator, idx: int) -> np.ndarray:
    t = np.arange(total_samples, dtype=float) / sr
    base_f = rng.uniform(90.0, 280.0) + idx * rng.uniform(20.0, 60.0)
    vibrato = 3.0 * np.sin(2 * np.pi * rng.uniform(1.0, 2.2) * t)
    fm = 2 * np.pi * (base_f * t + 0.002 * np.cumsum(vibrato) / sr)
    carrier = np.sin(fm)
    harmonic = 0.35 * np.sin(2 * fm + rng.uniform(0.0, 2 * np.pi))
    env = 0.5 + 0.5 * np.maximum(0.0, np.sin(2 * np.pi * rng.uniform(2.0, 4.5) * t + rng.uniform(0, 2 * np.pi)))
    breath = 0.03 * rng.standard_normal(total_samples)
    sig = (carrier + harmonic) * env + breath
    sig = sig.astype(np.float32)
    sig /= max(1e-6, float(np.max(np.abs(sig))))
    return 0.2 * sig


def _make_doa_trajectory(
    n_chunks: int,
    rng: np.random.Generator,
    moving_probability: float,
) -> np.ndarray:
    start = rng.uniform(0.0, 360.0)
    if rng.random() < moving_probability:
        vel = rng.uniform(-20.0, 20.0)  # deg/s approximated per chunk later
    else:
        vel = 0.0
    # per chunk increment around 0.2s baseline; exact chunk dt applied by caller scaling
    traj = start + vel * np.arange(n_chunks) * 0.2
    return _angle_wrap_deg(traj.astype(float))


def generate_scene(scene_id: str, cfg: SceneConfig, rng: np.random.Generator) -> SceneData:
    sr = cfg.sample_rate
    total_samples = int(cfg.duration_sec * sr)
    chunk_samples = max(1, int(cfg.chunk_ms * sr / 1000))
    n_chunks = int(np.ceil(total_samples / chunk_samples))

    mic_xy = _build_mic_geometry_xy(cfg.n_mics, cfg.mic_spacing_m)

    oracle_sources = np.stack(
        [_make_speaker_signal(sr, total_samples, rng, i) for i in range(cfg.n_speakers)],
        axis=0,
    )

    doa_by_chunk = np.stack(
        [_make_doa_trajectory(n_chunks, rng, cfg.moving_probability) for _ in range(cfg.n_speakers)],
        axis=1,
    )

    c = 343.0
    raw_mic = np.zeros((total_samples, cfg.n_mics), dtype=np.float32)

    separated_streams_by_chunk: list[list[np.ndarray]] = []
    stream_to_oracle_by_chunk: list[dict[int, int]] = []
    srp_peaks_by_chunk: list[list[float]] = []
    srp_scores_by_chunk: list[list[float] | None] = []

    for chunk_id in range(n_chunks):
        start = chunk_id * chunk_samples
        end = min(total_samples, start + chunk_samples)
        L = end - start

        src_chunks = [oracle_sources[k, start:end].astype(np.float32, copy=False) for k in range(cfg.n_speakers)]

        # Raw mic mixture via far-field delay model.
        mix_chunk_mc = np.zeros((L, cfg.n_mics), dtype=np.float32)
        for k in range(cfg.n_speakers):
            theta = np.deg2rad(doa_by_chunk[chunk_id, k])
            direction = np.array([np.cos(theta), np.sin(theta)], dtype=float)
            for m in range(cfg.n_mics):
                tau = -float(np.dot(mic_xy[m], direction)) / c
                d_samples = tau * sr
                delayed = _fractional_delay(src_chunks[k], d_samples).astype(np.float32, copy=False)
                mix_chunk_mc[:, m] += delayed

        mix_chunk_mc += (cfg.noise_std * rng.standard_normal(mix_chunk_mc.shape)).astype(np.float32)
        raw_mic[start:end, :] = mix_chunk_mc

        # Separated stream proxies with bleed + permutation.
        sep = []
        for i in range(cfg.n_speakers):
            y = src_chunks[i].copy()
            for j in range(cfg.n_speakers):
                if i == j:
                    continue
                y += cfg.bleed_ratio * src_chunks[j]
            y += (0.01 * rng.standard_normal(L)).astype(np.float32)
            sep.append(y)

        perm = rng.permutation(cfg.n_speakers)
        sep_perm = [sep[int(p)] for p in perm]
        map_stream_to_oracle = {sidx: int(perm[sidx]) for sidx in range(cfg.n_speakers)}

        separated_streams_by_chunk.append(sep_perm)
        stream_to_oracle_by_chunk.append(map_stream_to_oracle)

        # SRP proxy peaks (oracle + noise + distractors).
        peaks = []
        scores = []
        for k in range(cfg.n_speakers):
            peaks.append(float(_angle_wrap_deg(doa_by_chunk[chunk_id, k] + rng.normal(0.0, cfg.srp_peak_noise_deg))))
            scores.append(float(rng.uniform(0.65, 0.98)))
        for _ in range(max(0, int(cfg.srp_num_distractors))):
            peaks.append(float(rng.uniform(0.0, 360.0)))
            scores.append(float(rng.uniform(0.1, 0.45)))

        order = rng.permutation(len(peaks))
        srp_peaks_by_chunk.append([float(peaks[i]) for i in order])
        srp_scores_by_chunk.append([float(scores[i]) for i in order])

    return SceneData(
        scene_id=scene_id,
        sample_rate=sr,
        chunk_samples=chunk_samples,
        mic_geometry_xy=mic_xy,
        oracle_sources=oracle_sources,
        raw_mic=raw_mic,
        oracle_doa_deg_by_chunk=doa_by_chunk,
        separated_streams_by_chunk=separated_streams_by_chunk,
        stream_to_oracle_by_chunk=stream_to_oracle_by_chunk,
        srp_peaks_by_chunk=srp_peaks_by_chunk,
        srp_scores_by_chunk=srp_scores_by_chunk,
    )
