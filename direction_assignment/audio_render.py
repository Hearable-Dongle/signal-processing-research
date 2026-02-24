from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


class StitchAccumulator:
    def __init__(self, total_samples: int):
        self.total_samples = total_samples
        self.buffers: dict[int, np.ndarray] = {}
        self.weights: dict[int, np.ndarray] = {}

    def add_chunk(self, speaker_id: int, chunk: np.ndarray, start: int, end: int) -> None:
        sid = int(speaker_id)
        if sid not in self.buffers:
            self.buffers[sid] = np.zeros(self.total_samples, dtype=np.float32)
            self.weights[sid] = np.zeros(self.total_samples, dtype=np.float32)

        L = end - start
        x = np.asarray(chunk[:L], dtype=np.float32)
        self.buffers[sid][start:end] += x
        self.weights[sid][start:end] += 1.0

    def render(self) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for sid in sorted(self.buffers.keys()):
            b = self.buffers[sid]
            w = self.weights[sid]
            y = np.zeros_like(b)
            valid = w > 1e-8
            y[valid] = b[valid] / w[valid]
            out[sid] = y
        return out


def write_audio_bundle(
    out_dir: Path,
    sample_rate: int,
    mixture_mono: np.ndarray,
    oracle_sources: np.ndarray,
    estimated_tracks: dict[int, np.ndarray],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / "mixture.wav"), mixture_mono.astype(np.float32), sample_rate)
    for i in range(oracle_sources.shape[0]):
        sf.write(str(out_dir / f"oracle_source_{i}.wav"), oracle_sources[i].astype(np.float32), sample_rate)
    for sid, y in estimated_tracks.items():
        sf.write(str(out_dir / f"estimated_speaker_{sid}.wav"), y.astype(np.float32), sample_rate)
