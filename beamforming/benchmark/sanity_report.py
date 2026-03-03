from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if n <= 0:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    return a[:n], b[:n]


def plot_waveform_comparison(
    *,
    reference: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
    title: str,
    out_path: Path,
) -> None:
    ref, proc = _match_length(reference, processed)
    t = np.arange(len(ref), dtype=np.float64) / float(sample_rate)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, ref, color="black", linewidth=0.8)
    axes[0].set_title("Ground Truth (Target Speech Sum)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(t, proc, color="tab:blue", linewidth=0.8)
    axes[1].set_title("Beamformed Output")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spectrogram_comparison(
    *,
    reference: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
    title: str,
    out_path: Path,
) -> None:
    ref, proc = _match_length(reference, processed)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)

    axes[0].specgram(ref, NFFT=512, Fs=sample_rate, noverlap=256, cmap="magma")
    axes[0].set_title("Ground Truth Spectrogram")
    axes[0].set_ylabel("Frequency (Hz)")

    axes[1].specgram(proc, NFFT=512, Fs=sample_rate, noverlap=256, cmap="magma")
    axes[1].set_title("Beamformed Spectrogram")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Time (s)")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_noise_sweep_trends(
    *,
    snr_levels: list[float],
    delta_sii: list[float],
    delta_si_sdr: list[float],
    delta_stoi: list[float],
    title: str,
    out_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(snr_levels, delta_sii, marker="o", color="tab:green", label="Delta SII")
    ax1.plot(snr_levels, delta_stoi, marker="s", color="tab:orange", label="Delta STOI")
    ax2.plot(snr_levels, delta_si_sdr, marker="^", color="tab:blue", label="Delta SI-SDR (dB)")

    ax1.set_xlabel("Input SNR Target (dB)")
    ax1.set_ylabel("Delta SII / Delta STOI")
    ax2.set_ylabel("Delta SI-SDR (dB)")
    ax1.grid(True, alpha=0.25)
    ax1.set_title(title)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
