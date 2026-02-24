from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_doa_timeline(
    out_path: Path,
    chunk_times_s: list[float],
    oracle_doa_chunks: list[dict[int, float]],
    est_doa_chunks: list[dict[int, float]],
) -> None:
    labels = sorted({k for d in oracle_doa_chunks for k in d.keys()})
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for lab in labels:
        o = [d.get(lab, np.nan) for d in oracle_doa_chunks]
        e = [d.get(lab, np.nan) for d in est_doa_chunks]
        ax.plot(chunk_times_s, o, linestyle="--", linewidth=1.5, label=f"oracle {lab}")
        ax.plot(chunk_times_s, e, linestyle="-", linewidth=1.2, label=f"est {lab}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("DOA (deg)")
    ax.set_ylim(0, 360)
    ax.set_title("DOA Timeline: Oracle vs Estimated")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, out_path)


def plot_error_histogram(out_path: Path, errors_deg: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if errors_deg:
        ax.hist(np.asarray(errors_deg), bins=24, alpha=0.85)
    ax.set_xlabel("Absolute Error (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Direction Error Histogram")
    ax.grid(True, alpha=0.25)
    _save(fig, out_path)


def plot_room_topdown(
    out_path: Path,
    mic_xy: np.ndarray,
    oracle_doa_chunks: list[dict[int, float]],
) -> None:
    # Show mic layout and initial speaker bearings on unit circle.
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(mic_xy[:, 0], mic_xy[:, 1], marker="^", label="mics")

    if oracle_doa_chunks:
        d0 = oracle_doa_chunks[0]
        for sid, doa in d0.items():
            th = np.deg2rad(doa)
            ax.arrow(0.0, 0.0, np.cos(th), np.sin(th), head_width=0.03, length_includes_head=True)
            ax.text(1.08 * np.cos(th), 1.08 * np.sin(th), f"spk {sid}", fontsize=8)

    ax.axhline(0, linewidth=0.8, alpha=0.3)
    ax.axvline(0, linewidth=0.8, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Top-Down: Mic Layout + Initial Oracle Bearings")
    ax.legend()
    _save(fig, out_path)


def plot_weight_timeline(
    out_path: Path,
    chunk_times_s: list[float],
    target_speaker_ids_seq: list[list[int]],
    target_weights_seq: list[list[float]],
) -> None:
    # Flatten to per-speaker sparse timeline.
    all_ids = sorted({sid for ids in target_speaker_ids_seq for sid in ids})
    fig, ax = plt.subplots(figsize=(10, 4.0))

    for sid in all_ids:
        ys = []
        for ids, ws in zip(target_speaker_ids_seq, target_weights_seq):
            if sid in ids:
                ys.append(ws[ids.index(sid)])
            else:
                ys.append(np.nan)
        ax.plot(chunk_times_s, ys, label=f"speaker {sid}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Weight")
    ax.set_title("Target Weight Timeline")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, out_path)
