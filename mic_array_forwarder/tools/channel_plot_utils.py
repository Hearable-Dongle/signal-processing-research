from __future__ import annotations

from io import BytesIO

import numpy as np


def smooth_envelope(x: np.ndarray, win: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0 or win <= 1:
        return arr
    kernel = np.ones(int(win), dtype=np.float64) / float(win)
    return np.convolve(arr, kernel, mode="same")


def default_channel_labels(channel_count: int) -> list[str]:
    return [f"raw ch{idx} · mic {idx + 1}" for idx in range(int(channel_count))]


def render_multichannel_plot_png_bytes(
    *,
    data: np.ndarray,
    sample_rate_hz: int,
    channel_labels: list[str] | None = None,
    title: str = "Mic channels",
    subtitle: str = "",
    envelope_ms: float = 15.0,
    prompt_windows: list[tuple[float, float, str]] | None = None,
) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return b""

    sr = max(1, int(sample_rate_hz))
    channel_count = int(arr.shape[1])
    labels = list(channel_labels) if channel_labels is not None else default_channel_labels(channel_count)
    if len(labels) != channel_count:
        labels = default_channel_labels(channel_count)

    times = np.arange(arr.shape[0], dtype=np.float64) / float(sr)
    envelope_win = max(1, int(round((float(envelope_ms) / 1000.0) * sr)))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(channel_count, 2)))
    envelopes: list[np.ndarray] = []
    raw_peaks: list[float] = []

    for ch in range(channel_count):
        raw = np.abs(np.asarray(arr[:, ch], dtype=np.float64))
        env = smooth_envelope(raw, envelope_win)
        envelopes.append(env)
        raw_peaks.append(float(np.max(raw)) if raw.size else 0.0)

    global_ymax = max([float(np.max(env)) if env.size else 0.0 for env in envelopes] + [1e-6])
    fig, axes = plt.subplots(channel_count, 1, figsize=(14, max(3, 2.2 * channel_count)), sharex=True)
    axes_arr = np.atleast_1d(axes)

    for ch in range(channel_count):
        env = envelopes[ch]
        axes_arr[ch].plot(times, env, color=colors[ch], linewidth=1.0)
        axes_arr[ch].set_ylabel(labels[ch], fontsize=9)
        axes_arr[ch].grid(True, alpha=0.2)
        axes_arr[ch].set_ylim(0.0, global_ymax)
        axes_arr[ch].set_title(f"{labels[ch]} peak={raw_peaks[ch]:.4f}", loc="left", fontsize=10)
        if prompt_windows:
            for idx, (start_s, end_s, label) in enumerate(prompt_windows):
                axes_arr[ch].axvspan(start_s, end_s, color="black", alpha=0.03 if idx % 2 == 0 else 0.06)
                axes_arr[ch].axvline(start_s, color="gray", alpha=0.25, linewidth=0.8)
                if ch == 0:
                    axes_arr[ch].text((start_s + end_s) * 0.5, axes_arr[ch].get_ylim()[1] * 0.95, label, ha="center", va="top")
            axes_arr[ch].axvline(prompt_windows[-1][1], color="gray", alpha=0.25, linewidth=0.8)

    axes_arr[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    if subtitle:
        fig.text(0.5, 0.975, subtitle, ha="center", va="top", fontsize=10, color="#555555")
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, dpi=180, format="png")
    plt.close(fig)
    return buf.getvalue()
