#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _find_input_device(sd: Any, query: str | None, min_channels: int) -> int | None:
    devices = sd.query_devices()
    if query is None or not str(query).strip():
        for idx, dev in enumerate(devices):
            if int(dev.get("max_input_channels", 0)) >= min_channels:
                return idx
        return None

    query_lc = str(query).strip().lower()
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) < min_channels:
            continue
        if query_lc in str(dev.get("name", "")).lower():
            return idx
    return None


def _ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _smooth_envelope(x: np.ndarray, win: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0 or win <= 1:
        return arr
    kernel = np.ones(int(win), dtype=np.float64) / float(win)
    return np.convolve(arr, kernel, mode="same")


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a guided mic tap sequence and save per-channel amplitude plots.")
    parser.add_argument("--audio-device-query", default="ReSpeaker", help="Substring to match input device")
    parser.add_argument(
        "--channel-count",
        type=int,
        default=0,
        help="Number of channels to capture. Use 0 to capture all available input channels.",
    )
    parser.add_argument("--sample-rate-hz", type=int, default=48000)
    parser.add_argument("--mic-count", type=int, default=4, help="How many mic prompts to announce")
    parser.add_argument("--seconds-per-mic", type=float, default=2.0, help="Duration of each prompted tap window")
    parser.add_argument("--out-path", default="tap_plot.png", help="Output image path")
    parser.add_argument("--envelope-ms", type=float, default=15.0, help="Envelope smoothing window in ms")
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("matplotlib is required for this script", file=sys.stderr)
        raise SystemExit(1) from exc

    try:
        import sounddevice as sd
    except ImportError as exc:
        print("sounddevice is required for this script", file=sys.stderr)
        raise SystemExit(1) from exc

    min_channels = 1 if int(args.channel_count) <= 0 else int(args.channel_count)
    device_idx = _find_input_device(sd, args.audio_device_query, min_channels)
    if device_idx is None:
        print(f"No matching input device found for query={args.audio_device_query!r}.", file=sys.stderr)
        return 1

    device_info = sd.query_devices(device_idx)
    max_input_channels = int(device_info.get("max_input_channels", 0))
    requested_channels = int(args.channel_count)
    capture_channels = max_input_channels if requested_channels <= 0 else min(requested_channels, max_input_channels)
    if capture_channels <= 0:
        print(f"Selected device has no input channels: {device_info}", file=sys.stderr)
        return 1

    sr = int(args.sample_rate_hz)
    mic_count = max(1, int(args.mic_count))
    seconds_per_mic = max(0.25, float(args.seconds_per_mic))
    total_seconds = mic_count * seconds_per_mic
    total_frames = int(round(total_seconds * sr))
    chunks: list[np.ndarray] = []

    print(f"Using device {device_idx}: {device_info.get('name', 'unknown')}")
    print(f"Port max input channels: {max_input_channels}")
    print(f"Requested capture channels: {requested_channels}")
    print(f"Actual captured channels: {capture_channels}")
    print("Get ready. Recording starts in 2 seconds.")
    time.sleep(2.0)

    announced_idx = -1

    def _callback(indata: np.ndarray, _frames_in: int, _time_info: Any, status: Any) -> None:
        if status:
            print(f"audio status: {status}", file=sys.stderr, flush=True)
        chunks.append(np.asarray(indata, dtype=np.float32).copy())

    with sd.InputStream(
        samplerate=sr,
        channels=capture_channels,
        dtype="float32",
        device=device_idx,
        callback=_callback,
    ):
        start_t = time.monotonic()
        while True:
            elapsed = time.monotonic() - start_t
            prompt_idx = min(int(elapsed // seconds_per_mic), mic_count - 1)
            if prompt_idx != announced_idx and prompt_idx < mic_count:
                announced_idx = prompt_idx
                print(f"Tap {_ordinal(prompt_idx + 1)} mic now.", flush=True)
            if elapsed >= total_seconds:
                break
            sd.sleep(50)

    if not chunks:
        print("No audio captured.", file=sys.stderr)
        return 1

    data = np.concatenate(chunks, axis=0)[:total_frames]
    if data.ndim != 2:
        print(f"Unexpected recording shape: {data.shape}", file=sys.stderr)
        return 1

    times = np.arange(data.shape[0], dtype=np.float64) / float(sr)
    envelope_win = max(1, int(round((float(args.envelope_ms) / 1000.0) * sr)))
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(capture_channels, 1, figsize=(14, max(3, 2.2 * capture_channels)), sharex=True)
    axes_arr = np.atleast_1d(axes)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(capture_channels, 2)))
    envelopes: list[np.ndarray] = []
    raw_peaks: list[float] = []

    for ch in range(capture_channels):
        raw = np.abs(np.asarray(data[:, ch], dtype=np.float64))
        env = _smooth_envelope(raw, envelope_win)
        envelopes.append(env)
        raw_peaks.append(float(np.max(raw)) if raw.size else 0.0)

    global_ymax = max([float(np.max(env)) if env.size else 0.0 for env in envelopes] + [1e-6])

    for ch in range(capture_channels):
        env = envelopes[ch]
        axes_arr[ch].plot(times, env, color=colors[ch], linewidth=1.0)
        axes_arr[ch].set_ylabel(f"ch {ch}")
        axes_arr[ch].grid(True, alpha=0.2)
        axes_arr[ch].set_ylim(0.0, global_ymax)
        peak = raw_peaks[ch]
        axes_arr[ch].set_title(f"Channel {ch} peak={peak:.4f}", loc="left", fontsize=10)
        for mic_idx in range(mic_count):
            start_s = mic_idx * seconds_per_mic
            end_s = start_s + seconds_per_mic
            axes_arr[ch].axvspan(start_s, end_s, color="black", alpha=0.03 if mic_idx % 2 == 0 else 0.06)
            axes_arr[ch].axvline(start_s, color="gray", alpha=0.25, linewidth=0.8)
        axes_arr[ch].axvline(total_seconds, color="gray", alpha=0.25, linewidth=0.8)

    axes_arr[-1].set_xlabel("time (s)")
    for mic_idx in range(mic_count):
        center_s = (mic_idx + 0.5) * seconds_per_mic
        axes_arr[0].text(center_s, axes_arr[0].get_ylim()[1] * 0.95, f"tap {_ordinal(mic_idx + 1)} mic", ha="center", va="top")

    fig.suptitle("Mic Tap Capture by Channel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Saved plot to {out_path}")
    print(f"Port max input channels: {max_input_channels}")
    print(f"Captured channels in stream: {capture_channels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
