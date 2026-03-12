#!/usr/bin/env python3
"""
Mic Array Channel Order Detector (tap test)

Usage:
  1) Place the mic array on a stable surface.
  2) Run this script and follow the prompt.
  3) When it says START, tap near each mic in a known order
     (e.g., clockwise from a marked point), leaving ~0.5s between taps.
  4) The script prints which channel index corresponded to each tap.

Example:
  python -m mic_array_forwarder.tools.channel_order --audio-device-query "ReSpeaker" \
    --channel-count 4 --sample-rate-hz 48000 --tap-count 4

Output example:
  Tap 1 -> channel 2
  Tap 2 -> channel 0
  Tap 3 -> channel 3
  Tap 4 -> channel 1

Use the resulting order to build a channel map in the backend:
  channel_map = [2, 0, 3, 1]  # logical order = tap order
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
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


def _device_debug_string(sd: Any) -> str:
    parts: list[str] = []
    for idx, dev in enumerate(sd.query_devices()):
        parts.append(f"{idx}: {dev.get('name', 'unknown')} (inputs={dev.get('max_input_channels', 0)})")
    return "; ".join(parts)


def _pick_tap_times(energy: np.ndarray, sr: int, tap_count: int, min_gap_s: float) -> list[int]:
    if energy.size == 0:
        return []
    min_gap = max(1, int(sr * min_gap_s))
    threshold = np.percentile(energy, 98)
    candidates = np.where(energy >= threshold)[0].tolist()
    if not candidates:
        return []
    taps: list[int] = []
    last_idx = -min_gap
    for idx in candidates:
        if idx - last_idx < min_gap:
            continue
        taps.append(idx)
        last_idx = idx
        if len(taps) >= tap_count:
            break
    return taps


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect mic channel order via tap test.")
    parser.add_argument("--audio-device-query", default="ReSpeaker", help="Substring to match input device")
    parser.add_argument("--channel-count", type=int, default=4)
    parser.add_argument("--sample-rate-hz", type=int, default=48000)
    parser.add_argument("--tap-count", type=int, default=4)
    parser.add_argument("--min-gap-ms", type=int, default=400, help="Minimum gap between taps")
    parser.add_argument("--record-seconds", type=float, default=6.0)
    parser.add_argument(
        "--live-threshold-scale",
        type=float,
        default=1.0,
        help="Scale factor for the live tap-print threshold. Lower values trigger more easily.",
    )
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError as exc:
        print("sounddevice is required for this script", file=sys.stderr)
        raise SystemExit(1) from exc

    device_idx = _find_input_device(sd, args.audio_device_query, args.channel_count)
    if device_idx is None:
        print(
            "No matching input device found. "
            f"Requested query={args.audio_device_query!r}. Available devices: {_device_debug_string(sd)}",
            file=sys.stderr,
        )
        return 1

    sr = int(args.sample_rate_hz)
    frames = int(sr * float(args.record_seconds))
    min_gap_s = float(args.min_gap_ms) / 1000.0
    recorded_chunks: list[np.ndarray] = []
    chunk_frames_seen = 0
    last_live_tap_s = -min_gap_s
    live_tap_count = 0
    stop_requested = False
    baseline_peak = 1e-4
    recent_peaks: deque[float] = deque(maxlen=32)

    def _callback(indata: np.ndarray, frames_in: int, _time_info: Any, status: Any) -> None:
        nonlocal chunk_frames_seen, last_live_tap_s, live_tap_count, stop_requested, baseline_peak
        if status:
            print(f"audio status: {status}", file=sys.stderr, flush=True)
        chunk = np.asarray(indata, dtype=np.float32).copy()
        recorded_chunks.append(chunk)
        peak_by_channel = np.max(np.abs(chunk), axis=0)
        chunk_peak = float(np.max(peak_by_channel))
        recent_peaks.append(chunk_peak)
        baseline_peak = 0.97 * baseline_peak + 0.03 * chunk_peak
        live_scale = max(0.05, float(args.live_threshold_scale))
        live_threshold = max(
            (float(np.median(recent_peaks)) * 4.0 if recent_peaks else 0.0) * live_scale,
            (baseline_peak * 5.0) * live_scale,
            1e-4,
        )
        current_time_s = float(chunk_frames_seen) / float(sr)
        if chunk_peak >= live_threshold and (current_time_s - last_live_tap_s) >= min_gap_s:
            per_channel_str = ", ".join(f"ch{idx}={float(v):.4f}" for idx, v in enumerate(peak_by_channel))
            live_tap_count += 1
            print(
                f"detected tap {live_tap_count}/{int(args.tap_count)} ~{current_time_s:.2f}s "
                f"peak={chunk_peak:.4f} threshold={live_threshold:.4f} {per_channel_str}",
                flush=True,
            )
            last_live_tap_s = current_time_s
            if live_tap_count >= int(args.tap_count):
                stop_requested = True
        chunk_frames_seen += int(frames_in)

    print("Get ready to tap. Starting in 2s...")
    time.sleep(2.0)
    print("START: tap each mic in order, leave ~0.5s between taps.")

    with sd.InputStream(
        samplerate=sr,
        channels=int(args.channel_count),
        dtype="float32",
        device=device_idx,
        callback=_callback,
    ):
        deadline = time.monotonic() + float(args.record_seconds)
        while time.monotonic() < deadline and not stop_requested:
            sd.sleep(50)

    if not recorded_chunks:
        print("No audio captured.", file=sys.stderr)
        return 1
    data = np.concatenate(recorded_chunks, axis=0)[:frames]
    if data.ndim != 2 or data.shape[1] != int(args.channel_count):
        print(f"Unexpected recording shape: {data.shape}", file=sys.stderr)
        return 1

    energy = np.mean(np.abs(data), axis=1)
    tap_idxs = _pick_tap_times(energy, sr, int(args.tap_count), min_gap_s)
    if len(tap_idxs) < int(args.tap_count):
        print(
            f"Detected only {len(tap_idxs)} taps. Try again with louder taps or higher record duration.",
            file=sys.stderr,
        )
        return 1

    window = int(sr * 0.05)
    for i, idx in enumerate(tap_idxs, start=1):
        start = max(0, idx - window)
        end = min(data.shape[0], idx + window)
        window_data = data[start:end]
        per_ch = np.max(np.abs(window_data), axis=0)
        ch = int(np.argmax(per_ch))
        print(f"Tap {i} -> channel {ch}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
