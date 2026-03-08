#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import sys
import threading
import urllib.request
import wave

import numpy as np
import sounddevice as sd
import websockets

from mic_array_forwarder.ws_codec import decode_audio_chunk

SAMPLE_RATE_HZ = 48000


def _start_session(http_base: str, payload: dict, *, stop_existing: bool) -> str:
    url = f"{http_base}/api/session/start"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 409 and stop_existing:
            stop_req = urllib.request.Request(
                f"{http_base}/api/session/active/stop",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(stop_req):
                pass
            with urllib.request.urlopen(req) as resp:
                body = resp.read().decode("utf-8")
        else:
            raise RuntimeError(f"start session failed: HTTP {exc.code} {detail}") from exc
    parsed = json.loads(body)
    session_id = parsed.get("session_id")
    if not session_id:
        raise RuntimeError(f"start session failed: missing session_id in response: {parsed}")
    return str(session_id)


class AudioPlayer:
    def __init__(self, sample_rate_hz: int, buffer_ms: int) -> None:
        self.sample_rate_hz = int(sample_rate_hz)
        self.buffer_samples = max(1, int(self.sample_rate_hz * buffer_ms / 1000))
        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._lock = threading.Lock()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._wav_path: str | None = None
        self._wav: wave.Wave_write | None = None
        self._wav_lock = threading.Lock()

    def push(self, samples: np.ndarray) -> None:
        try:
            self.q.put_nowait(samples)
        except queue.Full:
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                return
            try:
                self.q.put_nowait(samples)
            except queue.Full:
                return

    def _refill(self) -> None:
        try:
            while self._buffer.size < self.buffer_samples:
                chunk = self.q.get_nowait()
                self._write_wav(chunk)
                if self._buffer.size == 0:
                    self._buffer = chunk
                else:
                    self._buffer = np.concatenate([self._buffer, chunk], axis=0)
        except queue.Empty:
            return

    def set_wav_output(self, path: str | None) -> None:
        with self._wav_lock:
            self._wav_path = None if path is None else str(path)

    def _ensure_wav(self) -> None:
        if not self._wav_path:
            return
        if self._wav is not None:
            return
        wav = wave.open(self._wav_path, "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(self.sample_rate_hz)
        self._wav = wav

    def _write_wav(self, samples: np.ndarray) -> None:
        with self._wav_lock:
            if not self._wav_path:
                return
            self._ensure_wav()
            if self._wav is None:
                return
            pcm = np.clip(samples, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype(np.int16, copy=False)
            self._wav.writeframes(pcm16.tobytes())

    def close(self) -> None:
        with self._wav_lock:
            if self._wav is not None:
                self._wav.close()
                self._wav = None

    def callback(self, outdata: np.ndarray, _frames: int, _time_info: dict, status: sd.CallbackFlags) -> None:
        if status:
            pass
        with self._lock:
            self._refill()
            if self._buffer.size < outdata.shape[0]:
                # underrun -> fill silence
                needed = outdata.shape[0]
                available = self._buffer.size
                if available > 0:
                    outdata[:available, 0] = self._buffer
                if available < needed:
                    outdata[available:needed, 0] = 0.0
                self._buffer = np.zeros(0, dtype=np.float32)
                return
            outdata[:, 0] = self._buffer[: outdata.shape[0]]
            self._buffer = self._buffer[outdata.shape[0] :]


async def _read_ws(ws_url: str, player: AudioPlayer) -> None:
    async with websockets.connect(ws_url, max_size=None) as ws:
        print(f"connected {ws_url}")
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                try:
                    _ts_ms, audio = decode_audio_chunk(msg)
                    if audio.size:
                        player.push(audio.astype(np.float32, copy=False))
                except Exception as exc:
                    print(f"audio decode error: {exc}")
            else:
                try:
                    payload = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "metrics":
                    tracker = payload.get("tracker_debug", {})
                    print(
                        "metrics",
                        f"device={payload.get('device_name')}",
                        f"channels={payload.get('channel_count')}",
                        f"sr={payload.get('sample_rate_hz')}",
                        f"monitor={payload.get('monitor_source')}",
                        f"frame={tracker.get('last_frame_shape')}",
                        f"rms={tracker.get('rms_by_channel')}",
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Play realtime audio from the backend WebSocket.")
    parser.add_argument("--http-base", default="http://localhost:8000", help="HTTP base URL")
    parser.add_argument("--ws-base", default="ws://localhost:8000", help="WebSocket base URL")
    parser.add_argument("--audio-device-query", default="ReSpeaker", help="Substring to match input device")
    parser.add_argument("--channel-count", type=int, default=4)
    parser.add_argument("--sample-rate-hz", type=int, default=SAMPLE_RATE_HZ)
    parser.add_argument("--monitor-source", choices=["processed", "raw_mixed"], default="raw_mixed")
    parser.add_argument("--buffer-ms", type=int, default=120, help="Jitter buffer (ms)")
    parser.add_argument("--wav-out", default="", help="Optional path to save output WAV")
    parser.add_argument("--stop-existing", action="store_true", help="Stop any active session before starting")
    parser.add_argument(
        "--mic-array-profile",
        choices=["respeaker_v3_0457", "respeaker_cross_0640"],
        default="respeaker_v3_0457",
    )
    parser.add_argument("--channel-map", default="", help="Comma-separated channel map, e.g. 0,1,2,3")
    args = parser.parse_args()

    payload = {
        "input_source": "respeaker_live",
        "audio_device_query": args.audio_device_query,
        "channel_count": args.channel_count,
        "sample_rate_hz": args.sample_rate_hz,
        "monitor_source": args.monitor_source,
        "mic_array_profile": args.mic_array_profile,
        "channel_map": [int(v) for v in args.channel_map.split(",") if v.strip()] if args.channel_map else None,
        "separation_mode": "mock",
    }

    session_id = _start_session(args.http_base, payload, stop_existing=bool(args.stop_existing))
    ws_url = f"{args.ws_base}/ws/session/{session_id}"
    print(f"session_id={session_id}")

    player = AudioPlayer(sample_rate_hz=args.sample_rate_hz, buffer_ms=args.buffer_ms)
    if args.wav_out:
        player.set_wav_output(args.wav_out)

    try:
        with sd.OutputStream(
            samplerate=args.sample_rate_hz,
            channels=1,
            dtype="float32",
            callback=player.callback,
        ):
            asyncio.run(_read_ws(ws_url, player))
    finally:
        player.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
