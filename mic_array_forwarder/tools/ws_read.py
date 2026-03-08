#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request

import websockets

from mic_array_forwarder.ws_codec import decode_audio_chunk


def _start_session(http_base: str, payload: dict) -> str:
    url = f"{http_base}/api/session/start"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"start session failed: HTTP {exc.code} {detail}") from exc
    parsed = json.loads(body)
    session_id = parsed.get("session_id")
    if not session_id:
        raise RuntimeError(f"start session failed: missing session_id in response: {parsed}")
    return str(session_id)


async def _read_ws(ws_url: str) -> None:
    async with websockets.connect(ws_url, max_size=None) as ws:
        print(f"connected {ws_url}")
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                try:
                    ts_ms, audio = decode_audio_chunk(msg)
                    print(f"audio ts_ms={ts_ms:.0f} samples={audio.size}")
                except Exception as exc:
                    print(f"audio decode error: {exc}")
            else:
                try:
                    payload = json.loads(msg)
                except json.JSONDecodeError:
                    print(f"text: {msg}")
                else:
                    msg_type = payload.get("type", "unknown")
                    print(f"json {msg_type}: {payload}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read realtime mic array messages over WebSocket.")
    parser.add_argument("--http-base", default="http://localhost:8000", help="HTTP base URL")
    parser.add_argument("--ws-base", default="ws://localhost:8000", help="WebSocket base URL")
    parser.add_argument("--audio-device-query", default="ReSpeaker", help="Substring to match input device")
    parser.add_argument("--channel-count", type=int, default=4)
    parser.add_argument("--sample-rate-hz", type=int, default=16000)
    parser.add_argument("--monitor-source", choices=["processed", "raw_mixed"], default="processed")
    args = parser.parse_args()

    payload = {
        "input_source": "respeaker_live",
        "audio_device_query": args.audio_device_query,
        "channel_count": args.channel_count,
        "sample_rate_hz": args.sample_rate_hz,
        "monitor_source": args.monitor_source,
        "separation_mode": "mock",
    }

    session_id = _start_session(args.http_base, payload)
    ws_url = f"{args.ws_base}/ws/session/{session_id}"
    print(f"session_id={session_id}")
    asyncio.run(_read_ws(ws_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
