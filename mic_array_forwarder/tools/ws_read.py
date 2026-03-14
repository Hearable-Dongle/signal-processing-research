#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request

import websockets

from mic_array_forwarder.ws_codec import decode_audio_chunk
from realtime_pipeline.tracking_modes import TRACKING_MODE_CHOICES, validate_tracking_mode


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
    parser.add_argument("--sample-rate-hz", type=int, default=48000)
    parser.add_argument(
        "--mic-array-profile",
        choices=["respeaker_v3_0457", "respeaker_xvf3800_0650"],
        default="respeaker_xvf3800_0650",
    )
    parser.add_argument("--channel-map", default="", help="Comma-separated channel map, e.g. 0,1,2,3")
    parser.add_argument("--monitor-source", choices=["processed", "raw_mixed"], default="processed")
    parser.add_argument(
        "--localization-backend",
        choices=["srp_phat_legacy", "srp_phat_localization", "capon_1src", "capon_multisrc", "capon_mvdr_refine_1src", "music_1src"],
        default="srp_phat_localization",
    )
    parser.add_argument("--tracking-mode", choices=TRACKING_MODE_CHOICES, default="doa_centroid_v1")
    args = parser.parse_args()
    tracking_mode = validate_tracking_mode(str(args.tracking_mode))

    payload = {
        "input_source": "respeaker_live",
        "audio_device_query": args.audio_device_query,
        "channel_count": args.channel_count,
        "sample_rate_hz": args.sample_rate_hz,
        "monitor_source": args.monitor_source,
        "mic_array_profile": args.mic_array_profile,
        "localization_backend": args.localization_backend,
        "tracking_mode": tracking_mode,
        "channel_map": [int(v) for v in args.channel_map.split(",") if v.strip()] if args.channel_map else None,
        "separation_mode": "mock",
    }

    session_id = _start_session(args.http_base, payload)
    ws_url = f"{args.ws_base}/ws/session/{session_id}"
    print(f"session_id={session_id}")
    asyncio.run(_read_ws(ws_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
