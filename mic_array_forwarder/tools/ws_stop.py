#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request


def _http_json(method: str, url: str) -> dict:
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop the active mic array session.")
    parser.add_argument("--http-base", default="http://localhost:8000", help="HTTP base URL")
    args = parser.parse_args()

    active = _http_json("GET", f"{args.http_base}/api/session/active")
    session_id = active.get("session_id")
    if not session_id:
        print("No active session.")
        return 0

    stopped = _http_json("POST", f"{args.http_base}/api/session/active/stop")
    print(f"Stopped session {stopped.get('session_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
