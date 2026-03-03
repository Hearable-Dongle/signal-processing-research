from __future__ import annotations

from typing import Any

import numpy as np


def _as_int_or_none(v: Any) -> int | None:
    if v == "" or v is None:
        return None
    return int(v)


def compute_catchup_metrics(observations: list[dict[str, Any]], stable_frames: int = 3) -> dict[str, Any]:
    """Compute lock/catchup metrics from frame-level observation rows."""
    if not observations:
        return {
            "startup_lock_ms": 0.0,
            "reacquire_catchup_ms_median": 0.0,
            "reacquire_catchup_count": 0,
            "nearest_change_catchup_ms_median": 0.0,
            "nearest_change_events": 0,
            "nearest_change_caught": 0,
        }

    rows = sorted(observations, key=lambda r: int(r["frame_idx"]))
    startup_lock_ms = 0.0
    for r in rows:
        if _as_int_or_none(r.get("locked_speaker_id")) is not None:
            startup_lock_ms = float(r["timestamp_ms"])
            break

    nearest_change_events = 0
    nearest_change_caught = 0
    nearest_catchups: list[float] = []
    stable_nearest: int | None = None
    pending_target: int | None = None
    pending_t0: float | None = None

    for idx, r in enumerate(rows):
        if idx + 1 < stable_frames:
            continue
        window = rows[idx - stable_frames + 1 : idx + 1]
        nearest_vals = [_as_int_or_none(w.get("nearest_speaker_id")) for w in window]
        if nearest_vals[0] is None:
            continue
        if not all(v == nearest_vals[0] for v in nearest_vals):
            continue
        candidate = nearest_vals[0]
        if candidate is None:
            continue

        if stable_nearest is None:
            stable_nearest = candidate
        elif candidate != stable_nearest:
            stable_nearest = candidate
            nearest_change_events += 1
            pending_target = candidate
            pending_t0 = float(r["timestamp_ms"])

        if pending_target is not None and pending_t0 is not None:
            locked = _as_int_or_none(r.get("locked_speaker_id"))
            if locked == pending_target:
                nearest_change_caught += 1
                nearest_catchups.append(max(0.0, float(r["timestamp_ms"]) - pending_t0))
                pending_target = None
                pending_t0 = None

    return {
        "startup_lock_ms": float(startup_lock_ms),
        "reacquire_catchup_ms_median": 0.0,
        "reacquire_catchup_count": 0,
        "nearest_change_catchup_ms_median": float(np.median(nearest_catchups)) if nearest_catchups else 0.0,
        "nearest_change_events": int(nearest_change_events),
        "nearest_change_caught": int(nearest_change_caught),
    }
