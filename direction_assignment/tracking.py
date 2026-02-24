from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DirectionAssignmentConfig
from .geometry import circular_diff_deg, circular_distance_deg, circular_mean_deg, normalize_angle_deg
from .types import SpeakerDirectionState


@dataclass
class SpeakerObservation:
    speaker_id: int
    doa_deg: float
    confidence: float


def snap_to_srp_peak(
    doa_deg: float,
    peaks_deg: list[float],
    peak_scores: list[float] | None,
    tolerance_deg: float,
) -> tuple[float, bool]:
    if not peaks_deg:
        return doa_deg, False

    peaks = np.asarray(peaks_deg, dtype=float)
    dists = np.asarray([circular_distance_deg(doa_deg, p) for p in peaks], dtype=float)
    valid = np.where(dists <= tolerance_deg)[0]
    if valid.size == 0:
        return doa_deg, False

    if peak_scores is not None and len(peak_scores) == len(peaks_deg):
        scores = np.asarray(peak_scores, dtype=float)[valid]
        idx = int(valid[int(np.argmax(scores))])
    else:
        idx = int(valid[int(np.argmin(dists[valid]))])

    return normalize_angle_deg(float(peaks[idx])), True


def aggregate_speaker_observations(
    observations: list[SpeakerObservation],
) -> tuple[dict[int, tuple[float, float]], dict]:
    by_speaker: dict[int, list[SpeakerObservation]] = {}
    for obs in observations:
        by_speaker.setdefault(obs.speaker_id, []).append(obs)

    out: dict[int, tuple[float, float]] = {}
    debug: dict[int, dict] = {}

    for sid, obs_list in by_speaker.items():
        angles = np.asarray([o.doa_deg for o in obs_list], dtype=float)
        weights = np.asarray([max(1e-6, o.confidence) for o in obs_list], dtype=float)
        mean_doa = circular_mean_deg(angles, weights)
        mean_conf = float(np.clip(np.average(weights), 0.0, 1.0))
        out[sid] = (mean_doa, mean_conf)
        debug[sid] = {"num_streams": len(obs_list)}

    return out, debug


def update_speaker_states(
    states: dict[int, SpeakerDirectionState],
    aggregated_obs: dict[int, tuple[float, float]],
    timestamp_ms: float,
    cfg: DirectionAssignmentConfig,
    srp_peaks_deg: list[float],
    srp_peak_scores: list[float] | None,
) -> tuple[dict[int, SpeakerDirectionState], dict]:
    snap_debug: dict[int, dict] = {}

    # Update from current observations.
    for sid, (raw_doa, conf) in aggregated_obs.items():
        snapped_doa, snapped = snap_to_srp_peak(
            doa_deg=raw_doa,
            peaks_deg=srp_peaks_deg,
            peak_scores=srp_peak_scores,
            tolerance_deg=cfg.srp_snap_tolerance_deg,
        )

        if sid not in states:
            states[sid] = SpeakerDirectionState(
                speaker_id=sid,
                direction_deg=snapped_doa,
                confidence=conf,
                last_update_ms=timestamp_ms,
                updates=1,
            )
        else:
            st = states[sid]
            diff = circular_diff_deg(snapped_doa, st.direction_deg)
            st.direction_deg = normalize_angle_deg(st.direction_deg + cfg.doa_ema_alpha * diff)
            st.confidence = float(np.clip((1.0 - cfg.doa_ema_alpha) * st.confidence + cfg.doa_ema_alpha * conf, 0.0, 1.0))
            st.last_update_ms = timestamp_ms
            st.updates += 1

        snap_debug[sid] = {
            "raw_doa": float(raw_doa),
            "snapped_doa": float(snapped_doa),
            "snapped": bool(snapped),
        }

    # Forget old states.
    forget_ids = [
        sid
        for sid, st in states.items()
        if (timestamp_ms - st.last_update_ms) > cfg.speaker_forget_timeout_ms
    ]
    for sid in forget_ids:
        del states[sid]

    stale_ids = [
        sid
        for sid, st in states.items()
        if (timestamp_ms - st.last_update_ms) > cfg.speaker_stale_timeout_ms
    ]

    debug = {
        "snap": snap_debug,
        "forgotten_speakers": forget_ids,
        "stale_speakers": stale_ids,
    }
    return states, debug
