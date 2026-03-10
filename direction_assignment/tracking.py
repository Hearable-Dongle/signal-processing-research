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
    identity_confidence: float = 0.0
    activity_confidence: float = 0.0
    identity_maturity: str = "unknown"
    last_separator_stream_index: int | None = None


@dataclass
class AggregatedSpeakerObservation:
    doa_deg: float
    doa_confidence: float
    identity_confidence: float = 0.0
    activity_confidence: float = 0.0
    identity_maturity: str = "unknown"
    last_separator_stream_index: int | None = None


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
) -> tuple[dict[int, AggregatedSpeakerObservation], dict]:
    by_speaker: dict[int, list[SpeakerObservation]] = {}
    for obs in observations:
        by_speaker.setdefault(obs.speaker_id, []).append(obs)

    out: dict[int, AggregatedSpeakerObservation] = {}
    debug: dict[int, dict] = {}

    for sid, obs_list in by_speaker.items():
        angles = np.asarray([o.doa_deg for o in obs_list], dtype=float)
        doa_weights = np.asarray([max(1e-6, o.confidence) for o in obs_list], dtype=float)
        mean_doa = circular_mean_deg(angles, doa_weights)
        mean_conf = float(np.clip(np.average(doa_weights), 0.0, 1.0))
        identity_conf = float(
            np.clip(np.average([max(0.0, o.identity_confidence) for o in obs_list], weights=doa_weights), 0.0, 1.0)
        )
        activity_conf = float(
            np.clip(np.average([max(0.0, o.activity_confidence) for o in obs_list], weights=doa_weights), 0.0, 1.0)
        )
        maturity = "stable" if any(o.identity_maturity == "stable" for o in obs_list) else obs_list[0].identity_maturity
        out[sid] = AggregatedSpeakerObservation(
            doa_deg=mean_doa,
            doa_confidence=mean_conf,
            identity_confidence=identity_conf,
            activity_confidence=activity_conf,
            identity_maturity=maturity,
            last_separator_stream_index=obs_list[-1].last_separator_stream_index,
        )
        debug[sid] = {
            "num_streams": len(obs_list),
            "identity_confidence": identity_conf,
            "activity_confidence": activity_conf,
            "identity_maturity": maturity,
        }

    return out, debug


def _normalize_obs_value(
    obs: tuple[float, float] | AggregatedSpeakerObservation,
) -> AggregatedSpeakerObservation:
    if isinstance(obs, AggregatedSpeakerObservation):
        return obs
    doa_deg, conf = obs
    return AggregatedSpeakerObservation(doa_deg=float(doa_deg), doa_confidence=float(conf))


def _effective_observation_confidence(obs: AggregatedSpeakerObservation) -> float:
    return float(
        np.clip(
            (0.65 * float(obs.doa_confidence))
            + (0.2 * float(obs.identity_confidence))
            + (0.15 * float(obs.activity_confidence)),
            0.0,
            1.0,
        )
    )


def _update_tracking_metadata(state: SpeakerDirectionState, obs: AggregatedSpeakerObservation) -> None:
    state.identity_confidence = float(np.clip(obs.identity_confidence, 0.0, 1.0))
    state.identity_maturity = str(obs.identity_maturity or state.identity_maturity)
    state.activity_confidence = float(np.clip(obs.activity_confidence, 0.0, 1.0))
    state.last_separator_stream_index = obs.last_separator_stream_index


def _has_anchor(state: SpeakerDirectionState, cfg: DirectionAssignmentConfig) -> bool:
    return bool(
        cfg.long_memory_enabled
        and state.anchor_direction_deg is not None
        and state.anchor_confidence >= cfg.min_persist_confidence
    )


def _anchor_limit(cfg: DirectionAssignmentConfig) -> int:
    chunk_ms = max(float(cfg.chunk_ms), 1.0)
    return max(1, int(round(float(cfg.long_memory_window_ms) / chunk_ms)))


def _blend_with_anchor(direction_deg: float, state: SpeakerDirectionState, cfg: DirectionAssignmentConfig) -> float:
    if not _has_anchor(state, cfg):
        return normalize_angle_deg(direction_deg)
    anchor = float(state.anchor_direction_deg)
    anchor_weight = float(np.clip(0.25 + 0.5 * state.anchor_confidence, 0.0, 0.8))
    vec = np.array(
        [
            (1.0 - anchor_weight) * np.cos(np.deg2rad(direction_deg)) + anchor_weight * np.cos(np.deg2rad(anchor)),
            (1.0 - anchor_weight) * np.sin(np.deg2rad(direction_deg)) + anchor_weight * np.sin(np.deg2rad(anchor)),
        ],
        dtype=np.float64,
    )
    if float(np.linalg.norm(vec)) < 1e-12:
        return normalize_angle_deg(direction_deg)
    return normalize_angle_deg(np.degrees(np.arctan2(vec[1], vec[0])))


def _update_anchor_state(
    state: SpeakerDirectionState,
    obs: AggregatedSpeakerObservation,
    timestamp_ms: float,
    cfg: DirectionAssignmentConfig,
    observed_doa: float,
) -> dict:
    debug: dict = {}
    if not cfg.long_memory_enabled:
        return debug

    obs_conf = _effective_observation_confidence(obs)
    state.anchor_confidence = float(np.clip(state.anchor_confidence * cfg.long_memory_decay, 0.0, 1.0))
    if state.anchor_direction_deg is None:
        state.anchor_direction_deg = float(observed_doa)
        state.anchor_confidence = float(obs_conf)
        state.anchor_last_confirmed_ms = float(timestamp_ms)
        state.anchor_observation_count = 1
        state.anchor_recent_observations_deg = (float(observed_doa),)
        state.anchor_locked = bool(
            state.anchor_observation_count >= max(1, int(cfg.long_memory_min_observations))
            and state.anchor_confidence >= float(cfg.long_memory_anchor_lock_confidence)
        )
        debug["anchor_update"] = "initialized"
        return debug

    if obs_conf < max(float(cfg.min_confidence_for_update), 0.35):
        debug["anchor_update"] = "skipped_low_conf"
        return debug

    distance_from_anchor = circular_distance_deg(float(observed_doa), float(state.anchor_direction_deg))
    if distance_from_anchor <= float(cfg.long_memory_max_anchor_spread_deg) or not state.anchor_locked:
        hist = list(state.anchor_recent_observations_deg)
        hist.append(float(observed_doa))
        hist = hist[-_anchor_limit(cfg) :]
        state.anchor_recent_observations_deg = tuple(hist)
        weights = np.linspace(0.5, 1.0, num=len(hist), dtype=np.float64)
        state.anchor_direction_deg = float(circular_mean_deg(np.asarray(hist, dtype=float), weights))
        state.anchor_confidence = float(
            np.clip((0.85 * state.anchor_confidence) + (0.15 * obs_conf), 0.0, 1.0)
        )
        state.anchor_last_confirmed_ms = float(timestamp_ms)
        state.anchor_observation_count += 1
        state.anchor_locked = bool(
            state.anchor_observation_count >= max(1, int(cfg.long_memory_min_observations))
            and state.anchor_confidence >= float(cfg.long_memory_anchor_lock_confidence)
        )
        state.anchor_relock_candidate_deg = None
        state.anchor_relock_count = 0
        debug["anchor_update"] = "confirmed"
        debug["anchor_distance_deg"] = float(distance_from_anchor)
        return debug

    same_candidate = (
        state.anchor_relock_candidate_deg is not None
        and circular_distance_deg(float(state.anchor_relock_candidate_deg), float(observed_doa))
        <= float(cfg.speaker_tracking_small_change_deg)
    )
    state.anchor_relock_candidate_deg = float(observed_doa)
    state.anchor_relock_count = int(state.anchor_relock_count + 1) if same_candidate else 1
    debug["anchor_update"] = "pending_relock"
    debug["anchor_distance_deg"] = float(distance_from_anchor)
    debug["anchor_relock_count"] = int(state.anchor_relock_count)
    if state.anchor_relock_count >= max(1, int(cfg.long_memory_relock_persist_chunks)):
        state.anchor_direction_deg = float(observed_doa)
        state.anchor_last_confirmed_ms = float(timestamp_ms)
        state.anchor_confidence = float(
            np.clip(max(state.anchor_confidence, obs_conf), 0.0, 1.0)
        )
        state.anchor_observation_count += 1
        state.anchor_locked = True
        state.anchor_recent_observations_deg = tuple(
            list(state.anchor_recent_observations_deg)[-_anchor_limit(cfg) + 1 :] + [float(observed_doa)]
        )
        state.anchor_relock_candidate_deg = None
        state.anchor_relock_count = 0
        debug["anchor_update"] = "relocked"
    return debug


def _init_state(
    sid: int,
    timestamp_ms: float,
    doa_deg: float,
    obs: AggregatedSpeakerObservation,
) -> SpeakerDirectionState:
    return SpeakerDirectionState(
        speaker_id=sid,
        direction_deg=doa_deg,
        confidence=float(np.clip(obs.doa_confidence, 0.0, 1.0)),
        last_update_ms=timestamp_ms,
        updates=1,
        last_observed_ms=timestamp_ms,
        last_raw_direction_deg=float(obs.doa_deg),
        velocity_deg_per_chunk=0.0,
        recent_direction_history_deg=(float(doa_deg),),
        predicted_direction_deg=float(doa_deg),
        identity_confidence=float(np.clip(obs.identity_confidence, 0.0, 1.0)),
        identity_maturity=str(obs.identity_maturity),
        activity_confidence=float(np.clip(obs.activity_confidence, 0.0, 1.0)),
        last_separator_stream_index=obs.last_separator_stream_index,
        anchor_direction_deg=float(doa_deg),
        anchor_confidence=float(np.clip(_effective_observation_confidence(obs), 0.0, 1.0)),
        anchor_last_confirmed_ms=timestamp_ms,
        anchor_observation_count=1,
        anchor_locked=False,
        anchor_recent_observations_deg=(float(doa_deg),),
    )


def _update_state_spatial_peak_mode(
    state: SpeakerDirectionState,
    obs: AggregatedSpeakerObservation,
    timestamp_ms: float,
    cfg: DirectionAssignmentConfig,
    snapped_doa: float,
) -> tuple[bool, dict]:
    predicted_velocity = float(
        np.clip(state.velocity_deg_per_chunk, -abs(cfg.max_angular_velocity_deg_per_chunk), abs(cfg.max_angular_velocity_deg_per_chunk))
    )
    predicted_doa = _blend_with_anchor(normalize_angle_deg(state.direction_deg + predicted_velocity), state, cfg)
    predicted_diff = circular_diff_deg(snapped_doa, predicted_doa)
    debug = {
        "predicted_doa": float(predicted_doa),
        "velocity_deg_per_chunk": float(predicted_velocity),
        "mode": "spatial_peak_mode",
    }
    if _has_anchor(state, cfg):
        debug["anchor_direction_deg"] = float(state.anchor_direction_deg)
        debug["anchor_confidence"] = float(state.anchor_confidence)
    transition_bypass = bool(
        state.updates < 2
        and len(state.recent_direction_history_deg) < 2
    )
    if (
        not transition_bypass
        and abs(predicted_diff) > (cfg.transition_penalty_deg + cfg.history_switch_penalty_deg)
        and obs.doa_confidence < cfg.min_confidence_for_switch
    ):
        state.confidence = float(np.clip(state.confidence * cfg.hold_confidence_decay, 0.0, 1.0))
        state.hold_count += 1
        state.last_raw_direction_deg = float(obs.doa_deg)
        state.last_observed_ms = timestamp_ms
        state.predicted_direction_deg = float(predicted_doa)
        _update_tracking_metadata(state, obs)
        debug["blocked_transition"] = True
        return False, debug
    if (
        _has_anchor(state, cfg)
        and circular_distance_deg(snapped_doa, float(state.anchor_direction_deg)) > float(cfg.long_memory_soft_prior_margin_deg)
        and obs.doa_confidence < cfg.min_confidence_for_switch
    ):
        state.confidence = float(np.clip(state.confidence * cfg.hold_confidence_decay, 0.0, 1.0))
        state.hold_count += 1
        state.last_raw_direction_deg = float(obs.doa_deg)
        state.last_observed_ms = timestamp_ms
        state.predicted_direction_deg = float(predicted_doa)
        _update_tracking_metadata(state, obs)
        debug["blocked_by_anchor"] = True
        return False, debug

    blended_target = normalize_angle_deg(
        predicted_doa + (1.0 - cfg.prediction_alpha) * circular_diff_deg(snapped_doa, predicted_doa)
    )
    diff = circular_diff_deg(blended_target, state.direction_deg)
    if cfg.max_angular_jump_deg_per_chunk is not None:
        diff = float(np.clip(diff, -abs(cfg.max_angular_jump_deg_per_chunk), abs(cfg.max_angular_jump_deg_per_chunk)))
    prev_direction = float(state.direction_deg)
    state.direction_deg = normalize_angle_deg(state.direction_deg + cfg.doa_ema_alpha * diff)
    state.confidence = float(np.clip((1.0 - cfg.doa_ema_alpha) * state.confidence + cfg.doa_ema_alpha * obs.doa_confidence, 0.0, 1.0))
    state.predicted_direction_deg = float(predicted_doa)
    observed_velocity = circular_diff_deg(state.direction_deg, prev_direction)
    state.velocity_deg_per_chunk = float(
        np.clip(
            (cfg.prediction_alpha * predicted_velocity) + ((1.0 - cfg.prediction_alpha) * observed_velocity),
            -abs(cfg.max_angular_velocity_deg_per_chunk),
            abs(cfg.max_angular_velocity_deg_per_chunk),
        )
    )
    state.large_deviation_count = 0
    state.pending_large_deviation_deg = None
    _update_tracking_metadata(state, obs)
    return True, debug


def _update_state_speaker_tracking_mode(
    state: SpeakerDirectionState,
    obs: AggregatedSpeakerObservation,
    timestamp_ms: float,
    cfg: DirectionAssignmentConfig,
    snapped_doa: float,
) -> tuple[bool, dict]:
    predicted_velocity = float(
        np.clip(state.velocity_deg_per_chunk, -abs(cfg.max_angular_velocity_deg_per_chunk), abs(cfg.max_angular_velocity_deg_per_chunk))
    )
    predicted_doa = _blend_with_anchor(normalize_angle_deg(state.direction_deg + predicted_velocity), state, cfg)
    distance_from_prediction = abs(circular_diff_deg(snapped_doa, predicted_doa))
    direction_conf = float(np.clip(state.confidence, 0.0, 1.0))
    stable_state = (
        str(state.identity_maturity) == "stable"
        and direction_conf >= float(cfg.speaker_tracking_stable_confidence_threshold)
    )
    debug = {
        "predicted_doa": float(predicted_doa),
        "distance_from_prediction_deg": float(distance_from_prediction),
        "velocity_deg_per_chunk": float(predicted_velocity),
        "mode": "speaker_tracking_mode",
    }
    if _has_anchor(state, cfg):
        debug["anchor_direction_deg"] = float(state.anchor_direction_deg)
        debug["anchor_confidence"] = float(state.anchor_confidence)
        debug["distance_from_anchor_deg"] = float(circular_distance_deg(snapped_doa, float(state.anchor_direction_deg)))

    if distance_from_prediction <= float(cfg.speaker_tracking_small_change_deg):
        effective_alpha = float(cfg.doa_ema_alpha)
        target_doa = float(snapped_doa)
        clip_jump = True
        state.large_deviation_count = 0
        state.pending_large_deviation_deg = None
        debug["adaptation"] = "fast"
    elif distance_from_prediction <= float(cfg.speaker_tracking_medium_change_deg):
        effective_alpha = float(np.clip(0.5 * cfg.doa_ema_alpha, 0.05, 1.0))
        target_doa = normalize_angle_deg(
            predicted_doa + (1.0 - cfg.prediction_alpha) * circular_diff_deg(snapped_doa, predicted_doa)
        )
        clip_jump = True
        state.large_deviation_count = 0
        state.pending_large_deviation_deg = None
        debug["adaptation"] = "smoothed"
    else:
        same_pending = (
            state.pending_large_deviation_deg is not None
            and circular_distance_deg(float(state.pending_large_deviation_deg), snapped_doa) <= float(cfg.speaker_tracking_small_change_deg)
        )
        state.large_deviation_count = int(state.large_deviation_count + 1) if same_pending else 1
        state.pending_large_deviation_deg = float(snapped_doa)
        persistence_required = max(1, int(cfg.speaker_tracking_large_change_persist_chunks))
        allow_relock = (
            not stable_state
            or state.large_deviation_count >= persistence_required
            or obs.identity_confidence >= float(state.identity_confidence + cfg.speaker_tracking_identity_hold_margin)
        )
        if (
            allow_relock
            and _has_anchor(state, cfg)
            and circular_distance_deg(snapped_doa, float(state.anchor_direction_deg)) > float(cfg.long_memory_soft_prior_margin_deg)
        ):
            allow_relock = state.anchor_relock_count >= max(1, int(cfg.long_memory_relock_persist_chunks))
        if not allow_relock:
            state.confidence = float(np.clip(state.confidence * cfg.hold_confidence_decay, 0.0, 1.0))
            state.hold_count += 1
            state.last_raw_direction_deg = float(obs.doa_deg)
            state.last_observed_ms = timestamp_ms
            state.predicted_direction_deg = float(predicted_doa)
            _update_tracking_metadata(state, obs)
            debug["blocked_large_change"] = True
            debug["large_deviation_count"] = int(state.large_deviation_count)
            return False, debug
        effective_alpha = 1.0
        target_doa = float(snapped_doa)
        clip_jump = False
        debug["adaptation"] = "relock"
        debug["large_deviation_count"] = int(state.large_deviation_count)

    diff = circular_diff_deg(target_doa, state.direction_deg)
    if clip_jump and cfg.max_angular_jump_deg_per_chunk is not None:
        diff = float(np.clip(diff, -abs(cfg.max_angular_jump_deg_per_chunk), abs(cfg.max_angular_jump_deg_per_chunk)))
    prev_direction = float(state.direction_deg)
    state.direction_deg = normalize_angle_deg(state.direction_deg + effective_alpha * diff)
    state.predicted_direction_deg = float(predicted_doa)
    obs_conf = _effective_observation_confidence(obs)
    state.confidence = float(np.clip((1.0 - effective_alpha) * state.confidence + effective_alpha * obs_conf, 0.0, 1.0))
    observed_velocity = circular_diff_deg(state.direction_deg, prev_direction)
    state.velocity_deg_per_chunk = float(
        np.clip(
            (cfg.prediction_alpha * predicted_velocity) + ((1.0 - cfg.prediction_alpha) * observed_velocity),
            -abs(cfg.max_angular_velocity_deg_per_chunk),
            abs(cfg.max_angular_velocity_deg_per_chunk),
        )
    )
    if debug.get("adaptation") == "relock":
        state.large_deviation_count = 0
        state.pending_large_deviation_deg = None
    _update_tracking_metadata(state, obs)
    return True, debug


def update_speaker_states(
    states: dict[int, SpeakerDirectionState],
    aggregated_obs: dict[int, tuple[float, float] | AggregatedSpeakerObservation],
    timestamp_ms: float,
    cfg: DirectionAssignmentConfig,
    srp_peaks_deg: list[float],
    srp_peak_scores: list[float] | None,
) -> tuple[dict[int, SpeakerDirectionState], dict]:
    snap_debug: dict[int, dict] = {}
    skipped_low_confidence: list[int] = []
    held_low_confidence: list[int] = []
    blocked_transitions: list[int] = []
    updated_ids: set[int] = set()
    normalized_obs = {sid: _normalize_obs_value(obs) for sid, obs in aggregated_obs.items()}

    for sid, obs in normalized_obs.items():
        if obs.doa_confidence < cfg.min_confidence_for_update:
            skipped_low_confidence.append(int(sid))
            if sid in states:
                st = states[sid]
                st.confidence = float(np.clip(st.confidence * cfg.hold_confidence_decay, 0.0, 1.0))
                st.hold_count += 1
                st.last_raw_direction_deg = float(obs.doa_deg)
                st.predicted_direction_deg = float(st.direction_deg + st.velocity_deg_per_chunk)
                _update_tracking_metadata(st, obs)
                held_low_confidence.append(int(sid))
            continue

        snapped_doa, snapped = snap_to_srp_peak(
            doa_deg=obs.doa_deg,
            peaks_deg=srp_peaks_deg,
            peak_scores=srp_peak_scores,
            tolerance_deg=cfg.srp_snap_tolerance_deg,
        )

        if sid not in states:
            states[sid] = _init_state(sid, timestamp_ms, snapped_doa, obs)
            updated_ids.add(int(sid))
            snap_debug[sid] = {
                "raw_doa": float(obs.doa_deg),
                "snapped_doa": float(snapped_doa),
                "snapped": bool(snapped),
                "predicted_doa": float(snapped_doa),
                "velocity_deg_per_chunk": 0.0,
                "mode": str(cfg.control_mode),
                "adaptation": "new",
                "anchor_update": "initialized",
            }
            continue

        st = states[sid]
        if str(cfg.control_mode) == "speaker_tracking_mode":
            updated, debug = _update_state_speaker_tracking_mode(st, obs, timestamp_ms, cfg, snapped_doa)
        else:
            updated, debug = _update_state_spatial_peak_mode(st, obs, timestamp_ms, cfg, snapped_doa)
        st.last_update_ms = timestamp_ms if updated else st.last_update_ms
        st.last_observed_ms = timestamp_ms
        st.last_raw_direction_deg = float(obs.doa_deg)
        anchor_debug: dict = {}
        if updated:
            st.updates += 1
            st.hold_count = 0
            st.stale_updates = 0
            hist = list(st.recent_direction_history_deg)
            hist.append(float(st.direction_deg))
            st.recent_direction_history_deg = tuple(hist[-max(1, int(cfg.history_window_chunks)) :])
            anchor_debug = _update_anchor_state(st, obs, timestamp_ms, cfg, snapped_doa)
            updated_ids.add(int(sid))
        else:
            blocked_transitions.append(int(sid))
        snap_debug[sid] = {
            "raw_doa": float(obs.doa_deg),
            "snapped_doa": float(snapped_doa),
            "snapped": bool(snapped),
            **debug,
            **anchor_debug,
        }

    for sid, st in list(states.items()):
        if sid in updated_ids:
            continue
        age_ms = float(timestamp_ms - st.last_update_ms)
        st.predicted_direction_deg = _blend_with_anchor(
            normalize_angle_deg(st.direction_deg + st.velocity_deg_per_chunk),
            st,
            cfg,
        )
        if _has_anchor(st, cfg) and st.anchor_locked and age_ms <= float(cfg.long_memory_stale_timeout_ms):
            anchor_diff = circular_diff_deg(float(st.anchor_direction_deg), float(st.direction_deg))
            st.direction_deg = normalize_angle_deg(float(st.direction_deg) + (0.08 * anchor_diff))
        if age_ms > cfg.speaker_stale_timeout_ms:
            st.confidence = float(np.clip(st.confidence * cfg.stale_confidence_decay, 0.0, 1.0))
            st.stale_updates += 1

    forget_ids = [
        sid
        for sid, st in states.items()
        if (
            (timestamp_ms - st.last_update_ms)
            > max(
                float(cfg.speaker_forget_timeout_ms),
                float(cfg.long_memory_stale_timeout_ms) if _has_anchor(st, cfg) else 0.0,
            )
            or (st.confidence < cfg.min_persist_confidence and not _has_anchor(st, cfg))
        )
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
        "skipped_low_confidence_speakers": skipped_low_confidence,
        "held_low_confidence_speakers": held_low_confidence,
        "blocked_transition_speakers": blocked_transitions,
        "forgotten_speakers": forget_ids,
        "stale_speakers": stale_ids,
    }
    return states, debug
