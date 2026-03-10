from __future__ import annotations

from direction_assignment.config import DirectionAssignmentConfig
from direction_assignment.tracking import AggregatedSpeakerObservation, update_speaker_states
from direction_assignment.types import SpeakerDirectionState


def test_tracking_confidence_gate_skips_low_confidence_updates() -> None:
    cfg = DirectionAssignmentConfig(min_confidence_for_update=0.5)
    states = {
        1: SpeakerDirectionState(
            speaker_id=1,
            direction_deg=30.0,
            confidence=0.9,
            last_update_ms=10.0,
            updates=2,
        )
    }
    out, debug = update_speaker_states(
        states=states,
        aggregated_obs={1: (120.0, 0.2)},
        timestamp_ms=50.0,
        cfg=cfg,
        srp_peaks_deg=[],
        srp_peak_scores=None,
    )
    assert out[1].direction_deg == 30.0
    assert out[1].last_update_ms == 10.0
    assert 1 in debug["skipped_low_confidence_speakers"]


def test_tracking_max_angular_jump_limits_per_chunk_change() -> None:
    cfg = DirectionAssignmentConfig(doa_ema_alpha=1.0, max_angular_jump_deg_per_chunk=20.0)
    states = {
        2: SpeakerDirectionState(
            speaker_id=2,
            direction_deg=10.0,
            confidence=0.8,
            last_update_ms=0.0,
            updates=1,
        )
    }
    out, _ = update_speaker_states(
        states=states,
        aggregated_obs={2: (200.0, 0.9)},
        timestamp_ms=200.0,
        cfg=cfg,
        srp_peaks_deg=[],
        srp_peak_scores=None,
    )
    assert out[2].direction_deg == 350.0


def test_tracking_history_penalizes_large_unexpected_switch() -> None:
    cfg = DirectionAssignmentConfig(
        doa_ema_alpha=1.0,
        prediction_alpha=0.5,
        transition_penalty_deg=18.0,
        history_switch_penalty_deg=10.0,
        min_confidence_for_switch=0.8,
    )
    states = {
        3: SpeakerDirectionState(
            speaker_id=3,
            direction_deg=40.0,
            confidence=0.9,
            last_update_ms=0.0,
            updates=3,
            velocity_deg_per_chunk=4.0,
            recent_direction_history_deg=(32.0, 36.0, 40.0),
        )
    }
    out, debug = update_speaker_states(
        states=states,
        aggregated_obs={3: (130.0, 0.4)},
        timestamp_ms=200.0,
        cfg=cfg,
        srp_peaks_deg=[],
        srp_peak_scores=None,
    )
    assert out[3].direction_deg == 40.0
    assert 3 in debug["blocked_transition_speakers"]


def test_speaker_tracking_mode_requires_persistence_for_large_jump() -> None:
    cfg = DirectionAssignmentConfig(
        control_mode="speaker_tracking_mode",
        doa_ema_alpha=1.0,
        speaker_tracking_small_change_deg=10.0,
        speaker_tracking_medium_change_deg=20.0,
        speaker_tracking_large_change_persist_chunks=3,
        speaker_tracking_stable_confidence_threshold=0.5,
    )
    states = {
        4: SpeakerDirectionState(
            speaker_id=4,
            direction_deg=20.0,
            confidence=0.9,
            last_update_ms=0.0,
            updates=5,
            identity_confidence=0.8,
            identity_maturity="stable",
            activity_confidence=0.9,
        )
    }
    obs = AggregatedSpeakerObservation(
        doa_deg=120.0,
        doa_confidence=0.9,
        identity_confidence=0.8,
        activity_confidence=0.9,
        identity_maturity="stable",
    )
    out, debug = update_speaker_states(
        states=states,
        aggregated_obs={4: obs},
        timestamp_ms=200.0,
        cfg=cfg,
        srp_peaks_deg=[],
        srp_peak_scores=None,
    )
    assert out[4].direction_deg == 20.0
    assert 4 in debug["blocked_transition_speakers"]

    out, _ = update_speaker_states(
        states=out,
        aggregated_obs={4: obs},
        timestamp_ms=400.0,
        cfg=cfg,
        srp_peaks_deg=[],
        srp_peak_scores=None,
    )
    assert out[4].direction_deg == 20.0

    out, _ = update_speaker_states(
        states=out,
        aggregated_obs={4: obs},
        timestamp_ms=600.0,
        cfg=cfg,
        srp_peaks_deg=[],
        srp_peak_scores=None,
    )
    assert out[4].direction_deg == 120.0


def test_long_memory_anchor_builds_and_locks_from_consistent_observations() -> None:
    cfg = DirectionAssignmentConfig(
        control_mode="speaker_tracking_mode",
        doa_ema_alpha=1.0,
        long_memory_enabled=True,
        long_memory_min_observations=3,
        long_memory_anchor_lock_confidence=0.5,
    )
    states: dict[int, SpeakerDirectionState] = {}
    obs = AggregatedSpeakerObservation(
        doa_deg=28.0,
        doa_confidence=0.9,
        identity_confidence=0.8,
        activity_confidence=0.8,
        identity_maturity="stable",
    )
    for idx in range(3):
        states, _ = update_speaker_states(
            states=states,
            aggregated_obs={7: obs},
            timestamp_ms=float((idx + 1) * 200),
            cfg=cfg,
            srp_peaks_deg=[],
            srp_peak_scores=None,
        )
    assert states[7].anchor_direction_deg is not None
    assert abs(states[7].anchor_direction_deg - 28.0) < 1e-6
    assert states[7].anchor_locked
    assert states[7].anchor_confidence >= 0.5


def test_long_memory_anchor_blocks_single_spatial_jump() -> None:
    cfg = DirectionAssignmentConfig(
        control_mode="spatial_peak_mode",
        doa_ema_alpha=1.0,
        long_memory_enabled=True,
        long_memory_min_observations=1,
        long_memory_anchor_lock_confidence=0.5,
        long_memory_soft_prior_margin_deg=15.0,
        min_confidence_for_switch=0.95,
    )
    states = {
        8: SpeakerDirectionState(
            speaker_id=8,
            direction_deg=15.0,
            confidence=0.9,
            last_update_ms=0.0,
            updates=5,
            anchor_direction_deg=15.0,
            anchor_confidence=0.9,
            anchor_locked=True,
            anchor_last_confirmed_ms=0.0,
            anchor_observation_count=5,
            anchor_recent_observations_deg=(15.0, 15.0, 15.0),
        )
    }
    out, debug = update_speaker_states(
        states=states,
        aggregated_obs={8: (160.0, 0.4)},
        timestamp_ms=200.0,
        cfg=cfg,
        srp_peaks_deg=[],
        srp_peak_scores=None,
    )
    assert out[8].direction_deg == 15.0
    assert debug["snap"][8].get("blocked_by_anchor") is True or debug["snap"][8].get("blocked_transition") is True


def test_long_memory_anchor_preserves_state_beyond_default_forget_timeout() -> None:
    cfg = DirectionAssignmentConfig(
        long_memory_enabled=True,
        long_memory_stale_timeout_ms=60000.0,
        speaker_forget_timeout_ms=8000.0,
    )
    states = {
        9: SpeakerDirectionState(
            speaker_id=9,
            direction_deg=45.0,
            confidence=0.7,
            last_update_ms=0.0,
            updates=4,
            anchor_direction_deg=45.0,
            anchor_confidence=0.8,
            anchor_locked=True,
            anchor_last_confirmed_ms=0.0,
            anchor_observation_count=4,
            anchor_recent_observations_deg=(45.0, 45.0, 45.0, 45.0),
        )
    }
    out, debug = update_speaker_states(
        states=states,
        aggregated_obs={},
        timestamp_ms=30000.0,
        cfg=cfg,
        srp_peaks_deg=[],
        srp_peak_scores=None,
    )
    assert 9 in out
    assert 9 not in debug["forgotten_speakers"]
