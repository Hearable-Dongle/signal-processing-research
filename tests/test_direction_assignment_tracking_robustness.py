from __future__ import annotations

from direction_assignment.config import DirectionAssignmentConfig
from direction_assignment.tracking import update_speaker_states
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
