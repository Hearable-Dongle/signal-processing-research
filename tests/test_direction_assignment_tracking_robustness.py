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

