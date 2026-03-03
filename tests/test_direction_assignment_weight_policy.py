from __future__ import annotations

import math

from direction_assignment.config import DirectionAssignmentConfig
from direction_assignment.types import SpeakerDirectionState
from direction_assignment.weight_policy import build_target_weights, db_to_linear


def _states() -> dict[int, SpeakerDirectionState]:
    return {
        1: SpeakerDirectionState(speaker_id=1, direction_deg=20.0, confidence=0.9, last_update_ms=0.0),
        2: SpeakerDirectionState(speaker_id=2, direction_deg=90.0, confidence=0.8, last_update_ms=0.0),
        3: SpeakerDirectionState(speaker_id=3, direction_deg=220.0, confidence=0.7, last_update_ms=0.0),
    }


def test_db_to_linear() -> None:
    assert math.isclose(db_to_linear(0.0), 1.0, rel_tol=1e-8)
    assert math.isclose(db_to_linear(6.0), 10.0 ** (6.0 / 20.0), rel_tol=1e-8)
    assert math.isclose(db_to_linear(-14.0), 10.0 ** (-14.0 / 20.0), rel_tol=1e-8)


def test_build_target_weights_no_focus_defaults_to_unity() -> None:
    cfg = DirectionAssignmentConfig()
    w = build_target_weights(_states(), [1, 2, 3], None, None, 0.0, cfg)
    assert w == [1.0, 1.0, 1.0]


def test_build_target_weights_focus_speakers_applies_db_policy() -> None:
    cfg = DirectionAssignmentConfig(focus_gain_db=0.0, non_focus_attenuation_db=-14.0)
    w = build_target_weights(_states(), [1, 2, 3], {2}, None, 6.0, cfg)
    assert math.isclose(w[1], db_to_linear(6.0), rel_tol=1e-6)
    assert math.isclose(w[0], db_to_linear(-14.0), rel_tol=1e-6)
    assert math.isclose(w[2], db_to_linear(-14.0), rel_tol=1e-6)


def test_build_target_weights_focus_direction_uses_nearest() -> None:
    cfg = DirectionAssignmentConfig(focus_gain_db=0.0, non_focus_attenuation_db=-14.0)
    w = build_target_weights(_states(), [1, 2, 3], None, 100.0, 3.0, cfg)
    assert math.isclose(w[1], db_to_linear(3.0), rel_tol=1e-6)
    assert math.isclose(w[0], db_to_linear(-14.0), rel_tol=1e-6)
    assert math.isclose(w[2], db_to_linear(-14.0), rel_tol=1e-6)

