from __future__ import annotations

from dataclasses import dataclass

from realtime_pipeline.contracts import SpeakerGainDirection
from realtime_pipeline.focus_sanity_check import (
    FocusControllerConfig,
    FocusLockController,
    compute_target_azimuth_from_scene,
    ratio_to_db,
)
from simulation.simulation_config import MicrophoneArray, Room, SimulationAudio, SimulationConfig, SimulationSource


class _FakeSharedState:
    def __init__(self) -> None:
        self._speaker_map: dict[int, SpeakerGainDirection] = {}

    def set_speaker_map(self, speaker_map: dict[int, SpeakerGainDirection]) -> None:
        self._speaker_map = dict(speaker_map)

    def get_speaker_map_snapshot(self) -> dict[int, SpeakerGainDirection]:
        return dict(self._speaker_map)


@dataclass
class _FocusCall:
    focused_speaker_ids: list[int] | None
    focused_direction_deg: float | None
    user_boost_db: float


class _FakePipe:
    def __init__(self) -> None:
        self.shared_state = _FakeSharedState()
        self.calls: list[_FocusCall] = []

    def set_focus_control(
        self,
        *,
        focused_speaker_ids: list[int] | None = None,
        focused_direction_deg: float | None = None,
        user_boost_db: float = 0.0,
    ) -> None:
        self.calls.append(
            _FocusCall(
                focused_speaker_ids=focused_speaker_ids,
                focused_direction_deg=focused_direction_deg,
                user_boost_db=float(user_boost_db),
            )
        )


def test_compute_target_azimuth_from_scene() -> None:
    scene = SimulationConfig(
        room=Room(dimensions=[5.0, 4.0, 3.0], absorption=0.25),
        microphone_array=MicrophoneArray(mic_center=[1.0, 1.0, 1.5], mic_radius=0.05, mic_count=4),
        audio=SimulationAudio(
            sources=[
                SimulationSource(loc=[2.0, 1.0, 1.5], audio_path="a.wav", gain=1.0, classification="speech"),
                SimulationSource(loc=[1.0, 2.0, 1.5], audio_path="b.wav", gain=1.0, classification="noise"),
            ],
            duration=1.0,
            fs=16000,
        ),
    )
    az = compute_target_azimuth_from_scene(scene)
    assert abs(az - 0.0) < 1e-6


def test_focus_controller_locks_on_stable_candidate() -> None:
    pipe = _FakePipe()
    ctrl = FocusLockController(
        FocusControllerConfig(
            target_azimuth_deg=90.0,
            boost_db=ratio_to_db(2.0),
            lock_tolerance_deg=20.0,
            lock_consecutive_hits=2,
            lock_timeout_ms=500.0,
        )
    )
    ctrl.initialize(pipe)  # direction mode

    speaker = SpeakerGainDirection(
        speaker_id=1,
        direction_degrees=95.0,
        gain_weight=2.0,
        confidence=0.9,
        active=True,
        activity_confidence=1.0,
        updated_at_ms=0.0,
    )
    pipe.shared_state.set_speaker_map({1: speaker, 2: SpeakerGainDirection(2, 240.0, 1.0, 0.8, True, 1.0, 0.0)})
    ctrl.on_frame(frame_idx=0, timestamp_ms=0.0, pipe=pipe)
    ctrl.on_frame(frame_idx=1, timestamp_ms=10.0, pipe=pipe)

    assert ctrl.locked_speaker_id == 1
    assert any(c.focused_speaker_ids == [1] for c in pipe.calls)


def test_focus_controller_reacquires_direction_after_timeout() -> None:
    pipe = _FakePipe()
    ctrl = FocusLockController(
        FocusControllerConfig(
            target_azimuth_deg=90.0,
            boost_db=ratio_to_db(2.0),
            lock_tolerance_deg=20.0,
            lock_consecutive_hits=1,
            lock_timeout_ms=100.0,
        )
    )
    ctrl.initialize(pipe)
    pipe.shared_state.set_speaker_map(
        {
            7: SpeakerGainDirection(7, 88.0, 2.0, 0.9, True, 1.0, 0.0),
            8: SpeakerGainDirection(8, 260.0, 1.0, 0.9, True, 1.0, 0.0),
        }
    )
    ctrl.on_frame(frame_idx=0, timestamp_ms=0.0, pipe=pipe)
    assert ctrl.locked_speaker_id == 7

    pipe.shared_state.set_speaker_map({})
    ctrl.on_frame(frame_idx=1, timestamp_ms=250.0, pipe=pipe)
    assert ctrl.locked_speaker_id is None
    assert ctrl.mode == "direction"

