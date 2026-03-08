from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RoomSpec:
    dimensions_m: list[float] = field(default_factory=lambda: [6.0, 5.0, 3.0])
    absorption: float = 0.28


@dataclass
class MicArraySpec:
    mic_center_m: list[float] = field(default_factory=lambda: [3.0, 2.5, 1.4])
    mic_radius_m: float = 0.045
    mic_count: int = 6


@dataclass
class TurnTakingSpec:
    min_speakers: int = 2
    max_speakers: int = 4
    utterance_sec_range: list[float] = field(default_factory=lambda: [1.1, 4.6])
    pause_sec_range: list[float] = field(default_factory=lambda: [0.12, 0.9])
    overlap_sec_range: list[float] = field(default_factory=lambda: [0.15, 0.4])
    overlap_probability: float = 0.38
    interruption_probability: float = 0.16
    persistence_probability: float = 0.58
    backchannel_probability: float = 0.12
    backchannel_sec_range: list[float] = field(default_factory=lambda: [0.22, 0.7])


@dataclass
class NoiseSpec:
    base_snr_db_range: list[float] = field(default_factory=lambda: [10.0, 22.0])
    transient_count_range: list[int] = field(default_factory=lambda: [1, 3])
    ambience_layers: list[str] = field(default_factory=lambda: ["hvac"])
    transient_types: list[str] = field(default_factory=lambda: ["door_click", "cough"])
    motion_update_sec: float = 1.0


@dataclass
class RenderSpec:
    duration_sec: float = 12.0
    sample_rate: int = 16000
    frame_ms: int = 20


@dataclass
class AssetManifest:
    speech: list[dict[str, str]]
    noise: list[dict[str, str]]

    @classmethod
    def from_file(cls, path: str | Path) -> "AssetManifest":
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            speech=[dict(item) for item in data.get("speech", [])],
            noise=[dict(item) for item in data.get("noise", [])],
        )


@dataclass
class ConversationScenarioConfig:
    preset: str
    room: RoomSpec = field(default_factory=RoomSpec)
    mic_array: MicArraySpec = field(default_factory=MicArraySpec)
    turn_taking: TurnTakingSpec = field(default_factory=TurnTakingSpec)
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    render: RenderSpec = field(default_factory=RenderSpec)
    moving_speaker: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    scene_name: str
    scene_dir: Path
    scene_config_path: Path
    metadata_path: Path
    frame_truth_path: Path
    metrics_path: Path


def _quiet_room() -> ConversationScenarioConfig:
    return ConversationScenarioConfig(
        preset="quiet_room",
        room=RoomSpec(dimensions_m=[5.5, 4.5, 2.8], absorption=0.34),
        mic_array=MicArraySpec(mic_center_m=[2.75, 2.25, 1.4], mic_radius_m=0.04, mic_count=4),
        turn_taking=TurnTakingSpec(
            utterance_sec_range=[1.4, 4.8],
            pause_sec_range=[0.2, 1.1],
            overlap_probability=0.28,
            interruption_probability=0.08,
            persistence_probability=0.64,
            backchannel_probability=0.08,
        ),
        noise=NoiseSpec(
            base_snr_db_range=[18.0, 28.0],
            transient_count_range=[0, 2],
            ambience_layers=["hvac"],
            transient_types=["door_click"],
        ),
    )


def _office() -> ConversationScenarioConfig:
    return ConversationScenarioConfig(
        preset="office",
        room=RoomSpec(dimensions_m=[7.2, 5.4, 3.0], absorption=0.22),
        mic_array=MicArraySpec(mic_center_m=[3.6, 2.7, 1.4], mic_radius_m=0.045, mic_count=6),
        noise=NoiseSpec(
            base_snr_db_range=[8.0, 18.0],
            transient_count_range=[1, 4],
            ambience_layers=["hvac", "keyboard"],
            transient_types=["door_click", "cough"],
        ),
    )


def _cafe() -> ConversationScenarioConfig:
    return ConversationScenarioConfig(
        preset="cafe",
        room=RoomSpec(dimensions_m=[8.5, 6.8, 3.2], absorption=0.18),
        mic_array=MicArraySpec(mic_center_m=[4.25, 3.4, 1.4], mic_radius_m=0.05, mic_count=6),
        turn_taking=TurnTakingSpec(
            utterance_sec_range=[1.0, 3.8],
            pause_sec_range=[0.1, 0.7],
            overlap_probability=0.42,
            interruption_probability=0.2,
            persistence_probability=0.54,
            backchannel_probability=0.16,
        ),
        noise=NoiseSpec(
            base_snr_db_range=[3.0, 14.0],
            transient_count_range=[2, 5],
            ambience_layers=["distant_chatter", "dishwasher"],
            transient_types=["dish_clink", "cough", "door_click"],
        ),
    )


def _moving_speaker() -> ConversationScenarioConfig:
    return ConversationScenarioConfig(
        preset="moving_speaker",
        room=RoomSpec(dimensions_m=[7.0, 5.2, 3.0], absorption=0.24),
        mic_array=MicArraySpec(mic_center_m=[3.5, 2.6, 1.4], mic_radius_m=0.045, mic_count=6),
        noise=NoiseSpec(
            base_snr_db_range=[7.0, 18.0],
            transient_count_range=[1, 3],
            ambience_layers=["hvac", "street"],
            transient_types=["door_click", "cough"],
            motion_update_sec=0.75,
        ),
        moving_speaker=True,
    )


def _noisy_home() -> ConversationScenarioConfig:
    return ConversationScenarioConfig(
        preset="noisy_home",
        room=RoomSpec(dimensions_m=[6.2, 5.8, 2.9], absorption=0.26),
        mic_array=MicArraySpec(mic_center_m=[3.1, 2.9, 1.35], mic_radius_m=0.045, mic_count=4),
        turn_taking=TurnTakingSpec(
            utterance_sec_range=[1.2, 4.2],
            pause_sec_range=[0.08, 0.8],
            overlap_probability=0.35,
            interruption_probability=0.18,
            persistence_probability=0.6,
            backchannel_probability=0.14,
        ),
        noise=NoiseSpec(
            base_snr_db_range=[4.0, 13.0],
            transient_count_range=[2, 6],
            ambience_layers=["hvac", "street", "keyboard"],
            transient_types=["dish_clink", "cough", "door_click"],
        ),
    )


PRESET_BUILDERS = {
    "quiet_room": _quiet_room,
    "office": _office,
    "cafe": _cafe,
    "moving_speaker": _moving_speaker,
    "noisy_home": _noisy_home,
}


def build_preset(name: str) -> ConversationScenarioConfig:
    if name not in PRESET_BUILDERS:
        raise KeyError(f"Unknown preset '{name}'. Expected one of {sorted(PRESET_BUILDERS)}")
    return PRESET_BUILDERS[name]()
