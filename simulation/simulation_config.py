import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from general_utils.constants import LIBRIMIX_PATH


@dataclass
class Room:
    dimensions: List[float]
    absorption: float


@dataclass
class MicrophoneArray:
    mic_center: List[float]
    mic_radius: float
    mic_count: int


@dataclass
class SimulationSource:
    loc: List[float]
    audio_path: str

    def get_absolute_path(self) -> Path:
        """Returns the absolute path to the audio file."""
        return LIBRIMIX_PATH / self.audio_path


@dataclass
class SimulationAudio:
    sources: List[SimulationSource]
    duration: float
    fs: int


@dataclass
class SimulationConfig:
    room: Room
    microphone_array: MicrophoneArray
    audio: SimulationAudio

    @classmethod
    def from_file(cls, path: str | Path) -> "SimulationConfig":
        """Loads configuration from a JSON file."""
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        room = Room(
            dimensions=data["room"]["dimensions"],
            absorption=data["room"]["absorption"],
        )

        mic_array = MicrophoneArray(
            mic_center=data["microphone_array"]["mic_center"],
            mic_radius=data["microphone_array"]["mic_radius"],
            mic_count=data["microphone_array"]["mic_count"],
        )

        sources = [
            SimulationSource(
                loc=s["loc"],
                audio_path=s["audio"],
            )
            for s in data["audio"]["sources"]
        ]

        audio = SimulationAudio(
            sources=sources,
            duration=data["audio"]["duration"],
            fs=data["audio"]["fs"],
        )

        return cls(room=room, microphone_array=mic_array, audio=audio)
