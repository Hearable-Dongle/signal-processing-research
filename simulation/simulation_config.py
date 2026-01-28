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
    gain: float = 1.0
    classification: str = "signal"

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
    def from_dict(cls, data: dict) -> "SimulationConfig":
        """Loads configuration from a dictionary."""
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
                gain=s.get("gain", 1.0),
                classification=s.get("classification", "signal")
            )
            for s in data["audio"]["sources"]
        ]

        audio = SimulationAudio(
            sources=sources,
            duration=data["audio"]["duration"],
            fs=data["audio"]["fs"],
        )

        return cls(room=room, microphone_array=mic_array, audio=audio)

    def create_noise_config(self) -> "SimulationConfig":
        """Creates a new SimulationConfig containing only noise sources."""
        noise_sources = [s for s in self.audio.sources if s.classification != "signal"]
        
        new_audio = SimulationAudio(
            sources=noise_sources,
            duration=self.audio.duration,
            fs=self.audio.fs
        )
        
        return SimulationConfig(
            room=self.room,
            microphone_array=self.microphone_array,
            audio=new_audio
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "SimulationConfig":
        """Loads configuration from a JSON file."""
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)
        
        return cls.from_dict(data)

    def to_file(self, path: str | Path) -> None:
        """Saves configuration to a JSON file."""
        path = Path(path)
        
        data = {
            "room": {
                "dimensions": self.room.dimensions,
                "absorption": self.room.absorption,
            },
            "microphone_array": {
                "mic_center": self.microphone_array.mic_center,
                "mic_radius": self.microphone_array.mic_radius,
                "mic_count": self.microphone_array.mic_count,
            },
            "audio": {
                "sources": [
                    {
                        "loc": s.loc,
                        "audio": s.audio_path,
                        "gain": s.gain,
                    }
                    for s in self.audio.sources
                ],
                "duration": self.audio.duration,
                "fs": self.audio.fs,
            },
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("w") as f:
            json.dump(data, f, indent=4)
