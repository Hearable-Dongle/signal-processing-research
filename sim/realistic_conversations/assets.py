from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from general_utils.constants import LIBRIMIX_PATH

from .config import AssetManifest


@dataclass
class SpeechAsset:
    speaker_id: str
    path: Path


@dataclass
class NoiseAsset:
    category: str
    path: Path


@dataclass
class AssetLibrary:
    speech_by_speaker: dict[str, list[SpeechAsset]]
    noise_assets: list[NoiseAsset]

    def choose_speakers(self, rng: np.random.Generator, count: int) -> list[str]:
        speaker_ids = sorted(self.speech_by_speaker)
        if len(speaker_ids) < count:
            raise ValueError(f"Requested {count} speakers but only found {len(speaker_ids)}")
        order = rng.permutation(len(speaker_ids))
        return [speaker_ids[int(idx)] for idx in order[:count]]

    def choose_speech_path(self, rng: np.random.Generator, speaker_id: str) -> Path:
        candidates = self.speech_by_speaker.get(speaker_id, [])
        if not candidates:
            raise KeyError(f"No speech assets for speaker '{speaker_id}'")
        return candidates[int(rng.integers(0, len(candidates)))].path

    def choose_noise_path(self, rng: np.random.Generator) -> Path:
        if not self.noise_assets:
            raise ValueError("No noise assets available")
        return self.noise_assets[int(rng.integers(0, len(self.noise_assets)))].path


def _normalize_entries(manifest: AssetManifest) -> AssetLibrary:
    speech_by_speaker: dict[str, list[SpeechAsset]] = defaultdict(list)
    for item in manifest.speech:
        speaker_id = str(item["speaker_id"])
        speech_by_speaker[speaker_id].append(SpeechAsset(speaker_id=speaker_id, path=Path(item["path"]).resolve()))
    noise_assets = [
        NoiseAsset(category=str(item.get("category", "noise")), path=Path(item["path"]).resolve())
        for item in manifest.noise
    ]
    return AssetLibrary(speech_by_speaker=dict(speech_by_speaker), noise_assets=noise_assets)


def load_asset_library(manifest_path: str | Path | None = None) -> AssetLibrary:
    if manifest_path is not None:
        return _normalize_entries(AssetManifest.from_file(manifest_path))

    speech_paths = list((LIBRIMIX_PATH / "LibriSpeech").glob("**/*.flac"))
    if not speech_paths:
        speech_paths = list((LIBRIMIX_PATH / "LibriSpeech").glob("**/*.wav"))
    noise_paths = list((LIBRIMIX_PATH / "wham_noise").glob("**/*.wav"))
    if not speech_paths:
        raise FileNotFoundError(f"No speech assets found under {LIBRIMIX_PATH / 'LibriSpeech'}")
    if not noise_paths:
        raise FileNotFoundError(f"No noise assets found under {LIBRIMIX_PATH / 'wham_noise'}")

    speech_by_speaker: dict[str, list[SpeechAsset]] = defaultdict(list)
    for path in speech_paths:
        speaker_id = path.parent.parent.name if path.parent.parent.name else path.parent.name
        speech_by_speaker[speaker_id].append(SpeechAsset(speaker_id=speaker_id, path=path.resolve()))

    noise_assets = [NoiseAsset(category="noise", path=path.resolve()) for path in noise_paths]
    return AssetLibrary(speech_by_speaker=dict(speech_by_speaker), noise_assets=noise_assets)
