from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from speaker_identity_grouping import IdentityChunkInput, IdentityChunkOutput, IdentityConfig, SpeakerIdentityGrouper
from speaker_identity_grouping.contracts import SpeakerState


@dataclass
class IdentityBridgeConfig:
    sample_rate_hz: int = 16000
    chunk_duration_ms: int = 200
    identity_vad_rms: float = 0.01
    match_threshold: float = 0.82
    ema_alpha: float = 0.1
    max_speakers: int = 16
    retire_after_chunks: int = 25
    new_speaker_confidence: float = 0.5
    identity_mode: str = "online_only"  # online_only|enroll_audio|seed_embeddings
    enroll_audio_manifest: str | None = None
    seed_embeddings_npz: str | None = None


class IdentityBridge:
    def __init__(self, cfg: IdentityBridgeConfig):
        self.cfg = cfg
        self.grouper = SpeakerIdentityGrouper(
            IdentityConfig(
                sample_rate_hz=cfg.sample_rate_hz,
                chunk_duration_ms=cfg.chunk_duration_ms,
                vad_rms_threshold=cfg.identity_vad_rms,
                match_threshold=cfg.match_threshold,
                ema_alpha=cfg.ema_alpha,
                max_speakers=cfg.max_speakers,
                retire_after_chunks=cfg.retire_after_chunks,
                new_speaker_confidence=cfg.new_speaker_confidence,
            )
        )
        self.bootstrap_debug: dict[str, object] = {}

    def maybe_seed(self, synthetic_oracle_sources: np.ndarray | None = None) -> None:
        mode = self.cfg.identity_mode
        if mode == "online_only":
            return
        if mode == "enroll_audio":
            self._seed_from_enroll_audio(synthetic_oracle_sources)
            return
        if mode == "seed_embeddings":
            self._seed_from_npz()
            return
        raise ValueError(f"Unsupported identity_mode: {mode}")

    def update(
        self,
        chunk_id: int,
        timestamp_ms: float,
        streams: list[np.ndarray],
    ) -> IdentityChunkOutput:
        return self.grouper.update(
            IdentityChunkInput(
                chunk_id=chunk_id,
                timestamp_ms=timestamp_ms,
                sample_rate_hz=self.cfg.sample_rate_hz,
                streams=streams,
            )
        )

    def _seed_state(self, speaker_id: int, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=float)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-8:
            emb = emb / norm

        self.grouper._speakers[int(speaker_id)] = SpeakerState(
            speaker_id=int(speaker_id),
            centroid=emb,
            sample_count=1,
            last_seen_chunk=-1,
            last_seen_timestamp_ms=-1.0,
        )
        self.grouper._next_speaker_id = max(self.grouper._next_speaker_id, int(speaker_id) + 1)

    def _seed_from_npz(self) -> None:
        path = self.cfg.seed_embeddings_npz
        if not path:
            raise ValueError("identity_mode=seed_embeddings requires --seed-embeddings-npz")

        data = np.load(path)
        seeded = []
        for key in data.files:
            sid = int(key)
            emb = np.asarray(data[key], dtype=float).reshape(-1)
            self._seed_state(sid, emb)
            seeded.append(sid)
        self.bootstrap_debug = {"seed_mode": "seed_embeddings", "seeded_speaker_ids": sorted(seeded)}

    def _read_mono(self, path: Path) -> np.ndarray:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != self.cfg.sample_rate_hz:
            raise ValueError(f"Enrollment sample rate mismatch for {path}: {sr} != {self.cfg.sample_rate_hz}")
        return np.asarray(audio, dtype=np.float64)

    def _seed_from_enroll_audio(self, synthetic_oracle_sources: np.ndarray | None) -> None:
        seeded = []
        manifest_path = self.cfg.enroll_audio_manifest

        if manifest_path:
            with Path(manifest_path).open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            for key, wav_path in manifest.items():
                sid = int(key)
                audio = self._read_mono(Path(wav_path))
                emb = self.grouper._extract_embedding(audio)
                if emb is None:
                    continue
                self._seed_state(sid, emb)
                seeded.append(sid)
        else:
            if synthetic_oracle_sources is None:
                raise ValueError("No enrollment manifest and no synthetic sources to auto-enroll")
            # Auto-enroll from first second of synthetic oracle tracks.
            n = min(self.cfg.sample_rate_hz, synthetic_oracle_sources.shape[1])
            for sid in range(synthetic_oracle_sources.shape[0]):
                audio = np.asarray(synthetic_oracle_sources[sid, :n], dtype=np.float64)
                emb = self.grouper._extract_embedding(audio)
                if emb is None:
                    continue
                self._seed_state(sid, emb)
                seeded.append(sid)

        self.bootstrap_debug = {"seed_mode": "enroll_audio", "seeded_speaker_ids": sorted(seeded)}
