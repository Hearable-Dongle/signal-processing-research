from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.fftpack import dct
from scipy.optimize import linear_sum_assignment

from speaker_identity_grouping.contracts import (
    IdentityChunkInput,
    IdentityChunkOutput,
    IdentityConfig,
    SpeakerState,
    StreamObservation,
)


class SpeakerIdentityGrouper:
    """Realtime speaker identity grouping over separated per-stream audio chunks."""

    def __init__(self, config: IdentityConfig | None = None):
        self.config = config or IdentityConfig()
        if self.config.max_speakers < 1:
            raise ValueError("max_speakers must be >= 1")
        if self.config.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if self.config.match_threshold < -1.0 or self.config.match_threshold > 1.0:
            raise ValueError("match_threshold must be within [-1.0, 1.0]")
        self._speakers: dict[int, SpeakerState] = {}
        self._next_speaker_id: int = 0

    def reset(self) -> None:
        self._speakers.clear()
        self._next_speaker_id = 0

    def get_state(self) -> dict[int, SpeakerState]:
        return {sid: replace(state) for sid, state in self._speakers.items()}

    def update(self, input_chunk: IdentityChunkInput) -> IdentityChunkOutput:
        cfg = self.config
        if input_chunk.sample_rate_hz != cfg.sample_rate_hz:
            raise ValueError(
                f"sample rate mismatch: expected {cfg.sample_rate_hz}, got {input_chunk.sample_rate_hz}"
            )

        retired_speakers = self._retire_stale_speakers(input_chunk.chunk_id)

        stream_to_speaker: dict[int, int | None] = {}
        per_stream_confidence: dict[int, float] = {}
        observations: list[StreamObservation] = []

        for i, stream in enumerate(input_chunk.streams):
            audio = np.asarray(stream, dtype=np.float64).reshape(-1)
            rms = float(np.sqrt(np.mean(audio**2) + cfg.eps))
            embedding = self._extract_embedding(audio) if rms >= cfg.vad_rms_threshold else None
            active = embedding is not None
            observations.append(StreamObservation(stream_index=i, rms=rms, embedding=embedding, active=active))
            stream_to_speaker[i] = None
            per_stream_confidence[i] = 0.0

        active_obs = [obs for obs in observations if obs.active and obs.embedding is not None]
        new_speakers: list[int] = []
        forced_reuse_streams: list[int] = []

        matched_stream_to_speaker: dict[int, int] = {}
        matched_stream_conf: dict[int, float] = {}

        if active_obs:
            stream_ids = [obs.stream_index for obs in active_obs]
            embeddings = np.stack([obs.embedding for obs in active_obs], axis=0)

            known_ids = sorted(self._speakers.keys())
            if known_ids:
                centroids = np.stack([self._speakers[sid].centroid for sid in known_ids], axis=0)
                sim = embeddings @ centroids.T

                rows, cols = linear_sum_assignment(1.0 - sim)
                assigned_streams = set()

                for r, c in zip(rows.tolist(), cols.tolist()):
                    score = float(np.clip(sim[r, c], -1.0, 1.0))
                    if score >= cfg.match_threshold:
                        sidx = stream_ids[r]
                        sid = known_ids[c]
                        matched_stream_to_speaker[sidx] = sid
                        matched_stream_conf[sidx] = score
                        self._update_speaker(sid, embeddings[r], input_chunk.chunk_id, input_chunk.timestamp_ms)
                        assigned_streams.add(sidx)

                unmatched = [r for r, sidx in enumerate(stream_ids) if sidx not in assigned_streams]
            else:
                unmatched = list(range(len(stream_ids)))

            for r in unmatched:
                sidx = stream_ids[r]
                emb = embeddings[r]

                if len(self._speakers) < cfg.max_speakers:
                    sid = self._register_new_speaker(emb, input_chunk.chunk_id, input_chunk.timestamp_ms)
                    new_speakers.append(sid)
                    matched_stream_to_speaker[sidx] = sid
                    matched_stream_conf[sidx] = float(cfg.new_speaker_confidence)
                else:
                    # Registry full: force map to nearest active speaker for continuity.
                    known_ids = sorted(self._speakers.keys())
                    centroids = np.stack([self._speakers[sid].centroid for sid in known_ids], axis=0)
                    sims = emb @ centroids.T
                    c = int(np.argmax(sims))
                    sid = known_ids[c]
                    score = float(np.clip(sims[c], -1.0, 1.0))
                    matched_stream_to_speaker[sidx] = sid
                    matched_stream_conf[sidx] = max(0.0, score)
                    self._update_speaker(sid, emb, input_chunk.chunk_id, input_chunk.timestamp_ms)
                    forced_reuse_streams.append(sidx)

        for sidx, sid in matched_stream_to_speaker.items():
            stream_to_speaker[sidx] = sid
            per_stream_confidence[sidx] = matched_stream_conf.get(sidx, 0.0)

        active_speakers = sorted({sid for sid in stream_to_speaker.values() if sid is not None})

        return IdentityChunkOutput(
            chunk_id=input_chunk.chunk_id,
            timestamp_ms=input_chunk.timestamp_ms,
            stream_to_speaker=stream_to_speaker,
            active_speakers=active_speakers,
            new_speakers=sorted(new_speakers),
            retired_speakers=sorted(retired_speakers),
            per_stream_confidence=per_stream_confidence,
            debug={
                "num_streams": len(input_chunk.streams),
                "num_active_streams": len(active_obs),
                "num_registry_speakers": len(self._speakers),
                "forced_reuse_streams": forced_reuse_streams,
            },
        )

    def _retire_stale_speakers(self, current_chunk_id: int) -> list[int]:
        retire_after = max(0, int(self.config.retire_after_chunks))
        retired: list[int] = []

        for sid, state in list(self._speakers.items()):
            if current_chunk_id - state.last_seen_chunk > retire_after:
                retired.append(sid)
                del self._speakers[sid]

        return retired

    def _register_new_speaker(self, embedding: np.ndarray, chunk_id: int, timestamp_ms: float) -> int:
        sid = self._next_speaker_id
        self._next_speaker_id += 1

        self._speakers[sid] = SpeakerState(
            speaker_id=sid,
            centroid=embedding.copy(),
            sample_count=1,
            last_seen_chunk=chunk_id,
            last_seen_timestamp_ms=timestamp_ms,
        )
        return sid

    def _update_speaker(self, speaker_id: int, embedding: np.ndarray, chunk_id: int, timestamp_ms: float) -> None:
        cfg = self.config
        state = self._speakers[speaker_id]
        centroid = (1.0 - cfg.ema_alpha) * state.centroid + cfg.ema_alpha * embedding
        centroid = self._normalize_embedding(centroid)

        self._speakers[speaker_id] = SpeakerState(
            speaker_id=speaker_id,
            centroid=centroid,
            sample_count=state.sample_count + 1,
            last_seen_chunk=chunk_id,
            last_seen_timestamp_ms=timestamp_ms,
        )

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray | None:
        if audio.size == 0:
            return None

        cfg = self.config
        emphasized = np.append(audio[0], audio[1:] - cfg.preemphasis * audio[:-1])

        frame_len = max(16, int(cfg.frame_length_ms * cfg.sample_rate_hz / 1000.0))
        frame_hop = max(8, int(cfg.frame_hop_ms * cfg.sample_rate_hz / 1000.0))
        frames = self._frame_signal(emphasized, frame_len, frame_hop)
        if frames.size == 0:
            return None

        frames = frames * np.hamming(frame_len)[None, :]

        spec = np.abs(np.fft.rfft(frames, n=cfg.n_fft))
        power = (1.0 / cfg.n_fft) * (spec**2)

        mel_filterbank = self._mel_filterbank(cfg.n_mels, cfg.n_fft, cfg.sample_rate_hz)
        mel_energy = np.maximum(power @ mel_filterbank.T, cfg.eps)
        log_mel = np.log(mel_energy)

        mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, : cfg.n_mfcc]
        stats = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)], axis=0)
        return self._normalize_embedding(stats)

    def _normalize_embedding(self, emb: np.ndarray) -> np.ndarray:
        cfg = self.config
        emb = emb.astype(np.float64, copy=False)
        emb = (emb - float(np.mean(emb))) / (float(np.std(emb)) + cfg.eps)
        norm = float(np.linalg.norm(emb))
        if norm <= cfg.eps:
            return emb
        return emb / norm

    @staticmethod
    def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _mel_filterbank(self, n_mels: int, n_fft: int, sample_rate_hz: int) -> np.ndarray:
        n_freq_bins = n_fft // 2 + 1
        low_mel = self._hz_to_mel(np.array([0.0]))[0]
        high_mel = self._hz_to_mel(np.array([sample_rate_hz / 2.0]))[0]

        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bins = np.floor((n_fft + 1) * hz_points / sample_rate_hz).astype(int)

        fb = np.zeros((n_mels, n_freq_bins), dtype=np.float64)
        for m in range(1, n_mels + 1):
            left = max(0, bins[m - 1])
            center = max(left + 1, bins[m])
            right = max(center + 1, bins[m + 1])

            for k in range(left, min(center, n_freq_bins)):
                fb[m - 1, k] = (k - left) / max(1, center - left)
            for k in range(center, min(right, n_freq_bins)):
                fb[m - 1, k] = (right - k) / max(1, right - center)

        return fb

    @staticmethod
    def _frame_signal(sig: np.ndarray, frame_len: int, frame_hop: int) -> np.ndarray:
        if sig.size < frame_len:
            pad = np.zeros(frame_len - sig.size, dtype=sig.dtype)
            sig = np.concatenate([sig, pad], axis=0)

        n_frames = 1 + int(np.ceil((sig.size - frame_len) / frame_hop))
        total_len = (n_frames - 1) * frame_hop + frame_len
        if total_len > sig.size:
            sig = np.concatenate([sig, np.zeros(total_len - sig.size, dtype=sig.dtype)], axis=0)

        idx = (
            np.tile(np.arange(frame_len), (n_frames, 1))
            + np.tile(np.arange(0, n_frames * frame_hop, frame_hop), (frame_len, 1)).T
        )
        return sig[idx]
