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
from speaker_identity_grouping.embeddings import SpeakerEmbeddingBackend, build_session_embedding_backend


class SpeakerIdentityGrouper:
    """Realtime speaker identity grouping over separated per-stream audio chunks."""

    def __init__(self, config: IdentityConfig | None = None, session_embedder: SpeakerEmbeddingBackend | None = None):
        self.config = config or IdentityConfig()
        if self.config.max_speakers < 1:
            raise ValueError("max_speakers must be >= 1")
        if self.config.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if self.config.match_threshold < -1.0 or self.config.match_threshold > 1.0:
            raise ValueError("match_threshold must be within [-1.0, 1.0]")
        self._speakers: dict[int, SpeakerState] = {}
        self._next_speaker_id: int = 0
        self._prev_stream_to_speaker: dict[int, int | None] = {}
        self._prev_stream_confidence: dict[int, float] = {}
        self._speaker_audio_buffers: dict[int, np.ndarray] = {}
        self._stream_audio_buffers: dict[int, np.ndarray] = {}
        self._stream_voiceprints: dict[int, np.ndarray] = {}
        self._stream_last_embed_chunk: dict[int, int] = {}
        self._session_embedder = session_embedder
        if self.config.backend == "speaker_embed_session" and self._session_embedder is None:
            self._session_embedder = build_session_embedding_backend(
                self.config.speaker_embedding_model,
                device=self.config.speaker_embedding_device,
            )

    def reset(self) -> None:
        self._speakers.clear()
        self._next_speaker_id = 0
        self._prev_stream_to_speaker.clear()
        self._prev_stream_confidence.clear()
        self._speaker_audio_buffers.clear()
        self._stream_audio_buffers.clear()
        self._stream_voiceprints.clear()
        self._stream_last_embed_chunk.clear()

    def get_state(self) -> dict[int, SpeakerState]:
        return {sid: replace(state) for sid, state in self._speakers.items()}

    def update(self, input_chunk: IdentityChunkInput) -> IdentityChunkOutput:
        cfg = self.config
        if input_chunk.sample_rate_hz != cfg.sample_rate_hz:
            raise ValueError(
                f"sample rate mismatch: expected {cfg.sample_rate_hz}, got {input_chunk.sample_rate_hz}"
            )

        retired_speakers = self._retire_stale_speakers(input_chunk.chunk_id)
        self._direction_priors_deg = {int(k): float(v) for k, v in input_chunk.speaker_direction_priors.items()}
        self._direction_prior_confidence = {
            int(k): float(v) for k, v in input_chunk.speaker_direction_prior_confidence.items()
        }

        stream_to_speaker: dict[int, int | None] = {}
        per_stream_confidence: dict[int, float] = {}
        observations: list[StreamObservation] = []

        for i, stream in enumerate(input_chunk.streams):
            audio = np.asarray(stream, dtype=np.float64).reshape(-1)
            rms = float(np.sqrt(np.mean(audio**2) + cfg.eps))
            speech_likelihood = float(np.clip(input_chunk.per_stream_speech_likelihood.get(i, 0.0), 0.0, 1.0))
            embedding = self._extract_embedding(audio) if rms >= cfg.vad_rms_threshold else None
            active = embedding is not None and speech_likelihood >= float(cfg.speech_likelihood_threshold)
            voiceprint = None
            if cfg.backend == "speaker_embed_session":
                if active:
                    voiceprint = self._update_stream_voiceprint(i, audio, input_chunk.chunk_id)
                else:
                    self._stream_audio_buffers.pop(int(i), None)
            observations.append(
                StreamObservation(
                    stream_index=i,
                    rms=rms,
                    embedding=embedding,
                    active=active,
                    voiceprint=voiceprint,
                    speech_likelihood=speech_likelihood,
                    direction_deg=(
                        None if i not in input_chunk.per_stream_direction_deg else float(input_chunk.per_stream_direction_deg[i])
                    ),
                    direction_confidence=float(input_chunk.per_stream_direction_confidence.get(i, 0.0)),
                )
            )
            stream_to_speaker[i] = None
            per_stream_confidence[i] = 0.0

        if cfg.backend == "speaker_embed_session":
            out = self._update_session_backend(input_chunk, observations, retired_speakers)
        else:
            out = self._update_legacy_backend(input_chunk, observations, retired_speakers)
        self._prev_stream_to_speaker = dict(out.stream_to_speaker)
        self._prev_stream_confidence = dict(out.per_stream_confidence)
        return out

    def _update_legacy_backend(
        self,
        input_chunk: IdentityChunkInput,
        observations: list[StreamObservation],
        retired_speakers: list[int],
    ) -> IdentityChunkOutput:
        cfg = self.config
        stream_to_speaker = {obs.stream_index: None for obs in observations}
        per_stream_confidence = {obs.stream_index: 0.0 for obs in observations}
        active_obs = [obs for obs in observations if obs.active and obs.embedding is not None]
        new_speakers: list[int] = []
        forced_reuse_streams: list[int] = []
        matched_stream_to_speaker: dict[int, int] = {}
        matched_stream_conf: dict[int, float] = {}
        held_streams: list[int] = []
        blocked_switch_streams: list[int] = []
        blocked_new_speaker_streams: list[int] = []
        continuity_tiebreak_margin = 0.03

        if active_obs:
            stream_ids = [obs.stream_index for obs in active_obs]
            embeddings = np.stack([obs.embedding for obs in active_obs], axis=0)

            known_ids = sorted(self._speakers.keys())
            if known_ids:
                centroids = np.stack([self._speakers[sid].centroid for sid in known_ids], axis=0)
                sim = embeddings @ centroids.T
                sim01 = np.clip((sim + 1.0) * 0.5, 0.0, 1.0)
                adjusted = sim01.copy()
                for row_idx, sidx in enumerate(stream_ids):
                    prev_sid = self._prev_stream_to_speaker.get(sidx)
                    if prev_sid in known_ids:
                        adjusted[row_idx, known_ids.index(prev_sid)] += cfg.continuity_bonus
                    for col_idx, sid in enumerate(known_ids):
                        adjusted[row_idx, col_idx] = self._combined_assignment_score(
                            identity_score=float(sim01[row_idx, col_idx]),
                            stream_obs=active_obs[row_idx],
                            speaker_id=int(sid),
                            carry_bonus=(cfg.continuity_bonus if prev_sid == sid else 0.0),
                        )

                rows, cols = linear_sum_assignment(1.0 - np.clip(adjusted, 0.0, 1.0))
                assigned_streams = set()

                for r, c in zip(rows.tolist(), cols.tolist()):
                    score = float(np.clip(adjusted[r, c], 0.0, 1.0))
                    best_raw_score = float(np.max(adjusted[r]))
                    raw_margin = best_raw_score - score
                    if score >= cfg.combined_match_threshold:
                        sidx = stream_ids[r]
                        sid = known_ids[c]
                        prev_sid = self._prev_stream_to_speaker.get(sidx)
                        if prev_sid in known_ids and sid == prev_sid and raw_margin > continuity_tiebreak_margin:
                            continue
                        if prev_sid in known_ids and sid != prev_sid:
                            prev_idx = known_ids.index(prev_sid)
                            prev_score = float(adjusted[r, prev_idx])
                            if score < prev_score + cfg.switch_penalty:
                                blocked_switch_streams.append(sidx)
                                continue
                        matched_stream_to_speaker[sidx] = sid
                        matched_stream_conf[sidx] = score
                        self._update_speaker_local(sid, embeddings[r], input_chunk.chunk_id, input_chunk.timestamp_ms, score, stream_index=sidx)
                        assigned_streams.add(sidx)

                unmatched = [r for r, sidx in enumerate(stream_ids) if sidx not in assigned_streams]
            else:
                unmatched = list(range(len(stream_ids)))

            for r in unmatched:
                sidx = stream_ids[r]
                emb = embeddings[r]
                prev_sid = self._prev_stream_to_speaker.get(sidx)
                prev_conf = float(self._prev_stream_confidence.get(sidx, 0.0))

                if prev_sid in self._speakers:
                    prev_identity = float(np.clip(((emb @ self._speakers[prev_sid].centroid.T) + 1.0) * 0.5, 0.0, 1.0))
                    prev_score = self._combined_assignment_score(
                        identity_score=prev_identity,
                        stream_obs=active_obs[r],
                        speaker_id=int(prev_sid),
                        carry_bonus=cfg.continuity_bonus,
                    )
                    age = input_chunk.chunk_id - self._speakers[prev_sid].last_seen_chunk
                    support = max(prev_score, prev_conf)
                    if age <= cfg.carry_forward_chunks and support >= cfg.hold_similarity_threshold:
                        matched_stream_to_speaker[sidx] = prev_sid
                        matched_stream_conf[sidx] = float(np.clip(support * cfg.confidence_decay, 0.0, 1.0))
                        self._touch_speaker(prev_sid, input_chunk.chunk_id, input_chunk.timestamp_ms, matched_stream_conf[sidx], stream_index=sidx)
                        held_streams.append(sidx)
                        continue

                best_existing_sid = None
                best_existing_score = -1.0
                if self._speakers:
                    known_ids = sorted(self._speakers.keys())
                    centroids = np.stack([self._speakers[sid].centroid for sid in known_ids], axis=0)
                    sims = np.clip(((emb @ centroids.T) + 1.0) * 0.5, 0.0, 1.0)
                    combined = np.asarray(
                        [
                            self._combined_assignment_score(
                                identity_score=float(sims[idx]),
                                stream_obs=active_obs[r],
                                speaker_id=int(sid),
                                carry_bonus=(cfg.continuity_bonus if self._prev_stream_to_speaker.get(sidx) == sid else 0.0),
                            )
                            for idx, sid in enumerate(known_ids)
                        ],
                        dtype=float,
                    )
                    best_idx = int(np.argmax(combined))
                    best_existing_sid = int(known_ids[best_idx])
                    best_existing_score = float(combined[best_idx])

                if len(self._speakers) < cfg.max_speakers:
                    if best_existing_sid is not None and best_existing_score > float(cfg.new_speaker_max_existing_score):
                        matched_stream_to_speaker[sidx] = int(best_existing_sid)
                        matched_stream_conf[sidx] = max(0.0, best_existing_score)
                        self._update_speaker_local(
                            int(best_existing_sid),
                            emb,
                            input_chunk.chunk_id,
                            input_chunk.timestamp_ms,
                            best_existing_score,
                            stream_index=sidx,
                        )
                        forced_reuse_streams.append(sidx)
                        blocked_new_speaker_streams.append(sidx)
                    else:
                        sid = self._register_new_speaker(
                            emb,
                            input_chunk.chunk_id,
                            input_chunk.timestamp_ms,
                            cfg.new_speaker_confidence,
                            stream_index=sidx,
                        )
                        new_speakers.append(sid)
                        matched_stream_to_speaker[sidx] = sid
                        matched_stream_conf[sidx] = float(cfg.new_speaker_confidence)
                else:
                    known_ids = sorted(self._speakers.keys())
                    centroids = np.stack([self._speakers[sid].centroid for sid in known_ids], axis=0)
                    sims = np.clip(((emb @ centroids.T) + 1.0) * 0.5, 0.0, 1.0)
                    combined = np.asarray(
                        [
                            self._combined_assignment_score(
                                identity_score=float(sims[idx]),
                                stream_obs=active_obs[r],
                                speaker_id=int(sid),
                                carry_bonus=(cfg.continuity_bonus if self._prev_stream_to_speaker.get(sidx) == sid else 0.0),
                            )
                            for idx, sid in enumerate(known_ids)
                        ],
                        dtype=float,
                    )
                    c = int(np.argmax(combined))
                    sid = known_ids[c]
                    score = float(np.clip(combined[c], 0.0, 1.0))
                    matched_stream_to_speaker[sidx] = sid
                    matched_stream_conf[sidx] = max(0.0, score)
                    self._update_speaker_local(sid, emb, input_chunk.chunk_id, input_chunk.timestamp_ms, score, stream_index=sidx)
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
                "num_streams": len(observations),
                "num_active_streams": len(active_obs),
                "num_registry_speakers": len(self._speakers),
                "backend": "mfcc_legacy",
                "forced_reuse_streams": forced_reuse_streams,
                "held_streams": held_streams,
                "blocked_switch_streams": blocked_switch_streams,
                "blocked_new_speaker_streams": blocked_new_speaker_streams,
            },
        )

    def _combined_assignment_score(
        self,
        *,
        identity_score: float,
        stream_obs: StreamObservation,
        speaker_id: int,
        carry_bonus: float = 0.0,
    ) -> float:
        cfg = self.config
        id_w = max(0.0, float(cfg.identity_match_weight))
        dir_w = max(0.0, float(cfg.direction_match_weight))
        total_w = max(id_w + dir_w, 1e-8)
        combined = id_w * float(identity_score)

        prior_deg = None
        prior_conf = 0.0
        if speaker_id in getattr(self, "_direction_priors_deg", {}):
            prior_deg = float(self._direction_priors_deg[speaker_id])
            prior_conf = float(getattr(self, "_direction_prior_confidence", {}).get(speaker_id, 0.0))

        if (
            stream_obs.direction_deg is not None
            and float(stream_obs.direction_confidence) >= float(cfg.direction_gate_confidence)
            and prior_deg is not None
            and prior_conf >= float(cfg.direction_gate_confidence)
        ):
            dist = self._circular_distance_deg(float(stream_obs.direction_deg), float(prior_deg))
            if dist >= float(cfg.direction_mismatch_block_deg):
                return -1.0
            dir_score = max(0.0, 1.0 - (dist / max(float(cfg.direction_match_max_distance_deg), 1e-6)))
            combined += dir_w * dir_score
        else:
            total_w = max(id_w, 1e-8)

        combined = (combined / total_w) + float(carry_bonus)
        return float(np.clip(combined, -1.0, 1.0))

    @staticmethod
    def _circular_distance_deg(a: float, b: float) -> float:
        return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)

    def _update_session_backend(
        self,
        input_chunk: IdentityChunkInput,
        observations: list[StreamObservation],
        retired_speakers: list[int],
    ) -> IdentityChunkOutput:
        cfg = self.config
        base_out = self._update_legacy_backend(input_chunk, observations, retired_speakers)
        stream_to_speaker = dict(base_out.stream_to_speaker)
        per_stream_confidence = dict(base_out.per_stream_confidence)
        merged_speakers: list[int] = []
        promoted_speakers: list[int] = []
        lookup_matches: list[dict[str, float | int]] = []

        obs_by_stream = {obs.stream_index: obs for obs in observations}
        # Update session-local voiceprints for speakers using the current stream assignments.
        for stream_idx, speaker_id in stream_to_speaker.items():
            if speaker_id is None:
                continue
            obs = obs_by_stream.get(stream_idx)
            if obs is None or not obs.active:
                continue
            self._append_speaker_audio(int(speaker_id), np.asarray(input_chunk.streams[stream_idx], dtype=np.float32))
            st = self._speakers.get(int(speaker_id))
            if st is None:
                continue
            support_ms = st.speech_support_ms + float(self.config.chunk_duration_ms)
            st = replace(st, speech_support_ms=support_ms)
            self._speakers[int(speaker_id)] = st
            if obs.voiceprint is not None:
                promoted = self._update_speaker_voiceprint(
                    int(speaker_id),
                    obs.voiceprint,
                    input_chunk.chunk_id,
                    provisional_threshold=cfg.speaker_embedding_match_threshold,
                )
                if promoted:
                    promoted_speakers.append(int(speaker_id))

        # Exact cosine lookup over confirmed session voiceprints for unresolved/newer cases.
        for stream_idx, speaker_id in list(stream_to_speaker.items()):
            obs = obs_by_stream.get(stream_idx)
            if obs is None or obs.voiceprint is None:
                continue
            best_sid, best_score, best_margin = self._lookup_voiceprint(obs.voiceprint, exclude_speaker_id=None)
            if best_sid is None:
                continue
            if best_score < cfg.speaker_embedding_match_threshold or best_margin < cfg.speaker_embedding_margin:
                continue
            current_sid = speaker_id
            current_state = self._speakers.get(int(current_sid)) if current_sid is not None else None
            current_score = -1.0
            if current_state is not None and current_state.voiceprint is not None:
                current_score = float(np.clip(obs.voiceprint @ current_state.voiceprint.T, -1.0, 1.0))
            if current_sid is None or best_score > current_score + cfg.speaker_embedding_margin:
                stream_to_speaker[stream_idx] = int(best_sid)
                per_stream_confidence[stream_idx] = max(float(per_stream_confidence.get(stream_idx, 0.0)), float(best_score))
                self._touch_speaker(int(best_sid), input_chunk.chunk_id, input_chunk.timestamp_ms, float(best_score), stream_index=stream_idx)
                lookup_matches.append(
                    {
                        "stream_index": int(stream_idx),
                        "speaker_id": int(best_sid),
                        "score": float(best_score),
                        "margin": float(best_margin),
                    }
                )

        # Merge provisional duplicates into confirmed voiceprint matches.
        for sid, st in list(self._speakers.items()):
            if st.voiceprint is None or not st.provisional:
                continue
            best_sid, best_score, best_margin = self._lookup_voiceprint(st.voiceprint, exclude_speaker_id=int(sid))
            if best_sid is None:
                continue
            if best_score < cfg.speaker_embedding_merge_threshold or best_margin < cfg.speaker_embedding_margin:
                continue
            target = self._speakers.get(int(best_sid))
            if target is None or target.provisional:
                continue
            for stream_idx, assigned_sid in list(stream_to_speaker.items()):
                if assigned_sid == sid:
                    stream_to_speaker[stream_idx] = int(best_sid)
                    per_stream_confidence[stream_idx] = max(float(per_stream_confidence.get(stream_idx, 0.0)), float(best_score))
            self._merge_speakers(source_id=int(sid), target_id=int(best_sid))
            merged_speakers.append(int(sid))

        active_speakers = sorted({sid for sid in stream_to_speaker.values() if sid is not None})
        return IdentityChunkOutput(
            chunk_id=input_chunk.chunk_id,
            timestamp_ms=input_chunk.timestamp_ms,
            stream_to_speaker=stream_to_speaker,
            active_speakers=active_speakers,
            new_speakers=sorted(base_out.new_speakers),
            retired_speakers=sorted(retired_speakers),
            per_stream_confidence=per_stream_confidence,
            debug={
                **base_out.debug,
                "backend": "speaker_embed_session",
                "confirmed_speakers": sorted([int(sid) for sid, st in self._speakers.items() if not st.provisional]),
                "provisional_speakers": sorted([int(sid) for sid, st in self._speakers.items() if st.provisional]),
                "promoted_speakers": sorted(promoted_speakers),
                "merged_speakers": sorted(merged_speakers),
                "lookup_matches": lookup_matches,
            },
        )

    def _retire_stale_speakers(self, current_chunk_id: int) -> list[int]:
        retire_after = max(0, int(self.config.retire_after_chunks))
        retired: list[int] = []
        for sid, state in list(self._speakers.items()):
            timeout = retire_after
            if state.provisional:
                timeout = min(timeout, max(1, int(self.config.provisional_speaker_timeout_chunks)))
            if current_chunk_id - state.last_seen_chunk > timeout:
                retired.append(sid)
                del self._speakers[sid]
                self._speaker_audio_buffers.pop(int(sid), None)
        return retired

    def _register_new_speaker(
        self,
        embedding: np.ndarray,
        chunk_id: int,
        timestamp_ms: float,
        confidence: float,
        *,
        stream_index: int | None = None,
    ) -> int:
        sid = self._next_speaker_id
        self._next_speaker_id += 1
        self._speakers[sid] = SpeakerState(
            speaker_id=sid,
            centroid=embedding.copy(),
            sample_count=1,
            last_seen_chunk=chunk_id,
            last_seen_timestamp_ms=timestamp_ms,
            last_confidence=float(confidence),
            provisional=(self.config.backend == "speaker_embed_session"),
            last_stream_index=stream_index,
        )
        return sid

    def _update_speaker_local(
        self,
        speaker_id: int,
        embedding: np.ndarray,
        chunk_id: int,
        timestamp_ms: float,
        confidence: float,
        *,
        stream_index: int | None = None,
    ) -> None:
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
            last_confidence=float(confidence),
            hold_count=0,
            voiceprint=state.voiceprint,
            voiceprint_updates=state.voiceprint_updates,
            speech_support_ms=state.speech_support_ms,
            provisional=state.provisional,
            last_stream_index=stream_index,
            last_voiceprint_chunk=state.last_voiceprint_chunk,
        )

    def _touch_speaker(
        self,
        speaker_id: int,
        chunk_id: int,
        timestamp_ms: float,
        confidence: float,
        *,
        stream_index: int | None = None,
    ) -> None:
        state = self._speakers[speaker_id]
        self._speakers[speaker_id] = SpeakerState(
            speaker_id=speaker_id,
            centroid=state.centroid,
            sample_count=state.sample_count,
            last_seen_chunk=chunk_id,
            last_seen_timestamp_ms=timestamp_ms,
            last_confidence=float(confidence),
            hold_count=state.hold_count + 1,
            voiceprint=state.voiceprint,
            voiceprint_updates=state.voiceprint_updates,
            speech_support_ms=state.speech_support_ms,
            provisional=state.provisional,
            last_stream_index=stream_index if stream_index is not None else state.last_stream_index,
            last_voiceprint_chunk=state.last_voiceprint_chunk,
        )

    def _update_stream_voiceprint(self, stream_index: int, audio: np.ndarray, chunk_id: int) -> np.ndarray | None:
        if self._session_embedder is None:
            return None
        self._append_stream_audio(stream_index, audio)
        buf = self._stream_audio_buffers.get(int(stream_index))
        if buf is None:
            return None
        support_ms = 1000.0 * len(buf) / max(1, int(self.config.sample_rate_hz))
        if support_ms < float(self.config.speaker_embedding_min_speech_ms):
            return self._stream_voiceprints.get(int(stream_index))
        last_chunk = int(self._stream_last_embed_chunk.get(int(stream_index), -10**9))
        if (chunk_id - last_chunk) < max(1, int(self.config.speaker_embedding_update_interval_chunks)):
            return self._stream_voiceprints.get(int(stream_index))
        voiceprint = self._session_embedder.embed(buf, self.config.sample_rate_hz)
        if voiceprint is not None:
            self._stream_voiceprints[int(stream_index)] = self._normalize_embedding(voiceprint)
            self._stream_last_embed_chunk[int(stream_index)] = int(chunk_id)
        return self._stream_voiceprints.get(int(stream_index))

    def _update_speaker_voiceprint(self, speaker_id: int, voiceprint: np.ndarray, chunk_id: int, provisional_threshold: float) -> bool:
        state = self._speakers[speaker_id]
        vp = self._normalize_embedding(voiceprint)
        if state.voiceprint is None:
            new_vp = vp
            updates = 1
        else:
            new_vp = self._normalize_embedding((1.0 - self.config.ema_alpha) * state.voiceprint + self.config.ema_alpha * vp)
            updates = int(state.voiceprint_updates) + 1
        provisional = state.provisional
        if provisional and state.speech_support_ms >= float(self.config.speaker_embedding_min_speech_ms):
            provisional = False
        self._speakers[speaker_id] = replace(
            state,
            voiceprint=new_vp,
            voiceprint_updates=updates,
            provisional=provisional,
            last_voiceprint_chunk=int(chunk_id),
        )
        return bool(state.provisional and not provisional)

    def _lookup_voiceprint(self, voiceprint: np.ndarray, exclude_speaker_id: int | None) -> tuple[int | None, float, float]:
        candidates: list[tuple[int, float]] = []
        for sid, st in self._speakers.items():
            if exclude_speaker_id is not None and int(sid) == int(exclude_speaker_id):
                continue
            if st.voiceprint is None or st.provisional:
                continue
            score = float(np.clip(voiceprint @ st.voiceprint.T, -1.0, 1.0))
            candidates.append((int(sid), score))
        if not candidates:
            return None, -1.0, 0.0
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_sid, best_score = candidates[0]
        second = candidates[1][1] if len(candidates) > 1 else -1.0
        return int(best_sid), float(best_score), float(best_score - second)

    def _merge_speakers(self, source_id: int, target_id: int) -> None:
        source = self._speakers.get(int(source_id))
        target = self._speakers.get(int(target_id))
        if source is None or target is None:
            return
        merged_centroid = self._normalize_embedding(
            ((target.sample_count * target.centroid) + (source.sample_count * source.centroid))
            / max(1, target.sample_count + source.sample_count)
        )
        merged_vp = target.voiceprint
        if source.voiceprint is not None:
            merged_vp = source.voiceprint if merged_vp is None else self._normalize_embedding(0.5 * (merged_vp + source.voiceprint))
        self._speakers[int(target_id)] = replace(
            target,
            centroid=merged_centroid,
            sample_count=int(target.sample_count + source.sample_count),
            last_seen_chunk=max(int(target.last_seen_chunk), int(source.last_seen_chunk)),
            last_seen_timestamp_ms=max(float(target.last_seen_timestamp_ms), float(source.last_seen_timestamp_ms)),
            last_confidence=max(float(target.last_confidence), float(source.last_confidence)),
            voiceprint=merged_vp,
            voiceprint_updates=int(target.voiceprint_updates + source.voiceprint_updates),
            speech_support_ms=float(target.speech_support_ms + source.speech_support_ms),
            provisional=False,
        )
        self._speaker_audio_buffers[int(target_id)] = self._truncate_audio_buffer(
            np.concatenate(
                [
                    self._speaker_audio_buffers.get(int(target_id), np.zeros(0, dtype=np.float32)),
                    self._speaker_audio_buffers.get(int(source_id), np.zeros(0, dtype=np.float32)),
                ]
            )
        )
        del self._speakers[int(source_id)]
        self._speaker_audio_buffers.pop(int(source_id), None)
        for sidx, prev_sid in list(self._prev_stream_to_speaker.items()):
            if prev_sid == int(source_id):
                self._prev_stream_to_speaker[sidx] = int(target_id)

    def _append_stream_audio(self, stream_index: int, audio: np.ndarray) -> None:
        prev = self._stream_audio_buffers.get(int(stream_index), np.zeros(0, dtype=np.float32))
        self._stream_audio_buffers[int(stream_index)] = self._truncate_audio_buffer(np.concatenate([prev, np.asarray(audio, dtype=np.float32).reshape(-1)]))

    def _append_speaker_audio(self, speaker_id: int, audio: np.ndarray) -> None:
        prev = self._speaker_audio_buffers.get(int(speaker_id), np.zeros(0, dtype=np.float32))
        self._speaker_audio_buffers[int(speaker_id)] = self._truncate_audio_buffer(np.concatenate([prev, np.asarray(audio, dtype=np.float32).reshape(-1)]))

    def _truncate_audio_buffer(self, audio: np.ndarray) -> np.ndarray:
        max_samples = max(1, int(self.config.sample_rate_hz * self.config.speaker_embedding_buffer_ms / 1000.0))
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size <= max_samples:
            return arr
        return arr[-max_samples:]

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
        emb = np.asarray(emb, dtype=np.float64).reshape(-1)
        emb = (emb - float(np.mean(emb))) / (float(np.std(emb)) + cfg.eps)
        norm = float(np.linalg.norm(emb))
        if norm <= cfg.eps:
            return emb.astype(np.float32, copy=False)
        return (emb / norm).astype(np.float32, copy=False)

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
