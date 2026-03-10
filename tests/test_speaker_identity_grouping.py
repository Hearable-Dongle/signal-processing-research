import numpy as np

from speaker_identity_grouping import IdentityChunkInput, IdentityConfig, SpeakerIdentityGrouper


def _tone(freq_hz: float, seconds: float = 0.2, sr: int = 16000, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    return amp * np.sin(2.0 * np.pi * freq_hz * t)


def test_same_speaker_keeps_id_across_chunks():
    cfg = IdentityConfig(match_threshold=0.75)
    grouper = SpeakerIdentityGrouper(cfg)

    s0 = _tone(180.0)
    out0 = grouper.update(IdentityChunkInput(chunk_id=0, timestamp_ms=0.0, sample_rate_hz=16000, streams=[s0]))
    sid0 = out0.stream_to_speaker[0]

    s1 = _tone(180.0) + 0.005 * np.random.default_rng(1).normal(size=s0.shape)
    out1 = grouper.update(IdentityChunkInput(chunk_id=1, timestamp_ms=200.0, sample_rate_hz=16000, streams=[s1]))

    assert sid0 is not None
    assert out1.stream_to_speaker[0] == sid0
    assert sid0 in out1.active_speakers


def test_silent_stream_is_not_assigned():
    grouper = SpeakerIdentityGrouper(IdentityConfig(vad_rms_threshold=0.01))
    silence = np.zeros(3200, dtype=np.float64)

    out = grouper.update(IdentityChunkInput(chunk_id=0, timestamp_ms=0.0, sample_rate_hz=16000, streams=[silence]))

    assert out.stream_to_speaker[0] is None
    assert out.active_speakers == []
    assert out.per_stream_confidence[0] == 0.0


def test_two_distinct_streams_split_into_two_ids():
    cfg = IdentityConfig(match_threshold=0.90)
    grouper = SpeakerIdentityGrouper(cfg)

    a = _tone(120.0)
    b = _tone(900.0)
    out = grouper.update(IdentityChunkInput(chunk_id=0, timestamp_ms=0.0, sample_rate_hz=16000, streams=[a, b]))

    sid_a = out.stream_to_speaker[0]
    sid_b = out.stream_to_speaker[1]

    assert sid_a is not None
    assert sid_b is not None
    assert sid_a != sid_b
    assert sorted(out.new_speakers) == sorted([sid_a, sid_b])


def test_ttl_retires_inactive_speaker():
    cfg = IdentityConfig(retire_after_chunks=1, match_threshold=0.75)
    grouper = SpeakerIdentityGrouper(cfg)

    out0 = grouper.update(
        IdentityChunkInput(chunk_id=0, timestamp_ms=0.0, sample_rate_hz=16000, streams=[_tone(200.0)])
    )
    sid = out0.stream_to_speaker[0]
    assert sid is not None

    silence = np.zeros(3200, dtype=np.float64)
    out1 = grouper.update(IdentityChunkInput(chunk_id=1, timestamp_ms=200.0, sample_rate_hz=16000, streams=[silence]))
    out2 = grouper.update(IdentityChunkInput(chunk_id=2, timestamp_ms=400.0, sample_rate_hz=16000, streams=[silence]))

    assert out1.retired_speakers == []
    assert sid in out2.retired_speakers


def test_max_speakers_forces_reuse():
    cfg = IdentityConfig(max_speakers=1, match_threshold=0.95)
    grouper = SpeakerIdentityGrouper(cfg)

    out0 = grouper.update(
        IdentityChunkInput(chunk_id=0, timestamp_ms=0.0, sample_rate_hz=16000, streams=[_tone(140.0)])
    )
    sid0 = out0.stream_to_speaker[0]
    assert sid0 is not None

    out1 = grouper.update(
        IdentityChunkInput(chunk_id=1, timestamp_ms=200.0, sample_rate_hz=16000, streams=[_tone(1100.0)])
    )

    assert out1.stream_to_speaker[0] == sid0


def test_weak_evidence_carries_forward_recent_assignment():
    cfg = IdentityConfig(
        match_threshold=0.95,
        hold_similarity_threshold=0.2,
        carry_forward_chunks=2,
        confidence_decay=0.8,
    )
    grouper = SpeakerIdentityGrouper(cfg)

    out0 = grouper.update(
        IdentityChunkInput(chunk_id=0, timestamp_ms=0.0, sample_rate_hz=16000, streams=[_tone(220.0)])
    )
    sid0 = out0.stream_to_speaker[0]
    assert sid0 is not None

    noisy = 0.03 * np.random.default_rng(2).normal(size=_tone(220.0).shape)
    out1 = grouper.update(
        IdentityChunkInput(chunk_id=1, timestamp_ms=200.0, sample_rate_hz=16000, streams=[noisy])
    )

    assert out1.stream_to_speaker[0] == sid0
    assert 0 in out1.debug["held_streams"]


class _FakeSessionEmbedder:
    def __init__(self, mapping: dict[int, np.ndarray]) -> None:
        self._mapping = {int(k): np.asarray(v, dtype=np.float32) for k, v in mapping.items()}

    def embed(self, audio: np.ndarray, sample_rate_hz: int) -> np.ndarray | None:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return None
        key = int(np.round(float(np.mean(x)) * 1000.0))
        return self._mapping.get(key)


def test_session_backend_merges_duplicate_voiceprints_after_support():
    cfg = IdentityConfig(
        backend="speaker_embed_session",
        chunk_duration_ms=200,
        speaker_embedding_min_speech_ms=400.0,
        speaker_embedding_update_interval_chunks=1,
        speaker_embedding_match_threshold=0.6,
        speaker_embedding_merge_threshold=0.7,
        speaker_embedding_margin=0.01,
        provisional_speaker_timeout_chunks=8,
    )
    voiceprint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    grouper = SpeakerIdentityGrouper(
        cfg,
        session_embedder=_FakeSessionEmbedder(
            {
                100: voiceprint,
                200: voiceprint,
            }
        ),
    )

    a = np.full(3200, 0.100, dtype=np.float64)
    b = np.full(3200, 0.200, dtype=np.float64)

    out0 = grouper.update(IdentityChunkInput(chunk_id=0, timestamp_ms=0.0, sample_rate_hz=16000, streams=[a]))
    sid0 = out0.stream_to_speaker[0]
    assert sid0 is not None

    out1 = grouper.update(IdentityChunkInput(chunk_id=1, timestamp_ms=200.0, sample_rate_hz=16000, streams=[a]))
    assert sid0 in out1.debug["promoted_speakers"] or sid0 in grouper.get_state()

    out2 = grouper.update(IdentityChunkInput(chunk_id=2, timestamp_ms=400.0, sample_rate_hz=16000, streams=[b]))
    assert out2.stream_to_speaker[0] == sid0


def test_session_backend_scales_to_five_distinct_speakers_without_reuse():
    cfg = IdentityConfig(
        backend="speaker_embed_session",
        max_speakers=5,
        speaker_embedding_min_speech_ms=200.0,
        speaker_embedding_update_interval_chunks=1,
        speaker_embedding_match_threshold=0.6,
        speaker_embedding_merge_threshold=0.8,
        speaker_embedding_margin=0.01,
    )
    mapping = {
        100: np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        110: np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        120: np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        130: np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        140: np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    grouper = SpeakerIdentityGrouper(cfg, session_embedder=_FakeSessionEmbedder(mapping))
    streams = [np.full(3200, k / 1000.0, dtype=np.float64) for k in [100, 110, 120, 130, 140]]

    out = grouper.update(IdentityChunkInput(chunk_id=0, timestamp_ms=0.0, sample_rate_hz=16000, streams=streams))
    ids = [out.stream_to_speaker[i] for i in range(5)]
    assert all(sid is not None for sid in ids)
    assert len(set(ids)) == 5
