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
    assert 0 in out1.debug["forced_reuse_streams"]
