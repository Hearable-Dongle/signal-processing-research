from __future__ import annotations

import numpy as np

from speaker_identity_grouping import IdentityChunkInput, IdentityConfig, SpeakerIdentityGrouper


def test_identity_grouper_ignores_non_speechlike_stream(monkeypatch) -> None:
    cfg = IdentityConfig(
        sample_rate_hz=16000,
        chunk_duration_ms=2000,
        speech_likelihood_threshold=0.45,
    )
    grouper = SpeakerIdentityGrouper(cfg)

    monkeypatch.setattr(
        grouper,
        "_extract_embedding",
        lambda audio: np.array([1.0, 0.0], dtype=np.float32),
    )

    out = grouper.update(
        IdentityChunkInput(
            chunk_id=0,
            timestamp_ms=0.0,
            sample_rate_hz=16000,
            streams=[np.ones(3200, dtype=np.float32) * 0.02],
            per_stream_speech_likelihood={0: 0.1},
        )
    )

    assert out.active_speakers == []
    assert out.stream_to_speaker[0] is None


def test_identity_grouper_blocks_reuse_when_direction_is_incompatible(monkeypatch) -> None:
    cfg = IdentityConfig(
        sample_rate_hz=16000,
        chunk_duration_ms=2000,
        speech_likelihood_threshold=0.0,
        identity_match_weight=0.7,
        direction_match_weight=0.3,
        combined_match_threshold=0.58,
        direction_match_max_distance_deg=35.0,
        direction_mismatch_block_deg=60.0,
        direction_gate_confidence=0.3,
        max_speakers=4,
    )
    grouper = SpeakerIdentityGrouper(cfg)

    emb = np.array([1.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(grouper, "_extract_embedding", lambda audio: emb.copy())

    first = grouper.update(
        IdentityChunkInput(
            chunk_id=0,
            timestamp_ms=0.0,
            sample_rate_hz=16000,
            streams=[np.ones(3200, dtype=np.float32) * 0.02],
            per_stream_speech_likelihood={0: 0.9},
            per_stream_direction_deg={0: 0.0},
            per_stream_direction_confidence={0: 0.9},
        )
    )
    assert first.stream_to_speaker[0] == 0

    second = grouper.update(
        IdentityChunkInput(
            chunk_id=1,
            timestamp_ms=2000.0,
            sample_rate_hz=16000,
            streams=[np.ones(3200, dtype=np.float32) * 0.02],
            per_stream_speech_likelihood={0: 0.9},
            per_stream_direction_deg={0: 150.0},
            per_stream_direction_confidence={0: 0.9},
            speaker_direction_priors={0: 0.0},
            speaker_direction_prior_confidence={0: 0.9},
        )
    )

    assert second.stream_to_speaker[0] == 1
    assert 1 in second.new_speakers


def test_identity_grouper_reuses_existing_speaker_in_gray_zone(monkeypatch) -> None:
    cfg = IdentityConfig(
        sample_rate_hz=16000,
        chunk_duration_ms=2000,
        speech_likelihood_threshold=0.0,
        identity_match_weight=0.7,
        direction_match_weight=0.3,
        combined_match_threshold=0.58,
        new_speaker_max_existing_score=0.32,
        direction_match_max_distance_deg=35.0,
        direction_mismatch_block_deg=60.0,
        direction_gate_confidence=0.3,
        max_speakers=4,
    )
    grouper = SpeakerIdentityGrouper(cfg)

    emb_seed = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.6, 0.8], dtype=np.float32),
    ]

    def _extract(_audio):
        return emb_seed.pop(0)

    monkeypatch.setattr(grouper, "_extract_embedding", _extract)

    first = grouper.update(
        IdentityChunkInput(
            chunk_id=0,
            timestamp_ms=0.0,
            sample_rate_hz=16000,
            streams=[np.ones(3200, dtype=np.float32) * 0.02],
            per_stream_speech_likelihood={0: 0.9},
            per_stream_direction_deg={0: 0.0},
            per_stream_direction_confidence={0: 0.9},
        )
    )
    assert first.stream_to_speaker[0] == 0

    second = grouper.update(
        IdentityChunkInput(
            chunk_id=1,
            timestamp_ms=2000.0,
            sample_rate_hz=16000,
            streams=[np.ones(3200, dtype=np.float32) * 0.02],
            per_stream_speech_likelihood={0: 0.9},
            per_stream_direction_deg={0: 5.0},
            per_stream_direction_confidence={0: 0.9},
            speaker_direction_priors={0: 0.0},
            speaker_direction_prior_confidence={0: 0.9},
        )
    )

    assert second.stream_to_speaker[0] == 0
    assert second.new_speakers == []
