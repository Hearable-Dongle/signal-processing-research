from __future__ import annotations

import numpy as np

from speaker_identity_grouping import IdentityChunkOutput

from .payload_adapter import BalancedPayloadBuildDebug, build_direction_assignment_input
from .types import DirectionAssignmentInput


def build_payload_for_chunk(
    *,
    chunk_id: int,
    timestamp_ms: float,
    raw_mic_chunk: np.ndarray,
    separated_streams: list[np.ndarray],
    identity_out: IdentityChunkOutput,
    srp_peaks: list[float],
    srp_scores: list[float] | None,
) -> tuple[DirectionAssignmentInput, BalancedPayloadBuildDebug]:
    return build_direction_assignment_input(
        chunk_id=chunk_id,
        timestamp_ms=timestamp_ms,
        raw_mic_chunk=raw_mic_chunk,
        separated_streams=separated_streams,
        stream_to_speaker=identity_out.stream_to_speaker,
        active_speakers=identity_out.active_speakers,
        srp_doa_peaks_deg=srp_peaks,
        srp_peak_scores=srp_scores,
    )
