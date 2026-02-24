from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import DirectionAssignmentInput


@dataclass
class BalancedPayloadBuildDebug:
    dropped_stream_mapping_keys: list[int] = field(default_factory=list)
    trimmed_stream_indices: list[int] = field(default_factory=list)
    padded_stream_indices: list[int] = field(default_factory=list)
    corrected_active_speakers: bool = False
    dropped_srp_scores: bool = False


def _as_2d_float(name: str, arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={out.shape}")
    return out


def _as_1d_float(name: str, arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return out


def _pad_or_trim_1d(x: np.ndarray, target_len: int) -> tuple[np.ndarray, str | None]:
    if len(x) == target_len:
        return x, None
    if len(x) > target_len:
        return x[:target_len], "trim"
    return np.pad(x, (0, target_len - len(x))), "pad"


def validate_balanced_payload(payload: DirectionAssignmentInput) -> None:
    if not isinstance(payload.chunk_id, int):
        raise ValueError("chunk_id must be int")
    if payload.chunk_id < 0:
        raise ValueError("chunk_id must be >= 0")

    if payload.raw_mic_chunk.ndim != 2:
        raise ValueError("raw_mic_chunk must be shape (samples, n_mics)")
    n_samples, _ = payload.raw_mic_chunk.shape
    if n_samples <= 0:
        raise ValueError("raw_mic_chunk must have at least 1 sample")

    for i, s in enumerate(payload.separated_streams):
        s_arr = np.asarray(s)
        if s_arr.ndim != 1:
            raise ValueError(f"separated_streams[{i}] must be 1D")
        if len(s_arr) != n_samples:
            raise ValueError(
                f"separated_streams[{i}] length mismatch: got {len(s_arr)} expected {n_samples}"
            )

    max_idx = len(payload.separated_streams) - 1
    for idx in payload.stream_to_speaker.keys():
        if not isinstance(idx, int):
            raise ValueError("stream_to_speaker keys must be int")
        if idx < 0 or idx > max_idx:
            raise ValueError(
                f"stream_to_speaker has out-of-range key {idx}; valid [0,{max_idx}]"
            )

    inferred_active = sorted({sid for sid in payload.stream_to_speaker.values() if sid is not None})
    if payload.active_speakers != inferred_active:
        raise ValueError(
            "active_speakers mismatch; expected derived active speaker list from stream_to_speaker"
        )

    if payload.srp_peak_scores is not None and len(payload.srp_peak_scores) != len(payload.srp_doa_peaks_deg):
        raise ValueError("srp_peak_scores length must match srp_doa_peaks_deg length")


def build_direction_assignment_input(
    *,
    chunk_id: int,
    timestamp_ms: float,
    raw_mic_chunk: np.ndarray,
    separated_streams: list[np.ndarray],
    stream_to_speaker: dict[int, int | None] | None,
    active_speakers: list[int] | None,
    srp_doa_peaks_deg: list[float] | None,
    srp_peak_scores: list[float] | None,
) -> tuple[DirectionAssignmentInput, BalancedPayloadBuildDebug]:
    """
    Build balanced payload with corrective normalization:
    - stream lengths are trimmed/padded to raw chunk length
    - invalid stream_to_speaker indices are dropped
    - active_speakers is always recomputed from stream_to_speaker
    - srp_peak_scores are dropped if length mismatch
    """
    debug = BalancedPayloadBuildDebug()

    raw = _as_2d_float("raw_mic_chunk", raw_mic_chunk)
    n_samples, _ = raw.shape

    norm_streams: list[np.ndarray] = []
    for i, s in enumerate(separated_streams):
        s_arr = _as_1d_float(f"separated_streams[{i}]", s)
        s_norm, action = _pad_or_trim_1d(s_arr, n_samples)
        if action == "trim":
            debug.trimmed_stream_indices.append(i)
        elif action == "pad":
            debug.padded_stream_indices.append(i)
        norm_streams.append(s_norm)

    if stream_to_speaker is None:
        raw_map: dict[int, int | None] = {i: None for i in range(len(norm_streams))}
    else:
        raw_map = dict(stream_to_speaker)

    valid_map: dict[int, int | None] = {}
    max_idx = len(norm_streams) - 1
    for k, v in raw_map.items():
        if not isinstance(k, int) or k < 0 or k > max_idx:
            try:
                debug.dropped_stream_mapping_keys.append(int(k))
            except Exception:
                pass
            continue
        valid_map[k] = None if v is None else int(v)

    # Ensure every stream has an explicit mapping entry.
    for i in range(len(norm_streams)):
        valid_map.setdefault(i, None)

    derived_active = sorted({sid for sid in valid_map.values() if sid is not None})
    if active_speakers is None or list(active_speakers) != derived_active:
        debug.corrected_active_speakers = True

    peaks = [float(x) for x in (srp_doa_peaks_deg or [])]
    scores: list[float] | None = None
    if srp_peak_scores is not None:
        if len(srp_peak_scores) == len(peaks):
            scores = [float(x) for x in srp_peak_scores]
        else:
            debug.dropped_srp_scores = True

    payload = DirectionAssignmentInput(
        chunk_id=int(chunk_id),
        timestamp_ms=float(timestamp_ms),
        raw_mic_chunk=raw,
        separated_streams=norm_streams,
        stream_to_speaker=valid_map,
        active_speakers=derived_active,
        srp_doa_peaks_deg=peaks,
        srp_peak_scores=scores,
    )

    # Final strict check to keep adapter guarantees explicit.
    validate_balanced_payload(payload)
    return payload, debug
