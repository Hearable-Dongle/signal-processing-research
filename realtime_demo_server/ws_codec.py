from __future__ import annotations

import struct

import numpy as np

MAGIC = b"RTA1"
VERSION = 1
_HEADER = struct.Struct("<4sB3xQ")


def encode_audio_chunk(timestamp_ms: float, samples: np.ndarray) -> bytes:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    ts = int(max(0, round(float(timestamp_ms))))
    return _HEADER.pack(MAGIC, VERSION, ts) + mono.astype("<f4", copy=False).tobytes()


def decode_audio_chunk(payload: bytes) -> tuple[float, np.ndarray]:
    if len(payload) < _HEADER.size:
        raise ValueError("audio payload is too short")
    magic, version, ts = _HEADER.unpack(payload[: _HEADER.size])
    if magic != MAGIC:
        raise ValueError("invalid magic")
    if int(version) != VERSION:
        raise ValueError("unsupported audio payload version")
    data = payload[_HEADER.size :]
    if len(data) % 4 != 0:
        raise ValueError("invalid float32 payload length")
    return float(ts), np.frombuffer(data, dtype="<f4").astype(np.float32, copy=False)
