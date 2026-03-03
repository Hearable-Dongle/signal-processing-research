from __future__ import annotations

import threading
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

from .contracts import FocusControlSnapshot, SRPPeakSnapshot, SpeakerGainDirection


@dataclass(slots=True)
class PipelineRuntimeStats:
    fast_frames: int = 0
    slow_chunks: int = 0
    dropped_fast_to_slow_frames: int = 0
    speaker_map_updates: int = 0
    fast_total_ms: float = 0.0
    slow_total_ms: float = 0.0


class SharedPipelineState:
    """Thread-safe state with lock-free snapshot reads for hot-path consumers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._speaker_map_ref: Mapping[int, SpeakerGainDirection] = MappingProxyType({})
        self._srp_ref: SRPPeakSnapshot = SRPPeakSnapshot(timestamp_ms=0.0, peaks_deg=(), peak_scores=None)
        self._focus_control_ref: FocusControlSnapshot = FocusControlSnapshot()
        self._stats = PipelineRuntimeStats()

    def get_speaker_map_snapshot(self) -> Mapping[int, SpeakerGainDirection]:
        return self._speaker_map_ref

    def publish_speaker_map(self, speaker_map: dict[int, SpeakerGainDirection]) -> None:
        snap = MappingProxyType(dict(speaker_map))
        with self._lock:
            self._speaker_map_ref = snap
            self._stats.speaker_map_updates += 1

    def get_srp_snapshot(self) -> SRPPeakSnapshot:
        return self._srp_ref

    def publish_srp_snapshot(self, snapshot: SRPPeakSnapshot) -> None:
        with self._lock:
            self._srp_ref = snapshot

    def get_focus_control_snapshot(self) -> FocusControlSnapshot:
        return self._focus_control_ref

    def publish_focus_control(self, snapshot: FocusControlSnapshot) -> None:
        with self._lock:
            self._focus_control_ref = snapshot

    def incr_fast_frame(self, elapsed_ms: float) -> None:
        with self._lock:
            self._stats.fast_frames += 1
            self._stats.fast_total_ms += float(elapsed_ms)

    def incr_slow_chunk(self, elapsed_ms: float) -> None:
        with self._lock:
            self._stats.slow_chunks += 1
            self._stats.slow_total_ms += float(elapsed_ms)

    def incr_dropped_fast_to_slow(self, count: int = 1) -> None:
        with self._lock:
            self._stats.dropped_fast_to_slow_frames += int(count)

    def get_stats(self) -> PipelineRuntimeStats:
        with self._lock:
            return PipelineRuntimeStats(
                fast_frames=self._stats.fast_frames,
                slow_chunks=self._stats.slow_chunks,
                dropped_fast_to_slow_frames=self._stats.dropped_fast_to_slow_frames,
                speaker_map_updates=self._stats.speaker_map_updates,
                fast_total_ms=self._stats.fast_total_ms,
                slow_total_ms=self._stats.slow_total_ms,
            )


class Timer:
    def __enter__(self) -> "Timer":
        self._start = perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (perf_counter() - self._start) * 1000.0
