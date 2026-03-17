from __future__ import annotations

import threading
from collections import deque
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
    # Fast-path stage timings
    fast_srp_total_ms: float = 0.0
    fast_beamform_total_ms: float = 0.0
    fast_safety_total_ms: float = 0.0
    fast_sink_total_ms: float = 0.0
    fast_enqueue_total_ms: float = 0.0
    # Slow-path stage timings
    slow_separation_total_ms: float = 0.0
    slow_identity_total_ms: float = 0.0
    slow_direction_total_ms: float = 0.0
    slow_publish_total_ms: float = 0.0
    beamforming_frames: int = 0
    beamforming_total_ms: float = 0.0
    beamforming_p50_ms: float = 0.0
    beamforming_p95_ms: float = 0.0
    postfilter_frames: int = 0
    postfilter_total_ms: float = 0.0
    postfilter_p50_ms: float = 0.0
    postfilter_p95_ms: float = 0.0
    pipeline_frames: int = 0
    pipeline_total_ms: float = 0.0
    pipeline_p50_ms: float = 0.0
    pipeline_p95_ms: float = 0.0
    interstage_queue_wait_p50_ms: float = 0.0
    interstage_queue_wait_p95_ms: float = 0.0
    interstage_queue_depth_max: int = 0
    dropped_interstage_frames: int = 0
    end_to_end_latency_p50_ms: float = 0.0
    end_to_end_latency_p95_ms: float = 0.0


class SharedPipelineState:
    """Thread-safe state with lock-free snapshot reads for hot-path consumers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._speaker_map_ref: Mapping[int, SpeakerGainDirection] = MappingProxyType({})
        self._srp_ref: SRPPeakSnapshot = SRPPeakSnapshot(timestamp_ms=0.0, peaks_deg=(), peak_scores=None)
        self._focus_control_ref: FocusControlSnapshot = FocusControlSnapshot()
        self._stats = PipelineRuntimeStats()
        self._beamforming_samples_ms: deque[float] = deque(maxlen=4096)
        self._postfilter_samples_ms: deque[float] = deque(maxlen=4096)
        self._pipeline_samples_ms: deque[float] = deque(maxlen=4096)
        self._queue_wait_samples_ms: deque[float] = deque(maxlen=4096)
        self._latency_samples_ms: deque[float] = deque(maxlen=4096)

    @staticmethod
    def _percentile_from_samples(samples: deque[float], q: float) -> float:
        if not samples:
            return 0.0
        arr = sorted(float(v) for v in samples)
        idx = int(round((len(arr) - 1) * float(q)))
        idx = max(0, min(len(arr) - 1, idx))
        return float(arr[idx])

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

    def incr_fast_stage_times(
        self,
        *,
        srp_ms: float,
        beamform_ms: float,
        safety_ms: float,
        sink_ms: float,
        enqueue_ms: float,
    ) -> None:
        with self._lock:
            self._stats.fast_srp_total_ms += float(srp_ms)
            self._stats.fast_beamform_total_ms += float(beamform_ms)
            self._stats.fast_safety_total_ms += float(safety_ms)
            self._stats.fast_sink_total_ms += float(sink_ms)
            self._stats.fast_enqueue_total_ms += float(enqueue_ms)

    def incr_slow_stage_times(
        self,
        *,
        separation_ms: float,
        identity_ms: float,
        direction_ms: float,
        publish_ms: float,
    ) -> None:
        with self._lock:
            self._stats.slow_separation_total_ms += float(separation_ms)
            self._stats.slow_identity_total_ms += float(identity_ms)
            self._stats.slow_direction_total_ms += float(direction_ms)
            self._stats.slow_publish_total_ms += float(publish_ms)

    def incr_dropped_fast_to_slow(self, count: int = 1) -> None:
        with self._lock:
            self._stats.dropped_fast_to_slow_frames += int(count)

    def incr_dropped_interstage(self, count: int = 1) -> None:
        with self._lock:
            self._stats.dropped_interstage_frames += int(count)

    def record_beamforming_stage(self, elapsed_ms: float) -> None:
        with self._lock:
            self._stats.beamforming_frames += 1
            self._stats.beamforming_total_ms += float(elapsed_ms)
            self._beamforming_samples_ms.append(float(elapsed_ms))

    def record_postfilter_stage(self, elapsed_ms: float) -> None:
        with self._lock:
            self._stats.postfilter_frames += 1
            self._stats.postfilter_total_ms += float(elapsed_ms)
            self._postfilter_samples_ms.append(float(elapsed_ms))

    def record_pipeline_latency(self, *, queue_wait_ms: float, end_to_end_latency_ms: float, queue_depth: int) -> None:
        with self._lock:
            self._stats.pipeline_frames += 1
            self._stats.pipeline_total_ms += float(end_to_end_latency_ms)
            self._queue_wait_samples_ms.append(float(queue_wait_ms))
            self._latency_samples_ms.append(float(end_to_end_latency_ms))
            self._pipeline_samples_ms.append(float(end_to_end_latency_ms))
            self._stats.interstage_queue_depth_max = max(int(self._stats.interstage_queue_depth_max), int(queue_depth))

    def get_stats(self) -> PipelineRuntimeStats:
        with self._lock:
            return PipelineRuntimeStats(
                fast_frames=self._stats.fast_frames,
                slow_chunks=self._stats.slow_chunks,
                dropped_fast_to_slow_frames=self._stats.dropped_fast_to_slow_frames,
                speaker_map_updates=self._stats.speaker_map_updates,
                fast_total_ms=self._stats.fast_total_ms,
                slow_total_ms=self._stats.slow_total_ms,
                fast_srp_total_ms=self._stats.fast_srp_total_ms,
                fast_beamform_total_ms=self._stats.fast_beamform_total_ms,
                fast_safety_total_ms=self._stats.fast_safety_total_ms,
                fast_sink_total_ms=self._stats.fast_sink_total_ms,
                fast_enqueue_total_ms=self._stats.fast_enqueue_total_ms,
                slow_separation_total_ms=self._stats.slow_separation_total_ms,
                slow_identity_total_ms=self._stats.slow_identity_total_ms,
                slow_direction_total_ms=self._stats.slow_direction_total_ms,
                slow_publish_total_ms=self._stats.slow_publish_total_ms,
                beamforming_frames=self._stats.beamforming_frames,
                beamforming_total_ms=self._stats.beamforming_total_ms,
                beamforming_p50_ms=self._percentile_from_samples(self._beamforming_samples_ms, 0.5),
                beamforming_p95_ms=self._percentile_from_samples(self._beamforming_samples_ms, 0.95),
                postfilter_frames=self._stats.postfilter_frames,
                postfilter_total_ms=self._stats.postfilter_total_ms,
                postfilter_p50_ms=self._percentile_from_samples(self._postfilter_samples_ms, 0.5),
                postfilter_p95_ms=self._percentile_from_samples(self._postfilter_samples_ms, 0.95),
                pipeline_frames=self._stats.pipeline_frames,
                pipeline_total_ms=self._stats.pipeline_total_ms,
                pipeline_p50_ms=self._percentile_from_samples(self._pipeline_samples_ms, 0.5),
                pipeline_p95_ms=self._percentile_from_samples(self._pipeline_samples_ms, 0.95),
                interstage_queue_wait_p50_ms=self._percentile_from_samples(self._queue_wait_samples_ms, 0.5),
                interstage_queue_wait_p95_ms=self._percentile_from_samples(self._queue_wait_samples_ms, 0.95),
                interstage_queue_depth_max=self._stats.interstage_queue_depth_max,
                dropped_interstage_frames=self._stats.dropped_interstage_frames,
                end_to_end_latency_p50_ms=self._percentile_from_samples(self._latency_samples_ms, 0.5),
                end_to_end_latency_p95_ms=self._percentile_from_samples(self._latency_samples_ms, 0.95),
            )


class Timer:
    def __enter__(self) -> "Timer":
        self._start = perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (perf_counter() - self._start) * 1000.0
