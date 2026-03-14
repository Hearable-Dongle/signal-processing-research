from __future__ import annotations

import queue
import threading

import numpy as np

from realtime_pipeline.contracts import PipelineConfig, SRPPeakSnapshot
from realtime_pipeline.separation_backends import MockSeparationBackend
from realtime_pipeline.shared_state import SharedPipelineState
from realtime_pipeline.slow_path import SlowPathWorker


def _worker() -> tuple[SlowPathWorker, SharedPipelineState]:
    state = SharedPipelineState()
    worker = SlowPathWorker(
        config=PipelineConfig(tracking_mode="doa_centroid_v1", max_speakers_hint=2),
        shared_state=state,
        slow_queue=queue.Queue(),
        separation_backend=MockSeparationBackend(n_streams=2),
        mic_geometry_xy=np.zeros((4, 2), dtype=np.float64),
        stop_event=threading.Event(),
    )
    return worker, state


def test_centroid_tracker_updates_existing_track_with_nearby_peak() -> None:
    worker, state = _worker()
    state.publish_srp_snapshot(
        SRPPeakSnapshot(timestamp_ms=0.0, peaks_deg=(45.0,), peak_scores=(0.9,))
    )
    worker._process_centroid_frame(np.zeros((160, 4), dtype=np.float32))
    first = dict(state.get_speaker_map_snapshot())
    assert len(first) == 1
    speaker_id = next(iter(first))

    state.publish_srp_snapshot(
        SRPPeakSnapshot(timestamp_ms=200.0, peaks_deg=(55.0,), peak_scores=(0.85,))
    )
    worker._process_centroid_frame(np.zeros((160, 4), dtype=np.float32))
    second = dict(state.get_speaker_map_snapshot())
    assert len(second) == 1
    assert next(iter(second)) == speaker_id
    assert 45.0 < second[speaker_id].direction_degrees < 55.0


def test_centroid_tracker_creates_new_track_for_far_peak() -> None:
    worker, state = _worker()
    state.publish_srp_snapshot(
        SRPPeakSnapshot(timestamp_ms=0.0, peaks_deg=(45.0,), peak_scores=(0.9,))
    )
    worker._process_centroid_frame(np.zeros((160, 4), dtype=np.float32))

    state.publish_srp_snapshot(
        SRPPeakSnapshot(timestamp_ms=250.0, peaks_deg=(140.0,), peak_scores=(0.88,))
    )
    worker._process_centroid_frame(np.zeros((160, 4), dtype=np.float32))
    speaker_map = dict(state.get_speaker_map_snapshot())
    assert len(speaker_map) == 2


def test_centroid_tracker_gaussian_mode_absorbs_nearby_jitter() -> None:
    state = SharedPipelineState()
    worker = SlowPathWorker(
        config=PipelineConfig(
            tracking_mode="doa_centroid_v1",
            max_speakers_hint=1,
            centroid_association_mode="gaussian",
            centroid_association_sigma_deg=10.0,
            centroid_association_min_score=0.15,
            speaker_match_window_deg=33.0,
        ),
        shared_state=state,
        slow_queue=queue.Queue(),
        separation_backend=MockSeparationBackend(n_streams=2),
        mic_geometry_xy=np.zeros((4, 2), dtype=np.float64),
        stop_event=threading.Event(),
    )
    state.publish_srp_snapshot(
        SRPPeakSnapshot(timestamp_ms=0.0, peaks_deg=(90.0,), peak_scores=(0.9,))
    )
    worker._process_centroid_frame(np.zeros((160, 4), dtype=np.float32))
    state.publish_srp_snapshot(
        SRPPeakSnapshot(timestamp_ms=200.0, peaks_deg=(103.0,), peak_scores=(0.8,))
    )
    worker._process_centroid_frame(np.zeros((160, 4), dtype=np.float32))
    speaker_map = dict(state.get_speaker_map_snapshot())
    assert len(speaker_map) == 1
