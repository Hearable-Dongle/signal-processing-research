from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .contracts import FastPathAudioPacket, FocusControlSnapshot, PipelineConfig, SRPPeakSnapshot
from .fast_path import FastPathWorker, PostFilterWorker
from .separation_backends import SeparationBackend, build_default_backend
from .shared_state import SharedPipelineState
from .slow_path import SlowPathWorker


FrameSink = Callable[[np.ndarray], None]


@dataclass(slots=True)
class PipelineStatsSnapshot:
    fast_frames: int
    slow_chunks: int
    dropped_fast_to_slow_frames: int
    speaker_map_updates: int
    fast_avg_ms: float
    slow_avg_ms: float
    fast_rtf: float
    slow_rtf: float
    fast_srp_avg_ms: float
    fast_beamform_avg_ms: float
    fast_safety_avg_ms: float
    fast_sink_avg_ms: float
    fast_enqueue_avg_ms: float
    slow_separation_avg_ms: float
    slow_identity_avg_ms: float
    slow_direction_avg_ms: float
    slow_publish_avg_ms: float
    dropped_interstage_frames: int
    beamforming_avg_ms: float
    beamforming_rtf: float
    beamforming_p50_ms: float
    beamforming_p95_ms: float
    postfilter_avg_ms: float
    postfilter_rtf: float
    postfilter_p50_ms: float
    postfilter_p95_ms: float
    pipeline_avg_ms: float
    pipeline_rtf: float
    interstage_queue_wait_p50_ms: float
    interstage_queue_wait_p95_ms: float
    interstage_queue_depth_max: int
    end_to_end_latency_p50_ms: float
    end_to_end_latency_p95_ms: float


class RealtimeSpeakerPipeline:
    def __init__(
        self,
        *,
        config: PipelineConfig,
        mic_geometry_xyz: np.ndarray,
        mic_geometry_xy: np.ndarray,
        frame_iterator: Iterator[np.ndarray],
        frame_sink: FrameSink | None = None,
        separation_backend: SeparationBackend | None = None,
        srp_override_provider: Callable[[int, float], SRPPeakSnapshot | None] | None = None,
        target_activity_override_provider: Callable[[int, float], float | None] | None = None,
        oracle_noise_frame_provider: Callable[[int, float], np.ndarray | None] | None = None,
        frame_packet_iterator: Iterator[FastPathAudioPacket] | None = None,
        frame_packet_sink: Callable[[FastPathAudioPacket], None] | None = None,
    ):
        self.config = config
        self._state = SharedPipelineState()
        self._stop = threading.Event()
        self._slow_q: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=config.slow_queue_max_frames)
        self._postfilter_q: "queue.Queue[FastPathAudioPacket | None]" = queue.Queue(maxsize=max(1, int(config.postfilter_queue_max_frames)))
        self._iter = frame_iterator
        self._packet_iter = frame_packet_iterator
        self._frame_sink = frame_sink or (lambda _x: None)
        self._frame_packet_sink = frame_packet_sink or (lambda _pkt: None)
        self._sep = separation_backend or build_default_backend(config)
        split_mode = str(getattr(config, "split_runtime_mode", "monolithic")).strip().lower()

        self._fast = None
        self._postfilter = None
        if split_mode != "postfilter_only":
            self._fast = FastPathWorker(
                config=config,
                shared_state=self._state,
                frame_source=self._next_frame,
                frame_sink=self._frame_sink,
                slow_queue=self._slow_q,
                mic_geometry_xyz=mic_geometry_xyz,
                stop_event=self._stop,
                srp_override_provider=srp_override_provider,
                target_activity_override_provider=target_activity_override_provider,
                oracle_noise_frame_provider=oracle_noise_frame_provider,
                postfilter_queue=self._postfilter_q if split_mode == "pipelined" else None,
                frame_packet_sink=self._frame_packet_sink,
            )
        if split_mode in {"pipelined", "postfilter_only"}:
            packet_source = self._next_packet if split_mode == "postfilter_only" else None
            self._postfilter = PostFilterWorker(
                config=config,
                shared_state=self._state,
                frame_sink=self._frame_sink,
                stop_event=self._stop,
                packet_queue=None if split_mode == "postfilter_only" else self._postfilter_q,
                packet_source=packet_source,
                mic_geometry_xyz=mic_geometry_xyz,
            )
        self._slow = None
        if bool(config.slow_path_enabled) and split_mode != "postfilter_only":
            self._slow = SlowPathWorker(
                config=config,
                shared_state=self._state,
                slow_queue=self._slow_q,
                separation_backend=self._sep,
                mic_geometry_xy=mic_geometry_xy,
                stop_event=self._stop,
            )

    def _next_frame(self) -> np.ndarray | None:
        try:
            return next(self._iter)
        except StopIteration:
            return None

    def _next_packet(self) -> FastPathAudioPacket | None:
        if self._packet_iter is None:
            return None
        try:
            return next(self._packet_iter)
        except StopIteration:
            return None

    @property
    def shared_state(self) -> SharedPipelineState:
        return self._state

    def start(self) -> None:
        if self._fast is not None:
            self._fast.start()
        if self._postfilter is not None:
            self._postfilter.start()
        if self._slow is not None:
            self._slow.start()

    def set_focus_control(
        self,
        *,
        focused_speaker_ids: list[int] | None = None,
        focused_direction_deg: float | None = None,
        user_boost_db: float = 0.0,
    ) -> None:
        ids: tuple[int, ...] | None = None
        if focused_speaker_ids:
            ids = tuple(sorted({int(v) for v in focused_speaker_ids}))
        direction = None if focused_direction_deg is None else float(focused_direction_deg % 360.0)
        boost = float(np.clip(float(user_boost_db), 0.0, float(self.config.max_user_boost_db)))
        self._state.publish_focus_control(
            FocusControlSnapshot(
                focused_speaker_ids=ids,
                focused_direction_deg=direction,
                user_boost_db=boost,
            )
        )

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._fast is not None:
            self._fast.join(timeout=timeout)
        if self._postfilter is not None:
            self._postfilter.join(timeout=timeout)
        if self._slow is not None:
            self._slow.join(timeout=timeout)

    def run_blocking(self) -> None:
        self.start()
        self.join()

    def stats_snapshot(self) -> PipelineStatsSnapshot:
        st = self._state.get_stats()
        fast_avg = st.fast_total_ms / st.fast_frames if st.fast_frames else 0.0
        slow_avg = st.slow_total_ms / st.slow_chunks if st.slow_chunks else 0.0
        fast_srp_avg = st.fast_srp_total_ms / st.fast_frames if st.fast_frames else 0.0
        fast_beamform_avg = st.fast_beamform_total_ms / st.fast_frames if st.fast_frames else 0.0
        fast_safety_avg = st.fast_safety_total_ms / st.fast_frames if st.fast_frames else 0.0
        fast_sink_avg = st.fast_sink_total_ms / st.fast_frames if st.fast_frames else 0.0
        fast_enqueue_avg = st.fast_enqueue_total_ms / st.fast_frames if st.fast_frames else 0.0
        slow_sep_avg = st.slow_separation_total_ms / st.slow_chunks if st.slow_chunks else 0.0
        slow_ident_avg = st.slow_identity_total_ms / st.slow_chunks if st.slow_chunks else 0.0
        slow_dir_avg = st.slow_direction_total_ms / st.slow_chunks if st.slow_chunks else 0.0
        slow_pub_avg = st.slow_publish_total_ms / st.slow_chunks if st.slow_chunks else 0.0
        beamforming_avg = st.beamforming_total_ms / st.beamforming_frames if st.beamforming_frames else 0.0
        postfilter_avg = st.postfilter_total_ms / st.postfilter_frames if st.postfilter_frames else 0.0
        pipeline_avg = st.pipeline_total_ms / st.pipeline_frames if st.pipeline_frames else 0.0
        return PipelineStatsSnapshot(
            fast_frames=st.fast_frames,
            slow_chunks=st.slow_chunks,
            dropped_fast_to_slow_frames=st.dropped_fast_to_slow_frames,
            speaker_map_updates=st.speaker_map_updates,
            fast_avg_ms=float(fast_avg),
            slow_avg_ms=float(slow_avg),
            fast_rtf=float(fast_avg / max(self.config.fast_frame_ms, 1e-6)),
            slow_rtf=float(slow_avg / max(self.config.slow_chunk_ms, 1e-6)),
            fast_srp_avg_ms=float(fast_srp_avg),
            fast_beamform_avg_ms=float(fast_beamform_avg),
            fast_safety_avg_ms=float(fast_safety_avg),
            fast_sink_avg_ms=float(fast_sink_avg),
            fast_enqueue_avg_ms=float(fast_enqueue_avg),
            slow_separation_avg_ms=float(slow_sep_avg),
            slow_identity_avg_ms=float(slow_ident_avg),
            slow_direction_avg_ms=float(slow_dir_avg),
            slow_publish_avg_ms=float(slow_pub_avg),
            dropped_interstage_frames=int(st.dropped_interstage_frames),
            beamforming_avg_ms=float(beamforming_avg),
            beamforming_rtf=float(beamforming_avg / max(self.config.fast_frame_ms, 1e-6)),
            beamforming_p50_ms=float(st.beamforming_p50_ms),
            beamforming_p95_ms=float(st.beamforming_p95_ms),
            postfilter_avg_ms=float(postfilter_avg),
            postfilter_rtf=float(postfilter_avg / max(self.config.fast_frame_ms, 1e-6)),
            postfilter_p50_ms=float(st.postfilter_p50_ms),
            postfilter_p95_ms=float(st.postfilter_p95_ms),
            pipeline_avg_ms=float(pipeline_avg),
            pipeline_rtf=float(pipeline_avg / max(self.config.fast_frame_ms, 1e-6)),
            interstage_queue_wait_p50_ms=float(st.interstage_queue_wait_p50_ms),
            interstage_queue_wait_p95_ms=float(st.interstage_queue_wait_p95_ms),
            interstage_queue_depth_max=int(st.interstage_queue_depth_max),
            end_to_end_latency_p50_ms=float(st.end_to_end_latency_p50_ms),
            end_to_end_latency_p95_ms=float(st.end_to_end_latency_p95_ms),
        )
