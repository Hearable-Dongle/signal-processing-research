from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .contracts import FocusControlSnapshot, PipelineConfig
from .fast_path import FastPathWorker
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
    ):
        self.config = config
        self._state = SharedPipelineState()
        self._stop = threading.Event()
        self._slow_q: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=config.slow_queue_max_frames)
        self._iter = frame_iterator
        self._frame_sink = frame_sink or (lambda _x: None)
        self._sep = separation_backend or build_default_backend(config)

        self._fast = FastPathWorker(
            config=config,
            shared_state=self._state,
            frame_source=self._next_frame,
            frame_sink=self._frame_sink,
            slow_queue=self._slow_q,
            mic_geometry_xyz=mic_geometry_xyz,
            stop_event=self._stop,
        )
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

    @property
    def shared_state(self) -> SharedPipelineState:
        return self._state

    def start(self) -> None:
        self._fast.start()
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
        self._fast.join(timeout=timeout)
        self._slow.join(timeout=timeout)

    def run_blocking(self) -> None:
        self.start()
        self.join()

    def stats_snapshot(self) -> PipelineStatsSnapshot:
        st = self._state.get_stats()
        fast_avg = st.fast_total_ms / st.fast_frames if st.fast_frames else 0.0
        slow_avg = st.slow_total_ms / st.slow_chunks if st.slow_chunks else 0.0
        return PipelineStatsSnapshot(
            fast_frames=st.fast_frames,
            slow_chunks=st.slow_chunks,
            dropped_fast_to_slow_frames=st.dropped_fast_to_slow_frames,
            speaker_map_updates=st.speaker_map_updates,
            fast_avg_ms=float(fast_avg),
            slow_avg_ms=float(slow_avg),
        )
