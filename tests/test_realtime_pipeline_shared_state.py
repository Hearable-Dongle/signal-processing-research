from __future__ import annotations

import threading

from realtime_pipeline.contracts import FocusControlSnapshot, SRPPeakSnapshot, SpeakerGainDirection
from realtime_pipeline.shared_state import SharedPipelineState


def test_shared_state_snapshot_publish_and_read() -> None:
    st = SharedPipelineState()

    st.publish_srp_snapshot(SRPPeakSnapshot(timestamp_ms=100.0, peaks_deg=(10.0, 80.0), peak_scores=(0.9, 0.7)))
    st.publish_speaker_map(
        {
            1: SpeakerGainDirection(1, 45.0, 1.0, 0.8, True, 0.9, 100.0),
            2: SpeakerGainDirection(2, 120.0, 0.2, 0.5, False, 0.2, 100.0),
        }
    )

    srp = st.get_srp_snapshot()
    smap = st.get_speaker_map_snapshot()

    assert srp.peaks_deg == (10.0, 80.0)
    assert set(smap.keys()) == {1, 2}
    assert smap[1].direction_degrees == 45.0


def test_shared_state_concurrent_read_write_does_not_break() -> None:
    st = SharedPipelineState()
    stop = threading.Event()
    errors: list[Exception] = []

    def writer() -> None:
        i = 0
        while not stop.is_set():
            try:
                st.publish_speaker_map(
                    {
                        i % 3: SpeakerGainDirection(i % 3, float(i % 360), 1.0, 1.0, True, 1.0, float(i)),
                    }
                )
                st.publish_srp_snapshot(SRPPeakSnapshot(timestamp_ms=float(i), peaks_deg=(float(i % 360),), peak_scores=(1.0,)))
                i += 1
            except Exception as exc:  # pragma: no cover
                errors.append(exc)
                break

    def reader() -> None:
        for _ in range(5000):
            try:
                _ = st.get_speaker_map_snapshot()
                _ = st.get_srp_snapshot()
            except Exception as exc:  # pragma: no cover
                errors.append(exc)
                break

    wt = threading.Thread(target=writer)
    rt = threading.Thread(target=reader)
    wt.start()
    rt.start()
    rt.join(timeout=3.0)
    stop.set()
    wt.join(timeout=3.0)

    assert not errors


def test_shared_state_focus_control_snapshot_publish_and_read() -> None:
    st = SharedPipelineState()
    st.publish_focus_control(FocusControlSnapshot(focused_speaker_ids=(1, 3), focused_direction_deg=90.0, user_boost_db=6.0))
    snap = st.get_focus_control_snapshot()
    assert snap.focused_speaker_ids == (1, 3)
    assert snap.focused_direction_deg == 90.0
    assert snap.user_boost_db == 6.0
