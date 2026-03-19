from __future__ import annotations

import numpy as np

from realtime_pipeline.streaming_adapter import RealtimeIntelligibilityAdapter


def _int16_multichannel(samples: int, channels: int = 4, freq_hz: float = 440.0, sample_rate_hz: int = 16000) -> list[np.ndarray]:
    t = np.arange(samples, dtype=np.float32) / float(sample_rate_hz)
    base = (0.2 * np.sin(2.0 * np.pi * float(freq_hz) * t) * 32767.0).astype(np.int16)
    return [base.copy() for _ in range(channels)]


def test_streaming_adapter_defaults_match_requested_realtime_shape() -> None:
    adapter = RealtimeIntelligibilityAdapter()
    try:
        assert adapter.request.localization_backend == "capon_1src"
        assert adapter.request.beamforming_mode == "delay_sum"
        assert adapter.request.postfilter_method == "rnnoise"
        assert adapter.request.separation_mode == "single_dominant_no_separator"
        assert adapter.request.single_active is True
        assert adapter.request.sample_rate_hz == 16000
        assert adapter.channel_count == 4
        assert adapter.frame_samples == 160
    finally:
        adapter.close()


def test_streaming_adapter_default_path_processes_audio() -> None:
    adapter = RealtimeIntelligibilityAdapter()
    try:
        emitted = 0
        for samples in (73, 211, 160, 97, 401, 120):
            emitted += int(adapter.process_chunk(_int16_multichannel(samples)).shape[0])
        emitted += int(adapter.flush().shape[0])
        assert emitted == sum((73, 211, 160, 97, 401, 120))
    finally:
        adapter.close()


def test_streaming_adapter_buffers_short_callback_until_frame_ready() -> None:
    adapter = RealtimeIntelligibilityAdapter(postfilter_method="off", postfilter_enabled=False)
    try:
        first = adapter.process_chunk(_int16_multichannel(80))
        second = adapter.process_chunk(_int16_multichannel(80))
        assert first.shape == (0,)
        assert second.shape == (adapter.frame_samples,)
        assert second.dtype == np.float32
    finally:
        adapter.close()


def test_streaming_adapter_accepts_variable_chunk_sizes() -> None:
    adapter = RealtimeIntelligibilityAdapter(postfilter_method="off", postfilter_enabled=False)
    try:
        out0 = adapter.process_chunk(_int16_multichannel(160))
        out1 = adapter.process_chunk(_int16_multichannel(320))
        out2 = adapter.process_chunk(_int16_multichannel(40))
        assert out0.shape == (adapter.frame_samples,)
        assert out1.shape == (adapter.frame_samples * 2,)
        assert out2.shape == (0,)
    finally:
        adapter.close()


def test_streaming_adapter_rejects_mismatched_channel_lengths() -> None:
    adapter = RealtimeIntelligibilityAdapter(postfilter_method="off", postfilter_enabled=False)
    try:
        bad = _int16_multichannel(160)
        bad[1] = bad[1][:120]
        try:
            adapter.process_chunk(bad)
        except ValueError as exc:
            assert "same number of samples" in str(exc)
        else:
            raise AssertionError("expected ValueError for mismatched channel lengths")
    finally:
        adapter.close()


def test_streaming_adapter_rejects_wrong_channel_count() -> None:
    adapter = RealtimeIntelligibilityAdapter(postfilter_method="off", postfilter_enabled=False)
    try:
        try:
            adapter.process_chunk(_int16_multichannel(160, channels=3))
        except ValueError as exc:
            assert "expected 4 channels" in str(exc)
        else:
            raise AssertionError("expected ValueError for wrong channel count")
    finally:
        adapter.close()


def test_streaming_adapter_resamples_to_processing_rate() -> None:
    adapter = RealtimeIntelligibilityAdapter(
        input_sample_rate_hz=48000,
        processing_sample_rate_hz=16000,
        enable_resample=True,
        postfilter_method="off",
        postfilter_enabled=False,
    )
    try:
        out = adapter.process_chunk(_int16_multichannel(480, sample_rate_hz=48000))
        assert out.shape == (adapter.frame_samples,)
        assert out.dtype == np.float32
        assert float(np.max(np.abs(out))) > 0.0
    finally:
        adapter.close()


def test_streaming_adapter_flushes_partial_frame() -> None:
    adapter = RealtimeIntelligibilityAdapter(postfilter_method="off", postfilter_enabled=False)
    try:
        _ = adapter.process_chunk(_int16_multichannel(120))
        flushed = adapter.flush()
        assert flushed.shape == (120,)
        assert flushed.dtype == np.float32
    finally:
        adapter.close()
