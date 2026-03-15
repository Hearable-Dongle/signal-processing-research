from __future__ import annotations

import numpy as np

from mic_array_forwarder.models import SessionStartRequest
from realtime_pipeline.contracts import PipelineConfig
from realtime_pipeline.fast_path import _PostFilterState, _apply_output_safety, _soft_clip
from realtime_pipeline.output_enhancer import apply_output_enhancer_audio
from realtime_pipeline.session_runtime import build_pipeline_config_from_request


def test_soft_clip_limits_peak_amplitude() -> None:
    x = np.array([-3.0, -1.2, 0.0, 1.2, 3.0], dtype=np.float32)
    y = _soft_clip(x, drive=1.2)
    assert np.max(np.abs(y)) <= 1.0 + 1e-6
    assert np.max(np.abs(y)) < np.max(np.abs(x))


def test_rms_normalization_moves_toward_target() -> None:
    cfg = PipelineConfig(
        output_soft_clip_enabled=False,
        output_target_rms=0.2,
        output_rms_ema_alpha=1.0,
        output_rms_max_gain_db=12.0,
        output_allow_amplification=True,
    )
    x = np.ones(160, dtype=np.float32) * 0.05
    y, _ = _apply_output_safety(x, cfg, rms_gain_ema=1.0)
    rms_y = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    assert abs(rms_y - 0.2) < 1e-3


def test_disabled_rms_normalization_keeps_level_when_no_clipping() -> None:
    cfg = PipelineConfig(
        output_soft_clip_enabled=False,
        output_target_rms=None,
    )
    x = np.linspace(-0.25, 0.25, num=128, dtype=np.float32)
    y, gain = _apply_output_safety(x, cfg, rms_gain_ema=1.0)
    assert np.allclose(y, x)
    assert gain == 1.0


def test_attenuation_only_normalization_does_not_amplify() -> None:
    cfg = PipelineConfig(
        output_soft_clip_enabled=False,
        output_normalization_enabled=True,
        output_allow_amplification=False,
        output_target_rms=0.2,
        output_rms_ema_alpha=1.0,
    )
    x = np.ones(160, dtype=np.float32) * 0.4
    y, gain = _apply_output_safety(x, cfg, rms_gain_ema=1.0)
    assert gain <= 1.0 + 1e-6
    assert float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) <= float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-6


def test_postfilter_preserves_shape_and_finite_values() -> None:
    cfg = PipelineConfig()
    state = _PostFilterState(frame_samples=160, cfg=cfg)
    rng = np.random.default_rng(0)
    frame = rng.standard_normal(160).astype(np.float32) * 0.05
    out = state.process(frame, speech_activity=0.2)
    assert out.shape == frame.shape
    assert np.all(np.isfinite(out))


def test_postfilter_reduces_stationary_noise_after_adaptation() -> None:
    cfg = PipelineConfig(
        postfilter_noise_ema_alpha=0.2,
        postfilter_speech_ema_alpha=0.1,
        postfilter_gain_floor=0.12,
        postfilter_gain_ema_alpha=0.4,
        postfilter_dd_alpha=0.9,
        postfilter_freq_smoothing_bins=2,
    )
    state = _PostFilterState(frame_samples=160, cfg=cfg)
    rng = np.random.default_rng(1)
    frame = rng.standard_normal(160).astype(np.float32) * 0.05
    for _ in range(20):
        out = state.process(frame, speech_activity=0.0)
    rms_in = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert rms_out < rms_in


def test_output_enhancer_auto_mode_resolves_to_shared_wiener() -> None:
    cfg = PipelineConfig(sample_rate_hz=16000, fast_frame_ms=50, postfilter_enabled=True, output_enhancer_mode="auto")
    frame_samples = int(cfg.sample_rate_hz * cfg.fast_frame_ms / 1000)
    rng = np.random.default_rng(2)
    audio = (0.02 * rng.standard_normal(frame_samples * 3)).astype(np.float32)
    out, meta = apply_output_enhancer_audio(audio, cfg=cfg, frame_samples=frame_samples, speech_activity=0.0)
    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))
    assert meta.output_enhancer_mode == "shared_wiener"


def test_rnnoise_wet_mix_zero_keeps_input_when_unavailable_or_bypassed() -> None:
    cfg = PipelineConfig(
        sample_rate_hz=16000,
        fast_frame_ms=50,
        postfilter_enabled=False,
        output_enhancer_mode="rnnoise",
        rnnoise_wet_mix=0.0,
    )
    frame_samples = int(cfg.sample_rate_hz * cfg.fast_frame_ms / 1000)
    audio = np.linspace(-0.1, 0.1, num=frame_samples * 2, dtype=np.float32)
    out, _meta = apply_output_enhancer_audio(audio, cfg=cfg, frame_samples=frame_samples, speech_activity=1.0)
    assert out.shape == audio.shape
    assert np.allclose(out, audio, atol=1e-5)


def test_output_enhancer_supports_smaller_internal_frame_than_fast_hop() -> None:
    cfg = PipelineConfig(
        sample_rate_hz=16000,
        fast_frame_ms=50,
        postfilter_enabled=True,
        output_enhancer_mode="shared_wiener",
        output_enhancer_frame_ms=10.0,
    )
    outer_frame_samples = int(cfg.sample_rate_hz * cfg.fast_frame_ms / 1000)
    rng = np.random.default_rng(3)
    audio = (0.03 * rng.standard_normal(outer_frame_samples * 2)).astype(np.float32)
    out, meta = apply_output_enhancer_audio(audio, cfg=cfg, frame_samples=outer_frame_samples, speech_activity=0.5)
    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))
    assert meta.output_enhancer_mode == "shared_wiener"


def test_build_pipeline_config_maps_output_enhancer_fields() -> None:
    req = SessionStartRequest(
        sample_rate_hz=16000,
        localization_hop_ms=50,
        postfilter_enabled=False,
        postfilter_noise_ema_alpha=0.11,
        postfilter_gain_floor=0.18,
        output_enhancer_mode="shared_wiener_rnnoise",
        rnnoise_input_gain_db=3.0,
        rnnoise_wet_mix=0.8,
    )
    cfg = build_pipeline_config_from_request(req, sample_rate_hz=16000, max_speakers_hint=4)
    assert cfg.fast_frame_ms == 50
    assert cfg.postfilter_enabled is False
    assert cfg.postfilter_noise_ema_alpha == 0.11
    assert cfg.postfilter_gain_floor == 0.18
    assert cfg.output_enhancer_mode == "shared_wiener_rnnoise"
    assert cfg.rnnoise_input_gain_db == 3.0
    assert cfg.rnnoise_wet_mix == 0.8
