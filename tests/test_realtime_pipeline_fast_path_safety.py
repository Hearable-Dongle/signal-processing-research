from __future__ import annotations

import numpy as np

from realtime_pipeline.contracts import PipelineConfig
from realtime_pipeline.fast_path import _apply_output_safety, _soft_clip


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
