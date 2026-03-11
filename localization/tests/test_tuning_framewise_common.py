from __future__ import annotations

from localization.tuning.framewise_common import (
    aggregate_framewise_rows,
    balanced_framewise_score,
    build_framewise_eval_config,
    hop_ms_to_samples,
    window_ms_to_samples,
)


def test_framewise_eval_config_sets_window_hop_and_freq_range() -> None:
    cfg = build_framewise_eval_config(
        base_cfg={"nfft": 512, "overlap": 0.5, "freq_range": [200, 3000]},
        fs=16000,
        window_ms=200,
        hop_ms=50,
        overlap=0.75,
        freq_low_hz=300,
        freq_high_hz=2500,
    )
    assert cfg["window_ms"] == 200
    assert cfg["hop_ms"] == 50
    assert cfg["nfft"] == 3200
    assert cfg["overlap"] == 0.75
    assert cfg["freq_range"] == [300, 2500]


def test_framewise_sample_converters_match_expected_sizes() -> None:
    assert window_ms_to_samples(50, 16000) == 800
    assert window_ms_to_samples(300, 16000) == 4800
    assert hop_ms_to_samples(30, 16000) == 480
    assert hop_ms_to_samples(100, 16000) == 1600


def test_aggregate_framewise_rows_computes_balanced_score() -> None:
    rows = [
        {
            "status": "ok",
            "mae_deg": 20.0,
            "acc_at_10": 0.4,
            "acc_at_25": 0.7,
            "coverage_rate": 0.95,
            "runtime_ms_mean": 12.0,
        },
        {
            "status": "ok",
            "mae_deg": 30.0,
            "acc_at_10": 0.3,
            "acc_at_25": 0.6,
            "coverage_rate": 0.90,
            "runtime_ms_mean": 10.0,
        },
    ]
    agg = aggregate_framewise_rows(rows)
    assert agg["n_scenes"] == 2
    assert agg["mae_deg_mean"] == 25.0
    assert agg["acc_at_25_mean"] == 0.6499999999999999
    assert agg["coverage_rate_mean"] == 0.925
    assert agg["balanced_score"] == balanced_framewise_score(mae_deg=25.0, acc25=0.6499999999999999, runtime_ms=11.0)
