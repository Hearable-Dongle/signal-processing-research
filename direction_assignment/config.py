from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DirectionAssignmentConfig:
    # Audio/STFT settings
    sample_rate: int = 16000
    chunk_ms: int = 200
    n_fft: int = 512
    win_length: int = 320
    hop_length: int = 160

    # Mask-backprojection
    mask_floor: float = 1e-3
    mask_power: float = 1.0

    # DOA estimation
    sound_speed_m_s: float = 343.0
    min_pair_baseline_m: float = 0.02
    pair_max_lag_scale: float = 1.1
    min_pair_coherence: float = 0.2
    doa_grid_step_deg: float = 1.0
    doa_refine_span_deg: float = 2.0
    doa_refine_step_deg: float = 0.25

    # Fusion / tracking
    srp_snap_tolerance_deg: float = 15.0
    doa_ema_alpha: float = 0.3
    min_stream_rms: float = 0.01
    min_confidence_for_update: float = 0.15
    min_confidence_for_switch: float = 0.55
    transition_penalty_deg: float = 35.0
    hold_confidence_decay: float = 0.9
    stale_confidence_decay: float = 0.96
    min_persist_confidence: float = 0.05
    max_angular_jump_deg_per_chunk: float | None = 30.0

    # Speaker lifecycle
    speaker_stale_timeout_ms: float = 2000.0
    speaker_forget_timeout_ms: float = 8000.0

    # Weight policy
    focus_gain_db: float = 0.0
    non_focus_attenuation_db: float = -14.0
    max_user_boost_db: float = 12.0
