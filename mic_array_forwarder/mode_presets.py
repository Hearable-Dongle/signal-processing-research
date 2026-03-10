from __future__ import annotations

from dataclasses import dataclass


METHOD_LOCALIZATION_ONLY = "localization_only"
METHOD_SPATIAL_BASELINE = "spatial_baseline"
METHOD_SPEAKER_TRACKING = "speaker_tracking"
METHOD_SPEAKER_TRACKING_LONG_MEMORY = "speaker_tracking_long_memory"
METHOD_SINGLE_DOMINANT_NO_SEPARATOR = "single_dominant_no_separator"


@dataclass(frozen=True)
class SimulationAlgorithmPreset:
    algorithm_mode: str
    control_mode: str
    fast_path_reference_mode: str
    direction_long_memory_enabled: bool
    direction_long_memory_window_ms: float
    use_single_dominant_no_separator: bool = False
    supports_ground_truth_speaker_sources: bool = True


@dataclass(frozen=True)
class LiveAlgorithmPreset:
    algorithm_mode: str
    max_sources: int
    match_angle_threshold_deg: float
    stale_track_ms: float
    inactive_hold_ms: float


@dataclass(frozen=True)
class LocalizationSmoothingPreset:
    srp_peak_min_score: float
    localization_min_relative_peak_score: float
    localization_min_peak_contrast: float
    localization_max_assoc_distance_deg: float
    localization_track_hold_frames: int
    localization_track_kill_frames: int
    localization_new_track_min_confidence: float
    localization_track_confidence_decay: float
    localization_velocity_alpha: float
    localization_angle_alpha: float
    direction_stable_confidence_threshold: float
    direction_large_change_persist_chunks: int
    direction_history_window_chunks: int
    speaker_map_min_confidence_for_refresh: float
    speaker_map_hold_ms: float
    doa_ema_alpha: float
    doa_max_step_deg_per_frame: float
    live_display_min_confidence: float
    live_display_min_score: float


LIGHT_SMOOTHING_PRESET = LocalizationSmoothingPreset(
    srp_peak_min_score=0.03,
    localization_min_relative_peak_score=0.24,
    localization_min_peak_contrast=0.06,
    localization_max_assoc_distance_deg=24.0,
    localization_track_hold_frames=4,
    localization_track_kill_frames=7,
    localization_new_track_min_confidence=0.35,
    localization_track_confidence_decay=0.90,
    localization_velocity_alpha=0.40,
    localization_angle_alpha=0.36,
    direction_stable_confidence_threshold=0.48,
    direction_large_change_persist_chunks=2,
    direction_history_window_chunks=3,
    speaker_map_min_confidence_for_refresh=0.12,
    speaker_map_hold_ms=180.0,
    doa_ema_alpha=0.28,
    doa_max_step_deg_per_frame=14.0,
    live_display_min_confidence=0.16,
    live_display_min_score=0.08,
)

BALANCED_SMOOTHING_PRESET = LocalizationSmoothingPreset(
    srp_peak_min_score=0.05,
    localization_min_relative_peak_score=0.28,
    localization_min_peak_contrast=0.08,
    localization_max_assoc_distance_deg=20.0,
    localization_track_hold_frames=5,
    localization_track_kill_frames=9,
    localization_new_track_min_confidence=0.42,
    localization_track_confidence_decay=0.88,
    localization_velocity_alpha=0.35,
    localization_angle_alpha=0.30,
    direction_stable_confidence_threshold=0.55,
    direction_large_change_persist_chunks=3,
    direction_history_window_chunks=4,
    speaker_map_min_confidence_for_refresh=0.20,
    speaker_map_hold_ms=300.0,
    doa_ema_alpha=0.20,
    doa_max_step_deg_per_frame=10.0,
    live_display_min_confidence=0.22,
    live_display_min_score=0.14,
)

STRONG_SMOOTHING_PRESET = LocalizationSmoothingPreset(
    srp_peak_min_score=0.08,
    localization_min_relative_peak_score=0.40,
    localization_min_peak_contrast=0.12,
    localization_max_assoc_distance_deg=14.0,
    localization_track_hold_frames=7,
    localization_track_kill_frames=12,
    localization_new_track_min_confidence=0.58,
    localization_track_confidence_decay=0.92,
    localization_velocity_alpha=0.22,
    localization_angle_alpha=0.18,
    direction_stable_confidence_threshold=0.68,
    direction_large_change_persist_chunks=4,
    direction_history_window_chunks=6,
    speaker_map_min_confidence_for_refresh=0.35,
    speaker_map_hold_ms=450.0,
    doa_ema_alpha=0.12,
    doa_max_step_deg_per_frame=6.0,
    live_display_min_confidence=0.32,
    live_display_min_score=0.22,
)


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _interp_presets(left: LocalizationSmoothingPreset, right: LocalizationSmoothingPreset, t: float) -> LocalizationSmoothingPreset:
    return LocalizationSmoothingPreset(
        srp_peak_min_score=_lerp(left.srp_peak_min_score, right.srp_peak_min_score, t),
        localization_min_relative_peak_score=_lerp(left.localization_min_relative_peak_score, right.localization_min_relative_peak_score, t),
        localization_min_peak_contrast=_lerp(left.localization_min_peak_contrast, right.localization_min_peak_contrast, t),
        localization_max_assoc_distance_deg=_lerp(left.localization_max_assoc_distance_deg, right.localization_max_assoc_distance_deg, t),
        localization_track_hold_frames=int(round(_lerp(left.localization_track_hold_frames, right.localization_track_hold_frames, t))),
        localization_track_kill_frames=int(round(_lerp(left.localization_track_kill_frames, right.localization_track_kill_frames, t))),
        localization_new_track_min_confidence=_lerp(left.localization_new_track_min_confidence, right.localization_new_track_min_confidence, t),
        localization_track_confidence_decay=_lerp(left.localization_track_confidence_decay, right.localization_track_confidence_decay, t),
        localization_velocity_alpha=_lerp(left.localization_velocity_alpha, right.localization_velocity_alpha, t),
        localization_angle_alpha=_lerp(left.localization_angle_alpha, right.localization_angle_alpha, t),
        direction_stable_confidence_threshold=_lerp(left.direction_stable_confidence_threshold, right.direction_stable_confidence_threshold, t),
        direction_large_change_persist_chunks=int(round(_lerp(left.direction_large_change_persist_chunks, right.direction_large_change_persist_chunks, t))),
        direction_history_window_chunks=int(round(_lerp(left.direction_history_window_chunks, right.direction_history_window_chunks, t))),
        speaker_map_min_confidence_for_refresh=_lerp(left.speaker_map_min_confidence_for_refresh, right.speaker_map_min_confidence_for_refresh, t),
        speaker_map_hold_ms=_lerp(left.speaker_map_hold_ms, right.speaker_map_hold_ms, t),
        doa_ema_alpha=_lerp(left.doa_ema_alpha, right.doa_ema_alpha, t),
        doa_max_step_deg_per_frame=_lerp(left.doa_max_step_deg_per_frame, right.doa_max_step_deg_per_frame, t),
        live_display_min_confidence=_lerp(left.live_display_min_confidence, right.live_display_min_confidence, t),
        live_display_min_score=_lerp(left.live_display_min_score, right.live_display_min_score, t),
    )


def get_localization_smoothing_preset(level: float) -> LocalizationSmoothingPreset:
    value = float(max(0.0, min(1.0, float(level))))
    if value <= 0.5:
        return _interp_presets(LIGHT_SMOOTHING_PRESET, BALANCED_SMOOTHING_PRESET, value / 0.5 if value > 0.0 else 0.0)
    return _interp_presets(BALANCED_SMOOTHING_PRESET, STRONG_SMOOTHING_PRESET, (value - 0.5) / 0.5)


def get_simulation_algorithm_preset(algorithm_mode: str) -> SimulationAlgorithmPreset:
    mode = str(algorithm_mode).strip().lower()
    if mode == METHOD_LOCALIZATION_ONLY:
        return SimulationAlgorithmPreset(
            algorithm_mode=mode,
            control_mode="spatial_peak_mode",
            fast_path_reference_mode="srp_peak",
            direction_long_memory_enabled=False,
            direction_long_memory_window_ms=2000.0,
            supports_ground_truth_speaker_sources=False,
        )
    if mode == METHOD_SPATIAL_BASELINE:
        return SimulationAlgorithmPreset(
            algorithm_mode=mode,
            control_mode="spatial_peak_mode",
            fast_path_reference_mode="speaker_map",
            direction_long_memory_enabled=False,
            direction_long_memory_window_ms=2000.0,
        )
    if mode == METHOD_SPEAKER_TRACKING:
        return SimulationAlgorithmPreset(
            algorithm_mode=mode,
            control_mode="speaker_tracking_mode",
            fast_path_reference_mode="speaker_map",
            direction_long_memory_enabled=False,
            direction_long_memory_window_ms=2000.0,
        )
    if mode == METHOD_SPEAKER_TRACKING_LONG_MEMORY:
        return SimulationAlgorithmPreset(
            algorithm_mode=mode,
            control_mode="speaker_tracking_mode",
            fast_path_reference_mode="speaker_map",
            direction_long_memory_enabled=True,
            direction_long_memory_window_ms=60000.0,
        )
    if mode == METHOD_SINGLE_DOMINANT_NO_SEPARATOR:
        return SimulationAlgorithmPreset(
            algorithm_mode=mode,
            control_mode="speaker_tracking_mode",
            fast_path_reference_mode="speaker_map",
            direction_long_memory_enabled=False,
            direction_long_memory_window_ms=2000.0,
            use_single_dominant_no_separator=True,
            supports_ground_truth_speaker_sources=False,
        )
    raise ValueError(f"Unsupported simulation algorithm_mode: {algorithm_mode}")


def get_live_algorithm_preset(algorithm_mode: str) -> LiveAlgorithmPreset:
    mode = str(algorithm_mode).strip().lower()
    if mode == METHOD_LOCALIZATION_ONLY:
        return LiveAlgorithmPreset(
            algorithm_mode=mode,
            max_sources=1,
            match_angle_threshold_deg=18.0,
            stale_track_ms=900.0,
            inactive_hold_ms=600.0,
        )
    if mode == METHOD_SPATIAL_BASELINE:
        return LiveAlgorithmPreset(
            algorithm_mode=mode,
            max_sources=3,
            match_angle_threshold_deg=25.0,
            stale_track_ms=1500.0,
            inactive_hold_ms=1200.0,
        )
    if mode == METHOD_SPEAKER_TRACKING:
        return LiveAlgorithmPreset(
            algorithm_mode=mode,
            max_sources=3,
            match_angle_threshold_deg=20.0,
            stale_track_ms=2200.0,
            inactive_hold_ms=1800.0,
        )
    if mode == METHOD_SPEAKER_TRACKING_LONG_MEMORY:
        return LiveAlgorithmPreset(
            algorithm_mode=mode,
            max_sources=3,
            match_angle_threshold_deg=18.0,
            stale_track_ms=4000.0,
            inactive_hold_ms=3200.0,
        )
    if mode == METHOD_SINGLE_DOMINANT_NO_SEPARATOR:
        return LiveAlgorithmPreset(
            algorithm_mode=mode,
            max_sources=1,
            match_angle_threshold_deg=16.0,
            stale_track_ms=2600.0,
            inactive_hold_ms=2200.0,
        )
    raise ValueError(f"Unsupported live algorithm_mode: {algorithm_mode}")
