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
