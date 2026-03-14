from __future__ import annotations

SUPPORTED_TRACKING_MODE = "doa_centroid_v1"
DEPRECATED_TRACKING_MODES = {
    "legacy",
    "multi_peak_v2",
    "dominant_lock_v1",
}
TRACKING_MODE_CHOICES = [SUPPORTED_TRACKING_MODE, *sorted(DEPRECATED_TRACKING_MODES)]


def validate_tracking_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in DEPRECATED_TRACKING_MODES:
        raise ValueError(
            f"tracking_mode '{normalized}' is deprecated; use '{SUPPORTED_TRACKING_MODE}' instead"
        )
    if normalized != SUPPORTED_TRACKING_MODE:
        raise ValueError(
            f"Unsupported tracking_mode '{mode}'; expected '{SUPPORTED_TRACKING_MODE}'"
        )
    return normalized
