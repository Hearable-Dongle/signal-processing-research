from .contracts import FocusControlSnapshot, PipelineConfig, SRPPeakSnapshot, SpeakerGainDirection
from .orchestrator import RealtimeSpeakerPipeline
from .sanity_checks import run_sanity_checks
from .separation_backends import (
    AsteroidConvTasNetBackend,
    MockSeparationBackend,
    MultispeakerModuleBackend,
    SeparationBackend,
    build_default_backend,
    probe_backend_support,
)
from .streaming_adapter import RealtimeIntelligibilityAdapter

__all__ = [
    "PipelineConfig",
    "FocusControlSnapshot",
    "SRPPeakSnapshot",
    "SpeakerGainDirection",
    "RealtimeSpeakerPipeline",
    "SeparationBackend",
    "MultispeakerModuleBackend",
    "AsteroidConvTasNetBackend",
    "MockSeparationBackend",
    "build_default_backend",
    "probe_backend_support",
    "RealtimeIntelligibilityAdapter",
    "run_sanity_checks",
    "run_focus_sanity_check",
    "run_simulation_pipeline",
]


def __getattr__(name: str):
    if name == "run_focus_sanity_check":
        from .focus_sanity_check import run_focus_sanity_check

        return run_focus_sanity_check
    if name == "run_simulation_pipeline":
        from .simulation_runner import run_simulation_pipeline

        return run_simulation_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
