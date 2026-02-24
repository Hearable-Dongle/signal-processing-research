from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SceneConfig:
    sample_rate: int = 16000
    duration_sec: float = 12.0
    chunk_ms: int = 200
    n_speakers: int = 3
    n_mics: int = 4
    mic_spacing_m: float = 0.04
    noise_std: float = 0.01
    bleed_ratio: float = 0.12
    moving_probability: float = 0.6
    srp_peak_noise_deg: float = 5.0
    srp_num_distractors: int = 1


@dataclass
class EnrollmentConfig:
    identity_mode: str = "online_only"  # online_only|enroll_audio|seed_embeddings
    enroll_audio_manifest: str | None = None
    seed_embeddings_npz: str | None = None


@dataclass
class OutputConfig:
    out_dir: str
    export_audio: bool = True
    export_plots: bool = True


@dataclass
class ValidationConfig:
    num_scenes: int = 20
    seed: int = 7
    speaker_choices: list[int] = field(default_factory=lambda: [2, 3, 4])
    run_prior_comparison: bool = True
