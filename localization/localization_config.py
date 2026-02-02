from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class LocalizationConfig:
    nfft: int
    overlap: float
    epsilon: float
    d_freq: int
    freq_range: Tuple[int, int]
    max_sources: int
    algo_type: str = "SSZ"
    mdl_beta: float = 0.6
    power_thresh_percentile: float = 90.0
    output_dir: str | None = None
    model_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "LocalizationConfig":
        return cls(
            nfft=data.get("nfft", 512),
            overlap=data.get("overlap", 0.5),
            epsilon=data.get("epsilon", 0.1),
            d_freq=data.get("d_freq", 8),
            freq_range=tuple(data.get("freq_range", [200, 3000])),
            max_sources=data.get("max_sources", 2),
            algo_type=data.get("type", "SSZ"),
            mdl_beta=data.get("mdl_beta", 0.6),
            power_thresh_percentile=data.get("power_thresh_percentile", 90.0),
            output_dir=data.get("output_dir"),
            model_path=data.get("model_path")
        )
