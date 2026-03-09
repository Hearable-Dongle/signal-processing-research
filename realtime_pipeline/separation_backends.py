from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Protocol

import numpy as np
from scipy import signal

from .contracts import PipelineConfig


class SeparationBackend(Protocol):
    def separate(self, mono_chunk: np.ndarray, expected_speakers: int | None = None) -> list[np.ndarray]:
        ...


@dataclass(slots=True)
class MockSeparationBackend:
    """Deterministic fast backend for tests/integration smoke."""

    n_streams: int = 2

    def separate(self, mono_chunk: np.ndarray, expected_speakers: int | None = None) -> list[np.ndarray]:
        x = np.asarray(mono_chunk, dtype=np.float32).reshape(-1)
        n = int(expected_speakers) if expected_speakers and expected_speakers > 0 else self.n_streams
        n = max(1, n)
        out: list[np.ndarray] = []
        for i in range(n):
            scale = 1.0 - (0.05 * i)
            out.append((scale * x).astype(np.float32, copy=False))
        return out


class MultispeakerModuleBackend:
    """Thin adapter over multispeaker_separation.SpeakerSeparationSystem."""

    def __init__(self, model_dir: str, backend: str = "pytorch", max_speakers_hint: int = 8) -> None:
        self._model_dir = model_dir
        self._backend = backend
        self._max_speakers = max(1, int(max_speakers_hint))
        self._torch = None
        self._system = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from multispeaker_separation.inference import SpeakerSeparationSystem
        except Exception as exc:
            raise RuntimeError(
                "multispeaker_separation backend unavailable. "
                "Ensure multispeaker_separation.inference and SpeakerSeparationSystem are present."
            ) from exc

        self._torch = torch
        self._system = SpeakerSeparationSystem(self._model_dir, backend=self._backend)

    def separate(self, mono_chunk: np.ndarray, expected_speakers: int | None = None) -> list[np.ndarray]:
        if self._system is None or self._torch is None:
            raise RuntimeError("Multispeaker separation system was not initialized")

        x = np.asarray(mono_chunk, dtype=np.float32).reshape(-1)
        t = self._torch.tensor(x, dtype=self._torch.float32).unsqueeze(0)

        n = int(expected_speakers) if expected_speakers and expected_speakers > 0 else 1
        n = max(1, min(self._max_speakers, n))

        est = self._system.separate(t, n)
        if hasattr(est, "detach"):
            arr = est.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            arr = np.asarray(est, dtype=np.float32)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim == 3:
            arr = np.squeeze(arr, axis=0)
        elif arr.ndim > 3:
            raise RuntimeError(f"unexpected separation output shape from multispeaker backend: {arr.shape}")

        streams: list[np.ndarray] = []
        for i in range(arr.shape[0]):
            s = np.asarray(arr[i], dtype=np.float32).reshape(-1)
            if s.shape[0] > x.shape[0]:
                s = s[: x.shape[0]]
            elif s.shape[0] < x.shape[0]:
                s = np.pad(s, (0, x.shape[0] - s.shape[0]))
            streams.append(s)

        return streams


class AsteroidConvTasNetBackend:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        *,
        model_sample_rate_hz: int = 16000,
        input_sample_rate_hz: int = 16000,
        resample_mode: str = "polyphase",
        expected_num_sources: int | None = None,
    ) -> None:
        self._device = device
        self._model_name = model_name
        self._model_sample_rate_hz = max(1, int(model_sample_rate_hz))
        self._input_sample_rate_hz = max(1, int(input_sample_rate_hz))
        self._resample_mode = str(resample_mode)
        self._expected_num_sources = None if expected_num_sources is None else max(1, int(expected_num_sources))
        self._torch = None
        self._model = None
        self._load_model()

    @staticmethod
    def _add_local_asteroid_site_packages() -> bool:
        repo_root = Path(__file__).resolve().parent.parent
        candidates = sorted((repo_root / "asteroid" / "asteroid-env" / "lib").glob("python*/site-packages"))
        added = False
        for site_dir in candidates:
            site_str = str(site_dir.resolve())
            if site_str not in sys.path:
                sys.path.insert(0, site_str)
                added = True
        if added:
            sys.modules.pop("asteroid", None)
        return added

    def _resample_1d(self, x: np.ndarray, *, from_hz: int, to_hz: int, target_len: int | None = None) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if from_hz == to_hz:
            out = arr.copy()
        elif self._resample_mode != "polyphase":
            raise ValueError(f"Unsupported convtasnet_resample_mode: {self._resample_mode}")
        else:
            gcd = int(np.gcd(int(from_hz), int(to_hz)))
            up = int(to_hz // gcd)
            down = int(from_hz // gcd)
            out = signal.resample_poly(arr, up=up, down=down).astype(np.float32, copy=False)
        if target_len is None:
            return out
        if out.shape[0] > target_len:
            return out[:target_len]
        if out.shape[0] < target_len:
            return np.pad(out, (0, target_len - out.shape[0])).astype(np.float32, copy=False)
        return out

    def _load_model(self) -> None:
        try:
            import torch
            from asteroid.models import ConvTasNet
        except Exception as exc:
            added = self._add_local_asteroid_site_packages()
            if not added:
                raise RuntimeError(
                    "Asteroid ConvTasNet backend unavailable. Install asteroid and torch in the active environment."
                ) from exc
            try:
                import torch
                from asteroid.models import ConvTasNet
            except Exception as exc2:
                raise RuntimeError(
                    "Asteroid ConvTasNet backend unavailable. Install asteroid and torch in the active environment."
                ) from exc2

        self._torch = torch
        orig_torch_load = torch.load

        def _torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_torch_load(*args, **kwargs)

        torch.load = _torch_load_compat
        try:
            model = ConvTasNet.from_pretrained(self._model_name)
        finally:
            torch.load = orig_torch_load
        self._model = model.eval().to(torch.device(self._device))

    def separate(self, mono_chunk: np.ndarray, expected_speakers: int | None = None) -> list[np.ndarray]:
        if self._model is None or self._torch is None:
            raise RuntimeError("ConvTasNet model was not initialized")

        x = np.asarray(mono_chunk, dtype=np.float32).reshape(-1)
        x_model = self._resample_1d(
            x,
            from_hz=self._input_sample_rate_hz,
            to_hz=self._model_sample_rate_hz,
        )
        t = self._torch.tensor(x_model, dtype=self._torch.float32, device=self._device).unsqueeze(0)

        with self._torch.no_grad():
            est = self._model.separate(t)

        arr = est.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        if arr.ndim == 1:
            arr = arr[None, :]

        streams = [
            self._resample_1d(
                arr[i].reshape(-1),
                from_hz=self._model_sample_rate_hz,
                to_hz=self._input_sample_rate_hz,
                target_len=x.shape[0],
            )
            for i in range(arr.shape[0])
        ]

        target_n = self._expected_num_sources if self._expected_num_sources is not None else expected_speakers
        if target_n is None or target_n <= 0:
            return streams

        target_n = int(target_n)
        if len(streams) > target_n:
            rms = [float(np.sqrt(np.mean(s**2) + 1e-12)) for s in streams]
            keep = np.argsort(rms)[::-1][:target_n]
            streams = [streams[int(i)] for i in keep]
        elif len(streams) < target_n:
            zeros = np.zeros_like(streams[0]) if streams else np.zeros_like(x)
            for _ in range(target_n - len(streams)):
                streams.append(zeros.copy())

        return streams


def probe_backend_support(config: PipelineConfig | None = None) -> dict[str, dict[str, object]]:
    cfg = config or PipelineConfig()
    report: dict[str, dict[str, object]] = {}

    try:
        from multispeaker_separation.inference import SpeakerSeparationSystem  # type: ignore

        report["multispeaker_module"] = {
            "available": True,
            "details": f"SpeakerSeparationSystem={SpeakerSeparationSystem.__name__}",
            "model_dir": cfg.multispeaker_model_dir,
            "backend": cfg.multispeaker_backend,
        }
    except Exception as exc:
        report["multispeaker_module"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "model_dir": cfg.multispeaker_model_dir,
            "backend": cfg.multispeaker_backend,
        }

    try:
        from asteroid.models import ConvTasNet  # type: ignore

        report["asteroid"] = {
            "available": True,
            "details": f"ConvTasNet={ConvTasNet.__name__}",
            "model_name": cfg.convtasnet_model_name,
            "device": cfg.convtasnet_device,
            "model_sample_rate_hz": cfg.convtasnet_model_sample_rate_hz,
            "input_sample_rate_hz": cfg.convtasnet_input_sample_rate_hz,
        }
    except Exception as exc:
        fallback_added = AsteroidConvTasNetBackend._add_local_asteroid_site_packages()
        try:
            from asteroid.models import ConvTasNet  # type: ignore

            report["asteroid"] = {
                "available": True,
                "details": f"ConvTasNet={ConvTasNet.__name__}",
                "model_name": cfg.convtasnet_model_name,
                "device": cfg.convtasnet_device,
                "model_sample_rate_hz": cfg.convtasnet_model_sample_rate_hz,
                "input_sample_rate_hz": cfg.convtasnet_input_sample_rate_hz,
                "fallback_env": bool(fallback_added),
            }
        except Exception as exc2:
            report["asteroid"] = {
                "available": False,
                "error": f"{type(exc2).__name__}: {exc2}",
                "model_name": cfg.convtasnet_model_name,
                "device": cfg.convtasnet_device,
                "model_sample_rate_hz": cfg.convtasnet_model_sample_rate_hz,
                "input_sample_rate_hz": cfg.convtasnet_input_sample_rate_hz,
                "fallback_env": bool(fallback_added),
                "initial_error": f"{type(exc).__name__}: {exc}",
            }

    return report


def _try_multispeaker_backend(config: PipelineConfig) -> SeparationBackend:
    return MultispeakerModuleBackend(
        model_dir=config.multispeaker_model_dir,
        backend=config.multispeaker_backend,
        max_speakers_hint=config.max_speakers_hint,
    )


def _try_asteroid_backend(config: PipelineConfig) -> SeparationBackend:
    return AsteroidConvTasNetBackend(
        model_name=config.convtasnet_model_name,
        device=config.convtasnet_device,
        model_sample_rate_hz=config.convtasnet_model_sample_rate_hz,
        input_sample_rate_hz=config.convtasnet_input_sample_rate_hz,
        resample_mode=config.convtasnet_resample_mode,
        expected_num_sources=config.convtasnet_expected_num_sources,
    )


def build_default_backend(config: PipelineConfig) -> SeparationBackend:
    attempts: list[str] = []

    order = ["multispeaker", "asteroid"] if config.prefer_multispeaker_module else ["asteroid", "multispeaker"]

    for name in order:
        try:
            if name == "multispeaker":
                return _try_multispeaker_backend(config)
            return _try_asteroid_backend(config)
        except Exception as exc:
            attempts.append(f"{name}: {type(exc).__name__}: {exc}")

    report = probe_backend_support(config)
    msg = (
        "No usable real separation backend found. "
        "Tried backends in order: " + ", ".join(order) + ". "
        "Failures: " + " | ".join(attempts) + ". "
        f"Probe report: {report}"
    )
    raise RuntimeError(msg)
