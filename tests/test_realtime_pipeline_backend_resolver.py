from __future__ import annotations

import numpy as np

from realtime_pipeline.contracts import PipelineConfig
from realtime_pipeline.separation_backends import AsteroidConvTasNetBackend, MockSeparationBackend, build_default_backend


class _DummyBackend(MockSeparationBackend):
    pass


def test_build_default_backend_prefers_multispeaker(monkeypatch) -> None:
    cfg = PipelineConfig(prefer_multispeaker_module=True)
    called: list[str] = []

    def _ms(_cfg: PipelineConfig):
        called.append("ms")
        return _DummyBackend(n_streams=2)

    def _ast(_cfg: PipelineConfig):
        called.append("ast")
        return _DummyBackend(n_streams=3)

    monkeypatch.setattr("realtime_pipeline.separation_backends._try_multispeaker_backend", _ms)
    monkeypatch.setattr("realtime_pipeline.separation_backends._try_asteroid_backend", _ast)

    backend = build_default_backend(cfg)
    assert isinstance(backend, _DummyBackend)
    assert called == ["ms"]


def test_build_default_backend_falls_back_to_asteroid(monkeypatch) -> None:
    cfg = PipelineConfig(prefer_multispeaker_module=True)
    called: list[str] = []

    def _ms(_cfg: PipelineConfig):
        called.append("ms")
        raise RuntimeError("ms missing")

    def _ast(_cfg: PipelineConfig):
        called.append("ast")
        return _DummyBackend(n_streams=3)

    monkeypatch.setattr("realtime_pipeline.separation_backends._try_multispeaker_backend", _ms)
    monkeypatch.setattr("realtime_pipeline.separation_backends._try_asteroid_backend", _ast)

    backend = build_default_backend(cfg)
    assert isinstance(backend, _DummyBackend)
    assert called == ["ms", "ast"]


def test_build_default_backend_raises_when_no_backend(monkeypatch) -> None:
    cfg = PipelineConfig(prefer_multispeaker_module=False)

    def _ms(_cfg: PipelineConfig):
        raise RuntimeError("ms missing")

    def _ast(_cfg: PipelineConfig):
        raise RuntimeError("ast missing")

    monkeypatch.setattr("realtime_pipeline.separation_backends._try_multispeaker_backend", _ms)
    monkeypatch.setattr("realtime_pipeline.separation_backends._try_asteroid_backend", _ast)

    try:
        _ = build_default_backend(cfg)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        msg = str(exc)
        assert "No usable real separation backend found" in msg
        assert "ms missing" in msg
        assert "ast missing" in msg


def test_asteroid_backend_resamples_and_preserves_chunk_length(monkeypatch) -> None:
    class _FakeModel:
        def eval(self):
            return self

        def to(self, _device):
            return self

        def separate(self, tensor):
            x = tensor.squeeze(0)
            return x.new_tensor(np.stack([x.detach().cpu().numpy(), 0.5 * x.detach().cpu().numpy()], axis=0)[None, ...])

    def _fake_load(self):
        import torch

        self._torch = torch
        self._model = _FakeModel()

    monkeypatch.setattr(AsteroidConvTasNetBackend, "_load_model", _fake_load)

    backend = AsteroidConvTasNetBackend(
        model_name="fake/model",
        device="cpu",
        model_sample_rate_hz=8000,
        input_sample_rate_hz=16000,
        expected_num_sources=2,
    )
    x = np.sin(2.0 * np.pi * 220.0 * np.arange(3200, dtype=np.float32) / 16000.0)
    streams = backend.separate(x, expected_speakers=2)

    assert len(streams) == 2
    assert all(s.shape == x.shape for s in streams)


def test_asteroid_backend_expected_source_override_pads_or_trims(monkeypatch) -> None:
    class _FakeModel:
        def eval(self):
            return self

        def to(self, _device):
            return self

        def separate(self, tensor):
            x = tensor.squeeze(0)
            arr = np.stack(
                [
                    x.detach().cpu().numpy(),
                    0.5 * x.detach().cpu().numpy(),
                    0.25 * x.detach().cpu().numpy(),
                ],
                axis=0,
            )
            return x.new_tensor(arr[None, ...])

    def _fake_load(self):
        import torch

        self._torch = torch
        self._model = _FakeModel()

    monkeypatch.setattr(AsteroidConvTasNetBackend, "_load_model", _fake_load)

    backend = AsteroidConvTasNetBackend(
        model_name="fake/model",
        device="cpu",
        model_sample_rate_hz=8000,
        input_sample_rate_hz=16000,
        expected_num_sources=None,
    )
    x = np.ones(1600, dtype=np.float32)
    streams = backend.separate(x, expected_speakers=2)
    assert len(streams) == 2
