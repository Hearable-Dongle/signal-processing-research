from __future__ import annotations

from realtime_pipeline.contracts import PipelineConfig
from realtime_pipeline.separation_backends import MockSeparationBackend, build_default_backend


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
