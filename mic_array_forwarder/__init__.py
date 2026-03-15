from __future__ import annotations

__all__ = ["app"]


def __getattr__(name: str):
    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .app import app as _app

    return _app
