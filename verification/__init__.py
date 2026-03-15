from __future__ import annotations

__all__ = ["run_all_verification"]


def run_all_verification(*args, **kwargs):
    from .run_all import run_all_verification as _run_all_verification

    return _run_all_verification(*args, **kwargs)
