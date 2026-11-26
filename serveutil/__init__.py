"""Collect and explore serving profiles."""

from .overview import export, kernel_label

__all__ = ["export", "kernel_label", "profile", "trace_handler", "instrument"]


def __getattr__(name):
    if name in {"profile", "trace_handler", "instrument"}:
        from . import recording

        return getattr(recording, name)
    raise AttributeError(name)
