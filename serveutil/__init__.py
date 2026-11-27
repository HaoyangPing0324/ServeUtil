"""Collect and explore serving profiles."""

from .overview import export, kernel_label

__all__ = ["export", "kernel_label", "profile", "trace_handler", "instrument", "observe"]


def __getattr__(name):
    if name in {"profile", "trace_handler", "instrument", "observe"}:
        from . import recording

        return getattr(recording, name)
    if name == "profiler":
        from importlib import import_module

        return import_module(".profiler", __name__)
    raise AttributeError(name)
