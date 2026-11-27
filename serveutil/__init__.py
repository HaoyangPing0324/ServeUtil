"""Collect and explore serving profiles."""

from .overview import export, kernel_label

__all__ = ["export", "kernel_label", "profile", "trace_handler", "instrument", "observe"]


def __getattr__(name):
    if name == "profile":
        from importlib.util import find_spec

        if find_spec("torch") is not None:
            from .profiler import profile

            return profile
    if name in {"profile", "trace_handler", "instrument", "observe"}:
        from . import recording

        return getattr(recording, name)
    if name == "profiler":
        from importlib import import_module

        return import_module(".profiler", __name__)
    raise AttributeError(name)
