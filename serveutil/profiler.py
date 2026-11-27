"""PyTorch Profiler's public API, with ServeUtil collection on profile()."""

from torch import profiler as _native

from .recording import profile

__all__ = list(_native.__all__)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_native, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_native)))
