"""PyTorch Profiler's public API, with ServeUtil collection on profile()."""

import inspect as _inspect

from torch import profiler as _native

from .recording import profile

__all__ = list(_native.__all__)

# Tools that filter profiler options through inspect.signature should see the
# constructor of the installed torch version, including newly added options.
_constructor = _inspect.signature(_native.profile)
_collector_options = [
    p for p in _inspect.signature(profile).parameters.values()
    if p.kind != _inspect.Parameter.VAR_KEYWORD and p.name not in _constructor.parameters
]
profile.__signature__ = _constructor.replace(
    parameters=[*_collector_options, *_constructor.parameters.values()],
    return_annotation=_native.profile,
)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_native, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_native)))
