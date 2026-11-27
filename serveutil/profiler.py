"""PyTorch Profiler's public API, with ServeUtil collection on profile()."""

import inspect as _inspect

from torch import profiler as _native

from .recording import _record_profile

__all__ = list(_native.__all__)

class _ProfileType(type(_native.profile)):
    def __call__(cls, *args, **kwargs):
        if cls is profile:
            # Preserve the exact native type received by external callbacks.
            return _record_profile(_native.profile, *args, **kwargs)
        return super().__call__(*args, **kwargs)

    def __instancecheck__(cls, instance):
        if cls is profile:
            return isinstance(instance, _native.profile)
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass):
        if cls is profile:
            return issubclass(subclass, _native.profile)
        return super().__subclasscheck__(subclass)


class profile(_native.profile, metaclass=_ProfileType):
    """Record native PyTorch profiler data together with ServeUtil measurements."""

    def __init__(self, *args, **kwargs):
        # Subclasses reach this constructor through their normal super() chain.
        native_init = super().__init__

        def initialize(**options):
            native_init(**options)
            return self

        _record_profile(initialize, *args, **kwargs)


# Tools that filter profiler options through inspect.signature should see the
# constructor of the installed torch version, including newly added options.
_constructor = _inspect.signature(_native.profile)
_collector_options = [
    p for p in _inspect.signature(_record_profile).parameters.values()
    if p.kind != _inspect.Parameter.VAR_KEYWORD and p.name != "create_profiler" and p.name not in _constructor.parameters
]
profile.__signature__ = _constructor.replace(
    parameters=[*_collector_options, *_constructor.parameters.values()],
    return_annotation=_native.profile,
)
profile.__init__.__signature__ = profile.__signature__.replace(
    parameters=[_inspect.signature(_native.profile.__init__).parameters["self"],
                *profile.__signature__.parameters.values()],
    return_annotation=_inspect.signature(_native.profile.__init__).return_annotation,
)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_native, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_native)))
