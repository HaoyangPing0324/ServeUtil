"""PyTorch capture entry points. Torch is only imported when recording."""

import gzip
import inspect
import os
import shutil
import tempfile
import weakref
from collections.abc import Mapping
from pathlib import Path
from types import MethodType

from .bundle import Capture, write_json


def _failed_callback(profiler, capture, exc):
    # PyTorch invokes callbacks between stop_trace and the next prepare_trace.
    # If a callback raises during step(), __exit__ must not stop Kineto twice.
    from torch.profiler import ProfilerAction

    profiler.current_action = ProfilerAction.NONE
    profiler.step_rec_fn = None
    capture.finish(exc)


def _result(profiler):
    return getattr(profiler.profiler, "kineto_results", None) or profiler.profiler


def _export_cycle(capture, profiler):
    """Archive Kineto's complete single-use export and replay that cycle on demand."""
    rank = os.getenv("RANK", "0")
    cycle = len(capture.manifest["traces"])
    path = capture.path / "traces" / f"rank-{rank}-cycle-{cycle}.json.gz"
    # Keep the native JSON and compression spools in this capture's directory.
    native_export = getattr(
        profiler, "_serveutil_native_export", profiler.export_chrome_trace
    )
    result = _result(profiler)
    previous = getattr(profiler, "_serveutil_result", None)
    if previous is not None and previous() is result:
        shutil.copyfile(profiler._serveutil_archive, path)
    else:
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            raw = Path(tmp) / "kineto.json"
            native_export(str(raw))
            with raw.open("rb") as src, gzip.open(path, "wb", compresslevel=3) as dst:
                shutil.copyfileobj(src, dst)
    capture.add_trace(path, rank)
    profiler.serveutil_directory = capture.path
    profiler._serveutil_native_export = native_export
    profiler._serveutil_archive = path
    profiler._serveutil_result = weakref.ref(result)
    profiler.export_chrome_trace = MethodType(_export_saved, profiler)


def _export_saved(profiler, *args, **kwargs):
    # A newly prepared cycle may not yet have a result. Let torch handle it.
    native = profiler._serveutil_native_export
    signature = inspect.signature(native)
    options = signature.bind(*args, **kwargs).arguments
    dest = options.pop("path")
    if _result(profiler) is not profiler._serveutil_result() or any(
        value != signature.parameters[name].default for name, value in options.items()
    ):
        # Native alternatives (such as use_python_export=True) use their own writer.
        return native(*args, **kwargs)
    source, dest = profiler._serveutil_archive, Path(dest)
    if source.resolve() == dest.resolve():
        return
    if dest.suffix == ".gz":
        shutil.copyfile(source, dest)
    else:
        with gzip.open(source, "rb") as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)


def trace_handler(output="serveutil-runs", *, on_trace_ready=None):
    """Attach only an export callback to an existing native profiler."""

    def save(profiler):
        capture = Capture(output, "torch")
        try:
            _export_cycle(capture, profiler)
            if on_trace_ready is not None:
                on_trace_ready(profiler)
            capture.finish()
        except BaseException as exc:
            _failed_callback(profiler, capture, exc)
            raise

    return save


def _torch_devices(torch, devices):
    if not torch.cuda.is_available():
        return [] if devices is None else devices
    requested = [torch.cuda.current_device()] if devices is None else devices
    return [
        str(torch.cuda.get_device_properties(int(d)).uuid)
        if str(d).isdecimal()
        else str(d)
        for d in requested
    ]


def _baseline(torch, output, devices, calibration):
    if isinstance(calibration, (str, Path)) and calibration != "auto":
        import json

        with Path(calibration).open(encoding="utf-8") as stream:
            return json.load(stream), None
    if not calibration or not devices or not torch.cuda.is_available():
        return None, None
    from .calibration import calibrate

    try:
        return calibrate(devices=devices, cache_dir=Path(output) / ".calibration"), None
    except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
        if calibration != "auto":
            raise
        return None, str(exc)


def _set_metadata(capture, torch, devices, base, error):
    if base is not None:
        write_json(capture.path / "calibration.json", base)
    if error:
        capture.manifest["calibration_error"] = error
    if devices and torch.cuda.is_available():
        from .telemetry import _identifier

        selected = {_identifier(d) for d in devices}
        ids = [
            _identifier(torch.cuda.get_device_properties(i).uuid)
            for i in range(torch.cuda.device_count())
        ]
        capture.manifest["device_map"] = {
            str(i): uuid for i, uuid in enumerate(ids) if uuid in selected
        }


def _configure(torch, devices, metrics, options):
    from .telemetry import kineto_activities

    if metrics:
        if options.get("activities") is not None:
            raise ValueError("choose metrics or explicit activities")
        options["activities"] = kineto_activities(torch, metrics)
    activities = options.get("activities")
    if activities is not None and not isinstance(activities, Mapping):
        activities = options["activities"] = tuple(activities)
    if devices is None and (
        options.get("use_cuda") is False
        or (activities and torch.profiler.ProfilerActivity.CUDA not in activities)
    ):
        return []
    return _torch_devices(torch, devices)


def profile(
    output="serveutil-runs",
    *,
    devices=None,
    interval_ms=100,
    metrics=(),
    calibration="auto",
    **profiler_kwargs,
):
    """Return a native torch profiler and collect a single ServeUtil run across its schedule."""
    import torch
    from .telemetry import TelemetrySampler

    selected = _configure(torch, devices, metrics, profiler_kwargs)
    user_callback = profiler_kwargs.pop("on_trace_ready", None)
    capture = sampler = None

    def ready(profiler):
        try:
            _export_cycle(capture, profiler)
            if user_callback is not None:
                user_callback(profiler)
        except BaseException as exc:
            _failed_callback(profiler, capture, exc)
            raise

    profiler = torch.profiler.profile(on_trace_ready=ready, **profiler_kwargs)
    native_start, native_stop = profiler.start, profiler.stop

    def start():
        nonlocal capture, sampler
        if sampler is not None:
            raise RuntimeError("ServeUtil profiler is already recording")
        capture = Capture(output, "torch")
        profiler.serveutil_directory = capture.path
        try:
            base, error = _baseline(torch, output, selected, calibration)
            _set_metadata(capture, torch, selected, base, error)
            sampler = TelemetrySampler(
                capture.path / "samples.jsonl.gz",
                devices=selected,
                interval_ms=interval_ms,
                include_hbm=not bool(metrics),
            ).start()
            return native_start()
        except BaseException as exc:
            if sampler is not None:
                capture.manifest["telemetry"] = sampler.stop()
                sampler = None
            capture.finish(exc)
            raise

    def stop():
        nonlocal sampler
        if sampler is None:
            return native_stop()
        active, sampler = sampler, None
        try:
            try:
                result = native_stop()
            finally:
                capture.manifest["telemetry"] = active.stop()
            if capture.manifest["status"] == "recording":
                capture.finish()
            else:
                capture.save()
            return result
        except BaseException as exc:
            capture.finish(exc)
            if profiler.execution_trace_observer is not None:
                profiler.execution_trace_observer.cleanup()
            raise

    profiler.start, profiler.stop = start, stop
    return profiler


class _CycleObserver:
    """Collect one hardware run for each active execution-trace observer interval."""

    def __init__(
        self, output, devices, interval_ms, base, error, observer, callback, metrics
    ):
        self.output = output
        self.devices = devices
        self.interval_ms = interval_ms
        self.base, self.error = base, error
        self.observer = observer
        self.callback = callback
        self.include_hbm = not bool(metrics)
        self.capture = self.sampler = None

    def start(self):
        import torch
        from .telemetry import TelemetrySampler

        if self.capture is not None:
            self._discard()
        self.capture = Capture(self.output, "torch")
        try:
            _set_metadata(self.capture, torch, self.devices, self.base, self.error)
            self.sampler = TelemetrySampler(
                self.capture.path / "samples.jsonl.gz",
                self.devices,
                self.interval_ms,
                include_hbm=self.include_hbm,
            ).start()
            if self.observer is not None:
                self.observer.start()
        except BaseException as exc:
            self._stop_sampler()
            self.capture.finish(exc)
            self.capture = None
            raise

    def _stop_sampler(self):
        if self.sampler is not None:
            sampler, self.sampler = self.sampler, None
            self.capture.manifest["telemetry"] = sampler.stop()

    def stop(self):
        try:
            if self.observer is not None:
                self.observer.stop()
        finally:
            self._stop_sampler()

    def on_trace_ready(self, profiler):
        self._stop_sampler()
        if self.capture is None:
            raise RuntimeError(
                "ServeUtil observer did not start with the native profiler"
            )
        capture, self.capture = self.capture, None
        try:
            _export_cycle(capture, profiler)
            if self.callback is not None:
                self.callback(profiler)
            capture.finish()
        except BaseException as exc:
            _failed_callback(profiler, capture, exc)
            self.cleanup()
            raise

    def _discard(self):
        self._stop_sampler()
        if self.capture is not None:
            self.capture.manifest["note"] = (
                "Schedule ended without a saved native trace."
            )
            self.capture.finish()
            self.capture = None

    def cleanup(self):
        try:
            self._discard()
        finally:
            if self.observer is not None:
                observer, self.observer = self.observer, None
                observer.cleanup()


def instrument(
    output="serveutil-runs",
    *,
    devices=None,
    interval_ms=100,
    calibration="auto",
    metrics=(),
    **profiler_kwargs,
):
    """Return native torch.profiler.profile kwargs with a composed observer and trace callback."""
    import torch

    selected = _configure(torch, devices, metrics, profiler_kwargs)
    base, error = _baseline(torch, output, selected, calibration)
    native_observer = profiler_kwargs.get("execution_trace_observer")
    if native_observer is None:
        observer_type = getattr(torch.profiler, "ExecutionTraceObserver", None)
        factory = getattr(observer_type, "build_execution_trace_obs_from_env", None)
        if factory is not None:
            native_observer = factory()
    observer = _CycleObserver(
        output,
        selected,
        interval_ms,
        base,
        error,
        native_observer,
        profiler_kwargs.get("on_trace_ready"),
        metrics,
    )
    return {
        **profiler_kwargs,
        "execution_trace_observer": observer,
        "on_trace_ready": observer.on_trace_ready,
    }
