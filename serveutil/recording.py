"""PyTorch capture entry points. Torch is only imported when recording."""

import gzip
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from .bundle import Capture, write_json


def _callback(capture):
    def save(prof):
        cycle = len(capture.manifest["traces"])
        rank = os.getenv("RANK", "0")
        path = capture.path / "traces" / f"rank-{rank}-cycle-{cycle}.json.gz"
        prof.export_chrome_trace(str(path))
        capture.add_trace(path, rank)
        return path

    return save


def trace_handler(output="serveutil-runs"):
    """Use as on_trace_ready in an existing torch.profiler.profile."""

    def save(prof):
        capture = Capture(output, "torch")
        try:
            _callback(capture)(prof)
            capture.finish()
        except BaseException as exc:
            capture.finish(exc)
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


@contextmanager
def profile(
    output="serveutil-runs",
    *,
    devices=None,
    interval_ms=100,
    metrics=(),
    calibration="auto",
    **profiler_kwargs,
):
    """Yield a native torch profiler. Its step(), schedule and summaries work as usual.

    A supplied on_trace_ready callback can read summaries and export the same native cycle.
    Set calibration=False to skip the local microbenchmarks.
    """
    import torch

    from .telemetry import TelemetrySampler, kineto_activities

    capture = Capture(output, "torch")
    sampling = None
    try:
        devices = _torch_devices(torch, devices)
        base, error = _baseline(torch, output, devices, calibration)
        _set_metadata(capture, torch, devices, base, error)
        if metrics:
            if "activities" in profiler_kwargs:
                raise ValueError("choose metrics or explicit activities")
            profiler_kwargs["activities"] = kineto_activities(torch, metrics)
        user_callback = profiler_kwargs.pop("on_trace_ready", None)
        export_cycle = _callback(capture)

        def ready(prof):
            path = export_cycle(prof)
            if user_callback is None:
                return

            # Kineto allows a native trace to be exported once. Replay that exact
            # export for torch.tensorboard_trace_handler and custom callbacks.
            def export_again(destination):
                destination = Path(destination)
                if destination.resolve() == path.resolve():
                    return
                if destination.suffix == ".gz":
                    shutil.copyfile(path, destination)
                else:
                    with gzip.open(path, "rb") as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)

            previous = vars(prof).get("export_chrome_trace")
            prof.export_chrome_trace = export_again
            try:
                user_callback(prof)
            finally:
                if previous is None:
                    del prof.export_chrome_trace
                else:
                    prof.export_chrome_trace = previous

        sampling = TelemetrySampler(
            capture.path / "samples.jsonl.gz",
            devices=devices,
            interval_ms=interval_ms,
            include_hbm=not bool(metrics),
        ).start()
        with torch.profiler.profile(
            on_trace_ready=ready, **profiler_kwargs
        ) as profiler:
            profiler.serveutil_directory = capture.path
            yield profiler
        capture.manifest["telemetry"] = sampling.stop()
        sampling = None
        capture.finish()
    except BaseException as exc:
        if sampling is not None:
            capture.manifest["telemetry"] = sampling.stop()
        capture.finish(exc)
        raise


class _CycleObserver:
    """Torch starts and stops this observer on each active Kineto recording cycle."""

    def __init__(self, output, devices, interval_ms, base, error):
        self.output = output
        self.devices = devices
        self.interval_ms = interval_ms
        self.base, self.error = base, error
        self.capture = None
        self.sampler = None

    def start(self):
        from .telemetry import TelemetrySampler

        if self.capture is not None:
            self.cleanup()
        self.capture = Capture(self.output, "torch")
        import torch

        selected = self.devices
        try:
            _set_metadata(self.capture, torch, selected, self.base, self.error)
            self.sampler = TelemetrySampler(
                self.capture.path / "samples.jsonl.gz", selected, self.interval_ms
            ).start()
        except BaseException as exc:
            self.capture.finish(exc)
            self.capture = None
            raise

    def stop(self):
        if self.sampler is not None:
            sampler, self.sampler = self.sampler, None
            self.capture.manifest["telemetry"] = sampler.stop()

    def on_trace_ready(self, profiler):
        self.stop()
        if self.capture is None:
            raise RuntimeError(
                "ServeUtil observer did not start with the native profiler"
            )
        capture, self.capture = self.capture, None
        try:
            _callback(capture)(profiler)
            capture.finish()
        except BaseException as exc:
            capture.finish(exc)
            raise

    def cleanup(self):
        self.stop()
        if self.capture:
            self.capture.finish("Profiler ended without a completed trace cycle")
            self.capture = None


def instrument(
    output="serveutil-runs", *, devices=None, interval_ms=100, calibration="auto"
):
    """Expand into torch.profiler.profile(**serveutil.instrument(...))."""
    import torch

    selected = _torch_devices(torch, devices)
    base, error = _baseline(torch, output, selected, calibration)
    observer = _CycleObserver(output, selected, interval_ms, base, error)
    return {
        "execution_trace_observer": observer,
        "on_trace_ready": observer.on_trace_ready,
    }
