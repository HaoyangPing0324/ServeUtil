"""PyTorch capture entry points. Torch is only imported when recording."""

import gzip
import inspect
import json
import os
import shutil
import stat
import tempfile
import threading
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from .bundle import Capture, write_json


def _failed_callback(profiler, capture, exc):
    # PyTorch invokes callbacks between stop_trace and the next prepare_trace.
    # If a callback raises during step(), __exit__ must not stop Kineto twice.
    from torch.profiler import ProfilerAction

    profiler.current_action = ProfilerAction.NONE
    profiler.step_rec_fn = None
    capture.finish(exc)


def _result(profiler):
    backend = getattr(profiler, "profiler", None)
    return getattr(backend, "kineto_results", None) or backend


def _watch_profiler_artifacts(profiler):
    def file_version(path):
        try:
            info = path.stat()
        except (OSError, ValueError):
            return None
        if stat.S_ISREG(info.st_mode):
            return info.st_ino, info.st_size, info.st_ctime_ns, info.st_mtime_ns
        return None

    def wrap(writer, kind):
        @wraps(writer)
        def export(*args, **kwargs):
            path = args[0] if args else kwargs.get("path")
            path = Path(os.fsdecode(path)) if isinstance(path, (str, bytes, os.PathLike)) else None
            before = file_version(path) if path is not None else None
            value = writer(*args, **kwargs)
            window = getattr(profiler, "_serveutil_artifact_window", None)
            if window and (result := window[0]()) is not None and result is _result(profiler):
                _, capture, cycle = window
                if path is not None and file_version(path) not in (None, before):
                    try:
                        capture.archive_profiler_artifact(cycle, kind, path)
                    except OSError as exc:
                        capture.warn_export(path, exc)
            return value

        export._serveutil_artifact_writer = True
        return export

    for name in ("export_stacks", "export_memory_timeline"):
        writer = getattr(profiler, name, None)
        if writer is not None and not getattr(writer, "_serveutil_artifact_writer", False):
            setattr(profiler, name, wrap(writer, name))


@contextmanager
def _preserve_cuspy_format(capture, profiler, json_path, binary):
    """Archive both native encodings from one finalized window.

    The Cuspy writer discards its CPU snapshot and GPU columns on first export.
    Its finalizer runs on the exporting thread. Observe just this instance and
    leave every other profiler and its callbacks untouched.
    """
    observer = getattr(profiler, "_cuspy_profiler_observer", None)
    finalize = getattr(observer, "_maybe_write", None)
    deferred = {}
    if finalize is None:
        yield deferred
        return
    from torch.profiler._cuspy.trace import merge_trace_window_into_chrome_trace

    selected_window = profiler._cuspy_window_id
    gate = threading.RLock()
    previous = vars(observer).get("_maybe_write")

    def restore():
        if previous is None:
            del observer._maybe_write
        else:
            observer._maybe_write = previous

    def finalize_formats(window_id):
        if window_id != selected_window:
            return finalize(window_id)
        # set_export runs on the training thread. A detached observer can join
        # on a different thread, even after its profiler has moved to a new cycle.
        with gate:
            with observer._lock:
                window = observer._windows.get(window_id)
                if not window or window["built"] is None or window["out"] is None:
                    return
                cpu, built, output = window["cpu"], window["built"], Path(window["out"])
            primary, companion = (binary, json_path) if ".pftrace" in str(output) else (json_path, binary)
            try:
                try:
                    merge_trace_window_into_chrome_trace(
                        cpu, companion, built, trace_name=str(companion),
                        pftrace_compression_level=observer._pftrace_compression_level,
                    )
                except Exception as exc:
                    companion.unlink(missing_ok=True)
                    capture.warn_export(companion, exc)
                result = finalize(window_id)
                if output.resolve() != primary.resolve():
                    shutil.copyfile(output, primary)
            except BaseException as exc:
                if deferred:
                    capture.complete_export(deferred["entry"], json_path, binary, exc)
                raise
            else:
                if deferred:
                    capture.complete_export(deferred["entry"], json_path, binary)
                return result
            finally:
                if deferred:
                    restore()

    observer._maybe_write = finalize_formats
    try:
        yield deferred
    finally:
        with gate:
            with observer._lock:
                window = observer._windows.get(selected_window)
                pending = window is not None and window["out"] is not None
            if pending and profiler._cuspy_profiler_observer is not observer:
                deferred["entry"] = capture.add_trace(
                    json_path, os.getenv("RANK", "0"), pending=True,
                    activities=[activity.name for activity in profiler.activities],
                )
            else:
                restore()


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
    binary = path.with_suffix("").with_suffix(".pftrace.gz")
    previous = getattr(profiler, "_serveutil_result", None)
    if previous is not None and previous() is result:
        if profiler._serveutil_archive.is_file():
            shutil.copyfile(profiler._serveutil_archive, path)
        if profiler._serveutil_binary.is_file():
            shutil.copyfile(profiler._serveutil_binary, binary)
    else:
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            raw = Path(tmp) / "kineto.json"
            if getattr(profiler, "_use_cuspy", False):
                # This backend always writes gzip and may complete its export asynchronously.
                with _preserve_cuspy_format(capture, profiler, path, binary):
                    native_export(str(path))
                    profiler.wait_for_exports()
                if not path.exists():
                    capture.manifest["note"] = (
                        "The native CUDA observer produced no trace for this cycle."
                    )
                    capture.save()
                    return
            else:
                native_export(str(raw))
                with (
                    raw.open("rb") as src,
                    gzip.open(path, "wb", compresslevel=3) as dst,
                ):
                    shutil.copyfileobj(src, dst)
    _register_export(capture, profiler, path, binary, native_export, result)


def _register_export(capture, profiler, path, binary, native_export, result):
    profiler.serveutil_directory = capture.path
    primary = path if path.exists() else binary
    metadata = {}
    if primary == binary:
        metadata["native_output"] = True
    elif binary.exists():
        metadata["native_binary"] = str(binary.relative_to(capture.path))
    entry = capture.add_trace(primary, os.getenv("RANK", "0"),
                              activities=[activity.name for activity in profiler.activities], **metadata)
    _bind_export(profiler, path, binary, native_export, result, capture, entry)


def _bind_export(profiler, path, binary, native_export, result, capture, entry):
    profiler._serveutil_export_entry = (capture, entry)
    profiler._serveutil_native_export = native_export
    profiler._serveutil_archive = path
    profiler._serveutil_binary = binary
    profiler._serveutil_result = weakref.ref(result)

    @wraps(native_export)
    def export(*args, **kwargs):
        return _export_saved(profiler, *args, **kwargs)

    profiler.export_chrome_trace = export


def _export_saved(profiler, *args, **kwargs):
    # A newly prepared cycle may not yet have a result. Let torch handle it.
    native = profiler._serveutil_native_export
    signature = inspect.signature(native)
    options = signature.bind(*args, **kwargs).arguments
    dest = options.pop("path")
    # Empty annotations and a stream number alone select the standard Kineto writer.
    if not options.get("cuda_graph_annotations"):
        options.pop("cuda_graph_annotations", None)
    if (
        not options.get("use_python_export")
        and not options.get("cuda_graph_annotations")
        and options.get("graph_lanes", "none") == "none"
    ):
        options.pop("default_stream", None)
    cuspy = getattr(profiler, "_use_cuspy", False)
    wants_binary = cuspy and ".pftrace" in str(dest)
    source = profiler._serveutil_binary if wants_binary else profiler._serveutil_archive
    capture, entry = profiler._serveutil_export_entry
    if (
        entry.get("pending") or entry.get("export_error")
        or not source.is_file()
        or _result(profiler) is not profiler._serveutil_result()
    ):
        return native(*args, **kwargs)
    if not cuspy and any(value != signature.parameters[name].default for name, value in options.items()):
        result = native(*args, **kwargs)
        capture.archive_native_export(entry, Path(dest))
        return result
    if cuspy and not str(dest).endswith(".gz"):
        dest = str(dest) + ".gz"
    dest = Path(dest)
    if source.resolve() == dest.resolve():
        return
    if dest.suffix == ".gz":
        shutil.copyfile(source, dest)
    else:
        with gzip.open(source, "rb") as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)


def _complete_cycle(capture, profiler, callback):
    _watch_profiler_artifacts(profiler)
    result = _result(profiler)
    if result is not None:
        profiler._serveutil_artifact_window = (
            weakref.ref(result), capture, len(capture.manifest["traces"])
        )
    if not callback:
        return _export_cycle(capture, profiler)
    if not getattr(profiler, "_use_cuspy", False):
        _export_cycle(capture, profiler)
        return callback(profiler)

    # Cuspy hands each window to one writer. Give the caller first choice of
    # format and preserve its native export instead of consuming the window.
    destinations = []
    native_export = getattr(profiler, "_serveutil_native_export", profiler.export_chrome_trace)
    previous = vars(profiler).get("export_chrome_trace")

    @wraps(native_export)
    def export(*args, **kwargs):
        path = inspect.signature(native_export).bind(*args, **kwargs).arguments["path"]
        result = native_export(*args, **kwargs)
        destinations.append(
            Path(path if str(path).endswith(".gz") else str(path) + ".gz")
        )
        return result

    profiler.serveutil_directory = capture.path
    index = len(capture.manifest["traces"])
    stem = f"rank-{os.getenv('RANK', '0')}-cycle-{index}"
    path = capture.path / "traces" / (stem + ".json.gz")
    binary = path.with_suffix("").with_suffix(".pftrace.gz")
    profiler.export_chrome_trace = export
    try:
        with _preserve_cuspy_format(capture, profiler, path, binary) as deferred:
            callback(profiler)
            if destinations:
                profiler.wait_for_exports()
    finally:
        if previous is None:
            del profiler.export_chrome_trace
        else:
            profiler.export_chrome_trace = previous
    if deferred:
        _bind_export(profiler, path, binary, native_export, _result(profiler), capture, deferred["entry"])
        return
    if not destinations:
        _export_cycle(capture, profiler)
        return
    if path.is_file() or binary.is_file():
        _register_export(capture, profiler, path, binary, native_export, _result(profiler))
        return
    for source in dict.fromkeys(destinations):
        if not source.is_file():
            capture.manifest["note"] = (
                "The native writer has not produced its output. For caller-managed deferred writers, finish the export inside serveutil.observe."
            )
            continue
        index = len(capture.manifest["traces"])
        folder = capture.path / "traces" / f"cycle-{index}"
        folder.mkdir()
        saved = folder / source.name
        shutil.copyfile(source, saved)
        capture.manifest["traces"].append(
            dict(
                file=str(saved.relative_to(capture.path)),
                rank=os.getenv("RANK", "0"),
                native_output=True,
            )
        )
    capture.save()


def trace_handler(output="serveutil-runs", *, on_trace_ready=None):
    """Attach only an export callback to an existing native profiler."""

    def save(profiler):
        capture = Capture(output, "torch")
        try:
            _complete_cycle(capture, profiler, on_trace_ready)
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
    cuda = torch.profiler.ProfilerActivity.CUDA
    if devices is None and (
        options.get("use_cuda") is False
        or (
            activities
            and not any(
                cuda in item if isinstance(item, Mapping) else item == cuda
                for item in activities
            )
        )
    ):
        return []
    return _torch_devices(torch, devices)


def _independent_hbm(torch, options):
    # A native PM or Cuspy configuration owns the CUDA profiler on this device.
    extensions = tuple(
        cls
        for name in ("CuspyConfig", "PerformanceMetricsConfig")
        if (cls := getattr(torch.profiler, name, None)) is not None
    )
    cuda = torch.profiler.ProfilerActivity.CUDA
    activities = options.get("activities") or ()
    for item in (activities,) if isinstance(activities, Mapping) else activities:
        if isinstance(item, Mapping):
            config = item.get(cuda)
            if any(
                isinstance(ext, extensions)
                for ext in getattr(config, "profiler_configs", ())
            ):
                return False
    legacy = getattr(options.get("experimental_config"), "custom_profiler_config", None)
    if not legacy:
        return True
    try:
        config = json.loads(legacy)
    except (TypeError, ValueError):
        return legacy != "cuspy"
    return not isinstance(config, dict) or not (
        config.get("backend") == "cuspy"
        or config.get("cuspy")
        or config.get("pm_metrics")
        or config.get("enable_pm_sampling")
    )


@contextmanager
def observe(
    output="serveutil-runs",
    *,
    devices=None,
    interval_ms=100,
    calibration="auto",
    include_hbm=True,
):
    """Sample hardware around an unmodified profiler. Yield its new capture directory.

    Point native exporters at the traces subdirectory. Complete deferred writes
    before leaving the context. Every original file remains in that directory.
    """
    import torch
    from .telemetry import TelemetrySampler

    capture = Capture(output, "torch")
    try:
        selected = _torch_devices(torch, devices)
        base, error = _baseline(torch, output, selected, calibration)
        _set_metadata(capture, torch, selected, base, error)
        sampler = TelemetrySampler(
            capture.path / "samples.jsonl.gz",
            selected,
            interval_ms,
            include_hbm=include_hbm,
        ).start()
        try:
            yield capture.path
        finally:
            capture.manifest["telemetry"] = sampler.stop()
            folder = capture.path / "traces"
            for file in sorted(folder.rglob("*")):
                if not file.is_file() or not file.resolve().is_relative_to(
                    folder.resolve()
                ):
                    continue
                # Every file is retained; the summary understands Chrome JSON.
                capture.manifest["traces"].append(
                    {
                        "file": str(file.relative_to(capture.path)),
                        "rank": os.getenv("RANK", "0"),
                        "native_output": True,
                    }
                )
    except BaseException as exc:
        capture.finish(exc)
        raise
    capture.finish()


def profile(*args, **kwargs):
    """Load the native profiler class when an application starts recording."""
    from .profiler import profile as native_recorder

    return native_recorder(*args, **kwargs)


def _record_profile(
    create_profiler,
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
    capture = sampler = None

    profiler = create_profiler(**profiler_kwargs)
    _watch_profiler_artifacts(profiler)

    def save_cycle():
        try:
            _complete_cycle(capture, profiler, profiler.on_trace_ready)
        except BaseException as exc:
            _failed_callback(profiler, capture, exc)
            raise

    # The native scheduler binds its methods during construction. Attach the
    # archive to the saved-cycle action so callers can replace or clear their
    # callback between steps just as they can on a native profiler.
    for actions in profiler.action_map.values():
        for index, action in enumerate(actions):
            if action == profiler._trace_ready:
                actions[index] = save_cycle
    native_start, native_stop = profiler.start, profiler.stop
    native_start_trace, native_stop_trace = profiler.start_trace, profiler.stop_trace
    direct_window = False

    def begin(operation):
        nonlocal capture, sampler, direct_window
        if sampler is not None:
            raise RuntimeError("ServeUtil profiler is already recording")
        direct_window = operation is native_start_trace
        capture = Capture(output, "torch")
        profiler.serveutil_directory = capture.path
        try:
            base, error = _baseline(torch, output, selected, calibration)
            _set_metadata(capture, torch, selected, base, error)
            sampler = TelemetrySampler(
                capture.path / "samples.jsonl.gz",
                devices=selected,
                interval_ms=interval_ms,
                include_hbm=_independent_hbm(torch, profiler_kwargs),
            ).start()
            return operation()
        except BaseException as exc:
            if sampler is not None:
                capture.manifest["telemetry"] = sampler.stop()
                sampler = None
            capture.finish(exc)
            raise

    def finish(operation):
        nonlocal sampler
        if sampler is None:
            return operation()
        active, sampler = sampler, None
        try:
            try:
                result = operation()
            finally:
                capture.manifest["telemetry"] = active.stop()
            if operation is native_stop_trace:
                # PyTorch's low-level pair has no scheduler or ready callback.
                _complete_cycle(capture, profiler, None)
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

    @wraps(native_start)
    def start():
        return begin(native_start)

    @wraps(native_stop)
    def stop():
        return finish(native_stop)

    @wraps(native_start_trace)
    def start_trace():
        if sampler is not None:
            return native_start_trace()
        return begin(native_start_trace)

    @wraps(native_stop_trace)
    def stop_trace():
        if not direct_window:
            return native_stop_trace()
        return finish(native_stop_trace)

    profiler.start, profiler.stop = start, stop
    profiler.start_trace, profiler.stop_trace = start_trace, stop_trace
    return profiler


class _CycleObserver:
    """Collect one hardware run for each active execution-trace observer interval."""

    def __init__(
        self, output, devices, interval_ms, base, error, observer, callback, include_hbm
    ):
        self.output = output
        self.devices = devices
        self.interval_ms = interval_ms
        self.base, self.error = base, error
        self.observer = observer
        self.callback = callback
        self.include_hbm = include_hbm
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
            _complete_cycle(capture, profiler, self.callback)
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
        _independent_hbm(torch, profiler_kwargs),
    )
    return {
        **profiler_kwargs,
        "execution_trace_observer": observer,
        "on_trace_ready": observer.on_trace_ready,
    }
