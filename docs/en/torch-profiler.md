# Record with PyTorch Profiler

Use `serveutil.profile` in the same environment as your model. It returns a native `torch.profiler.profile` instance. Its Python signature includes the installed PyTorch options. You can replace or clear `prof.on_trace_ready` between saved steps. Use the context manager or call `start()`, `step()` and `stop()` yourself.

```python
import torch
import serveutil

with serveutil.profile("runs", record_shapes=True, with_flops=True) as prof:
    x = torch.ones(128, 128, device="cuda" if torch.cuda.is_available() else "cpu")
    with torch.profiler.record_function("Projection"):
        y = x @ x
    prof.step()

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))
prof.export_chrome_trace("last-cycle.json.gz")
print(prof.serveutil_directory)
```

The same class is available as `serveutil.profiler.profile`. Existing profiler subclasses can inherit from it and use `super().__init__(...)`. Code that manages individual windows can call `prepare_trace()`, `start_trace()` and `stop_trace()`. Each standalone window produces a capture and keeps the original trace available for export. These three low-level calls do not invoke the ready callback.

## Keep an existing recording schedule

```python
handler = torch.profiler.tensorboard_trace_handler("tensorboard", use_gzip=True)
options = serveutil.instrument(
    "runs", on_trace_ready=handler, record_shapes=True,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=2),
)
with torch.profiler.profile(**options) as prof:
    for batch in batches:
        model(**batch)
        prof.step()
```

Pass an existing registered `execution_trace_observer` in the same options. PyTorch drives its lifecycle. Each completed activity window is archived independently. Use `serveutil.trace_handler` for export-only integrations. With `profile` the hardware sampler spans start to stop. With `instrument` it follows the active observer windows.

## Preserve caller-managed exports

If `on_trace_ready` raises, the exception propagates to the application. Open the capture folder to inspect completed windows and the recorded error. A caller that took ownership of an asynchronous Cuspy export can finish it through the same native observer after the exception.

Call `prof.export_stacks(...)` or `prof.export_memory_timeline(...)` in the ready callback or after recording. PyTorch writes the requested file. ServeUtil saves a byte-for-byte copy under `artifacts/` and lists its path, export method and zero-based recording cycle in `capture.json` under `profiler_artifacts`. An `ExecutionTraceObserver` writes its graph to the path supplied to `register_callback` when PyTorch finalizes the observer. One graph can cover multiple recording cycles.

```python
with serveutil.observe("runs") as run:
    handler = torch.profiler.tensorboard_trace_handler(str(run / "traces"), use_gzip=True)
    with torch.profiler.profile(on_trace_ready=handler, record_shapes=True) as prof:
        model(inputs)
    if hasattr(prof, "wait_for_exports"):
        prof.wait_for_exports()
```

Write timelines to `run / "traces"`. Write other profiler outputs to `run / "artifacts"`, including an execution graph from `observer.register_callback(str(run / "artifacts" / "execution.json.gz"))`. ServeUtil Viewer opens the timelines together and offers the other files under **Download profiler output**. Subdirectories and original bytes are preserved. Finish deferred writes before leaving the context. Use `include_hbm=False` with a native device profiler that needs exclusive CUPTI access.

Use `from serveutil import profiler` to keep calls such as `profiler.schedule(...)`, `profiler.record_function(...)`, and `profiler.profile(...)` together. These helpers come from the installed PyTorch version. The recording context returns the native profiler instance, including its `step()` method.

## Options and device availability

Native schedules, shape recording, memory, stacks, FLOP estimates, execution graphs, dynamic activity toggling and export APIs retain the installed PyTorch version's prerequisites. ServeUtil adds `devices`, `interval_ms`, `calibration` and `metrics`. CPU-only recording works. CUDA defaults to the current device. Use `calibration=False` to skip the first microbenchmark or pass saved calibration JSON.

Tested combinations include CPU PyTorch 2.10 and 2.15 and H20 CUDA builds of 2.13 and 2.15. On the tested 2.13 build, recording only CPU first can leave later GPU windows without kernels, including with native PyTorch alone. Record CUDA in a fresh process in that case. The report flags missing device events and individually damaged JSON cycles. On 2.15, native Cuspy synchronous, asynchronous and binary exports were exercised. Native directional PM sampling was tested with CUPTI 13.4.92. The tested 13.2.2 runtime was below Cuspy's requirement. [Metrics and reference workloads](metrics.md)

On the tested PyTorch 2.15 Cuspy backend, the default writer and callbacks that complete a native export during the callback retain both encodings of the same window. The capture keeps a copy even if your callback moves or deletes its output. You can request a `.pftrace` after `stop()` or select the binary in the viewer’s source list. Both encodings use the same GPU observations. Keeping both adds encoding time and disk space.

If a callback transfers an asynchronous writer with `take_pending_cuspy_export()`, finish it with its native `join()`. The capture shows `exporting` while writes are pending and creates the report once every writer finishes. Writers may finish in any order.

When you call `export_chrome_trace(..., use_python_export=True)` on a supported Kineto build, ServeUtil also saves the file produced by that writer. You can select it in ServeUtil Viewer to inspect its annotations or export-time metadata. The default trace and alternate exports belong to the same recording window.
