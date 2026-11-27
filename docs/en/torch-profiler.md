# Record with PyTorch Profiler

Use `serveutil.profile` in the same environment as your model. It returns a native `torch.profiler.profile` instance. Use the context manager or call `start()`, `step()` and `stop()` yourself.

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

```python
with serveutil.observe("runs") as run:
    handler = torch.profiler.tensorboard_trace_handler(str(run / "traces"), use_gzip=True)
    with torch.profiler.profile(on_trace_ready=handler, record_shapes=True) as prof:
        model(inputs)
    if hasattr(prof, "wait_for_exports"):
        prof.wait_for_exports()
```

Write native outputs to the capture's `traces/` directory and finish deferred writes inside the context. This leaves the application's profiler, callbacks and export methods under its control. Use `include_hbm=False` with a native device profiler that needs exclusive CUPTI access. Binary traces open on the same ServeUtil Viewer timeline.

Use `from serveutil import profiler` to keep calls such as `profiler.schedule(...)`, `profiler.record_function(...)`, and `profiler.profile(...)` together. These helpers come from the installed PyTorch version. The recording context returns the native profiler instance, including its `step()` method.

## Options and device availability

Native schedules, shape recording, memory, stacks, FLOP estimates, execution graphs, dynamic activity toggling and export APIs retain the installed PyTorch version's prerequisites. ServeUtil adds `devices`, `interval_ms`, `calibration` and `metrics`. CPU-only recording works. CUDA defaults to the current device. Use `calibration=False` to skip the first microbenchmark or pass saved calibration JSON.

Tested combinations include CPU PyTorch 2.10 and 2.15 and H20 CUDA builds of 2.13 and 2.15. On the tested 2.13 build, recording only CPU first can leave later GPU windows without kernels, including with native PyTorch alone. Record CUDA in a fresh process in that case. The report flags missing device events and individually damaged JSON cycles. On 2.15, native Cuspy synchronous, asynchronous and binary exports were exercised. Native directional PM sampling was tested with CUPTI 13.4.92. The tested 13.2.2 runtime was below Cuspy's requirement. [Metrics and reference workloads](metrics.md)

On the tested PyTorch 2.15 Cuspy backend, a capture with the default writer retains both native encodings of each completed window. You can request a `.pftrace` after `stop()` or select the binary in the viewer’s source list. Both encodings use the same GPU observations. Keeping both adds encoding time and disk space.
