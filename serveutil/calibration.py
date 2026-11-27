"""Measure sustainable rates with the same devices and counter scopes as a capture."""

import gzip
import hashlib
import importlib.metadata
import json
import os
import shutil
import statistics
import tempfile
import time
from bisect import bisect_right
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from .telemetry import TelemetrySampler, _identifier

VERSION = 3
COPY_BYTES = 192 * 1024 * 1024
MATRIX_SIZE = 4096


def _identity(torch, devices):
    props = []
    for dev in devices:
        p = torch.cuda.get_device_properties(dev)
        if not hasattr(p, "uuid"):
            raise RuntimeError(
                "Calibration requires CUDA device UUIDs from torch.cuda.get_device_properties"
            )
        props.append(
            {
                "uuid": _identifier(p.uuid),
                "name": p.name,
                "memory_bytes": p.total_memory,
            }
        )
    try:
        import pynvml as nv

        nv.nvmlInit()
    except (ImportError, OSError, RuntimeError):
        driver = None
    except Exception:
        driver = None
    else:
        try:
            driver = str(nv.nvmlSystemGetDriverVersion())
            for row in props:
                h = nv.nvmlDeviceGetHandleByUUID(row["uuid"])
                fields = {}
                for name, function in (
                    ("power_limit_mw", "nvmlDeviceGetPowerManagementLimit"),
                    ("pcie_max_gen", "nvmlDeviceGetMaxPcieLinkGeneration"),
                    ("pcie_max_width", "nvmlDeviceGetMaxPcieLinkWidth"),
                ):
                    try:
                        fields[name] = getattr(nv, function)(h)
                    except (AttributeError, nv.NVMLError):
                        pass
                row.update(fields)
        finally:
            nv.nvmlShutdown()
    try:
        pm_version = importlib.metadata.version("cupti-python")
    except importlib.metadata.PackageNotFoundError:
        pm_version = None
    profile_access = {}
    if Path("/proc/self/status").is_file():
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("CapEff:"):
                profile_access["sys_admin"] = bool(int(line.split()[1], 16) & (1 << 21))
    tools = {}
    for name in ("nvbandwidth", "all_reduce_perf"):
        binary = shutil.which(name)
        if binary:
            st = Path(binary).stat()
            tools[name] = [st.st_size, st.st_mtime_ns]
    return {
        "tools": tools,
        "pm_version": pm_version,
        "profile_access": profile_access,
        "version": VERSION,
        "devices": props,
        "driver": driver,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "peer_access": [
            [a != b and torch.cuda.can_device_access_peer(a, b) for b in devices]
            for a in devices
        ],
    }


def _measure(torch, op, dev):
    """Time batches with CUDA events and run enough batches for coarse hardware samples."""
    with torch.cuda.device(dev):
        for _ in range(3):
            op()
        torch.cuda.synchronize(dev)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        count = 1
        while True:
            start.record()
            for _ in range(count):
                op()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end)
            if ms >= 8 or count >= 256:
                break
            count = min(count * 4, 256)
        start_ns = time.time_ns()
        deadline = time.monotonic() + 0.85
        durations = []
        while len(durations) < 5 or time.monotonic() < deadline:
            start.record()
            for _ in range(count):
                op()
            end.record()
            end.synchronize()
            durations.append(start.elapsed_time(end) / count / 1000)
        return (
            statistics.median(durations),
            start_ns,
            time.time_ns(),
            len(durations),
            count,
        )


def _standard_cases(torch, dev):
    with torch.cuda.device(dev):
        cuda = torch.device("cuda", dev)
        generator = torch.Generator(device=cuda).manual_seed(0)
        a = torch.empty(COPY_BYTES, dtype=torch.uint8, device=cuda)
        b = torch.empty_like(a)
        host = torch.empty(COPY_BYTES, dtype=torch.uint8, pin_memory=True)
        # Materialize host pages and populate GPU memory before timing.
        a.fill_(11)
        host.fill_(7)
        for metric, func, size, tracks in [
            (
                "copy.device",
                partial(b.copy_, a, non_blocking=True),
                COPY_BYTES * 2,
                ["hbm.util", "hbm.total_estimated", "hbm.read", "hbm.write"],
            ),
            (
                "copy.h2d",
                partial(b.copy_, host, non_blocking=True),
                COPY_BYTES,
                ["pcie.rx"],
            ),
            (
                "copy.d2h",
                partial(host.copy_, a, non_blocking=True),
                COPY_BYTES,
                ["pcie.tx"],
            ),
        ]:
            yield (
                metric,
                func,
                size,
                "GB/s",
                {"bytes": COPY_BYTES, "pinned_host": metric != "copy.device"},
                tracks,
            )
        del func, a, b, host
        data = torch.randn(
            COPY_BYTES // 4, device=cuda, dtype=torch.float32, generator=generator
        )
        scalar = torch.empty((), dtype=torch.float32, device=cuda)
        yield (
            "memory.reduce_read",
            partial(torch.sum, data, dim=(0,), out=scalar),
            COPY_BYTES,
            "GB/s",
            {"bytes": COPY_BYTES, "operation": "FP32 sum"},
            ["hbm.read"],
        )
        yield (
            "memory.arange_write",
            partial(torch.arange, data.numel(), out=data),
            COPY_BYTES,
            "GB/s",
            {"bytes": COPY_BYTES, "operation": "FP32 arange"},
            ["hbm.write"],
        )
        del data, scalar
        for name in ("float16", "bfloat16", "float32", "float64"):
            dtype = getattr(torch, name)
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            n = MATRIX_SIZE
            # Distinct inputs keep compression/zero optimizations out of the measurement.
            lhs = torch.randn((n, n), device=cuda, dtype=dtype, generator=generator)
            rhs = torch.randn((n, n), device=cuda, dtype=dtype, generator=generator)
            dst = torch.empty_like(lhs)
            pipe = {
                "float16": "tensor.hmma",
                "bfloat16": "tensor.hmma",
                "float32": "fp32.active",
                "float64": "fp64.active",
            }[name]
            yield (
                f"gemm.{name}",
                partial(torch.mm, lhs, rhs, out=dst),
                2 * n**3,
                "TFLOP/s",
                {
                    "m": n,
                    "n": n,
                    "k": n,
                    "dtype": name,
                    "fp32_precision": torch.get_float32_matmul_precision(),
                },
                ["sm.active", "tensor.active", pipe],
            )
            del lhs, rhs, dst
        n = MATRIX_SIZE
        if torch.cuda.get_device_capability(dev)[0] >= 9 and hasattr(
            torch, "_scaled_mm"
        ):
            dtype = torch.float8_e4m3fn
            lhs = torch.randn(
                (n, n), device=cuda, dtype=torch.float16, generator=generator
            ).to(dtype)
            rhs = (
                torch.randn(
                    (n, n), device=cuda, dtype=torch.float16, generator=generator
                )
                .to(dtype)
                .T
            )
            scale = torch.ones((), dtype=torch.float32, device=cuda)
            yield (
                "gemm.float8_e4m3fn",
                partial(
                    torch._scaled_mm, lhs, rhs, scale, scale, out_dtype=torch.bfloat16
                ),
                2 * n**3,
                "TFLOP/s",
                {
                    "m": n,
                    "n": n,
                    "k": n,
                    "dtype": "float8_e4m3fn",
                    "out_dtype": "bfloat16",
                },
                ["sm.active", "tensor.active"],
            )
            del lhs, rhs, scale
        if hasattr(torch, "_int_mm"):
            lhs = torch.randint(
                -4, 5, (n, n), device=cuda, dtype=torch.int8, generator=generator
            )
            rhs = torch.randint(
                -4, 5, (n, n), device=cuda, dtype=torch.int8, generator=generator
            ).T
            yield (
                "gemm.int8",
                partial(torch._int_mm, lhs, rhs),
                2 * n**3,
                "TOP/s",
                {"m": n, "n": n, "k": n, "dtype": "int8", "out_dtype": "int32"},
                ["sm.active", "tensor.active", "tensor.imma"],
            )


def _validate_devices(torch, requested):
    # Use CUDA logical indices for operations, resolve UUIDs before touching NVML.
    devices = (
        list(range(torch.cuda.device_count())) if requested is None else list(requested)
    )
    resolved = []
    for dev in devices:
        if str(dev).isdecimal():
            index = int(dev)
        else:
            uuid = _identifier(dev)
            ids = [
                _identifier(torch.cuda.get_device_properties(i).uuid)
                for i in range(torch.cuda.device_count())
            ]
            if uuid not in ids:
                raise ValueError(f"Device {dev} is not visible to CUDA")
            index = ids.index(uuid)
        if index < 0 or index >= torch.cuda.device_count() or index in resolved:
            raise ValueError(f"Invalid or duplicate CUDA device {dev}")
        resolved.append(index)
    if not resolved:
        raise RuntimeError("Calibration needs at least one CUDA device")
    return resolved


def calibrate(devices=None, *, cache_dir=None, refresh=False):
    """Benchmark on selected CUDA ordinals/UUIDs. Optional cache is keyed by hardware/config.

    Each benchmark records logical payload or operation throughput and matched
    on-device counters. Calibration should run during an idle period on these devices.
    """
    import torch

    devices = _validate_devices(torch, devices)
    identity = _identity(torch, devices)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if cache_dir is None:
        return _benchmark(torch, devices, identity, None)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"calibration-{key}.json"
    with _lock(path.with_suffix(".lock")):
        if path.exists() and not refresh:
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                cached = {}
            if cached.get("identity") == identity and cached.get("benchmarks"):
                return cached
        result = _benchmark(torch, devices, identity, cache_dir)
        if result["benchmarks"]:
            from .bundle import write_json

            write_json(path, result)
        return result


@contextmanager
def _lock(path):
    # On the shared Linux GPU host, simultaneous callers should reuse the
    # result of one hardware test. The kernel releases this lock on process exit.
    with path.open("a+b") as stream:
        if os.name == "posix":
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def _benchmark(torch, devices, identity, cache_dir):
    result = {
        "schema": VERSION,
        "identity": identity,
        "devices": identity["devices"],
        "benchmarks": [],
        "unavailable": [],
        "created_ns": time.time_ns(),
    }
    uuids = {idx: row["uuid"] for idx, row in zip(devices, identity["devices"])}
    windows = []

    def run(op, task, index, size, unit, parameters, tracks):
        try:
            duration, start, end, batches, count = _measure(torch, op, index)
        except (RuntimeError, MemoryError) as exc:
            result["unavailable"].append(
                {"device": uuids[index], "metric": task, "reason": str(exc)}
            )
            return
        value = size / duration / (1e9 if unit == "GB/s" else 1e12)
        result["benchmarks"].append(
            {
                "metric": task,
                "device": uuids[index],
                "unit": unit,
                "value": value,
                "source": "CUDA event microbenchmark",
                "parameters": parameters,
                "batches": batches,
                "iterations_per_batch": count,
            }
        )
        windows.append((task, parameters, tracks, start, end))

    with tempfile.TemporaryDirectory(dir=cache_dir) as temp, torch.no_grad():
        sample_path = Path(temp) / "calibration.jsonl.gz"
        sampler = TelemetrySampler(
            sample_path, list(uuids.values()), include_network=False
        ).start()
        try:
            for dev in devices:
                for task, op, size, unit, params, metrics in _standard_cases(
                    torch, dev
                ):
                    tracks = [(uuids[dev], metric) for metric in metrics]
                    run(op, task, dev, size, unit, params, tracks)
                    del op
            for i, src in enumerate(devices):
                for j, dst in enumerate(devices):
                    if not identity["peer_access"][i][j]:
                        continue
                    with torch.cuda.device(src):
                        a = torch.full(
                            (COPY_BYTES,), 19, dtype=torch.uint8, device=f"cuda:{src}"
                        )
                        b = torch.empty_like(a, device=f"cuda:{dst}")
                        torch.cuda.synchronize(dst)
                        params = {
                            "bytes": COPY_BYTES,
                            "src": uuids[src],
                            "dst": uuids[dst],
                        }
                        tracks = [(uuids[src], "nvlink.tx"), (uuids[dst], "nvlink.rx")]
                        run(
                            partial(b.copy_, a, non_blocking=True),
                            "copy.peer",
                            src,
                            COPY_BYTES,
                            "GB/s",
                            params,
                            tracks,
                        )
                        del a, b
        finally:
            result["sampler"] = sampler.stop()
        result["benchmarks"].extend(_observed_references(sample_path, windows))
        from .reference import run_installed

        upstream, missing = run_installed(list(uuids.values()), Path(temp).resolve())
        result["benchmarks"].extend(upstream)
        result["unavailable"].extend(missing)
    return result


def _observed_references(path, windows):
    # PM samples can be numerous and appear in device order after finalization.
    # Index disjoint benchmark windows, then stream the file exactly once.
    intervals = defaultdict(list)
    for task, params, tracks, start, end in windows:
        for device, metric in tracks:
            intervals[device, metric].append((start, end, task, params))
    starts = {key: [r[0] for r in rows] for key, rows in intervals.items()}
    observed = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            key = sample["device"], sample["metric"]
            if key not in starts:
                continue
            index = bisect_right(starts[key], sample["start_ns"]) - 1
            if index >= 0 and sample["end_ns"] <= intervals[key][index][1]:
                observed[key, index, sample["unit"], sample["source"]].append(
                    sample["value"]
                )
    for (key, index, unit, source), values in observed.items():
        value = statistics.median(values)
        if value > 0:
            _, _, task, params = intervals[key][index]
            yield {
                "metric": key[1],
                "device": key[0],
                "value": value,
                "unit": unit,
                "source": source,
                "workload": task,
                "parameters": params,
                "samples": len(values),
            }


def reference_for(sample, baseline):
    """Highest sustained median for the same physical device, metric, unit, and source."""
    possible = [
        r
        for r in baseline.get("benchmarks", ())
        if r.get("value", 0) > 0
        and all(
            sample.get(k) == r.get(k) for k in ("device", "metric", "unit", "source")
        )
        and (
            r.get("scope") not in ("port", "interface")
            or sample.get("node") == r.get("node")
        )
        and (
            r.get("scope") not in ("collective", "copy test")
            or sample.get("parameters") == r.get("parameters")
        )
    ]
    return max(possible, key=lambda r: r["value"], default=None)
