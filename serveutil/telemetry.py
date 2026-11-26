"""Interval samples from NVIDIA NVML and Linux RDMA counters.

Sampling is node-local. Timestamps are Unix nanoseconds, derived from a paired
wall/monotonic clock anchor. Device totals include traffic from other processes.
"""

import gzip
import json
import math
import os
import socket
import threading
import time
from contextlib import ExitStack
from pathlib import Path

HBM_PM_METRICS = ("dram__bytes_read.sum.per_second", "dram__bytes_write.sum.per_second")
# NVML specifies MiB/s for GPM throughput fields.
MIB_PER_SEC_TO_GB_PER_SEC = (1024**2) / 1e9
GPM_METRICS = {
    "SM_UTIL": ("sm.active", "SM active", "%"),
    "SM_OCCUPANCY": ("sm.occupancy", "SM occupancy", "%"),
    "ANY_TENSOR_UTIL": ("tensor.active", "Tensor pipe active", "%"),
    "HMMA_TENSOR_UTIL": ("tensor.hmma", "HMMA active", "%"),
    "DMMA_TENSOR_UTIL": ("tensor.dmma", "DMMA active", "%"),
    "IMMA_TENSOR_UTIL": ("tensor.imma", "IMMA active", "%"),
    "FP64_UTIL": ("fp64.active", "FP64 pipe active", "%"),
    "FP32_UTIL": ("fp32.active", "FP32 pipe active", "%"),
    "FP16_UTIL": ("fp16.active", "FP16 pipe active", "%"),
    "DRAM_BW_UTIL": ("hbm.util", "HBM bandwidth utilization (NVML)", "%"),
    "PCIE_TX_PER_SEC": ("pcie.tx", "PCIe TX", "GB/s"),
    "PCIE_RX_PER_SEC": ("pcie.rx", "PCIe RX", "GB/s"),
    "NVLINK_TOTAL_TX_PER_SEC": ("nvlink.tx", "NVLink TX (all links)", "GB/s"),
    "NVLINK_TOTAL_RX_PER_SEC": ("nvlink.rx", "NVLink RX (all links)", "GB/s"),
}


def kineto_activities(torch, metrics=HBM_PM_METRICS, interval_ms=1):
    """Return the public torch.profiler CUDA PM configuration when available."""
    prof = torch.profiler
    if not hasattr(prof, "PerformanceMetricsConfig"):
        raise RuntimeError(
            "This torch build lacks PerformanceMetricsConfig for CUDA PM sampling"
        )
    config = prof.PerformanceMetricsConfig(
        metric_names=list(metrics), sampling_interval_ms=interval_ms
    )
    return {
        prof.ProfilerActivity.CPU: prof.ProfilerActivityConfig(),
        prof.ProfilerActivity.CUDA: prof.ProfilerActivityConfig(
            profiler_configs=[config]
        ),
    }


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def gpu_selection(devices=None):
    """Prefer CUDA UUIDs, accounting for CUDA device ordering and visibility."""
    if devices is not None:
        return list(devices)
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        return []
    try:
        import torch

        count = torch.cuda.device_count()
        if count:
            return [str(torch.cuda.get_device_properties(i).uuid) for i in range(count)]
    except (ImportError, AttributeError, RuntimeError):
        pass
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None  # A lightweight host monitor may run without a CUDA-enabled torch build.
    return [i.strip() for i in visible.split(",") if i.strip()]


def _identifier(value):
    value = str(value)
    if value.isdecimal():
        return int(value)
    if not value.startswith(("GPU-", "MIG-")):
        return "GPU-" + value
    return value


def rdma_counters(root=Path("/sys/class/infiniband")):
    """Yield one byte source per port/direction. Standard IB data counts are 4-byte words."""
    for port in sorted(Path(root).glob("*/ports/*")):
        for direction, short in (("tx", "xmit"), ("rx", "rcv")):
            candidates = [
                (
                    port / "hw_counters" / f"hw_{direction}_bytes_cnt",
                    1,
                    "eRDMA hw_counters",
                ),
                (port / "counters" / f"port_{short}_data", 4, "InfiniBand port data"),
            ]
            for path, scale, source in candidates:
                if path.is_file():
                    yield (
                        f"{port.parents[1].name}/{port.name}",
                        direction,
                        path,
                        scale,
                        source,
                    )
                    break


def netdev_counters(root, interfaces=None):
    """Linux interface bytes, including bond masters. Explicit names also support virtual NICs."""
    root = Path(root)
    for iface in sorted(root.glob("*")):
        if interfaces is not None:
            if iface.name not in interfaces:
                continue
        elif (
            iface.name == "lo"
            or (iface / "master").exists()
            or not ((iface / "device").exists() or (iface / "bonding").exists())
        ):
            continue
        for direction in ("rx", "tx"):
            path = iface / "statistics" / (direction + "_bytes")
            if path.is_file():
                yield iface.name, direction, path, 1, "Linux netdev statistics"


class _Network:
    def __init__(self, sink):
        self.sink = sink
        rdma_root = Path(os.environ.get("SERVEUTIL_RDMA_ROOT", "/sys/class/infiniband"))
        net_root = os.environ.get("SERVEUTIL_NET_ROOT", rdma_root.parent / "net")
        names = os.environ.get("SERVEUTIL_NET_INTERFACES")
        requested = (
            {name.strip() for name in names.split(",")} if names is not None else None
        )
        rdma = list(rdma_counters(rdma_root))
        netdev = list(netdev_counters(net_root, requested))
        self.counters = [("rdma", *row) for row in rdma] + [
            ("netdev", *row) for row in netdev
        ]
        self.previous = {}
        if not rdma:
            sink.unavailable("node", "rdma", "No RDMA port counter files found")
        if requested:
            for missing in sorted(requested - {row[0] for row in netdev}):
                sink.unavailable(
                    missing, "netdev", "Requested network interface unavailable"
                )

    def sample(self):
        for kind, device, direction, path, scale, source in self.counters:
            key = kind, device, direction
            try:
                value = int(path.read_text())
            except (OSError, ValueError) as exc:
                self.sink.unavailable(device, f"{kind}.{direction}", exc)
                self.previous.pop(key, None)
                continue
            now = self.sink.now()
            previous = self.previous.get(key)
            # Preserve the byte delta when two polls fall on the same clock tick.
            if previous and now == previous[0] and value >= previous[1]:
                continue
            self.previous[key] = now, value
            if previous is None:
                continue
            start, before = previous
            if now <= start or value < before:
                self.sink.unavailable(
                    device, f"{kind}.{direction}", "Port counter discontinuity"
                )
                continue
            # B/ns and decimal GB/s have the same numeric value.
            rate = (value - before) * scale / (now - start)
            self.sink.emit(
                f"{kind}.{direction}",
                f"{device} {direction.upper()}",
                "GB/s",
                device,
                "port" if kind == "rdma" else "interface",
                source,
                start,
                now,
                rate,
            )


class _GPU:
    def __init__(self, nv, handle, sink):
        self.nv, self.handle, self.sink = nv, handle, sink
        self.uuid = _text(nv.nvmlDeviceGetUUID(handle))
        self.previous = None
        self.buffers = []
        self.failed = set()
        self.link_previous = {}
        self.dram_peak = None
        self.fields = {}
        self.device = {
            "uuid": self.uuid,
            "name": _text(nv.nvmlDeviceGetName(handle)),
            "index": nv.nvmlDeviceGetIndex(handle),
        }
        sink.metadata["devices"].append(self.device)
        try:
            # GPM defines DRAM_BW_UTIL against theoretical DRAM bandwidth.
            # The DDR clock formula is an estimate and cannot separate reads and writes.
            width = nv.nvmlDeviceGetMemoryBusWidth(handle)
            mhz = nv.nvmlDeviceGetMaxClockInfo(handle, nv.NVML_CLOCK_MEM)
            if width > 0 and mhz > 0:
                self.dram_peak = 2 * mhz * width / 8000
                self.device.update(memory_bus_bits=width, max_memory_clock_mhz=mhz)
        except (nv.NVMLError, AttributeError):
            pass
        try:
            if not nv.nvmlGpmQueryDeviceSupport(handle).isSupportedDevice:
                raise RuntimeError("GPU does not support NVML GPM")
            for _ in range(2):
                self.buffers.append(nv.nvmlGpmSampleAlloc())
            for field, spec in GPM_METRICS.items():
                field_id = getattr(nv, "NVML_GPM_METRIC_" + field, None)
                if field_id is not None:
                    self.fields[field_id] = spec
            if not self.fields:
                raise RuntimeError("NVML bindings expose no GPM fields")
        except (nv.NVMLError, AttributeError, RuntimeError) as exc:
            sink.unavailable(self.uuid, "nvml.gpm", exc)
            self.close()

    def close(self):
        while self.buffers:
            self.nv.nvmlGpmSampleFree(self.buffers.pop())

    def sample(self):
        n = self.nv
        if self.buffers:
            try:
                self._sample_gpm()
                return
            except (n.NVMLError, AttributeError) as exc:
                self.sink.unavailable(self.uuid, "nvml.gpm", exc)
                self.close()
        self._sample_basic()

    def _sample_gpm(self):
        n = self.nv
        now = self.sink.now()
        if self.previous is not None and now - self.previous < 101_000_000:
            return  # GPM's refresh period is >100 ms.
        left, right = self.buffers
        n.nvmlGpmSampleGet(self.handle, right)
        start = self.previous
        self.previous = now
        self.buffers.reverse()
        if start is None:
            return
        metrics = n.c_nvmlGpmMetricsGet_t()
        metrics.version = n.NVML_GPM_METRICS_GET_VERSION
        metrics.sample1, metrics.sample2 = left, right
        metrics.numMetrics = len(self.fields)
        for field, output in zip(self.fields, metrics.metrics):
            output.metricId = field
        n.nvmlGpmMetricsGet(metrics)
        for item in metrics.metrics[: metrics.numMetrics]:
            metric, label, unit = self.fields[item.metricId]
            if item.nvmlReturn != n.NVML_SUCCESS or not math.isfinite(item.value):
                reason = (
                    n.NVMLError(item.nvmlReturn)
                    if item.nvmlReturn
                    else "Invalid sample"
                )
                self.sink.unavailable(self.uuid, metric, reason)
                continue
            value = item.value * (MIB_PER_SEC_TO_GB_PER_SEC if unit == "GB/s" else 1)
            self.sink.emit(
                metric, label, unit, self.uuid, "gpu", "NVML GPM", start, now, value
            )
            if metric == "hbm.util" and self.dram_peak is not None:
                self.sink.emit(
                    "hbm.total_estimated",
                    "HBM total (estimated)",
                    "GB/s",
                    self.uuid,
                    "gpu",
                    "NVML GPM × theoretical DDR bandwidth",
                    start,
                    now,
                    self.dram_peak * value / 100,
                    estimated=True,
                )

    def _sample_basic(self):
        n = self.nv
        if "gpu.busy" not in self.failed:
            try:
                util = n.nvmlDeviceGetUtilizationRates(self.handle)
                now = self.sink.now()
                for name, value in (
                    ("gpu.busy", util.gpu),
                    ("memory.busy", util.memory),
                ):
                    self.sink.emit(
                        name,
                        name.replace(".", " ").capitalize(),
                        "%",
                        self.uuid,
                        "gpu",
                        "NVML utilization rates",
                        now,
                        now,
                        value,
                        window="device-defined",
                    )
            except n.NVMLError as exc:
                self.sink.unavailable(self.uuid, "gpu.busy", exc)
                self.failed.add("gpu.busy")
        if "nvlink" not in self.failed:
            try:
                fields = [
                    (n.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX, 0xFFFFFFFF),
                    (n.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX, 0xFFFFFFFF),
                ]
                for direction, item in zip(
                    ("tx", "rx"), n.nvmlDeviceGetFieldValues(self.handle, fields)
                ):
                    metric = "nvlink." + direction
                    if (
                        item.nvmlReturn
                        or item.valueType != n.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG
                    ):
                        self.sink.unavailable(
                            self.uuid, metric, "NVML payload field unavailable"
                        )
                        continue
                    now = item.timestamp * 1000
                    value = item.value.ullVal * 1024
                    before = self.link_previous.get(metric)
                    self.link_previous[metric] = now, value
                    if before and now > before[0] and value >= before[1]:
                        self.sink.emit(
                            metric,
                            "NVLink " + direction.upper() + " (payload)",
                            "GB/s",
                            self.uuid,
                            "gpu",
                            "NVML NVLink data field",
                            before[0],
                            now,
                            (value - before[1]) / (now - before[0]),
                        )
            except (n.NVMLError, AttributeError) as exc:
                self.sink.unavailable(self.uuid, "nvlink", exc)
                self.failed.add("nvlink")
        for suffix, counter in (
            ("tx", n.NVML_PCIE_UTIL_TX_BYTES),
            ("rx", n.NVML_PCIE_UTIL_RX_BYTES),
        ):
            metric = "pcie." + suffix
            if metric in self.failed:
                continue
            try:
                rate = n.nvmlDeviceGetPcieThroughput(self.handle, counter)
            except n.NVMLError as exc:
                self.sink.unavailable(self.uuid, metric, exc)
                self.failed.add(metric)
                continue
            # NVML's PCIe throughput query covers a hardware-defined 20 ms window.
            now = self.sink.now()
            self.sink.emit(
                metric,
                f"PCIe {suffix.upper()}",
                "GB/s",
                self.uuid,
                "gpu",
                "NVML PCIe throughput (KiB/s)",
                now - 20_000_000,
                now,
                rate * 1024 / 1e9,
            )


class TelemetrySampler:
    """Stream samples to an exclusive JSONL/JSONL.GZ output; inspect metadata after stop()."""

    def __init__(
        self,
        path,
        devices=None,
        interval_ms=150,
        *,
        include_network=True,
        include_hbm=True,
    ):
        if not math.isfinite(interval_ms) or interval_ms <= 0:
            raise ValueError("interval_ms must be positive and finite")
        self.path = Path(path)
        self.selection = devices
        self.interval_ms = interval_ms
        self.include_network = include_network
        self.include_hbm = include_hbm
        self.metadata = {
            "schema": 1,
            "node": os.environ.get("SERVEUTIL_NODE_NAME") or socket.gethostname(),
            "devices": [],
            "unavailable": [],
            "samples": 0,
            "interval_ms": interval_ms,
            "clock": "unix_ns",
            "metric_counts": {},
        }
        self._issues = set()
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._sources = []
        self._nv = None
        self._stream = None

    def unavailable(self, device, metric, reason):
        key = device, metric, str(reason)
        if key not in self._issues:
            self._issues.add(key)
            self.metadata["unavailable"].append(
                dict(zip(("device", "metric", "reason"), key))
            )

    def now(self):
        return self._wall + (time.perf_counter_ns() - self._mono)

    def emit(
        self, metric, label, unit, device, scope, source, start, end, value, **details
    ):
        row = dict(
            metric=metric,
            label=label,
            unit=unit,
            device=device,
            node=self.metadata["node"],
            scope=scope,
            source=source,
            start_ns=start,
            end_ns=end,
            value=value,
            **details,
        )
        self._stream.write(
            json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
        )
        self.metadata["samples"] += 1
        counts = self.metadata["metric_counts"]
        counts[metric] = counts.get(metric, 0) + 1

    def start(self):
        if self._thread is not None:
            raise RuntimeError("A TelemetrySampler instance can be started once")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if self.path.suffix == ".gz" else open
        self._stream = opener(self.path, "xt", encoding="utf-8")
        self._mono = time.perf_counter_ns()
        self._wall = time.time_ns()
        self.metadata["start_ns"] = self._wall
        try:
            self._initialize_sources()
            self._sample()  # Seed delta counters before returning control to the workload.
            self._thread = threading.Thread(
                target=self._run, name="serveutil-telemetry", daemon=True
            )
            self._thread.start()
        except BaseException:
            self._close()
            raise
        return self

    def _initialize_sources(self):
        devices = gpu_selection(self.selection)
        if devices != []:
            try:
                import pynvml as nv

                nv.nvmlInit()
                self._nv = nv
            except Exception as exc:
                self.unavailable("node", "nvml", exc)
            if self._nv is not None:
                self.metadata["driver"] = _text(nv.nvmlSystemGetDriverVersion())
                visible = (
                    devices if devices is not None else range(nv.nvmlDeviceGetCount())
                )
                seen = set()
                for dev in visible:
                    try:
                        key = _identifier(dev)
                        handle = (
                            nv.nvmlDeviceGetHandleByIndex(key)
                            if isinstance(key, int)
                            else nv.nvmlDeviceGetHandleByUUID(key)
                        )
                        uuid = _text(nv.nvmlDeviceGetUUID(handle))
                        if uuid not in seen:
                            self._sources.append(_GPU(nv, handle, self))
                            seen.add(uuid)
                    except nv.NVMLError as exc:
                        self.unavailable(str(dev), "gpu", exc)
        if self.include_hbm:
            from .pm import HbmSampler

            for gpu in self.metadata["devices"]:
                if not HbmSampler.available():
                    self.unavailable(
                        gpu["uuid"],
                        "hbm.read/write",
                        "Optional cupti-python PM collector unavailable",
                    )
                    continue
                try:
                    self._sources.append(
                        HbmSampler(gpu["uuid"], self, min(self.interval_ms, 5))
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    self.unavailable(gpu["uuid"], "hbm.read/write", exc)
        if self.include_network:
            self._sources.append(_Network(self))

    def _sample(self):
        for source in self._sources:
            source.sample()

    def _run(self):
        try:
            while not self._stop.wait(self.interval_ms / 1000):
                self._sample()
        except Exception as exc:
            self._error = exc

    def _close(self):
        with ExitStack() as cleanup:
            cleanup.callback(self._stream.close)
            if self._nv is not None:
                cleanup.callback(self._nv.nvmlShutdown)
            for source in self._sources:
                if hasattr(source, "close"):
                    cleanup.callback(source.close)

    def stop(self):
        if self._thread is None:
            raise RuntimeError("Sampler has not started")
        if not self._stream.closed:
            self._stop.set()
            self._thread.join()
            try:
                self._sample()
            finally:
                self.metadata["end_ns"] = self.now()
                self._close()
        if self._error:
            raise RuntimeError("Telemetry sampler failed") from self._error
        return self.metadata
