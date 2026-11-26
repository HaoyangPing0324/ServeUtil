"""Isolate CUPTI PM collection from Kineto's CUDA activity subscriber.

The helper process sees exactly one GPU, specified by UUID in CUDA_VISIBLE_DEVICES.
Native CUPTI timestamps are translated to the same Unix clock used by the sampler.
"""

import gzip
import importlib.metadata
import importlib.util
import json
import math
import os
import select
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

METRICS = ("dram__bytes_read.sum.per_second", "dram__bytes_write.sum.per_second")


def _clock(cupti):
    # Choose the shortest of several paired queries to bound scheduling jitter.
    anchors = []
    for _ in range(5):
        start = time.time_ns()
        counter = cupti.get_timestamp()
        end = time.time_ns()
        anchors.append((end - start, (start + end) // 2 - counter))
    return min(anchors)


def collect(filepath, interval_ms):
    # Initialize CUDA through the same runtime that Kineto uses. Older drivers
    # require runtime initialization before CUPTI allocates its profiler objects.
    import torch

    torch.cuda.init()
    from cupti import cupti, pm_sampling

    collector = pm_sampling.Collector(device_index=0)
    collector.enable()
    with ExitStack() as cleanup:
        cleanup.callback(collector.disable)
        collector.configure(
            metrics=list(METRICS),
            hardware_buffer_size=16 * 1024**2,
            sampling_interval=max(1, round(interval_ms * 1e6)),
            trigger_mode=pm_sampling.TriggerMode.GPU_TIME_INTERVAL,
        )
        error_ns, offset = _clock(cupti)
        start_ns = time.time_ns()
        with gzip.open(filepath, "wt", encoding="utf-8") as out:
            collector.start()
            running = True
            try:
                print(
                    json.dumps({"ready": True, "clock_error_ns": error_ns}), flush=True
                )
                last_end = 0
                while running:
                    readable, _, _ = select.select([sys.stdin], [], [], 0.20)
                    if readable:
                        sys.stdin.readline()  # EOF also releases GPU resources on parent exit.
                        running = False
                        collector.stop()
                    data = collector.decode(max_samples=10_000)
                    for s in data:
                        if s.start_timestamp <= 0 or s.end_timestamp <= max(
                            s.start_timestamp, last_end
                        ):
                            continue
                        start = s.start_timestamp + offset
                        end = s.end_timestamp + offset
                        if (
                            start < start_ns - error_ns
                            or end > time.time_ns() + error_ns
                        ):
                            continue
                        last_end = s.end_timestamp
                        values = [float(v) / 1e9 for v in s.metric_values]
                        if all(math.isfinite(v) and v >= 0 for v in values):
                            out.write(json.dumps([start, end, *values]) + "\n")
            finally:
                if running:
                    collector.stop()
        final_error, final_offset = _clock(cupti)
        print(
            json.dumps(
                {
                    "complete": True,
                    "clock_drift_ns": final_offset - offset,
                    "clock_error_ns": max(error_ns, final_error),
                }
            ),
            flush=True,
        )


def _library_dir():
    """Locate the matching vendor wheel's dynamic dependencies for a fresh subprocess."""
    try:
        dist = importlib.metadata.distribution("nvidia-cuda-cupti")
    except importlib.metadata.PackageNotFoundError:
        return None
    for file in dist.files or ():
        if file.name == "libnvperf_host.so":
            return str(dist.locate_file(file).parent)
    return None


class HbmSampler:
    """A PM subprocess writes bounded chunks to a temporary stream consumed by stop()."""

    def __init__(self, uuid, sink, interval_ms):
        self.uuid, self.sink = uuid, sink
        self.temp = tempfile.TemporaryDirectory(
            prefix=".serveutil-pm-", dir=sink.path.parent
        )
        self.path = Path(self.temp.name).resolve() / "hbm.jsonl.gz"
        self.error = open(self.path.with_suffix(".log"), "w+", encoding="utf-8")
        self.process = None
        self.ready = False
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = uuid
            lib = _library_dir()
            if lib:
                env["LD_LIBRARY_PATH"] = os.pathsep.join(
                    filter(None, (lib, env.get("LD_LIBRARY_PATH")))
                )
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(self.path),
                    str(interval_ms),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.error,
                text=True,
                env=env,
            )
            readable, _, _ = select.select([self.process.stdout], [], [], 15)
            if not readable:
                raise RuntimeError("CUPTI PM initialization timed out")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(self._failure())
            status = json.loads(line)
            if not status.get("ready"):
                raise RuntimeError("CUPTI PM failed to start")
            self.sink.metadata.setdefault("cupti_pm", {})[uuid] = status
            self.ready = True
        except Exception:
            self.close()
            raise

    @staticmethod
    def available():
        return (
            sys.platform.startswith("linux")
            and importlib.util.find_spec("cupti") is not None
        )

    def _failure(self):
        self.error.flush()
        self.error.seek(0)
        lines = self.error.read().strip().splitlines()
        return (
            lines[-1][:500]
            if lines
            else f"CUPTI PM process exited with status {self.process.poll()}"
        )

    def sample(self):
        pass  # The subprocess decodes continuously, independently of NVML's refresh period.

    def close(self):
        try:
            if self.process is None:
                return
            if self.ready:
                try:
                    output, _ = self.process.communicate(input="stop\n", timeout=30)
                    if self.process.returncode:
                        raise RuntimeError(self._failure())
                    status = json.loads(output)
                    if not status.get("complete"):
                        raise RuntimeError("CUPTI PM did not finalize its measurements")
                    self.sink.metadata["cupti_pm"][self.uuid].update(status)
                    count = 0
                    with gzip.open(self.path, "rt", encoding="utf-8") as file:
                        for line in file:
                            start, end, read, write = json.loads(line)
                            count += 1
                            for direction, value in (("read", read), ("write", write)):
                                self.sink.emit(
                                    "hbm." + direction,
                                    f"HBM {direction}",
                                    "GB/s",
                                    self.uuid,
                                    "gpu",
                                    "CUPTI PM",
                                    start,
                                    end,
                                    value,
                                )
                    if count == 0:
                        self.sink.unavailable(
                            self.uuid,
                            "hbm.read/write",
                            "No complete CUPTI PM interval in this window",
                        )
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    self.sink.unavailable(self.uuid, "hbm.read/write", exc)
        finally:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.kill()
                self.process.communicate()
            self.error.close()
            self.temp.cleanup()


if __name__ == "__main__":
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        collect(sys.argv[1], float(sys.argv[2]))
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
