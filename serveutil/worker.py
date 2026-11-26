"""Node-local capture worker. Stdout carries a framed control/data stream."""

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

from .bundle import write_json
from .telemetry import TelemetrySampler


def send(data):
    print(json.dumps(data, separators=(",", ":")), flush=True)


def snapshot(roots):
    current = {}
    for item in roots:
        root = Path(item["directory"]).resolve()
        if not root.is_dir():
            raise ValueError(f"trace directory does not exist: {root}")
        for path in root.rglob("*"):
            if path.is_file() and path.name.endswith(
                (".pt.trace.json", ".trace.json.gz", ".trace.json")
            ):
                st = path.stat()
                current[path] = (
                    item["role"],
                    str(path.relative_to(root)),
                    st.st_mtime_ns,
                    st.st_size,
                    st.st_ino,
                )
    return current


def transmit(path, **info):
    size = path.stat().st_size
    send(dict(type="file", name=path.name, size=size, checksum="sha256", **info))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise EOFError(f"{path.name} changed while reading")
            digest.update(chunk)
            sys.stdout.buffer.write(chunk)
            remaining -= len(chunk)
    sys.stdout.buffer.flush()
    send({"sha256": digest.hexdigest()})


def main():
    sampler = None
    temp = None
    try:
        for line in sys.stdin.buffer:
            msg = json.loads(line)
            op = msg["op"]
            if op == "clock":
                send({"time_ns": time.time_ns()})
            elif op == "link":
                from .link import run_endpoint

                send(run_endpoint(msg, send))
            elif op == "start":
                roots = msg["roots"]
                before = snapshot(roots)
                work = Path(msg["workdir"])
                work.mkdir(parents=True, exist_ok=True)
                temp = tempfile.TemporaryDirectory(dir=work)
                path = Path(temp.name) / "samples.jsonl.gz"
                baseline = None
                if msg.get("calibration_file"):
                    baseline = Path(msg["calibration_file"])
                    if not baseline.is_file():
                        raise ValueError(f"baseline does not exist: {baseline}")
                elif msg.get("calibrate"):
                    from .calibration import calibrate
                    from .telemetry import _identifier

                    devices = msg["devices"]
                    if devices:
                        # Explicit numeric telemetry selectors are physical NVML indexes.
                        import pynvml as nv

                        nv.nvmlInit()
                        try:
                            devices = [
                                nv.nvmlDeviceGetUUID(
                                    nv.nvmlDeviceGetHandleByIndex(int(d))
                                )
                                if str(d).isdigit()
                                else _identifier(d)
                                for d in devices
                            ]
                        finally:
                            nv.nvmlShutdown()
                    result = calibrate(devices, cache_dir=work / ".calibration")
                    baseline = Path(temp.name) / "calibration.json"
                    write_json(baseline, result)
                sampler = TelemetrySampler(
                    path, devices=msg["devices"], interval_ms=msg["interval_ms"]
                )
                sampler.start()
                send({"ready": True})
            elif op == "stop":
                metadata = sampler.stop()
                sampler = None
                after = snapshot(roots)
                transmit(path, role="telemetry")
                if baseline:
                    transmit(baseline, role="calibration")
                for file, fingerprint in sorted(after.items()):
                    if before.get(file) != fingerprint:
                        transmit(file, role=fingerprint[0], relative=fingerprint[1])
                send({"type": "done", "telemetry": metadata})
                break
            else:
                raise ValueError(f"unknown command {op}")
    except Exception as exc:
        send({"error": str(exc)})
        raise
    finally:
        if sampler is not None:
            sampler.stop()
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    main()
