"""Coordinate engine-native profile windows and node-local hardware telemetry."""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .bundle import Capture


def control(target, engine, start):
    route = "start_profile" if start else "stop_profile"
    body = {}
    if start and engine == "sglang":
        body = {
            "activities": ["CPU", "GPU"],
            "detailed_annotations": True,
            "with_stack": target.get("with_stack", False),
        }
        if target.get("output_dir"):
            body["output_dir"] = target["output_dir"]
    headers = {"Content-Type": "application/json"}
    token = os.getenv(
        target.get(
            "key_env", "SGLANG_ADMIN_API_KEY" if engine == "sglang" else "VLLM_API_KEY"
        )
    )
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(
        target["url"].rstrip("/") + "/" + route, json.dumps(body).encode(), headers
    )
    sent = time.time_ns()
    with urlopen(request, timeout=target.get("timeout", 240)) as response:
        content = response.read()
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(f"{target['url']} {route} failed: {payload}")
    return {"sent_ns": sent, "received_ns": time.time_ns()}


def map_all(fn, items):
    with ThreadPoolExecutor(max_workers=len(items)) as executor:
        return list(executor.map(fn, items))


def _recover_all(fn, items):
    """Finish each independent teardown operation even if a peer fails."""

    def attempt(item):
        try:
            return fn(item), None
        except Exception as exc:
            return None, exc

    return map_all(attempt, items)


class Worker:
    def __init__(self, config):
        self.config = config
        self.name = config["name"]
        command = config.get("command", [sys.executable, "-m", "serveutil.worker"])
        if not isinstance(command, list) or not command:
            raise ValueError("node command must be a nonempty argument list")
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )

    def send(self, **data):
        self.proc.stdin.write(json.dumps(data).encode() + b"\n")
        self.proc.stdin.flush()

    def receive(self):
        line = self.proc.stdout.readline(1024 * 1024)
        if not line:
            raise RuntimeError(f"{self.name}: worker closed the control channel")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"{self.name}: {result['error']}")
        return result

    def clock(self):
        probes = []
        for _ in range(5):
            start = time.time_ns()
            self.send(op="clock")
            peer = self.receive()["time_ns"]
            end = time.time_ns()
            probes.append((end - start, (end + start) // 2 - peer))
        latency, offset = min(probes)
        return {"offset_ns": offset, "uncertainty_ns": (latency + 1) // 2}

    def start(self, interval):
        self.anchor = self.clock()
        self.send(
            op="start",
            roots=self.config["roots"],
            devices=self.config.get("devices"),
            interval_ms=interval,
            workdir=self.config["workdir"],
            calibrate=self.config.get("calibrate", False),
            calibration_file=self.config.get("calibration_file"),
        )
        if self.receive().get("ready") is not True:
            raise RuntimeError("worker did not start")

    def collect(self, capture):
        end_clock = self.clock()
        self.send(op="stop")
        folder = capture.path / "nodes" / self.name
        folder.mkdir(parents=True)
        records = []
        calibration = None
        while True:
            msg = self.receive()
            if msg.get("type") == "done":
                meta = msg["telemetry"]
                return records, dict(
                    name=self.name,
                    device_map=self.config.get("device_map", {}),
                    rank_device_maps=self.config.get("rank_device_maps", {}),
                    collective_groups=self.config.get("collective_groups", {}),
                    clock=self.anchor,
                    end_clock=end_clock,
                    telemetry=meta,
                    calibration=calibration,
                    samples=str(
                        (folder / "samples.jsonl.gz").relative_to(capture.path)
                    ),
                )
            if msg.get("type") != "file":
                raise ValueError("invalid worker frame")
            role = msg["role"]
            if not re.fullmatch(r"[\w.-]+", role) or role in {".", ".."}:
                raise ValueError("invalid role")
            fname = Path(msg["name"]).name
            relative = Path(msg.get("relative", fname))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("worker sent a file outside the role directory")
            dest = (
                folder / "samples.jsonl.gz"
                if role == "telemetry"
                else folder / role / relative
            )
            if role == "calibration":
                dest = folder / "calibration.json"
                calibration = str(dest.relative_to(capture.path))
            dest.parent.mkdir(parents=True, exist_ok=True)
            remaining = msg["size"]
            digest = hashlib.sha256()
            with dest.open("xb") as out:
                while remaining:
                    chunk = self.proc.stdout.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise EOFError(f"incomplete {fname}")
                    digest.update(chunk)
                    out.write(chunk)
                    remaining -= len(chunk)
            if msg.get("checksum") != "sha256":
                raise ValueError(
                    "update the worker to the same ServeUtil version as the controller"
                )
            if self.receive().get("sha256") != digest.hexdigest():
                raise ValueError(f"transfer checksum mismatch for {fname}")
            if role not in {"telemetry", "calibration"}:
                match = re.search(r"(?:TP-|rank[_-]?)(\d+)", fname, re.I)
                parts = re.findall(
                    r"(?:^|[-_.])(DP|PP|EP|CP|DCP|TP)[-_]?(\d+)", fname, re.I
                )
                worker = (
                    ".".join(axis.lower() + number for axis, number in parts)
                    if len(parts) > 1
                    else (match[1] if match else fname)
                )
                records.append(
                    dict(
                        file=str(dest.relative_to(capture.path)),
                        rank=f"{self.name}/{role}/{worker}",
                        role=role,
                        node=self.name,
                        sha256=digest.hexdigest(),
                        auxiliary="async_llm" in fname,
                    )
                )

    def close(self):
        self.proc.stdin.close()
        self.proc.stdout.close()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(10)


def capture_engine(
    engine,
    targets,
    nodes,
    output,
    seconds,
    interval_ms=100,
    transfers=None,
    workload=None,
    links=(),
):
    if (
        engine not in {"vllm", "sglang"}
        or not targets
        or not nodes
        or not 0 < seconds <= 86400
    ):
        raise ValueError(
            "choose an engine, control endpoint(s), nodes and positive duration"
        )
    if len({node["name"] for node in nodes}) != len(nodes):
        raise ValueError("node names must be unique")
    for node in nodes:
        if not node.get("roots") or any(r.get("ranks", 1) < 1 for r in node["roots"]):
            raise ValueError(
                "each node needs a trace directory and a positive expected rank count"
            )
        for item in node["roots"]:
            role = item["role"]
            if role in {"telemetry", "calibration", ".", ".."} or not re.fullmatch(
                r"[\w.-]+", role
            ):
                raise ValueError(f"invalid trace role {role}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", node["name"]):
            raise ValueError("invalid node name")
    from .bundle import write_json

    session = Capture(output, engine)
    workers = []
    running_load = None
    errors = []

    def drain_load():
        nonlocal running_load
        if running_load is not None:
            current, running_load = running_load, None
            session.manifest["workload"] = current.stop()

    try:
        link_results = []
        if links:
            from .link import calibrate_link

            for spec in links:
                link_results.append(calibrate_link(spec))
            write_json(session.path / "links.json", link_results)
        for node in nodes:
            workers.append(Worker(node))
        map_all(lambda worker: worker.start(interval_ms), workers)
        if workload:
            running_load = workload
            workload.start()
        from .transfers import nixl_deltas, snapshot_nixl

        kv_targets = [
            t
            for t in targets
            if engine == "vllm" and t.get("role") in {"prefill", "decode"}
        ]
        kv_before = map_all(snapshot_nixl, kv_targets) if kv_targets else []
        # A rejected start (e.g. 409 already profiling) belongs to another session.
        # Unknown outcomes after a disconnect still need a bounded cleanup attempt.
        cleanup = []

        def start(target):
            try:
                result = control(target, engine, True)
            except HTTPError as exc:
                if exc.code >= 500:
                    cleanup.append(target)
                raise
            except BaseException:
                cleanup.append(target)
                raise
            cleanup.append(target)
            return result

        try:
            session.manifest["start"] = map_all(start, targets)
            time.sleep(seconds)
            # Some engines suspend scheduling while serializing their trace.
            # Finish outstanding requests while all serving roles can still run.
            drain_load()
            if kv_targets:
                kv_after = map_all(snapshot_nixl, kv_targets)
                session.manifest["kv_metrics"] = [
                    row
                    for before, after in zip(kv_before, kv_after)
                    for row in nixl_deltas(before, after)
                ]
        finally:
            try:
                drain_load()
            finally:
                if cleanup:
                    stops = _recover_all(lambda t: control(t, engine, False), cleanup)
                    session.manifest["stop"] = [
                        value for value, err in stops if err is None
                    ]
                    for endpoint, (_, err) in zip(cleanup, stops):
                        if err is not None:
                            errors.append(f"{endpoint['url']} stop_profile: {err}")
        session.manifest["nodes"] = []
        for worker, (result, err) in zip(
            workers, _recover_all(lambda w: w.collect(session), workers)
        ):
            if err is not None:
                errors.append(f"{worker.name} collection: {err}")
                continue
            records, meta = result
            session.manifest["traces"].extend(records)
            port_rows = [
                r
                for link in link_results
                for r in link["benchmarks"]
                if r.get("node") == worker.name
            ]
            if port_rows:
                path = session.path / "nodes" / worker.name / "calibration.json"
                base = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else {"benchmarks": []}
                )
                base["benchmarks"].extend(port_rows)
                write_json(path, base)
                meta["calibration"] = str(path.relative_to(session.path))
            session.manifest["nodes"].append(meta)
            for root in worker.config["roots"]:
                ranks = {
                    r["rank"]
                    for r in records
                    if r["role"] == root["role"] and not r["auxiliary"]
                }
                if len(ranks) < root.get("ranks", 1):
                    errors.append(
                        f"{worker.name}/{root['role']}: only {len(ranks)} worker traces collected"
                    )
        if transfers:
            from .transfers import load_transfers

            transfer_file = session.path / "transfers.json"
            write_json(transfer_file, load_transfers(transfers))
            session.manifest["transfers"] = transfer_file.name
        if workload:
            workload.check()
        if errors:
            raise RuntimeError("; ".join(errors))
        session.finish()
        return session.path
    except BaseException as exc:
        try:
            drain_load()
        finally:
            if errors:
                session.manifest["collection_errors"] = errors
            session.finish(exc)
            if session.manifest["traces"]:
                from .report import make_report

                try:
                    write_json(session.path / "report.json", make_report(session.path))
                except Exception as rendering_error:
                    session.manifest["report_error"] = str(rendering_error)
                    session.save()
        raise
    finally:
        for worker in workers:
            worker.close()
