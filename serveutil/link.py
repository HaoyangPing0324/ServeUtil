"""Measure directional TCP and RDMA WRITE throughput between two ServeUtil workers."""

import gzip
import ipaddress
import json
import math
import os
import secrets
import socket
import struct
import tempfile
import time
from pathlib import Path

from .reference import port_reference, steady_rdma_reference
from .telemetry import TelemetrySampler


def _receive(sock, count):
    data = bytearray()
    while len(data) < count:
        block = sock.recv(count - len(data))
        if not block:
            raise EOFError("Network calibration peer disconnected")
        data.extend(block)
    return bytes(data)


def _test_duration(seconds):
    if not math.isfinite(seconds) or not 1 <= seconds <= 60:
        raise ValueError("TCP reference duration must be between 1 and 60 seconds")
    return seconds


def _transfer(msg, send, sampler):
    seconds = _test_duration(float(msg.get("seconds", 2)))
    address = ipaddress.ip_address(msg["address"])
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    token = bytes.fromhex(msg["token"])
    if len(token) != 32:
        raise ValueError("Invalid link calibration session token")
    server = msg["role"] == "receive"
    with socket.socket(family, socket.SOCK_STREAM) as endpoint:
        endpoint.settimeout(seconds + 15)
        endpoint.bind((str(address), 0))
        if server:
            endpoint.listen(1)
            send({"port": endpoint.getsockname()[1]})
            conn, _ = endpoint.accept()
        else:
            endpoint.connect((msg["peer_address"], msg["port"]))
            conn = endpoint
        with conn:
            conn.settimeout(seconds + 10)
            block = bytearray(1024**2)
            total = 0
            if server:
                if not secrets.compare_digest(_receive(conn, len(token)), token):
                    raise ValueError("Network calibration session token mismatch")
                conn.sendall(b"G")
            else:
                conn.sendall(token)
                if _receive(conn, 1) != b"G":
                    raise ValueError("Network calibration handshake failed")
            start = sampler.now()
            deadline = time.monotonic() + seconds
            if server:
                while time.monotonic() < deadline + 10:
                    size = conn.recv_into(block)
                    if not size:
                        break
                    total += size
                else:
                    raise TimeoutError("Network calibration stream did not finish")
                end = sampler.now()
                conn.sendall(struct.pack("!Q", total))
            else:
                while time.monotonic() < deadline:
                    conn.sendall(block)
                    total += len(block)
                conn.shutdown(socket.SHUT_WR)
                if struct.unpack("!Q", _receive(conn, 8))[0] != total:
                    raise ValueError("The receiver observed a different payload length")
                end = sampler.now()
    return start, end, {"bytes": total, "payload_GBps": total / (end - start)}


def run_endpoint(msg, send):
    """Handle one authorized link experiment in a node-local worker."""
    if msg.get("role") not in {"send", "receive"}:
        raise ValueError("Unknown link endpoint role")
    work = Path(msg["workdir"])
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        path = Path(tmp) / "port.jsonl.gz"
        # Worker processes service one request at a time. Restore explicit interface
        # selection even if a later calibration in the same worker fails.
        old = os.environ.get("SERVEUTIL_NET_INTERFACES")
        os.environ["SERVEUTIL_NET_INTERFACES"] = msg.get("interface", "")
        try:
            sampler = TelemetrySampler(path, devices=[], interval_ms=100).start()
        finally:
            if old is None:
                del os.environ["SERVEUTIL_NET_INTERFACES"]
            else:
                os.environ["SERVEUTIL_NET_INTERFACES"] = old
        try:
            if msg.get("transport") == "rdma-write":
                from . import rdma

                start, end, stats = rdma.transfer(msg, send, sampler)
                device = msg["rdma_port"]
                local_memory = (
                    "GPU memory"
                    if msg.get("cuda_device") is not None
                    else "host memory"
                )
                peer_memory = (
                    "GPU memory"
                    if msg.get("peer_cuda_device") is not None
                    else "host memory"
                )
                source_memory, dest_memory = (
                    (local_memory, peer_memory)
                    if msg["role"] == "send"
                    else (peer_memory, local_memory)
                )
                workload = f"ib_write_bw (RC, {source_memory} to {dest_memory}, 64 KiB)"
            else:
                start, end, stats = _transfer(msg, send, sampler)
                device = msg["interface"]
                workload = "single TCP stream (host memory)"
        finally:
            sampler.stop()
        direction = "tx" if msg["role"] == "send" else "rx"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = (dict(json.loads(line), node=msg["name"]) for line in stream)
            if msg.get("transport") == "rdma-write":
                # Only the sending process is guaranteed to report payload bandwidth.
                # Locate each endpoint's sustained counter interval on its own clock.
                windows = [
                    s
                    for s in rows
                    if s["metric"] == "rdma." + direction
                    and s["device"] == device
                    and s["unit"] == "GB/s"
                    and start <= s["start_ns"] < s["end_ns"] <= end
                ]
                stats.update(port_samples=windows, workload=workload, device=device)
            else:
                stats["benchmarks"] = port_reference(
                    rows,
                    start_ns=start,
                    end_ns=end,
                    peer=msg["peer_name"],
                    workload=workload,
                    device=device,
                    direction=direction,
                )
        return {
            "node": msg["name"],
            "peer": msg["peer_name"],
            "direction": direction,
            "duration_s": (end - start) / 1e9,
            **stats,
        }


def calibrate_link(spec):
    """Measure both directions by default. Each endpoint uses a regular serveutil.worker command.

    Pass source and destination dictionaries with name, workdir, address and an
    optional worker command. TCP measures the named network interface. RDMA
    WRITE uses each endpoint's rdma_port (device/port) with ib_write_bw; perftest
    can name a custom binary and cuda_device selects a CUDA-enabled memory
    buffer on that endpoint. Omit cuda_device for host memory.

    The benchmarks describe entire interfaces or ports. The transfers list
    reports payload throughput separately from those physical counters.
    """
    from .engines import Worker

    seconds = _test_duration(float(spec.get("seconds", 2)))
    transport = spec.get("transport", "tcp")
    if transport not in ("tcp", "rdma-write"):
        raise ValueError("Choose tcp or rdma-write")
    if transport == "rdma-write" and (seconds < 4 or not seconds.is_integer()):
        raise ValueError(
            "RDMA WRITE needs an integer duration of at least four seconds"
        )
    left, right = spec["source"], spec["destination"]
    for endpoint in (left, right):
        device_key = "rdma_port" if transport == "rdma-write" else "interface"
        if not endpoint.get(device_key) or not endpoint.get("workdir"):
            raise ValueError(f"Network endpoints require {device_key} and workdir")
        ipaddress.ip_address(endpoint["address"])
    if left["name"] == right["name"]:
        raise ValueError("Use distinct source and destination node names")
    routes = [(left, right)]
    if spec.get("bidirectional", True):
        routes.append((right, left))
    result = {"schema": 1, "transport": transport, "benchmarks": [], "transfers": []}
    for src, dst in routes:
        workers = []
        try:
            for config in (dst, src):
                workers.append(Worker(config))
            listener, client = workers
            token = secrets.token_hex(32)

            def begin(worker, cfg, role, peer, **extra):
                worker.send(
                    op="link",
                    role=role,
                    name=cfg["name"],
                    peer_name=peer["name"],
                    address=cfg["address"],
                    peer_address=peer["address"],
                    workdir=cfg["workdir"],
                    transport=transport,
                    interface=cfg.get("interface", ""),
                    rdma_port=cfg.get("rdma_port"),
                    perftest=cfg.get("perftest"),
                    cuda_device=cfg.get("cuda_device"),
                    peer_cuda_device=peer.get("cuda_device"),
                    token=token,
                    seconds=seconds,
                    **extra,
                )

            begin(listener, dst, "receive", src)
            port = listener.receive()["port"]
            begin(client, src, "send", dst, port=port)
            sending = client.receive()
            receiving = listener.receive()
            if transport == "tcp" and (
                sending["bytes"] <= 0 or sending["bytes"] != receiving["bytes"]
            ):
                raise ValueError("Incomplete link calibration transfer")
            for observed in (sending, receiving):
                if transport == "rdma-write":
                    refs = steady_rdma_reference(
                        observed.pop("port_samples"),
                        payload_gbs=sending["payload_GBps"],
                        seconds=seconds,
                        device=observed["device"],
                        peer=observed["peer"],
                        workload=observed["workload"],
                        direction=observed["direction"],
                    )
                else:
                    refs = observed.pop("benchmarks")
                result["benchmarks"].extend(refs)
                result["transfers"].append(observed)
        finally:
            for worker in workers:
                worker.close()
    return result
