"""Run perftest RDMA WRITE between host or GPU buffers on a named HCA port."""

import ipaddress
import json
import math
import shutil
import socket
import subprocess
import time
from pathlib import Path


def _ready(port, proc, rdma_cm=False):
    """Observe the control socket without connecting a competing handshake."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and proc.poll() is None:
        if rdma_cm:
            # RDMA CM listeners do not appear in /proc/net/tcp on eRDMA.
            observed = subprocess.run(
                ["rdma", "-j", "resource", "show", "cm_id"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            for item in json.loads(observed.stdout):
                if (
                    item.get("state") == "LISTEN"
                    and item.get("pid") == proc.pid
                    and item.get("src-addr", "").endswith(f":{port}")
                ):
                    return
            time.sleep(0.05)
            continue
        for path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
            if not path.exists():
                continue
            for line in path.read_text().splitlines()[1:]:
                fields = line.split()
                if fields[3] == "0A" and int(fields[1].split(":")[-1], 16) == port:
                    return
        time.sleep(0.05)
    raise RuntimeError("ib_write_bw did not open its control socket")


def transfer(msg, send, sampler):
    program = shutil.which(msg.get("perftest") or "ib_write_bw")
    if not program:
        raise RuntimeError(
            "ib_write_bw (linux-rdma/perftest) is required on both RDMA endpoints"
        )
    hca, sep, port_num = msg["rdma_port"].partition("/")
    if not hca or not sep or not port_num.isdecimal() or int(port_num) <= 0:
        raise ValueError("Use an RDMA port in the form device/port")
    rdma_cm = (
        Path("/sys/class/infiniband") / hca / "device/driver"
    ).resolve().name == "erdma"
    seconds = int(msg["seconds"])
    if not 4 <= seconds <= 60:
        raise ValueError("RDMA WRITE reference duration must be 4–60 seconds")
    server = msg["role"] == "receive"
    address = ipaddress.ip_address(msg["address"])
    if server:
        family = socket.AF_INET if address.version == 4 else socket.AF_INET6
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((str(address), 0))
            port = sock.getsockname()[1]
    else:
        port = int(msg["port"])
    args = [
        program,
        "--report_gbits",
        "--output=bandwidth",
        "-s",
        "65536",
        "-F",
        "-d",
        hca,
        "-i",
        port_num,
        "-p",
        str(port),
        "-D",
        str(seconds),
        "-f",
        "1",
        "--bind_source_ip",
        str(address),
    ]
    cuda = msg.get("cuda_device")
    if rdma_cm:
        args += ["--rdma_cm"]
    if cuda is not None:
        if type(cuda) is not int or cuda < 0:
            raise ValueError("cuda_device must be a nonnegative CUDA ordinal")
        args += ["--use_cuda", str(cuda)]
    if not server:
        args.append(str(ipaddress.ip_address(msg["peer_address"])))
    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        if server:
            _ready(port, proc, rdma_cm)
            send({"port": port})
        start = sampler.now()
        stdout, stderr = proc.communicate(timeout=seconds + 20)
        end = sampler.now()
        if proc.returncode:
            detail = (stderr or stdout).strip().splitlines()
            raise RuntimeError(
                "ib_write_bw failed: "
                + (detail[-1][:500] if detail else str(proc.returncode))
            )
        # --report_gbits --output=bandwidth selects decimal Gbit/s. The server
        # may emit no logical bandwidth; each endpoint still measures its port.
        measured = None
        for line in reversed(stdout.splitlines()):
            try:
                value = float(line.strip()) / 8
            except ValueError:
                continue
            if math.isfinite(value) and value > 0:
                measured = value
                break
        if not server and measured is None:
            raise ValueError("ib_write_bw returned no positive client bandwidth")
        return (
            start,
            end,
            {
                "payload_GBps": measured,
                "connection_manager": "rdma_cm" if rdma_cm else "socket",
            },
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
