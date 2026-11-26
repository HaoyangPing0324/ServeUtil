"""Import upstream microbenchmark results without mixing logical and physical bandwidth."""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _positive(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _nccl(data, devices, ranks):
    if data.get("out_of_bounds", {}).get("okay") is False:
        raise ValueError("nccl-tests failed correctness checks")
    group = list(devices or ())
    if ranks is None:
        ranks = len(group) or None
    collective = Path(data["args"][0]).name.removesuffix("_perf")
    rows = []
    for experiment in data["results"]:
        for placement in ("in_place", "out_of_place"):
            sample = experiment.get(placement)
            if not sample or sample.get("nwrong") not in (None, 0):
                continue
            params = {
                "collective": collective,
                "bytes": experiment["size"],
                "dtype": experiment["type"],
                "placement": placement,
                "ranks": ranks,
                "devices": group,
                "reduction": experiment.get("redop"),
                "version": data.get("nccl_version"),
            }
            root = str(experiment.get("root", "")).strip()
            if collective in ("broadcast", "reduce") and root.isdecimal():
                params["root_rank"] = int(root)
            for field in ("alg_bw", "bus_bw"):
                value = _positive(sample.get(field))
                if value is not None:
                    rows.append(
                        {
                            "metric": f"nccl.{collective}.{field}",
                            "device": "collective group",
                            "value": value,
                            "unit": "GB/s",
                            "scope": "collective",
                            "source": "nccl-tests",
                            "parameters": params,
                            "time_us": sample.get("time"),
                        }
                    )
    return rows


def _nvbandwidth(data, devices):
    if data.get("error"):
        raise ValueError("nvbandwidth reported a failed run")
    rows = []
    for case in data["testcases"]:
        desc = case.get("bandwidth_description", "")
        if case.get("status") != "Passed" or case.get("error") or "(GB/s)" not in desc:
            continue
        for i, row in enumerate(case.get("bandwidth_matrix", ())):
            for j, cell in enumerate(row):
                rate = _positive(cell)
                if rate is None:
                    continue
                # Preserve upstream's row/column orientation, including simultaneous-copy cases.
                params = {
                    "case": case["name"],
                    "row": i,
                    "column": j,
                    "description": desc,
                    "devices": list(devices or ()),
                    "version": data.get("version"),
                }
                device = (
                    str(devices[j])
                    if devices and "GPU(column)" in desc
                    else f"column {j}"
                )
                rows.append(
                    {
                        "metric": "nvbandwidth." + case["name"],
                        "device": device,
                        "scope": "copy test",
                        "value": rate,
                        "unit": "GB/s",
                        "source": "nvbandwidth",
                        "parameters": params,
                    }
                )
    return rows


def import_reference(path, *, devices=None, ranks=None):
    """Read nvbandwidth JSON, nccl-tests -J, or a saved ServeUtil reference file.

    Supply device UUIDs in the upstream run's order and the global group size
    for multi-node nccl-tests. Only performance fields are copied.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "nvbandwidth" in data:
        rows = _nvbandwidth(data["nvbandwidth"], devices)
    elif "nccl_version" in data and "results" in data and data.get("args"):
        rows = _nccl(data, devices, ranks)
    elif isinstance(data.get("benchmarks"), list):
        rows = data["benchmarks"]
        required = {"metric", "device", "source", "unit", "value"}
        if any(
            not isinstance(r, dict)
            or not required <= r.keys()
            or _positive(r["value"]) is None
            for r in rows
        ):
            raise ValueError("Invalid ServeUtil benchmark record")
    else:
        raise ValueError("Expected nvbandwidth, nccl-tests or ServeUtil reference JSON")
    if not rows:
        raise ValueError("No successful bandwidth measurements in reference")
    return rows


def port_reference(samples, *, start_ns, end_ns, peer, workload, device, direction):
    """Reduce whole, same-source port windows during a known sustained transfer."""
    if direction not in ("tx", "rx") or start_ns >= end_ns:
        raise ValueError("Expected tx or rx and a positive reference interval")
    groups = defaultdict(list)
    for s in samples:
        if s["device"] != device or s["metric"] not in (
            "rdma." + direction,
            "netdev." + direction,
        ):
            continue
        if start_ns <= s["start_ns"] < s["end_ns"] <= end_ns and s["unit"] == "GB/s":
            if math.isfinite(s["value"]) and s["value"] >= 0:
                groups[
                    (s["node"], s["device"], s["metric"], s["source"], s["scope"])
                ].append(s["value"])
    rows = []
    for (node, dev, metric, source, scope), values in groups.items():
        median = statistics.median(values)
        if median > 0 and len(values) >= 3:
            rows.append(
                {
                    "node": node,
                    "device": dev,
                    "metric": metric,
                    "source": source,
                    "scope": scope,
                    "unit": "GB/s",
                    "value": median,
                    "workload": workload,
                    "parameters": {"peer": peer, "direction": direction},
                    "samples": len(values),
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                }
            )
    if not rows:
        raise ValueError(
            "Need three complete port samples and positive median throughput"
        )
    return rows


def steady_rdma_reference(
    samples, *, payload_gbs, seconds, peer, workload, device, direction
):
    """Locate the sustained WRITE inside perftest's setup, data and teardown interval."""
    if _positive(payload_gbs) is None:
        raise ValueError("Missing client payload bandwidth for RDMA port calibration")
    if not samples:
        raise ValueError("No complete RDMA port counter intervals during perftest")
    samples.sort(key=lambda row: row["start_ns"])
    rates = [row["value"] for row in samples]
    runs = []
    current = []
    for index, row in enumerate(samples):
        # A three-window median tolerates a single delayed port counter update.
        # Only select bounds this way; retain *all* raw rates inside each run.
        neighboring = rates[max(index - 1, 0) : index + 2]
        active = statistics.median(neighboring) >= 0.65 * payload_gbs
        if current and (not active or current[-1]["end_ns"] != row["start_ns"]):
            runs.append(current)
            current = []
        if active:
            current.append(row)
    if current:
        runs.append(current)

    def duration(run):
        return run[-1]["end_ns"] - run[0]["start_ns"] if run else 0

    run = max(runs, key=duration, default=[])
    if duration(run) < seconds * 0.5e9:
        raise ValueError("Insufficient sustained RDMA port activity during perftest")
    refs = port_reference(
        run,
        start_ns=run[0]["start_ns"],
        end_ns=run[-1]["end_ns"],
        peer=peer,
        workload=workload,
        device=device,
        direction=direction,
    )
    for row in refs:
        if not 0.8 * payload_gbs <= row["value"] <= 1.35 * payload_gbs:
            raise ValueError(
                "RDMA port median disagrees with measured client payload bandwidth"
            )
        row.update(
            covered_seconds=duration(run) / 1e9,
            client_payload_GBps=payload_gbs,
            selection="sustained perftest interval",
        )
    return refs


def run_installed(devices, directory):
    """Use installed upstream tools on the selected GPUs; discard their private raw output."""
    import os
    import shutil
    import subprocess

    directory = Path(directory)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices))
    rows, missing = [], []
    cases = ["host_to_device_memcpy_ce", "device_to_host_memcpy_ce"]
    if len(devices) > 1:
        cases.append("device_to_device_memcpy_write_ce")
    tasks = [
        ("nvbandwidth", ["--format", "json", "-i", "3", "-b", "64", "-t", *cases], True)
    ]
    if len(devices) > 1:
        tasks.append(
            (
                "all_reduce_perf",
                [
                    "-b",
                    "1M",
                    "-e",
                    "64M",
                    "-f",
                    "8",
                    "-g",
                    str(len(devices)),
                    "-w",
                    "2",
                    "-n",
                    "10",
                    "-J",
                    str(directory / "nccl.json"),
                ],
                False,
            )
        )
    for tool, args, writes_stdout in tasks:
        program = shutil.which(tool)
        if not program:
            continue
        path = directory / ("nvbandwidth.json" if writes_stdout else "nccl.json")
        try:
            # Some versions include the process environment in their raw JSON.
            # The temporary directory is private; publish only whitelisted fields.
            completed = subprocess.run(
                [program, *args],
                check=True,
                env=env,
                cwd=directory,
                timeout=180,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if writes_stdout:
                path.write_bytes(completed.stdout)
            extra = import_reference(path, devices=devices, ranks=len(devices))
            if writes_stdout:
                for item in extra:
                    item["parameters"]["bytes"] = 64 * (1024**2)
                    item["parameters"]["samples"] = 3
            rows.extend(extra)
        except (subprocess.SubprocessError, OSError, ValueError, KeyError) as exc:
            missing.append({"device": "group", "metric": tool, "reason": str(exc)})
    return rows, missing
