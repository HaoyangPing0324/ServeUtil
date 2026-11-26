"""Build a compact Perfetto timeline from a single PyTorch Chrome trace."""

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import re
import tempfile

import ijson


ANNOTATIONS = {"user_annotation", "gpu_user_annotation"}
GPU_EVENTS = {"kernel", "gpu_memcpy", "gpu_memset"}
COLLECTIVES = re.compile(r"allreduce|allgather|reducescatter|alltoall|broadcast|sendrecv", re.I)
COLLECTIVE_LABELS = {
    "allreduce": "AllReduce", "allgather": "AllGather",
    "reducescatter": "ReduceScatter", "alltoall": "AllToAll",
    "broadcast": "Broadcast", "sendrecv": "SendRecv",
}


def kernel_label(name):
    lower = name.lower()
    if "nccl" in lower:
        match = COLLECTIVES.search(lower.replace("_", ""))
        if match:
            return COLLECTIVE_LABELS[match.group().lower()]
    # Test the outer function: a GEMM template argument alone does not identify a kernel.
    outer = name.split("<", 1)[0].split("(", 1)[0].strip()
    if "gemm" in outer.lower():
        return "GEMM"
    if "attention" in outer.lower() or "flash_fwd" in outer.lower():
        return "Attention"
    tokens = outer.rsplit("::", 1)[-1].split()
    short = tokens[-1] if tokens else name
    return short if len(short) <= 55 else short[:52] + "…"


def overview_event(event):
    phase = event.get("ph")
    if phase in {"C", "M"}:
        return event
    if phase != "X":
        return None
    categories = set(event.get("cat", "").split(","))
    if categories & ANNOTATIONS:
        return event
    if not categories & GPU_EVENTS:
        return None
    name = event.get("name", "")
    if "kernel" in categories:
        short = kernel_label(name)
    elif "gpu_memcpy" in categories:
        short = name.split(" (", 1)[0]
    else:
        short = name
    if short != name:
        event["args"] = {**(event.get("args") or {}), "serveutil.original_name": name}
        event["name"] = short
    return event


def trace_items(stream, header):
    # Keep the Kineto time origin even if the producer put it after traceEvents.
    found_events = False

    def tokens():
        nonlocal found_events
        for prefix, kind, value in ijson.parse(stream):
            if prefix == "traceEvents" and kind == "start_array":
                found_events = True
            if prefix in {"baseTimeNanoseconds", "displayTimeUnit", "trace_id"} and kind in {"number", "string"}:
                header[prefix] = value
            yield prefix, kind, value

    yield from ijson.items(tokens(), "traceEvents.item")
    if not found_events:
        raise ValueError("expected a PyTorch Chrome trace object with a traceEvents array")


def export(source, destination):
    if source.resolve() == destination.resolve() or (destination.exists() and source.samefile(destination)):
        raise ValueError("choose an output different from the input")
    if destination.suffix != ".json":
        raise ValueError("output must end in .json")
    counts = Counter()
    header = {}
    opener = gzip.open if source.suffix == ".gz" else open
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        with opener(source, "rb") as stream, temp_path.open("w", encoding="utf-8") as out:
            out.write('{"traceEvents":[')
            for event in trace_items(stream, header):
                counts["read"] += 1
                if not isinstance(event, dict):
                    raise ValueError("invalid event in traceEvents array")
                event = overview_event(event)
                if event is None:
                    continue
                if counts["kept"]:
                    out.write(",")
                out.write("\n")
                json.dump(event, out, ensure_ascii=True, separators=(",", ":"), default=float)
                counts["kept"] += 1
                counts[event.get("ph")] += 1
            out.write("\n]")
            for key, value in header.items():
                out.write("," + json.dumps(key) + ":" + json.dumps(value, default=float))
            out.write("}\n")
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="input .json or .json.gz")
    parser.add_argument("-o", "--output", required=True, type=Path, help="overview .json to create or replace")
    args = parser.parse_args()
    try:
        counts = export(args.trace, args.output)
    except (OSError, EOFError, ValueError, ijson.JSONError) as exc:
        parser.exit(1, f"serveutil: {exc}\n")
    print(f"{args.output}: kept {counts['kept']:,}/{counts['read']:,} events "
          f"({counts['X']:,} slices, {counts['C']:,} counter samples)")
    if not counts["X"] and not counts["C"]:
        print("No matching GPU work, complete annotations or counters found in this capture.")


if __name__ == "__main__":
    main()
