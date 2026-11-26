"""Export the aligned report lanes and hardware rates as native Chrome events."""

import json
import os
import tempfile
from pathlib import Path

from .report import make_report


def export_capture(directory, output):
    output = Path(output)
    if output.suffix != ".json":
        raise ValueError("Perfetto output must have the .json extension")
    # A run is the authoritative input for reconstruction, including all its JSON files.
    if output.exists() and output.resolve().is_relative_to(Path(directory).resolve()):
        raise ValueError("use a new file or an output outside the capture")
    data = make_report(directory)

    def events():
        for pid, trace in enumerate(data["traces"], 1):
            yield {
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "name": "process_name",
                "args": {"name": trace["rank"]},
            }
            lanes = {}
            for e in trace["events"]:
                lane = e["device"], e["lane"], e["kind"]
                if lane not in lanes:
                    lanes[lane] = len(lanes) + 1
                    yield {
                        "ph": "M",
                        "pid": pid,
                        "tid": lanes[lane],
                        "name": "thread_name",
                        "args": {"name": f"{e['kind']} · {e['device']} / {e['lane']}"},
                    }
                args = dict(e.get("args", {}))
                if e.get("original"):
                    args["serveutil.original_name"] = e["original"]
                if e.get("gemm"):
                    args.update(e["gemm"])
                if e.get("collective"):
                    args["serveutil.collective"] = e["collective"]
                if e.get("copy"):
                    args["serveutil.copy"] = e["copy"]
                yield {
                    "ph": "X",
                    "pid": pid,
                    "tid": lanes[lane],
                    "ts": e["ts"],
                    "dur": e["dur"],
                    "name": e["name"],
                    "cat": e["kind"],
                    "args": args,
                }
            for c in trace["counters"]:
                for ts, value in c["points"]:
                    yield dict(
                        ph="C",
                        pid=pid,
                        tid=0,
                        ts=ts,
                        name=c["device"] + " · " + c["name"],
                        args={"value": value},
                    )
        first = 1 + len(data["traces"])
        for idx, c in enumerate(data["hardware"], first):
            label = " / ".join(filter(None, (c.get("node"), c["device"], c["name"])))
            yield {
                "ph": "M",
                "pid": idx,
                "tid": 0,
                "name": "process_name",
                "args": {"name": label},
            }
            for ts, value, start in c["points"]:
                yield dict(
                    ph="C",
                    pid=idx,
                    tid=0,
                    ts=ts,
                    name=c["name"] + " (" + c["unit"] + ")",
                    args={"value": value},
                )
                if c["baseline"]:
                    yield dict(
                        ph="C",
                        pid=idx,
                        tid=0,
                        ts=ts,
                        name="Measured reference (%)",
                        args={"value": 100 * value / c["baseline"]["value"]},
                    )
        idx = first + len(data["hardware"])
        yield dict(
            ph="M",
            pid=idx,
            tid=0,
            name="process_name",
            args={"name": "Optimization opportunities"},
        )
        for f in data["findings"]:
            yield dict(
                ph="X",
                pid=idx,
                tid=0,
                ts=f["ts"],
                dur=f["dur"],
                name=f["name"],
                cat="opportunity",
                args=f,
            )

        if data.get("transfers") or data.get("kv_metrics"):
            idx += 1
            yield dict(ph="M", pid=idx, tid=0, name="process_name", args={"name": "KV transfers"})
            for i, row in enumerate(data.get("transfers", []), 1):
                yield dict(
                    ph="X", pid=idx, tid=i, ts=row["ts"], dur=row["dur"],
                    name="KV handoff", cat="kv_transfer", args=row,
                )
            for i, row in enumerate(data.get("kv_metrics", []), 1 + len(data.get("transfers", []))):
                if row.get("status") == "ok" and "ts" in row:
                    yield dict(
                        ph="X", pid=idx, tid=i, ts=row["ts"], dur=row["dur"],
                        name=f'{row["role"]} · NIXL counter interval', cat="kv_interval", args=row,
                    )

    with tempfile.NamedTemporaryFile(
        "w", dir=output.parent, delete=False, encoding="utf-8"
    ) as stream:
        path = Path(stream.name)
        count = 0
        try:
            stream.write('{"traceEvents":[')
            for event in events():
                if count:
                    stream.write(",")
                json.dump(event, stream, allow_nan=False, separators=(",", ":"))
                count += 1
            stream.write("]}")
            stream.close()
            os.replace(path, output)
        finally:
            path.unlink(missing_ok=True)
    return {"kept": count}
