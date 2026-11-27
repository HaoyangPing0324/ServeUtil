"""Merge original Chrome trace events into one JSON without summary filtering.

Timestamps in Chrome events are microseconds. Kineto's baseTimeNanoseconds is
the Unix time at ts=0. A ServeUtil worker clock offset maps node to controller.
"""

import gzip
import json
import os
import shutil
import tempfile
from collections import Counter
from decimal import Decimal
from itertools import count
from json.encoder import encode_basestring_ascii
from pathlib import Path

import ijson
from ijson.common import ObjectBuilder


def _open(path):
    return gzip.open(path, "rb") if path.suffix.lower() == ".gz" else path.open("rb")


def _header(path):
    """Read the entire envelope, including fields after the event array."""
    header = {}
    with _open(path) as stream:
        tokens = iter(ijson.basic_parse(stream))
        first = next(tokens, None)
        if first == ("start_array", None):
            return header, "item"
        if first != ("start_map", None):
            raise ValueError(
                f"{path.name}: expected a Chrome trace object or event array"
            )
        found = False
        for kind, key in tokens:
            if kind == "end_map":
                break
            if kind != "map_key":
                raise ValueError(f"{path.name}: invalid trace header")
            typ, value = next(tokens)
            skip = key == "traceEvents"
            if skip:
                if typ != "start_array" or found:
                    raise ValueError(
                        f"{path.name}: expected exactly one traceEvents array"
                    )
                found = True
            builder = None if skip else ObjectBuilder()
            depth = 0
            while True:
                if builder is not None:
                    builder.event(typ, value)
                if typ in ("start_map", "start_array"):
                    depth += 1
                elif typ in ("end_map", "end_array"):
                    depth -= 1
                if not depth:
                    break
                typ, value = next(tokens)
            if builder is not None:
                header[key] = builder.value
        # Exhaust the parser to reject trailing JSON.
        if next(tokens, None) is not None or not found:
            raise ValueError(f"{path.name}: expected a Chrome traceEvents array")
    return header, "traceEvents.item"


def _json(value):
    """Keep ijson's exact decimal numbers, including numeric private arguments."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, str):
        return encode_basestring_ascii(value)
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                encode_basestring_ascii(k) + ":" + _json(v) for k, v in value.items()
            )
            + "}"
        )
    if isinstance(value, list):
        return "[" + ",".join(_json(v) for v in value) + "]"
    return json.dumps(value, allow_nan=False)


def _sources(source):
    paths = [source] if isinstance(source, (str, os.PathLike)) else list(source)
    if not paths:
        raise ValueError("provide at least one trace file or capture directory")
    paths = [Path(p) for p in paths]
    if len(paths) == 1 and paths[0].is_dir():
        root = paths[0].resolve()
        manifest = json.loads((root / "capture.json").read_text(encoding="utf8"))
        nodes = {n["name"]: n for n in manifest.get("nodes", [])}
        result = []
        for trace in manifest["traces"]:
            path = (root / trace["file"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("trace path is outside the capture")
            node = nodes.get(trace.get("node"), {})
            result.append(
                (path, dict(trace), node.get("clock", {}), node.get("end_clock"))
            )
        return result
    return [(p, {"file": str(p), "rank": p.name}, {}, None) for p in paths]


class _Ids:
    """Allocate integer identities usable by both Chrome and Perfetto importers."""

    def __init__(self, serial):
        self.serial = serial
        self.values = {}

    def get(self, key, name=None):
        if key not in self.values:
            identifier = next(self.serial)
            # Chrome string pid/tid fields also name otherwise anonymous tracks.
            if isinstance(name, str) and name:
                identifier = f"{identifier} · {name}"
            self.values[key] = identifier
        return self.values[key]


class _Tracks:
    def __init__(self, serial):
        self.processes = _Ids(serial)
        self.threads = _Ids(serial)
        self.ids = _Ids(serial)
        self.frames = _Ids(serial)

    def frame(self, value):
        # Frame keys in JSON objects are strings, even for numeric references.
        return str(self.frames.get(str(value)))

    def event(self, e, shift):
        if not isinstance(e, dict):
            raise ValueError("expected event and sample records to be JSON objects")
        pid = e.get("pid", 0)
        e["pid"] = self.processes.get(pid, pid)
        tid = e.get("tid", 0)
        if tid == pid:
            # Perfetto identifies a process's main thread by tid == pid.
            self.threads.values[(pid, tid)] = e["pid"]
        e["tid"] = self.threads.get((pid, tid), tid)
        if "ts" in e and shift:
            e["ts"] += shift
        for key in ("id", "bind_id"):
            if key in e:
                e[key] = self.ids.get(e[key])
        if isinstance(e.get("id2"), dict):
            for key in ("local", "global"):
                if key in e["id2"]:
                    e["id2"][key] = self.ids.get(e["id2"][key])
        for key in ("sf", "esf"):
            if key in e:
                e[key] = self.frame(e[key])
        return e


def export_full(source, destination, *, clocks=None):
    """Write a full Chrome JSON from a capture, one trace, or a list of traces.

    Optional clocks map a manifest file path, rank, node or unique basename to
    an object. `origin_ns` specifies the reference epoch of raw ts=0 when a
    Kineto origin is absent. `offset_ns` is added to that epoch. An uncorrected
    wall clock is identified explicitly in the exported provenance.
    """
    dst = Path(destination)
    if dst.suffix != ".json":
        raise ValueError("full trace output must use the .json extension")
    if not isinstance(source, (str, os.PathLike)):
        source = list(source)
        if len(source) == 1:
            source = source[0]
    sources = _sources(source)
    if not sources:
        raise ValueError("capture has no raw traces")
    paths = [s[0].resolve() for s in sources]
    if len(paths) != len(set(paths)) or dst.resolve() in paths:
        raise ValueError("provide distinct input and output files")
    if (
        isinstance(source, (str, os.PathLike))
        and Path(source).is_dir()
        and dst.exists()
        and dst.resolve().is_relative_to(Path(source).resolve())
    ):
        raise ValueError("choose a new output file outside the existing capture inputs")
    clocks = {} if clocks is None else clocks
    if not isinstance(clocks, dict):
        raise ValueError("clocks must map source names to clock objects")
    basenames = Counter(path.name for path in paths)
    ranks = Counter(meta.get("rank") for _, meta, _, _ in sources)
    used = set()
    prepared = []
    for path, meta, clock, end_clock in sources:
        header, prefix = _header(path)
        candidates = [
            meta["file"],
            str(path.resolve()),
            meta.get("rank") if ranks[meta.get("rank")] == 1 else None,
            meta.get("node"),
        ]
        if basenames[path.name] == 1:
            candidates.append(path.name)
        override = next((k for k in candidates if k is not None and k in clocks), None)
        merged = dict(clock)
        if override is not None:
            if not isinstance(clocks[override], dict):
                raise ValueError(f"expected a clock object for {override}")
            merged.update(clocks[override])
            used.add(override)
        origin = merged.get("origin_ns", header.get("baseTimeNanoseconds"))
        offset = merged.get("offset_ns", 0)
        if type(offset) is not int or (origin is not None and type(origin) is not int):
            raise ValueError(
                f"{path.name}: origin_ns and offset_ns must be integer nanoseconds"
            )
        if origin is None and (len(sources) > 1 or offset):
            raise ValueError(
                f"{path.name}: missing Kineto time origin; set origin_ns in --clocks"
            )
        time_basis = "source-relative time"
        if origin is not None:
            time_basis = "recorded wall clock; host correction unmeasured"
        if clock and origin is not None:
            time_basis = "capture clock"
        if override is not None and origin is not None:
            time_basis = "explicit clock"
        provenance = {
            "source": meta,
            "clock": merged,
            "end_clock": end_clock,
            "time_basis": time_basis,
        }
        prepared.append((path, prefix, header, provenance, (origin or 0) + offset))
    if set(clocks) - used:
        raise ValueError(
            "clock entries did not match a source: "
            + ", ".join(sorted(set(clocks) - used))
        )
    # A single standalone source already is a full trace. Preserve its JSON
    # bytes (including vendor-specific fields and formatting) without a rewrite.
    if (
        len(prepared) == 1
        and not clocks
        and isinstance(source, (str, os.PathLike))
        and Path(source).is_file()
    ):
        path, prefix, *_ = prepared[0]
        with _open(path) as stream:
            events = sum(1 for _ in ijson.items(stream, prefix))
        with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False) as f:
            temp = Path(f.name)
            try:
                with _open(path) as stream:
                    shutil.copyfileobj(stream, f, length=1024 * 1024)
                f.close()
                os.replace(temp, dst)
            finally:
                temp.unlink(missing_ok=True)
        return {"read": events, "kept": events, "sources": 1}
    origin = min(item[-1] for item in prepared)
    serial = count(1)
    frames = {}
    samples = []
    sources_out = []
    events = 0
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf8", dir=dst.parent, delete=False
    ) as f:
        temp = Path(f.name)
        try:
            f.write('{"traceEvents":[')
            for path, prefix, header, provenance, epoch in prepared:
                shift = Decimal(epoch - origin) / 1000
                tracks = _Tracks(serial)
                n = 0
                low = high = None
                with _open(path) as stream:
                    for event in ijson.items(stream, prefix):
                        if events:
                            f.write(",\n")
                        row = tracks.event(event, shift)
                        f.write(_json(row))
                        if "ts" in row and row.get("ph") != "M":
                            start = row["ts"]
                            end = start + row.get("dur", 0)
                            low = min(start, low) if low is not None else start
                            high = max(end, high) if high is not None else end
                        n += 1
                        events += 1
                for key, frame in header.pop("stackFrames", {}).items():
                    if "parent" in frame:
                        frame["parent"] = tracks.frame(frame["parent"])
                    frames[tracks.frame(key)] = frame
                for sample in header.pop("samples", []):
                    samples.append(tracks.event(sample, shift))
                provenance.update(
                    header=header,
                    shift_us=shift,
                    events=n,
                    range_us=[low, high],
                    processes=[[k, v] for k, v in tracks.processes.values.items()],
                    threads=[[p, t, v] for (p, t), v in tracks.threads.values.items()],
                    identifiers=list(tracks.ids.values.items()),
                    frame_keys=list(tracks.frames.values.items()),
                )
                sources_out.append(provenance)
            remainder = {
                "stackFrames": frames,
                "samples": samples,
                "serveutilFullTrace": {"version": 1, "sources": sources_out},
            }
            if any(s["time_basis"] != "source-relative time" for s in sources_out):
                remainder["baseTimeNanoseconds"] = origin
            # Present consistent header fields in their native place as well as
            # storing all original headers under their individual sources.
            for key, value in prepared[0][2].items():
                if key not in remainder and all(
                    s[2].get(key) == value and key in s[2] for s in prepared
                ):
                    remainder[key] = value
            remainder["serveutilFullTrace"]["source_specific_header_fields"] = sorted(
                set().union(*(s["header"] for s in sources_out)) - remainder.keys()
            )
            f.write("],")
            f.write(_json(remainder)[1:])
            f.flush()
            f.close()
            os.replace(temp, dst)
        finally:
            temp.unlink(missing_ok=True)
    return {"read": events, "kept": events, "sources": len(prepared)}
