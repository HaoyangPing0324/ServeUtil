"""Self-contained original attachments for the full trace workspace."""

import base64
import gzip
import json
import re

import ijson


def from_html(html):
    match = re.search(
        r'<script id="raw-sources" type="application/json">([\s\S]*?)</script>', html
    )
    if not match:
        return []
    records = json.loads(match[1])
    if not isinstance(records, list):
        raise ValueError("Invalid raw trace attachment list")
    return records


def _record(path, title, rank=None, node=None, kind="trace"):
    raw = path.read_bytes()
    already_gz = raw.startswith(b"\x1f\x8b")
    payload = raw if already_gz else gzip.compress(raw, compresslevel=1, mtime=0)
    return dict(
        name=path.name,
        title=title,
        rank=rank,
        node=node,
        kind=kind,
        compress_attachment=not already_gz,
        payload=base64.b64encode(payload).decode("ascii"),
    )


def attachments(path, report=None):
    if path.is_file():
        return [_record(path, path.name)]
    manifest = path / "capture.json"
    if not manifest.exists():
        return []
    m = json.loads(manifest.read_text(encoding="utf-8"))
    result = []
    parsed = {(t["rank"], t["file"]): t for t in (report or {}).get("traces", [])}
    nodes = {n["name"]: n for n in m.get("nodes", [])}
    for i, entry in enumerate(m["traces"]):
        f = (path / entry["file"]).resolve()
        if not f.is_relative_to(path.resolve()):
            raise ValueError("Trace outside capture directory")
        rank = str(entry["rank"] if entry.get("rank") is not None else i)
        item = _record(
            f,
            f"{rank} · {f.name}",
            rank,
            entry.get("node"),
            "auxiliary" if entry.get("auxiliary") else "trace",
        )
        trace = parsed.get((rank, f.name), {})
        clock = nodes.get(entry.get("node"), {}).get("clock") or {}
        has_clock = not entry.get("node") or (
            isinstance(clock.get("offset_ns"), int)
            and isinstance(clock.get("uncertainty_ns"), int)
        )
        if entry.get("auxiliary") and has_clock:
            opener = gzip.open if f.suffix == ".gz" else open
            with opener(f, "rb") as src:
                base = next(ijson.items(src, "baseTimeNanoseconds"), None)
            if base is not None:
                item["epoch"] = str(int(base) + clock.get("offset_ns", 0))
        if (
            not entry.get("auxiliary")
            and has_clock
            and trace.get("source_epoch_ns") is not None
        ):
            item["epoch"] = trace["source_epoch_ns"]
        result.append(item)
    return result
