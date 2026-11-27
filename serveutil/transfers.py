"""Read SGLang PD transfers from OpenTelemetry exports.

Accept an OTLP JSON export, line-delimited exports, or a descriptor such as
{"spans":"spans.jsonl","histograms":"isolated-requests.jsonl"}. Paths are
relative to the descriptor. Standard OTLP resource attributes carry host.name.
When absent, an optional service_nodes object maps service.name to a known node.
Per-request size is optional. A histogram observation needs the trace ID sent
by the client, its response ID, request timestamps and before/after values
of SGLang's KV transfer histogram on the source rank.
"""

import gzip
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

STAGES = {"prefill_transfer_kv_cache", "decode_transferred", "decode_forward"}


def _read(path):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        if path.name.removesuffix(".gz").endswith((".jsonl", ".ndjson")):
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        else:
            yield json.load(stream)


def _attributes(pairs):
    result = {}
    for pair in pairs:
        value = pair.get("value", {})
        for typ in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if typ in value:
                result[pair["key"]] = value[typ]
                break
    return result


def _expand(value):
    if isinstance(value, list):
        for item in value:
            yield from _expand(item)
    elif "resourceSpans" in value:
        for resource in value["resourceSpans"]:
            attrs = _attributes(resource.get("resource", {}).get("attributes", []))
            for scope in resource.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    yield dict(
                        trace_id=span["traceId"],
                        span_id=span["spanId"],
                        parent_id=span.get("parentSpanId", ""),
                        name=span["name"],
                        start_ns=int(span["startTimeUnixNano"]),
                        end_ns=int(span["endTimeUnixNano"]),
                        node=attrs.get("host.name"),
                        service=attrs.get("service.name"),
                        attrs=_attributes(span.get("attributes", [])),
                    )
    elif "trace_id" in value and "span_id" in value:
        yield value
    else:
        raise ValueError("expected an OTLP resourceSpans export or span record")


def _hex_id(value, length):
    value = str(value).lower()
    if re.fullmatch(f"[0-9a-f]{{{length}}}", value) and int(value, 16):
        return value
    return None


def _request(span, by_id):
    """Find an observed bootstrap room on this span's own ancestor chain."""
    seen = set()
    while span and span["span_id"] not in seen:
        seen.add(span["span_id"])
        attrs = span.get("attrs", {})
        if attrs.get("bootstrap_room") is not None:
            try:
                text = str(attrs["bootstrap_room"]).lower()
                room = int(text, 16 if text.startswith("0x") else 10)
            except ValueError:
                return None
            if room <= 0:  # SGLang uses room zero for disaggregation warmup.
                return None
            return room, attrs.get("rid") or attrs.get("gen_ai.request.id")
        span = by_id.get(span.get("parent_id"))
    return None


def _unique_node(spans):
    nodes = {s["node"] for s in spans if s.get("node")}
    return next(iter(nodes)) if len(nodes) == 1 else None


def _transfers(trace_id, spans):
    by_id = {}
    for span in spans:
        span_id = _hex_id(span["span_id"], 16)
        if not span_id:
            continue
        item = {
            **span,
            "span_id": span_id,
            "parent_id": str(span.get("parent_id", "")).lower(),
        }
        if span_id in by_id and by_id[span_id] != item:
            raise ValueError(f"conflicting OTLP records for span {span_id}")
        by_id[span_id] = item
    rooms = defaultdict(lambda: defaultdict(list))
    for span in by_id.values():
        if span["name"] not in STAGES:
            continue
        request = _request(span, by_id)
        if request and 0 < int(span["start_ns"]) <= int(span["end_ns"]):
            room, rid = request
            rooms[room][span["name"]].append({**span, "rid": rid})
    for room, stages in rooms.items():
        sends = stages["prefill_transfer_kv_cache"]
        if not sends:
            continue
        receives, decodes = stages["decode_transferred"], stages["decode_forward"]
        yield dict(
            id=f"{trace_id}/{room}",
            trace_id=trace_id,
            bootstrap_room=str(room),
            logical_bytes=None,
            send_start_ns=min(int(s["start_ns"]) for s in sends),
            send_end_ns=max(int(s["end_ns"]) for s in sends),
            receive_end_ns=max((int(s["end_ns"]) for s in receives), default=None),
            decode_start_ns=min((int(s["start_ns"]) for s in decodes), default=None),
            source_node=_unique_node(sends),
            destination_node=_unique_node(receives + decodes),
            prefill_request_ids=sorted({s["rid"] for s in sends if s["rid"]}),
            decode_request_ids=sorted(
                {s["rid"] for s in receives + decodes if s["rid"]}
            ),
            source="SGLang OpenTelemetry",
            notes=["Scheduler-observed KV transfer stages."],
        )


def _counter_delta(before, after, field):
    count, total = f"sglang:{field}_count", f"sglang:{field}_sum"
    if (count in before) != (total in before):
        return None
    if Decimal(str(after[count])) - Decimal(str(before.get(count, 0))) != 1:
        return None
    value = Decimal(str(after[total])) - Decimal(str(before.get(total, 0)))
    return value if value.is_finite() and value >= 0 else None


def _measurement(row, observation):
    if (
        not observation.get("response_id")
        or observation["response_id"] not in row["decode_request_ids"]
    ):
        return None
    try:
        if (
            not int(observation["sent_ns"])
            <= row["send_start_ns"]
            <= row["send_end_ns"]
            <= int(observation["end_ns"])
        ):
            return None
        before, after = observation["before"], observation["after"]
        mib = _counter_delta(before, after, "kv_transfer_total_mb")
        if mib is None:
            return None
        byte_delta = mib * (1024**2)
        if byte_delta != byte_delta.to_integral_value():
            return None
        result = {"logical_bytes": int(byte_delta)}
        if "sglang:kv_transfer_latency_ms_count" in after:
            elapsed_ms = _counter_delta(before, after, "kv_transfer_latency_ms")
            if elapsed_ms:
                result["engine_transfer_ms"] = float(elapsed_ms)
                result["effective_gb_s"] = float(byte_delta / elapsed_ms / 1_000_000)
        return result
    except (KeyError, ValueError, InvalidOperation, TypeError):
        return None


def load_transfers(path):
    path = Path(path)
    by_trace = defaultdict(list)
    histograms = defaultdict(list)
    for record in _read(path):
        if isinstance(record, dict) and isinstance(record.get("spans"), str):
            service_nodes = record.get("service_nodes", {})
            for item in _read(path.parent / record["spans"]):
                for s in _expand(item):
                    if not s.get("node") and s.get("service") in service_nodes:
                        s = {**s, "node": service_nodes[s["service"]]}
                    by_trace[_hex_id(s["trace_id"], 32)].append(s)
            if record.get("histograms"):
                for item in _read(path.parent / record["histograms"]):
                    for obs in item if isinstance(item, list) else [item]:
                        histograms[_hex_id(obs.get("sent_trace_id"), 32)].append(obs)
        else:
            for span in _expand(record):
                by_trace[_hex_id(span["trace_id"], 32)].append(span)
    rows = []
    for trace_id, spans in by_trace.items():
        if not trace_id:
            continue
        transfers = list(_transfers(trace_id, spans))
        measured = defaultdict(list)
        for obs in histograms[trace_id]:
            matches = []
            for row in transfers:
                measurement = _measurement(row, obs)
                if measurement is not None:
                    matches.append((row["id"], measurement))
            if len(matches) == 1:
                row_id, measurement = matches[0]
                measured[row_id].append(measurement)
        for row in transfers:
            observations = measured[row["id"]]
            if len(observations) == 1:
                row.update(observations[0])
                row["notes"].append(
                    "Prefill histogram increased by one observation in the matching isolated request."
                )
            rows.append(row)
    return sorted(rows, key=lambda row: row["send_start_ns"])


_NIXL_FIELDS = {
    "bytes_transferred_count",
    "bytes_transferred_sum",
    "bytes_transferred_created",
    "xfer_time_seconds_count",
    "xfer_time_seconds_sum",
    "xfer_time_seconds_created",
    "num_failed_transfers_total",
    "num_failed_transfers_created",
    "num_failed_notifications_total",
    "num_failed_notifications_created",
}


def _nixl_groups(text):
    from prometheus_client.parser import text_string_to_metric_families

    # Drop unrelated histograms (and their large bucket arrays) before parsing.
    # The upstream parser handles escaped label values and numeric syntax.
    prefix = "vllm:nixl_"
    relevant = (
        line
        for line in text.splitlines()
        if line.startswith(prefix)
        and re.split(r"[{\s]", line[len(prefix) :], maxsplit=1)[0] in _NIXL_FIELDS
    )
    groups = defaultdict(dict)
    for family in text_string_to_metric_families("\n".join(relevant)):
        for sample in family.samples:
            field = sample.name[len(prefix) :]
            if field in _NIXL_FIELDS:
                key = tuple(sorted(sample.labels.items()))
                value = Decimal(str(sample.value))
                if not value.is_finite() or value < 0 or field in groups[key]:
                    raise ValueError("invalid or duplicate NIXL metric sample")
                groups[key][field] = str(value)
    return [
        {"labels": dict(labels), "values": values} for labels, values in groups.items()
    ]


def snapshot_nixl(target):
    """Read the endpoint's existing NIXL histograms, without changing the service."""
    import os
    import time
    from http.client import HTTPException
    from urllib.parse import urlsplit
    from urllib.request import Request, urlopen

    url = target["url"].rstrip("/")
    location = urlsplit(url)
    endpoint = (location.hostname or "") + (
        f":{location.port}" if location.port else ""
    )
    result = {
        "endpoint": endpoint,
        "role": target.get("role"),
        "groups": [],
        "sent_ns": time.time_ns(),
    }
    try:
        headers = {"Accept": "text/plain; version=0.0.4"}
        key = os.getenv(target.get("key_env", "VLLM_API_KEY"))
        if key:
            headers["Authorization"] = "Bearer " + key
        with urlopen(
            Request(url + "/metrics", headers=headers),
            timeout=target.get("metrics_timeout", 3),
        ) as response:
            text = response.read(16 * 1024 * 1024 + 1)
        if len(text) > 16 * 1024 * 1024:
            raise ValueError("metrics response exceeds 16 MiB")
        result["groups"] = _nixl_groups(text.decode())
        if not result["groups"]:
            result["error"] = "Endpoint did not expose NixlConnector histograms"
    except (OSError, HTTPException, ImportError, ValueError, InvalidOperation) as exc:
        result["error"] = f"NIXL metrics unavailable ({type(exc).__name__})"
    result["received_ns"] = time.time_ns()
    return result


def _nixl_delta_fields(left, right, interval_ns):
    values = {}
    for key in left.keys() & right.keys():
        a, b = Decimal(left[key]), Decimal(right[key])
        if key.endswith("_created") and a != b:
            raise ValueError("NIXL metric series restarted")
        if b < a:
            raise ValueError("NIXL metric series reset")
        values[key] = b - a
    count = values.get("bytes_transferred_count")
    size = values.get("bytes_transferred_sum")
    if count is None or size is None:
        raise ValueError("incomplete NIXL byte histogram")
    if int(count) != count or int(size) != size or (count == 0 and size):
        raise ValueError("inconsistent NIXL byte histogram")
    seconds = values.get("xfer_time_seconds_sum")
    rate = None
    if seconds and count > 0 and values.get("xfer_time_seconds_count") == count:
        rate = float(size / seconds / 1_000_000_000)
    return {
        "status": "ok",
        "transfers": int(count),
        "logical_bytes": int(size),
        "interval_gb_s": float(size / interval_ns),
        "active_gb_s": rate,
        "failures": int(values["num_failed_transfers_total"])
        if "num_failed_transfers_total" in values
        else None,
        "notification_failures": int(values["num_failed_notifications_total"])
        if "num_failed_notifications_total" in values
        else None,
    }


def nixl_deltas(before, after):
    """Difference equal-label NIXL counters, retaining endpoint/rank scope.

    interval_gb_s uses elapsed wall time between scrape midpoints. active_gb_s
    uses the SUM of NIXL-reported successful transfer durations and therefore
    also works for overlapping transfers; it is a per-transfer weighted rate.
    """
    start = (before["sent_ns"] + before["received_ns"]) // 2
    end = (after["sent_ns"] + after["received_ns"]) // 2
    base = {
        "endpoint": before["endpoint"],
        "role": before["role"],
        "source": "vLLM NixlConnector",
        "start_ns": start,
        "end_ns": end,
        "interval_ns": end - start,
    }
    error = before.get("error") or after.get("error")
    if (
        error
        or end <= start
        or before["endpoint"] != after["endpoint"]
        or before["role"] != after["role"]
    ):
        return [
            {**base, "status": "unavailable", "reason": error or "unmatched snapshots"}
        ]
    left, right = [
        {
            tuple(sorted(group["labels"].items())): group["values"]
            for group in snapshot["groups"]
        }
        for snapshot in (before, after)
    ]
    result = []
    for key in sorted(left.keys() | right.keys()):
        row = {**base, "labels": dict(key)}
        if key not in left or key not in right:
            row.update(
                status="unavailable", reason="metric labels changed between scrapes"
            )
        else:
            try:
                row.update(_nixl_delta_fields(left[key], right[key], end - start))
            except ValueError as exc:
                row.update(status="unavailable", reason=str(exc))
        result.append(row)
    return result
