"""Interval arithmetic for finding GPU timeline regions worth inspecting."""

from bisect import bisect_right


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def subtract(intervals, occupied):
    """Both inputs are sorted, disjoint half-open intervals."""
    ends = [e for _, e in occupied]
    for start, end in intervals:
        cursor = start
        index = bisect_right(ends, start)
        while index < len(occupied) and occupied[index][0] < end:
            left, right = occupied[index]
            if left > cursor:
                yield cursor, left
            cursor = max(cursor, right)
            index += 1
        if cursor < end:
            yield cursor, end


def findings(events, rank):
    gpu = [e for e in events if e["kind"] in {"compute", "communication", "copy"}]
    result = []
    devices = {e["device"] for e in gpu}
    for device in devices:
        groups = {}
        for event in gpu:
            if event["device"] == device:
                groups.setdefault(event["kind"], []).append(
                    (event["ts"], event["ts"] + event["dur"])
                )
        busy = union(span for spans in groups.values() for span in spans)
        gaps = [
            (prev[1], nxt[0])
            for prev, nxt in zip(busy, busy[1:])
            if nxt[0] - prev[1] >= 100
        ]
        observations = []
        if len(gaps) >= 2:
            observations.extend(
                (
                    "Repeated GPU gap",
                    start,
                    end,
                    "No recorded GPU work on this device between adjacent activities. Compare host launches and synchronization.",
                )
                for start, end in gaps
            )
        for start, end in subtract(
            union(groups.get("communication", [])), union(groups.get("compute", []))
        ):
            if end - start >= 100:
                observations.append(
                    (
                        "Exposed communication",
                        start,
                        end,
                        "Collective activity with no overlapping recorded compute on this device. Inspect dependencies and message size.",
                    )
                )
        for label, start, end, reason in observations:
            result.append(
                dict(
                    name=label,
                    ts=start,
                    dur=end - start,
                    device=device,
                    rank=rank,
                    reason=reason,
                )
            )
    return sorted(result, key=lambda item: -item["dur"])[:100]


def matmul_work(name, args):
    """Fused multiply-add counts twice. Only match explicit dense GEMM shapes."""
    shapes = args.get("Input Dims") or []
    types = args.get("Input type") or []
    if name not in {
        "aten::mm",
        "aten::bmm",
        "aten::addmm",
        "aten::_int_mm",
        "aten::_scaled_mm",
    }:
        return None
    index = 1 if name == "aten::addmm" else 0
    try:
        left, right = shapes[index : index + 2]
        dtype = str(types[index]).split("::")[-1].lower()
        right_type = str(types[index + 1]).split("::")[-1].lower()
        if right_type != dtype:
            return None
        dtype = {"signed char": "int8", "unsigned char": "uint8", "char": "int8"}.get(
            dtype, dtype
        )
        floating = {
            "half",
            "float",
            "double",
            "float16",
            "float32",
            "float64",
            "bfloat16",
        }
        integer = dtype in {
            "char",
            "byte",
            "short",
            "int",
            "long",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
        }
        if not (integer or dtype in floating or dtype.startswith("float8_")):
            return None
        if (
            len(left) != len(right)
            or len(left) not in (2, 3)
            or left[:-2] != right[:-2]
        ):
            return None
        if left[-1] != right[-2] or min(*left, *right) <= 0:
            return None
        batch = left[0] if len(left) == 3 else 1
        count = 2 * batch * left[-2] * left[-1] * right[-1]
        return dict(
            **({"operations": count} if integer else {"flops": count}),
            rate_unit="TOP/s" if integer else "TFLOP/s",
            dtype=dtype,
            shape=[left, right],
            method="Dense GEMM input shapes; one multiply-add counts as two operations",
        )
    except (IndexError, TypeError, ValueError):
        return None


def saturation(series):
    """Flag sustained device-counter saturation against a measured, comparable baseline."""
    result = []
    for s in series:
        ref = s.get("baseline")
        if not ref or s["unit"] != "GB/s" or ref["value"] <= 0:
            continue
        run = []

        def flush():
            if len(run) < 3:
                return
            minimum = min(p[1] for p in run)
            result.append(
                dict(
                    name="Sustained bandwidth pressure",
                    ts=run[0][2],
                    dur=run[-1][0] - run[0][2],
                    device=s["device"],
                    rank=s.get("node") or "local",
                    node=s.get("node"),
                    metric=s["name"],
                    source=s["source"],
                    reason=f"{len(run)} consecutive samples at or above 90% of the measured reference.",
                    minimum_GB_s=minimum,
                    reference_GB_s=ref["value"],
                    scope=s["scope"],
                )
            )

        for point in s["points"]:
            if run and point[2] - run[-1][0] > 0.5 * (point[0] - point[2]):
                flush()
                run.clear()
            if point[1] >= 0.9 * ref["value"]:
                run.append(point)
            else:
                flush()
                run.clear()
        flush()
    return result


def rank_skew(traces):
    """Compare only uniquely identified copies of a logical step in the same role."""
    import re
    from collections import defaultdict
    from statistics import median

    groups = defaultdict(lambda: defaultdict(list))
    for trace in traces:
        for e in trace["events"]:
            if e["kind"] != "stage":
                continue
            args = e.get("args", {})
            step = args.get("step_id")
            if step is None:
                step = args.get("iteration_id")
            if step is None and re.fullmatch(r"ProfilerStep#\d+", e["name"]):
                step = e["name"]
            if step is not None:
                key = trace.get("role"), e.get("domain"), str(step), e["name"]
                groups[key][trace["rank"]].append(e)
    for (_, domain, step, _), by_rank in groups.items():
        if len(by_rank) < 2 or any(len(vals) != 1 for vals in by_rank.values()):
            continue
        for rank, (event,) in by_rank.items():
            typical = median(vals[0]["dur"] for r, vals in by_rank.items() if r != rank)
            if event["dur"] - typical >= 100 and event["dur"] >= 1.5 * typical:
                yield dict(
                    name="Longer rank step",
                    ts=event["ts"],
                    dur=event["dur"],
                    rank=rank,
                    device=event["device"],
                    step=step,
                    domain=domain,
                    peer_median_us=typical,
                    reason="This identified step took at least 1.5× the peer median and 100 µs longer.",
                )
