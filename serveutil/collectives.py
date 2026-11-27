"""Derive per-rank NCCL operation rates from Kineto GPU kernel metadata."""

import json
import math
import re
from bisect import bisect_right
from collections import defaultdict

_OPERATIONS = {
    "allreduce": "all_reduce",
    "allgather": "all_gather",
    "allgatherbase": "all_gather",
    "allgatherintotensor": "all_gather",
    "reducescatter": "reduce_scatter",
    "reducescatterbase": "reduce_scatter",
    "reducescattertensor": "reduce_scatter",
    "alltoall": "all_to_all",
    "alltoallbase": "all_to_all",
    "broadcast": "broadcast",
    "reduce": "reduce",
    "send": "send",
    "recv": "recv",
}
_TYPES = {
    alias: (names[0], size)
    for size, names in (
        (1, ("int8", "char")),
        (1, ("uint8", "byte")),
        (1, ("bool",)),
        (2, ("int16", "short")),
        (2, ("uint16",)),
        (2, ("bfloat16", "bf16")),
        (2, ("float16", "half", "fp16")),
        (4, ("int32", "int")),
        (4, ("uint32",)),
        (4, ("float32", "float", "fp32")),
        (8, ("int64", "long")),
        (8, ("uint64",)),
        (8, ("float64", "double", "fp64")),
        (8, ("complex64", "complexfloat")),
        (16, ("complex128", "complexdouble")),
    )
    for alias in names
}


def _integer(value):
    if type(value) is int and value >= 0:
        return value
    if (
        isinstance(value, str)
        and len(value) <= 20
        and value.isascii()
        and value.isdigit()
    ):
        return int(value)
    return None


def _numbers(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, list) and all(_integer(item) is not None for item in value):
        return tuple(map(int, value))
    return None


def _type(value):
    return _TYPES.get(str(value).lower().removeprefix("torch."))


def _geometry(op, args, ranks):
    incoming = _integer(args.get("In msg nelems"))
    outgoing = _integer(args.get("Out msg nelems"))
    if ("In msg nelems" in args and incoming is None) or (
        "Out msg nelems" in args and outgoing is None
    ):
        return None, None
    if op == "send":
        return (incoming, None) if outgoing in (None, 0, incoming) else (None, None)
    if op == "recv":
        return (outgoing, None) if incoming in (None, 0, outgoing) else (None, None)
    if incoming is None or outgoing is None:
        return None, None
    fraction = (ranks - 1) / ranks
    if op == "all_gather":
        return (outgoing, fraction) if incoming * ranks == outgoing else (None, None)
    if op == "reduce_scatter":
        return (incoming, fraction) if outgoing * ranks == incoming else (None, None)
    if op == "all_to_all":
        # Uniform counts can be inferred from explicit equal splits or the
        # equal-size convention of all_to_all_single (both split lists empty).
        lists = [_numbers(args.get(key)) for key in ("In split size", "Out split size")]
        uniform = incoming % ranks == 0 and all(
            splits == ()
            or (
                splits is not None
                and len(splits) == ranks
                and len(set(splits)) == 1
                and splits[0] > 0
                and incoming % (ranks * splits[0]) == 0
            )
            for splits in lists
        )
        if incoming != outgoing:
            return None, None
        return incoming, fraction if uniform else None
    if incoming != outgoing:
        return None, None
    return incoming, 2 * fraction if op == "all_reduce" else 1.0


class CollectiveRecorder:
    """Group kernels within one rank trace before counting their logical payload."""

    def __init__(self):
        self.groups = defaultdict(list)
        self.inplace_calls = defaultdict(list)
        self.markers = []

    def observe_operator(self, event, name, args, thread):
        if name == "c10d::allreduce_":
            self.inplace_calls[thread].append((event["ts"], event["ts"] + event["dur"]))
        elif (
            name == "record_param_comms" and args.get("Collective name") == "allreduce"
        ):
            external = args.get("External id")
            if external is not None:
                identity = (
                    str(args.get("Process Group Name")),
                    _integer(args.get("Seq")),
                    external,
                )
                self.markers.append(
                    (thread, event["ts"], event["ts"] + event["dur"], identity)
                )

    def observe(self, event, args):
        name = str(args.get("Collective name", "")).lower().replace("_", "")
        op = _OPERATIONS.get(name)
        if op is None:
            return
        group_id = args.get("Process Group Name")
        size = _integer(args.get("Group size"))
        members = _numbers(args.get("Process Group Ranks"))
        group = {"id": str(group_id) if group_id is not None else None, "size": size}
        if members is not None:
            group["ranks"] = list(members)
        description = args.get("Process Group Description")
        if description and description != "undefined":
            group["description"] = str(description)
        seq = _integer(args.get("Seq"))
        details = event["collective"] = {"op": op, "group": group, "seq": seq}
        comms_id = _integer(args.get("Comms Id"))
        if comms_id is not None and comms_id != 0:
            details["comms_id"] = str(comms_id)
        # Generic kernels carry no reduction in their symbol. Only accept an
        # explicit specialization for the operation recorded by Kineto.
        symbol = event.get("original", event.get("name", ""))
        reduction = re.search(
            r"\bnccl(?:DevKernel|Kernel)_(AllReduce|ReduceScatter|Reduce)_(Sum|Prod|Min|Max|Avg)_",
            symbol,
        )
        if reduction and _OPERATIONS[reduction[1].lower()] == op:
            details["reduction"] = reduction[2].lower()
        if "Root Rank" in args and _integer(args["Root Rank"]) is not None:
            details["root_rank"] = _integer(args["Root Rank"])
        peer = None
        if op in ("send", "recv"):
            details["direction"] = op
            peer = _integer(args.get("Dst Rank" if op == "send" else "Src Rank"))
            if peer is not None:
                details["peer_rank"] = peer
        dtype = _type(args.get("dtype"))
        if dtype:
            details["dtype"] = dtype[0]
        if (
            not isinstance(group_id, (str, int))
            or group["id"] in ("", "undefined")
            or seq is None
        ):
            details["unavailable"] = "Missing process group or sequence number"
            return
        valid = size is not None and size > 1
        if "Process Group Ranks" in args:
            valid = (
                valid
                and members is not None
                and len(set(members)) == len(members) == size
            )
        elements, factor = _geometry(op, args, size) if valid else (None, None)
        byte_count = elements * dtype[1] if dtype and elements is not None else None
        key = (event["device"], group["id"], seq, op, peer)
        signature = (
            size,
            members,
            dtype,
            byte_count,
            factor,
            args.get("Comms Id"),
            description,
        )
        self.groups[key].append((event, signature, args.get("External id")))

    def finish(self):
        inplace = set()
        for calls in self.inplace_calls.values():
            calls.sort()
        for thread, start, end, identity in self.markers:
            calls = self.inplace_calls[thread]
            i = bisect_right(calls, (start, math.inf)) - 1
            if i >= 0 and calls[i][1] >= end:
                inplace.add(identity)
        for kernels in self.groups.values():
            events = [event for event, _, _ in kernels]
            signature = kernels[0][1]
            if any(item != signature for _, item, _ in kernels) or signature[3] is None:
                for event in events:
                    event["collective"]["unavailable"] = (
                        "Incomplete or inconsistent NCCL metadata"
                    )
                continue
            first = min(events, key=lambda e: e["ts"])
            span = max(e["ts"] - first["ts"] + e["dur"] for e in events)
            if not math.isfinite(span) or span <= 0:
                continue
            info = first["collective"]
            if info["op"] == "all_reduce" and all(
                (info["group"]["id"], info["seq"], external) in inplace
                for _, _, external in kernels
            ):
                info["placement"] = "in_place"
            byte_count, factor = signature[3:5]
            rate = byte_count / (1000 * span)
            info.update(
                bytes=byte_count,
                duration_us=span,
                kernel_count=len(events),
                algorithm_GB_s=rate,
            )
            if factor is not None:
                info.update(bus_GB_s=rate * factor, bus_factor=factor)
            for event in events:
                if event is not first:
                    event["collective"]["part_of_operation"] = True


def _version(value):
    parsed = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", str(value))
    if parsed:
        major, minor, patch = map(int, parsed.groups())
        return (
            major * (1000 if (major, minor) < (2, 9) else 10000) + minor * 100 + patch
        )
    return _integer(value)


def match_nccl_reference(
    operation,
    benchmarks,
    *,
    devices=None,
    placement=None,
    reduction=None,
    root_rank=None,
    nccl_version=None,
):
    """Compare a rank span with an nccl-tests run for an explicitly identified group.

    The caller supplies the verified device UUIDs in rank order, buffer placement,
    and (for reductions) operator. Kineto commonly omits these details.
    """
    op = operation.get("op")
    group_size = operation.get("group", {}).get("size")
    placement = operation.get("placement", placement)
    reduction = operation.get("reduction", reduction)
    root_rank = operation.get("root_rank", root_rank)
    if (
        op in ("send", "recv")
        or "algorithm_GB_s" not in operation
        or not isinstance(devices, (list, tuple))
        or any(not isinstance(uuid, str) or not uuid for uuid in devices)
        or len(devices) != group_size
        or len(set(devices)) != group_size
        or placement not in ("in_place", "out_of_place")
        or (op in ("all_reduce", "reduce_scatter", "reduce") and not reduction)
        or (op in ("broadcast", "reduce") and root_rank is None)
        or (op == "all_to_all" and "bus_factor" not in operation)
    ):
        return None
    matches = []
    for row in benchmarks:
        params = row.get("parameters", {})
        dtype = _type(params.get("dtype"))
        if (
            row.get("source") == "nccl-tests"
            and row.get("unit") == "GB/s"
            and row.get("metric") == f"nccl.{op}.alg_bw"
            and params.get("collective") == op
            and params.get("ranks") == group_size
            and (
                nccl_version is None
                or params.get("version") is None
                or (
                    _version(nccl_version) is not None
                    and _version(nccl_version) == _version(params["version"])
                )
            )
            and params.get("devices") == list(devices)
            and params.get("placement") == placement
            and params.get("bytes") == operation.get("bytes")
            and dtype
            and dtype[0] == operation.get("dtype")
            and (
                op not in ("all_reduce", "reduce_scatter", "reduce")
                or params.get("reduction") == reduction
            )
            and (
                op not in ("broadcast", "reduce")
                or params.get("root_rank") == root_rank
            )
        ):
            rate = row.get("value")
            if isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0:
                matches.append(row)
    if len(matches) == 1:
        reference = matches[0]
        return {
            "reference_GB_s": reference["value"],
            "percent": 100 * operation["algorithm_GB_s"] / reference["value"],
            "source": reference["source"],
            "parameters": reference["parameters"],
        }
    return None
