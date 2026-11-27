"""Build a portable metrics report from native traces and device samples."""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import ijson

from .analysis import findings, matmul_work, rank_skew, saturation
from .bundle import open_text
from .calibration import reference_for
from .collectives import CollectiveRecorder, match_nccl_reference
from .overview import ANNOTATIONS, GPU_EVENTS, kernel_label, trace_items
from .relationships import communication_links

COMM = re.compile(
    r"nccl|all.?reduce|all.?gather|reduce.?scatter|all.?to.?all|sendrecv|cross_device_reduce",
    re.I,
)


def stage_label(name):
    step = re.match(r"step\[(EXTEND|PREFILL|DECODE|MIXED|IDLE)\b", name, re.I)
    if step:
        return {"extend": "Prefill", "mixed": "Mixed batch"}.get(
            step[1].lower(), step[1].capitalize()
        )
    match = re.match(r"execute_context_(\d+)\(\d+\)_generation_(\d+)\(\d+\)", name)
    if match:
        context, generation = map(int, match.groups())
        return (
            "Mixed batch"
            if context and generation
            else "Prefill"
            if context
            else "Decode"
            if generation
            else "Idle"
        )
    return name


def load_trace(path, rank, analysis=None):
    header = {}
    events = []
    counters = defaultdict(list)
    stacks = defaultdict(list)
    gemms = {}
    launches = {}
    linked = defaultdict(list)
    collectives = CollectiveRecorder()
    with open_text(path) as source:
        for raw in trace_items(source.buffer, header):
            phase = raw.get("ph")
            if phase not in {"B", "E", "X", "C"} or "ts" not in raw:
                continue
            ts = float(raw["ts"])
            name = str(raw.get("name", ""))
            args = raw.get("args") or {}
            pid, tid = str(raw.get("pid", "")), str(raw.get("tid", ""))
            if phase == "C":
                for key, value in args.items():
                    try:
                        number = float(value)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if math.isfinite(number):
                        title = name if key in ("", "value") else f"{name} · {key}"
                        counters[(title, pid, tid)].append([ts, number])
                continue
            if phase == "B":
                stacks[pid, tid].append(raw)
                continue
            if phase == "E":
                if not stacks[pid, tid]:
                    continue
                raw = stacks[pid, tid].pop()
                end = ts
                ts = float(raw["ts"])
                duration = end - ts
                name = str(raw.get("name", ""))
                args = raw.get("args") or {}
            else:
                duration = float(raw.get("dur", 0))
            cats = set(raw.get("cat", "").split(","))
            if duration <= 0:
                continue
            if (
                cats & {"cuda_runtime", "cuda_driver"}
                and "correlation" in args
                and "External id" in args
            ):
                launches[args["correlation"]] = args["External id"]
            if cats & ANNOTATIONS:
                kind = "stage"
            elif cats & GPU_EVENTS:
                kind = (
                    "copy"
                    if "kernel" not in cats
                    else (
                        "communication"
                        if args.get("Collective name") or COMM.search(name)
                        else "compute"
                    )
                )
            elif "cpu_op" in cats:
                kind = "operator"
            elif cats & {"cuda_runtime", "cuda_driver"}:
                kind = "runtime"
            elif "python_function" in cats:
                kind = "python"
            else:
                continue
            label = kernel_label(name) if kind in {"communication", "compute"} else name
            device = str(args.get("device", pid)) if cats & (GPU_EVENTS | {"gpu_user_annotation"}) else pid
            event = dict(
                ts=ts, dur=duration, name=label, kind=kind, device=device, lane=tid
            )
            if kind == "stage":
                event["name"] = stage_label(name)
                event["domain"] = "GPU" if "gpu_user_annotation" in cats else "CPU"
            original = args.get("serveutil.original_name", name)
            if original != event["name"]:
                event["original"] = original
            # Keep bounded identifying metadata. Full stacks and tensor payloads remain in the raw file.
            details = {
                k: str(v)[:500]
                for k, v in args.items()
                if k.lower()
                in {
                    "stream",
                    "correlation",
                    "external id",
                    "bytes",
                    "input dims",
                    "step_id",
                    "iteration_id",
                    "input type",
                    "collective name",
                    "dtype",
                    "process group name",
                }
            }
            if details:
                event["args"] = details
            if kind == "copy" and "bytes" in args and "gpu_memcpy" in cats:
                directions = {
                    "htod": "h2d",
                    "dtoh": "d2h",
                    "dtod": "device",
                    "ptop": "peer",
                }
                direction = next(
                    (key for token, key in directions.items() if token in name.lower()),
                    None,
                )
                if direction and isinstance(args["bytes"], int) and args["bytes"] >= 0:
                    event["copy"] = dict(
                        direction=direction,
                        bytes=args["bytes"],
                        GB_s=args["bytes"] / (duration * 1000),
                        method="Payload bytes / GPU copy duration",
                    )
                    for field, keys in {
                        "source_device": (
                            "srcDeviceId",
                            "srcDevice",
                            "src device",
                            "source device",
                        ),
                        "destination_device": (
                            "dstDeviceId",
                            "dstDevice",
                            "dst device",
                            "destination device",
                        ),
                    }.items():
                        ordinal = next((args[k] for k in keys if k in args), None)
                        if ordinal is not None:
                            event["copy"][field] = str(ordinal)
                    copy = event["copy"]
                    if direction == "device":
                        if (
                            "source_device" in copy
                            and "destination_device" in copy
                            and copy["source_device"] != copy["destination_device"]
                        ):
                            copy["direction"] = "peer"
                        else:
                            copy["read_write_GB_s"] = 2 * copy["GB_s"]
            if kind == "operator" and "External id" in args:
                collectives.observe_operator(event, name, args, (pid, tid))
                work = matmul_work(name, args)
                if work is not None:
                    gemms[args["External id"]] = work
            if kind == "compute":
                event["_link"] = args.get("External id"), args.get("correlation")
            if kind == "communication":
                collectives.observe(event, args)
            events.append(event)
    collectives.finish()
    for event in events:
        if "_link" in event:
            external, correlation = event.pop("_link")
            key = external if external in gemms else launches.get(correlation)
            if key in gemms:
                linked[key].append(event)
    for key, kernels in linked.items():
        if len(kernels) == 1:
            event = kernels[0]
            event["gemm"] = {
                **gemms[key],
                gemms[key]["rate_unit"]: gemms[key].get(
                    "operations", gemms[key].get("flops")
                )
                / (event["dur"] * 1e6),
            }
    begin = min(
        [e["ts"] for e in events]
        + [p[0] for series in counters.values() for p in series],
        default=0,
    )
    events.sort(key=lambda e: (e["ts"], -e["dur"]))
    series = [
        dict(
            name=name,
            device=pid,
            lane=tid,
            source="Trace counter",
            points=sorted(points),
        )
        for (name, pid, tid), points in counters.items()
    ]
    upstream = {}
    if analysis:
        from .integrations import analyze_trace

        upstream[analysis] = analyze_trace(path, analysis)
    return dict(
        rank=str(rank),
        file=path.name,
        upstream=upstream,
        base=header.get("baseTimeNanoseconds"),
        nccl_version=header.get("distributedInfo.nccl_version"),
        global_rank=header.get("distributedInfo.rank"),
        begin=begin,
        events=events,
        counters=series,
    )


def inputs(path, analysis=None):
    if path.is_dir():
        manifest = json.loads((path / "capture.json").read_text(encoding="utf-8"))
        traces = []
        for i, trace in enumerate(manifest["traces"]):
            file = (path / trace["file"]).resolve()
            if not file.is_relative_to(path.resolve()):
                raise ValueError("trace outside capture directory")
            rank = trace.get("rank")
            if trace.get("auxiliary"):
                continue
            rank = str(rank if rank is not None else i)
            metadata = dict(
                rank=rank,
                file=file.name,
                node=trace.get("node"),
                role=trace.get("role"),
            )
            if trace.get("pending") or (trace.get("export_error") and not file.is_file()):
                traces.append(dict(**metadata, base=None, begin=0, events=[], counters=[],
                                   parse_error="Native export is pending" if trace.get("pending") else trace["export_error"]))
                continue
            if trace.get("native_output") and not file.name.lower().endswith(
                (".json", ".json.gz")
            ):
                traces.append(
                    dict(
                        **metadata,
                        base=None,
                        begin=0,
                        events=[],
                        counters=[],
                        native_only=True,
                    )
                )
                continue
            try:
                parsed = load_trace(file, rank, analysis)
            except (ijson.JSONError, ValueError) as exc:
                if manifest.get("engine") != "torch":
                    raise
                if isinstance(exc, ijson.JSONError):
                    note = {
                        "parse_error": f"Cannot parse the original Kineto JSON: {exc}"
                    }
                elif trace.get(
                    "native_output"
                ) and "expected a PyTorch Chrome trace object" in str(exc):
                    note = {"native_only": True}
                else:
                    raise
                traces.append(
                    dict(**metadata, base=None, begin=0, events=[], counters=[], **note)
                )
                continue
            has_device_events = any(
                e["kind"] in {"compute", "copy", "communication"}
                for e in parsed["events"]
            )
            if not has_device_events and "CUDA" in trace.get("activities", ()):
                parsed["device_activity_unavailable"] = (
                    "CUDA profiling was requested. This cycle contains no GPU kernels, copies or memsets."
                )
            if not has_device_events and manifest.get("engine") in {"sglang", "vllm"}:
                if manifest.get("status") != "failed":
                    raise ValueError(
                        f"{file.name}: worker trace contains no device activities"
                    )
                parsed["device_activity_unavailable"] = (
                    "The failed capture recovered no device activities on this worker."
                )
            parsed.update(
                node=trace.get("node"),
                role=trace.get("role"),
                source_path=trace["file"],
            )
            traces.append(parsed)
        return manifest, traces
    return {"engine": "Imported trace"}, [load_trace(path, 0, analysis)]


def make_report(path, *, analysis=None):
    path = Path(path)
    if (
        path.is_dir()
        and not (path / "capture.json").exists()
        and (path / "run.json").exists()
    ):
        from .integrations import llmperfscope_report

        return llmperfscope_report(path)
    manifest, traces = inputs(path, analysis)
    if not traces:
        raise ValueError("capture contains no traces")
    # Kineto stores event timestamps in microseconds relative to an integer ns epoch.
    # Subtract whole epochs first; keep unrelated or unanchored captures independent.
    nodes = {n["name"]: n for n in manifest.get("nodes", [])}

    def correction(name):
        return (nodes.get(name, {}).get("clock") or {}).get("offset_ns", 0)

    epochs = [
        int(t["base"]) + correction(t.get("node")) + round(t["begin"] * 1000)
        for t in traces
        if t["base"] is not None
    ]
    origin = min(epochs, default=None)
    common_clock = bool(epochs) and len(epochs) == sum(
        "parse_error" not in t and not t.get("native_only") for t in traces
    )
    trace_nodes = {t.get("node") for t in traces}
    if len(trace_nodes) > 1:
        common_clock = common_clock and all(
            isinstance((nodes.get(n, {}).get("clock") or {}).get("offset_ns"), int)
            and isinstance(
                (nodes.get(n, {}).get("clock") or {}).get("uncertainty_ns"), int
            )
            for n in trace_nodes
        )
    # A native binary has its own clock domain. Node-local samples still use the capture wall clock.
    sample_origin = origin if common_clock else None
    if (
        sample_origin is None
        and all(t.get("native_only") for t in traces)
        and not nodes
    ):
        sample_origin = manifest.get("started_ns")
    opportunities = []
    for trace in traces:
        base = trace.pop("base")
        if base is not None:
            trace["source_epoch_ns"] = str(int(base) + correction(trace.get("node")))
        begin = trace.pop("begin")
        trace["raw_begin"] = begin
        offset = (
            (int(base) + correction(trace.get("node")) - origin) / 1000
            if common_clock and base is not None
            else -begin
        )
        for e in trace["events"]:
            e["ts"] += offset
        trace["rank_origin_us"] = begin + offset
        if base is not None:
            trace["rank_origin_unix_ns"] = str(int(base) + round(begin * 1000))
        for series in trace["counters"]:
            for point in series["points"]:
                point[0] += offset
        opportunities.extend(findings(trace["events"], trace["rank"]))
    telemetry = manifest.get("telemetry", {})
    sources = [(None, "samples.jsonl.gz", "calibration.json")]
    if nodes:
        sources = [
            (name, n["samples"], n.get("calibration")) for name, n in nodes.items()
        ]
        telemetry = {
            name: {
                "clock": n.get("clock"),
                "end_clock": n.get("end_clock"),
                **n["telemetry"],
            }
            for name, n in nodes.items()
        }
    hardware = []
    baselines = {}
    for name, filename, calibration_file in sources:
        if not path.is_dir():
            continue
        sample_path = (path / filename).resolve()
        if not sample_path.is_relative_to(path.resolve()):
            raise ValueError("sample file outside capture directory")
        baseline = {}
        if calibration_file:
            cal_path = (path / calibration_file).resolve()
            if not cal_path.is_relative_to(path.resolve()):
                raise ValueError("calibration outside capture directory")
            if cal_path.is_file():
                baseline = json.loads(cal_path.read_text(encoding="utf-8"))
                baselines[name or "local"] = baseline.get("benchmarks", [])
        if sample_origin is None or not sample_path.is_file():
            continue
        groups = {}
        shift = correction(name) - sample_origin
        with open_text(sample_path) as stream:
            for line in stream:
                sample = json.loads(line)
                key = (
                    sample["device"],
                    sample["metric"],
                    sample["unit"],
                    sample["source"],
                )
                if key not in groups:
                    groups[key] = {
                        k: sample.get(k)
                        for k in ("device", "metric", "unit", "scope", "source")
                    }
                    groups[key].update(
                        node=name,
                        name=sample.get("label", key[1]),
                        points=[],
                        sample_node=sample.get("node"),
                        baseline=reference_for(
                            {**sample, "node": name or sample.get("node")}, baseline
                        ),
                    )
                groups[key]["points"].append(
                    [
                        (int(sample["end_ns"]) + shift) / 1000,
                        sample["value"],
                        (int(sample["start_ns"]) + shift) / 1000,
                    ]
                )
        hardware.extend(groups.values())
    for trace in traces:
        node = trace.get("node")
        refs = baselines.get(node or "local", [])
        meta = nodes[node]["telemetry"] if node in nodes else telemetry
        devices = meta.get("devices", [])
        config = nodes.get(node, manifest)
        local_rank = trace["rank"].rsplit("/", 1)[-1]
        overrides = config.get("rank_device_maps", {}).get(trace.get("role"), {})
        explicit = overrides.get(local_rank, config.get("device_map", {}))
        ordinals = {
            e["device"]
            for e in trace["events"]
            if e["kind"] in {"compute", "communication", "copy"}
        }
        trace["devices"] = (
            sorted({explicit[i] for i in ordinals})
            if ordinals <= explicit.keys()
            else []
        )
        if not trace["devices"] and len(devices) == 1:
            trace["devices"] = [devices[0]["uuid"]]
        collective_groups = (
            nodes.get(node, {}).get("collective_groups", {}).get(trace.get("role"), {})
        )
        nccl_refs = defaultdict(list)
        for ref in refs:
            if ref.get("source") == "nccl-tests":
                params = ref.get("parameters", {})
                nccl_refs[params.get("collective"), params.get("bytes")].append(ref)
        for e in trace["events"]:
            op = e.get("collective")
            if op and "algorithm_GB_s" in op:
                cfg = collective_groups.get(op["group"]["id"], {})
                options = cfg.get("operations", {}).get(op["op"], {})
                reference = match_nccl_reference(
                    op,
                    nccl_refs[op["op"], op["bytes"]],
                    devices=cfg.get("devices"),
                    placement=options.get("placement"),
                    reduction=options.get("reduction"),
                    root_rank=options.get("root_rank"),
                    nccl_version=trace.get("nccl_version"),
                )
                if reference:
                    op["reference_pct"] = reference["percent"]
                    op["reference"] = {
                        "value": reference["reference_GB_s"],
                        "unit": "GB/s",
                        "source": reference["source"],
                        "parameters": reference["parameters"],
                    }
            if "gemm" not in e and "copy" not in e:
                continue
            uuid = explicit.get(str(e["device"]))
            if uuid is None and len(devices) == 1:
                uuid = devices[0]["uuid"]
            if uuid is None:
                continue
            if "gemm" in e:
                value = e["gemm"]
                dtype = {
                    "char": "int8",
                    "half": "float16",
                    "float": "float32",
                    "double": "float64",
                }.get(value["dtype"], value["dtype"])
                unit = value["rate_unit"]
                metric, absolute = "gemm." + dtype, value[unit]
            else:
                value = e["copy"]
                unit = "GB/s"
                metric = "copy." + value["direction"]
                absolute = value.get("read_write_GB_s", value["GB_s"])
                if value["direction"] == "peer":
                    src = explicit.get(value.get("source_device"))
                    dst = explicit.get(value.get("destination_device"))
                    if not src or not dst or src == dst:
                        continue
                    uuid = src
            candidates = [
                r
                for r in refs
                if r.get("metric") == metric
                and r.get("device") == uuid
                and r.get("unit") == unit
                and r.get("value", 0) > 0
            ]
            if metric == "copy.peer":
                candidates = [
                    r
                    for r in candidates
                    if r.get("parameters", {}).get("src") == src
                    and r.get("parameters", {}).get("dst") == dst
                ]
            if len(candidates) == 1:
                ref = candidates[0]
                value["reference"] = {
                    k: ref[k]
                    for k in ("value", "unit", "source", "parameters")
                    if k in ref
                }
                value["reference_pct"] = 100 * absolute / ref["value"]
    transfers = []
    unplaced_transfers = []
    if manifest.get("transfers"):
        transfer_file = (path / manifest["transfers"]).resolve()
        if not transfer_file.is_relative_to(path.resolve()):
            raise ValueError("transfer data outside capture directory")
        periods = [
            (e["ts"], e["ts"] + e["dur"])
            for tr in traces
            for e in tr["events"]
            if e["kind"] == "stage"
        ]
        left = min((v[0] for v in periods), default=None)
        right = max((v[1] for v in periods), default=None)
        for transfer in json.loads(transfer_file.read_text(encoding="utf-8")):
            node = transfer.get("source_node")
            if not common_clock or (len(nodes) > 1 and node not in nodes):
                unplaced_transfers.append(transfer)
                continue
            offset = correction(node or next(iter(nodes), None))
            start = (int(transfer["send_start_ns"]) + offset - origin) / 1000
            end = (int(transfer["send_end_ns"]) + offset - origin) / 1000
            if left is None or end < left or start > right or end <= start:
                continue
            size = transfer.get("logical_bytes")
            row = dict(
                name="KV handoff",
                ts=start,
                dur=end - start,
                scheduler_span_GB_s=size / ((end - start) * 1000)
                if size is not None
                else None,
                **transfer,
            )
            destination = row.get("destination_node") or (
                next(iter(nodes), None) if len(nodes) <= 1 else None
            )
            if destination in nodes or len(nodes) <= 1:
                shift = correction(destination) - origin
                for src, target in [
                    ("receive_end_ns", "receive_us"),
                    ("decode_start_ns", "decode_us"),
                ]:
                    if row.get(src) is not None:
                        row[target] = (int(row[src]) + shift) / 1000
                if (
                    row.get("receive_us") is not None
                    and row.get("decode_us") is not None
                ):
                    row["receiver_to_decode_us"] = row["decode_us"] - row["receive_us"]
            transfers.append(row)
    opportunities.extend(saturation(hardware))
    opportunities.extend(rank_skew(traces))
    timing = {}
    for name, n in nodes.items():
        start = n.get("clock")
        if not start or "uncertainty_ns" not in start:
            continue
        end = n.get("end_clock") or start
        timing[name] = {
            "max_error_us": (
                max(start["uncertainty_ns"], end["uncertainty_ns"])
                + abs(start["offset_ns"] - end["offset_ns"])
            )
            / 1000
        }
    return dict(
        schema="serveutil.report/1",
        time_origin_unix_ns=str(origin) if common_clock else None,
        communication_links=communication_links(traces),
        timing=timing,
        calibration_error=manifest.get("calibration_error"),
        transfers=transfers,
        unplaced_transfers=unplaced_transfers,
        kv_metrics=[
            {
                **row,
                **(
                    {
                        "ts": (row["start_ns"] - origin) / 1000,
                        "dur": row["interval_ns"] / 1000,
                    }
                    if common_clock
                    and row.get("status") == "ok"
                    and "interval_ns" in row
                    else {}
                ),
            }
            for row in manifest.get("kv_metrics", [])
        ],
        engine=manifest.get("engine"),
        status=manifest.get("status"),
        error=manifest.get("error"),
        collection_errors=manifest.get("collection_errors", []),
        export_warnings=manifest.get("export_warnings", []),
        aligned=common_clock or len(traces) == 1,
        alignment_note=None
        if common_clock
        else (
            "A trace or node lacks an absolute clock anchor. Each rank uses its first event as zero. "
            "Absolute-time hardware samples and KV handoffs cannot be placed on these axes."
        ),
        traces=traces,
        hardware=hardware,
        baselines=baselines,
        telemetry=telemetry,
        findings=sorted(opportunities, key=lambda x: -x["dur"]),
    )
