"""ServeUtil console entry point."""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    import ijson

    from .overview import export

    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"export", "report", "capture", "view", "calibrate"}
    if argv and argv[0] not in commands and not argv[0].startswith("-"):
        argv.insert(0, "export")
    p = argparse.ArgumentParser(
        prog="serveutil", description="Capture and explore inference performance."
    )
    sub = p.add_subparsers(dest="action", required=True)
    convert = sub.add_parser("export", help="write a Perfetto-compatible Chrome trace")
    convert.add_argument("trace", type=Path, nargs="+")
    convert.add_argument(
        "--full", action="store_true", help="merge original events from every source"
    )
    convert.add_argument("--clocks", type=Path, help="JSON source clock overrides for --full")
    convert.add_argument("-o", "--output", type=Path, required=True)
    report = sub.add_parser("report", help="generate a self-contained offline report")
    report.add_argument(
        "--analysis",
        choices=["hta", "tracelens"],
        help="include an upstream per-rank activity breakdown",
    )
    report.add_argument("trace", type=Path)
    report.add_argument("-o", "--output", type=Path, required=True)
    cap = sub.add_parser("capture", help="record a running service under load")
    cap.add_argument("--engine", required=True, choices=["vllm", "sglang"])
    cap.add_argument(
        "--generate",
        metavar="PROMPT",
        help="drive real completions throughout the capture",
    )
    cap.add_argument(
        "--model",
        help="served model ID (default: first model advertised by the service)",
    )
    cap.add_argument("--max-tokens", type=int, default=32)
    cap.add_argument("--concurrency", type=int, default=1)
    cap.add_argument("--url", help="engine API (default: vLLM :8000, SGLang :30000)")
    cap.add_argument("--trace-dir", type=Path)
    cap.add_argument(
        "--server-trace-dir", help="SGLang trace directory as seen by the server"
    )
    cap.add_argument("--topology", type=Path, help="multi-node / PD deployment JSON")
    cap.add_argument(
        "--calibration", type=Path, help="reuse an offline hardware calibration"
    )
    cap.add_argument(
        "--calibrate",
        action="store_true",
        help="benchmark the selected idle devices before capture (cached)",
    )
    cap.add_argument("--devices", help="comma-separated GPU indices or UUIDs")
    cap.add_argument("--ranks", type=int, default=1)
    cap.add_argument("--seconds", type=float, default=5)
    cap.add_argument("--interval-ms", type=float, default=100)
    cap.add_argument("-o", "--output", type=Path, default=Path("serveutil-runs"))
    viewer = sub.add_parser("view", help="open the viewer")
    viewer.add_argument("path", type=Path, nargs="?")
    viewer.add_argument("--desktop", action="store_true")
    cal = sub.add_parser(
        "calibrate", help="measure sustainable hardware reference rates"
    )
    cal.add_argument(
        "--link",
        type=Path,
        help="benchmark a node-to-node network link defined by JSON",
    )
    cal.add_argument("--devices")
    cal.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=[],
        help="also import nvbandwidth, nccl-tests or a ServeUtil reference JSON",
    )
    cal.add_argument(
        "--ranks", type=int, help="global communicator size of imported nccl-tests"
    )
    cal.add_argument("--cache-dir", type=Path)
    cal.add_argument("-o", "--output", type=Path, default=Path("calibration.json"))
    args = p.parse_args(argv)
    try:
        if args.action == "export":
            if args.full:
                from .raw_trace import export_full

                clocks = (
                    json.loads(args.clocks.read_text(encoding="utf8"))
                    if args.clocks else None
                )
                source = args.trace[0] if len(args.trace) == 1 else args.trace
                counts = export_full(source, args.output, clocks=clocks)
            elif len(args.trace) != 1 or args.clocks:
                p.error("use --full to merge multiple traces or choose clocks")
            elif args.trace[0].is_dir():
                from .perfetto import export_capture

                counts = export_capture(args.trace[0], args.output)
            else:
                counts = export(args.trace[0], args.output)
            print(f"{args.output}: {counts['kept']:,} events")
        elif args.action == "report":
            from .report import render

            render(args.trace, args.output, analysis=args.analysis)
            print(args.output)
        elif args.action == "capture":
            from .engines import capture_engine

            args.url = args.url or ("http://localhost:30000" if args.engine == "sglang" else "http://localhost:8000")
            transfers = None
            if args.topology:
                config = json.loads(args.topology.read_text(encoding="utf-8"))
                nodes, targets = config["nodes"], config["targets"]
                if config.get("transfers"):
                    transfers = args.topology.parent / config["transfers"]
            else:
                if not args.trace_dir:
                    p.error("--trace-dir or --topology is required")
                targets = [{"url": args.url, "output_dir": args.server_trace_dir}]
                nodes = [
                    dict(
                        name="local",
                        calibrate=args.calibrate,
                        calibration_file=str(args.calibration.resolve())
                        if args.calibration
                        else None,
                        workdir=str(args.output.resolve()),
                        devices=args.devices.split(",") if args.devices else None,
                        roots=[
                            dict(
                                role="worker",
                                directory=str(args.trace_dir),
                                ranks=args.ranks,
                            )
                        ],
                    )
                ]
            workload = None
            if args.generate is not None:
                from .workload import Workload

                endpoint = config.get("client_url") if args.topology else args.url
                if not endpoint:
                    p.error(
                        "set client_url to the serving router in the topology for --generate"
                    )
                workload = Workload(
                    endpoint.rstrip("/") + "/v1",
                    args.generate,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    concurrency=args.concurrency,
                )
            links = config.get("links", []) if args.topology else []
            path = capture_engine(
                args.engine,
                targets,
                nodes,
                args.output,
                args.seconds,
                args.interval_ms,
                transfers=transfers,
                workload=workload,
                links=links,
            )
            print(path / "report.html")
        elif args.action == "view":
            from .desktop import view

            view(args.path, args.desktop)
        elif args.action == "calibrate":
            from .bundle import write_json
            from .calibration import calibrate
            from .reference import import_reference

            if args.link:
                from .link import calibrate_link

                baseline = calibrate_link(
                    json.loads(args.link.read_text(encoding="utf-8"))
                )
            else:
                baseline = calibrate(
                    devices=args.devices.split(",") if args.devices else None,
                    cache_dir=args.cache_dir,
                )
            for reference in args.reference:
                ids = [row["uuid"] for row in baseline.get("devices", [])]
                baseline["benchmarks"].extend(
                    import_reference(reference, devices=ids, ranks=args.ranks)
                )
            write_json(args.output, baseline)
            print(args.output)
    except (OSError, ValueError, RuntimeError, ijson.JSONError) as exc:
        p.exit(1, f"serveutil: {exc}\n")
