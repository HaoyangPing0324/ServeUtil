"""Read an existing llmperfscope run using its sampler and GPU field definitions."""

import json
from collections import defaultdict
from pathlib import Path


def llmperfscope_report(directory):
    try:
        from llmperfscope.gpu import G_MEM, G_POWER, G_SM, G_UTIL
        from llmperfscope.sampler import load_ticks
    except ImportError as exc:
        raise RuntimeError(
            "Install the llmperfscope extra to read lifecycle runs"
        ) from exc

    directory = Path(directory)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "llmperfscope.run/1":
        raise ValueError("unsupported llmperfscope run schema")
    ticks = (
        load_ticks(directory / "samples.jsonl.gz")
        if (directory / "samples.jsonl.gz").exists()
        else []
    )
    origin = metadata.get("t0_monotonic_ns") or (ticks[0]["t"] if ticks else 0)
    counters = defaultdict(list)
    for tick in ticks:
        for device, fields in enumerate(tick.get("g", [])):
            for index, name, unit in (
                (G_UTIL, "GPU utilization", "%"),
                (G_POWER, "GPU power", "W"),
                (G_MEM, "GPU memory", "MiB"),
                (G_SM, "SM clock", "MHz"),
            ):
                if index >= len(fields) or fields[index] is None:
                    continue
                counters[device, name, unit].append(
                    [(tick["t"] - origin) / 1000, fields[index]]
                )
    phases = []
    for p in metadata.get("phases", []):
        if p["end_s"] <= p["start_s"]:
            continue
        phases.append(
            dict(
                name=p["name"],
                kind="stage",
                domain="Engine",
                device="engine",
                lane="lifecycle",
                ts=p["start_s"] * 1e6,
                dur=(p["end_s"] - p["start_s"]) * 1e6,
                args={"group": p.get("group"), "tokens": p.get("tokens")},
            )
        )
    series = [
        dict(
            name=name,
            device=str(device),
            lane="GPU",
            source="llmperfscope",
            unit=unit,
            points=points,
        )
        for (device, name, unit), points in counters.items()
    ]
    return dict(
        schema="serveutil.report/1",
        engine="llmperfscope",
        aligned=True,
        hardware=[],
        findings=[],
        telemetry={
            "diagnosis": metadata.get("diagnosis", {}),
            "notes": metadata.get("notes", []),
        },
        traces=[dict(rank="0", file="run.json", events=phases, counters=series)],
    )


def analyze_trace(path, engine):
    """Compute a per-rank breakdown using an installed upstream analysis library."""
    path = Path(path).resolve()
    if engine == "hta":
        try:
            from hta.trace_analysis import TraceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "Install HolisticTraceAnalysis and its dependencies to use --analysis hta"
            ) from exc
        analysis = TraceAnalysis(
            trace_files={0: str(path)}, include_last_profiler_step=True
        )
        tables = {
            "GPU temporal breakdown": analysis.get_temporal_breakdown(visualize=False),
            "Compute and communication overlap": analysis.get_comm_comp_overlap(
                visualize=False
            ),
        }
    elif engine == "tracelens":
        try:
            from TraceLens.TreePerf import TreePerfAnalyzer
        except ImportError as exc:
            raise RuntimeError(
                "Install AMD TraceLens to use --analysis tracelens"
            ) from exc
        analysis = TreePerfAnalyzer.from_file(str(path), add_python_func=False)
        tables = {"GPU activity breakdown": analysis.get_df_gpu_timeline()}
    else:
        raise ValueError("supported upstream analysis engines are hta and tracelens")
    return {
        label: json.loads(table.to_json(orient="records"))
        for label, table in tables.items()
    }
