import json
import tempfile
import unittest
from pathlib import Path

from serveutil.collectives import CollectiveRecorder, match_nccl_reference
from serveutil.report import load_trace, make_report


def trace_args(op="allreduce", **fields):
    return {
        "Collective name": op,
        "Group size": 2,
        "Process Group Name": "3",
        "Process Group Description": "tp:device",
        "Process Group Ranks": "[0, 1]",
        "Seq": 3597,
        "In msg nelems": 1024,
        "Out msg nelems": 1024,
        "dtype": "BFloat16",
        **fields,
    }


def record(*items):
    recorder = CollectiveRecorder()
    events = []
    for start, duration, args in items:
        event = {"ts": start, "dur": duration, "device": "0"}
        recorder.observe(event, args)
        events.append(event)
    recorder.finish()
    return [event.get("collective", {}) for event in events]


class CollectiveTests(unittest.TestCase):
    def test_sglang_allreduce_and_gather(self):
        reduce, gather = record(
            (1, 8, trace_args()),
            (
                100,
                64,
                trace_args(
                    "_allgather_base",
                    **{
                        "Seq": 3654,
                        "In msg nelems": 75968,
                        "Out msg nelems": 151936,
                    },
                ),
            ),
        )
        self.assertEqual(
            reduce["group"],
            {"id": "3", "size": 2, "ranks": [0, 1], "description": "tp:device"},
        )
        self.assertEqual(reduce["algorithm_GB_s"], 2048 / 8000)
        self.assertEqual(reduce["bus_factor"], 1)
        self.assertEqual(gather["bytes"], 2 * 151936)
        self.assertEqual(gather["bus_GB_s"], gather["algorithm_GB_s"] / 2)

    def test_vllm_send_recv_and_broadcast_same_sequence(self):
        rows = record(
            (0, 10, trace_args("send", **{"Dst Rank": 1})),
            (0, 10, trace_args("recv", **{"Src Rank": 0, "In msg nelems": 0})),
            (
                1,
                10,
                trace_args(
                    "broadcast",
                    **{
                        "Process Group Name": "21",
                        "In msg nelems": 1,
                        "Out msg nelems": 1,
                        "dtype": "Long",
                    },
                ),
            ),
        )
        for row, (direction, peer) in zip(rows, [("send", 1), ("recv", 0)]):
            self.assertEqual((row["direction"], row["peer_rank"]), (direction, peer))
            self.assertEqual(row["bytes"], 2048)
            self.assertNotIn("bus_GB_s", row)
        self.assertEqual(rows[2]["group"]["id"], "21")
        self.assertEqual(rows[2]["bytes"], 8)

    def test_overlapping_and_gapped_kernels_share_one_byte_count(self):
        a, b, c, other = record(
            (14, 3, trace_args()),
            (10, 5, trace_args()),
            (18, 2, trace_args()),
            (12, 1, trace_args(**{"Process Group Name": "21"})),
        )
        self.assertEqual(
            (b["kernel_count"], b["duration_us"], b["algorithm_GB_s"]), (3, 10, 0.2048)
        )
        self.assertTrue(a["part_of_operation"] and c["part_of_operation"])
        self.assertNotIn("bytes", c)
        self.assertEqual(other["kernel_count"], 1)

    def test_inconsistent_or_incomplete_metadata(self):
        for fields in (
            {"Out msg nelems": 1023},
            {"dtype": "unknown"},
            {"Process Group Ranks": "[0,0]"},
            {"In msg nelems": -1},
            {"Group size": 1},
            {"Seq": None},
            {"Process Group Name": None},
        ):
            with self.subTest(fields=fields):
                (row,) = record((0, 10, trace_args(**fields)))
                self.assertNotIn("algorithm_GB_s", row)
        x, y = record((0, 2, trace_args()), (3, 4, trace_args(dtype="float")))
        self.assertEqual(x["unavailable"], y["unavailable"])

    def test_scatter_and_variable_alltoall(self):
        scatter, equal, variable = record(
            (1, 8, trace_args("_reduce_scatter_base", **{"Out msg nelems": 512})),
            (
                2,
                8,
                trace_args(
                    "alltoall_base", **{"In split size": "[]", "Out split size": "[]"}
                ),
            ),
            (
                3,
                8,
                trace_args(
                    "alltoall_base",
                    **{"In split size": "[1,3]", "Out split size": "[3,1]", "Seq": 50},
                ),
            ),
        )
        self.assertEqual(scatter["bytes"], 2048)
        self.assertEqual(scatter["bus_factor"], 0.5)
        self.assertEqual(equal["bus_factor"], 0.5)
        self.assertEqual(variable["bytes"], 2048)
        self.assertNotIn("bus_GB_s", variable)

    def test_exact_reference_requires_external_context(self):
        (op,) = record((0, 8, trace_args()))
        config = {
            "devices": ["GPU-A", "GPU-B"],
            "placement": "in_place",
            "reduction": "sum",
        }
        row = {
            "source": "nccl-tests",
            "unit": "GB/s",
            "metric": "nccl.all_reduce.alg_bw",
            "value": 0.512,
            "parameters": {
                "collective": "all_reduce",
                "bytes": 2048,
                "ranks": 2,
                "dtype": "bfloat16",
                **config,
            },
        }
        self.assertEqual(match_nccl_reference(op, [row], **config)["percent"], 50)
        row["parameters"]["version"] = 23007
        self.assertIsNotNone(
            match_nccl_reference(op, [row], nccl_version="2.30.7", **config)
        )
        self.assertIsNone(
            match_nccl_reference(op, [row], nccl_version="2.28.3", **config)
        )
        for changed in (
            {"devices": ["GPU-B", "GPU-A"]},
            {"placement": None},
            {"reduction": "max"},
        ):
            self.assertIsNone(match_nccl_reference(op, [row], **{**config, **changed}))
        self.assertIsNone(
            match_nccl_reference(op, [row, {**row, "value": 0.3}], **config)
        )

    def test_kineto_report_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "trace.json")
            path.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "cat": "kernel",
                                "ph": "X",
                                "pid": 1,
                                "tid": 3,
                                "ts": 200,
                                "dur": 10,
                                "name": "user kernel",
                                "args": trace_args("recv", **{"Src Rank": 0}),
                            }
                        ]
                    }
                )
            )
            (event,) = load_trace(path, "rank0")["events"]
            self.assertEqual(event["kind"], "communication")
            self.assertEqual(event["collective"]["algorithm_GB_s"], 0.2048)

    def test_capture_report_matches_only_configured_role_and_primary_kernel(self):
        devices = ["GPU-A", "GPU-B"]
        group = {
            "devices": devices,
            "operations": {
                "all_reduce": {"placement": "out_of_place", "reduction": "max"}
            },
        }

        def event(name, cat, ts, dur, args):
            return {
                "name": name,
                "cat": cat,
                "ph": "X",
                "pid": 2,
                "tid": 3,
                "ts": ts,
                "dur": dur,
                "args": args,
            }

        args = trace_args(**{"External id": 92})
        trace = {
            "traceEvents": [
                event("c10d::allreduce_", "cpu_op", 1, 5, {"External id": 91}),
                event("record_param_comms", "cpu_op", 2, 1, args),
                event(
                    "ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "kernel", 10, 3, args
                ),
                event(
                    "ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "kernel", 14, 4, args
                ),
            ]
        }
        row = {
            "source": "nccl-tests",
            "unit": "GB/s",
            "metric": "nccl.all_reduce.alg_bw",
            "value": 0.512,
            "parameters": {
                "collective": "all_reduce",
                "devices": devices,
                "ranks": 2,
                "bytes": 2048,
                "placement": "in_place",
                "reduction": "sum",
                "dtype": "bfloat16",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trace.json").write_text(json.dumps(trace))
            (root / "ref.json").write_text(json.dumps({"benchmarks": [row]}))
            manifest = {
                "traces": [
                    {"file": "trace.json", "node": "head", "role": role}
                    for role in ("worker", "decode")
                ],
                "nodes": [
                    {
                        "name": "head",
                        "telemetry": {},
                        "calibration": "ref.json",
                        "samples": "samples.jsonl",
                        "clock": {"offset_ns": 0, "uncertainty_ns": 0},
                        "collective_groups": {"worker": {"3": group}},
                    }
                ],
            }
            (root / "capture.json").write_text(json.dumps(manifest))
            result = make_report(root)
            measured = [
                [e["collective"] for e in t["events"] if "collective" in e]
                for t in result["traces"]
            ]
            self.assertEqual(measured[0][0]["reference_pct"], 50)
            self.assertEqual(
                measured[0][0]["reference"]["parameters"], row["parameters"]
            )
            self.assertEqual(measured[0][0]["reduction"], "sum")
            self.assertEqual(measured[0][0]["placement"], "in_place")
            self.assertNotIn("reference", measured[0][1])
            self.assertNotIn("reference", measured[1][0])
            # Generic kernels and missing CPU operators use explicitly configured facts.
            trace["traceEvents"] = [event("nccl generic", "kernel", 10, 8, args)]
            (root / "trace.json").write_text(json.dumps(trace))
            self.assertNotIn(
                "reference", make_report(root)["traces"][0]["events"][0]["collective"]
            )
            group["operations"]["all_reduce"] = {
                "placement": "in_place",
                "reduction": "sum",
            }
            (root / "capture.json").write_text(json.dumps(manifest))
            self.assertEqual(
                make_report(root)["traces"][0]["events"][0]["collective"][
                    "reference_pct"
                ],
                50,
            )
            trace["distributedInfo"] = {"nccl_version": "2.30.7"}
            (root / "trace.json").write_text(json.dumps(trace))
            row["parameters"]["version"] = 22803
            (root / "ref.json").write_text(json.dumps({"benchmarks": [row]}))
            tr = make_report(root)["traces"][0]
            self.assertEqual(tr["nccl_version"], "2.30.7")
            self.assertNotIn("reference_pct", tr["events"][0]["collective"])
