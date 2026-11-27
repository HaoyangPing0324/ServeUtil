import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from serveutil.analysis import findings, matmul_work, subtract, union
from serveutil.bundle import Capture
from serveutil.engines import Worker
from serveutil.report import make_report, stage_label


class CaptureTest(unittest.TestCase):
    def test_clock_offsets_samples_and_compatible_reference(self):
        with TemporaryDirectory() as tmp:
            c = Capture(tmp, "sglang")
            c.manifest["nodes"] = []
            for name, offset in (("a", 500_000), ("b", -700_000)):
                epoch = 1_700_000_000_000_000_000 - offset
                file = c.path / "traces" / (name + ".json")
                file.write_text(
                    json.dumps(
                        {
                            "baseTimeNanoseconds": epoch,
                            "traceEvents": [
                                {
                                    "ph": "X",
                                    "cat": "kernel",
                                    "name": "gemm",
                                    "pid": 1,
                                    "tid": 2,
                                    "ts": 100,
                                    "dur": 8,
                                }
                            ],
                        }
                    )
                )
                c.manifest["traces"].append(
                    dict(
                        file=str(file.relative_to(c.path)),
                        node=name,
                        role="decode",
                        rank=name,
                    )
                )
                f = c.path / (name + ".jsonl.gz")
                with gzip.open(f, "wt") as stream:
                    stream.write(
                        json.dumps(
                            dict(
                                metric="nvlink.tx",
                                device="GPU-1",
                                unit="GB/s",
                                scope="device",
                                source="NVML",
                                start_ns=epoch + 100000,
                                end_ns=epoch + 107000,
                                value=33,
                            )
                        )
                    )
                cal = c.path / (name + "-cal.json")
                key = dict(metric="nvlink.tx", device="GPU-1", unit="GB/s", value=66)
                cal.write_text(
                    json.dumps(
                        {
                            "benchmarks": [
                                {**key, "source": "NVML"},
                                {**key, "source": "other", "value": 999},
                            ]
                        }
                    )
                )
                c.manifest["nodes"].append(
                    dict(
                        name=name,
                        samples=f.name,
                        calibration=cal.name,
                        telemetry={},
                        clock=dict(offset_ns=offset, uncertainty_ns=1),
                    )
                )
            c.save()
            r = make_report(c.path)
            self.assertEqual([t["events"][0]["ts"] for t in r["traces"]], [0, 0])
            self.assertEqual(
                [h["points"] for h in r["hardware"]], [[[7, 33, 0]], [[7, 33, 0]]]
            )
            self.assertTrue(all(h["baseline"]["value"] == 66 for h in r["hardware"]))

    def test_temporal_and_compute_semantics(self):
        self.assertEqual(union([(0, 9), (5, 12), (15, 18)]), [[0, 12], [15, 18]])
        self.assertEqual(
            list(subtract([[2, 15]], [[1, 4], [8, 9], [10, 20]])), [(4, 8), (9, 10)]
        )
        events = [
            dict(kind="compute", device="a", ts=0, dur=400),
            dict(kind="communication", device="a", ts=20, dur=250),
            dict(kind="communication", device="b", ts=20, dur=250),
        ]
        self.assertEqual([f["device"] for f in findings(events, "rank")], ["b"])
        self.assertEqual(
            stage_label("execute_context_1(4)_generation_2(2)"), "Mixed batch"
        )
        self.assertEqual(stage_label("step[EXTEND bs=1]"), "Prefill")
        self.assertEqual(
            matmul_work(
                "aten::bmm",
                {
                    "Input Dims": [[2, 3, 4], [2, 4, 5]],
                    "Input type": ["float", "float"],
                },
            )["flops"],
            240,
        )
        self.assertIsNone(matmul_work("aten::mm", {"Input Dims": [[2, 3], [2, 3]]}))

    def test_integer_gemm_reports_operations_and_skips_mixed_inputs(self):
        args = {"Input Dims": [[4, 8], [8, 2]], "Input type": ["signed char", "signed char"]}
        work = matmul_work("aten::_int_mm", args)
        self.assertEqual((work["operations"], work["rate_unit"]), (128, "TOP/s"))
        self.assertNotIn("flops", work)
        args["Input type"] = ["float8_e4m3fn", "float8_e5m2"]
        self.assertIsNone(matmul_work("aten::_scaled_mm", args))

    def test_worker_only_transfers_new_native_outputs(self):
        with TemporaryDirectory() as tmp:
            c = Capture(tmp, "fixture")
            raw = Path(tmp) / "native"
            raw.mkdir()
            (raw / "old.trace.json").write_text("old")
            w = Worker(
                dict(
                    name="node",
                    roots=[dict(role="prefill", directory=str(raw))],
                    devices=[],
                    workdir=tmp,
                )
            )
            try:
                w.start(40)
                (raw / "TP-0.trace.json").write_text('{"traceEvents":[]}')
                records, meta = w.collect(c)
                self.assertEqual([r["rank"] for r in records], ["node/prefill/0"])
                self.assertTrue((c.path / meta["samples"]).is_file())
            finally:
                w.close()


class ProfilerIntegrationTest(unittest.TestCase):
    def test_scheduled_native_callback_can_export_its_own_file(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional")
        from serveutil import profile

        with TemporaryDirectory() as tmp:
            callbacks = []

            def ready(p):
                dest = Path(tmp) / f"user-{len(callbacks)}.json"
                p.export_chrome_trace(str(dest))
                callbacks.append(json.loads(dest.read_text())["traceEvents"])

            with profile(
                tmp,
                calibration=False,
                devices=[],
                on_trace_ready=ready,
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=2),
            ) as p:
                for _ in range(4):
                    with torch.profiler.record_function("Decode step"):
                        torch.ones(4) + 1
                    p.step()
            self.assertEqual(len(callbacks), 2)
            self.assertTrue(
                all(
                    any(e.get("name") == "Decode step" for e in batch)
                    for batch in callbacks
                )
            )
            self.assertEqual(
                len(
                    json.loads((p.serveutil_directory / "capture.json").read_text())[
                        "traces"
                    ]
                ),
                2,
            )

    def test_native_profiler_observer_follows_active_cycle(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional")
        from serveutil import instrument

        with TemporaryDirectory() as tmp:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                **instrument(tmp, devices=[]),
            ) as p:
                with torch.profiler.record_function("Prefill"):
                    torch.ones(16) + 2
                p.step()
            manifest = next(Path(tmp).glob("*/capture.json"))
            data = json.loads(manifest.read_text())
            self.assertEqual(data["status"], "complete")
            self.assertGreater(
                data["telemetry"]["end_ns"], data["telemetry"]["start_ns"]
            )
            self.assertTrue((manifest.parent / "report.json").is_file())
