import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serveutil.engines import capture_engine
from serveutil.report import make_report


class EngineLifecycleTests(unittest.TestCase):
    def capture(
        self, output, events, *, failure=False, stop_failure=False, lost_node=False
    ):
        class Load:
            def start(self):
                events.append("load started")

            def stop(self):
                events.append("requests drained")
                return {
                    "requests": [{"output_tokens": 1}],
                    "errors": ["generation failed"] if failure else [],
                }

            def check(self):
                events.append("load checked")
                if failure:
                    raise RuntimeError("generation failed")

        class Node:
            def __init__(self, config):
                self.config = config
                self.name = config["name"]

            def start(self, interval):
                pass

            def collect(self, capture):
                if self.name == "lost":
                    raise OSError("worker offline")
                events.append("trace collected")
                path = capture.path / "worker.trace.json"
                path.write_text(
                    json.dumps(
                        {
                            "traceEvents": [
                                {
                                    "cat": "kernel",
                                    "ph": "X",
                                    "name": "run",
                                    "ts": 1,
                                    "dur": 2,
                                }
                            ]
                        }
                    )
                )
                return [
                    {
                        "role": "worker",
                        "rank": "local/0",
                        "file": path.name,
                        "auxiliary": False,
                    }
                ], {
                    "name": self.name,
                    "samples": "samples.gz",
                    "telemetry": {},
                    "clock": {"uncertainty_ns": 0, "offset_ns": 0},
                }

            def close(self):
                events.append("node closed")

        def control(target, engine, start):
            events.append("profiler started" if start else "profiler stopped")
            if not start and stop_failure:
                raise RuntimeError("503 during export")
            return {}

        nodes = [
            {"name": "local", "roots": [{"role": "worker", "directory": "traces"}]}
        ]
        if lost_node:
            nodes.append({**nodes[0], "name": "lost"})
        with (
            patch("serveutil.engines.Worker", Node),
            patch("serveutil.engines.control", control),
        ):
            return capture_engine(
                "sglang", [{"url": "unused"}], nodes, output, 0.001, workload=Load()
            )

    def test_drain_before_stop_and_recover_before_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            destination = self.capture(tmp, events)
            self.assertEqual(
                events,
                [
                    "load started",
                    "profiler started",
                    "requests drained",
                    "profiler stopped",
                    "trace collected",
                    "load checked",
                    "node closed",
                ],
            )
            self.assertTrue((destination / "report.html").is_file())

    def test_decode_failure_preserves_rebuildable_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                self.capture(tmp, events, failure=True)
            (destination,) = Path(tmp).iterdir()
            manifest = json.loads((destination / "capture.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["workload"]["errors"], ["generation failed"])
            self.assertEqual(
                make_report(destination)["traces"][0]["events"][0]["kind"], "compute"
            )
            self.assertEqual(events.count("requests drained"), 1)
            self.assertLess(
                events.index("trace collected"), events.index("load checked")
            )

    def test_stop_failure_still_recovers_available_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "503 during export"):
                self.capture(tmp, [], stop_failure=True, lost_node=True)
            (destination,) = Path(tmp).iterdir()
            manifest = json.loads((destination / "capture.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn(
                "lost collection: worker offline", manifest["collection_errors"]
            )
            self.assertEqual(len(manifest["traces"]), 1)
            self.assertTrue((destination / "report.html").is_file())

    def test_cpu_only_recovery_is_allowed_only_for_failed_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = self.capture(tmp, [])
            trace_path = destination / "worker.trace.json"
            trace = json.loads(trace_path.read_text())
            trace["traceEvents"][0]["cat"] = "cpu_op"
            trace_path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "no device activities"):
                make_report(destination)
            manifest_path = destination / "capture.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "failed"
            manifest_path.write_text(json.dumps(manifest))
            recovered = make_report(destination)
            self.assertEqual(recovered["status"], "failed")
            self.assertIn("device_activity_unavailable", recovered["traces"][0])
