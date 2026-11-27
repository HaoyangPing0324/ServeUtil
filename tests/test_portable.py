import json
import tempfile
import unittest
from pathlib import Path


class CaptureDataTests(unittest.TestCase):
    def test_cuda_cycle_without_device_events_is_identified_individually(self):
        from serveutil.bundle import Capture
        from serveutil.report import make_report

        with tempfile.TemporaryDirectory() as directory:
            capture = Capture(directory, "torch")
            for index, activities in enumerate((["CPU", "CUDA"], ["CPU"])):
                path = capture.path / "traces" / f"{index}.json"
                path.write_text('{"traceEvents":[]}')
                capture.add_trace(path, 0, activities=activities)
            report = make_report(capture.path)
            self.assertIn(
                "CUDA profiling was requested",
                report["traces"][0]["device_activity_unavailable"],
            )
            self.assertNotIn("device_activity_unavailable", report["traces"][1])

    def test_malformed_native_cycle_retains_bytes_and_valid_siblings(self):
        from serveutil.bundle import Capture
        from serveutil.perfetto import export_capture

        with tempfile.TemporaryDirectory() as directory:
            capture = Capture(directory, "torch")
            good = capture.path / "traces" / "good.json"
            bad = good.with_name("bad.json")
            good.write_bytes(
                b'{"baseTimeNanoseconds":123,"traceEvents":[{"ph":"C","ts":5,"name":"n","args":{"value":3}}]}'
            )
            bad.write_bytes(b'{"label":"a/"b","traceEvents":[]}')
            for source in (bad, good):
                capture.add_trace(source, 0)
            capture.finish()
            data = json.loads((capture.path / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(data["aligned"])
            self.assertEqual(data["time_origin_unix_ns"], "5123")
            self.assertIn("parse_error", data["traces"][0])
            self.assertEqual(data["traces"][1]["counters"][0]["points"], [[0, 3]])
            self.assertEqual(bad.read_bytes(), b'{"label":"a/"b","traceEvents":[]}')
            with self.assertRaisesRegex(ValueError, "bad.json"):
                export_capture(capture.path, Path(directory) / "compact.json")
