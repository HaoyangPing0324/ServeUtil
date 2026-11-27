import base64
import gzip
import json
import re
import tempfile
import unittest
from pathlib import Path

from serveutil.report import render, render_data


def payload(html):
    return json.loads(
        re.search(
            r'<script id="capture-data" type="application/json">(.*?)</script>', html
        )[1]
    )


class PortableReportTests(unittest.TestCase):
    def test_reopening_large_encoded_html_preserves_one_compression_layer(self):
        import random

        # Even after compression the encoded payload exceeds the compression threshold.
        content = base64.b64encode(random.Random(31).randbytes(1_100_000)).decode(
            "ascii"
        )
        data = {"schema": "serveutil.report/1", "note": content}
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "source.html"
            render_data(data, original)
            with original.open("a") as f:
                f.write("<script>window.foreignReportScript = true;</script>")
            rebuilt = render(original)
            self.assertNotIn("window.foreignReportScript", rebuilt)
            encoded = payload(rebuilt)
            self.assertEqual(
                json.loads(gzip.decompress(base64.b64decode(encoded["payload"]))), data
            )

    def test_portable_report_json_and_gzip(self):
        data = {"schema": "serveutil.report/1", "engine": "torch", "traces": []}
        with tempfile.TemporaryDirectory() as directory:
            for name in ("report.json", "report.json.gz", "REPORT.JSON.GZ"):
                path = Path(directory) / name
                binary = json.dumps(data).encode()
                path.write_bytes(
                    gzip.compress(binary) if name.lower().endswith(".gz") else binary
                )
                self.assertEqual(payload(render(path)), data)

    def test_case_insensitive_gzip_trace_through_report_and_full_export(self):
        from serveutil.raw_trace import export_full

        raw = b'{"traceEvents":[{"ph":"C","name":"bytes","ts":12,"args":{"value":9}}]}'
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "worker.JSON.GZ"
            source.write_bytes(gzip.compress(raw))
            self.assertEqual(
                payload(render(source))["traces"][0]["counters"][0]["points"], [[0, 9]]
            )
            out = Path(directory) / "full.json"
            self.assertEqual(export_full(source, out)["kept"], 1)
            self.assertEqual(out.read_bytes(), raw)

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
        from serveutil.originals import from_html
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
            html = (capture.path / "report.html").read_text(encoding="utf-8")
            data = payload(html)
            self.assertTrue(data["aligned"])
            self.assertEqual(data["time_origin_unix_ns"], "5123")
            self.assertIn("parse_error", data["traces"][0])
            self.assertEqual(data["traces"][1]["counters"][0]["points"], [[0, 3]])
            for item, source in zip(from_html(html), (bad, good), strict=True):
                self.assertEqual(
                    gzip.decompress(base64.b64decode(item["payload"])),
                    source.read_bytes(),
                )
            with self.assertRaisesRegex(ValueError, "bad.json"):
                export_capture(capture.path, Path(directory) / "compact.json")
