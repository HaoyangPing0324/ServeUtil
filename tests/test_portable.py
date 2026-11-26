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
            for name in ("report.json", "report.json.gz"):
                path = Path(directory) / name
                binary = json.dumps(data).encode()
                path.write_bytes(
                    gzip.compress(binary) if name.endswith(".gz") else binary
                )
                self.assertEqual(payload(render(path)), data)
