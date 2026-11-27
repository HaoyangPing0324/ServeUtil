import gzip
import json
import tempfile
import unittest

import ijson
from decimal import Decimal
from pathlib import Path

from serveutil.cli import main
from serveutil.raw_trace import export_full


class RawTraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.dst = self.dir / "merged.json"

    def write(self, name, data):
        path = self.dir / name
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt", encoding="utf8") as f:
            json.dump(data, f)
        return path

    def test_all_events_clock_correction_id_namespaces_and_headers(self):
        paths = []
        traces = []
        nodes = []
        for i in range(2):
            base = 1_750_000_000_000_000_000 + i * 1000
            events = [
                {
                    "ph": "M",
                    "pid": 42,
                    "tid": 7,
                    "name": "thread_name",
                    "args": {"name": "main"},
                },
                {
                    "ph": "B",
                    "ts": 1.125,
                    "tts": 8,
                    "pid": 42,
                    "tid": 7,
                    "name": "child",
                    "sf": 1,
                },
                {"ph": "E", "ts": 2.25, "pid": 42, "tid": 7, "esf": "1"},
                {
                    "ph": "s",
                    "cat": "flow",
                    "name": "x",
                    "ts": 1,
                    "pid": 42,
                    "tid": 7,
                    "id": 9,
                },
                {
                    "ph": "f",
                    "cat": "flow",
                    "name": "x",
                    "ts": 2,
                    "pid": 0,
                    "tid": 1,
                    "id": 9,
                },
                {
                    "ph": "b",
                    "cat": "a",
                    "name": "op",
                    "id2": {"global": "0xbeef"},
                    "scope": "root",
                    "ts": 1,
                    "pid": 42,
                    "tid": 7,
                    "args": {"deep": [False, None, {"custom": "field"}]},
                },
                {
                    "ph": "e",
                    "cat": "a",
                    "name": "op",
                    "id2": {"global": "0xbeef"},
                    "scope": "root",
                    "ts": 3,
                    "pid": 42,
                    "tid": 7,
                },
                {
                    "ph": "X",
                    "ts": 2,
                    "dur": 2.5,
                    "pid": 0,
                    "tid": 1,
                    "bind_id": 9,
                    "flow_in": True,
                },
                {
                    "ph": "O",
                    "ts": 2,
                    "pid": 42,
                    "tid": 7,
                    "id": 8,
                    "args": {"snapshot": {"x": 1}},
                },
            ]
            record = {
                "traceEvents": events,
                "samples": [{"ts": 2, "pid": 42, "tid": 7, "sf": 1, "weight": 4}],
                "stackFrames": {
                    "0": {"name": "root"},
                    "1": {"name": "child", "parent": "0"},
                },
                "displayTimeUnit": "ns",
                "baseTimeNanoseconds": base,
                "vendor": {"nested": [i, {"flag": True}]},
                "deviceProperties": [{"id": 0, "uuid": str(i)}],
            }
            path = self.write(f"{i}.json.gz", record)
            paths.append(path)
            traces.append({"file": path.name, "rank": f"rank{i}", "node": f"host{i}"})
            nodes.append(
                {
                    "name": f"host{i}",
                    "clock": {"offset_ns": i * 2000, "uncertainty_ns": 10},
                }
            )
        self.write("capture.json", {"traces": traces, "nodes": nodes})
        counts = export_full(self.dir, self.dst)
        result = json.loads(self.dst.read_text(), parse_float=Decimal)
        self.assertEqual(counts["kept"], 18)
        rows = result["traceEvents"]
        for i in range(2):
            group = rows[9 * i : 9 * i + 9]
            self.assertEqual(
                [r["ph"] for r in group], ["M", "B", "E", "s", "f", "b", "e", "X", "O"]
            )
            self.assertEqual(group[3]["id"], group[4]["id"])
            self.assertEqual(group[3]["id"], group[7]["bind_id"])
            self.assertEqual(group[5]["id2"], group[6]["id2"])
            self.assertEqual(group[1]["tts"], 8)
            self.assertEqual(group[1]["ts"], Decimal("1.125") + 3 * i)
            self.assertEqual(group[5]["scope"], "root")
            self.assertEqual(group[5]["args"]["deep"][2], {"custom": "field"})
            sample = result["samples"][i]
            self.assertEqual(
                (sample["sf"], sample["tid"]), (group[1]["sf"], group[1]["tid"])
            )
            frame = result["stackFrames"][sample["sf"]]
            self.assertEqual(result["stackFrames"][frame["parent"]]["name"], "root")
            self.assertEqual(
                result["serveutilFullTrace"]["sources"][i]["header"]["vendor"][
                    "nested"
                ][0],
                i,
            )
        self.assertNotEqual(rows[1]["pid"], rows[10]["pid"])
        self.assertNotEqual(rows[1]["tid"], rows[10]["tid"])
        self.assertNotEqual(rows[3]["id"], rows[12]["id"])
        self.assertNotEqual(rows[5]["id2"], rows[14]["id2"])
        self.assertEqual(result["displayTimeUnit"], "ns")
        self.assertEqual(result["baseTimeNanoseconds"], 1_750_000_000_000_000_000)

    def test_multiple_raw_inputs_require_an_origin_and_leave_previous_output(self):
        a = self.write(
            "a.json", [{"ph": "i", "pid": 0, "tid": 1, "ts": 0, "name": "a"}]
        )
        b = self.write("b.json", {"traceEvents": [{"ph": "i", "ts": 0, "name": "b"}]})
        self.dst.write_text("prior")
        with self.assertRaisesRegex(ValueError, "origin_ns"):
            export_full([a, b], self.dst)
        self.assertEqual(self.dst.read_text(), "prior")
        cfg = self.write(
            "clocks.json", {a.name: {"origin_ns": 100}, b.name: {"origin_ns": 150}}
        )
        main(
            [
                "export",
                "--full",
                str(a),
                str(b),
                "--clocks",
                str(cfg),
                "-o",
                str(self.dst),
            ]
        )
        out = json.loads(self.dst.read_text())
        self.assertEqual([r["ts"] for r in out["traceEvents"]], [0, 0.05])
        with self.assertRaisesRegex(ValueError, "distinct"):
            export_full(a, a)

    def test_main_thread_string_lanes_and_vendor_headers(self):
        files = []
        for i in range(2):
            files.append(
                self.write(
                    f"lane{i}.json",
                    {
                        "baseTimeNanoseconds": 123,
                        "vendorContext": {"source": i},
                        "traceEvents": [
                            {"ph": "i", "ts": 10, "pid": 42, "tid": 42, "name": "main"},
                            {
                                "ph": "i",
                                "ts": 11,
                                "pid": "Spans",
                                "tid": "Profiler",
                                "name": "trace",
                            },
                        ],
                    },
                )
            )
        export_full(files, self.dst)
        data = json.loads(self.dst.read_text())
        r0, named0, r1, named1 = data["traceEvents"]
        for row in (r0, r1):
            self.assertEqual(row["tid"], row["pid"])
        self.assertNotEqual(r0["pid"], r1["pid"])
        self.assertNotEqual(named0["pid"], named1["pid"])
        self.assertIn("Profiler", named1["tid"])
        meta = data["serveutilFullTrace"]
        self.assertIn("vendorContext", meta["source_specific_header_fields"])
        self.assertEqual(meta["sources"][1]["header"]["vendorContext"], {"source": 1})
        self.assertIn("unmeasured", meta["sources"][0]["time_basis"])

    def test_single_gzip_preserves_json_bytes_and_truncated_input_is_atomic(self):
        original = b'[  {"ts": 1, "ph":"i", "pid": "GPU", "tid":"0"}  ]\n'
        src = self.dir / "raw.gz"
        src.write_bytes(gzip.compress(original))
        self.assertEqual(export_full(src, self.dst)["kept"], 1)
        self.assertEqual(self.dst.read_bytes(), original)
        src.write_bytes(gzip.compress(original[:-3]))
        with self.assertRaises(ijson.JSONError):
            export_full(src, self.dst)
        self.assertEqual(self.dst.read_bytes(), original)

    def test_exact_private_numbers_and_capture_escape(self):
        src = self.dir / "raw.json"
        src.write_text(
            '{"traceEvents":[{"ph":"i","name":"n","ts":0,"args":{"val":0.123456789012345678901234567890123456789}}]}'
        )
        export_full(src, self.dst)
        self.assertEqual(src.read_bytes(), self.dst.read_bytes())
        self.assertIn("0.123456789012345678901234567890123456789", self.dst.read_text())
        self.write(
            "capture.json", {"traces": [{"file": "../raw.json", "rank": "other"}]}
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            export_full(self.dir, self.dst)
