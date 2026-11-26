import gzip
import json
from pathlib import Path
import tempfile
import unittest

import ijson

from serveutil import export, kernel_label


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "decode.synthetic.json"


class OverviewTest(unittest.TestCase):
    def test_gzip_preserves_timing_overlaps_names_and_counters(self):
        trace = json.loads(EXAMPLE.read_bytes())
        trace["baseTimeNanoseconds"] = 1750000000000000123
        trace["distributedInfo"] = {"nccl_version": "2.30.7"}
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "in.json.gz"
            dst = Path(directory) / "out.json"
            src.write_bytes(gzip.compress(json.dumps(trace).encode()))
            export(src, dst)
            actual = json.loads(dst.read_bytes())
        self.assertEqual(actual["baseTimeNanoseconds"], trace["baseTimeNanoseconds"])
        self.assertEqual(actual["distributedInfo"], trace["distributedInfo"])
        expected = [e for e in trace["traceEvents"] if e.get("cat") != "cuda_runtime"]
        self.assertEqual(len(expected), len(actual["traceEvents"]))
        for before, after in zip(expected, actual["traceEvents"]):
            if before.get("cat") == "kernel":
                self.assertIn(after["name"], {"Attention", "GEMM", "AllReduce"})
                self.assertEqual(
                    after["args"].pop("serveutil.original_name"), before["name"]
                )
                after["name"] = before["name"]
            self.assertEqual(before, after)

    def test_read_failure_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "broken.json"
            dst = Path(directory) / "out.json"
            src.write_text('{"traceEvents":[{"ph":"C"},')
            dst.write_text("previous")
            with self.assertRaises(ijson.JSONError):
                export(src, dst)
            self.assertEqual(dst.read_text(), "previous")
            with self.assertRaises(ValueError):
                export(src, src)

    def test_gemm_in_template_argument_does_not_relabel_wrapper(self):
        self.assertEqual(
            kernel_label("void lib::wrapper<lib::gemm>(Params)"), "wrapper"
        )


if __name__ == "__main__":
    unittest.main()
