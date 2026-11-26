import json
import tempfile
import unittest
from pathlib import Path

from serveutil.report import make_report


class CopyTests(unittest.TestCase):
    def test_local_dram_traffic_and_peer_direction_use_comparable_references(self):
        events = []
        for name, info in [
            ("Memcpy DtoD", {}),
            ("Memcpy PtoP", {}),
            ("Memcpy PtoP", {"srcDeviceId": 0, "dstDeviceId": 1}),
            ("Memcpy DtoD", {"srcDeviceId": 0, "dstDeviceId": 2}),
        ]:
            events.append(
                dict(
                    name=name,
                    cat="gpu_memcpy",
                    ph="X",
                    ts=len(events) * 10,
                    dur=10,
                    pid=1,
                    tid=0,
                    args={"bytes": 10000, "device": 0, **info},
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)

            def write(name, data):
                (base / name).write_text(json.dumps(data), encoding="utf-8")

            write("raw.json", {"traceEvents": events})
            write(
                "capture.json",
                {
                    "engine": "torch",
                    "device_map": {"0": "GPU-a", "1": "GPU-b", "2": "GPU-c"},
                    "traces": [{"file": "raw.json", "rank": "0"}],
                },
            )
            write(
                "calibration.json",
                {
                    "benchmarks": [
                        dict(
                            metric="copy." + metric,
                            device="GPU-a",
                            value=4,
                            unit="GB/s",
                            source="CUDA event microbenchmark",
                            parameters=params,
                        )
                        for metric, params in [
                            ("device", {"bytes": 10000}),
                            ("peer", {"src": "GPU-a", "dst": "GPU-b"}),
                        ]
                    ]
                },
            )
            copies = [e["copy"] for e in make_report(base)["traces"][0]["events"]]
        self.assertEqual(copies[0]["GB_s"], 1)
        self.assertEqual(copies[0]["read_write_GB_s"], 2)
        self.assertEqual(copies[0]["reference_pct"], 50)
        self.assertNotIn("reference_pct", copies[1])
        self.assertEqual(copies[2]["reference_pct"], 25)
        self.assertNotIn("reference_pct", copies[3])
