import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serveutil.calibration import reference_for
from serveutil.reference import import_reference, port_reference
from serveutil.telemetry import TelemetrySampler, netdev_counters, rdma_counters


class MeasurementTests(unittest.TestCase):
    def test_rdma_scale_direction_precedence_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            port = root / "rdma0/ports/1"
            hw = port / "hw_counters/hw_tx_bytes_cnt"
            tx = port / "counters/port_xmit_data"
            rx = port / "counters/port_rcv_data"
            for file in (hw, tx, rx):
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("10")
            self.assertEqual([c[3] for c in rdma_counters(root)], [1, 4])
            path = root / "samples.jsonl.gz"
            with patch.dict("os.environ", {"SERVEUTIL_RDMA_ROOT": str(root)}):
                sampler = TelemetrySampler(path, devices=[], interval_ms=60_000).start()
                hw.write_text("90")
                rx.write_text("30")
                sampler._sample()
                hw.write_text("2")  # reset: skip this window only
                metadata = sampler.stop()
            with gzip.open(path, "rt") as f:
                rows = [json.loads(line) for line in f]
            for metric in ("rdma.tx", "rdma.rx"):
                row = next(r for r in rows if r["metric"] == metric)
                self.assertAlmostEqual(
                    row["value"] * (row["end_ns"] - row["start_ns"]), 80
                )
            self.assertTrue(
                any("discontinuity" in r["reason"] for r in metadata["unavailable"])
            )
            self.assertEqual(sum(r["metric"] == "rdma.tx" for r in rows), 1)

    def test_interface_selection_excludes_bond_slaves_and_virtual_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("lo", "eth0", "bond0", "veth0"):
                directory = root / name / "statistics"
                directory.mkdir(parents=True)
                for dirn in ("rx", "tx"):
                    (directory / (dirn + "_bytes")).write_text("0")
            (root / "eth0/device").mkdir()
            (root / "eth0/master").mkdir()  # sysfs exposes the master at this path
            (root / "bond0/bonding").mkdir()
            self.assertEqual({r[0] for r in netdev_counters(root)}, {"bond0"})
            self.assertEqual(
                {r[0] for r in netdev_counters(root, {"veth0"})}, {"veth0"}
            )

    def test_port_window_and_cross_node_denominators(self):
        rows = [
            dict(
                node="a",
                device="bond0",
                value=v,
                start_ns=i,
                end_ns=i + 1,
                unit="GB/s",
                source="Linux netdev statistics",
                scope="interface",
                metric="netdev.tx",
            )
            for i, v in enumerate([99, 3, 1, 2, 77])
        ]
        base = port_reference(
            rows,
            start_ns=1,
            end_ns=4,
            peer="b",
            workload="one stream",
            device="bond0",
            direction="tx",
        )
        self.assertEqual(base[0]["value"], 2)
        self.assertIs(reference_for(rows[1], {"benchmarks": base}), base[0])
        self.assertIsNone(reference_for({**rows[1], "node": "b"}, {"benchmarks": base}))
        self.assertIsNone(
            reference_for({**rows[1], "source": "NVML"}, {"benchmarks": base})
        )

    def test_upstream_formats_preserve_fields_and_reject_invalid_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bench.json"
            sample = dict(
                size=1024,
                type="float",
                redop="sum",
                in_place=dict(alg_bw=1, bus_bw=1.5, nwrong=0, time=1.024),
                out_of_place=dict(alg_bw=2, bus_bw=3, nwrong=2),
            )
            path.write_text(
                json.dumps(
                    dict(nccl_version=23000, args=["all_reduce_perf"], results=[sample])
                )
            )
            rows = import_reference(path, ranks=4)
            self.assertEqual([r["value"] for r in rows], [1, 1.5])
            self.assertEqual(rows[0]["parameters"]["ranks"], 4)
            self.assertEqual(rows[1]["metric"], "nccl.all_reduce.bus_bw")
            doc = {
                "nvbandwidth": {
                    "testcases": [
                        dict(
                            name="case",
                            status="Passed",
                            bandwidth_description="memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)",
                            bandwidth_matrix=[["N/A", "26.3"], ["27.5", "N/A"]],
                        )
                    ]
                }
            }
            path.write_text(json.dumps(doc))
            nvrows = import_reference(path, devices=["GPU-a", "GPU-b"])
            self.assertEqual(
                (nvrows[0]["device"], nvrows[0]["parameters"]["row"]), ("GPU-b", 0)
            )
