import socket
import tempfile
import unittest
from pathlib import Path

from serveutil.link import calibrate_link
from serveutil.reference import steady_rdma_reference


class LinkTests(unittest.TestCase):
    def test_rdma_reference_excludes_registration_and_teardown(self):
        def measure(rates):
            rows = [
                dict(
                    node="host",
                    device="nic/1",
                    metric="rdma.tx",
                    scope="port",
                    source="IB port",
                    unit="GB/s",
                    value=value,
                    start_ns=i * 100_000_000,
                    end_ns=(i + 1) * 100_000_000,
                )
                for i, value in enumerate(rates)
            ]
            return steady_rdma_reference(
                rows,
                payload_gbs=40,
                seconds=4,
                peer="peer",
                workload="ib_write_bw",
                device="nic/1",
                direction="tx",
            )

        [ref] = measure([0] * 40 + [41] * 19 + [5] + [41] * 20 + [0] * 21)
        self.assertEqual(ref["value"], 41)
        self.assertEqual(ref["samples"], 40)
        self.assertEqual(ref["covered_seconds"], 4)
        with self.assertRaisesRegex(ValueError, "Insufficient sustained"):
            measure([0] * 10 + [41] * 5 + [0] * 20 + [41] * 5)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            measure([85] * 40)

    @unittest.skipUnless(
        Path("/sys/class/net/lo/statistics").is_dir(),
        "requires Linux interface counters",
    )
    def test_workers_confirm_payload_and_match_port_scopes(self):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", 0))
            except OSError:
                self.skipTest("Loopback sockets unavailable")
        with tempfile.TemporaryDirectory() as directory:
            cfg = {"address": "127.0.0.1", "interface": "lo", "workdir": directory}
            result = calibrate_link(
                {
                    "source": dict(cfg, name="sender"),
                    "destination": dict(cfg, name="receiver"),
                    "seconds": 1,
                    "bidirectional": False,
                }
            )
        sender, receiver = result["transfers"]
        self.assertGreater(sender["bytes"], 0)
        self.assertEqual(receiver["bytes"], sender["bytes"])
        self.assertEqual(
            [r["metric"] for r in result["benchmarks"]], ["netdev.tx", "netdev.rx"]
        )
        self.assertEqual(
            [r["node"] for r in result["benchmarks"]], ["sender", "receiver"]
        )
        self.assertTrue(all(r["samples"] >= 3 for r in result["benchmarks"]))
