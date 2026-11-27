import json
import tempfile
import unittest
from pathlib import Path

from serveutil.report import make_report


class CommunicationRelationships(unittest.TestCase):
    def capture(
        self,
        root,
        ops=("send", "recv"),
        *,
        peers=(1, 0),
        ids=(2**60 + 1, 2**60 + 1),
        ranks=(0, 1),
        anchor=True,
        sizes=(128, 128),
    ):
        manifest = {"traces": [], "nodes": []}
        for i, (op, peer, comm, rank, size) in enumerate(
            zip(ops, peers, ids, ranks, sizes)
        ):
            node = f"node{i}"
            meta = {"name": node, "samples": "missing", "telemetry": {}}
            if anchor:
                meta["clock"] = {"offset_ns": i * 100, "uncertainty_ns": 40}
            manifest["nodes"].append(meta)
            args = {
                "Collective name": op,
                "Group size": 2,
                "Process Group Ranks": "[0, 1]",
                "Process Group Name": "5",
                "Seq": 7,
                "dtype": "Float",
                "In msg nelems": size,
                "Out msg nelems": size,
            }
            if comm is not None:
                args["Comms Id"] = comm
            args["Dst Rank" if op == "send" else "Src Rank"] = peer
            path = root / f"{node}.json"
            path.write_text(
                json.dumps(
                    {
                        "distributedInfo": {"rank": rank},
                        "baseTimeNanoseconds": 1750000000000000000,
                        "traceEvents": [
                            {
                                "name": "ncclKernel",
                                "cat": "kernel",
                                "ph": "X",
                                "ts": 100 + i,
                                "dur": 6,
                                "args": args,
                            }
                        ],
                    }
                )
            )
            manifest["traces"].append(
                {
                    "file": path.name,
                    "node": node,
                    "role": "worker",
                    "rank": f"{node}/worker/{rank}",
                }
            )
        (root / "capture.json").write_text(json.dumps(manifest))
        return make_report(root)

    def test_pair_uses_reciprocal_peers_payload_and_exact_64bit_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self.capture(root)
            self.assertEqual(
                report["communication_links"],
                [
                    {
                        "kind": "send_recv",
                        "evidence": "comms_id",
                        "endpoints": [[0, 0], [1, 0]],
                    }
                ],
            )
            self.assertEqual(
                report["traces"][0]["events"][0]["collective"]["comms_id"],
                str(2**60 + 1),
            )
            for changed in (
                {"ids": (2**60, 2**60 + 1)},
                {"peers": (0, 1)},
                {"sizes": (128, 256)},
                {"ranks": (0, 0)},
            ):
                with self.subTest(changed=changed):
                    self.assertEqual(
                        self.capture(root, **changed)["communication_links"], []
                    )

    def test_collective_without_comms_id_and_unanchored_clocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.capture(
                Path(tmp),
                ops=("allreduce", "allreduce"),
                ids=(None, None),
                anchor=False,
            )
            self.assertFalse(report["aligned"])
            self.assertIsNone(report["time_origin_unix_ns"])
            self.assertEqual(report["traces"][0]["events"][0]["ts"], 0)
            self.assertEqual(
                report["communication_links"][0]["evidence"], "group_sequence"
            )
