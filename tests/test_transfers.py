import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from serveutil.transfers import _nixl_groups, load_transfers, nixl_deltas, snapshot_nixl

TRACE = "1234" * 8
DECODE_ID = "answer-1"


def request_export(service, room, offset):
    def span(i, name, start, end, parent=None, attrs=None):
        return {
            "traceId": TRACE,
            "spanId": f"{offset + i:016x}",
            "parentSpanId": f"{offset + parent:016x}" if parent else "",
            "name": name,
            "startTimeUnixNano": str(start),
            "endTimeUnixNano": str(end),
            "attributes": [
                {"key": key, "value": {"stringValue": value}}
                for key, value in (attrs or {}).items()
            ],
        }

    stages = [
        span(
            1,
            "request",
            100,
            200,
            attrs={
                "rid": DECODE_ID if service == "decode" else "sender-2",
                "bootstrap_room": room,
            },
        ),
        span(2, "scheduler", 110, 190, parent=1),
    ]
    if service == "prefill":
        stages.append(span(3, "prefill_transfer_kv_cache", 130, 150, parent=2))
    else:
        stages.extend(
            [
                span(3, "decode_transferred", 120, 152, parent=2),
                span(4, "decode_forward", 161, 185, parent=2),
            ]
        )
    return {
        "resource": {
            "attributes": [
                {"key": "host.name", "value": {"stringValue": service + "-node"}}
            ]
        },
        "scopeSpans": [{"spans": stages}],
    }


class TransferTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.src = self.path / "spans.jsonl"
        self.obs = self.path / "requests.json"
        self.description = self.path / "input.json"
        self.description.write_text(
            json.dumps({"spans": self.src.name, "histograms": self.obs.name})
        )
        self.resources = [
            request_export("prefill", "0xff", 100),
            request_export("decode", "255", 200),
        ]
        self.record = {
            "sent_trace_id": TRACE,
            "response_id": DECODE_ID,
            "sent_ns": 90,
            "end_ns": 250,
            "before": {
                "sglang:kv_transfer_total_mb_count": 3,
                "sglang:kv_transfer_total_mb_sum": 4,
            },
            "after": {
                "sglang:kv_transfer_total_mb_count": 4,
                "sglang:kv_transfer_total_mb_sum": 5.25,
            },
        }

    def load(self):
        self.obs.write_text(json.dumps(self.record))
        self.src.write_text(
            "\n".join(json.dumps({"resourceSpans": [r]}) for r in self.resources)
        )
        return load_transfers(self.description)

    def test_different_request_ids_join_on_observed_ancestry(self):
        (row,) = self.load()
        self.assertEqual(
            (
                row["send_start_ns"],
                row["send_end_ns"],
                row["receive_end_ns"],
                row["decode_start_ns"],
            ),
            (130, 150, 152, 161),
        )
        self.assertEqual(row["logical_bytes"], 1310720)
        self.assertEqual(
            (row["source_node"], row["destination_node"]),
            ("prefill-node", "decode-node"),
        )
        self.assertEqual(row["prefill_request_ids"], ["sender-2"])
        self.assertIsNone(load_transfers(self.src)[0]["logical_bytes"])

    def test_effective_rate_uses_engine_latency_histogram(self):
        field = "sglang:kv_transfer_latency_ms"
        self.record["after"].update({field + "_sum": 2.0, field + "_count": 1})
        (row,) = self.load()
        self.assertEqual(row["engine_transfer_ms"], 2.0)
        self.assertAlmostEqual(row["effective_gb_s"], 0.65536)

    def test_room_mismatch_and_disconnected_stage(self):
        self.resources[1] = request_export("decode", "256", 200)
        (row,) = self.load()
        self.assertIsNone(row["logical_bytes"])
        self.assertIsNone(row["receive_end_ns"])
        self.resources[0]["scopeSpans"][0]["spans"][-1]["parentSpanId"] = (
            "eeeeeeeeeeeeeeee"
        )
        self.assertEqual(self.load(), [])

    def test_overlapping_or_stale_histograms_cannot_supply_payload_bytes(self):
        for change in (
            {"response_id": "unrelated"},
            {"end_ns": 129},
            {"sent_trace_id": "4321" * 8},
            {"before": {}},
            {
                "after": {
                    "sglang:kv_transfer_total_mb_count": 5,
                    "sglang:kv_transfer_total_mb_sum": 8,
                }
            },
        ):
            with self.subTest(change=change):
                original = self.record
                self.record = {**original, **change}
                self.assertIsNone(self.load()[0]["logical_bytes"])
                self.record = original


class NixlMetricsTests(unittest.TestCase):
    @staticmethod
    def scrape(count, size, elapsed, timestamp):
        text = (
            f'vllm:nixl_bytes_transferred_count{{engine="0",model_name="test,1"}} {count}\n'
            f'vllm:nixl_bytes_transferred_sum{{model_name="test,1",engine="0"}} {size}\n'
            f'vllm:nixl_xfer_time_seconds_count{{model_name="test,1",engine="0"}} {count}\n'
            f'vllm:nixl_xfer_time_seconds_sum{{model_name="test,1",engine="0"}} {elapsed}\n'
            'vllm:nixl_num_failed_transfers_total{model_name="test,1",engine="0"} 2\n'
            'vllm:nixl_bytes_transferred_created{model_name="test,1",engine="0"} 123\n'
        )
        with patch("urllib.request.urlopen", return_value=BytesIO(text.encode())):
            result = snapshot_nixl({"url": "http://localhost:8000", "role": "decode"})
        return {**result, "sent_ns": timestamp, "received_ns": timestamp}

    def test_active_and_wall_bandwidth_use_different_denominators(self):
        a = self.scrape(10, 50, 0.5, 1_000_000_000)
        b = self.scrape(12, 500000050, 0.75, 2_000_000_000)
        (row,) = nixl_deltas(a, b)
        self.assertEqual(row["logical_bytes"], 500000000)
        self.assertEqual(row["transfers"], 2)
        self.assertEqual(row["active_gb_s"], 2.0)
        self.assertEqual(row["interval_gb_s"], 0.5)
        self.assertEqual(row["failures"], 0)
        self.assertEqual(row["labels"]["model_name"], "test,1")
        b["groups"][0]["values"]["xfer_time_seconds_count"] = "14"
        self.assertIsNone(nixl_deltas(a, b)[0]["active_gb_s"])

    def test_unavailable_or_reset_metrics_preserve_the_reason(self):
        with patch("urllib.request.urlopen", side_effect=OSError("unavailable")):
            failed = snapshot_nixl({"url": "http://localhost:8000", "role": "decode"})
        a = self.scrape(10, 50, 0.5, 1)
        b = self.scrape(2, 25, 0.3, 2)
        self.assertEqual(nixl_deltas(a, failed)[0]["status"], "unavailable")
        self.assertEqual(nixl_deltas(a, b)[0]["status"], "unavailable")
        b = self.scrape(15, 99, 0.7, 2)
        b["groups"][0]["values"]["bytes_transferred_created"] = "456"
        self.assertEqual(nixl_deltas(a, b)[0]["status"], "unavailable")
        self.assertEqual(_nixl_groups("other_metric 1"), [])


if __name__ == "__main__":
    unittest.main()
