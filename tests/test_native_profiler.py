import gzip
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import serveutil

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires PyTorch")
class NativeProfilerTests(unittest.TestCase):
    def workload(self, step, device):
        with torch.profiler.record_function(f"batch-{step}"):
            x = torch.ones(16, 16, device=device, requires_grad=True)
            (x @ x).sum().backward()
            if device == "cuda":
                torch.cuda.synchronize()

    def check_recorder(self, instrumented, device):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = []
            data = []
            observer = torch.profiler.ExecutionTraceObserver().register_callback(
                str(root / "execution.json")
            )
            tb = torch.profiler.tensorboard_trace_handler(
                str(root / "tb"), worker_name="worker", use_gzip=True
            )

            def ready(p):
                self.assertIs(type(p), torch.profiler.profile)
                self.assertTrue(any(e.key == "aten::mm" for e in p.key_averages()))
                self.assertTrue(p.events())
                if hasattr(p, "get_trace_id"):
                    self.assertTrue(p.get_trace_id())
                dirs.append(p.serveutil_directory)
                dest = root / f"user-{len(dirs)}.json"
                p.export_chrome_trace(str(dest))
                data.append(dest.read_bytes())
                tb(p)

            activities = [torch.profiler.ProfilerActivity.CPU]
            if device == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            options = dict(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=2),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                acc_events=True,
                on_trace_ready=ready,
                execution_trace_observer=observer,
            )
            if (
                "custom_trace_id_callback"
                in inspect.signature(torch.profiler.profile).parameters
            ):
                options["custom_trace_id_callback"] = lambda: (
                    "1234567890abcdef1234567890abcdef"
                )
            options["experimental_config"] = torch._C._profiler._ExperimentalConfig(
                verbose=True
            )
            run_options = dict(
                output=root / "captures",
                devices=[] if device == "cpu" else [0],
                calibration=False,
            )
            if instrumented:
                p = torch.profiler.profile(
                    **serveutil.instrument(**run_options, **options)
                )
            else:
                p = serveutil.profile(**run_options, **options)
            p.preset_metadata_json("test_metadata", '{"label":"example"}')
            with p:
                for step in range(4):
                    self.workload(step, device)
                    p.step()
            self.assertFalse(observer.is_registered)
            self.assertEqual(len(dirs), 2)
            self.assertEqual(len(set(dirs)), 2 if instrumented else 1)
            self.assertTrue(json.loads((root / "execution.json").read_text())["nodes"])
            tensorboard = sorted((root / "tb").glob("*.pt.trace.json.gz"))
            self.assertEqual(
                [gzip.decompress(f.read_bytes()) for f in tensorboard], data
            )
            p.export_chrome_trace(str(root / "again.json.gz"))
            self.assertEqual(
                gzip.decompress((root / "again.json.gz").read_bytes()), data[-1]
            )
            self.assertTrue(p.events())
            if (
                "use_python_export"
                in inspect.signature(type(p).export_chrome_trace).parameters
            ):
                python_trace = root / "python-writer.json"
                p.export_chrome_trace(path=str(python_trace), use_python_export=True)
                self.assertIn("traceEvents", json.loads(python_trace.read_text()))
            p.export_stacks(str(root / "stacks.txt"), "self_cpu_time_total")
            self.assertTrue((root / "stacks.txt").is_file())
            # Recent versions deprecate this API; the native dependency and output format are preserved.
            if hasattr(p, "export_memory_timeline"):
                p.export_memory_timeline(str(root / "memory.json.gz"), device=device)
                self.assertTrue(
                    json.loads(gzip.decompress((root / "memory.json.gz").read_bytes()))
                )
            for i, raw in enumerate(data):
                trace = json.loads(raw)
                self.assertEqual(trace["test_metadata"], {"label": "example"})
                if "custom_trace_id_callback" in options and "trace_id" in trace:
                    self.assertEqual(
                        trace["trace_id"], options["custom_trace_id_callback"]()
                    )
                events = trace["traceEvents"]
                names = {e.get("name") for e in events}
                self.assertIn(f"batch-{2 * i + 1}", names)
                self.assertIn("[memory]", names)
                self.assertTrue(any(e.get("cat") == "python_function" for e in events))
                self.assertTrue(
                    any(
                        e.get("name") == "aten::mm"
                        and e.get("args", {}).get("Input Dims")
                        for e in events
                    )
                )
                if device == "cuda":
                    self.assertTrue(any(e.get("cat") == "kernel" for e in events))
                    self.assertTrue({"s", "f"} <= {e.get("ph") for e in events})
                manifest = json.loads((dirs[i] / "capture.json").read_text())
                saved = manifest["traces"][0 if instrumented else i]["file"]
                self.assertEqual(gzip.decompress((dirs[i] / saved).read_bytes()), raw)
                self.assertEqual(manifest["status"], "complete")

    def test_native_cpu_results(self):
        for instrumented in (False, True):
            with self.subTest(instrumented=instrumented):
                self.check_recorder(instrumented, "cpu")

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "requires CUDA"
    )
    def test_native_cuda_results(self):
        for instrumented in (False, True):
            with self.subTest(instrumented=instrumented):
                self.check_recorder(instrumented, "cuda")

    def test_manual_start_stop_default_and_wait_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = serveutil.profile(tmp, devices=[], calibration=False)
            self.assertIs(type(p), torch.profiler.profile)
            p.start()
            self.workload(0, "cpu")
            p.stop()
            self.assertTrue(p.events())
            p.export_chrome_trace(str(Path(tmp) / "manual.json"))
            self.assertTrue((p.serveutil_directory / "report.html").exists())
            with serveutil.profile(
                tmp,
                devices=[],
                calibration=False,
                schedule=torch.profiler.schedule(wait=3, warmup=1, active=2),
            ) as idle:
                torch.ones(2)
                idle.step()
            m = json.loads((idle.serveutil_directory / "capture.json").read_text())
            self.assertEqual(m["traces"], [])
            self.assertEqual(m["status"], "complete")

    def test_callback_error_ends_native_cycle_once(self):
        def fail(_):
            raise RuntimeError("callback failed")

        for entry in (serveutil.profile, serveutil.instrument):
            with (
                self.subTest(entry=entry.__name__),
                tempfile.TemporaryDirectory() as tmp,
            ):
                observer = torch.profiler.ExecutionTraceObserver().register_callback(
                    str(Path(tmp) / "graph.json")
                )
                options = entry(
                    tmp,
                    devices=[],
                    calibration=False,
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    execution_trace_observer=observer,
                    on_trace_ready=fail,
                    schedule=torch.profiler.schedule(
                        wait=0, warmup=1, active=1, repeat=2
                    ),
                )
                p = (
                    torch.profiler.profile(**options)
                    if isinstance(options, dict)
                    else options
                )
                with self.assertRaisesRegex(RuntimeError, "callback failed"):
                    with p:
                        for _ in range(4):
                            torch.ones(3)
                            p.step()
                self.assertFalse(observer.is_registered)
                self.assertTrue((Path(tmp) / "graph.json").exists())
                manifest = json.loads(
                    (p.serveutil_directory / "capture.json").read_text()
                )
                self.assertEqual(manifest["status"], "failed")
