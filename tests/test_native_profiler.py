import gzip
import inspect
import json
import subprocess
import sys
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
            self.assertTrue(json.loads((root / "execution.json").read_text(encoding="utf-8"))["nodes"])
            tensorboard = sorted((root / "tb").glob("*.pt.trace.json.gz"))
            self.assertEqual(
                [gzip.decompress(f.read_bytes()) for f in tensorboard], data
            )
            p.export_chrome_trace(str(root / "again.json.gz"))
            self.assertEqual(inspect.signature(p.export_chrome_trace),
                             inspect.signature(type(p).export_chrome_trace.__get__(p)))
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
                self.assertIn("traceEvents", json.loads(python_trace.read_text(encoding="utf-8")))
                last = json.loads((dirs[-1] / "capture.json").read_text())["traces"][-1]
                alternate, = last["native_exports"]
                self.assertEqual((dirs[-1] / alternate).read_bytes(), python_trace.read_bytes())
            if (
                "cuda_graph_annotations"
                in inspect.signature(type(p).export_chrome_trace).parameters
            ):
                standard = root / "standard.json"
                p.export_chrome_trace(
                    str(standard), cuda_graph_annotations={}, default_stream=9
                )
                self.assertEqual(standard.read_bytes(), data[-1])
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
                manifest = json.loads((dirs[i] / "capture.json").read_text(encoding="utf-8"))
                saved = manifest["traces"][0 if instrumented else i]["file"]
                self.assertEqual(gzip.decompress((dirs[i] / saved).read_bytes()), raw)
                self.assertEqual(manifest["status"], "complete")

    def test_profiler_namespace_preserves_native_helpers(self):
        import serveutil.profiler as profiler

        self.assertEqual(set(profiler.__all__), set(torch.profiler.__all__))
        for name in torch.profiler.__all__:
            if name != "profile":
                self.assertIs(getattr(profiler, name), getattr(torch.profiler, name))
        self.assertIs(profiler.profile, serveutil.profile)
        accepted = inspect.signature(serveutil.profile).parameters
        for name, param in inspect.signature(torch.profiler.profile).parameters.items():
            self.assertEqual(accepted[name], param)

    def test_native_class_api_and_subclass_lifecycle(self):
        from serveutil import profiler

        native = torch.profiler.profile
        self.assertTrue(inspect.isclass(profiler.profile))
        for name in dir(native):
            if not name.startswith("_"):
                self.assertIs(getattr(profiler.profile, name), getattr(native, name))
        self.assertTrue(issubclass(profiler.profile, native))
        self.assertIsInstance(native(), profiler.profile)
        self.assertNotIsInstance(object(), profiler.profile)
        for name, param in inspect.signature(native.__init__).parameters.items():
            self.assertEqual(inspect.signature(profiler.profile.__init__).parameters[name], param)

        class NamedProfiler(profiler.profile):
            def __init__(self, label, output):
                self.label = label
                super().__init__(output, activities=[torch.profiler.ProfilerActivity.CPU])

            def step(self):
                self.steps_seen = getattr(self, "steps_seen", 0) + 1
                return super().step()

        with tempfile.TemporaryDirectory() as tmp:
            with NamedProfiler("projection", tmp) as recorder:
                with profiler.record_function(recorder.label):
                    torch.ones(4).sum()
                recorder.step()
            self.assertIs(type(recorder), NamedProfiler)
            self.assertIsInstance(recorder, profiler.profile)
            self.assertNotIsInstance(native(), NamedProfiler)
            self.assertEqual(recorder.steps_seen, 1)
            self.assertIn("projection", {row.key for row in recorder.key_averages()})
            capture = recorder.serveutil_directory
            entry, = json.loads((capture / "capture.json").read_text())["traces"]
            with gzip.open(capture / entry["file"], "rt") as f:
                self.assertIn("projection", {e.get("name") for e in json.load(f)["traceEvents"]})

    def test_callback_can_change_between_saved_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            called = []
            def first(p):
                called.append(("first", p.step_num))
                p.on_trace_ready = second

            def second(p):
                called.append(("second", p.step_num))
                p.on_trace_ready = None

            p = serveutil.profile(tmp, activities=[torch.profiler.ProfilerActivity.CPU],
                                  schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=3))
            self.assertIsNone(p.on_trace_ready)
            p.on_trace_ready = first
            with p:
                for step in range(3):
                    with torch.profiler.record_function(f"cycle-{step}"):
                        torch.ones(2) + 1
                    p.step()
            self.assertEqual(called, [("first", 1), ("second", 2)])
            self.assertIsNone(p.on_trace_ready)
            manifest = json.loads((p.serveutil_directory / "capture.json").read_text())
            self.assertEqual(len(manifest["traces"]), 3)
            for i, entry in enumerate(manifest["traces"]):
                with gzip.open(p.serveutil_directory / entry["file"], "rt") as stream:
                    events = json.load(stream)["traceEvents"]
                self.assertIn(f"cycle-{i}", [event.get("name") for event in events])

    def test_native_cpu_results(self):
        for instrumented in (False, True):
            with self.subTest(instrumented=instrumented):
                self.check_recorder(instrumented, "cpu")

    def run_in_fresh_process(self):
        # Kineto initialization can depend on the first activity set in this process.
        if __name__ == "__main__":
            return False
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "NativeProfilerTests." + self._testMethodName,
            ],
            check=True,
        )
        return True

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "requires CUDA"
    )
    def test_native_cuda_results(self):
        if self.run_in_fresh_process():
            return
        for instrumented in (False, True):
            with self.subTest(instrumented=instrumented):
                self.check_recorder(instrumented, "cuda")

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "requires CUDA"
    )
    def test_cpu_then_cuda_matches_native_device_visibility(self):
        if self.run_in_fresh_process():
            return
        from serveutil.report import make_report

        cpu, cuda = (
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with torch.profiler.profile(activities=[cpu]):
                self.workload(0, "cpu")
            with torch.profiler.profile(activities=[cpu, cuda]) as native:
                self.workload(1, "cuda")
            source = Path(tmp) / "native.json"
            native.export_chrome_trace(str(source))
            with serveutil.profile(
                tmp, devices=[0], activities=[cpu, cuda], calibration=False
            ) as recorded:
                self.workload(1, "cuda")
            upstream = make_report(source)["traces"][0]
            captured = make_report(recorded.serveutil_directory)["traces"][0]

            def kernels(trace):
                return sum(e["kind"] == "compute" for e in trace["events"])

            self.assertEqual(bool(kernels(captured)), bool(kernels(upstream)))
            self.assertEqual(
                "device_activity_unavailable" in captured, not kernels(upstream)
            )

    def test_observe_preserves_native_writer_and_all_cycles(self):
        from serveutil.report import make_report

        with tempfile.TemporaryDirectory() as tmp:
            with serveutil.observe(tmp, devices=[], calibration=False) as capture:
                options = dict(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    schedule=torch.profiler.schedule(
                        wait=0, warmup=1, active=1, repeat=2
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(capture / "traces"), use_gzip=True
                    ),
                )
                with torch.profiler.profile(**options) as p:
                    for i in range(4):
                        self.workload(i, "cpu")
                        p.step()
                self.assertNotIn("export_chrome_trace", vars(p))
                # Additional binary outputs remain byte-exact beside the two JSON cycles.
                (capture / "traces" / "os.pftrace").write_bytes(b"\x0a\x00")
            self.assertTrue((capture / "report.json").is_file())
            report = make_report(capture)
            self.assertEqual(sum(bool(t["events"]) for t in report["traces"]), 2)
            self.assertEqual(
                sum(bool(t.get("native_only")) for t in report["traces"]), 1
            )
            self.assertTrue(report["aligned"])

    def test_native_metric_names_reach_torch_activity_config(self):
        from serveutil.telemetry import kineto_activities

        if not hasattr(torch.profiler, "PerformanceMetricsConfig"):
            with self.assertRaisesRegex(RuntimeError, "lacks PerformanceMetricsConfig"):
                kineto_activities(torch, ["test_metric"])
            return
        activities = kineto_activities(torch, ["test_metric"], interval_ms=5)
        native = torch.profiler.profile(activities=activities)
        config = native.activity_configs[
            torch.profiler.ProfilerActivity.CUDA
        ].profiler_configs[0]
        self.assertEqual(config.metric_names, ["test_metric"])
        self.assertEqual(config.sampling_interval_ms, 5)
        options = serveutil.instrument(
            devices=[], calibration=False, metrics=["test_metric"]
        )
        configured = torch.profiler.profile(**options)
        actual = configured.activity_configs[
            torch.profiler.ProfilerActivity.CUDA
        ].profiler_configs[0]
        self.assertEqual(actual.metric_names, ["test_metric"])
        self.assertEqual(actual.sampling_interval_ms, 1)
        if hasattr(torch.profiler, "CuspyConfig"):
            self.assertTrue(configured._use_cuspy)
        self.assertFalse(options["execution_trace_observer"].include_hbm)

    def test_deferred_native_binary_export_runs_user_writer_first(self):
        from serveutil.bundle import Capture
        from serveutil.recording import _complete_cycle

        class DeferredProfiler:
            _use_cuspy = True
            pending = None

            def export_chrome_trace(self, path):
                if self.pending is not None:
                    raise AssertionError("window exported twice")
                self.pending = Path(str(path) + ".gz")

            def wait_for_exports(self):
                self.pending.write_bytes(gzip.compress(b"\x0a\x00"))

        with tempfile.TemporaryDirectory() as tmp:
            capture = Capture(tmp, "torch")
            recorder = DeferredProfiler()
            original = Path(tmp) / "native.pftrace"
            _complete_cycle(
                capture, recorder, lambda p: p.export_chrome_trace(str(original))
            )
            self.assertNotIn("export_chrome_trace", vars(recorder))
            saved = capture.path / capture.manifest["traces"][0]["file"]
            self.assertEqual(saved.read_bytes(), recorder.pending.read_bytes())
            capture.finish()
            self.assertTrue((capture.path / "report.json").exists())

    def test_native_binary_survives_unavailable_companion_encoding(self):
        from threading import Lock
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from serveutil.bundle import Capture
        from serveutil.recording import _preserve_cuspy_format

        with tempfile.TemporaryDirectory() as tmp:
            capture = Capture(tmp, "torch")
            target = Path(tmp) / "caller.pftrace.gz"
            raw = b"native output"
            window = dict(built={}, out=str(target), cpu="cpu.json")
            def write(window_id):
                windows.pop(window_id)
                target.write_bytes(raw)
            windows = {1: window}
            observer = SimpleNamespace(_windows=windows, _lock=Lock(), _maybe_write=write,
                                       _pftrace_compression_level=1)
            prof = SimpleNamespace(_cuspy_profiler_observer=observer, _cuspy_window_id=1)
            binary = capture.path / "traces" / "cycle.pftrace.gz"
            companion = binary.with_name("cycle.json.gz")
            encoder = Mock(side_effect=ValueError("unavailable companion"))
            with patch.dict(sys.modules, {"torch.profiler._cuspy.trace": SimpleNamespace(
                    merge_trace_window_into_chrome_trace=encoder)}):
                with _preserve_cuspy_format(capture, prof, companion, binary):
                    observer._maybe_write(1)
                    target.unlink()
            self.assertEqual(binary.read_bytes(), raw)
            self.assertFalse(companion.exists())
            self.assertIs(observer._maybe_write, write)
            self.assertEqual(capture.manifest["export_warnings"], [
                {"file": companion.name, "error": "unavailable companion"}
            ])

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available() and hasattr(torch.profiler, "CuspyConfig"),
        "requires the native Cuspy CUDA backend",
    )
    def test_cuspy_post_stop_binary_and_json(self):

        if self.run_in_fresh_process():
            return
        cpu, cuda = torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
        activities = [cpu, {cuda: torch.profiler.ProfilerActivityConfig(
            profiler_configs=[torch.profiler.CuspyConfig()]
        )}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for entry in (torch.profiler.profile, serveutil.profile):
                options = {} if entry is torch.profiler.profile else dict(output=root, calibration=False)
                with entry(activities=activities, **options) as recorder:
                    self.workload(0, "cuda")
                dest = root / (entry.__module__ + ".pftrace")
                recorder.export_chrome_trace(str(dest))
                recorder.wait_for_exports()
                binary = Path(str(dest) + ".gz")
                if entry is torch.profiler.profile and not binary.exists():
                    self.skipTest("native Cuspy observer could not collect a window")
                self.assertIn(b"batch-0", gzip.decompress(binary.read_bytes()))
            capture = recorder.serveutil_directory
            entry = json.loads((capture / "capture.json").read_text())["traces"][0]
            self.assertEqual((capture / entry["native_binary"]).read_bytes(), binary.read_bytes())
            recorder.export_chrome_trace(str(root / "last.json.gz"))
            trace = json.loads(gzip.decompress((root / "last.json.gz").read_bytes()))
            self.assertTrue(any(e.get("cat") == "kernel" for e in trace["traceEvents"]))

            # A user writer can consume its temporary output before returning.
            # The capture must keep that native file and the alternate encoding.
            written = []

            def consume(prof):
                name = "callback.json.gz" if written else "callback.pftrace.gz"
                target = root / name
                prof.export_chrome_trace(str(target))
                prof.wait_for_exports()
                written.append(target.read_bytes())
                target.unlink()

            with serveutil.profile(root, activities=activities, calibration=False,
                                   schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=2),
                                   on_trace_ready=consume) as recorder:
                for step in range(2):
                    self.workload(step, "cuda")
                    recorder.step()
            capture = recorder.serveutil_directory
            entries = json.loads((capture / "capture.json").read_text())["traces"]
            self.assertEqual(len(entries), 2)
            for i, entry in enumerate(entries):
                self.assertEqual((capture / entry["file" if i else "native_binary"]).read_bytes(), written[i])
                for field in ("file", "native_binary"):
                    self.assertIn(f"batch-{i}".encode(), gzip.decompress((capture / entry[field]).read_bytes()))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available()
        and hasattr(torch.profiler.profile, "take_pending_cuspy_export"),
        "requires detached native CUDA exports",
    )
    def test_caller_can_finalize_detached_windows_in_reverse_order(self):
        from concurrent.futures import ThreadPoolExecutor

        if self.run_in_fresh_process():
            return
        cpu, cuda = torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
        config = torch.profiler.ProfilerActivityConfig(profiler_configs=[torch.profiler.CuspyConfig()])
        pending = []
        with tempfile.TemporaryDirectory() as tmp:
            def detach(prof):
                target = Path(tmp) / ("first.pftrace.gz" if not pending else "second.json.gz")
                prof.export_chrome_trace(str(target))
                pending.append((prof.take_pending_cuspy_export(), target))

            with serveutil.profile(tmp, activities=[cpu, {cuda: config}], calibration=False,
                    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=2),
                    experimental_config=torch._C._profiler._ExperimentalConfig(
                        custom_profiler_config='{"cuspy_async_export":true}'),
                    on_trace_ready=detach) as prof:
                for i in range(2):
                    self.workload(i, "cuda")
                    prof.step()
            capture = prof.serveutil_directory
            with ThreadPoolExecutor(1) as worker:
                for writer, target in reversed(pending):
                    self.assertIsNotNone(writer)
                    worker.submit(writer.join, force=False, timeout_s=0.2).result()
            manifest = json.loads((capture / "capture.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["traces"]), len(pending))
            self.assertTrue((capture / "report.json").is_file())
            for i, (record, (_, target)) in enumerate(zip(manifest["traces"], pending)):
                self.assertNotIn("pending", record)
                field = "file" if i else "native_binary"
                self.assertEqual((capture / record[field]).read_bytes(), target.read_bytes())
                for field in ("file", "native_binary"):
                    self.assertIn(f"batch-{i}".encode(), gzip.decompress((capture / record[field]).read_bytes()))

    def test_low_level_trace_windows_keep_native_callback_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = serveutil.profile(tmp, activities=[torch.profiler.ProfilerActivity.CPU],
                                  on_trace_ready=lambda _: self.fail("unscheduled callback"))
            runs = set()
            for step in range(2):
                p.prepare_trace()
                p.start_trace()
                self.workload(step, "cpu")
                p.stop_trace()
                capture = p.serveutil_directory
                runs.add(capture)
                manifest = json.loads((capture / "capture.json").read_text())
                self.assertEqual(manifest["status"], "complete")
                entry, = manifest["traces"]
                archived = gzip.decompress((capture / entry["file"]).read_bytes())
                target = Path(tmp) / "user.json"
                p.export_chrome_trace(str(target))
                self.assertEqual(target.read_bytes(), archived)
                self.assertIn(f"batch-{step}", {r.get("name") for r in json.loads(archived)["traceEvents"]})
            self.assertEqual(len(runs), 2)

    def test_manual_start_stop_default_and_wait_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = serveutil.profile(tmp, devices=[], calibration=False)
            self.assertIs(type(p), torch.profiler.profile)
            p.start()
            self.workload(0, "cpu")
            p.stop()
            self.assertTrue(p.events())
            p.export_chrome_trace(str(Path(tmp) / "manual.json"))
            self.assertTrue((p.serveutil_directory / "report.json").exists())
            with serveutil.profile(
                tmp,
                devices=[],
                calibration=False,
                schedule=torch.profiler.schedule(wait=3, warmup=1, active=2),
            ) as idle:
                torch.ones(2)
                idle.step()
            m = json.loads((idle.serveutil_directory / "capture.json").read_text(encoding="utf-8"))
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
                    (p.serveutil_directory / "capture.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["status"], "failed")


if __name__ == "__main__":
    unittest.main()
