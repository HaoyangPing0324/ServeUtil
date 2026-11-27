"""Open the static site, inspect a measured collective and import a local Kineto trace."""

import argparse
import functools
import gzip
import json
import tarfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import sync_playwright
from native_trace import BINARY_TRACE


def verify_sql(page, sql, expected):
    frame = page.frame_locator("#nativeFrame iframe")
    page.wait_for_function(
        """() => document.querySelector("#nativeFrame iframe")?.contentDocument?.body?.innerText.includes("CURRENT TRACE")""",
        timeout=90000,
    )
    frame.locator("canvas").first.wait_for(timeout=90000)
    frame.locator("a[href='#!/query']").evaluate("link => link.click()")
    editor = frame.locator(".cm-content")
    editor.fill(sql)
    frame.get_by_role("button", name="Run Query").click()
    frame.get_by_text(expected, exact=True).wait_for()


def verify_full_workspace(page):
    """Query two compressed workers with overlapping process IDs using upstream SQL."""
    originals = []
    for i in range(2):
        trace = {
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "custom",
                    "name": f"native-worker-{i}",
                    "ts": 1000,
                    "dur": 50,
                    "pid": 42,
                    "tid": 42,
                    "args": {"worker": i},
                }
            ]
        }
        raw = json.dumps(trace).encode()
        originals.append(
            {
                "name": f"worker-{i}.json.gz",
                "mimeType": "application/gzip",
                "buffer": gzip.compress(raw, mtime=0),
            }
        )
    page.route("https://**", lambda route: route.abort())
    page.locator("#file").set_input_files(originals)
    page.locator("#nativePanel").wait_for()
    verify_sql(
        page,
        "select 'verified ' || count(*) || ' slices across ' || "
        "(select count(distinct machine_id) from process) || ' workers' "
        "from slice where name glob 'native-worker-*'",
        "verified 2 slices across 2 workers",
    )
    with page.expect_download() as capture:
        page.locator("#saveOriginal").click()
    with tarfile.open(capture.value.path()) as archive:
        manifest = json.load(archive.extractfile("perfetto_manifest.json"))
        for entry, original in zip(
            manifest["perfetto_manifest"]["files"], originals, strict=True
        ):
            assert archive.extractfile(entry["path"]).read() == gzip.decompress(
                original["buffer"]
            )
    page.locator("#leaveNative").click()
    page.locator("#file").set_input_files(
        {
            "name": "native.pftrace",
            "mimeType": "application/octet-stream",
            "buffer": BINARY_TRACE,
        }
    )
    page.locator("#nativePanel").wait_for()
    verify_sql(
        page, "select name || ' lasted ' || dur from slice", "Binary slice lasted 3000"
    )


def verify_capture_cycles(page):
    # Use only the raw folder, with no precomputed report.
    base = 1_760_000_000_000_000_001
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        entries = []
        for i, rank in enumerate((0, 0, 1)):
            name = f"cycle-{i}.JSON.GZ"
            trace = {
                "baseTimeNanoseconds": base + 10_000 * i,
                "traceEvents": [
                    {
                        "name": "bytes",
                        "ph": "C",
                        "pid": 1,
                        "tid": 1,
                        "ts": 20,
                        "args": {"value": i},
                    }
                ],
            }
            (root / name).write_bytes(gzip.compress(json.dumps(trace).encode()))
            entries.append({"file": name, "rank": rank, "node": "host"})
        (root / "aux.json").write_text(
            json.dumps({"baseTimeNanoseconds": base, "traceEvents": []})
        )
        entries.append(
            {"file": "aux.json", "rank": 2, "node": "host", "auxiliary": True}
        )
        manifest = {
            "engine": "torch",
            "status": "complete",
            "traces": entries,
            "nodes": [
                {"name": "host", "clock": {"offset_ns": 100, "uncertainty_ns": 10}}
            ],
        }

        def open_folder():
            (root / "capture.json").write_text(json.dumps(manifest))
            page.goto(page.url)
            page.locator("#captureFolder").set_input_files(root)
            page.wait_for_function(
                "report?.engine === 'torch' && report.traces.length === 3"
            )

        open_folder()
        assert page.evaluate("report.time_origin_unix_ns") == str(base + 20_100)
        assert page.evaluate("report.traces.map(t => t.rank_origin_us)") == [0, 10, 20]
        assert page.evaluate("report.traces.map(t => t.rank_origin_unix_ns)") == [
            str(base + 20_000 + i * 10_000) for i in range(3)
        ]
        assert page.locator("#rank option").count() == 3
        assert page.evaluate("rows.filter(r=>r.kind==='counter').map(r=>r.points)") == [
            [[0, 0], [10, 1]],
            [[20, 2]],
        ]
        page.locator("#node").select_option("host")
        page.locator("#fullTrace").click()
        with page.expect_download() as download:
            page.locator("#saveOriginal").click()
        with tarfile.open(download.value.path()) as archive:
            files = json.load(archive.extractfile("perfetto_manifest.json"))[
                "perfetto_manifest"
            ]["files"]
            assert len(files) == len(entries)
            for member, entry in zip(files, entries, strict=True):
                raw = (root / entry["file"]).read_bytes()
                assert archive.extractfile(member["path"]).read() == (
                    gzip.decompress(raw) if entry["file"].endswith(".GZ") else raw
                )
        page.locator("#leaveNative").click()
        page.locator("#rank").select_option("0")
        page.locator("#fullTrace").click()
        assert (
            page.locator("#nativeSources option:checked").inner_text()
            == "0 · all cycles"
        )
        entries[2]["node"] = "uncalibrated"
        open_folder()
        assert not page.evaluate("report.aligned")
        assert page.locator("#clock option[value=utc]").count() == 0
        assert page.locator("#clock option[value=local_utc]").count() == 1


def verify_device_finding_scope(page):
    hardware = dict(
        device="GPU-a",
        name="HBM read",
        metric="hbm.read",
        source="fixture",
        scope="device",
        unit="GB/s",
        points=[[10, 95, 0], [20, 94, 10], [30, 93, 20]],
        baseline={"value": 100},
    )
    from serveutil.analysis import saturation

    findings = saturation([hardware])
    assert len(findings) == 1
    report = dict(
        schema="serveutil.report/1",
        engine="scope test",
        aligned=True,
        traces=[
            dict(
                rank=str(i),
                file=str(i),
                devices=[uuid],
                counters=[],
                events=[
                    dict(
                        kind="compute",
                        device="0",
                        lane="0",
                        name="matmul",
                        ts=0,
                        dur=30,
                    )
                ],
            )
            for i, uuid in enumerate(("GPU-a", "GPU-b"))
        ],
        hardware=[hardware],
        telemetry={},
        findings=findings,
    )
    page.locator("#file").set_input_files(
        {
            "name": "report.json",
            "mimeType": "application/json",
            "buffer": json.dumps(report).encode(),
        }
    )
    page.wait_for_function("report.engine === 'scope test'")
    for rank, count in (("", 1), ("0", 1), ("1", 0)):
        page.locator("#rank").select_option(rank)
        assert page.locator("#issues button").count() == count

    page.locator("#rank").select_option("0")
    assert "GPU-a" in page.locator("#issues button").inner_text()
    # All 45 observations remain reachable even when they share a label.
    report["findings"] = [dict(name="Repeated GPU gap", rank="0", device="0",
                                ts=0, dur=i / 2) for i in range(1, 46)]
    report["engine"] = "repeated gaps"
    page.locator("#file").set_input_files({
        "name": "repeated.json", "mimeType": "application/json",
        "buffer": json.dumps(report).encode(),
    })
    page.wait_for_function("report.engine === 'repeated gaps'")
    assert page.locator("#issues button").count() == 1
    assert "22.5 µs" in page.locator("#issues button").inner_text()
    page.get_by_text("Other intervals (44)").click()
    page.wait_for_function("document.querySelectorAll('#issues button').length === 45")
    page.locator("#issues button").last.click()
    assert "Repeated GPU gap" in page.locator("#details").inner_text()


def verify(site):
    with ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(SimpleHTTPRequestHandler, directory=site)
    ) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as driver:
                browser = driver.chromium.launch()
                page = browser.new_page(viewport={"width": 1400, "height": 900})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                root = f"http://127.0.0.1:{server.server_port}/"
                page.goto(root)
                assert page.locator("html").get_attribute("lang") == "en"
                page.locator('a.card[href="bandwidth.html"]').click()
                page.locator("a.language").click()
                assert page.url.endswith("/zh/bandwidth.html")
                assert "0.51%" in page.locator("article").inner_text()
                page.locator('article a[href*="sglang-erdma.html"]').click()
                page.wait_for_function("document.documentElement.lang === 'zh-CN'")
                assert "选区资源" in page.locator("main").inner_text()
                page.locator("#viewerLanguage").click()
                assert page.locator("html").get_attribute("lang") == "en"
                page.goto(root + "zh/pd.html")
                page.locator("a.language").click()
                assert page.url.endswith("/pd.html")
                page.locator('article a[href*="sglang-pd.html"]').click()
                page.locator("#kvTable button").click()
                assert "7225344" in page.locator("#details").inner_text()
                rank = page.locator("#rank option").last.get_attribute("value")
                page.locator("#rank").select_option(rank)
                page.locator("#viewerLanguage").click()
                assert page.locator("#rank").input_value() == rank
                page.goto(root + "zh/downloads.html")
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.locator("a.language").click()
                assert page.url.endswith("/downloads.html")
                page.set_viewport_size({"width": 1400, "height": 900})
                page.goto(f"http://127.0.0.1:{server.server_port}/viewer.html")
                assert not page.locator("#fullTrace").is_visible()
                page.get_by_role("link", name="SGLang AllReduce on two H20s").click()
                card = page.get_by_role(
                    "button", name="Rank 0 · all_reduce", exact=False
                ).first
                card.click()
                assert "Logical bytes" in page.locator("#details").inner_text()
                # Center a visible NCCL slice and exercise the actual canvas hit test.
                point = page.evaluate("""() => {
                    const candidates = targets.flatMap(row => (row.visible || [])
                        .filter(v => v.e.collective?.reference_pct != null)
                        .map(v => ({x: (v.a + v.b) / 2, y: row.y + 15, width: v.b - v.a})));
                    const item = candidates.sort((a, b) => b.width - a.width)[0];
                    return [item.x, item.y];
                }""")
                page.locator("canvas").click(position={"x": point[0], "y": point[1]})
                assert "Relative to nccl-tests" in page.locator("#details").inner_text()
                page.goto(f"http://127.0.0.1:{server.server_port}/viewer.html")
                trace = (
                    Path(__file__).resolve().parents[1]
                    / "examples/decode.synthetic.json"
                )
                page.locator("#file").set_input_files(trace)
                page.get_by_role("heading", name="Imported trace timeline").wait_for()
                verify_full_workspace(page)
                verify_capture_cycles(page)
                verify_device_finding_scope(page)
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="dist/site")
    verify(parser.parse_args().directory)
