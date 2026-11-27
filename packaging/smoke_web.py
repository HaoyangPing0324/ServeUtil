"""Open the static site, inspect a measured collective and import a local Kineto trace."""

import argparse
import functools
import gzip
import json
import tarfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright


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
    # TrackDescriptor and two TrackEvents emitted by Perfetto's text_to_binary.
    binary = bytes.fromhex(
        "0a1d400150016801e20314082a12104e617469766520475055207175657565"
        "0a1a40e80750015a134801582aba010c42696e61727920736c696365"
        "0a0b40a01f50015a044802582a"
    )
    page.locator("#file").set_input_files(
        {
            "name": "native.pftrace",
            "mimeType": "application/octet-stream",
            "buffer": binary,
        }
    )
    page.locator("#nativePanel").wait_for()
    verify_sql(
        page, "select name || ' lasted ' || dur from slice", "Binary slice lasted 3000"
    )


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
                page.goto(f"http://127.0.0.1:{server.server_port}/")
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
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                trace = (
                    Path(__file__).resolve().parents[1]
                    / "examples/decode.synthetic.json"
                )
                page.locator("#file").set_input_files(trace)
                page.get_by_role("heading", name="Imported trace timeline").wait_for()
                verify_full_workspace(page)
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="dist/site")
    verify(parser.parse_args().directory)
