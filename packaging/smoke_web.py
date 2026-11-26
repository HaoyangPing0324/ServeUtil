"""Open the static site, inspect a measured collective and import a local Kineto trace."""

import argparse
import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright


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
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="dist/site")
    verify(parser.parse_args().directory)
