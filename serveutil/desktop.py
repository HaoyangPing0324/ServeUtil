"""Open a trace in a browser or a native application window."""

import argparse
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

from .report import render


def _verify(window):
    # The native load event precedes decompression and drawing for large reports.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if window.evaluate_js("!document.getElementById('png').disabled"):
            return window.evaluate_js("""
                document.getElementById('view').value = 'operator';
                document.getElementById('view').dispatchEvent(new Event('change'));
                document.querySelector('#summary').textContent.includes('trace(s)') &&
                !document.getElementById('depthControl').hidden && document.querySelector('canvas').width > 0;
            """)
        time.sleep(0.1)
    return False


def _desktop(path, verify):
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError(
            "Install the desktop extra with pip install 'serveutil[desktop]'"
        ) from exc
    with tempfile.TemporaryDirectory(prefix="serveutil-viewer-") as tmp:
        document = Path(tmp) / "report.html"
        if path and path.is_dir() and (path / "report.html").is_file():
            path = path / "report.html"
        render(path, document)
        window = webview.create_window(
            "ServeUtil", url=document.as_uri(), width=1340, height=860
        )
        if not verify:
            webview.start(private_mode=True)
            return
        passed = []

        def inspect():
            try:
                passed.append(_verify(window))
            finally:
                window.destroy()

        window.events.loaded += inspect
        timeout = threading.Timer(85, window.destroy)
        timeout.start()
        try:
            webview.start(private_mode=True)
        finally:
            timeout.cancel()
        if passed != [True]:
            raise RuntimeError("The desktop runtime could not render the report")


def view(path=None, desktop=False, verify=False):
    path = Path(path).resolve() if path else None
    if desktop:
        return _desktop(path, verify)
    if path and path.is_file() and path.suffix.lower() == ".html":
        document = path
    elif path and path.is_dir():
        document = path / "report.html"
        if not document.is_file():
            render(path, document)
    else:
        if path:
            document = path.with_name(path.name + ".html")
        else:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as stream:
                document = Path(stream.name)
        render(path, document)
    if not webbrowser.open(document.as_uri()):
        print(document)


def main():
    parser = argparse.ArgumentParser(
        description="Open an inference trace in the ServeUtil desktop viewer."
    )
    parser.add_argument("path", nargs="?")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="open the report, check the native renderer, then exit",
    )
    args = parser.parse_args()
    view(args.path, desktop=True, verify=args.verify)
