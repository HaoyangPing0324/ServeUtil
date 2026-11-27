"""Serve the independently packaged upstream Perfetto UI and one caller-selected report."""

import mimetypes
import secrets
import shutil
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit
from zipfile import ZipFile


def unpack_perfetto(directory):
    """Populate an HTTP static site with the pinned official distribution."""
    directory.mkdir(exist_ok=True, parents=True)
    with files("serveutil").joinpath("vendor/perfetto-ui.zip").open("rb") as src:
        with ZipFile(src) as archive:
            archive.extractall(directory)
    with files("serveutil").joinpath("vendor/Perfetto-LICENSE.txt").open("rb") as src:
        with (directory / "LICENSE.txt").open("wb") as dst:
            shutil.copyfileobj(src, dst)


@contextmanager
def viewer_server(document):
    """Expose only the report and packaged assets on a private loopback URL."""
    prefix = "/" + secrets.token_urlsafe(20) + "/"
    with (
        files("serveutil").joinpath("vendor/perfetto-ui.zip").open("rb") as src,
        ZipFile(src) as archive,
    ):
        members = set(archive.namelist())

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = urlsplit(self.path).path
                if path in (prefix, prefix + "index.html"):
                    f = document.open("rb")
                    length = document.stat().st_size
                    name = "index.html"
                elif path.startswith(prefix + "perfetto/"):
                    name = path.removeprefix(prefix + "perfetto/") or "index.html"
                    if name not in members:
                        self.send_error(404)
                        return
                    f = archive.open(name)
                    length = archive.getinfo(name).file_size
                else:
                    self.send_error(404)
                    return
                with f:
                    self.send_response(200)
                    content_type = (
                        {"wasm": "application/wasm", "js": "text/javascript"}.get(
                            name.rsplit(".", 1)[-1]
                        )
                        or mimetypes.guess_type(name)[0]
                        or "application/octet-stream"
                    )
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(length))
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.end_headers()
                    try:
                        shutil.copyfileobj(f, self.wfile)
                    except (ConnectionResetError, BrokenPipeError):
                        pass

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield f"http://127.0.0.1:{server.server_port}{prefix}"
            finally:
                server.shutdown()
                thread.join()
