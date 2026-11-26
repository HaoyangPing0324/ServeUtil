"""A capture is an immutable directory of inputs with a rebuildable report."""

import gzip
import json
import os
import socket
import tempfile
import time
import uuid
from pathlib import Path


def write_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temp = Path(stream.name)
        try:
            json.dump(
                data, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            stream.flush()
            stream.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def open_text(path):
    return (gzip.open if str(path).endswith(".gz") else open)(
        path, "rt", encoding="utf-8"
    )


class Capture:
    def __init__(self, output, engine):
        parent = Path(output)
        parent.mkdir(parents=True, exist_ok=True)
        self.path = parent / (
            time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        self.path.mkdir()
        (self.path / "traces").mkdir()
        self.manifest = dict(
            schema="serveutil.capture/1",
            engine=engine,
            host=socket.gethostname(),
            started_ns=time.time_ns(),
            status="recording",
            traces=[],
        )
        self.save()

    def save(self):
        write_json(self.path / "capture.json", self.manifest)

    def add_trace(self, path, rank=None):
        self.manifest["traces"].append(
            {"file": str(path.relative_to(self.path)), "rank": rank}
        )
        self.save()

    def finish(self, error=None):
        self.manifest["ended_ns"] = time.time_ns()
        self.manifest["status"] = "failed" if error else "complete"
        if error:
            self.manifest["error"] = str(error)
        self.save()
        if error is None:
            from .report import render

            render(self.path, self.path / "report.html")
