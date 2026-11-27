"""A capture is an immutable directory of inputs with a rebuildable report."""

import gzip
import json
import os
import socket
import tempfile
import threading
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
    return (gzip.open if Path(path).suffix.lower() == ".gz" else open)(
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
        self._lock = threading.RLock()
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
        with self._lock:
            write_json(self.path / "capture.json", self.manifest.copy())

    def warn_export(self, path, error):
        with self._lock:
            self.manifest.setdefault("export_warnings", []).append(
                {"file": path.name, "error": str(error)}
            )

    def add_trace(self, path, rank=None, *, activities=None, **metadata):
        trace = {"file": str(path.relative_to(self.path)), "rank": rank, **metadata}
        if activities is not None:
            trace["activities"] = sorted(activities)
        with self._lock:
            self.manifest["traces"].append(trace)
            self.save()
        return trace

    def complete_export(self, entry, path, binary, error=None):
        """Publish a caller-owned deferred export when its native writer finishes."""
        with self._lock:
            entry.pop("pending", None)
            if error is not None:
                entry["export_error"] = str(error)
                self.manifest.update(status="failed", error=str(error))
            elif path.exists() and binary.exists():
                entry["native_binary"] = str(binary.relative_to(self.path))
            elif binary.exists():
                entry.update(file=str(binary.relative_to(self.path)), native_output=True)
            if self.manifest["status"] == "exporting":
                self._finish_exports()
            else:
                self.save()

    def _finish_exports(self):
        if self.manifest["status"] != "failed":
            pending = any(t.get("pending") for t in self.manifest["traces"])
            self.manifest["status"] = "exporting" if pending else "complete"
        self.save()
        if self.manifest["status"] == "complete" and self.manifest["traces"]:
            from .report import make_report

            write_json(self.path / "report.json", make_report(self.path))

    def finish(self, error=None):
        with self._lock:
            self.manifest["ended_ns"] = time.time_ns()
            if error is not None:
                self.manifest.update(status="failed", error=str(error))
            self._finish_exports()
