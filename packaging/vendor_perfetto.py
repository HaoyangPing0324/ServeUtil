"""Rebuild the offline Perfetto UI from the pinned upstream distribution."""

import base64
import hashlib
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VERSION = "v59.1-491832c0d"
BASE = f"https://ui.perfetto.dev/{VERSION}/"
MANIFEST_SHA = "5b7c5eea44cd954dc5338f302bb347d3e8c866ac32c762b266e81a57b632258d"
INDEX_SHA = "33ae569b0cfd7ed66b2be9f649de4aaadae0dd42177643a552460a6225d2da99"
OUTPUT = (
    Path(__file__).resolve().parents[1] / "serveutil" / "vendor" / "perfetto-ui.zip"
)


def fetch(path, digest):
    for retry in range(3):
        try:
            with urllib.request.urlopen(BASE + path, timeout=120) as stream:
                data = stream.read()
            break
        except OSError:
            if retry == 2:
                raise
            time.sleep(2**retry)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected SHA-256 for {path}")
    return data


def main():
    manifest = fetch("manifest.json", MANIFEST_SHA)
    index = fetch("index.html", INDEX_SHA)
    entries = sorted(json.loads(manifest)["resources"].items())

    def asset(item):
        name, sri = item
        digest = base64.b64decode(sri.removeprefix("sha256-")).hex()
        return name, fetch(name, digest)

    OUTPUT.parent.mkdir(exist_ok=True)
    tmp = OUTPUT.with_suffix(".tmp")
    try:
        with (
            ThreadPoolExecutor(max_workers=8) as pool,
            zipfile.ZipFile(tmp, "w") as archive,
        ):
            for name, data in [
                ("index.html", index),
                ("manifest.json", manifest),
                *pool.map(asset, entries),
            ]:
                entry = zipfile.ZipInfo(name)
                entry.external_attr = 0o100644 << 16
                archive.writestr(
                    entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6
                )
        tmp.replace(OUTPUT)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
