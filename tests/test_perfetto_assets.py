import base64
import hashlib
import json
import runpy
import unittest
from pathlib import Path
from zipfile import ZipFile


class EmbeddedPerfettoTests(unittest.TestCase):
    def test_every_pinned_upstream_resource_is_present_and_unmodified(self):
        root = Path(__file__).resolve().parents[1]
        release = runpy.run_path(str(root / "packaging" / "vendor_perfetto.py"))
        with ZipFile(release["OUTPUT"]) as archive:
            manifest = archive.read("manifest.json")
            self.assertEqual(
                hashlib.sha256(manifest).hexdigest(), release["MANIFEST_SHA"]
            )
            self.assertEqual(
                hashlib.sha256(archive.read("index.html")).hexdigest(),
                release["INDEX_SHA"],
            )
            resources = json.loads(manifest)["resources"]
            self.assertEqual(
                set(archive.namelist()),
                set(resources) | {"index.html", "manifest.json"},
            )
            for name, sri in resources.items():
                with self.subTest(resource=name):
                    digest = hashlib.sha256(archive.read(name)).digest()
                    self.assertEqual(
                        "sha256-" + base64.b64encode(digest).decode("ascii"), sri
                    )
