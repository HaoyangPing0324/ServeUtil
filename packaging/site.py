"""Build the portable viewer and the English/Chinese documentation website."""

import argparse
import json
import shutil
from pathlib import Path

from serveutil.report import render, render_data
from serveutil.native_view import unpack_perfetto

EXAMPLES = [
    ("h20-decode.html", "Qwen3 decode on H20"),
    ("h20-allreduce.html", "SGLang AllReduce on two H20s"),
    ("native-example.html", "Full trace example (synthetic)"),
]


def build_site(destination):
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / "examples"
    site = source.resolve() != destination.resolve()
    guides = (
        '<a class="projectDocs" href="index.html">Project guides</a>'
        '<a class="repoIcon" href="https://github.com/HaoyangPing0324/ServeUtil" aria-label="ServeUtil on GitHub">'
        '<img width="20" height="20" src="assets/github.svg" alt=""></a>'
    )
    if site:
        for name, _ in EXAMPLES[:2]:
            html = (source / name).read_text(encoding="utf-8")
            (destination / name).write_text(html.replace("<!-- PROJECT GUIDES -->", guides), encoding="utf-8")
        unpack_perfetto(destination / "perfetto")
    native = destination / "native-example.html"
    native.write_text(render(source / "decode.synthetic.json").replace("<!-- PROJECT GUIDES -->", guides if site else ""), encoding="utf-8")
    links = "".join(f'<a href="{name}">{label}</a>' for name, label in EXAMPLES)
    viewer = destination / ("viewer.html" if site else "index.html")
    html = render().replace("<!-- EXAMPLES -->", links)
    viewer.write_text(html.replace("<!-- PROJECT GUIDES -->", guides if site else ""), encoding="utf-8")
    if site:
        from portal import build_docs
        for path in source.glob("*.report.json"):
            html = render_data(json.loads(path.read_text(encoding="utf-8")))
            (destination / path.name.replace(".report.json", ".html")).write_text(html.replace("<!-- PROJECT GUIDES -->", guides), encoding="utf-8")
        build_docs(destination)
    return destination / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path, default=Path("dist/site"))
    print(build_site(parser.parse_args().destination))
