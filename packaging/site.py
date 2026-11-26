"""Build the static, offline-capable viewer and its measured examples."""

import argparse
import shutil
from pathlib import Path

from serveutil.report import render

EXAMPLES = [
    ("h20-decode.html", "Qwen3 decode on H20"),
    ("h20-allreduce.html", "SGLang AllReduce on two H20s"),
]


def build_site(destination):
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / "examples"
    for name, _ in EXAMPLES:
        if source.resolve() != destination.resolve():
            shutil.copyfile(source / name, destination / name)
    links = "".join(f'<a href="{name}">{label}</a>' for name, label in EXAMPLES)
    index = destination / "index.html"
    index.write_text(render().replace("<!-- EXAMPLES -->", links), encoding="utf-8")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path, default=Path("dist/site"))
    print(build_site(parser.parse_args().destination))
