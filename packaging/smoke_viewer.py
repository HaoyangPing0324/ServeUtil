"""Open an actual native webview and check rendering of a known trace."""
from pathlib import Path
from tempfile import TemporaryDirectory
from serveutil.desktop import view
from serveutil.report import render

if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        doc = Path(tmp)/"example.html"
        render(Path(__file__).resolve().parents[1]/"examples/decode.synthetic.json", doc)
        view(doc, desktop=True, verify=True)
