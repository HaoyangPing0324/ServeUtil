# Install and download

| Task | What to get | Where it runs |
|---|---|---|
| Record a model or generate a report from a trace | [Python package on PyPI](https://pypi.org/project/serveutil/0.1.1/), installed with `pip install serveutil` | Measure in the inference environment or generate a report on another computer |
| Analyze existing reports in a desktop window | v0.1.1 desktop viewer: [Windows x86-64](https://github.com/HaoyangPing0324/ServeUtil/releases/download/v0.1.1/ServeUtil-Windows-AMD64.zip) · [macOS Apple silicon](https://github.com/HaoyangPing0324/ServeUtil/releases/download/v0.1.1/ServeUtil-Darwin-arm64.tar.gz) · [Linux x86-64](https://github.com/HaoyangPing0324/ServeUtil/releases/download/v0.1.1/ServeUtil-Linux-x86_64.tar.gz) | Extract on the viewing computer. The archive includes the Python runtime. |
| Inspect an existing report in the browser | [Open the web viewer](https://haoyangping0324.github.io/ServeUtil/viewer.html) | Files are processed in the browser. No installation is required. |

A **package** is software installed by a package manager. The ServeUtil Python package provides the `serveutil` command and the `serveutil.profile` API. A **release** is a published version tied to a Git tag. Its GitHub page provides release notes, source archives and platform downloads. The same version ships as a Python package on PyPI and as a desktop viewer on GitHub Releases.

## Browser viewer

Choose **Open viewer** in the navigation bar, then drop a trace or select a capture folder. Files are processed in your browser. You can start immediately with a measured example.

## Desktop app

The table above links to the fixed v0.1.1 archives. The builds below correspond to the website revision shown next to each file. Extract the complete archive. On Windows run **ServeUtil.exe**. On macOS open **ServeUtil.app**. On Linux run **ServeUtil** from the extracted folder. Open a JSON or gzip trace, a ServeUtil report, or a native Perfetto trace. The desktop window includes the offline viewer.

## Python CLI and model capture

Use Python 3.10 or later. The PyPI installation command for published versions is:

```bash
pip install serveutil
serveutil --help
```

To install the latest source from GitHub, run:

```bash
python -m pip install "serveutil @ git+https://github.com/HaoyangPing0324/ServeUtil.git"
```

Use the model's existing PyTorch environment for recording. Install `serveutil[pm]` for independent CUPTI HBM sampling on compatible Linux devices and `serveutil[desktop]` to run the native window from Python. [First capture](start.md)

## v0.1.1 requirements and tested versions

| Component | Requirements and verification |
|---|---|
| CLI and report generation | Python ≥3.10. pip installs `ijson>=3.2,<4`, `prometheus-client>=0.17,<1` and `nvidia-ml-py>=12.550.52`. Existing reports work on a CPU-only machine. |
| In-process PyTorch capture | Use the PyTorch installation in the model environment. CPU capture was tested on 2.10.0 and 2.15.0. H20 capture was tested on 2.13.0+cu130 and 2.15.0+cu132. GPU capture needs a driver compatible with the selected PyTorch build. |
| Model-server capture | Install the serving framework for your deployment and enable its PyTorch profiler. Single-host tests used vLLM 0.29.0 and SGLang 0.5.21. Two-host tests used vLLM 0.32.0 and SGLang 0.5.22. Install the same ServeUtil version on each collector. |
| Optional extras | Independent CUPTI PM sampling on Linux uses `serveutil[pm]` with `cupti-python>=13.4,<14` and requires device, driver and counter access. The Python desktop extra, `serveutil[desktop]`, installs `pywebview>=6,<7` with the Qt backend on Linux. For llmperfscope captures, `serveutil[llmperfscope]` installs `llmperfscope>=0.2,<1`. |
| Viewer | The web viewer uses a modern browser and was tested in Chromium. The desktop archives target the architectures listed above. The Linux build was tested on Ubuntu 24.04. Full-trace viewing bundles Perfetto UI and Trace Processor v59.1. |

The framework versions above identify configurations exercised in tests. To reproduce a setup, pin the package with `pip install serveutil==0.1.1` and select the PyTorch, CUDA and serving framework builds for that deployment. [Model validation](validation.md)

## Release history

[GitHub Releases](https://github.com/HaoyangPing0324/ServeUtil/releases) lists tagged versions and their desktop assets. The website also offers tested builds for its own revision. Compare the revision shown next to a download when reporting a problem.
