# Install and download

## Browser viewer

Choose **Open viewer** in the navigation bar, then drop a trace or select a capture folder. Files are processed in your browser. You can start immediately with a measured example.

## Desktop app

Choose the build for your operating system and architecture below. Extract the complete archive. On Windows run **ServeUtil.exe**. On macOS open **ServeUtil.app**. On Linux run **ServeUtil** from the extracted folder. Open a JSON or gzip trace, a ServeUtil report, or a native Perfetto trace. The desktop window includes the offline viewer.

## Python CLI and model capture

Use Python 3.10 or later. The PyPI installation command for published versions is:

```bash
pip install serveutil
serveutil --help
```

Before the first PyPI release, or to use the current source:

```bash
python -m pip install "serveutil @ git+https://github.com/HaoyangPing0324/ServeUtil.git"
```

Use the model's existing PyTorch environment for recording. Install `serveutil[pm]` for independent CUPTI HBM sampling on compatible Linux devices and `serveutil[desktop]` to run the native window from Python. [First capture](start.md)

## Release history

[GitHub Releases](https://github.com/HaoyangPing0324/ServeUtil/releases) lists tagged versions and their desktop assets. The website also offers tested builds for its own revision. Compare the revision shown next to a download when reporting a problem.
