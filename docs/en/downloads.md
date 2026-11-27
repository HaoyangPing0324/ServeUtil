## Install and record

Use Python 3.10 or later and the PyTorch installation from the model environment.

```bash
pip install serveutil
serveutil --help
```

| Component | Requirements and tested versions |
|---|---|
| CLI and report generation | Python ≥3.10. pip installs `ijson>=3.2,<4`, `prometheus-client>=0.17,<1` and `nvidia-ml-py>=12.550.52`. Existing reports work on a CPU-only machine. |
| In-process PyTorch capture | Use the PyTorch installation in the model environment. CPU capture was tested on 2.10.0 and 2.15.0. H20 capture was tested on 2.13.0+cu130 and 2.15.0+cu132. GPU capture needs a driver compatible with the selected PyTorch build. |
| Model-server capture | Install the serving framework for your deployment and enable its PyTorch profiler. Single-host tests used vLLM 0.29.0 and SGLang 0.5.21. Two-host tests used vLLM 0.32.0 and SGLang 0.5.22. Install the same ServeUtil version on each collector. |

Optional GPU counter dependencies — `serveutil[pm]` (Linux, cupti-python ≥13.4,<14), `serveutil[llmperfscope]` (llmperfscope ≥0.2,<1).

[Desktop downloads and offline installation](https://haoyangping0324.github.io/ServeUtil-Viewer/downloads.html)
