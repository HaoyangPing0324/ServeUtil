<p align="center"><img src="assets/wordmark.png" width="360" alt="ServeUtil"></p>
<h1 align="center">Compute and bandwidth profiling for LLM inference</h1>
<p align="center"><strong><a href="https://haoyangping0324.github.io/ServeUtil/">Documentation</a> &nbsp;|&nbsp; <a href="https://haoyangping0324.github.io/ServeUtil-Viewer/viewer.html">Trace viewer</a> &nbsp;|&nbsp; <a href="https://haoyangping0324.github.io/ServeUtil/downloads.html">Installation</a> &nbsp;|&nbsp; <a href="README.zh-CN.md">简体中文</a></strong></p>
<p align="center">
  <a href="https://github.com/HaoyangPing0324/ServeUtil/actions/workflows/collector.yml"><img alt="Collector CI" src="https://github.com/HaoyangPing0324/ServeUtil/actions/workflows/collector.yml/badge.svg"></a>
  <a href="#install-and-record"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB"></a>
  <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-337b57"></a>
  <a href="https://haoyangping0324.github.io/ServeUtil/capture.html"><img alt="PyTorch · vLLM · SGLang" src="assets/engines.svg"></a>
</p>

ServeUtil collects original PyTorch traces, GPU counters and network traffic during the same inference window. It supports PyTorch, vLLM and SGLang, including tensor parallelism and disaggregated prefill/decode. A measured reference for the same device, direction and port gives both the absolute rate and its utilization.

On two H20 hosts, the sending port averaged 58.5 MB/s over a 3.05-second SGLang TP2 window, about 0.51% of its directional eRDMA reference. Of 4,469 complete AllReduce operations, 3,956 carried only 2 KiB. The message sizes and time between operations help explain the low window average. [Hardware validation and model results](docs/en/validation.md)

## Install and record

Use Python 3.10 or later and the PyTorch installation from the model environment.

```bash
pip install serveutil
```

```python
import torch
import serveutil

device = "cuda" if torch.cuda.is_available() else "cpu"
projection = torch.nn.Linear(256, 256, bias=False).to(device).eval()
x = torch.ones(32, 256, device=device)

with torch.inference_mode(), serveutil.profile("runs", record_shapes=True) as prof:
    with torch.profiler.record_function("Projection"):
        projection(x)
    prof.step()

print(prof.serveutil_directory)
```

The returned object is a native PyTorch profiler. The first CUDA run calibrates the selected device by default and later runs reuse the cache. Integrations using an existing schedule, TensorBoard callback or asynchronous writer can use `instrument` or `observe`. [PyTorch integration](docs/en/torch-profiler.md)

## Capture a running service

Enable the framework's PyTorch profiler when starting the service. For vLLM add `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/profiles"}'`. For SGLang set `SGLANG_TORCH_PROFILER_DIR=/profiles`. The collector needs access to the worker's trace directory. Run the command for your engine:

```bash
serveutil capture --engine vllm --url http://localhost:8000 \
  --trace-dir /profiles --seconds 5 --generate "Explain prefill and decode."

serveutil capture --engine sglang --url http://localhost:30000 \
  --trace-dir /profiles --seconds 5 --generate "Explain prefill and decode."
```

ServeUtil warms up the service, generates requests, starts and stops profiling in its workers, samples hardware and prints the capture directory. Omit `--generate` to capture existing traffic. Add `--calibrate` on idle test devices or pass an existing reference with `--calibration calibration.json`. Use a topology file for multi-node and disaggregated serving. [Deployment guide](docs/en/capture.md) · [Multi-node topology](examples/distributed-topology.json) · [PD topology](examples/pd-topology.json)

In [ServeUtil Viewer](https://haoyangping0324.github.io/ServeUtil-Viewer/), choose **Open capture folder** and select the printed directory to load the original traces alongside measurements and calibration. [Collector documentation](https://haoyangping0324.github.io/ServeUtil/) · [Viewer controls](https://haoyangping0324.github.io/ServeUtil-Viewer/viewer-guide.html) · [Viewer repository](https://github.com/HaoyangPing0324/ServeUtil-Viewer)
