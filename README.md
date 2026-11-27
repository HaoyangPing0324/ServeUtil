<p align="center"><img src="assets/wordmark.png" width="360" alt="ServeUtil"></p>
<h1 align="center">Compute and bandwidth profiling for LLM inference</h1>
<p align="center"><strong><a href="https://haoyangping0324.github.io/ServeUtil/">Documentation</a> &nbsp;|&nbsp; <a href="https://haoyangping0324.github.io/ServeUtil/viewer.html">Trace viewer</a> &nbsp;|&nbsp; <a href="https://haoyangping0324.github.io/ServeUtil/downloads.html">Downloads</a> &nbsp;|&nbsp; <a href="README.zh-CN.md">简体中文</a></strong></p>
<p align="center">
  <a href="https://github.com/HaoyangPing0324/ServeUtil/actions/workflows/desktop.yml"><img alt="Desktop and web CI" src="https://github.com/HaoyangPing0324/ServeUtil/actions/workflows/desktop.yml/badge.svg"></a>
  <a href="#installation"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB"></a>
  <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-337b57"></a>
  <a href="https://haoyangping0324.github.io/ServeUtil/capture.html"><img alt="PyTorch · vLLM · SGLang" src="https://img.shields.io/badge/inference-PyTorch%20%C2%B7%20vLLM%20%C2%B7%20SGLang-14695c"></a>
</p>

ServeUtil combines PyTorch, vLLM and SGLang traces with GPU and network counters. Select a Decode interval to see which ranks were computing, which were waiting on communication and how the measured bandwidth compares with a reference from the same hardware.

<img src="assets/h20-decode.png" width="100%" alt="Qwen3 Decode on H20 with GPU gaps, HBM read and write rates, Tensor activity and BF16 GEMM throughput">

## What ServeUtil adds

| Starting point | What you can inspect |
|---|---|
| PyTorch operators, kernels and copies | Device-level HBM reads and writes, PCIe and NVLink traffic, network and RDMA port throughput, and SM and Tensor pipeline activity where supported. The original trace is preserved. |
| An absolute rate such as 58 MB/s | Compare with a microbenchmark on the same device, port and direction. Dense GEMMs with recorded shapes and one correlated kernel can also show effective TFLOP/s. |
| A deeply nested, multi-rank timeline | Start with Prefill and Decode. Focus on a node, rank, operator depth or leaf. Follow matched Send/Recv operations or inspect KV handoffs between Prefill and Decode. Open Full Perfetto for original events, flows and SQL. |
| A slow interval | Jump to repeated GPU gaps, communication without overlapping compute, sustained bandwidth pressure or comparable steps with different rank durations. Each marker gives its evidence. |

On two H20 hosts, the sending port averaged 58.5 MB/s over a 3.05-second SGLang TP2 window, about 0.51% of its directional eRDMA reference. Of 4,469 complete AllReduce operations, 3,956 carried only 2 KiB. The message sizes and time between operations help explain the low window average. [Hardware validation and model results](docs/en/validation.md)

## Installation

Use Python 3.10 or later. Install a published release with the first command. Before the first PyPI release, install from a checkout with the second command.

```bash
pip install serveutil
# From a source checkout:
python -m pip install .
```

The base package includes the CLI, the offline-capable web viewer and NVIDIA NVML bindings. Viewing existing traces works on a CPU-only computer. Model capture uses the PyTorch installation in your inference environment. For independent directional HBM sampling on compatible Linux systems, install `pip install "serveutil[pm]"` and enable GPU counter access. For the desktop window, install `pip install "serveutil[desktop]"` or download a platform build. [Install and download](https://haoyangping0324.github.io/ServeUtil/downloads.html)

## Open a trace

```bash
serveutil view rank0.pt.trace.json.gz
serveutil report rank0.pt.trace.json.gz -o report.html
```

The first command opens the viewer. The second creates a shareable HTML report. Import a capture folder to recover rank metadata, hardware samples and clock calibration. [Viewer guide](docs/en/viewer.md)

## Profile your own PyTorch code

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

print(prof.serveutil_directory / "report.html")
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

ServeUtil warms up the service, generates requests, starts and stops profiling in its workers, samples hardware and prints the report path. Omit `--generate` to capture existing traffic. Add `--calibrate` on idle test devices or pass an existing reference with `--calibration calibration.json`. Use a topology file for multi-node and disaggregated serving. [Deployment guide](docs/en/capture.md) · [Multi-node topology](examples/distributed-topology.json) · [PD topology](examples/pd-topology.json)

## Guides and examples

- Explore two measured reports. [Qwen3 Decode](https://haoyangping0324.github.io/ServeUtil/h20-decode.html) · [SGLang AllReduce](https://haoyangping0324.github.io/ServeUtil/h20-allreduce.html)
- Check each metric's units, denominator and scope. [Metrics](docs/en/metrics.md)
- Reuse nccl-tests, nvbandwidth, perftest, HTA, TraceLens or llmperfscope. [Tools and integrations](docs/en/tools.md)

ServeUtil is licensed under Apache 2.0. Bundled dependencies retain their upstream notices. [License](LICENSE) · [Third-party notices](NOTICE)
