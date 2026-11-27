# ServeUtil

Profile PyTorch, vLLM and SGLang inference with GPU and network counters. Inspect Prefill and Decode by rank, select an interval for directional bandwidth and compare it with a measured hardware reference.

```bash
pip install serveutil
serveutil view rank0.pt.trace.json.gz
```

Create a shareable report with `serveutil report TRACE -o report.html`. Model capture uses the PyTorch installation in your serving environment. With a running vLLM server configured to write profiler traces to a shared directory:

```bash
serveutil capture --engine vllm --url http://localhost:8000 \
  --trace-dir /profiles --seconds 5 --generate "Explain prefill and decode."
```

Use `--engine sglang` and the SGLang service URL for SGLang. Add `--calibrate` on idle test devices to measure a local reference. For independent HBM read/write sampling on compatible Linux devices install `serveutil[pm]`.

[Documentation](https://haoyangping0324.github.io/ServeUtil/) · [简体中文](https://haoyangping0324.github.io/ServeUtil/zh/) · [Desktop downloads](https://haoyangping0324.github.io/ServeUtil/downloads.html)
