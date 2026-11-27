# Your first inference report

## Open a trace you already have

Open the browser viewer and drop a PyTorch trace, or select **Open capture folder** for a multi-worker run. A generated report includes the original capture's hardware samples and calibration when available. The browser reads your selected files locally. Use the desktop app for offline work. [Install or download](downloads.md)

## Capture a small workload

In an environment with PyTorch installed, run this file. It works on CPU and uses the current CUDA device when available.

```python
import torch
import serveutil

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(256, 256, device=device)
with serveutil.profile("runs", record_shapes=True) as p:
    with torch.profiler.record_function("Projection"):
        y = x @ x
    p.step()
print(p.serveutil_directory)
```

Open the printed directory in ServeUtil Viewer. Use **Stages & resources** to locate Projection and **GPU kernels** to inspect device work. CPU-only captures have CPU activity. CUDA captures can add hardware counters. Select a card to see which resources were measured in that environment. [PyTorch options](torch-profiler.md)

## Profile a live model

Enable the engine's torch profiler when starting vLLM or SGLang, then point the collector at its endpoint and trace directory. The collector uses the framework API to start and stop profiling in the model workers.

```bash
serveutil capture --engine vllm --url http://localhost:8000 \
  --trace-dir /profiles --seconds 5 --generate "Explain prefill and decode."
```

For existing traffic omit `--generate`. For multi-node or PD deployments use `--topology deployment.json`. [Configure service and worker paths](capture.md)

## Read a measured capture

Start with the [Decode walkthrough](https://haoyangping0324.github.io/ServeUtil-Viewer/decode.html), [eRDMA reference comparison](https://haoyangping0324.github.io/ServeUtil-Viewer/bandwidth.html) or [one KV handoff](https://haoyangping0324.github.io/ServeUtil-Viewer/pd.html). Each tutorial opens its own measured example.
