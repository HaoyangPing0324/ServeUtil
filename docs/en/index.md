# ServeUtil documentation

ServeUtil records PyTorch, vLLM and SGLang inference. Its timeline connects operators and GPU kernels with communication between ranks, hardware counter samples and measured bandwidth references.

## Install

Use Python 3.10 or later in the model environment.

```bash
pip install serveutil
```

For a desktop viewer or optional GPU counter dependencies, see [Install and download](downloads.md).

## View a trace

Open a trace you already have in the browser viewer, or run the following command locally.

```bash
serveutil view rank0.pt.trace.json.gz
```

[Open a Qwen3 Decode capture on H20](../../examples/h20-decode.html). Click a GPU slice to inspect its duration, then drag across tracks to compare work over a time interval. Press **F** to zoom to that selection.

## Record inference

For a Python model, follow the [PyTorch quick start](start.md). For a running service, [configure vLLM or SGLang](capture.md) and record one interval across its workers.
