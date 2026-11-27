# Locate a GPU gap during Decode

[Open the measured Qwen3 Decode report](../../../examples/h20-decode.html). This excerpt contains three consecutive Decode steps on one H20, selected from a run with a 16-token prompt and 16 generated tokens.

## Select the step

Start in **Stages & resources** and choose a Decode step under **Interval**. Follow the compute and copy tracks across the same time range. Click **Fit** to compare all three steps. With **Optimization opportunities**, the longest overlapping **Repeated GPU gap** appears first. Click it to zoom to the adjacent work.

## Read the evidence

A gap means that no recorded kernel, copy or memset occupied that device interval across the observed streams. Inspect the host launches and synchronization immediately before the next kernel. A missing upstream event can also leave a gap. In a complete recording, **Original tracks** lets you follow the CPU-to-GPU flow; this public excerpt contains only the selected events and hardware samples.

## Compare bandwidth and compute

Click the **HBM read** or **HBM write** card. Compare its covered time with the gap duration. A sample that spans the gap and adjacent kernels describes both. For a dense GEMM with input shapes and a unique kernel, switch to **GPU kernels** and inspect its effective BF16 rate. Record shapes when repeating this capture on your own model. [Record with PyTorch](../torch-profiler.md)
