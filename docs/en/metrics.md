# Read hardware metrics and references

Drag over an interval, switch to **Hardware counters** and click a resource card. Its selection mean weights each sample by the portion of its interval inside the selection. Two adjacent short steps can share one device sample.

| Resource | Measurement and interpretation |
|---|---|
| HBM reads and writes | CUPTI PM directional throughput in decimal GB/s, subject to hardware, driver and counter permissions. The estimated total uses NVML DRAM activity and device memory parameters. |
| SM and Tensor pipelines | Device activity in percent over each sample interval. Supported instructions and pipeline counters vary by GPU. |
| GEMM | For dense mm, bmm and addmm with shapes and one correlated GPU kernel, count a multiply-add as two operations. Supported same-input-precision integer and scaled variants use their recorded dtype. Rates are TFLOP/s or TOP/s. Selection totals use complete kernels and their summed durations. |
| GPU-to-GPU | NVLink TX/RX measures device link traffic. NCCL algorithm bandwidth uses logical bytes and operation duration. Bus bandwidth uses the nccl-tests formula where defined. |
| CPU-to-GPU | PCIe TX/RX covers the GPU's PCIe traffic. HtoD and DtoH events with known sizes also provide effective payload rates. |
| Node network | Differences in eRDMA, IB or Linux network-interface byte counters yield directional port rates. Different layers can count the same bytes. |
| KV transfer | SGLang can associate scheduler spans and unique engine observations with requests. vLLM NIXL counters describe endpoint and measurement-window totals. |

## Calibrated utilization

A 100% reading means the selection reached the measured reference. Rates can exceed a microbenchmark. Match the metric, source, device and units, as well as node, port and direction for network rates. For example, 58.508 MB/s divided by a TX reference of 11.4658 GB/s is approximately 0.5103%. The card includes the reference workload. A GEMM reference also includes matrix shape and precision.

Use nvbandwidth for transfer pairs, nccl-tests for collectives and perftest for RDMA. AllGather uses its full output size, ReduceScatter its full input size. A multi-kernel collective counts its payload once, from its first kernel start to its last end. Confirm group members, layout and message size before interpreting a collective percentage. [nccl-tests bandwidth definitions](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

## Clocks and optimization markers

The collector measures host offsets before and after recording. The details retain round-trip uncertainty and drift. Compare cross-host timestamps only within the accuracy of those estimates. GPU gaps use the union of observed activity across streams on a device. Exposed communication subtracts overlapping compute from communication intervals. Bandwidth pressure requires consecutive samples and a comparable baseline. Rank-duration comparisons require the same identifiable logical step. A marker gives a place to investigate the workload and its dependencies.

## Additional analysis

Run `serveutil report TRACE --analysis hta -o report.html` or use `--analysis tracelens` with the corresponding analyzer installed. Details preserve the upstream per-rank fields. See [tools and integrations](tools.md).
