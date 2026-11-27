# Model validation and capture overhead

The examples use BF16 Qwen3-0.6B on NVIDIA H20. Absolute rates depend on message size, concurrency, topology and the window selected.

## Two physical hosts

One GPU per host served real inference requests. SGLang 0.5.22 used TP2, vLLM 0.32.0 used PP2, and each framework was also run with separate Prefill and Decode services. The TCP configurations used the physical network directly. Independent samplers on both hosts read cumulative bytes around the same GPU activity window.

| TCP configuration | Sender GB/s | Receiver GB/s | Sender utilization · report / independent |
|---|---:|---:|---:|
| vLLM PP2 | 0.03283 | 0.03410 | 1.281% / 1.281% |
| vLLM PD | 0.31408 | 0.32203 | 12.253% / 12.248% |
| SGLang TP2 | 0.03395 | 0.03554 | 1.325% / 1.327% |
| SGLang PD | 0.02869 | 0.03163 | 1.119% / 1.210% |

The reference rates for these directions were 2.5633 and 2.6557 GB/s. Bursty KV traffic is sensitive to where the sampling interval crosses the selection boundary. Complete interval byte counts fell within the independent bounds. The vLLM PD engine reported 1,027,604,480 logical bytes and zero failed transfers in the repeat run.

## NCCL over eRDMA

Both NCCL model configurations recorded NET/IB with GDRDMA at both ends. A compatible eRDMA provider accepted the zero receive depth requested for a sending queue pair. The older provider in the SGLang image returned EINVAL. The reference used continuous 64 KiB GPU-to-GPU RDMA WRITE on the same pair of devices and ports.

| H20 configuration | Direction | Reference GB/s | Report / independent MB/s | Report / independent utilization |
|---|---|---:|---:|---:|
| SGLang TP2 | A TX | 11.4658 | 58.508 / 58.519 | 0.5103% / 0.5104% |
| SGLang TP2 | B RX | 11.4149 | 58.444 / 58.490 | 0.5120% / 0.5124% |
| vLLM PP2 | A TX | 11.4658 | 34.876 / 34.775 | 0.3042% / 0.3033% |
| vLLM PP2 | B RX | 11.4149 | 34.891 / 34.921 | 0.3057% / 0.3059% |

In the 3.050 s TP2 selection, 3,956 of 4,469 complete AllReduce operations carried just 2 KiB. A continuous two-GPU AllReduce microbenchmark on this link reached 0.073 GB/s with 2 KiB and 10.048 GB/s with 64 MiB. PP2 was asymmetric, with most bytes moving in the direction of activation sends. Port counters include other traffic on the device. One endpoint GPU was shared. The two disaggregated model runs used NIXL over UCX/TCP. Bundled UCX returned no usable device when explicitly asked for RC on the eRDMA port.

## Profiler checks

PyTorch CPU 2.10 and 2.15 and CUDA 2.13 and 2.15 were tested. Two H20 model windows using the unmodified PyTorch observer path each kept 3,290 kernels, collected 17 hardware metrics and produced the same tokens as the reference. Native Cuspy JSON and binary exports were tested. A real multi-file PP2 archive retained 1,216,989 slices, 130,746 flows, 127,280 kernels and matching endpoint kernel timestamps under independent Perfetto SQL queries.

## Short-window overhead

Two H20s ran eager TP2 serving at concurrency 2 and up to 8 generated tokens per request. The table gives the median of three roughly 3 s windows. Throughput counts completed output tokens during the window. TTFT and TPOT use requests wholly inside it.

| Framework | Mode | Output tok/s | TTFT ms | TPOT ms |
|---|---|---:|---:|---:|
| vLLM | No capture | 90.6 | 50.9 | 17.1 |
| vLLM | Native profiler | 63.9 | 72.9 | 24.4 |
| vLLM | ServeUtil | 61.3 | 73.2 | 24.6 |
| SGLang | No capture | 48.0 | 89.8 | 34.4 |
| SGLang | Native profiler | 21.3 | 168.7 | 65.9 |
| SGLang | ServeUtil | 26.7 | 137.6 | 61.6 |

For a deployment comparison, hold the model, input mix, concurrency and CUDA graph settings fixed, and use longer windows. Short-window differences can reflect requests crossing a boundary.

## Trace fidelity

[Viewer track and event validation](https://haoyangping0324.github.io/ServeUtil-Viewer/viewer-guide.html)

The Python recorder returns the installed PyTorch profiler object. Scheduled cycles, repeated Chrome trace export, metadata, shape and memory recording, stacks, FLOP estimates, execution observers and TensorBoard callbacks have regression tests. `serveutil.profiler` also exposes the installed version’s helpers. Available activities and backends depend on that installation.
