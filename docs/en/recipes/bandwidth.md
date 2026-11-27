# Explain a 0.51% eRDMA reading

[Open the measured port window](../../../examples/sglang-erdma.report.json). This two-host excerpt keeps the directional port samples and measured references from the common 3.050 s GPU window in an SGLang TP2 Qwen3 run. Use the node selector to compare Node A with Node B.

## Recompute the percentage

Click Node A's TX card. Its selection mean is about 58.508 MB/s. The directional reference on this port was 11.4658 GB/s during continuous 64 KiB GPUDirect RDMA WRITE. Divide 0.058508 by 11.4658 and multiply by 100. An independent process reading the sender port obtained 58.519 MB/s.

## Inspect the workload behind the average

The complete worker trace contained 4,469 AllReduce operations, 3,956 carrying only 2 KiB. Compute and scheduling time separate the transfers. Continuous AllReduce on the same hardware produced 0.073 GB/s with 2 KiB messages and 10.048 GB/s with 64 MiB. These port measurements describe different workloads.

## Follow an individual collective

The port excerpt deliberately contains only network samples. In the separate [two-rank AllReduce example](../../../examples/h20-allreduce.html), click a matched operation to inspect its group, bytes and nccl-tests reference. That example used a different transport. Keep its algorithm bandwidth separate from the eRDMA port rate. [Metric definitions](../metrics.md)
