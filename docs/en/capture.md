# Capture a model server

Start the service with its PyTorch profiler enabled. The worker writes operator and device traces. ServeUtil controls the recording window, samples hardware, collects each worker's files and builds one report.

## Single node

For vLLM, add `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/profiles"}'` to the serve command. For SGLang set `SGLANG_TORCH_PROFILER_DIR=/profiles` before launch. Mount the same directory into the collector if the server runs in a container. Some SGLang deployments require `SGLANG_ADMIN_API_KEY`. Verify launch flags for your installed version. [vLLM profiling](https://docs.vllm.ai/en/latest/contributing/profiling/) · [SGLang profiling](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling)

```bash
serveutil capture --engine vllm --url http://localhost:8000 \
  --trace-dir /profiles --ranks 2 --seconds 5 --generate "Explain how a GPU works."
```

Set `--ranks` to the expected worker count. For SGLang use `--engine sglang` and its service URL. If the worker sees a different output path, pass `--server-trace-dir`. The optional generator warms up and sends streaming requests. Adjust `--concurrency` and `--max-tokens` or omit `--generate` to observe existing traffic.

## Calibrate resources

Use `--calibrate` on idle test GPUs or pass `--calibration calibration.json` to reuse a prior run. The capture option `--devices` accepts NVML physical indices or GPU UUIDs. Numeric indices in standalone `serveutil calibrate --devices 0,1` refer to CUDA-visible devices. Use UUIDs where mappings differ.

## Multi-node and disaggregated serving

Start with the [distributed topology](../../examples/distributed-topology.json) or [PD topology](../../examples/pd-topology.json). Substitute your hosts, paths and role names. Install the same ServeUtil version on each collector node.

```bash
serveutil capture --engine sglang --topology deployment.json \
  --seconds 5 --generate "Explain the difference between prefill and decode."
```

| Topology field | How to fill it |
|---|---|
| `client_url` | Generation endpoint, usually the router for PD. Pass `--model` if the router cannot enumerate models. |
| `targets` | Each profiling endpoint's URL, role, worker-visible `output_dir`, optional `key_env` and `timeout`. SGLang accepts `with_stack` when Python stacks are needed. |
| `nodes[].command` | Argument array launching `python -m serveutil.worker` locally, through SSH or inside a container. SSH can collect files while model data travels over the actual serving network. |
| `roots`, `workdir` | Give each worker trace exactly one owning collector. Set the role, directory, rank count and writable working directory. |
| `devices`, `device_map`, `rank_device_maps` | Map visible CUDA ordinals to physical UUIDs. Per-rank mappings distinguish workers that each see a local GPU 0. |
| `calibrate`, `calibration_file`, `links` | Acquire or load node references and describe links for directional baselines. |
| `transfers` | SGLang request spans or their collection description. Paths are relative to the topology file. vLLM uses configured NIXL endpoint counters for PD. |

For network calibration, edit the addresses, device, port and buffer type in the [RDMA link example](../../examples/rdma-link.json). Install compatible perftest binaries on both hosts and run `serveutil calibrate --link link.json -o network-reference.json`. Both endpoints default to host memory. Specify CUDA devices at both ends for a GPU-to-GPU reference when GPUDirect is available. For TCP use the [TCP link example](../../examples/link.json).

## Collectives and outputs

To compare against nccl-tests, supply a matching calibration file. Set `collective_groups` by role and Kineto process-group name when group layout is missing. Specify ordered GPU UUIDs and the operation. The report matches message size, layout, operation, root or reduction where applicable, and NCCL version when recorded.

The command prints the capture directory. Keep the capture directory for original traces, calibration, samples and clock metadata. Rebuild with `serveutil report CAPTURE -o report.json`. Failed collection records which endpoints or metrics were unavailable.
