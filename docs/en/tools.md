# Tools and integrations

| Tool | How ServeUtil uses it |
|---|---|
| [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html) / Kineto | Native activities, schedules, execution observers, callbacks and original trace files. |
| [NVML GPM](https://docs.nvidia.com/deploy/nvml-api/group__nvmlGpmFunctions.html) / [CUPTI PM](https://docs.nvidia.com/cupti/main/main.html#pm-sampling-api) | Device activity and interconnect counters, plus directional HBM rates when available. |
| [nvbandwidth](https://github.com/NVIDIA/nvbandwidth), [nccl-tests](https://github.com/NVIDIA/nccl-tests), [perftest](https://github.com/linux-rdma/perftest) | Calibrate specific copy paths, collectives, message sizes and RDMA directions. Application logical bandwidth and physical port counters retain separate definitions. |
| [HTA](https://github.com/facebookresearch/HolisticTraceAnalysis), [TraceLens](https://github.com/AMD-AGI/TraceLens) | Install the analyzer and pass `--analysis hta` or `--analysis tracelens` to `serveutil report`. Results appear alongside the trace details. |
| [llmperfscope](https://github.com/developer0hye/llmperfscope) | Install `serveutil[llmperfscope]` and pass its run directory to `serveutil report` to import lifecycle stages and samples. |
| [Perfetto](https://perfetto.dev/docs/visualization/perfetto-ui) | The bundled upstream UI and Trace Processor v59.1 open original files and query all ranks in the native workspace. |
| @streamparser/json and pywebview | A browser worker streams raw JSON. Windows, Linux and macOS desktop windows run the same inference viewer. |

For complementary system-level evidence, Nsight Systems can record GPU and supported NIC metrics. XProf can provide roofline analysis for XLA/HLO workloads. Keep each tool's time range, units and measurement scope when comparing results.
