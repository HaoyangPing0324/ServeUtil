# 采集和分析工具的选择

| 现有工具 | 在 ServeUtil 工作流中的用途 |
|---|---|
| PyTorch Profiler / Kineto | 通过原生 schedule、回调和 observer 获取算子、kernel、stream 与关联 ID。原始 trace 随报告保存，可继续使用其他分析器。[PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html) |
| NVIDIA NVML / CUPTI | GPM 读取设备期间活动与链路速率。CUPTI PM 在硬件和权限允许时读取 DRAM 分向速率；结果保留采样区间。[NVML GPM](https://docs.nvidia.com/deploy/nvml-api/group__nvmlGpmFunctions.html) · [CUPTI PM Sampling](https://docs.nvidia.com/cupti/main/main.html#pm-sampling-api) |
| nvbandwidth、nccl-tests、perftest | 直接调用已有微测或导入其结果。保留设备对、数据类型、消息大小、收发方向、负载以及算法/总线口径。[nvbandwidth](https://github.com/NVIDIA/nvbandwidth) · [nccl-tests](https://github.com/NVIDIA/nccl-tests) · [perftest](https://github.com/linux-rdma/perftest) |
| Holistic Trace Analysis、TraceLens | 需要现成的 GPU 时间分类或通信重叠统计时，通过 `serveutil report --analysis` 执行上游分析器。结果与 ServeUtil 的区间标记放在同一份 HTML。[HTA](https://github.com/facebookresearch/HolisticTraceAnalysis) · [TraceLens](https://github.com/AMD-AGI/TraceLens) |
| llmperfscope | 导入现有运行目录，直接使用上游的采样读取器和 GPU 字段定义，查看服务生命周期与设备曲线。[llmperfscope](https://github.com/developer0hye/llmperfscope) |
| Perfetto | 随包分发经官方资源哈希校验的 v59.1 UI 和 WebAssembly Trace Processor。原始轨道 直接加载原始 trace，跨 rank 读取上游的多机 archive manifest，可在本地执行 SQL 或使用 Data Explorer。[Perfetto UI](https://perfetto.dev/docs/visualization/perfetto-ui) · [合并 trace](https://perfetto.dev/docs/analysis/merging-traces) |
| @streamparser/json、pywebview | 前者在浏览器 Worker 中流式读取原始 trace，后者在 Windows、Linux 和 macOS 的本地窗口载入相同界面。[streamparser](https://github.com/juanjoDiaz/streamparser-json) · [pywebview](https://pywebview.flowrl.com/) |

系统级排查还有两个互补入口。Nsight Systems 可以在系统时间线上显示 GPU Metrics 和受支持的 NIC 硬件采样。XProf 的 Roofline 分析可以按运算强度检查计算或访存约束，尤其适合已有 XLA/HLO 采集的工作负载。ServeUtil 的百分比使用本机微测分母，详情保留实际设备和条件，以便跨工具核对。[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) · [XProf Roofline](https://openxla.org/xprof/roofline_model)
