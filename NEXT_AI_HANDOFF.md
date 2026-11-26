# ServeUtil 项目交接

更新于 2026-11-26。接手时先读本文件和 [README](README.md)，再用仓库中的合成样例跑通一次转换。最先值得验证的是：在一次双卡推理里，能否把真实硬件采样与已有 step 对齐。离线导出器已经有可运行的起点。

## 用户确定的方向

- **计算与带宽同等重要。** 名称已定为 ServeUtil，英文描述可用 “Trace GPU compute activity and bandwidth across LLM inference stages in a single run with torch.profiler. Explore the results in Perfetto.” 目标场景是推理过程中各卡及各通信路径的资源利用情况。
- **一轮推理很贵。** 目标是在同一次 workload、同一个采集窗口中收集框架事件与硬件样本，保留原始输入。解析器、名称映射、统计口径和页面布局都应能离线重做。原型目前从已有 trace 开始，尚无启动采样的集成。
- **在推理时采样，窗口结束后查看。** 用户选择 Perfetto 网页离线导入。第一版沿用其缩放、选区、轨道和详情。框架未产生的端口计数、阶段摘要、跨机对时需要由 ServeUtil 补充。
- **能用的直接用。** 优先使用推理引擎已有的 step 标记、torch.profiler/Kineto 的 trace 与硬件指标、llmperfscope 的导出和已有设备接口。图太乱、kernel 名太长的诉求，已确定从导出层解决。只在上游没有记录所需语义时，考虑在最了解它的引擎或传输层补标记。
- **主标已通过审核。** 芯片内的柱形表示计算利用率，穿过边界的双向箭头表示带宽收发。用户驳回了更抽象的双轨 S 形草案，确认新主标后才派生另外两件。沿用 BenchToTune 的透明 PNG 命名：`assets/mark.png` 主标、`assets/icon.png` 小图标、`assets/wordmark.png` 横向字标。README 另有 Perfetto 样例截图。

项目采用 [Apache License 2.0](LICENSE)，[NOTICE](NOTICE) 中的署名为 ServeUtil contributors。

## 从哪里开始

仓库为 [HaoyangPing0324/ServeUtil](https://github.com/HaoyangPing0324/ServeUtil)。以下命令在检出目录的根目录运行。

| 文件 | 职责 |
|---|---|
| [serveutil.py](serveutil.py) | 命令行、流式读取、事件筛选与命名、写出 Chrome trace。 |
| [pyproject.toml](pyproject.toml) | 控制台入口 `serveutil`，依赖 ijson。可从当前仓库本地安装。 |
| [examples/decode.synthetic.json](examples/decode.synthetic.json) | 可公开使用的手造输入。包含三个 Decode step、并行 stream、长名、CUDA runtime 调用和模拟 counter。 |
| [examples/decode.perfetto.json](examples/decode.perfetto.json) | 当前导出器的样例输出，可直接导入 Perfetto。 |
| [tests/test_overview.py](tests/test_overview.py) | 检查 gzip、精确的时间原点、轨道与重叠、短名与原名、计数器透传和失败时保护原文件。 |

在仓库根目录复现：

```bash
python --version  # Python 3.10+
python -m pip install .
serveutil examples/decode.synthetic.json -o decode.overview.json
python -m unittest discover -s tests -v
```

**当前边界。** 一次处理一个 PyTorch Chrome trace 对象（`traceEvents` 数组，支持 JSON/gzip），不是跨 rank 合并器。保留 complete event 中的 kernel、GPU memcpy/memset、CPU/GPU 用户标记，以及 metadata 和现成的 counter。CPU 算子、Python 栈、launch、flow 箭头等详情仍在原图里；未采成 complete event 的自定义区间（例如 B/E 事件对）暂不覆盖。已保留标签更改事件的完整名称、pid/tid、时戳、时长及参数。导出不推断模型层、精度或硬件流量，不会自动对同一时段的算子归因。当前也没有阶段统计、在线面板、独立采样器、多芯片后端或 TFLOP/s 计算。

## 已验证与仍待验证

| 证据 | 可以据此判断什么 |
|---|---|
| 合成输入 56 个事件，输出 53 个（12 个区间、36 个 counter 样本、5 个元数据事件），3 个回归测试通过。浏览器实际在 Perfetto 中打开并展开两组轨道。 | 能同时看到 Compute、Communication、HBM read、SM Active 和 Decode step。README 截图来自这个样例，曲线数值仅作展示。 |
| 一份单机 SGLang prefill 的历史 H20 单 rank trace 转换成功。输入 688,271 个事件，输出 10,968 个，其中 10,900 个区间、68 个元数据事件、0 个 counter。 | 证明导出器能处理真实压缩输入。没有运行新推理，没有新增 GPU 实测指标。实际 trace 未收入仓库。 |
| 11 月 26 日只读检查四台 H20 节点。四台各 8 张 NVIDIA H20，同节点任意 GPU 对显示 NV18。32 张卡当前/最大 PCIe 链路均为 Gen5 ×16；每台 GPU 0–3 的 NUMA affinity 为 0，4–7 为 1。 | 节点内关注 NVLink，主机与卡之间关注 PCIe 和 NUMA。这是拓扑与协商速率，持续吞吐需在负载下测量。 |
| 四台各有两个 eRDMA 端口，erdma_0/1 对应 eth0，erdma_1/1 对应 eth1，均 ACTIVE，Ethernet 链路层，各报告 100 Gb/s。全部八个端口的 `/sys/class/infiniband/erdma_*/ports/1/hw_counters/hw_{tx,rx}_bytes_cnt` 已读到。 | 可做端口级 TX/RX 字节差分。单口标称 100 Gb/s 折 12.5 GB/s；多口聚合、负载实际选路和请求有效速率还没有验证。 |
| 11 月 24 日六对节点间小包主机内存 rping 通过。11 月 26 日 其中一个节点的 dcgmi 存在，但当时无法连接本地 DCGM host engine。 | 主机内存 RDMA 验证与 GPU 显存直传要分别记录。H20 实际推理容器中的 PyTorch/Kineto 版本、可采指标、权限、采样分辨率和开销仍需探测。 |

公开样例仅含合成事件。复验时可使用自己的单 rank trace。开展新实验前，先确认目标 GPU 和计数器的当前占用情况。

## 指标怎么定义

| 轨道 | 单位、来源和解释 |
|---|---|
| SM Active / Tensor Pipe Active | 分别展示窗口内 SM 及矩阵流水线活跃度（%），注明卡号和采样周期。SM 有活跃 warp 时也可能在等数据。整卡 GPU Util、SM 活跃、指令吞吐与峰值算力各有不同定义。 |
| 阶段有效计算吞吐（可选） | 用本 rank 覆盖到的算子 FLOPs 除以阶段墙钟时间，注明 dtype、并行分片、估算模型和覆盖率。除以 kernel 时长得到另一种口径。活跃度乘标称峰值无法给出实测 TFLOP/s。 |
| HBM | 使用设备支持的实际显存读/写字节或吞吐率，按卡区分读、写、单位 GB/s。DRAM Active 是接口活跃周期占比，不能作为 GB/s。 |
| 节点内 GPU 互连 | 用 NVLink 等链路的 TX/RX 计数或速率。NCCL 操作同时保留消息量和耗时；algbw 是逻辑消息量/耗时，busbw 是按 collective 和 rank 数折算的值。 |
| 主机 ↔ 卡 | 关联 PCIe TX/RX 与 HtoD/DtoH 拷贝及 NUMA。整卡 PCIe 流量还可能含网卡 DMA 等其他流量；显式拷贝字节和 PCIe 总量应分别显示。 |
| 节点 ↔ 节点 | 整端口带宽以字节差/时间差计，区分方向、端口和节点。单笔 KV 带宽以该传输逻辑字节/起止时长计。同一传输经过 HBM、互连和网卡会在多层留下记录，不能把这些数据相加为业务字节。 |

如展示“带宽利用率”，同时给出实测 GB/s 和参考分母，注明单向/双向、理论峰值/本机可持续带宽。没有可靠分母时只报绝对速率。为每项观测保存单位、来源、设备/端口、窗口起止、时钟域、有效性。记录计数器回绕/复位和采样中断；无法得到的指标应显示缺测。并发 step 或共享网卡上的其他任务可能影响整个窗口。一个 100 ms 的样本不能精确归给其中某个 8 ms step。

## 下一轮建议按这个顺序做

1. **离线精读。** 先用已有 SGLang trace 补齐有代表性的名字与阶段布局，用合成测试覆盖重名、未知 kernel 和异步重叠。保留用户标记作为阶段依据。一个名为 GEMM 的 kernel 本身不足以认定它属于 QKV 或 MLP Down。
2. **H20 能力探测。** 在实际推理进程使用的环境中检查 PyTorch、Kineto、CUDA、驱动及计数器权限。PyTorch 文档有 CUDA `PerformanceMetricsConfig`，具体支持取决于构建和硬件。确认 HBM 字节/速率、SM/Tensor 活跃度、NVLink/PCIe 的可用指标、方向和计量类型。llmperfscope 可复用 `run.json`、`samples.jsonl.gz`；阶段汇总和原始样本粒度应分别处理。对同一个硬件计数器选取一个采集方，核对 DCGM 与其他 profiler 的资源争用。
3. **单节点双卡首验。** 目标是一次 SGLang 推理同时获得原始 torch trace 与带时间戳的侧录文件。标记公共时间锚点，记录 GPU UUID、rank、采样周期及计时误差。分别选一个 Prefill 和 Decode step，核对每卡计算活跃度、HBM、PCIe、NVLink 及通信 kernel；分清 CPU 发起区间与 GPU 执行区间。重跑导出器应能离线重现视图。用短窗口先验证设备可承受的采样频率，另行说明采集开销和争用情况。
4. **双节点 PD。** 为 Prefill→Decode 的 KV 传输取得 transfer ID、逻辑字节、源/目的节点、提交与完成时间。校准跨机时钟并记录残余误差，通过同窗内两端 eRDMA 端口实测与传输层记录核对选路。分别验证主机 staging 与 GPUDirect RDMA 的实际条件；有设备并不代表当前任务走了它。
5. **阶段摘要及后续平台。** 先在 Perfetto/PerfettoSQL 上实现 step 和 rank 的聚合、缺测和重叠说明。SGLang/H20 是首验环境；vLLM、Ascend 910C、摩尔线程 S5000 是可能的后续方向，需分别检查框架标记、厂商计数器及数据权限，尚无平台支持承诺。

## 主要参考和可复用项目

- PyTorch Profiler 和 Kineto 提供框架事件、GPU 时间线与有条件的硬件计数支持。若将来复制其实现，复核对应源文件及第三方许可。当前仅消费导出格式。[Profiler API](https://docs.pytorch.org/docs/2.15/profiler.html) · [Kineto](https://github.com/pytorch/kineto)
- llmperfscope 提供阶段数据、PCIe 流量估计、NVLink 和采样输出，适合做适配器。复用时保留指标原始定义、设备范围和采样精度。[仓库](https://github.com/developer0hye/llmperfscope) · [格式文档](https://github.com/developer0hye/llmperfscope/blob/main/docs/run-json.md)
- Perfetto 原生接受 Chrome JSON 的区间与 counter；已有查看器足以支撑第一版。[格式](https://perfetto.dev/docs/getting-started/other-formats) · [SQL 派生轨道](https://perfetto.dev/docs/analysis/debug-tracks)
- SGLang 的 step 标记、传输层的 KV 语义、以及 eRDMA 端口观测分别描述不同层。优先复用引擎的标记与传输 ID。[SGLang profiling](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx) · [Mooncake](https://github.com/kvcache-ai/Mooncake) · [阿里云 eRDMA](https://www.alibabacloud.com/help/en/ecs/user-guide/erdma-monitoring-and-diagnosis)
- 查询设备可采集合时参考 DCGM 和 CUPTI；校验硬件/算法带宽口径时参考 NCCL Tests。[DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html) · [CUPTI](https://docs.nvidia.com/cupti/main/main.html) · [NCCL 带宽定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- 相关工程与研究已有持续遥测、trace 分析和推理算子关联经验，可直接借鉴。Nsight Systems 的 NIC 支持范围须与 eRDMA 单独核对。[Dynolog 与 HTA](https://pytorch.org/blog/automated-trace-collection/) · [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/) · [ProfInfer](https://proceedings.mlsys.org/paper_files/paper/2026/hash/03dbc11a22e79cd38bea53cf518c2371-Abstract-Conference.html)
