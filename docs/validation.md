# 真实模型与采集开销

示例使用 BF16 Qwen3-0.6B 和 NVIDIA H20。GPU 时间来自 Kineto；硬件曲线保留设备、方向、采集后端及采样区间。运行窗口的短期波动、输入形状和通信拓扑都会影响读数。[模型配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json) · [指标定义](metrics.md)

## 可以对照的观测

| 运行 | 可核对的结果 |
|---|---|
| PyTorch 原生 schedule | 8 步，每步生成 6 个 token。wait 1、warmup 1、active 2，重复两轮。两个活动窗口各保存 19,816 个计算 kernel，可重用同一组 32 项本机标定结果。 |
| vLLM 0.29.0、SGLang 0.5.21 | 单 GPU、双 GPU TP，以及同机的 Prefill/Decode 分离均以实际请求采集。SGLang 请求 span 可关联到交接区间。vLLM NIXL 的一次采集在 Decode 端得到 36 笔传输、66,060,288 字节、0 次失败。 |
| 多进程 GPU 通信 | vLLM 0.32.0 使用两实例 PP2。SGLang 0.5.22 使用两实例 TP2。两实例位于同一台 H20 主机。独立 rank trace 保留实际的 NCCL 点对点及集合通信元数据。 |
| 跨物理 GPU 的 vLLM PD | vLLM ExampleConnector 导出 28 层、共 5,505,024 字节的 KV，经显式文件中继交给另一台主机。Decode 端命中 48 个外部缓存 token，生成的 8 个 token 与完整模型逐项吻合。trace 中分别有 28 次等大小的 DtoH 和 HtoD。 |
| 跨物理 GPU 的 SGLang PD | SGLang 0.5.22 使用两台主机上的独立 Prefill、Decode 服务，经 NIXL 和 UCX/TCP 转发 KV。在包含预热的隔离请求中，6 笔可用引擎直方图与请求 span 唯一对应，各 7,225,344 字节。单次请求的 8 个 token 与单 GPU 参考逐一吻合。两端分别保留了 1,470 和 9,258 个 GPU kernel。 |
| 跨物理 GPU 的 SGLang TP2 | 两台主机各运行一个 SGLang 0.5.22 rank，经 NCCL NET/Socket 完成 6 笔真实流式请求。两个物理 GPU 各记录 116 个 NCCL kernel，两个节点分别采到 17,340 和 17,295 个硬件及端口样本。 |
| 跨物理 GPU 的 vLLM PP2 | 两台主机各运行一个 vLLM 0.32.0 rank，使用 NCCL NET/Socket 经中继连接。13 笔请求生成 39 个输出 token；两端 38 对 Send/Recv 按组、序号与消息大小逐一匹配，激活载荷合计 200,704 字节。另一次生成的 8 个 token 逐项等于单卡参考。 |
| 跨物理 GPU 的 PyTorch PP 与 PD | 两台主机各使用一张 H20，经应用专用 SSH 中继传输激活或 KV。重组后的两个模型均与单卡对照逐个生成相同的 5 个 token，两端载荷哈希一致。这组实验用于验证分布式采集和传递结果。 |
| 跨节点 RDMA | ib_write_bw 的 64 KiB WRITE 测得主存到主存 41.36 GB/s、GPU 到远端主存 47.34 GB/s。后者源和目的端口各约 47.82 GB/s。校准记录注明缓冲区位置和方向。 |

双卡 AllReduce 样例还使用与 trace 相同的 NCCL 2.30.7、同一对 GPU、同一 Socket 路径运行 nccl-tests。截取的 Decode 窗口中，两个 rank 各有 57 个事件能按操作、载荷和布局匹配实测分母。[查看通信样例](../examples/h20-allreduce.html)

## 采集时服务会慢多少

下表记录两张 H20 上 TP2 eager 服务的短窗口观测。使用持续流式请求、并发 2、每请求最多输出 8 个 token。每种模式跑三个约 3 秒的窗口，表内为三轮中位数。吞吐按窗口内已完成的输出 token 统计，TTFT 与 TPOT 按完整落入窗口的请求统计。原生 profiler 与 ServeUtil 两组均开启框架的 PyTorch trace。

| 服务 | 采集模式 | 已完成输出 tok/s | TTFT (ms) | TPOT (ms) |
|---|---|---:|---:|---:|
| vLLM | 关闭采集 | 90.6 | 50.9 | 17.1 |
| vLLM | 原生 profiler | 63.9 | 72.9 | 24.4 |
| vLLM | ServeUtil | 61.3 | 73.2 | 24.6 |
| SGLang | 关闭采集 | 48.0 | 89.8 | 34.4 |
| SGLang | 原生 profiler | 21.3 | 168.7 | 65.9 |
| SGLang | ServeUtil | 26.7 | 137.6 | 61.6 |

在部署环境里复查开销时，可固定模型、到达率、并发、生成长度和 CUDA Graph 设置，分别运行以上三种模式并增加测量时长。短窗口内少数请求跨过边界就能改变吞吐排序。
