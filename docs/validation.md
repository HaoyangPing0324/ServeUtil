# 真实模型与采集开销

示例使用 BF16 Qwen3-0.6B 和 NVIDIA H20。GPU 时间来自 Kineto；硬件曲线保留设备、方向、采集后端及采样区间。运行窗口的短期波动、输入形状和通信拓扑都会影响读数。[模型配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json) · [指标定义](metrics.md)

## 验证范围

| 能力 | 验证方法与结果 |
|---|---|
| 原生 PyTorch 入口 | 在 PyTorch 2.10 和 2.15 上核对对象类型、公开方法和可用构造参数。实际切换 CPU 活动、写入 metadata、运行 schedule、查询事件并执行两种版本各自支持的导出。H20 上另外验证 CUDA kernel、内存、调用栈和 flow。 |
| PyTorch 2.15 CUDA 后端 | H20 上实测普通和细粒度 CUDA 活动。Cuspy 的同步和异步 JSON 可解析，原生 protobuf 归档逐字节一致，官方 Trace Processor 从中读取 309,140 个 slice。相同负载用原生 profiler 和 ServeUtil 各录一次，Cuspy 均记录 1,600 个 kernel、2,406 个 HBM 读写采样事件。 |
| 保留原生程序 | 用 `serveutil.observe` 包住原样的 PyTorch 与 TensorBoard handler。H20 上运行 Qwen3，两轮分别导出 3,290 个 kernel，生成 token 与未采集参考逐项相同，同时取得 HBM 读写等 17 类硬件指标。 |
| 官方查看器 | 核对官方稳定版入口、manifest 以及清单中全部 76 个资源的 SHA-256。在浏览器离线执行 SQL，打开 Data Explorer、Metrics、Plugins 和独立录制页。三种桌面构建均包含原生渲染测试，覆盖模型报告、Chrome JSON、二进制 Perfetto trace。 |
| 原件与聚合 | 三份 vLLM PP2 原件归档后逐文件校验。官方 Trace Processor 对聚合数据和参考数据得出相同的 1,216,989 个 slice、130,746 条 flow、127,280 个 kernel，以及相同的首末 kernel 时间。 |
| 跨节点资源比例 | 从六组双机运行的原始数据重建报告，在浏览器点击资源卡片。按相同窗口独立计算 48 组网口或 eRDMA 收发读数，每组完整采样累计字节都落在独立读数的时间边界范围内。 |
| 可选分析器 | 真实 vLLM trace 的 HTA 两张表和 TraceLens 设备时间表，逐行核对上游独立运行结果。llmperfscope 运行目录可导入生命周期阶段。上游分析所需的调用栈、step 等信息取决于原始录制选项。 |

## 双物理节点直连

两台主机各使用一张 H20，运行 BF16 Qwen3-0.6B。vLLM 0.32.0 使用 PP2 和 NIXL PD，SGLang 0.5.22 使用 TP2 和 NIXL PD。NCCL Socket 和 UCX/TCP 通过物理网络直连。以下读数取两端实际 GPU 活动的共同窗口。独立采样进程直接读取网卡累计字节，用于复核报告选区。

| 场景 | 发送端 GB/s | 接收端 GB/s | 发送端利用率 · 报告 / 独立 | 接收端利用率 · 报告 / 独立 | 请求与通信 |
|---|---:|---:|---:|---:|---|
| vLLM PP2 | 0.03283 | 0.03410 | 1.281% / 1.281% | 1.284% / 1.283% | 56 个请求，592 对 Send/Recv |
| vLLM PD | 0.31408 | 0.32203 | 12.253% / 12.248% | 12.126% / 12.237% | 24 个请求，16 次 NIXL 读取 |
| SGLang TP2 | 0.03395 | 0.03554 | 1.325% / 1.327% | 1.338% / 1.324% | 13 个正式请求，4,959 组 AllReduce |
| SGLang PD | 0.02869 | 0.03163 | 1.119% / 1.210% | 1.191% / 1.131% | 14 个正式请求，12 次窗内 KV 交接 |

利用率的分母是同一节点、方向、端口在单 TCP 流标定中测得的网卡速率，发送端 2.5633 GB/s，接收端 2.6557 GB/s。端口字节包含协议开销和其他接口流量。由于 GPU 活动边缘可能落在两个读数之间，短促的 KV 突发对窗口边界敏感。四组实验中，ServeUtil 各整采样区间的累计字节均落在独立读数的上下界内。vLLM PD 复测由引擎报告 1,027,604,480 个逻辑字节和零次失败。

以上四组 TCP 模型运行的 eRDMA 端口增量均为零。

## NCCL 经 eRDMA 的模型推理

NCCL 2.30.7 的 NET/IB 发送队列请求 `max_recv_wr=0`。同一台机器、同一块 erdma_0，SGLang 镜像内 Ubuntu 的 ibverbs-providers 50.0-2ubuntu0.2 返回 EINVAL，把接收深度改为 1 即成功。只读挂载宿主机配套的 libibverbs 和 eRDMA provider 56.2-1.0.5 后，保留请求值 0 也能成功，实际返回 1。两端内核 eRDMA 0.2.39、compat_mode=Y。厂商软件包的 56.2-1.0.3 更新记录明确修复了零接收深度问题。先前通过的 ib_write_bw 使用宿主机 provider 和 RDMA CM，验证的是另一条建连路径。[NCCL 发送队列源码](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/transport/net_ib/connect.cc#L649) · [厂商修复记录](https://www.alibabacloud.com/help/en/ecs/user-guide/erdma-usage/)

两端各使用一张 H20，同一对 GPU 通过 erdma_0/1 实测 64 KiB 的连续 GPUDirect RDMA WRITE，两方向分别标定。表内每个分母取对应端口与方向的硬件计数速率，每个分子取两端 GPU 活动交集内本端口的平均速率。独立采样进程读取同一个 sysfs 硬件计数器，按相同边界复算。指标反映整个端口，包括控制包和并行作业流量。计数器复核检验采样、归一化与展示。节点 B 的 GPU 与已有任务共享。两种模型配置的双端 NCCL 均记录了 `NET/IB/0/GDRDMA`。

| 模型 | 节点 / 端口方向 | 同向标定 GB/s | 报告 / 独立 MB/s | 报告 / 独立利用率 |
|---|---|---:|---:|---:|
| SGLang TP2 | A TX | 11.4658 | 58.508 / 58.519 | 0.5103% / 0.5104% |
| SGLang TP2 | A RX | 12.5698 | 58.509 / 58.519 | 0.4655% / 0.4655% |
| SGLang TP2 | B TX | 12.6076 | 58.443 / 58.490 | 0.4636% / 0.4639% |
| SGLang TP2 | B RX | 11.4149 | 58.444 / 58.490 | 0.5120% / 0.5124% |
| vLLM PP2 | A TX | 11.4658 | 34.876 / 34.775 | 0.3042% / 0.3033% |
| vLLM PP2 | A RX | 12.5698 | 0.285 / 0.285 | 0.0023% / 0.0023% |
| vLLM PP2 | B TX | 12.6076 | 0.286 / 0.286 | 0.0023% / 0.0023% |
| vLLM PP2 | B RX | 11.4149 | 34.891 / 34.921 | 0.3057% / 0.3059% |

SGLang 的 3.050 秒窗口中，发送端有 4,469 次完整 AllReduce，其中 3,956 次载荷只有 2 KiB，其他 513 次为 126 KiB。单并发逐 token 生成产生大量短消息，两次传输之间还有计算和调度时间。为观察消息大小的影响，用同一对 GPU 连续执行两卡 PyTorch NCCL AllReduce。2 KiB 时端口发送速率为 0.073 GB/s，64 MiB 时达到 10.048 GB/s。微测的应用有效载荷、端口字节和模型窗口平均速率有各自的计数范围。

vLLM PP2 的 3.227 秒选区中，发送端 595 个完整 Send 有 504 个为 4 KiB，595 个 Broadcast 的载荷均为 8 或 16 字节。发送端通信 kernel 的持续时间并集占该窗口的 98.8%，其中也包含同步和等待对端的时间。对应端口平均发送速率为 34.88 MB/s。

### PD 使用的传输后端

双机 PD 的两组端到端数据来自 NIXL 的 UCX/TCP。在两端分别加载 vLLM 0.32.0 和 SGLang 0.5.22 随 NIXL 分发的 UCX，显式选择 TCP/eth0 时四次均成功建立 UCP context。对照实验将配套的宿主机 verbs 库及 provider 只读映射进容器，确认 NIXL 实际加载的 verbs 能枚举 erdma_0，再要求 UCX 使用 `rc` 和 `erdma_0:1`，四次均返回 `no usable transports/devices`。因此当前有跨机 eRDMA 正向证据的模型配置是 NCCL TP2 和 PP2；PD 的带宽与利用率以实际 TCP 网口及同向标定为准。

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
