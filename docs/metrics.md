# 指标与时间

每个硬件样本有来源、范围、单位和采样起止时间。选择一段时间后，资源卡片按样本在选区内覆盖的时长加权。一个样本跨越多个 step 时，几个 step 会共享同一观测。点击资源卡片可以查看覆盖时长与参考负载。

| 指标 | 单位与口径 |
|---|---|
| HBM 读、写 | CUPTI PM sampling 提供分方向的设备级读写速率，十进制 GB/s。需要 GPU、驱动和性能计数器权限支持。 |
| HBM 总带宽估算 | 用 NVML GPM 的 DRAM 带宽利用率和设备报告的显存参数推算。查看器以 estimated 区分此口径。 |
| SM、Tensor、精度流水线 | 设备采样周期内的活跃度，单位 %。支持情况取决于 GPU 和驱动。 |
| GEMM 有效算力 | 对具有输入形状、数据类型且唯一关联到一个 GPU kernel 的 dense mm、bmm、addmm 以及同输入精度的 _int_mm、_scaled_mm，以乘加计两次操作。浮点运算显示 TFLOP/s，整数运算显示 TOP/s，速率按形状推算的操作数除以实测 kernel 时长。选区卡片仅计完整落入选区的匹配 kernel，分母为这些 kernel 耗时之和。 |
| GPU↔GPU | NVLink TX/RX 为设备全部所选链路的合计速率。集合通信 kernel 的名称和时间可在独立轨道查看。nccl-tests 的算法带宽与总线折算带宽分别存储。 |
| CPU↔GPU | PCIe TX/RX 涵盖该 GPU 的总 PCIe 流量。可识别的 HtoD、DtoH 事件另按拷贝有效字节和 GPU 时长计算。 |
| 节点网络 | eRDMA 字节计数、IB 端口计数或 Linux 网口计数按时间差换算 GB/s。同一份数据可能经过多个设备，跨层汇总须使用对应层的计数。 |
| PD 的 KV 传输 | 有请求关联时，显示源端传输阶段和相应请求标识。字节来自可唯一归属的引擎观测。vLLM NixlConnector 的累计计数差分按端点及指标标签报告。 |

## 实测参考值

`serveutil calibrate` 在选定 CUDA 设备上运行微测。显存规模、计算精度、GEMM 形状、拷贝方向、设备身份和采样来源随结果保存。资源卡片使用同来源、设备、指标和单位的基线。100% 表示达到该微测的参考值，工作负载超过参考值时保留实际百分比。GEMM 的详情列出参考矩阵大小，供读者比较服务实际形状。

可用的外部工具包括 nvbandwidth、nccl-tests 和 RDMA perftest。网络端口基线来自负载持续期间的同一端口和方向。应用有效负载带宽与端口实测速率分别保存。[nvbandwidth](https://github.com/NVIDIA/nvbandwidth) · [NCCL tests 带宽定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) · [perftest](https://github.com/linux-rdma/perftest)

## 时钟与优化标记

每个节点在采集前后与控制端进行多次往返对时，保留最短往返时间和偏移估计。查看器把 Kineto 的整数纳秒原点和硬件样本时间映射到同一坐标，详情记录往返不确定度和观测到的漂移。比较两台机器上接近的事件时，应计入两端的不确定度。

橙色标记提供回到原始区间的入口。GPU 空档按同设备所有已记录 stream 上的 kernel、copy、memset 的区间并集计算。通信暴露时长取同卡通信区间扣除计算区间后的部分。持续带宽压力使用连续样本和可比实测基线。rank 耗时对比要求能唯一标识同一个逻辑 step。标记记录触发条件，根因仍需结合模型依赖和框架调度判断。

## 复用已有分析

`serveutil report TRACE --analysis hta -o report.html` 调用 HTA 的 GPU 时间分解、通信与计算重叠接口。`--analysis tracelens` 使用 TraceLens 的 GPU activity breakdown。先安装相应分析器。结果以引擎原始字段保存在页面详情里，各 rank 独立显示。两者对时间精度和 kernel 类别的处理可不同，比较时应保留各自口径。[Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) · [TraceLens SDK](https://github.com/AMD-AGI/TraceLens/blob/main/docs/how-to/sdk-analysis.md)

## 集合通信详情

在 **Communication** 视图选择完整操作，可读到该 rank 的有效载荷、进程组、序号和 GPU 执行时间。多 kernel 操作从首个 kernel 开始计到最后一个结束，有效载荷只计一次。AllGather 按聚合后的总消息、ReduceScatter 按散发前的总消息计算有效带宽。支持的等量集合另显示依照 nccl-tests 定义折算的 bus bandwidth。点对点收发显示对端和方向。各 rank 的执行时间包含其等待对端的时间，因而同一序号可以有不同的速率。

`serveutil.profile` 和 `serveutil.instrument` 都接受 `calibration="auto"`（默认，在首次录制前测量并缓存）、`False` 或已保存的标定 JSON 路径。使用已有 `torch.profiler.schedule` 时，`instrument` 为每个活动窗口单独保存原始 trace 和对应的硬件样本。

同设备拷贝的有效载荷为复制的字节数。与显存 copy 微测比较时，读、写各计一次，详情中的 `read_write_GB_s` 为两者之和。GPU 对 GPU 的 peer copy 使用有效载荷速率，仅在能确认发送和接收设备、并找到同方向的设备对基线时显示百分比。
