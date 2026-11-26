# 部署与采集

模型服务须启用其原生 PyTorch profiler。将 GPU 可见范围、worker、采样器和 trace 输出目录配置在同一部署中。ServeUtil 可在已有流量上录制，也可用 `--generate` 自动预热并持续发出真实请求。设置 `--max-tokens` 和 `--concurrency` 控制该负载。

## 单节点

vLLM 的 PyTorch profiler 启动配置示例为 `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/profiles"}'`。SGLang 可设置 `SGLANG_TORCH_PROFILER_DIR=/profiles`；不同版本的管理接口可能需要 `SGLANG_ADMIN_API_KEY`。按实际部署版本核对启动参数。运行 ServeUtil 的进程须能读取 worker 写出的文件，容器内外的目录可通过挂载对应。[vLLM profiling](https://docs.vllm.ai/en/latest/contributing/profiling/) · [SGLang profiling](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling)

```bash
serveutil capture --engine vllm --url http://localhost:8000 \
  --trace-dir /profiles --ranks 2 --seconds 5 --generate "Explain how a GPU works."
```

SGLang 使用 `--engine sglang`，需要指定服务端目录时加 `--server-trace-dir /profiles`。首次在空闲测试设备上获取本机基线可加 `--calibrate`，也可用 `--calibration path/to/calibration.json` 复用先前的结果。设备选择支持 NVML 物理编号或 GPU UUID，如 `--devices 0,1`。单独使用 `serveutil calibrate --devices 0,1` 时，数字指 CUDA 可见序号。

## 多节点与 PD 分离

```bash
serveutil capture --engine sglang --topology deployment.json \
  --seconds 5 --generate "Explain the difference between prefill and decode."
```

普通分布式引擎使用一个控制端点，分别从每台 worker 主机收集文件。PD 分离把生成请求送到 router，直接在 Prefill 和 Decode 端点开始和停止录制。[多节点配置样例](../examples/distributed-topology.json) · [PD 配置样例](../examples/pd-topology.json)

- `targets` 列出实际提供 profiling API 的服务。每个条目可有 `url`、`role`、`output_dir`（SGLang 进程看到的路径）、`key_env` 和请求超时 `timeout`。凭据从指定环境变量读取。
- `client_url` 是生成请求的基础 URL。生成器自动读取模型名，也可以通过 `--model` 指定，尤其适用于未提供模型枚举的 PD router。
- `nodes` 列出采集器进程。每个节点安装同一版本的 ServeUtil。`command` 是启动 `python -m serveutil.worker` 的参数数组，支持本机、SSH 和容器命令。标准输出用作数据通道。每个 rank 的 trace 由一个节点负责，共享目录应按实际写入范围配置。
- 每个节点的 `roots` 填写角色、worker 可见的 trace 目录和预期 rank 数。`workdir` 存放节点临时样本和选用的标定缓存。`devices` 可以限制采样设备。`device_map` 可明确将 Kineto 本地设备号映射到 GPU UUID，用于多 GPU 的算力基线查找。
- 同节点各 worker 使用独立的 CUDA 设备编号时，可用 `rank_device_maps` 按角色和 rank 覆盖节点映射。例如 `{"worker":{"0":{"0":"GPU-…"},"1":{"0":"GPU-…"}}}` 将两个进程各自的 CUDA 0 对应到实际设备。选中 rank 后，查看器按映射筛选 GPU 曲线，并保留节点级网络曲线。
- 节点可配置 `calibrate: true` 或 `calibration_file`。拓扑的 `links` 数组可加入基线测量配置。独立标定入口是 `serveutil calibrate --link link.json -o network-reference.json`。[网络配置样例](../examples/link.json)
- SGLang 请求级 KV 关联可在拓扑中加 `transfers` 指向 OpenTelemetry 采集文件或描述文件，路径相对于拓扑 JSON。vLLM 在配置了角色的 PD 端点上读取可用的 NixlConnector 指标。

## 采集包

命令返回本轮的 `report.html`。原始 trace、压缩硬件样本、配置结果、节点时钟和可用性在同一目录。可用 `serveutil report CAPTURE -o report.html` 重建页面，或用 `serveutil export CAPTURE -o overview.json` 输出供 Perfetto 打开的多轨概览。设备权限或接口不足的原因见页面底部的采集详情。

浏览器与桌面查看器可直接打开 ServeUtil HTML、report JSON 和 PyTorch JSON/gzip。直接打开原始 trace 时以已有事件及 counter 绘图；离线 report 命令另行执行关联和优化机会分析。安装 `.[llmperfscope]` 后，还可用 report 命令读取 llmperfscope 的 run 目录，复用其阶段和原始设备采样。[llmperfscope 格式](https://github.com/developer0hye/llmperfscope/blob/main/docs/run-json.md)

## NCCL 集合通信与基线

点击通信 kernel 可查看本 rank 的进程组、序号、消息字节和算法带宽。一个操作包含多个 kernel 时，按第一个开始到最后一个结束的时间计算，并在第一个 kernel 展示速率。AllGather 取完整输出大小，ReduceScatter 取完整输入大小。支持的集合操作同时显示按 nccl-tests 公式折算的总线带宽，send/recv 显示本 rank 方向及 trace 记录的对端。[NCCL 带宽定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

要对照实测基线，在节点的 `calibration_file` 中提供相同传输配置下的 nccl-tests 标定结果。节点的 `collective_groups` 按 `roots[].role` 和 Kineto 的 `Process Group Name` 指定 GPU UUID，顺序与该组的 rank 顺序一致。以下是角色 `worker`、进程组 `3` 的节点配置片段。请将示意 UUID 替换为实际值。

```json
{
  "collective_groups": {
    "worker": {
      "3": {
        "devices": ["GPU-uuid-of-first-rank", "GPU-uuid-of-second-rank"],
        "operations": {
          "all_reduce": {"placement": "in_place", "reduction": "sum"},
          "all_gather": {"placement": "out_of_place"}
        }
      }
    }
  }
}
```

ServeUtil 优先使用 trace 中能验证的规约类型和缓冲区布局，再使用配置补全缺失项。节点、角色、进程组、设备顺序、操作、消息大小、数据类型与布局均有对应记录时，页面显示本 rank 算法带宽相对 nccl-tests 基线的百分比。规约操作还要匹配规约类型，有根操作还要匹配 `root_rank`。trace 和基线都记录了 NCCL 版本时，还会检查版本一致性。该百分比可以超过 100%，受 kernel 内等待和测试条件影响。
