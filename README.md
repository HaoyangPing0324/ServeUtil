<p align="center"><img src="assets/wordmark.png" width="360" alt="ServeUtil"></p>

# 看清一次推理里的计算与带宽

ServeUtil 把 PyTorch、vLLM 和 SGLang 的推理阶段、GPU 工作、硬件计数器放到同一条时间轴。打开一次 Decode，可以看到 GEMM 和 AllReduce 重叠了多久，选择其中一段可以查看 HBM 实测带宽，以及它相对这台机器基线的百分比。

<img src="assets/h20-decode.png" width="100%" alt="Qwen3 在 H20 上的一次 Decode，橙色标记突出 GPU 空档，时间线下方显示 HBM 读写、Tensor 活跃度与 BF16 GEMM 算力">

克隆后双击 `examples/index.html` 即可打开网页版。文件拖入浏览器后在本地解析。两个实测样例也可单独下载，在浏览器中直接打开。[网页版入口](examples/index.html) · [Qwen3 Decode 与 HBM 读写](examples/h20-decode.html) · [SGLang 双卡 AllReduce 与实测参考值](examples/h20-allreduce.html)

## 在 PyTorch 中录制

从仓库根目录安装，需要 Python 3.10 或更新版本。使用现有推理环境里的 PyTorch。

```bash
python -m pip install '.[gpu]'
```

```python
import torch
import serveutil

with serveutil.profile("runs", record_shapes=True) as prof:
    with torch.profiler.record_function("Prefill"):
        output = model(**inputs)
    prof.step()

print(prof.serveutil_directory / "report.html")
```

默认在首次采集前标定当前 CUDA 设备，后续运行复用缓存。使用 `calibration=False` 可跳过微测，也可以传入已保存的标定 JSON。记录输入形状后，ServeUtil 对能唯一关联到 GPU kernel 的 dense GEMM 显示有效 TFLOP/s。Linux 环境可用 `python -m pip install '.[pm]'` 安装 CUPTI 采样依赖。直接采集 HBM 读写取决于驱动、GPU 性能计数器权限以及可用的 CUPTI 或 PyTorch PM 接口。详情区列出本次采集成功的指标和缺失原因。[指标、硬件后端和利用率](docs/metrics.md)

已有原生 profiler 的代码可以保留原来的 schedule 和 `prof.step()`，在构造函数中增加一项。

```python
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    record_shapes=True,
    **serveutil.instrument("runs"),
) as prof:
    for batch in batches:
        model(**batch)
        prof.step()
```

## 在 vLLM 或 SGLang 中采集

服务启动时启用框架的 PyTorch profiler，并把 trace 写到 ServeUtil 能读取的目录。下列命令会预热模型、发出持续请求、采集硬件并生成 HTML；已有负载时省略 `--generate`。命令结束时会打印报告路径。

```bash
serveutil capture --engine vllm --url http://localhost:8000 \
  --trace-dir /profiles --seconds 5 --generate "How does a GPU work?"

serveutil capture --engine sglang --url http://localhost:30000 \
  --trace-dir /profiles --seconds 5 --generate "How does a GPU work?"
```

在空闲测试设备上加 `--calibrate` 获取服务基线，或用 `--calibration calibration.json` 复用已有结果。多卡服务可用 `--ranks` 指定预期 worker 数。跨节点或 Prefill/Decode 分离部署使用一份拓扑文件，记下请求入口、各角色的 profiler 端点和 worker 所在主机。后续每次采集只需一条命令。[服务启动参数和拓扑字段](docs/capture.md) · [多节点示例](examples/distributed-topology.json) · [PD 示例](examples/pd-topology.json)

```bash
serveutil capture --engine sglang --topology deployment.json \
  --seconds 5 --generate "Explain prefill and decode."
```

## 读报告

| 想弄清楚的问题 | 查看方式 |
|---|---|
| 一个 step 的时间花在哪里？ | 默认显示阶段和资源。切到 Operator depth，可选一二级、指定层或叶子。选择 rank 或拖动时间轴可以缩小范围。 |
| 两张卡在什么时候相互等待？ | Communication 展示 NCCL 分组、序号、对端、消息大小、有效带宽。存在可比的 nccl-tests 基线时，同步显示百分比。 |
| 哪一种资源接近实测上限？ | Hardware counters 展示 HBM、PCIe、NVLink、节点网络和各计算流水线的可用样本。选区卡片给出绝对量，有同口径实测基线时也显示百分比。 |
| 哪段值得优化？ | 橙色标记突出重复 GPU 空档、暴露的通信、持续带宽压力及可比较的 rank 耗时差异。点击可回到对应区间并查看触发依据。 |
| KV 交接影响了哪个请求？ | 有请求关联的 SGLang 采集展示发送区间、接收完成和 Decode 起点。vLLM NIXL 提供对应端点计数窗口的总字节、传输数和失败数。 |

**已有 trace** 可直接拖入网页或桌面窗口。使用以下命令可离线生成包含完整分析的报告，或导出可在 Perfetto 打开的轨道。

```bash
serveutil report rank0.pt.trace.json.gz -o report.html
serveutil export capture-directory -o overview.json
```

**桌面版** 和网页版使用同一套时间轴。桌面程序支持从命令行打开文件，也支持拖放。从源码运行可执行以下两行。CI 工作流提供 Linux、Windows、macOS 的构建入口及原生渲染检验。[桌面与网页构建产物](https://github.com/HaoyangPing0324/ServeUtil/actions/workflows/desktop.yml)

```bash
python -m pip install '.[desktop]'
serveutil view report.html --desktop
```

## 验证与复用

实测覆盖 Qwen3-0.6B、单卡与多卡推理、Prefill/Decode 分离和两台物理主机的数据交接。结果页列出框架版本、传输后端、GPU kernel 证据和短窗口 profiling 开销。[真实模型与采集开销](docs/validation.md)

ServeUtil 直接使用 PyTorch/Kineto 时间线、NVML GPM 与 CUPTI。标定可调用 nvbandwidth、nccl-tests 和 RDMA perftest。已有 llmperfscope run 可以导入，安装 HTA 或 TraceLens 后也能把上游分析结果放入同一份报告。[上游工具与对应入口](docs/tools.md) · [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html) · [Perfetto](https://ui.perfetto.dev/)

项目采用 Apache License 2.0。浏览器流式 JSON 解析器遵循其 MIT 许可证。[LICENSE](LICENSE) · [NOTICE](NOTICE)
