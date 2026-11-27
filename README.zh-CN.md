<p align="center"><img src="assets/wordmark.png" width="360" alt="ServeUtil"></p>
<h1 align="center">面向大模型推理的计算与带宽分析</h1>
<p align="center"><strong><a href="https://haoyangping0324.github.io/ServeUtil/zh/">项目文档</a> &nbsp;|&nbsp; <a href="https://haoyangping0324.github.io/ServeUtil-Viewer/viewer.html?lang=zh">网页版查看器</a> &nbsp;|&nbsp; <a href="https://haoyangping0324.github.io/ServeUtil/zh/downloads.html">安装指南</a> &nbsp;|&nbsp; <a href="README.md">English</a></strong></p>
<p align="center">
  <a href="https://github.com/HaoyangPing0324/ServeUtil/actions/workflows/collector.yml"><img alt="Collector CI" src="https://github.com/HaoyangPing0324/ServeUtil/actions/workflows/collector.yml/badge.svg"></a>
  <a href="#安装与采集"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB"></a>
  <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-337b57"></a>
  <a href="https://haoyangping0324.github.io/ServeUtil/zh/capture.html"><img alt="PyTorch · vLLM · SGLang" src="assets/engines.svg"></a>
</p>

ServeUtil 在同一次推理中收集 PyTorch 原始 trace、GPU 硬件计数器与节点间网络流量。支持 PyTorch 程序、vLLM 和 SGLang，包括张量并行与 Prefill/Decode 分离。用同一设备、方向和端口的实测带宽作为分母，可以同时报告 GB/s 和利用率。

例如，两台 H20 上的 SGLang TP2 在一次单并发推理的 3.05 秒选区内平均发送约 58.5 MB/s，约为同向 eRDMA 端口实测基线的 0.51%。同一选区 4,469 次完整 AllReduce 中有 3,956 次载荷仅 2 KiB。把消息大小、计算间隔和端口读数放在一起，才能解释这个百分比。[模型与两端计数器实测](docs/validation.md)

## 安装与采集

在模型环境中使用 Python 3.10 或更高版本。保留服务现有的 PyTorch 安装。

```bash
pip install serveutil
```

```python
import torch
import serveutil

device = "cuda" if torch.cuda.is_available() else "cpu"
projection = torch.nn.Linear(256, 256, bias=False).to(device).eval()
x = torch.ones(32, 256, device=device)

with torch.inference_mode(), serveutil.profile("runs", record_shapes=True) as prof:
    with torch.profiler.record_function("Projection"):
        projection(x)
    prof.step()

print(prof.serveutil_directory)
```

`prof` 就是原生 `torch.profiler.profile` 对象。默认在首次 CUDA 采集前标定所选设备，后续复用缓存。已有原生 schedule、TensorBoard 回调或异步导出的项目还可用 `instrument` 或 `observe` 接入。[PyTorch 集成示例](docs/torch-profiler.md)

**已经运行的 vLLM 或 SGLang 服务**需要先启用框架的 PyTorch profiler。以单 GPU 为例，vLLM 启动时增加 `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/profiles"}'`。SGLang 启动前设置 `SGLANG_TORCH_PROFILER_DIR=/profiles`。服务和采集端都要能访问对应的 trace 目录。随后选择与服务对应的一条命令。

```bash
serveutil capture --engine vllm --url http://localhost:8000 \
  --trace-dir /profiles --seconds 5 --generate "Explain prefill and decode."

serveutil capture --engine sglang --url http://localhost:30000 \
  --trace-dir /profiles --seconds 5 --generate "Explain prefill and decode."
```

命令会预热并发出请求，通知模型 worker 启停原生 profiler，同时采集硬件指标，最后打印采集目录。已有业务流量时省略 `--generate`。空闲测试设备可用 `--calibrate` 获取基线，或用 `--calibration calibration.json` 复用已有结果。多节点或 PD 分离只需先写一次拓扑文件，之后每次仍用一条采集命令。[服务采集指南](docs/capture.md) · [多节点拓扑](examples/distributed-topology.json) · [PD 拓扑](examples/pd-topology.json)

在 [ServeUtil Viewer](https://haoyangping0324.github.io/ServeUtil-Viewer/viewer.html?lang=zh) 中选择「打开采集目录」，导入打印出的整个目录。原始 trace、采样和标定会一起打开。[采集文档](https://haoyangping0324.github.io/ServeUtil/zh/) · [查看器操作指南](https://haoyangping0324.github.io/ServeUtil-Viewer/zh/viewer-guide.html) · [查看器仓库](https://github.com/HaoyangPing0324/ServeUtil-Viewer)
