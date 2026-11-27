# ServeUtil 文档

ServeUtil 采集 PyTorch、vLLM 和 SGLang 的推理过程。在同一条时间线上可以查看算子、GPU kernel、rank 间通信和硬件采样，并将带宽与同机实测的参考值比较。

## 安装

在模型环境中使用 Python 3.10 或更高版本。

```bash
pip install serveutil
```

桌面客户端和 GPU 计数器的可选依赖见[安装与下载](downloads.md)。

## 查看 trace

可以在网页版打开已有 trace，也可以在本机运行以下命令。

```bash
serveutil view rank0.pt.trace.json.gz
```

[打开 H20 上实测的 Qwen3 Decode 时间线](../examples/h20-decode.html)。点击 GPU 事件查看耗时，跨轨道拖选一段时间比较计算和通信。按 **F** 放大到选区。

## 采集推理

运行自己的 Python 模型时，从 [PyTorch 快速开始](start.md) 入手。采集已启动的服务时，先[配置 vLLM 或 SGLang](capture.md)，再录制各 worker 的同一段推理。
