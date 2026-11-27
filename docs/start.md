# 生成第一份推理报告

## 打开已有的 trace

在网页版查看器中拖入 PyTorch trace，多 worker 运行可选「打开采集目录」。采集生成的报告还可包含硬件样本和标定数据。浏览器在本地读取选定的文件。离线分析可使用桌面客户端。[安装与下载](downloads.md)

## 采集一小段计算

在装有 PyTorch 的环境运行下例。它支持 CPU，在 CUDA 可用时使用当前 GPU。

```python
import torch
import serveutil

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(256, 256, device=device)
with serveutil.profile("runs", record_shapes=True) as p:
    with torch.profiler.record_function("Projection"):
        y = x @ x
    p.step()
print(p.serveutil_directory / "report.html")
```

打开打印出的 HTML。先在「阶段与资源」找到 Projection，然后进入「GPU kernel」查看设备活动。纯 CPU 采集显示主机活动。CUDA 采集可额外显示硬件计数器。点击资源卡片可查实际测量范围。[PyTorch 选项](torch-profiler.md)

## 采集模型服务

启动 vLLM 或 SGLang 时开启引擎的 torch profiler，然后把服务地址和 trace 目录传给采集器。采集器通过框架接口在实际模型 worker 里启停录制。

```bash
serveutil capture --engine vllm --url http://localhost:8000 \
  --trace-dir /profiles --seconds 5 --generate "Explain prefill and decode."
```

已有业务流量可省略 `--generate`。多节点和 PD 分离使用 `--topology deployment.json`。[配置服务和 worker 路径](capture.md)

## 阅读真实报告

可从 [Decode 空档](recipes/decode.md)、[eRDMA 带宽基线](recipes/bandwidth.md)或[一笔 KV 交接](recipes/pd.md)入手。每篇教程都有相应的实测样例。
