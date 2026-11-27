# ServeUtil

ServeUtil 把 PyTorch、vLLM 和 SGLang 的模型活动与 GPU、互连和网口读数放在同一条推理时间轴。可以从 Prefill 或 Decode 进入某个节点和 rank，查看计算、通信、GPU 空档、绝对带宽和有实测基线的利用率。随包提供的完整 Perfetto 工作区可继续查看原始事件和运行 SQL。

## 安装和使用

```bash
pip install serveutil
serveutil view rank0.pt.trace.json.gz
```

已有 trace 可在普通电脑上离线查看。要生成可分享的 HTML 报告，运行 `serveutil report rank0.pt.trace.json.gz -o report.html`。采集使用推理环境已有的 PyTorch。对启用了 PyTorch profiler 并将 trace 写入共享目录的 vLLM 服务，运行：

```bash
serveutil capture --engine vllm --url http://localhost:8000 \
  --trace-dir /profiles --seconds 5 --generate "Explain prefill and decode."
```

SGLang 使用 `--engine sglang` 和对应的服务地址。首次在空闲设备上标定可加 `--calibrate`。独立 CUPTI HBM 采样依赖可在兼容的 Linux 环境用 `pip install "serveutil[pm]"` 安装。用 `serveutil --help` 查看命令。PyPI 下载页的源码包内含完整 README、使用指南和可离线查看的 H20 实测报告。
