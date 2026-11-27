## 安装与采集

在模型环境中使用 Python 3.10 或更高版本。保留服务现有的 PyTorch 安装。

```bash
pip install serveutil
serveutil --help
```

| 组件 | 版本与验证 |
|---|---|
| 基础 CLI 与报告 | Python ≥3.10。pip 自动安装 `ijson>=3.2,<4`、`prometheus-client>=0.17,<1`、`nvidia-ml-py>=12.550.52`。分析已有报告时可以使用无 GPU 的电脑。 |
| PyTorch 进程内采集 | 使用模型环境自带的 PyTorch。CPU 采集验证了 2.10.0、2.15.0；H20 上验证了 2.13.0+cu130、2.15.0+cu132。GPU 采集还需与该 PyTorch 构建匹配的驱动。 |
| 服务端采集 | 只需安装实际运行的框架并开启其 PyTorch profiler。同机测过 vLLM 0.29.0、SGLang 0.5.21；双物理机测过 vLLM 0.32.0、SGLang 0.5.22。多节点的各采集进程使用同一 ServeUtil 版本。 |

可选 GPU 计数器依赖 — `serveutil[pm]` (Linux, cupti-python ≥13.4,<14), `serveutil[llmperfscope]` (llmperfscope ≥0.2,<1).

[桌面客户端下载与离线安装](https://haoyangping0324.github.io/ServeUtil-Viewer/zh/downloads.html)
