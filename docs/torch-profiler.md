# PyTorch Profiler 集成

`serveutil.profile` 返回本环境的 `torch.profiler.profile` 实例。使用 `with` 或手工调用 `start()`、`step()`、`stop()` 均可。每次运行的 `p.serveutil_directory` 包含原始 Kineto trace、硬件采样和交互报告。以下代码在 CPU 上也能运行。

```python
import torch
import serveutil

with serveutil.profile("runs", record_shapes=True, with_flops=True) as p:
    x = torch.ones(128, 128, device="cuda" if torch.cuda.is_available() else "cpu")
    with torch.profiler.record_function("Prefill"):
        y = x @ x
    p.step()

print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))
p.export_chrome_trace("last-cycle.json.gz")
print(p.serveutil_directory / "report.html")
```

## 保留已有采集代码

已有 `torch.profiler.profile` 的项目可将采集参数一并传给 `serveutil.instrument`。它组合原生回调和 observer，每个活动周期生成独立目录。下例保留 TensorBoard trace 和独立的 PyTorch execution graph。

```python
from pathlib import Path
import torch
import serveutil

root = Path("profiles")
root.mkdir(exist_ok=True)
observer = torch.profiler.ExecutionTraceObserver().register_callback(str(root / "execution.json"))
tb = torch.profiler.tensorboard_trace_handler(str(root / "tensorboard"), use_gzip=True)
options = serveutil.instrument(
    root / "runs",
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=2),
    on_trace_ready=tb,
    execution_trace_observer=observer,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
)
with torch.profiler.profile(**options) as p:
    for i in range(8):
        with torch.profiler.record_function("batch"):
            torch.ones(32, 32).sum()
        p.step()
```

`execution_trace_observer` 的 `start`、`stop` 和 `cleanup` 由 PyTorch 的生命周期驱动，原生 observer 仍写入注册的路径。采用 `ENABLE_PYTORCH_EXECUTION_TRACE` 自动生成 observer 时，ServeUtil 调用本版 PyTorch 的环境配置入口。原生 execution graph 描述算子和数据依赖，Kineto trace 则保存在各轮的 `traces/` 目录。[PyTorch Profiler API](https://docs.pytorch.org/docs/stable/profiler.html)

只需增加 trace 导出时，可将 `serveutil.trace_handler("runs", on_trace_ready=tb)` 用作原生 `on_trace_ready`。这种用法只接收已经录制完成的 trace。要采集同窗的硬件计数器，请使用 `profile` 或 `instrument`。

## 数据与原生选项

| 入口或选项 | 结果 |
| --- | --- |
| `activities`、`schedule`、`acc_events`、`experimental_config`，以及本版本新增的构造参数 | 传给安装的 PyTorch。省略的选项使用该版本的默认值。 |
| `record_shapes`、`with_flops`、`with_stack`、`with_modules`、`profile_memory` | 原生记录形状、受支持算子的 FLOP 估算、调用栈、受支持模型的 module hierarchy 和内存事件。字段是否存在取决于运行方式与本版 Kineto。 |
| `on_trace_ready` | 回调收到同一原生 profiler，可查询 `events()`、`key_averages()`、trace ID，或导出本周期 trace。 |
| `export_chrome_trace` | ServeUtil 每个完成周期归档一次完整的默认 Kineto 导出。回调内以及退出后重复导出普通 JSON 或 gzip 时，解压后的内容与该周期的归档逐字节一致。新版 PyTorch 的可选 Python exporter 仍使用原生 writer。 |
| `export_stacks`、`export_memory_timeline`、`toggle_collection_dynamic`、metadata API | 这些方法保留原生行为、前置条件与弃用提示。例如旧版 memory timeline 导出要求同时打开 `record_shapes`、`profile_memory` 和 `with_stack`。 |
| `devices`、`interval_ms`、`calibration`、`metrics` | ServeUtil 的硬件采样和标定选项。显式 CPU-only 活动默认跳过 CUDA 采样。默认选当前 CUDA 设备，可用 `devices=[0, 1]` 指定多个 CUDA 可见序号。已有标定可传 JSON 路径，`calibration=False` 可跳过微测。 |

`profile` 在一次 `start()` 到 `stop()` 之间保留连续硬件采样，在同一目录内保存所有完成周期。用 `instrument` 时，硬件采样随原生 observer 的活动周期启停。未进入保存阶段的 schedule 可以产生零份原始 trace。各轮时间线、CPU/GPU flow、内存事件、Python 栈和其他原始事件保存在 `traces/*.json.gz`；简化报告展示其中适合所选视图的内容。设备采样范围和采样分辨率可在报告详情查看。[硬件指标与基线](metrics.md)

标准 CPU / CUDA activity 可与设备支持的 CUPTI PM 采样配合。原生 `PerformanceMetricsConfig` 需使用提供此类的 PyTorch 版本。独立 CUPTI 设备采样需要兼容的库、驱动和性能计数器权限。当某项采样不可用时，硬件元数据记录原因。CPU-only、缺少 CUDA 权限、仅有原始 trace 的情形可继续查看原生已记录的事件。[PyTorch 性能指标配置](https://docs.pytorch.org/docs/2.15/profiler.html)

## 已验证的版本能力

PyTorch 2.10.0 的 CPU 录制已验证多周期回调、原生 execution graph、内存时间线、调用栈和重复导出。PyTorch 2.13.0+cu130 的 NVIDIA H20 录制已验证同组功能、CUDA kernel 与 CPU/CUDA flow，以及 TensorBoard 的 `use_python_export` 选项。两种接入方式均完成过 Qwen3 的多周期生成，原始文件保留设备活动、内存分配和 Python 调用事件。上述 H20 构建通过独立 CUPTI PM 采集方向性 HBM 计数器；可用的原生硬件指标配置取决于 PyTorch 构建。
