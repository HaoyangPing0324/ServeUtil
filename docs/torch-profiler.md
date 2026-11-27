# PyTorch Profiler 集成

`serveutil.profile` 返回本环境的 `torch.profiler.profile` 实例。函数签名列出本机 PyTorch 支持的参数。两个采集周期之间可以替换或清空 `p.on_trace_ready`。使用 `with` 或手工调用 `start()`、`step()`、`stop()` 均可。每次运行的 `p.serveutil_directory` 包含原始 Kineto trace、硬件采样和交互报告。以下代码在 CPU 上也能运行。

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
print(p.serveutil_directory)
```

`serveutil.profiler.profile` 和 `serveutil.profile` 是同一个类。已有 profiler 子类可以继承它，并通过 `super().__init__(...)` 初始化。手工管理窗口时，可依次调用 `prepare_trace()`、`start_trace()` 和 `stop_trace()`。每个独立窗口都会生成一次采集，之后仍可导出原始 trace。直接调用这三个方法时，PyTorch 不会触发 `on_trace_ready`。

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

## 保留原生导出器和异步写入

需要自己控制原生回调、导出格式或后台写入时，可以用 `serveutil.observe` 包住现有代码。上下文返回新建的采集目录。将要在查看器打开的原始 trace 写入其中的 `traces/`，其他产物也可以写在这个采集目录内。硬件采样覆盖整个上下文。退出时，ServeUtil 从实际落盘的文件生成报告。

```python
with serveutil.observe("runs") as run:
    handler = torch.profiler.tensorboard_trace_handler(str(run / "traces"), use_gzip=True)
    with torch.profiler.profile(on_trace_ready=handler, record_shapes=True) as p:
        model(inputs)
    if hasattr(p, "wait_for_exports"):
        p.wait_for_exports()
```

这个入口使用应用构造的原生 profiler，保留其导出方法、活动筛选和 observer。如果应用自行接管后台导出任务，请在退出 `observe` 前等待任务完成。使用需要独占 CUPTI 的原生设备采样时，可设 `include_hbm=False`，其余硬件采样仍随上下文启停。Chrome JSON 可以生成推理视图，二进制 trace 可在 原始轨道 中打开。

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

PyTorch 2.10.0 和 2.15.0 的 CPU 录制已验证多周期回调、原生 execution graph、内存时间线、调用栈和重复导出。PyTorch 2.13.0+cu130 的 NVIDIA H20 录制已验证同组功能、CUDA kernel 与 CPU/CUDA flow，以及 TensorBoard 的 `use_python_export` 选项。`profile`、`instrument` 和 `observe` 均完成过 Qwen3 的多周期生成，原始文件保留设备活动、内存分配和 Python 调用事件。上述 H20 构建通过独立 CUPTI PM 采集方向性 HBM 计数器；可用的原生硬件指标配置取决于 PyTorch 构建。

同机切换活动类型时，PyTorch 2.13.0+cu130 在 H20 上表现出进程级初始化差异。先录仅 CPU 的任务，再用 CPU 和 CUDA 录制 GPU 矩阵乘，纯 PyTorch 与 ServeUtil 的默认导出都可能缺少 kernel。将包含 CUDA 的活动放在进程首次采集，或在新进程启动 GPU 采集，可得到 kernel。ServeUtil 在报告摘要及详情中标明请求了 CUDA、却没有记录到 kernel、拷贝或 memset 的周期。

PyTorch 2.10 和 2.15 的默认 Kineto writer 在字符串 metadata 含内嵌引号时可能产生无效 JSON。ServeUtil 保留原始字节，在报告中标出无法解析的文件，继续展示同一采集中可解析的周期。可单独下载原件核查上游输出。

在 PyTorch 2.15.0+cu132 的 H20 上，普通 CUDA 活动、按 `CONCURRENT_KERNEL` 筛选、Cuspy 的同步和异步 JSON 导出、原生压缩 Perfetto protobuf 均经过实测。使用 `metrics=["dram__bytes_read.sum.per_second", "dram__bytes_write.sum.per_second"]` 可请求当前 GPU 的原生 HBM 采样。该版本的快捷入口同时启用 `CuspyConfig` 和 `PerformanceMetricsConfig`。实验使用 CUPTI 13.4.92，随 cu132 环境提供的 CUPTI 13.2.2 低于 Cuspy 要求的 13.3。显式传入 `activities` 时可自行控制上游后端。在同一台 H20 上仅启用 Kineto 与 PM 配置时，原生 PyTorch 和 ServeUtil 均有 kernel、零个 PM 计数事件。

也可以使用 `from serveutil import profiler`，由同一命名空间调用当前 PyTorch 的 `schedule`、`record_function` 等辅助 API。使用 `profiler.profile(...)` 得到原生 profiler 实例，可直接调用 `step()`。

在已验证的 PyTorch 2.15 Cuspy 后端上，默认导出以及在回调内完成的自定义导出，都会保留同一窗口的原生 JSON 和二进制编码。回调移动或删除自己的文件后，采集目录仍保留原件。在 `stop()` 后仍可导出 `.pftrace`，也可在查看器的来源列表中选择二进制文件。两份编码使用同一组 GPU 记录，额外开销是编码时间和磁盘空间。

回调通过 `take_pending_cuspy_export()` 将写入移交给其他线程时，在该线程调用原生 `join()` 即可。等待写入期间状态为 `exporting`，所有周期写完后自动生成报告。各周期可以乱序完成。

在支持 `export_chrome_trace(..., use_python_export=True)` 的 Kineto 版本上，ServeUtil 会一并保存该写入器生成的文件。在 ServeUtil Viewer 的来源列表中选中它，即可查看导出时添加的标注和元数据。默认 trace 和这些导出文件对应同一个采集窗口。
