# 从推理总览进入完整 trace

运行 `serveutil view capture-directory`，或在网页版用 **Open capture folder** 选择采集目录。在推理视图选择 **All nodes / All ranks** 可同时看所有 worker。选择节点或 rank 会过滤时间线、资源和通信路径。选择 **Original** 查看原始算子名，选择 **Readable** 查看推理阶段简称。**Operator depth** 支持一二级、指定深度、叶子和全部层级。时间轴下方的资源卡片随选区更新。

**Communication paths** 可以从一端跳到另一端，并在时间线上聚焦一次操作。集合通信和点对点传输优先使用可核对的通信 ID，只有完整的组成员、序号、数据类型和字节数相符时才按元数据推断。缺失或冲突的记录保留在对应 rank。PD 的请求级 KV 关联使用发送和接收证据，引擎仅提供直方图的场景展示窗口总量。跨机时间轴提供时钟校准的不确定度；独立导入的文件可查看 rank 内顺序，跨机时差需要共同时间基准。

## 官方 Perfetto 工作区

选择 **Full Perfetto** 可在同一份报告打开官方 Perfetto UI v59.1，包括 Timeline、Query (SQL)、Data Explorer、Metrics、工作区和该版本内置的插件。下拉框可选当前 rank、一个节点或所有原始文件。全局工作区保留辅助进程 trace，按来源区分重复 PID 和 TID，通过上游 archive manifest 应用采集时测得的时钟偏差。单文件直接交给上游解析器，可读取原生二进制 Perfetto trace 及上游支持的其他格式。浏览器只在切换到完整工作区时载入 WebAssembly。[官方 UI](https://perfetto.dev/docs/visualization/perfetto-ui) · [多文件与时钟配置](https://perfetto.dev/docs/analysis/merging-traces)

**Save original** 可下载选定文件，保留原始压缩字节。选择全部文件时下载可直接交给 Trace Processor 的 TAR，内部 gzip 信封会展开，成员内容保持原样。完整采集生成的 HTML 携带原始文件，分享前请确认 trace 中的调用栈、用户 metadata 和主机信息适合接收方。公开的实测节选经过裁剪，原件只存在于原采集。构建产物中的合成样例可以直接体验两种视图。

| 打开方式 | 可用入口 |
|---|---|
| `serveutil view PATH` 或桌面应用 | 通过本机回环地址加载随包提供的官方 UI 和 WASM，离线使用。 |
| 自托管静态网页 | 把 Web 构建产物完整放在同一目录，使用 HTTP(S) 打开。可以从空白页多选 trace 或打开保存的报告。 |
| 双击单个 HTML | 直接使用推理视图和原件下载。要自动打开内置的完整工作区，使用桌面版或 `serveutil view report.html`。 |
| 独立 Perfetto 标签页 | 选择 **Open Perfetto in a tab** 可在顶层浏览器环境使用官方录制界面和原生文件选择器。设备录制依赖浏览器、操作系统及设备权限；外部扩展或在线服务依赖各自的网络和服务配置。 |

## 导出

```bash
serveutil export capture-directory -o overview.json
serveutil export --full capture-directory -o full.json
serveutil export --full rank0.json rank1.json.gz --clocks clocks.json -o full.json
```

默认导出生成推理概要轨道。`--full` 导出原始 Chrome 事件，包括辅助 trace、flow、counter、metadata、stackFrames、samples 和事件参数。多文件 JSON 使用独立的进程、线程和标识符映射，保留各文件原始 header；单一共享 JSON 没有 Perfetto 的原生多机身份表，需要上游原生机器和线程语义时使用完整工作区或其 TAR。裸文件校时表按文件名提供 `origin_ns` 和 `offset_ns`。已有 capture 可直接使用采集记录的时间基准。
