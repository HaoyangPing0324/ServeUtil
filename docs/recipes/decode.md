# 定位一次 Decode 中的 GPU 空档

[打开 Qwen3 Decode 实测报告](../../examples/h20-decode.html)。节选来自单张 H20，保留了一次 16-token 输入、16-token 输出推理中的三个连续 Decode step。

## 选择一步

在「阶段与资源」视图用「区间」选中一个 Decode step，沿计算和拷贝轨道观察。点击「完整范围」比较三步。在「优化机会」找到重复的 GPU 空档，点击排在首位的区间即可放大相邻活动。

## 查证空档

标记表示在同一设备已录制的各个 stream 上，这段时间没有 kernel、拷贝或 memset。检查下一个 kernel 之前的主机启动和同步行为。上游漏录设备事件也可能留下空白。完整采集可用 原始轨道 沿 CPU 到 GPU 的 flow 追踪；这个公开节选只保留选定事件和硬件样本。

## 核对访存与计算

点击 HBM 读或写的资源卡片，对照样本覆盖时长和空档长度。横跨相邻 kernel 的样本反映整段活动。切到「GPU kernel」，检查可唯一关联到 dense GEMM 的 BF16 有效算力。在自己的模型上复测时启用输入形状记录。[用 PyTorch 采集](../torch-profiler.md)
