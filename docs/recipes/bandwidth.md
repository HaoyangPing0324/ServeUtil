# 解释 0.51% 的 eRDMA 利用率

[打开真实端口采样窗口](../../examples/sglang-erdma.report.json)。这份双机节选保留 SGLang TP2 运行 Qwen3 时，两端 GPU 共同活动的 3.050 秒窗口内的方向性端口样本和实测基线。用节点选择器在 A、B 之间切换。

## 复算百分比

点击节点 A 的 TX 卡片，选区均值约为 58.508 MB/s。同一端口在连续 64 KiB GPUDirect RDMA WRITE 标定中的发送速率为 11.4658 GB/s。计算 0.058508 ÷ 11.4658 × 100，得到 0.5103%。独立采样进程读取该发送端口得到了 58.519 MB/s。

## 解释窗口均值

完整 worker trace 中共 4,469 次 AllReduce，其中 3,956 次只有 2 KiB。传输之间还有计算和调度时间。相同硬件上连续 AllReduce 的端口发送速率，2 KiB 消息为 0.073 GB/s，64 MiB 消息为 10.048 GB/s。比较读数时要保留负载条件。

## 查看一笔集合通信

端口节选只保留网络采样。在另一份[双 rank AllReduce 样例](../../examples/h20-allreduce.html)中点击匹配操作，可查看通信组、消息字节和 nccl-tests 参考值。这份样例使用不同的传输路径，其算法带宽与上述 eRDMA 端口速率应分别解读。[指标定义](../metrics.md)
