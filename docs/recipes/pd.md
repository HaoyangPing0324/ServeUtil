# 跟踪一次 Prefill 到 Decode 的 KV 交接

[打开一笔实测的 SGLang 交接](../../examples/sglang-pd.report.json)。两个角色运行在不同的 H20 主机上，通过 NIXL、UCX/TCP 传输。样例匿名化了端点名称，保留这次请求的相对时间和对应的引擎观测。

## 找到两端

切到「通信」，点击「KV 传输」中的记录。源端 scheduler 传输区间约 22.707 ms。目的端从记录到接收完成到开始 Decode 约 180 µs。分别选中两个节点可查看各自角色的活动。

## 区分三种读数

对应引擎观测为该请求记录了 7,225,344 个 KV 逻辑字节和约 22.525 ms，有效速率约 0.321 GB/s。用同样字节数除以源端 scheduler span 得到约 0.318 GB/s。完整采集中，网口卡片统计该接口经过的全部字节。三种读数的计时窗口与覆盖范围不同。

## 采集自己的部署

在 [PD 拓扑样例](../../examples/pd-topology.json)中把请求指向 router，把 profiler 控制指向两个角色端点。通过拓扑的 `transfers` 字段配置 SGLang 请求 span，然后执行 `serveutil capture --engine sglang --topology deployment.json --seconds 5 --generate "Explain KV caching."`。vLLM 的 NIXL 计数面板给出端点窗口总量。按请求归属字节时，需要同一请求的匹配证据。[部署与采集](../capture.md)
