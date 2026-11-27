// Translate viewer controls while retaining original trace names, payloads and IDs.
const languageToggle = document.getElementById("viewerLanguage");
const uiChinese = {
  "Global rank":"全局 rank", "Operation":"操作", "Operation number":"操作序号", "Peer rank":"对端 rank", "Original kernel":"原始 kernel",
  "Communication ID":"通信 ID", "Group members":"通信组成员", "Bootstrap room":"Bootstrap room", "Endpoint attribution":"端点归属",
  "Clock uncertainty (µs)":"时钟不确定度（µs）", "Data type":"数据类型", "Operation time":"操作耗时", "Rank group":"Rank 组",
  "Algorithm bandwidth (GB/s)":"算法带宽（GB/s）", "NCCL bus bandwidth (GB/s)":"NCCL 总线带宽（GB/s）", "Relative to nccl-tests":"相对于 nccl-tests",
  "Shape-derived GEMM throughput":"根据形状估算的 GEMM 吞吐", "Sum of GPU operation durations":"GPU 操作总时长", "Measured GEMM reference":"GEMM 实测参考值",
  "Matched kernels":"匹配 kernel 数", "Estimated integer operations":"估算整数运算量", "Estimated FLOPs":"估算浮点运算量",
  "Sum of GPU kernel durations":"GPU kernel 总时长", "Sample window":"采样窗口", "Original sample":"原始采样值", "Baseline utilization":"相对于参考值的利用率",
  "source_node":"发送节点", "destination_node":"接收节点", "logical_bytes":"逻辑字节数", "engine_transfer_ms":"引擎传输时间（ms）",
  "effective_gb_s":"引擎有效带宽（GB/s）", "scheduler_span_GB_s":"调度区间带宽（GB/s）", "receiver_to_decode_us":"接收至 Decode（µs）",
  "Switch language":"切换语言", "Levels 1–2":"第 1–2 层", "All levels":"全部层级", "Full trace":"完整 trace",
  "Independent origins":"独立时间原点", "Rank-local UTC":"rank 本地 UTC", "Payload unavailable":"缺少载荷数据",
  "Inference traces":"推理时间线", "Open capture folder":"打开采集目录", "Open trace":"打开 trace",
  "Open a trace":"打开 trace", "Choose a file":"选择文件", "Inference timeline":"推理时间线",
  "Drop a PyTorch trace or a ServeUtil report here to explore the timeline.":"拖入 PyTorch trace 或 ServeUtil 报告，查看时间线。",
  "PyTorch trace · ServeUtil report · Perfetto trace":"PyTorch trace · ServeUtil 报告 · Perfetto trace",
  "Open a ServeUtil report to inspect a capture.":"打开 ServeUtil 报告以查看采集结果。",
  "Export image":"导出图片", "Full Perfetto":"完整 Perfetto", "Save original":"保存原件", "Back to ServeUtil":"返回 ServeUtil",
  "View":"视图", "Stages & resources":"阶段与资源", "Stages only":"仅阶段", "Hardware counters":"硬件计数器",
  "Operator depth":"算子层级", "GPU kernels":"GPU kernel", "Communication":"通信", "Depth":"层级",
  "Leaf only":"仅叶子", "Node":"节点", "All nodes":"全部节点", "Rank":"Rank", "All ranks":"全部 rank",
  "Names":"名称", "Readable":"简洁名称", "Original":"原始名称", "Clock":"时钟", "Capture elapsed":"采集相对时间",
  "Rank elapsed":"rank 相对时间", "Interval":"区间", "Full capture":"完整采集", "Fit":"完整范围",
  "Timeline":"时间线", "Compute":"计算", "Copy":"拷贝", "Opportunity":"优化机会", "Opportunities":"优化机会",
  "Drag to zoom · Click to inspect":"拖动缩放 · 点击查看详情", "Communication paths":"通信路径", "KV transfers":"KV 传输",
  "Resources in selection":"选区资源", "Optimization opportunities":"优化机会", "Selection details":"选区详情",
  "Select an event, sample or highlighted interval.":"选择事件、样本或标记区间。", "Capture and metric availability":"采集详情与可用指标",
  "Repeated GPU gap":"重复 GPU 空档", "Exposed communication":"未与计算重叠的通信", "Sustained bandwidth pressure":"持续带宽压力",
  "No matching opportunities in this interval.":"此区间内无匹配的优化机会。", "Focus operation":"定位操作",
  "Selection mean":"选区均值", "Covered time":"覆盖时长", "Measured reference":"实测参考值", "Device":"设备",
  "Unit":"单位", "Source":"来源", "Scope":"范围", "Time resolution":"时间分辨率", "Duration":"时长",
  "Logical bytes":"逻辑字节数", "Complete operations":"完整操作数", "Method":"计算方法", "Annotation":"标注",
  "Original trace":"原始 trace", "Example traces":"实测与示例", "View settings":"视图设置", "Zoom out":"缩小", "Zoom in":"放大",
  "Activity timeline, drag horizontally to select an interval":"活动时间线，横向拖动以选择区间", "Communication path":"通信路径",
  "Open this report with serveutil view or the desktop app for Full Perfetto.":"请使用 serveutil view 或桌面客户端打开报告以使用完整 Perfetto。",
  "Qwen3 decode on H20":"H20 上的 Qwen3 Decode", "SGLang AllReduce on two H20s":"双 H20 的 SGLang AllReduce",
  "Full trace example (synthetic)":"完整 trace 示例（合成）", "Project guides":"项目指南", "KV handoff":"KV 交接",
};
let viewerChinese = new URLSearchParams(location.search).get("lang") === "zh";
function uiText(text) {
  if (!viewerChinese) return text;
  if (Object.hasOwn(uiChinese, text)) return uiChinese[text];
  return text.replace(/^Level (\d+)$/, "第 $1 层").replace(/ · Prefill transfer /g, " · Prefill 传输 ").replace(/ · Received → decode /g, " · 接收完成至 Decode ")
    .replace(/ trace\(s\) · /g, " 份 trace · ")
    .replace(/ · capture time/g, " · 采集时间").replace(/ · independent trace origins/g, " · 独立时间原点")
    .replace(/ timeline$/, " 时间线").replace(/^Other intervals \((\d+)\)$/, "其他区间（$1）")
    .replace(/^([\d,]+) operations in selection$/, "选区内 $1 个操作")
    .replace(/ in selection$/, "（选区内）").replace(/% of reference/g, "% 实测参考值");
}
const localizedLabels = new WeakMap();
function translateLabel(node) {
  let original = node.nodeValue;
  const cached = localizedLabels.get(node);
  if (cached && cached.display === original) original = cached.original;
  const display = uiText(original.trim());
  const value = original.replace(original.trim(), () => display);
  if (node.nodeValue !== value) node.nodeValue = value;
  localizedLabels.set(node, {original, display:value});
}
// The renderer owns raw event values, kernel names and native Perfetto content.
const labelSelector = "header .muted, header label, header a, .welcome h1, .welcome p, .welcome small, #examples a, button:not(#viewerLanguage), .controls label, option, main h1, main h2, .legend span, .foot span, summary, #details dt, #linkList label, #summary, #issues, #linksCount, #nativeStatus, #details";
function translateControls() {
  for (const el of document.querySelectorAll(labelSelector)) {
    for (const n of el.childNodes) {
      if (n.nodeType === Node.TEXT_NODE && n.nodeValue.trim()) translateLabel(n);
    }
  }
  for (const el of document.querySelectorAll(".issue small, .metric span")) {
    if (el.firstChild?.nodeType === Node.TEXT_NODE) translateLabel(el.firstChild);
  }
  for (const el of document.querySelectorAll("[aria-label]")) {
    const en = el.dataset.englishAria || el.getAttribute("aria-label");
    el.dataset.englishAria = en;
    el.setAttribute("aria-label", uiText(en));
  }
  for (const link of document.querySelectorAll("#examples a, a.projectDocs")) {
    const url = new URL(link.href);
    if (url.origin !== location.origin) continue;
    if (link.classList.contains("projectDocs")) {
      link.href = viewerChinese ? "zh/index.html" : "index.html";
    } else {
      if (viewerChinese) url.searchParams.set("lang","zh"); else url.searchParams.delete("lang");
      link.href = url.href;
    }
  }
}
function applyViewerLanguage() {
  document.documentElement.lang = viewerChinese ? "zh-CN" : "en";
  languageToggle.textContent = viewerChinese ? "English" : "简体中文";
  translateControls();
}
languageToggle.onclick = () => {
  viewerChinese = !viewerChinese;
  const url = new URL(location.href);
  if (viewerChinese) url.searchParams.set("lang","zh"); else url.searchParams.delete("lang");
  try { history.replaceState(history.state, "", url); } catch (_) { /* file: viewers may restrict history */ }
  applyViewerLanguage();
  if (typeof draw === "function" && report) draw();
};
applyViewerLanguage();
let translationQueued = false;
new MutationObserver((changes) => {
  if (!viewerChinese || translationQueued || !changes.some(c =>
    c.type === "childList" && c.target.nodeType === Node.ELEMENT_NODE && !c.target.closest("#metadata, #nativeFrame")
  )) return;
  translationQueued = true;
  queueMicrotask(() => { translationQueued = false; translateControls(); });
}).observe(document.body, {childList:true, subtree:true});
