"use strict";
const $ = (id) => document.getElementById(id);
const canvas = $("chart"),
  ctx = canvas.getContext("2d");
const color = {
  stage: "#385775",
  compute: "#087e81",
  communication: "#8458b3",
  copy: "#4283ab",
  operator: "#64779c",
  signal: "#bb632e",
  counter: "#268b73",
};
let report,
  rows = [],
  stages = [],
  items = [],
  windowStart = 0,
  windowEnd = 1,
  bounds = [0, 1],
  drag = null,
  targets = [],
  titleWidth = 200;
const fmt = (us) =>
  Math.abs(us) >= 1e6
    ? (us / 1e6).toFixed(2) + " s"
    : Math.abs(us) >= 1000
      ? (us / 1000).toFixed(2) + " ms"
      : us.toFixed(1) + " µs";
function option(select, value, label) {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = label;
  select.append(o);
}
function rankLabel(rank) {
  const t = report?.traces.find((t) => t.rank === rank);
  if (!t?.node) return "Rank " + rank;
  const parts = rank.split("/");
  return (
    (new Set(report.traces.map((t) => t.node)).size > 1 ? t.node + " / " : "") +
    (t.role && t.role !== "worker" ? t.role + " / " : "") +
    "Rank " +
    parts.at(-1)
  );
}
function short(text, n) {
  return text.length <= n ? text : text.slice(0, n - 1) + "…";
}
function details(title, data) {
  const root = $("details"),
    head = document.createElement("strong"),
    list = document.createElement("dl");
  root.replaceChildren(head, list);
  head.textContent = title;
  for (const [k, v] of Object.entries(data)) {
    if (v == null) continue;
    const dt = document.createElement("dt"),
      dd = document.createElement("dd");
    dt.textContent = k;
    dd.textContent = typeof v === "object" ? JSON.stringify(v) : String(v);
    list.append(dt, dd);
  }
}
function depthOf(events) {
  const lanes = new Map();
  for (const e of events) {
    if (e.kind !== "operator" && e.kind !== "stage") continue;
    const key = e.kind + "/" + e.device + "/" + e.lane;
    const stack = lanes.get(key) || [];
    lanes.set(key, stack);
    while (
      stack.length &&
      (stack.at(-1).ts + stack.at(-1).dur < e.ts + e.dur ||
        stack.at(-1).ts + stack.at(-1).dur <= e.ts)
    )
      stack.pop();
    if (stack.length) stack.at(-1).leaf = false;
    e.depth = stack.length + 1;
    e.leaf = true;
    stack.push(e);
  }
}
function load(data) {
  if (data?.schema !== "serveutil.report/1")
    throw Error("Choose a ServeUtil report (.html or report JSON).");
  report = data;
  document.body.classList.remove('empty');
  $("view").value = data.initial_view === "communication" ? "communication" : "overview";
  let lo = Infinity,
    hi = -Infinity;
  stages = [];
  $("rank").replaceChildren();
  option($("rank"), "", "All ranks");
  const ranks = new Set(),
    depths = new Set();
  for (const t of data.traces) {
    ranks.add(t.rank);
    depthOf(t.events);
    t.hasPhases = t.events.some(
      (e) =>
        e.kind === "stage" &&
        /^(Prefill|Decode|Mixed batch|Extend)(?: step|$)/i.test(e.name),
    );
    for (const e of t.events) {
      if (e.depth) depths.add(e.depth);
      lo = Math.min(lo, e.ts);
      hi = Math.max(hi, e.ts + e.dur);
      if (e.kind === "stage") stages.push({ e, rank: t.rank });
    }
    for (const c of t.counters) {
      for (const p of c.points) {
        lo = Math.min(lo, p[0]);
        hi = Math.max(hi, p[0]);
      }
    }
  }
  $("depth").replaceChildren();
  option($("depth"), "top", "Levels 1–2");
  for (const d of [...depths].sort((a, b) => a - b))
    option($("depth"), String(d), "Level " + d);
  option($("depth"), "leaf", "Leaf only");
  option($("depth"), "all", "All levels");
  $("depth").value = "top";
  for (const rank of ranks) option($("rank"), rank, rankLabel(rank));
  bounds = Number.isFinite(lo) ? [lo, Math.max(hi, lo + 1)] : [0, 1];
  [windowStart, windowEnd] = bounds;
  $("interval").replaceChildren();
  option($("interval"), "", "Full capture");
  const preferred = stages.filter((s) =>
    /prefill|extend|decode|step|forward/i.test(s.e.name),
  );
  for (const s of (preferred.length ? preferred : stages).slice(0, 1000))
    option(
      $("interval"),
      String(stages.indexOf(s)),
      rankLabel(s.rank) + " · " + short(s.e.name, 44) + " · " + fmt(s.e.ts),
    );
  $("title").textContent = data.engine + " timeline";
  $("summary").textContent =
    data.traces.length +
    " trace(s) · " +
    fmt(bounds[1] - bounds[0]) +
    " · " +
    (data.aligned ? "capture time" : "independent trace origins");
  if (Object.keys(data.timing || {}).length > 1) {
    const worst = Math.max(
      ...Object.values(data.timing).map((x) => x.max_error_us),
    );
    $("summary").textContent +=
      " · Clock uncertainty up to ±" + fmt(worst) + " per host";
  }
  $("metadata").textContent = JSON.stringify(
    {
      engine: data.engine,
      status: data.status,
      aligned: data.aligned,
      clockAlignment: data.alignment_note,
      telemetry: data.telemetry,
      calibration: data.baselines,
      calibrationError: data.calibration_error,
      kvTransferMetrics: data.kv_metrics,
      upstreamAnalysis: data.traces
        .filter((t) => Object.keys(t.upstream || {}).length)
        .map((t) => ({ rank: t.rank, ...t.upstream })),
      files: data.traces.map((t) => t.file),
    },
    null,
    2,
  );
  $("png").disabled = false;
  const candidates = stages.filter(
    (s) =>
      s.rank === data.traces[0]?.rank &&
      s.e.domain === "GPU" &&
      s.e.name === "Decode",
  );
  if (candidates.length > 1) {
    const s = candidates[Math.floor(candidates.length / 2)];
    $("interval").value = String(stages.indexOf(s));
    windowStart = s.e.ts;
    windowEnd = s.e.ts + s.e.dur;
    details(s.e.name, {
      Rank: rankLabel(s.rank),
      Duration: fmt(s.e.dur),
      Annotation: s.e.original,
    });
  }
  refresh();
}
function refresh() {
  if (!report) return;
  const groups = new Map(),
    mode = $("view").value,
    rank = $("rank").value,
    depth = $("depth").value;
  $("depthControl").hidden = !["operator", "stage"].includes(mode);
  const matchesDepth = (e) =>
    depth === "all" ||
    (depth === "leaf"
      ? e.leaf
      : depth === "top"
        ? e.depth <= 2
        : e.depth === Number(depth));
  function row(key, label, kind) {
    if (!groups.has(key)) groups.set(key, { label, kind, events: [] });
    return groups.get(key);
  }
  for (const t of report.traces) {
    if (rank && rank !== t.rank) continue;
    for (const e of t.events) {
      const k = e.kind;
      if (mode === "hardware") continue;
      if (
        mode === "overview" &&
        (k === "operator" ||
          (k === "stage" &&
            (t.hasPhases
              ? !/^(Prefill|Decode|Mixed batch|Extend)(?: step|$)/i.test(e.name)
              : e.depth > 2)))
      )
        continue;
      if (
        mode === "kernel" &&
        !["communication", "compute", "copy"].includes(k)
      )
        continue;
      if (mode === "stage" && (k !== "stage" || !matchesDepth(e))) continue;
      if (mode === "communication" && k !== "communication") continue;
      if (mode === "operator" && (k !== "operator" || !matchesDepth(e)))
        continue;
      const group =
        k === "stage" || k === "operator"
          ? k +
            " · " +
            (e.domain || "CPU") +
            " " +
            (mode === "overview" ? "" : e.depth + " / " + e.lane)
          : mode === "overview"
            ? k
            : k + " · GPU " + e.device + " / stream " + e.lane;
      row(
        t.rank + "/" + group,
        rankLabel(t.rank) + " · " + group,
        k,
      ).events.push(e);
    }
    if (mode === "overview" || mode === "hardware")
      for (const c of t.counters) {
        groups.set(
          t.rank + "/counter/" + c.device + "/" + c.lane + "/" + c.name,
          { ...c, label: "R" + t.rank + " · " + c.name, kind: "counter" },
        );
      }
  }
  const selected = report.traces.find((t) => t.rank === rank);
  const hw = report.hardware.filter((c) => {
    if (!selected) return true;
    if (c.node && c.node !== selected.node) return false;
    const deviceMetric = /^(GPU|MIG)-/.test(c.device);
    return !deviceMetric || !selected.devices?.length || selected.devices.includes(c.device);
  });
  if (mode === "overview" || mode === "hardware")
    for (const c of hw) {
      if (
        mode === "overview" &&
        ![
          "hbm.read",
          "hbm.write",
          "hbm.total_estimated",
          "tensor.active",
          "nvlink.tx",
          "rdma.tx",
        ].includes(c.metric)
      )
        continue;
      groups.set("hw/" + c.node + "/" + c.device + "/" + c.metric, {
        ...c,
        label: c.name + " · " + deviceLabel(c),
        kind: "counter",
      });
    }
  resources(hw);
  transferDetails();
  computeResources(rank);
  if (mode === "communication") collectiveResources(rank);
  if (report.transfers?.length && ["overview", "communication"].includes(mode))
    groups.set("kv-handoff", {
      label: "KV handoff · Prefill → Decode",
      kind: "communication",
      events: report.transfers,
    });
  items = report.findings.filter(
    (e) =>
      (!rank ||
        e.rank === rank ||
        (e.node &&
          report.traces.some((t) => t.rank === rank && t.node === e.node))) &&
      e.ts < windowEnd &&
      e.ts + e.dur > windowStart,
  );
  if (items.length)
    groups.set("opportunities", {
      label: "Opportunities",
      kind: "signal",
      events: items,
    });
  rows = [...groups.values()];
  if (mode === "overview")
    rows.sort((a, b) => {
      const order = [
        "stage",
        "compute",
        "communication",
        "copy",
        "signal",
        "counter",
      ];
      return order.indexOf(a.kind) - order.indexOf(b.kind);
    });
  $("issues").replaceChildren();
  $("issueCount").textContent = items.length;
  for (const e of items.slice(0, 40)) {
    const b = document.createElement("button"),
      time = document.createElement("small");
    b.className = "issue";
    b.textContent = e.name;
    time.textContent = rankLabel(e.rank) + " · " + fmt(e.dur);
    b.append(time);
    b.onclick = () => {
      details(e.name, e);
      setWindow(e.ts - e.dur, e.ts + 2 * e.dur);
    };
    $("issues").append(b);
  }
  if (!items.length)
    $("issues").textContent = "No matching opportunities in this interval.";
  draw();
}
function deviceLabel(c) {
  const meta = c.node ? report.telemetry?.[c.node] : report.telemetry;
  const id = meta?.devices?.find((d) => d.uuid === c.device)?.index;
  return (
    (c.node && Object.keys(report.telemetry).length > 1 ? c.node + " / " : "") +
    (id == null ? c.device : "GPU " + id)
  );
}
function resources(hw) {
  const panel = $("rates"),
    root = $("metrics");
  root.replaceChildren();
  panel.hidden = !hw.length;
  for (const c of hw) {
    if (
      ![
        "hbm.read",
        "hbm.write",
        "hbm.total_estimated",
        "tensor.active",
        "nvlink.tx",
        "pcie.tx",
        "rdma.tx",
      ].includes(c.metric)
    )
      continue;
    let area = 0,
      covered = 0;
    for (const [end, value, start] of c.points) {
      const duration = Math.max(
        0,
        Math.min(end, windowEnd) - Math.max(start, windowStart),
      );
      if (duration > 0) {
        area += duration * value;
        covered += duration;
      }
    }
    if (covered === 0) continue;
    const mean = area / covered,
      b = document.createElement("button"),
      label = document.createElement("small"),
      v = document.createElement("strong"),
      fraction = document.createElement("span");
    b.className = "metric";
    label.textContent = deviceLabel(c) + " · " + c.name;
    v.textContent = mean.toFixed(1) + " " + c.unit;
    fraction.textContent = c.baseline
      ? ((100 * mean) / c.baseline.value).toFixed(1) + "% of reference"
      : "";
    b.append(label, v, fraction);
    b.onclick = () =>
      details(c.name, {
        Device: c.device,
        "Selection mean": mean,
        Unit: c.unit,
        "Covered time": fmt(covered),
        Source: c.source,
        "Measured reference": c.baseline,
        Scope: c.scope,
        "Time resolution":
          "Interval-weighted device samples; a short selection may share samples with adjoining work.",
      });
    root.append(b);
  }
  panel.hidden = !root.childElementCount;
}
function transferDetails() {
  const section = $("kvSection"),
    table = $("kvTable");
  table.replaceChildren();
  const records = (report.transfers || []).filter(
    (t) => t.ts < windowEnd && t.ts + t.dur > windowStart,
  );
  for (const row of records.slice(0, 12)) {
    const b = document.createElement("button");
    b.className = "kv-item";
    const bytes =
      row.logical_bytes == null
        ? "Payload unavailable"
        : (row.logical_bytes / 1048576).toFixed(2) + " MiB";
    b.textContent =
      bytes +
      " · Prefill transfer " +
      fmt(row.dur) +
      (row.receiver_to_decode_us >= 0
        ? " · Received → decode " + fmt(row.receiver_to_decode_us)
        : "");
    b.onclick = () => details("KV handoff", row);
    table.append(b);
  }
  for (const m of report.kv_metrics || []) {
    if (m.status !== "ok") continue;
    const b = document.createElement("button");
    b.className = "kv-item";
    b.textContent =
      m.role +
      " · " +
      m.transfers +
      " transfers · " +
      (m.logical_bytes / 1048576).toFixed(2) +
      " MiB in " +
      fmt(m.interval_ns / 1000);
    b.onclick = () => details("NIXL counter interval", m);
    table.append(b);
  }
  section.hidden = !table.childElementCount;
}
function computeResources(rank) {
  const root = $("metrics");
  for (const t of report.traces) {
    if (rank && t.rank !== rank) continue;
    const stats = new Map();
    for (const e of t.events) {
      if (!e.gemm || e.ts < windowStart || e.ts + e.dur > windowEnd) continue;
      const key = e.device + "/" + e.gemm.dtype;
      let g = stats.get(key);
      if (!g) {
        g = {
          flops: 0,
          duration: 0,
          count: 0,
          dtype: e.gemm.dtype,
          device: e.device,
          ref: e.gemm.reference,
        };
        stats.set(key, g);
      }
      g.flops += e.gemm.flops;
      g.duration += e.dur;
      g.count++;
    }
    for (const g of stats.values()) {
      const rate = g.flops / g.duration / 1e6,
        b = document.createElement("button"),
        label = document.createElement("small"),
        v = document.createElement("strong"),
        fraction = document.createElement("span");
      b.className = "metric";
      label.textContent = rankLabel(t.rank) + " · " + g.dtype + " GEMM";
      v.textContent = rate.toFixed(2) + " TFLOP/s";
      fraction.textContent = g.ref
        ? ((100 * rate) / g.ref.value).toFixed(1) + "% of GEMM reference"
        : "";
      b.append(label, v, fraction);
      b.onclick = () =>
        details("Shape-derived GEMM throughput", {
          "Matched kernels": g.count,
          "Estimated FLOPs": g.flops,
          "Sum of GPU kernel durations": fmt(g.duration),
          "Measured GEMM reference": g.ref,
          Scope:
            "Complete dense GEMM operations with one correlated device kernel inside the selection.",
        });
      root.append(b);
      $("rates").hidden = false;
    }
  }
}
function collectiveResources(rank) {
  const root = $("metrics");
  for (const trace of report.traces) {
    if (rank && rank !== trace.rank) continue;
    const groups = new Map();
    for (const event of trace.events) {
      const op = event.collective;
      if (op?.algorithm_GB_s == null || event.ts < windowStart ||
          event.ts + op.duration_us > windowEnd) continue;
      const key = JSON.stringify([event.device, op.group.id, op.op, op.peer_rank]);
      const g = groups.get(key) || { op, count: 0, bytes: 0, us: 0 };
      g.count++;
      g.bytes += op.bytes;
      g.us += op.duration_us;
      groups.set(key, g);
    }
    for (const {op, count, bytes, us} of groups.values()) {
      const card = document.createElement("button");
      card.className = "metric";
      const label = document.createElement("small");
      const value = document.createElement("strong");
      const caption = document.createElement("span");
      const direction = op.peer_rank == null ? "" :
        (op.direction === "recv" ? " ← " : " → ") + "rank " + op.peer_rank;
      label.textContent = rankLabel(trace.rank) + " · " + op.op + direction;
      value.textContent = (bytes / (1000 * us)).toPrecision(3) + " GB/s";
      caption.textContent = count + " operations · group " + op.group.id;
      card.append(label, value, caption);
      card.onclick = () => details(label.textContent, {
        "Logical bytes": bytes,
        "Complete operations": count,
        "Sum of GPU operation durations": fmt(us),
        "Rank group": op.group.ranks,
        "Method": "Logical payload divided by GPU operation time. Select an event for its size, peers and reference."
      });
      root.append(card);
      $("rates").hidden = false;
    }
  }
}
function setWindow(a, b) {
  const span = Math.min(bounds[1] - bounds[0], Math.max(0.01, b - a));
  windowStart = Math.max(bounds[0], Math.min(a, bounds[1] - span));
  windowEnd = windowStart + span;
  refresh();
}
function draw() {
  if (!report) return;
  const w = Math.max(330, Math.floor(canvas.parentElement.clientWidth)),
    height = 34 + rows.length * 32,
    ratio = Math.min(window.devicePixelRatio || 1, 2);
  // Limit the backing store on captures containing many streams.
  const scale = Math.min(ratio, 16000 / Math.max(height, 1));
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(height * scale);
  canvas.style.height = height + "px";
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, w, height);
  titleWidth = Math.min(205, w * 0.34);
  const left = titleWidth,
    width = w - left - 12,
    delta = windowEnd - windowStart;
  const x = (t) => left + ((t - windowStart) / delta) * width;
  ctx.font = "11px system-ui";
  ctx.fillStyle = "#506575";
  ctx.fillText("Capture time", 10, 19);
  for (let i = 0; i <= 5; i++) {
    const xx = left + (width * i) / 5;
    ctx.fillStyle = "#e2e9ed";
    ctx.fillRect(xx, 25, 1, height);
    ctx.fillStyle = "#506575";
    ctx.fillText(
      fmt(windowStart + (delta * i) / 5),
      Math.min(xx + 3, w - 55),
      19,
    );
  }
  targets = [];
  rows.forEach((r, index) => {
    const y = 29 + index * 32;
    ctx.fillStyle = "#e1e8eb";
    ctx.fillRect(0, y + 31, w, 1);
    ctx.fillStyle = "#304957";
    ctx.fillText(short(r.label, Math.floor(left / 6.4)), 9, y + 19);
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, y, width, 30);
    ctx.clip();
    if (r.kind === "counter") {
      const points = r.points.filter((p) =>
        p.length >= 3
          ? p[2] <= windowEnd && p[0] >= windowStart
          : p[0] >= windowStart && p[0] <= windowEnd,
      );
      let max = 0;
      for (const p of points) max = Math.max(max, p[1]);
      ctx.strokeStyle = color.counter;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      points.forEach((p, i) => {
        const xx = x(p[0]),
          yy = y + 28 - (23 * p[1]) / (max || 1);
        if (p.length >= 3) {
          ctx.moveTo(x(p[2]), yy);
          ctx.lineTo(xx, yy);
        } else if (i) ctx.lineTo(xx, yy);
        else ctx.moveTo(xx, yy);
        if (points.length < 180) ctx.fillRect(xx - 1, yy - 1, 2, 2);
      });
      ctx.stroke();
      if (points.length) {
        const unit = r.unit || "";
        ctx.fillStyle = "#385c55";
        ctx.font = "10px system-ui";
        ctx.fillText(
          "↑ " +
            max.toLocaleString(undefined, { maximumFractionDigits: 2 }) +
            " " +
            unit,
          left + 4,
          y + 10,
        );
      }
      targets.push({ y, row: r, points });
    } else {
      const visible = [];
      for (const e of r.events) {
        if (e.ts >= windowEnd || e.ts + e.dur <= windowStart) continue;
        const a = Math.max(left, x(e.ts)),
          b = Math.min(w - 12, x(e.ts + e.dur)),
          sw = Math.max(1, b - a);
        ctx.fillStyle = color[r.kind];
        ctx.fillRect(a, y + 5, sw, 21);
        if (sw > 42) {
          ctx.fillStyle = "white";
          ctx.fillText(short(e.name, Math.floor(sw / 6.8)), a + 3, y + 20);
        }
        visible.push({ a, b, e });
      }
      targets.push({ y, row: r, visible });
    }
    ctx.restore();
  });
  $("range").textContent = fmt(windowStart) + " – " + fmt(windowEnd);
}
function selectAt(px, py) {
  const hit = targets.find((t) => py >= t.y && py < t.y + 32);
  if (!hit) return;
  if (hit.points) {
    const ts =
      windowStart +
      ((px - titleWidth) / (canvas.clientWidth - titleWidth - 12)) *
        (windowEnd - windowStart);
    let best = null;
    for (const p of hit.points)
      if (!best || Math.abs(p[0] - ts) < Math.abs(best[0] - ts)) best = p;
    if (best) {
      const r = hit.row;
      details(r.name, {
        Device: r.device,
        Time: fmt(best[0]),
        Value: best[1],
        Unit: r.unit,
        Source: r.source,
        Scope: r.scope,
        "Sample window": best.length > 2 ? fmt(best[0] - best[2]) : undefined,
        Baseline: r.baseline?.value,
        "Baseline utilization": r.baseline?.value
          ? ((100 * best[1]) / r.baseline.value).toFixed(1) + "%"
          : undefined,
      });
    }
    return;
  }
  const target = hit.visible.findLast(
    (t) => px >= t.a && px <= Math.max(t.a + 1, t.b),
  );
  if (target?.e.collective) {
    const op = target.e.collective;
    details(op.op, {
      "Kernel": target.e.original || target.e.name,
      "Direction": op.direction, "Peer rank": op.peer_rank,
      "Group": op.group, "Sequence": op.seq, "Data type": op.dtype,
      "Logical bytes": op.bytes, "Operation time": op.duration_us == null ? null : fmt(op.duration_us),
      "GPU kernels": op.kernel_count,
      "Algorithm bandwidth (GB/s)": op.algorithm_GB_s,
      "NCCL bus bandwidth (GB/s)": op.bus_GB_s,
      "Relative to nccl-tests": op.reference_pct == null ? null : op.reference_pct.toFixed(1) + "%",
      "Measured reference": op.reference,
      "Part of a multi-kernel operation": op.part_of_operation,
      "Unavailable": op.unavailable
    });
  } else if (target)
    details(target.e.name, {
      ...target.e,
      ts: fmt(target.e.ts),
      dur: fmt(target.e.dur),
    });
}
function loc(ev) {
  const box = canvas.getBoundingClientRect();
  return [ev.clientX - box.left, ev.clientY - box.top];
}
canvas.onpointerdown = (e) => {
  const [x] = loc(e);
  drag = x;
  canvas.setPointerCapture(e.pointerId);
};
canvas.onpointerup = (e) => {
  if (drag === null) return;
  const [x, y] = loc(e),
    a = drag;
  drag = null;
  if (Math.abs(x - a) < 5) {
    selectAt(x, y);
    return;
  }
  const toTime = (px) =>
    windowStart +
    ((px - titleWidth) / (canvas.clientWidth - titleWidth - 12)) *
      (windowEnd - windowStart);
  setWindow(toTime(Math.min(x, a)), toTime(Math.max(x, a)));
};
canvas.onpointercancel = () => {
  drag = null;
};
for (const id of ["view", "rank", "depth"]) $(id).onchange = refresh;
$("interval").onchange = (e) => {
  if (!e.target.value) {
    setWindow(...bounds);
    return;
  }
  const s = stages[Number(e.target.value)];
  $("rank").value = s.rank;
  setWindow(s.e.ts, s.e.ts + s.e.dur);
};
$("fit").onclick = () => {
  $("interval").value = "";
  setWindow(...bounds);
};
function zoom(f) {
  const mid = (windowEnd + windowStart) / 2,
    half = ((windowEnd - windowStart) * f) / 2;
  setWindow(mid - half, mid + half);
}
$("plus").onclick = () => zoom(0.5);
$("minus").onclick = () => zoom(2);
$("png").onclick = () => {
  const figure = document.createElement("canvas");
  const w = canvas.clientWidth,
    h = canvas.clientHeight,
    margin = 20;
  const metrics = [...$("metrics").children].map((e) =>
    e.innerText.replaceAll("\n", "   "),
  );
  const notes = items
    .slice(0, 3)
    .map(
      (e) =>
        e.name +
        " · " +
        rankLabel(e.rank) +
        " · " +
        fmt(e.dur) +
        " at " +
        fmt(e.ts),
    );
  const footer = metrics.length + notes.length;
  const scale = Math.min(
    2,
    16000 / (h + 145 + 24 * footer),
    Math.sqrt(25e6 / (w * (h + 145 + 24 * footer))),
  );
  figure.width = Math.floor(w * scale);
  figure.height = Math.floor((h + 145 + 24 * footer) * scale);
  const g = figure.getContext("2d");
  g.scale(scale, scale);
  g.fillStyle = "white";
  g.fillRect(0, 0, w, figure.height / scale);
  g.fillStyle = "#152f3e";
  g.fillRect(0, 0, w, 72);
  g.fillStyle = "white";
  g.font = "bold 20px system-ui";
  g.fillText("ServeUtil  ·  " + $("title").innerText, margin, 29);
  g.fillStyle = "#c7d9df";
  g.font = "13px system-ui";
  g.fillText(
    $("view").selectedOptions[0].text +
      "   |   " +
      ($("rank").selectedOptions[0]?.text || "All ranks") +
      "   |   " +
      fmt(windowStart) +
      " – " +
      fmt(windowEnd),
    margin,
    52,
  );
  g.drawImage(canvas, 0, 76, w, h);
  let y = h + 99;
  for (const [name, lines] of [
    ["Resources in selection", metrics],
    ["Optimization opportunities", notes],
  ]) {
    if (!lines.length) continue;
    g.fillStyle = "#1b4048";
    g.font = "bold 13px system-ui";
    g.fillText(name, margin, y);
    y += 19;
    g.fillStyle = "#41535a";
    g.font = "12px system-ui";
    for (const line of lines) {
      g.fillText(line, margin, y);
      y += 22;
    }
  }
  figure.toBlob((blob) => {
    if (!blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "serveutil-timeline.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
};
async function unpack(raw) {
  const data = JSON.parse(raw);
  if (data?.encoding !== "gzip+base64") return data;
  const binary = atob(data.payload);
  const bytes = Uint8Array.from(binary, (ch) => ch.charCodeAt(0));
  const stream = new Blob([bytes])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}
async function importFile(file) {
  if (file.name.endsWith(".html")) {
    const html = await file.text();
    const content =
      /<script\s+id="capture-data"\s+type="application\/json">([\s\S]*?)<\/script>/i.exec(
        html,
      );
    if (!content)
      throw Error("This HTML file does not contain a ServeUtil capture.");
    return unpack(content[1]);
  }
  if (
    !file.name.endsWith(".gz") &&
    /"schema"\s*:\s*"serveutil\.report\/1"/.test(
      await file.slice(0, 1000).text(),
    )
  ) {
    return JSON.parse(await file.text());
  }
  const url = URL.createObjectURL(
    new Blob([$("import-worker").textContent], { type: "text/javascript" }),
  );
  const worker = new Worker(url);
  try {
    return await new Promise((resolve, reject) => {
      worker.onmessage = ({ data }) =>
        data.error ? reject(Error(data.error)) : resolve(data);
      worker.onerror = (e) => reject(Error(e.message));
      worker.postMessage(file);
    });
  } finally {
    worker.terminate();
    URL.revokeObjectURL(url);
  }
}
async function openFile(file) {
  if (!file) return;
  const status = document.body.classList.contains("empty") ? $("importStatus") : $("summary");
  try {
    status.textContent = "Reading " + file.name + "…";
    load(await importFile(file));
  } catch (err) {
    status.textContent = err.message;
  }
}
$("choose").onclick = () => $("file").click();
$("file").onchange = async (e) => {
  await openFile(e.target.files[0]);
  e.target.value = "";
};
window.addEventListener("dragover", (e) => {
  if (!e.dataTransfer.types.includes("Files")) return;
  e.preventDefault();
  document.body.classList.add("dragging");
});
window.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) document.body.classList.remove("dragging");
});
window.addEventListener("drop", (e) => {
  if (!e.dataTransfer.files.length) return;
  e.preventDefault();
  document.body.classList.remove("dragging");
  openFile(e.dataTransfer.files[0]);
});
new ResizeObserver(draw).observe(canvas.parentElement);
unpack($("capture-data").textContent)
  .then((initial) => {
    if (initial) load(initial);
  })
  .catch((err) => {
    $("importStatus").textContent = "Unable to read capture: " + err.message;
  });
