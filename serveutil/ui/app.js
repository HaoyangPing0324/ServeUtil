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
  links = [],
  eventLinks = new WeakMap(),
  chosenLink = null,
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
  if (!t?.node) return "Rank " + (t?.global_rank ?? rank);
  const parts = rank.split("/");
  return (
    (new Set(report.traces.map((t) => t.node)).size > 1 ? t.node + " / " : "") +
    (t.role && t.role !== "worker" ? t.role + " / " : "") +
    "Rank " +
    (t.global_rank ?? parts.at(-1))
  );
}
function displayName(event) {
  return $("names").value === "original"
    ? event.original || event.name
    : event.name;
}
function measure(value) {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  if (Math.abs(value) < 0.001) return value.toExponential(2);
  return Number(value.toPrecision(3)).toLocaleString("en-US", {
    maximumSignificantDigits: 3,
  });
}
function rateLabel(value, unit) {
  if (unit === "GB/s" && value > 0 && value < 1) {
    if (value >= 0.001) return measure(value * 1000) + " MB/s";
    if (value >= 0.000001) return measure(value * 1000000) + " kB/s";
    return measure(value * 1e9) + " B/s";
  }
  return measure(value) + (unit ? " " + unit : "");
}
function visibleTrace(t) {
  return (
    (!$("node").value || t.node === $("node").value) &&
    (!$("rank").value || t.rank === $("rank").value)
  );
}
function repopulateRanks() {
  $("rank").replaceChildren();
  option($("rank"), "", "All ranks");
  const ranks = new Set(
    report.traces
      .filter((t) => !$("node").value || t.node === $("node").value)
      .map((t) => t.rank),
  );
  for (const rank of ranks) option($("rank"), rank, rankLabel(rank));
}
function updateIntervals() {
  $("interval").replaceChildren();
  option($("interval"), "", "Full capture");
  const selectedRanks = new Set(
    report.traces.filter(visibleTrace).map((t) => t.rank),
  );
  const available = stages
    .map((s, i) => ({ ...s, i }))
    .filter((s) => selectedRanks.has(s.rank));
  const preferred = available.filter((s) =>
    /prefill|extend|decode|step|forward/i.test(s.e.name),
  );
  for (const s of (preferred.length ? preferred : available).slice(0, 1000))
    option(
      $("interval"),
      String(s.i),
      rankLabel(s.rank) +
        " · " +
        short(displayName(s.e), 44) +
        " · " +
        fmt(s.e.ts),
    );
}
function selectedClock(us) {
  const trace = report.traces.find((t) => t.rank === $("rank").value);
  const local = $("clock").value === "local_utc";
  const epoch = local ? trace?.rank_origin_unix_ns : report.time_origin_unix_ns;
  if ((local || $("clock").value === "utc") && epoch) {
    const elapsed = us - (local ? trace.rank_origin_us : 0);
    const ns = BigInt(epoch) + BigInt(Math.round(elapsed * 1000));
    const iso = new Date(Number(ns / 1000000n)).toISOString();
    return (
      iso.slice(11, 19) +
      "." +
      String(ns % 1000000000n)
        .padStart(9, "0")
        .slice(0, 6)
    );
  }
  return fmt(
    us -
      ($("clock").value === "rank" && trace ? (trace.rank_origin_us ?? 0) : 0),
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
  for (const t of data.traces) {
    if (
      !t.rank_origin_unix_ns &&
      /^[0-9]+$/.test(t.raw_epoch) &&
      Number.isFinite(t.raw_begin)
    ) {
      t.rank_origin_unix_ns = String(
        BigInt(t.raw_epoch) + BigInt(Math.round(t.raw_begin * 1000)),
      );
      t.rank_origin_us ??= 0;
      if (!data.time_origin_unix_ns && data.traces.length === 1)
        data.time_origin_unix_ns = t.rank_origin_unix_ns;
    }
  }
  chosenLink = null;
  $("node").replaceChildren();
  option($("node"), "", "All nodes");
  const nodes = [...new Set(data.traces.map((t) => t.node).filter(Boolean))];
  for (const node of nodes) option($("node"), node, node);
  $("nodeControl").hidden = !nodes.length;
  $("clock").replaceChildren();
  option(
    $("clock"),
    "capture",
    data.aligned ? "Capture elapsed" : "Independent origins",
  );
  option($("clock"), "rank", "Rank elapsed");
  if (data.time_origin_unix_ns) option($("clock"), "utc", "UTC");
  else if (data.traces.some((t) => t.rank_origin_unix_ns))
    option($("clock"), "local_utc", "Rank-local UTC");
  $("names").value = "short";
  document.body.classList.remove("empty");
  $("view").value =
    data.initial_view === "communication" ? "communication" : "overview";
  let lo = Infinity,
    hi = -Infinity;
  stages = [];
  $("rank").replaceChildren();
  option($("rank"), "", "All ranks");
  const depths = new Set();
  for (const t of data.traces) {
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
  repopulateRanks();
  links = buildLinks(data);
  eventLinks = new WeakMap();
  for (const link of links) {
    if (link.transfer) eventLinks.set(link.transfer, link);
    for (const end of link.ends) if (end.event) eventLinks.set(end.event, link);
  }
  if (!Number.isFinite(lo))
    for (const series of data.hardware)
      for (const [end, , start] of series.points) {
        lo = Math.min(lo, start ?? end);
        hi = Math.max(hi, end);
      }
  bounds = Number.isFinite(lo) ? [lo, Math.max(hi, lo + 1)] : [0, 1];
  [windowStart, windowEnd] = bounds;
  updateIntervals();
  $("title").textContent = data.engine + " timeline";
  $("summary").textContent =
    data.traces.length +
    " trace(s) · " +
    fmt(bounds[1] - bounds[0]) +
    " · " +
    (data.aligned ? "capture time" : "independent trace origins");
  if (data.status === "failed") {
    $("summary").textContent +=
      " · Incomplete capture — see collection details";
  }
  if (data.traces.some((t) => t.native_only))
    $("summary").textContent += " · Open native tracks in Full Perfetto";
  if (data.traces.some((t) => t.parse_error))
    $("summary").textContent +=
      " · Unreadable original trace (see collection details)";
  if (data.traces.some((t) => t.device_activity_unavailable))
    $("summary").textContent +=
      " · Missing device activity (see collection details)";
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
      error: data.error || undefined,
      collectionErrors: data.collection_errors?.length
        ? data.collection_errors
        : undefined,
      unreadableTraces: data.traces
        .filter((t) => t.parse_error)
        .map((t) => ({ file: t.file, rank: t.rank, reason: t.parse_error })),
      missingDeviceActivity: data.traces
        .filter((t) => t.device_activity_unavailable)
        .map((t) => ({
          rank: t.rank,
          file: t.file,
          reason: t.device_activity_unavailable,
        })),
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
    if (!visibleTrace(t)) continue;
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
        const key = JSON.stringify([
          t.node,
          t.rank,
          c.device,
          c.lane,
          c.name,
          report.aligned ? null : t.file,
        ]);
        const existing = groups.get(key);
        if (existing) {
          existing.points = existing.points
            .concat(c.points)
            .sort((a, b) => a[0] - b[0]);
        } else {
          groups.set(key, {
            ...c,
            label: rankLabel(t.rank) + " · " + c.name,
            kind: "counter",
          });
        }
      }
  }
  const selected = report.traces.find((t) => t.rank === rank);
  const hw = report.hardware.filter((c) => {
    if ($("node").value && c.node !== $("node").value) return false;
    if (!selected) return true;
    if (c.node && c.node !== selected.node) return false;
    const deviceMetric = /^(GPU|MIG)-/.test(c.device);
    return (
      !deviceMetric ||
      !selected.devices?.length ||
      selected.devices.includes(c.device)
    );
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
  resources(hw, mode);
  const transfers = (report.transfers || []).filter(transferInScope);
  transferDetails(transfers);
  showLinks(mode);
  if (["overview", "hardware", "kernel"].includes(mode)) computeResources(rank);
  if (mode === "communication") collectiveResources(rank);
  if (transfers.length && ["overview", "communication"].includes(mode))
    for (const t of transfers) {
      if (roleInScope(t.source_node, "prefill")) {
        const key = "kv/source/" + (t.source_node || "");
        const send = row(
          key,
          (t.source_node || "Prefill") + " · KV send",
          "communication",
        );
        send.events.push(t);
      }
      if (t.receive_us != null && roleInScope(t.destination_node, "decode")) {
        const key = "kv/receive/" + (t.destination_node || "");
        const receive = row(
          key,
          (t.destination_node || "Decode") + " · KV received",
          "communication",
        );
        receive.events.push({
          name: "KV received",
          ts: t.receive_us,
          dur: 0.001,
          transfer: t,
        });
      }
    }
  const visibleTraces = report.traces.filter(visibleTrace);
  items = report.findings.filter((e) => {
    if (e.ts >= windowEnd || e.ts + e.dur <= windowStart) return false;
    if (e.scope && e.metric)
      return hw.some(
        (series) =>
          (series.node ?? null) === (e.node ?? null) &&
          series.device === e.device &&
          series.name === e.metric &&
          series.source === e.source,
      );
    return visibleTraces.some(
      (t) => t.rank === e.rank && (!e.node || t.node === e.node),
    );
  });
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
function resources(hw, mode) {
  const panel = $("rates"),
    root = $("metrics");
  root.replaceChildren();
  panel.hidden = !hw.length;
  for (const c of hw) {
    const primary = [
      "hbm.read",
      "hbm.write",
      "hbm.total_estimated",
      "tensor.active",
      "nvlink.tx",
      "pcie.tx",
      "rdma.tx",
      "netdev.tx",
    ];
    if (
      mode === "communication" &&
      !/^(nvlink|pcie|rdma|netdev)\./.test(c.metric)
    )
      continue;
    if (
      !["hardware", "communication"].includes(mode) &&
      (mode !== "overview" || !primary.includes(c.metric))
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
    v.textContent = rateLabel(mean, c.unit);
    fraction.textContent = c.baseline
      ? measure((100 * mean) / c.baseline.value) + "% of reference"
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
function importedLinks(traces) {
  const unique = new Map(),
    groups = new Map(),
    linked = [];
  for (const t of traces)
    unique.set(t.global_rank, (unique.get(t.global_rank) || 0) + 1);
  for (const [ti, t] of traces.entries()) {
    if (
      t.raw_begin == null ||
      !Number.isInteger(t.global_rank) ||
      unique.get(t.global_rank) !== 1
    )
      continue;
    for (const [ei, e] of t.events.entries()) {
      const op = e.collective,
        ranks = op?.group.ranks;
      if (
        !op?.comms_id ||
        !op.group.id ||
        !Number.isSafeInteger(op.seq) ||
        !Number.isSafeInteger(op.bytes) ||
        !ranks?.includes(t.global_rank) ||
        ranks.length !== op.group.size ||
        new Set(ranks).size !== ranks.length
      )
        continue;
      const kind = ["send", "recv"].includes(op.op) ? "send_recv" : op.op;
      const key = JSON.stringify([
        op.comms_id,
        op.group.id,
        ranks,
        op.seq,
        kind,
        op.dtype,
        op.bytes,
      ]);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push({ ti, ei, rank: t.global_rank, op, kind });
    }
  }
  for (const parts of groups.values()) {
    const { kind, op } = parts[0];
    if (
      parts.length < 2 ||
      new Set(parts.map((p) => p.rank)).size !== parts.length
    )
      continue;
    if (kind === "send_recv") {
      parts.sort(
        (a, b) => Number(a.op.op === "recv") - Number(b.op.op === "recv"),
      );
      const [send, recv] = parts;
      if (parts.length !== 2 || send.op.op !== "send" || recv.op.op !== "recv")
        continue;
      function unambiguousPeer(endpoint, actual) {
        const n = endpoint.op.peer_rank,
          ranks = op.group.ranks;
        if (!Number.isSafeInteger(n) || n < 0) return false;
        const candidates = new Set();
        if (ranks.includes(n)) candidates.add(n);
        if (n < ranks.length) candidates.add(ranks[n]);
        return candidates.size === 1 && candidates.has(actual);
      }
      if (
        !unambiguousPeer(send, recv.rank) ||
        !unambiguousPeer(recv, send.rank)
      )
        continue;
    } else if (parts.length !== op.group.ranks.length) continue;
    linked.push({
      kind,
      evidence: "imported_ids",
      endpoints: parts.map((p) => [p.ti, p.ei]),
    });
  }
  return linked;
}
function buildLinks(data) {
  const result = [],
    paired = new Set();
  const endpoint = (t, e) => ({
    trace: t,
    node: t.node,
    rank: t.rank,
    event: e,
    ts: e.ts,
    dur: e.collective?.duration_us ?? e.dur,
  });
  for (const match of data.communication_links ?? importedLinks(data.traces)) {
    const ends = match.endpoints.map(([ti, ei]) => {
      const t = data.traces[ti],
        e = t?.events[ei];
      return e?.collective ? endpoint(t, e) : null;
    });
    if (ends.some((e) => !e)) continue;
    for (const end of ends) paired.add(end.event);
    result.push({
      kind: match.kind,
      evidence: match.evidence,
      ends,
      op: ends[0].event.collective,
    });
  }
  for (const t of data.traces)
    for (const event of t.events) {
      const op = event.collective;
      if (!op || op.part_of_operation || paired.has(event)) continue;
      result.push({
        kind: op.op,
        evidence: "single",
        ends: [endpoint(t, event)],
        op,
      });
    }
  function roleEnd(row, node, role, ts) {
    const candidates =
      node == null
        ? []
        : data.traces.filter((t) => t.node === node && t.role === role);
    const trace = candidates.length === 1 ? candidates[0] : undefined;
    return {
      trace,
      node,
      rank: trace?.rank,
      role,
      ts,
      dur: role === "prefill" ? row.dur : 0,
    };
  }
  for (const row of [
    ...(data.transfers || []),
    ...(data.unplaced_transfers || []),
  ]) {
    const ends = [roleEnd(row, row.source_node, "prefill", row.ts)];
    if (row.destination_node || row.receive_end_ns || row.decode_start_ns)
      ends.push(
        roleEnd(
          row,
          row.destination_node,
          "decode",
          row.receive_us ?? row.decode_us,
        ),
      );
    result.push({
      kind: "kv",
      evidence: ends.length === 2 ? "otel" : "single",
      ends,
      transfer: row,
    });
  }
  return result;
}
const linkTitles = {
  send_recv: "Send → receive",
  all_reduce: "AllReduce",
  all_gather: "AllGather",
  all_to_all: "AllToAll",
  reduce_scatter: "ReduceScatter",
  broadcast: "Broadcast",
  kv: "KV handoff",
};
const evidenceNames = {
  imported_ids: "Matching IDs in imported traces; shared capture unverified",
  comms_id: "Matching communication ID",
  group_sequence: "Inferred from group and sequence",
  single: "One recorded endpoint",
  otel: "Matching request trace and bootstrap room",
};
function routeLabel(end) {
  if (end.role) return (end.node ? end.node + " / " : "") + end.role;
  return end.trace ? rankLabel(end.trace.rank) : end.node || "Unknown node";
}
function byteLabel(n) {
  if (n == null) return "Payload unavailable";
  if (n >= 1048576) return measure(n / 1048576) + " MiB";
  if (n >= 1024) return measure(n / 1024) + " KiB";
  return n + " B";
}
function linkMatches(link) {
  const node = $("node").value,
    rank = $("rank").value;
  return (
    link.ends.some(
      (e) => (!node || e.node === node) && (!rank || e.rank === rank),
    ) &&
    (link.ends.every((e) => e.ts == null) ||
      link.ends.some(
        (e) =>
          e.ts != null &&
          e.ts <= windowEnd &&
          e.ts + (e.dur || 0) >= windowStart,
      ))
  );
}
function inspectLink(link) {
  chosenLink = link;
  const box = $("linkInspector");
  box.replaceChildren();
  const title = document.createElement("strong"),
    context = document.createElement("p");
  title.textContent =
    (linkTitles[link.kind] || link.kind) +
    " · " +
    byteLabel(link.transfer?.logical_bytes ?? link.op?.bytes);
  const receipt = link.transfer;
  context.textContent = [
    evidenceNames[link.evidence],
    receipt
      ? "Trace " +
        (receipt.trace_id || "?") +
        " · room " +
        (receipt.bootstrap_room || "?")
      : "Group " +
        (link.op.group?.id ?? "?") +
        " · sequence " +
        (link.op.seq ?? "?"),
    link.ends.length > 1 && !report.aligned
      ? "Independent rank clocks; cross-rank time order unavailable"
      : null,
  ]
    .filter(Boolean)
    .join(" · ");
  box.append(title, context);
  const path = document.createElement("div");
  path.className = "link-endpoints";
  for (const [index, end] of link.ends.entries()) {
    if (index) {
      const arrow = document.createElement("span");
      arrow.className = "link-connector";
      arrow.textContent = ["send_recv", "kv"].includes(link.kind) ? "→" : "↔";
      path.append(arrow);
    }
    const b = document.createElement("button"),
      node = document.createElement("strong"),
      note = document.createElement("small");
    node.textContent = routeLabel(end);
    const action =
      end.event?.collective?.direction || end.role || "Participant";
    note.textContent =
      action +
      (end.ts != null ? " · " + selectedClock(end.ts) : " · Time unavailable");
    b.append(node, note);
    b.onclick = () => {
      const trace = end.trace;
      if (trace) {
        $("node").value = trace.node || "";
        repopulateRanks();
        $("rank").value = trace.rank;
        updateIntervals();
      }
      details(routeLabel(end), {
        Evidence: evidenceNames[link.evidence],
        Node: end.node,
        "Global rank": trace?.global_rank,
        Role: end.role || trace?.role,
        Operation: link.kind,
        "Original kernel": end.event?.original || end.event?.name,
        "Communication ID": link.op?.comms_id,
        "Group members": link.op?.group?.ranks,
        Sequence: link.op?.seq,
        Bytes: receipt?.logical_bytes ?? end.event?.collective?.bytes,
        "Trace ID": receipt?.trace_id,
        "Bootstrap room": receipt?.bootstrap_room,
        "Endpoint attribution": receipt
          ? "Scheduler span on this serving role"
          : "Device kernel in the rank trace",
        "Clock uncertainty (µs)": report.timing?.[end.node]?.max_error_us,
      });
      if (end.ts != null) {
        const pad = Math.max(40, (end.dur || 0) * 0.8);
        setWindow(end.ts - pad, end.ts + (end.dur || 0) + pad);
      } else refresh();
    };
    path.append(b);
  }
  box.append(path);
  if (
    report.aligned &&
    link.ends.length > 1 &&
    link.ends.every((e) => e.ts != null)
  ) {
    const zoom = document.createElement("button");
    zoom.className = "link-focus";
    zoom.textContent = "Focus operation";
    zoom.onclick = () => {
      const a = Math.min(...link.ends.map((e) => e.ts));
      const b = Math.max(...link.ends.map((e) => e.ts + (e.dur || 0)));
      const pad = Math.max(25, (b - a) * 0.25);
      setWindow(a - pad, b + pad);
    };
    box.append(zoom);
  }
}
function showLinks(mode) {
  const section = $("linksSection"),
    table = $("linkList"),
    inspector = $("linkInspector");
  table.replaceChildren();
  inspector.replaceChildren();
  section.hidden = mode !== "communication" || !links.length;
  if (section.hidden) return;
  const families = new Map();
  for (const link of links) {
    if (!linkMatches(link)) continue;
    const key = JSON.stringify([
      link.kind,
      link.evidence,
      link.op?.group?.id,
      ...link.ends.map((e) => [e.node, e.rank, e.role]),
    ]);
    if (!families.has(key)) families.set(key, []);
    families.get(key).push(link);
  }
  const n = [...families.values()].reduce(
    (sum, group) => sum + group.length,
    0,
  );
  $("linksCount").textContent = n + " operations in selection";
  if (!n) return;
  const select = document.createElement("select");
  select.setAttribute("aria-label", "Communication path");
  const all = [...families.values()];
  for (const [i, group] of all.entries()) {
    const first = group[0];
    option(
      select,
      String(i),
      (linkTitles[first.kind] || first.kind) +
        " · " +
        first.ends
          .map(routeLabel)
          .join(["send_recv", "kv"].includes(first.kind) ? " → " : " ↔ ") +
        " · " +
        group.length,
    );
  }
  const chooser = document.createElement("input"),
    count = document.createElement("span");
  chooser.type = "number";
  chooser.min = "1";
  chooser.step = "1";
  chooser.setAttribute("aria-label", "Operation number");
  const selectedFamily = all.findIndex((group) => group.includes(chosenLink));
  select.selectedIndex = Math.max(0, selectedFamily);
  chooser.value = String(
    Math.max(0, all[select.selectedIndex].indexOf(chosenLink)) + 1,
  );
  function choose() {
    const group = all[select.selectedIndex];
    chooser.max = String(group.length);
    const index = Math.max(
      0,
      Math.min(group.length - 1, Math.floor(Number(chooser.value) || 1) - 1),
    );
    chooser.value = String(index + 1);
    count.textContent = "of " + group.length;
    inspectLink(group[index]);
  }
  select.onchange = () => {
    chooser.value = "1";
    choose();
  };
  chooser.onchange = choose;
  const lab = document.createElement("label");
  lab.append("Operation ", chooser, " ", count);
  table.append(select, lab);
  choose();
}
function transferInScope(t) {
  return (
    roleInScope(t.source_node, "prefill") ||
    roleInScope(t.destination_node, "decode")
  );
}
function roleInScope(node, role) {
  const selected = report.traces.find((tr) => tr.rank === $("rank").value);
  const selectedNode = selected?.node || $("node").value;
  if (selectedNode && selectedNode !== node) return false;
  return (
    !selected ||
    (selected.role === role &&
      report.traces.filter((t) => t.node === node && t.role === role).length ===
        1)
  );
}
function transferDetails(transfers) {
  const section = $("kvSection"),
    table = $("kvTable");
  table.replaceChildren();
  const records = transfers.filter(
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
    if (m.node && !roleInScope(m.node, m.role)) continue;
    const role = report.traces.find((t) => t.rank === $("rank").value)?.role;
    if (role && m.role && role !== m.role) continue;
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
    if (!visibleTrace(t)) continue;
    const stats = new Map();
    for (const e of t.events) {
      if (!e.gemm || e.ts < windowStart || e.ts + e.dur > windowEnd) continue;
      const key = e.device + "/" + e.gemm.dtype;
      let g = stats.get(key);
      if (!g) {
        g = {
          operations: 0,
          unit: e.gemm.rate_unit || "TFLOP/s",
          duration: 0,
          count: 0,
          dtype: e.gemm.dtype,
          device: e.device,
          ref: e.gemm.reference,
        };
        stats.set(key, g);
      }
      g.operations += e.gemm.operations ?? e.gemm.flops;
      g.duration += e.dur;
      g.count++;
    }
    for (const g of stats.values()) {
      const rate = g.operations / g.duration / 1e6,
        b = document.createElement("button"),
        label = document.createElement("small"),
        v = document.createElement("strong"),
        fraction = document.createElement("span");
      b.className = "metric";
      label.textContent = rankLabel(t.rank) + " · " + g.dtype + " GEMM";
      v.textContent = rateLabel(rate, g.unit);
      fraction.textContent = g.ref
        ? measure((100 * rate) / g.ref.value) + "% of GEMM reference"
        : "";
      b.append(label, v, fraction);
      b.onclick = () =>
        details("Shape-derived GEMM throughput", {
          "Matched kernels": g.count,
          [g.unit === "TOP/s"
            ? "Estimated integer operations"
            : "Estimated FLOPs"]: g.operations,
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
    if (!visibleTrace(trace)) continue;
    const groups = new Map();
    for (const event of trace.events) {
      const op = event.collective;
      if (
        op?.algorithm_GB_s == null ||
        event.ts < windowStart ||
        event.ts + op.duration_us > windowEnd
      )
        continue;
      const key = JSON.stringify([
        event.device,
        op.group.id,
        op.op,
        op.peer_rank,
      ]);
      const g = groups.get(key) || { op, count: 0, bytes: 0, us: 0 };
      g.count++;
      g.bytes += op.bytes;
      g.us += op.duration_us;
      groups.set(key, g);
    }
    for (const { op, count, bytes, us } of groups.values()) {
      const card = document.createElement("button");
      card.className = "metric";
      const label = document.createElement("small");
      const value = document.createElement("strong");
      const caption = document.createElement("span");
      const direction =
        op.peer_rank == null
          ? ""
          : (op.direction === "recv" ? " ← " : " → ") + "rank " + op.peer_rank;
      label.textContent = rankLabel(trace.rank) + " · " + op.op + direction;
      value.textContent = rateLabel(bytes / (1000 * us), "GB/s");
      caption.textContent = count + " operations · group " + op.group.id;
      card.append(label, value, caption);
      card.onclick = () =>
        details(label.textContent, {
          "Logical bytes": bytes,
          "Complete operations": count,
          "Sum of GPU operation durations": fmt(us),
          "Rank group": op.group.ranks,
          Method:
            "Logical payload divided by GPU operation time. Select an event for its size, peers and reference.",
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
  ctx.fillText($("clock").selectedOptions[0].text, 10, 19);
  for (let i = 0; i <= 5; i++) {
    const xx = left + (width * i) / 5;
    ctx.fillStyle = "#e2e9ed";
    ctx.fillRect(xx, 25, 1, height);
    ctx.fillStyle = "#506575";
    const tick = selectedClock(windowStart + (delta * i) / 5);
    ctx.fillText(
      tick,
      Math.min(xx + 3, w - ctx.measureText(tick).width - 3),
      19,
    );
  }
  targets = [];
  const linkedPositions = [];
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
        ctx.fillText("↑ " + rateLabel(max, unit), left + 4, y + 10);
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
          ctx.fillText(
            short(displayName(e), Math.floor(sw / 6.8)),
            a + 3,
            y + 20,
          );
        }
        visible.push({ a, b, e });
        if ($("view").value === "communication" && chosenLink) {
          const member = chosenLink.ends.findIndex((end) =>
            end.event
              ? end.event === e
              : chosenLink.transfer &&
                ((end.role === "prefill" && chosenLink.transfer === e) ||
                  (end.role === "decode" &&
                    chosenLink.transfer === e.transfer)),
          );
          if (member >= 0 && e.ts >= windowStart)
            linkedPositions.push({ member, x: x(e.ts), y: y + 15 });
        }
      }
      targets.push({ y, row: r, visible });
    }
    ctx.restore();
  });
  if (linkedPositions.length) {
    linkedPositions.sort((a, b) => a.member - b.member);
    ctx.save();
    ctx.strokeStyle = "#56356f";
    ctx.fillStyle = "white";
    ctx.lineWidth = 2;
    if (report.aligned && linkedPositions.length > 1) {
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      linkedPositions.forEach((p, i) =>
        i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y),
      );
      ctx.stroke();
      ctx.setLineDash([]);
    }
    for (const p of linkedPositions) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }
  $("range").textContent =
    selectedClock(windowStart) + " – " + selectedClock(windowEnd);
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
        Value: rateLabel(best[1], r.unit),
        "Original sample": best[1],
        Unit: r.unit,
        Source: r.source,
        Scope: r.scope,
        "Sample window": best.length > 2 ? fmt(best[0] - best[2]) : undefined,
        Baseline: r.baseline?.value,
        "Baseline utilization": r.baseline?.value
          ? measure((100 * best[1]) / r.baseline.value) + "%"
          : undefined,
      });
    }
    return;
  }
  const target = hit.visible.findLast(
    (t) => px >= t.a && px <= Math.max(t.a + 1, t.b),
  );
  const match = target && eventLinks.get(target.e.transfer || target.e);
  if (match && $("view").value === "communication") {
    chosenLink = match;
    refresh();
  }
  if (target?.e.collective) {
    const op = target.e.collective;
    details(op.op, {
      Kernel: target.e.original || target.e.name,
      Direction: op.direction,
      "Peer rank": op.peer_rank,
      Group: op.group,
      Sequence: op.seq,
      "Data type": op.dtype,
      "Logical bytes": op.bytes,
      "Operation time": op.duration_us == null ? null : fmt(op.duration_us),
      "GPU kernels": op.kernel_count,
      "Algorithm bandwidth (GB/s)": op.algorithm_GB_s,
      "NCCL bus bandwidth (GB/s)": op.bus_GB_s,
      "Relative to nccl-tests":
        op.reference_pct == null ? null : measure(op.reference_pct) + "%",
      "Measured reference": op.reference,
      "Part of a multi-kernel operation": op.part_of_operation,
      Unavailable: op.unavailable,
    });
  } else if (target)
    details(displayName(target.e), {
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
for (const id of ["view", "depth"]) $(id).onchange = refresh;
$("node").onchange = () => {
  repopulateRanks();
  if (["rank", "local_utc"].includes($("clock").value))
    $("clock").value = "capture";
  updateIntervals();
  refresh();
};
$("rank").onchange = () => {
  if (!$("rank").value && ["rank", "local_utc"].includes($("clock").value))
    $("clock").value = "capture";
  if (
    $("clock").value === "local_utc" &&
    !report.traces.find((t) => t.rank === $("rank").value)?.rank_origin_unix_ns
  )
    $("clock").value = "rank";
  updateIntervals();
  refresh();
};
$("clock").onchange = () => {
  if (["rank", "local_utc"].includes($("clock").value) && !$("rank").value) {
    const candidate = report.traces.find(
      (t) =>
        visibleTrace(t) &&
        ($("clock").value !== "local_utc" || t.rank_origin_unix_ns),
    );
    if (candidate) $("rank").value = candidate.rank;
    else $("clock").value = "capture";
    updateIntervals();
  }
  refresh();
};
$("names").onchange = () => {
  updateIntervals();
  refresh();
};
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
async function importFile(file, metadataOnly = false) {
  if (/\.html$/i.test(file.name)) {
    const html = await file.text();
    const content =
      /<script\s+id="capture-data"\s+type="application\/json">([\s\S]*?)<\/script>/i.exec(
        html,
      );
    if (!content)
      throw Error("This HTML file does not contain a ServeUtil capture.");
    return unpack(content[1]);
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
      worker.postMessage({ file, metadataOnly });
    });
  } finally {
    worker.terminate();
    URL.revokeObjectURL(url);
  }
}
async function portableFile(file) {
  const stream = () =>
    /\.gz$/i.test(file.name)
      ? file.stream().pipeThrough(new DecompressionStream("gzip"))
      : file.stream();
  const reader = stream().getReader();
  let prefix;
  try {
    const chunk = await reader.read();
    prefix = new TextDecoder().decode(chunk.value?.subarray(0, 8192));
  } finally {
    reader.cancel().catch(() => {});
  }
  if (
    !/"schema"\s*:\s*"serveutil\.report\/1"|"encoding"\s*:\s*"gzip\+base64"/.test(
      prefix,
    )
  )
    return null;
  const data = await unpack(await new Response(stream()).text());
  return data?.schema === "serveutil.report/1" ? data : null;
}
async function openFiles(files) {
  if (!files.length) return;
  const status = document.body.classList.contains("empty")
    ? $("importStatus")
    : $("summary");
  if (nativeFrame || !nativePanel.hidden) closeNative();
  let sources = [],
    traces = [],
    hardware = [],
    findings = [];
  let last, problem;
  for (const file of files) {
    status.textContent = "Reading " + file.name + "…";
    if (
      /\.html$/i.test(file.name) &&
      /id="capture-data"/i.test(await file.text())
    ) {
      if (files.length > 1) {
        status.textContent = "Open standalone reports one at a time.";
        return;
      }
      try {
        const html = await file.text();
        const data = await importFile(file);
        setOriginals(rawSourcesIn(html));
        if (data) load(data);
        else await openNative();
      } catch (err) {
        status.textContent = err.message;
      }
      return;
    }
    const source = fileOriginal(file);
    sources.push(source);
    if (!/\.(json|gz)$/i.test(file.name)) continue;
    try {
      const portable = await portableFile(file);
      if (portable && files.length === 1) {
        setOriginals([]);
        load(portable);
        return;
      }
      if (portable) {
        status.textContent = "Open portable reports one at a time.";
        return;
      }
      last = await importFile(file);
      if (
        !last.traces?.length ||
        last.traces.every((t) => !t.events.length && !t.counters.length)
      )
        continue;
      for (const t of last.traces) {
        const rank = files.length > 1 ? String(traces.length) : t.rank;
        source.rank = rank;
        traces.push({ ...t, rank });
      }
      hardware.push(...(last.hardware || []));
      findings.push(...(last.findings || []));
    } catch (err) {
      problem = err;
    }
  }
  setOriginals(sources);
  if (traces.length) {
    load({
      ...last,
      traces,
      hardware,
      findings,
      aligned: traces.length === 1,
      alignment_note:
        traces.length > 1
          ? "Each imported worker has its own origin. Open a capture report for clock-aligned global timing."
          : last.alignment_note,
    });
  } else {
    if (problem) status.textContent = problem.message;
    await openNative();
  }
}
$("choose").onclick = () => $("file").click();
$("file").onchange = async (e) => {
  await openFiles([...e.target.files]);
  e.target.value = "";
};
$("captureFolder").onchange = async (event) => {
  const status = document.body.classList.contains("empty")
    ? $("importStatus")
    : $("summary");
  const picked = [...event.target.files];
  event.target.value = "";
  if (!picked.length) return;
  try {
    const byPath = new Map(
      picked.map((f) => [
        f.webkitRelativePath.split("/").slice(1).join("/"),
        f,
      ]),
    );
    const manifestFile = byPath.get("capture.json");
    if (!manifestFile)
      throw Error("Select the capture folder containing capture.json.");
    const manifest = JSON.parse(await manifestFile.text()),
      traces = [],
      sources = [];
    const reportFile = byPath.get("report.html");
    let data = reportFile ? await importFile(reportFile) : null;
    const clocks = new Map(
      (manifest.nodes || []).map((n) => [n.name, n.clock]),
    );
    const epochs = [];
    const sameHost =
      new Set(manifest.traces.map((entry) => entry.node)).size <= 1;
    for (const [i, entry] of manifest.traces.entries()) {
      const f = byPath.get(entry.file.replaceAll("\\", "/"));
      if (!f) throw Error("Missing capture file: " + entry.file);
      const rank = String(entry.rank ?? i);
      const source = {
        ...fileOriginal(f),
        rank,
        node: entry.node,
        title: rank + " · " + f.name,
        kind: entry.auxiliary ? "auxiliary" : "trace",
      };
      sources.push(source);
      const clock = clocks.get(entry.node);
      const knownClock =
        sameHost ||
        (Number.isSafeInteger(clock?.uncertainty_ns) &&
          Number.isSafeInteger(clock?.offset_ns));
      if (entry.auxiliary) {
        const auxiliary = (await importFile(f, true)).traces[0];
        if (auxiliary?.raw_epoch != null && knownClock)
          source.epoch = String(
            BigInt(auxiliary.raw_epoch) + BigInt(clock?.offset_ns ?? 0),
          );
        continue;
      }
      if (data) {
        const t = data.traces.find(
          (t) =>
            t.rank === rank &&
            (t.source_path
              ? t.source_path === entry.file
              : t.file === f.name) &&
            t.node == entry.node,
        );
        if (t?.source_epoch_ns && knownClock) source.epoch = t.source_epoch_ns;
      } else {
        let worker;
        if (!entry.native_output || /\.json(?:\.gz)?$/i.test(f.name)) {
          try {
            worker = (await importFile(f)).traces[0];
          } catch (err) {
            if (!entry.native_output && manifest.engine !== "torch") throw err;
            worker = { parse_error: err.message, events: [], counters: [] };
          }
        }
        const t = {
          ...(worker || { native_only: true, events: [], counters: [] }),
          file: f.name,
          rank,
          node: entry.node,
          role: entry.role,
        };
        traces.push(t);
        if (t.raw_epoch != null && knownClock)
          source.epoch = String(
            BigInt(t.raw_epoch) + BigInt(clock?.offset_ns ?? 0),
          );
        epochs.push(
          source.epoch != null
            ? BigInt(source.epoch) + BigInt(Math.round(t.raw_begin * 1000))
            : null,
        );
      }
    }
    if (!data) {
      if (!traces.length) throw Error("No worker traces found in capture.");
      const timed = epochs.filter(
        (_, i) => !traces[i].native_only && !traces[i].parse_error,
      );
      const aligned = timed.length > 0 && timed.every((x) => x != null);
      const zero = aligned ? timed.reduce((a, b) => (a < b ? a : b)) : null;
      if (aligned) {
        traces.forEach((t, i) => {
          if (epochs[i] == null) return;
          const off = Number(epochs[i] - zero) / 1000;
          t.rank_origin_us = off;
          for (const e of t.events) e.ts += off;
          for (const c of t.counters)
            for (const point of c.points) point[0] += off;
        });
      }
      data = {
        schema: "serveutil.report/1",
        engine: manifest.engine,
        traces,
        aligned,
        status: manifest.status,
        time_origin_unix_ns: zero?.toString(),
        telemetry: {
          analysis:
            "Raw worker traces. Generate report.html with serveutil report to include hardware samples and interval findings.",
        },
        hardware: [],
        findings: [],
        alignment_note: aligned ? undefined : "Independent worker clocks",
      };
    }
    setOriginals(sources);
    closeNative();
    load(data);
  } catch (err) {
    status.textContent = err.message;
  }
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
  openFiles([...e.dataTransfer.files]);
});
new ResizeObserver(draw).observe(canvas.parentElement);
unpack($("capture-data").textContent)
  .then((initial) => {
    if (initial) load(initial);
    else if (originals.length) openNative();
  })
  .catch((err) => {
    $("importStatus").textContent = "Unable to read capture: " + err.message;
  });
