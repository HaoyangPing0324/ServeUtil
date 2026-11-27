import { Tokenizer, TokenParser } from "@streamparser/json";

// Kineto writes its epoch and some communicator IDs as 64-bit integers.
class TraceTokenizer extends Tokenizer {
  parseNumber(token) {
    const value = super.parseNumber(token);
    return Number.isInteger(value) &&
      !Number.isSafeInteger(value) &&
      /^-?\d+$/.test(token)
      ? token
      : value;
  }
}

function communicator(args) {
  const name = String(args["Collective name"] || "")
    .replaceAll("_", "")
    .toLowerCase();
  const op = {
    send: "send",
    recv: "recv",
    allreduce: "all_reduce",
    broadcast: "broadcast",
    allgather: "all_gather",
    allgatherbase: "all_gather",
    reducescatter: "reduce_scatter",
    reducescatterbase: "reduce_scatter",
    reduce: "reduce",
  }[name];
  if (!op) return;
  let members = args["Process Group Ranks"];
  if (typeof members === "string") {
    try {
      members = JSON.parse(members);
    } catch {
      members = null;
    }
  }
  const group = {
    id:
      args["Process Group Name"] == null
        ? null
        : String(args["Process Group Name"]),
    size: args["Group size"],
  };
  if (Array.isArray(members) && members.every(Number.isSafeInteger))
    group.ranks = members;
  const result = { op, group, seq: args.Seq, dtype: args.dtype };
  const comms = args["Comms Id"];
  if (/^[1-9]\d*$/.test(String(comms))) result.comms_id = String(comms);
  if (op === "send" || op === "recv") {
    result.direction = op;
    result.peer_rank = args[op === "send" ? "Dst Rank" : "Src Rank"];
  }
  const dtype = String(args.dtype || "")
    .replace(/^torch\./, "")
    .toLowerCase();
  const width = {
    bool: 1,
    byte: 1,
    uint8: 1,
    char: 1,
    int8: 1,
    short: 2,
    int16: 2,
    uint16: 2,
    half: 2,
    float16: 2,
    bfloat16: 2,
    bf16: 2,
    float: 4,
    float32: 4,
    int: 4,
    int32: 4,
    uint32: 4,
    long: 8,
    int64: 8,
    uint64: 8,
    double: 8,
    float64: 8,
    complex64: 8,
    complex128: 16,
  }[dtype];
  const incoming = args["In msg nelems"],
    outgoing = args["Out msg nelems"];
  let elements;
  if (
    op === "send" &&
    (outgoing == null || outgoing === 0 || incoming === outgoing)
  )
    elements = incoming;
  else if (
    op === "recv" &&
    (incoming == null || incoming === 0 || incoming === outgoing)
  )
    elements = outgoing;
  else if (op === "all_gather" && incoming * group.size === outgoing)
    elements = outgoing;
  else if (op === "reduce_scatter" && outgoing * group.size === incoming)
    elements = incoming;
  else if (
    ["all_reduce", "reduce", "broadcast"].includes(op) &&
    incoming === outgoing
  )
    elements = incoming;
  if (
    Number.isSafeInteger(elements) &&
    elements >= 0 &&
    Number.isSafeInteger(elements * width)
  )
    result.bytes = elements * width;
  return result;
}

// Parse in a worker so a long Kineto capture does not block pan and zoom.
self.onmessage = async ({ data: { file, metadataOnly } }) => {
  try {
    let stream = file.stream();
    if (/\.gz$/i.test(file.name)) {
      stream = stream.pipeThrough(new DecompressionStream("gzip"));
    }
    const reader = stream.getReader();
    const events = [],
      counters = new Map(),
      stack = new Map();
    let origin = Infinity,
      count = 0,
      epoch,
      globalRank;
    const tokenizer = new TraceTokenizer();
    const parser = new TokenParser({
      paths: metadataOnly
        ? ["$.baseTimeNanoseconds"]
        : [
            "$.traceEvents.*",
            "$.baseTimeNanoseconds",
            "$.distributedInfo.rank",
          ],
      keepStack: false,
    });
    tokenizer.onToken = (token) => parser.write(token);
    tokenizer.onEnd = () => {
      if (!parser.isEnded) parser.end();
    };
    parser.onValue = ({ value: e, key: property }) => {
      if (property === "baseTimeNanoseconds") {
        epoch = String(e);
        return;
      }
      if (property === "rank") {
        if (Number.isSafeInteger(e)) globalRank = e;
        return;
      }
      count++;
      if (!e || !Number.isFinite(e.ts)) return;
      const lane = String(e.tid ?? ""),
        track = String(e.pid ?? "");
      const key = track + "/" + lane;
      if (e.ph === "C") {
        for (const [field, val] of Object.entries(e.args || {})) {
          if (!Number.isFinite(val)) continue;
          const name = !field || field === "value" ? e.name : e.name + " · " + field;
          const id = key + "/" + name;
          if (!counters.has(id))
            counters.set(id, {
              name,
              device: track,
              lane,
              source: "Trace counter",
              points: [],
            });
          counters.get(id).points.push([e.ts, val]);
          origin = Math.min(origin, e.ts);
        }
        return;
      }
      if (e.ph === "B") {
        if (!stack.has(key)) stack.set(key, []);
        stack.get(key).push(e);
        return;
      }
      if (e.ph === "E") {
        const begin = stack.get(key)?.pop();
        if (!begin) return;
        e = { ...begin, ph: "X", dur: e.ts - begin.ts };
      }
      if (e.ph !== "X" || !(e.dur > 0)) return;
      const cat = (e.cat || "").split(","),
        name = e.name || "",
        args = e.args || {};
      let kind,
        label = name;
      if (cat.includes("kernel")) {
        kind =
          args["Collective name"] ||
          /nccl|all.?reduce|all.?gather|reduce.?scatter|all.?to.?all/i.test(
            name,
          )
            ? "communication"
            : "compute";
        if (kind === "compute" && /gemm/i.test(name.split("<")[0]))
          label = "GEMM";
      } else if (cat.includes("gpu_memcpy") || cat.includes("gpu_memset"))
        kind = "copy";
      else if (
        cat.includes("user_annotation") ||
        cat.includes("gpu_user_annotation")
      )
        kind = "stage";
      else if (cat.includes("cpu_op")) kind = "operator";
      else return;
      const event = {
        kind,
        name: label,
        ts: e.ts,
        dur: e.dur,
        device: String(args.device ?? track),
        lane,
      };
      if (kind === "stage") {
        event.domain = cat.includes("gpu_user_annotation") ? "GPU" : "CPU";
        const sg = /^step\[(EXTEND|DECODE|PREFILL|MIXED)\b/i.exec(name);
        const vl = /^execute_context_(\d+)\(\d+\)_generation_(\d+)\(\d+\)/.exec(
          name,
        );
        if (sg)
          event.name = {
            EXTEND: "Prefill",
            PREFILL: "Prefill",
            DECODE: "Decode",
            MIXED: "Mixed batch",
          }[sg[1].toUpperCase()];
        if (vl)
          event.name = Number(vl[1])
            ? Number(vl[2])
              ? "Mixed batch"
              : "Prefill"
            : Number(vl[2])
              ? "Decode"
              : "Idle";
      }
      if (name !== event.name) event.original = name;
      if (kind === "communication") event.collective = communicator(args);
      for (const key of [
        "stream",
        "correlation",
        "External id",
        "bytes",
        "Input Dims",
        "Input type",
      ]) {
        if (args[key] !== undefined) (event.args ||= {})[key] = args[key];
      }
      origin = Math.min(origin, e.ts);
      events.push(event);
    };
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      tokenizer.write(value);
    }
    if (!tokenizer.isEnded) tokenizer.end();
    if (!count && !metadataOnly)
      throw Error("No trace events found in this file.");
    for (const e of events) e.ts -= origin;
    events.sort((a, b) => a.ts - b.ts || b.dur - a.dur);
    const series = [...counters.values()];
    for (const c of series) {
      for (const p of c.points) p[0] -= origin;
      c.points.sort((a, b) => a[0] - b[0]);
    }
    self.postMessage({
      schema: "serveutil.report/1",
      engine: "Imported trace",
      aligned: true,
      hardware: [],
      telemetry: {
        analysis:
          "Raw browser import. Use serveutil report for interval findings and correlated GEMM estimates.",
      },
      findings: [],
      traces: [
        {
          rank: "0",
          global_rank: globalRank,
          file: file.name,
          events,
          counters: series,
          raw_epoch: epoch,
          raw_begin: Number.isFinite(origin) ? origin : 0,
        },
      ],
    });
  } catch (e) {
    self.postMessage({ error: e.message });
  }
};
