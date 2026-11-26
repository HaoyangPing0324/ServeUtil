import { JSONParser } from "@streamparser/json";

// Parse in a worker so a long Kineto capture does not block pan and zoom.
self.onmessage = async ({ data: file }) => {
  try {
    let stream = file.stream();
    if (file.name.endsWith(".gz")) {
      stream = stream.pipeThrough(new DecompressionStream("gzip"));
    }
    const reader = stream.getReader();
    const events = [],
      counters = new Map(),
      stack = new Map();
    let origin = Infinity,
      count = 0;
    const parser = new JSONParser({
      paths: ["$.traceEvents.*"],
      keepStack: false,
    });
    parser.onValue = ({ value: e }) => {
      count++;
      if (!e || !Number.isFinite(e.ts)) return;
      const lane = String(e.tid ?? ""),
        track = String(e.pid ?? "");
      const key = track + "/" + lane;
      if (e.ph === "C") {
        for (const [field, val] of Object.entries(e.args || {})) {
          if (!Number.isFinite(val)) continue;
          const name = field === "value" ? e.name : e.name + " · " + field;
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
      parser.write(value);
    }
    if (!parser.isEnded) parser.end();
    if (!count) throw Error("No trace events found in this file.");
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
      traces: [{ rank: "0", file: file.name, events, counters: series }],
    });
  } catch (e) {
    self.postMessage({ error: e.message });
  }
};
