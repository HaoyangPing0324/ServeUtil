// Hand unchanged trace bytes to the upstream engine through its documented API.
const nativePanel = document.getElementById("nativePanel");
const nativeSources = document.getElementById("nativeSources");
const nativeStatus = document.getElementById("nativeStatus");
let originals = JSON.parse(document.getElementById("raw-sources").textContent);
let nativeFrame,
  nativeOrigin,
  nativeGeneration = 0,
  pendingSource;

function setOriginals(sources) {
  const members = sources.filter(
    (s) => s.kind === "trace" || s.kind === "auxiliary",
  );
  const groups = [];
  if (members.length > 1) {
    groups.push({
      kind: "global",
      name: "all-ranks.tar",
      title: "All ranks · original traces",
      members,
    });
    const cyclesByRank = new Map();
    for (const source of members) {
      if (source.rank == null || source.kind !== "trace") continue;
      const key = JSON.stringify([source.node, source.rank]);
      if (!cyclesByRank.has(key)) cyclesByRank.set(key, []);
      cyclesByRank.get(key).push(source);
    }
    for (const cycles of cyclesByRank.values()) {
      if (cycles.length < 2) continue;
      const { rank, node } = cycles[0];
      groups.push({
        kind: "cycles",
        rank,
        node,
        name: "rank-cycles.tar",
        title: rank + " · all cycles",
        members: cycles,
      });
    }
    for (const node of new Set(members.map((s) => s.node).filter(Boolean))) {
      const local = members.filter((s) => s.node === node);
      if (local.length > 1)
        groups.push({
          kind: "node",
          node,
          name: "node-traces.tar",
          title: node + " · all ranks",
          members: local,
        });
    }
  }
  originals = [...groups, ...sources];
  nativeSources.replaceChildren();
  for (let i = 0; i < originals.length; i++) {
    const src = originals[i];
    if (src.kind === "unavailable") continue;
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = src.title;
    nativeSources.append(option);
  }
  document.getElementById("rawView").hidden = !nativeSources.length || location.protocol === "file:";
  nativeSources.hidden = !nativeSources.length;
  document.getElementById("saveOriginal").hidden = nativeSources.hidden;
}
function fileOriginal(file) {
  return { name: file.name, title: file.name, file, kind: "trace" };
}
async function originalBlob(src) {
  if (src.members) return traceArchive(src.members);
  if (src.file) return src.file;
  const parts = [];
  for (let i = 0; i < src.payload.length; i += 65536) {
    const binary = atob(src.payload.slice(i, i + 65536));
    parts.push(Uint8Array.from(binary, (ch) => ch.charCodeAt(0)));
  }
  const blob = new Blob(parts);
  return src.compress_attachment
    ? new Response(
        blob.stream().pipeThrough(new DecompressionStream("gzip")),
      ).blob()
    : blob;
}
function rawSourcesIn(html) {
  const hit =
    /<script id="raw-sources" type="application\/json">([\s\S]*?)<\/script>/i.exec(
      html,
    );
  return hit ? JSON.parse(hit[1]) : [];
}
async function postOriginal(frame, origin, src, generation) {
  if (!src) return;
  const blob = await originalBlob(src);
  const buffer = await blob.arrayBuffer();
  if (nativeGeneration !== generation || nativeFrame !== frame) return;
  frame.contentWindow.postMessage(
    {
      perfetto: {
        buffer,
        title: src.title,
        fileName: src.name,
        keepApiOpen: true,
        downloadable: true,
        shareable: false,
      },
    },
    origin,
    [buffer],
  );
  nativeStatus.textContent = "";
}
window.addEventListener("message", (e) => {
  if (
    !nativeFrame ||
    e.source !== nativeFrame.contentWindow ||
    e.origin !== nativeOrigin ||
    e.data !== "PONG" ||
    !pendingSource
  )
    return;
  const { src, generation } = pendingSource;
  pendingSource = undefined;
  postOriginal(nativeFrame, nativeOrigin, src, generation).catch(
    (err) => (nativeStatus.textContent = err.message),
  );
});
async function openNative(preferredRank = true) {
  const generation = ++nativeGeneration;
  nativePanel.hidden = false;
  document.body.classList.add("full-trace");
  const selected = document.getElementById("rank")?.value;
  if (preferredRank && selected) {
    const i = originals.findIndex(
      (r) =>
        r.rank === selected &&
        (r.kind === "cycles" || r.kind === "trace") &&
        (!document.getElementById("node")?.value ||
          r.node === document.getElementById("node").value),
    );
    if (i >= 0) nativeSources.value = String(i);
  } else if (preferredRank && document.getElementById("node")?.value) {
    const node = document.getElementById("node").value;
    const i = originals.findIndex(
      (s) => s.node === node && (s.kind === "node" || s.kind === "trace"),
    );
    if (i >= 0) nativeSources.value = String(i);
  } else if (preferredRank) {
    const all = originals.findIndex((r) => r.kind === "global");
    if (all >= 0) nativeSources.value = String(all);
  }
  const src =
    originals[Number(nativeSources.value)]?.kind === "unavailable"
      ? undefined
      : originals[Number(nativeSources.value)];
  document.getElementById("rawClockNote").textContent = src?.members?.some(
    (s) => s.epoch == null,
  )
    ? "Independent clocks. Cross-worker time differences require clock calibration."
    : "";
  let base;
  if (location.protocol !== "file:") {
    const local = new URL("perfetto/", location.href);
    try {
      const response = await fetch(new URL("manifest.json", local));
      if (
        response.ok &&
        (await response.json()).resources?.["trace_processor.wasm"]
      )
        base = local;
    } catch (_) {
      /* Standalone reports can be opened with the local viewer. */
    }
  }
  if (generation !== nativeGeneration) return;
  if (!base) {
    nativeStatus.textContent =
      "Open this report with serveutil view or the desktop app to read original tracks.";
    return;
  }
  nativeStatus.textContent = src
    ? "Opening " + src.title + "…"
    : "This report has no original trace attached. Open the original file in Perfetto.";
  nativeFrame?.remove();
  const frame = document.createElement("iframe");
  frame.title = "Perfetto full trace workspace";
  frame.allow = "fullscreen";
  frame.src = base.href;
  document.getElementById("nativeFrame").append(frame);
  nativeFrame = frame;
  nativeOrigin = base.origin;
  pendingSource = src ? { src, generation } : undefined;
  const deadline = Date.now() + 90000;
  const timer = setInterval(() => {
    if (generation !== nativeGeneration || !pendingSource) {
      clearInterval(timer);
      return;
    }
    if (Date.now() > deadline) {
      clearInterval(timer);
      pendingSource = undefined;
      nativeStatus.textContent =
        "Perfetto did not start. Reopen the workspace or save the original file.";
      return;
    }
    frame.contentWindow.postMessage("PING", nativeOrigin);
  }, 200);
}
function closeNative() {
  ++nativeGeneration;
  pendingSource = undefined;
  nativeFrame?.remove();
  nativeFrame = undefined;
  nativePanel.hidden = true;
  document.body.classList.remove("full-trace");
}
document.getElementById("leaveNative").onclick = closeNative;
nativeSources.onchange = () => openNative(false);
document.getElementById("saveOriginal").onclick = async () => {
  const src = originals[Number(nativeSources.value)];
  if (!src) return;
  const url = URL.createObjectURL(await originalBlob(src));
  const link = document.createElement("a");
  link.href = url;
  link.download = src.name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
};
setOriginals(originals);

// Standard ustar archive understood by upstream Trace Processor. The trace files
// retain the original uncompressed data. The manifest supplies identity and clocks.
function tarHeader(name, size) {
  const header = new Uint8Array(512),
    enc = new TextEncoder();
  const field = (pos, text) => header.set(enc.encode(text), pos);
  const oct = (v, width) => v.toString(8).padStart(width - 1, "0") + "\0";
  field(0, name);
  field(100, oct(420, 8));
  field(108, oct(0, 8));
  field(116, oct(0, 8));
  if (size > 0o77777777777)
    throw Error("Trace exceeds the ustar member size limit.");
  field(124, oct(size, 12));
  field(136, oct(0, 12));
  field(148, "        ");
  field(156, "0");
  field(257, "ustar\0");
  field(263, "00");
  field(
    148,
    header
      .reduce((sum, x) => sum + x, 0)
      .toString(8)
      .padStart(6, "0") + "\0 ",
  );
  return header;
}
async function traceArchive(members) {
  const parts = [],
    files = [],
    names = new Map(),
    machines = new Set();
  const add = (name, blob) =>
    parts.push(
      tarHeader(name, blob.size),
      blob,
      new Uint8Array((512 - (blob.size % 512)) % 512),
    );
  let anchor;
  if (members.every((s) => s.epoch != null)) {
    anchor = members.reduce((a, b) =>
      BigInt(a.epoch) <= BigInt(b.epoch) ? a : b,
    );
  }
  // Trace Processor applies clock and machine overrides to trace members.
  // An additional gzip envelope makes a member a nested archive upstream.
  const contents = new Map();
  for (const [i, s] of members.entries()) {
    let blob = await originalBlob(s);
    const magic = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
    let name = s.name;
    if (magic[0] === 0x1f && magic[1] === 0x8b) {
      blob = await new Response(
        blob.stream().pipeThrough(new DecompressionStream("gzip")),
      ).blob();
      name = name.replace(/\.gz$/i, "");
    }
    contents.set(s, blob);
    names.set(
      s,
      "worker-" + i + "-" + name.replace(/[^a-zA-Z0-9._-]/g, "_").slice(-72),
    );
  }
  for (const [i, s] of members.entries()) {
    const path = names.get(s);
    let machine =
      [s.node, s.rank].filter((x) => x != null).join(" / ") ||
      "source " + (i + 1);
    if (machines.has(machine)) machine += " · " + (i + 1);
    machines.add(machine);
    const entry = { path, machine: { name: machine } };
    if (anchor && s !== anchor) {
      const offset = BigInt(s.epoch) - BigInt(anchor.epoch);
      if (offset > BigInt(Number.MAX_SAFE_INTEGER))
        throw Error(
          "Clock offset exceeds exact browser precision. Open the originals individually.",
        );
      entry.clocks = {
        sync_to: { file: names.get(anchor) },
        offset_ns: Number(offset),
      };
    }
    files.push(entry);
    add(path, contents.get(s));
  }
  add(
    "perfetto_manifest.json",
    new Blob([JSON.stringify({ perfetto_manifest: { version: 1, files } })]),
  );
  parts.push(new Uint8Array(1024));
  return new Blob(parts, { type: "application/x-tar" });
}
