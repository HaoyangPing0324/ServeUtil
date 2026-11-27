// The wheel and drag navigation below adapts Perfetto v59.1 timeline_interactions.ts.
// Copyright (C) 2024 The Android Open Source Project. Licensed under Apache-2.0.
function analysisInterval() {
  return selection?.type === "area" ? [selection.start, selection.end] : [windowStart, windowEnd];
}
function hideInspector() {
  $("inspector").hidden = true;
  document.body.classList.remove("inspecting");
}
function resetSelection() {
  selection = null;
  chosenLink = null;
  $("details").replaceChildren();
  hideInspector();
}
function clearSelection() {
  resetSelection();
  refresh();
}
function plotWidth() {
  return Math.max(1, canvas.clientWidth - titleWidth - 12);
}
function fractionAt(x) {
  return Math.max(0, Math.min(1, (x - titleWidth) / plotWidth()));
}
function timeAt(x) {
  return windowStart + fractionAt(x) * (windowEnd - windowStart);
}
function zoomAt(factor, x) {
  const fraction = fractionAt(x);
  const anchor = timeAt(x);
  const span = (windowEnd - windowStart) * factor;
  setWindow(anchor - span * fraction, anchor + span * (1 - fraction));
}
function panPixels(px) {
  const delta = px * (windowEnd - windowStart) / plotWidth();
  setWindow(windowStart + delta, windowEnd + delta);
}
function fitSelection() {
  if (!selection) return;
  const pad = Math.max((selection.end - selection.start) * 0.1, 0.01);
  setWindow(selection.start - pad, selection.end + pad);
}
function describeArea() {
  const {start, end, tracks} = selection;
  details("Time selection", {
    Start: selectedClock(start), End: selectedClock(end), Duration: fmt(end - start),
    Tracks: [...tracks].join("\n") || "—",
  });
}
function toggleAreaTrack(row) {
  const tracks = selection.tracks;
  if (tracks.has(row.label)) tracks.delete(row.label);
  else tracks.add(row.label);
  describeArea();
  draw();
}
function paintSelection(x, width, height) {
  if (!selection) return;
  const start = Math.max(titleWidth, x(selection.start));
  const end = Math.min(width - 12, x(selection.end));
  if (end < start) return;
  if (selection.type === "area") {
    ctx.fillStyle = "rgba(42, 98, 183, 0.18)";
    for (const {row, y} of targets) {
      if (!selection.tracks.has(row.label)) continue;
      ctx.fillRect(start, y, Math.max(1, end - start), 30);
      ctx.fillRect(1, y + 3, 4, 25);
    }
    ctx.fillStyle = "#235c9e";
    for (const edge of [selection.start, selection.end]) {
      const pos = x(edge);
      if (pos < titleWidth || pos > width - 12) continue;
      ctx.fillRect(pos, 23, 1, height - 23);
      ctx.fillRect(pos - 3, 22, 7, 5);
    }
  } else if (selection.type === "sample") {
    const row = targets.find(t => t.row.label === selection.label);
    if (row) {
      ctx.strokeStyle = "#163e72";
      ctx.lineWidth = 2;
      ctx.strokeRect(start, row.y + 2, Math.max(2, end - start), 26);
    }
  }
}
let cursorX;
function areaEdge(x, y) {
  if (selection?.type !== "area" || y > 29 || x < titleWidth) return;
  const tolerance = (windowEnd - windowStart) / plotWidth() * 7;
  for (const edge of ["start", "end"]) {
    if (Math.abs(timeAt(x) - selection[edge]) <= tolerance) return edge;
  }
}
canvas.onpointerdown = (e) => {
  if (!report || drag || (e.button !== 0 && e.button !== 1)) return;
  const [x, y] = loc(e);
  canvas.focus({preventScroll:true});
  const pan = e.shiftKey || e.button === 1;
  drag = {x, y, lastX:x, id:e.pointerId, anchor:timeAt(x), mode:pan ? "pan" : areaEdge(x,y) || "select", moved:false};
  canvas.setPointerCapture(e.pointerId);
  e.preventDefault();
};
function movePointer(e) {
  const [x, y] = loc(e);
  cursorX = x;
  if (!drag) {
    const lane = targets.find(t => y >= t.y && y < t.y + 32);
    const item = x >= titleWidth && lane?.visible && eventAt(lane.visible, x);
    canvas.title = item ? displayName(item.e) + " · " + fmt(item.e.dur) : lane?.row.label || "";
    canvas.style.cursor = areaEdge(x,y) ? "ew-resize" : e.shiftKey ? "grab" : x < titleWidth ? "pointer" : "crosshair";
    return;
  }
  if (drag.id !== e.pointerId) return;
  if (!drag.moved && Math.hypot(x - drag.x, y - drag.y) < 4) return;
  drag.moved = true;
  if (drag.mode === "pan") {
    canvas.style.cursor = "grabbing";
    if (x !== drag.lastX) panPixels(drag.lastX - x);
  } else if (drag.mode === "select") {
    const current = timeAt(x);
    selection = {type:"area", start:Math.min(drag.anchor,current), end:Math.max(drag.anchor,current),
      tracks:new Set(targets.filter(t => t.y <= Math.max(y,drag.y) && t.y + 31 >= Math.min(y,drag.y)).map(t => t.row.label))};
    draw();
  } else {
    const current = timeAt(x);
    if (drag.mode === "start") selection.start = Math.min(current,selection.end);
    else selection.end = Math.max(current,selection.start);
    draw();
  }
  drag.lastX = x;
}
canvas.onpointermove = movePointer;
canvas.onpointerup = e => {
  if (drag?.id !== e.pointerId) return;
  movePointer(e);
  const {moved, mode} = drag;
  drag = null;
  if (!moved && mode !== "pan") selectAt(...loc(e));
  if (moved && mode !== "pan") {
    if (selection.end <= selection.start) return clearSelection();
    describeArea();
    refresh();
  }
  canvas.style.cursor = "crosshair";
};
canvas.onlostpointercapture = canvas.onpointercancel = () => {
  if (drag && drag.mode !== "pan" && drag.moved) clearSelection();
  drag = null;
};
canvas.addEventListener("wheel", e => {
  const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? plotWidth() : 1;
  if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
    panPixels((e.shiftKey && !e.deltaX ? e.deltaY : e.deltaX) * unit);
  } else if (e.ctrlKey || e.metaKey) {
    const delta = e.deltaY * unit;
    zoomAt(1 + Math.sign(delta) * Math.log2(1 + Math.abs(delta)) * 0.02, loc(e)[0]);
  } else return;
  e.preventDefault();
}, {passive:false});
function adjacentEvent(direction) {
  if (selection?.type !== "event") return;
  const row = rows.find(r => r.label === selection.label);
  if (!row?.events) return;
  const sorted = [...row.events].sort((a,b) => a.ts - b.ts);
  const index = sorted.indexOf(selection.event);
  const event = index >= 0 && sorted[index + direction];
  if (!event) return;
  inspectEvent(event, row.label);
  if (event.ts >= windowEnd || event.ts + event.dur <= windowStart) {
    const span = windowEnd - windowStart, center = event.ts + event.dur / 2;
    setWindow(center - span / 2, center + span / 2);
  }
}
$("keyHelp").onclick = () => $("shortcutDialog").showModal();
$("closeHelp").onclick = () => $("shortcutDialog").close();
$("closeDetails").onclick = () => { hideInspector(); canvas.focus({preventScroll:true}); };
document.addEventListener("keydown", e => {
  if (!report || !nativePanel.hidden || document.querySelector("dialog[open]") || e.altKey || e.metaKey || e.ctrlKey ||
      e.target.closest?.("input, select, textarea, [contenteditable=true]")) return;
  const x = cursorX ?? titleWidth + plotWidth() / 2;
  if (e.key === "?") $("shortcutDialog").showModal();
  else switch (e.code) {
    case "KeyW": zoomAt(0.9, x); break;
    case "KeyS": zoomAt(1 / 0.9, x); break;
    case "KeyA": panPixels(-50); break;
    case "KeyD": panPixels(50); break;
    case "KeyF": fitSelection(); break;
    case "KeyQ":
      if ($("inspector").hidden && $("details").childElementCount) {
        $("inspector").hidden = false;
        document.body.classList.add("inspecting");
      } else hideInspector();
      break;
    case "KeyR":
      if (!selection || selection.type === "area" || selection.end <= selection.start) return;
      selection = {...selection, type:"area", tracks:new Set([selection.label])};
      describeArea(); refresh(); break;
    case "Period": adjacentEvent(1); break;
    case "Comma": adjacentEvent(-1); break;
    case "Escape": clearSelection(); break;
    default: return;
  }
  e.preventDefault();
});

// Use the active timeline tracks, so the finder respects the rank and depth filters.
let finderCommands = false;
function finderEntries() {
  if (finderCommands) return [
    ["Fit selection", fitSelection], ["Full capture", () => $("fit").click()],
    ["Zoom in", () => zoom(0.5)], ["Zoom out", () => zoom(2)],
    ["Find track", () => openFinder(false)], ["Keyboard shortcuts", () => $("keyHelp").click()],
    ["Clear selection", clearSelection],
  ].map(([name, run]) => ({name:uiText(name), run}));
  return targets.map(({row,y}) => ({
    name:row.label,
    run() {
      $("scroller").scrollTop = Math.max(0,y - 35);
      resetSelection();
      details(row.label, {Track:row.label, Type:row.kind, Events:row.events?.length, Samples:row.points?.length});
      refresh();
    },
  }));
}
function filterFinder() {
  const query = $("finderInput").value.toLowerCase().trim();
  const result = $("finderResults");
  result.replaceChildren();
  for (const {name,run} of finderEntries().filter(item => item.name.toLowerCase().includes(query))) {
    const button = document.createElement("button");
    button.textContent = name;
    button.onclick = () => {
      $("finder").close();
      canvas.focus({preventScroll:true});
      run();
    };
    result.append(button);
  }
  if (!result.childElementCount) result.textContent = uiText("No matches");
}
function openFinder(commands) {
  finderCommands = commands;
  $("finderTitle").textContent = uiText(commands ? "Find command" : "Find track");
  $("finderInput").value = "";
  filterFinder();
  $("finder").showModal();
  $("finderInput").focus();
}
$("finderInput").oninput = filterFinder;
$("closeFinder").onclick = () => $("finder").close();
$("finder").onkeydown = e => {
  const buttons = Array.from($("finderResults").children);
  const i = buttons.indexOf(document.activeElement);
  if (e.key === "ArrowDown") buttons[Math.min(i + 1, buttons.length - 1)]?.focus();
  else if (e.key === "ArrowUp") (buttons[i - 1] || $("finderInput")).focus();
  else if (e.key === "Enter" && e.target === $("finderInput")) buttons[0]?.click();
  else return;
  e.preventDefault();
};
document.addEventListener("keydown", e => {
  if (!report || !nativePanel.hidden || document.querySelector("dialog[open]") || e.altKey ||
      !(e.ctrlKey || e.metaKey) || e.code !== "KeyP") return;
  e.preventDefault();
  openFinder(e.shiftKey);
});
