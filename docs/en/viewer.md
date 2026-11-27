# Explore a capture

Open the web viewer and choose **Open capture folder**, or run `serveutil view CAPTURE`. A single trace, a gzip-compressed PyTorch JSON file, an existing ServeUtil report or a native Perfetto trace can also be opened directly.

## Start with the inference timeline

**All nodes / All ranks** shows the global capture. Select a node or rank to filter the timeline, resources and communication paths. Use **Readable** for inference names or **Original** for the recorded operator name. **Operator depth** can show one level, two levels, a specific depth or leaves. Drag to zoom. Click **Fit** to restore the capture window.

Select a step under **Interval**, then click a resource card for its absolute rate, source, covered time and reference. Optimization opportunities are grouped by type. The interval with the longest overlap with your selection appears first. Open the other intervals or click a marker for its evidence.

## Follow a communication path

Choose **Communication**. Click an operation or a paired endpoint in **Communication paths** to examine the peer on the other rank or node. Inferred matches require consistent group, sequence, type and payload. Ambiguous observations stay rank-local. Request spans allow SGLang PD handoffs to be followed through to Decode. Engine histograms without request identity are reported at endpoint scope.

## Inspect the original in Perfetto

Choose **Full Perfetto** in the loaded report. Open one original worker, a node, or all sources. The bundled official UI provides its native tracks, SQL, Data Explorer and plugins. **Save original** retrieves an individual input or a TAR of uncompressed original members. Clock metadata and distinct process identities are preserved in the multi-file workspace. Capture reports can contain original stacks and application metadata. Measured public excerpts contain only selected data.

A local HTML file opens the inference timeline directly. To use the bundled WebAssembly workspace, open the report with the desktop app, `serveutil view report.html`, or serve the complete static viewer over HTTP(S).

## Export for another tool

```bash
serveutil export CAPTURE -o overview.json
serveutil export --full CAPTURE -o full.json
serveutil export --full rank0.json rank1.json.gz --clocks clocks.json -o full.json
```

The default produces inference overview tracks. Full Chrome JSON export preserves original events, metadata, counters, flows, samples and stack frames with distinct IDs per source. Prefer the native TAR workspace for full multi-machine clock and identity semantics.
