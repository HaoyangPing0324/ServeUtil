# Follow a KV handoff from Prefill to Decode

[Open one measured SGLang handoff](../../../examples/sglang-pd.report.json). The two roles ran on separate H20 hosts and used NIXL over UCX/TCP. Names are anonymized. The example preserves relative times and the matched engine observation for one request.

## Find the two endpoints

Choose **Communication**, then select the entry in **KV transfers**. The source scheduler transfer interval lasted about 22.707 ms. On the destination, the start of Decode followed the recorded receive completion by about 180 µs. Select one node at a time to see the role-local activities.

## Distinguish the three measurements

The matched engine observation attributed 7,225,344 logical KV bytes to this request over about 22.525 ms, yielding approximately 0.321 GB/s. Dividing the same bytes by the source scheduler span yields about 0.318 GB/s. The network card in a complete capture instead measures all bytes crossing the monitored interface. Each value has its own time window and scope.

## Record a deployment

Edit the [PD topology](../../../examples/pd-topology.json) so requests go to the router and profiler control goes to the two role endpoints. Configure SGLang request spans through the topology's `transfers` field. Capture with `serveutil capture --engine sglang --topology deployment.json --seconds 5 --generate "Explain KV caching."`. With vLLM, the NIXL counter panel provides endpoint totals; attribute those bytes to an individual request only when the deployment supplies matching evidence. [Deployment and collection](../capture.md)
