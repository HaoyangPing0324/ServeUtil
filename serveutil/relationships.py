"""Correlate recorded communication endpoints without relying on clock proximity."""

from collections import Counter, defaultdict


def communication_links(traces):
    """Reference primary device events by their trace and event indices."""
    rank_counts = Counter(
        (t.get("role"), t["global_rank"])
        for t in traces
        if t.get("global_rank") is not None
    )
    groups = defaultdict(list)
    for ti, trace in enumerate(traces):
        rank = trace.get("global_rank")
        role = trace.get("role")
        if rank is None or rank_counts[role, rank] != 1:
            continue
        for ei, event in enumerate(trace["events"]):
            op = event.get("collective", {})
            group = op.get("group", {})
            members = tuple(group.get("ranks", []))
            if (
                rank not in members
                or len(set(members)) != group.get("size")
                or op.get("seq") is None
                or op.get("bytes") is None
                or not group.get("id")
                or not op.get("dtype")
            ):
                continue
            name = op.get("op")
            kind = "send_recv" if name in {"send", "recv"} else name
            key = (role, group["id"], members, op["seq"], op.get("comms_id"), kind)
            groups[key].append((ti, ei, rank, op))

    links = []
    for (_, _, members, _, comms, kind), rows in groups.items():
        if len(rows) < 2 or len({r[2] for r in rows}) != len(rows):
            continue
        if len({(r[3]["dtype"], r[3]["bytes"]) for r in rows}) != 1:
            continue
        if kind == "send_recv":
            if len(rows) != 2:
                continue
            rows.sort(key=lambda r: r[3]["op"] != "send")
            sender, receiver = rows
            if (sender[3]["op"], receiver[3]["op"]) != ("send", "recv"):
                continue

            # Peer indices can be group-local or global across producer versions.
            # Only pair when both conventions agree or just one is applicable.
            def peers(row):
                n = row[3].get("peer_rank")
                if not isinstance(n, int):
                    return set()
                return ({n} if n in members else set()) | (
                    {members[n]} if 0 <= n < len(members) else set()
                )

            if peers(sender) != {receiver[2]} or peers(receiver) != {sender[2]}:
                continue
        elif {r[2] for r in rows} != set(members):
            continue
        else:
            rows.sort(key=lambda r: members.index(r[2]))
        links.append(
            {
                "kind": kind,
                "evidence": "comms_id" if comms is not None else "group_sequence",
                "endpoints": [[t, e] for t, e, _, _ in rows],
            }
        )
    return links
