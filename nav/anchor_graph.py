"""Stage 1b anchor graph (A3): load the fiducial anchor graph, annotate planned paths, give look headings.

Pure helpers plus two loaders. Nothing here touches the planner or the map. The graph and look table are
built offline by reports/dev/v5/A3/combined/build_anchor_graph_live.py against the exact live planning map.
Every load is fail-closed: the graph's self-hash, the look table's hash and the pinned expected hash must match.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

LOOK_DTYPE = [('i', '<i4'), ('j', '<i4'), ('x', '<f8'), ('y', '<f8'), ('room', 'i1'), ('nearest_anchor', '<i4'),
              ('anchor_dist', '<f8'), ('look_heading_idx', 'i1'), ('look_yaw', '<f8'), ('look_marker', '<i2'),
              ('look_hpx', '<f8'), ('look_n_reliable', 'i1')]
LOOK_MATCH_RADIUS_M = 0.10       # a look is only offered from a pose within this distance of a table cell


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_anchor_graph(graph_path, look_path, expected_sha256):
    """Return (graph dict, look table ndarray, provenance dict). Raises on any mismatch (fail closed)."""
    if not expected_sha256:
        raise RuntimeError("anchor graph: expected_sha256 is required (fail closed)")
    graph_path, look_path = Path(graph_path), Path(look_path)
    for p in (graph_path, look_path):
        if not p.is_file():
            raise RuntimeError(f"anchor graph: missing {p}")
    graph = json.loads(graph_path.read_text())
    body = {k: v for k, v in graph.items() if k not in ("sha256", "runtime_s")}
    self_hash = hashlib.sha256(_canonical(body)).hexdigest()
    if self_hash != graph.get("sha256") or self_hash != str(expected_sha256).lower():
        raise RuntimeError("anchor graph: self-hash differs from the pinned ANCHOR_GRAPH_SHA256")
    if _sha256_file(look_path) != graph["look_table"]["sha256"]:
        raise RuntimeError("anchor graph: look table sha256 differs from the graph record")
    look = np.load(look_path, allow_pickle=False)
    if look.dtype != np.dtype(LOOK_DTYPE):
        raise RuntimeError("anchor graph: look table dtype mismatch")
    return graph, look, {"anchor_graph_sha256": self_hash, "anchor_look_sha256": graph["look_table"]["sha256"],
                         "n_anchors": int(graph["n_anchors"])}


def annotate(path_poses, graph: dict, radius: float) -> list[dict]:
    """Pass records of the path through each anchor disc: {index, node_id, distance} (see annotate_path.py)."""
    if not radius > 0:
        raise ValueError("radius must be > 0")
    anchors = graph.get("anchors", [])
    pts = np.asarray(path_poses, dtype=float)
    if pts.size == 0 or not anchors:
        return []
    pts = pts[:, :2]
    out: list[dict] = []
    for a in anchors:
        d = np.hypot(pts[:, 0] - float(a["x"]), pts[:, 1] - float(a["y"]))
        near = d <= radius
        if not near.any():
            continue
        edges = np.diff(np.concatenate([[0], near.astype(np.int8), [0]]))
        for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
            k = s + int(np.argmin(d[s:e]))
            out.append({"index": int(k), "node_id": str(a["id"]), "distance": float(d[k])})
    out.sort(key=lambda r: (r["index"], r["node_id"]))
    return out


def anchor_fn_for(graph: dict, radius_m: float, slow_cruise_mps_for):
    """Build the FSM hook: poses -> anchor dicts for the passes on this path.

    slow_cruise_mps_for(): current default cruise (m/s) or None. The slow-down value is a fraction of it that
    the caller chooses; the FSM clamps it below at the follower's measured minimum speed.
    """
    anchors = graph["anchors"]

    def fn(poses):
        out = []
        for hit in annotate(poses, graph, radius_m):
            a = anchors[next(i for i, x in enumerate(anchors) if x["id"] == hit["node_id"])]
            out.append(dict(xy=[float(a["x"]), float(a["y"])], radius_m=float(radius_m), node_id=a["id"],
                            index=hit["index"], slow_cruise_mps=slow_cruise_mps_for(),
                            request_fix=True, look_yaw=float(a["yaw"])))   # anchor yaw was render-verified
        return out
    return fn


def look_provider(look: np.ndarray):
    """pose -> look yaw (rad) from the nearest table cell with a reliable marker, or None (default sweep)."""
    rows = look[look["look_heading_idx"] >= 0]
    if len(rows) == 0:
        return lambda pose: None
    xy = np.stack([rows["x"], rows["y"]], axis=1)
    yaw = rows["look_yaw"].astype(float)

    def provider(pose):
        d = np.hypot(xy[:, 0] - float(pose[0]), xy[:, 1] - float(pose[1]))
        k = int(np.argmin(d))
        return float(yaw[k]) if d[k] <= LOOK_MATCH_RADIUS_M else None
    return provider
