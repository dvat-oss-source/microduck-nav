#!/usr/bin/env python3
"""Compare benchmark traces without treating profiling time as trajectory state.

Directory inputs are paired by ``*.npz`` basename. The deterministic comparison
requires byte-identical ``qpos`` and ``t`` arrays, equal task JSON, and equal
trace metadata after removing only timing-only fields: ``plan_time_s``,
``plan_times_s``, and ``wall_s`` (at any metadata nesting level). In particular,
``plan_log`` entries remain compared except for their individual
``plan_time_s`` values. The per-step ``commands``, ``estimates`` and
``transitions`` therefore remain subject to exact metadata equality.

NPZ container bytes are intentionally not compared: ZIP metadata/compression
can vary while the decoded simulation trace remains identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


TIMING_KEYS = {"plan_time_s", "plan_times_s", "wall_s"}
REQUIRED_METADATA = ("commands", "estimates", "transitions")


def _strip_timing(value: Any) -> Any:
    """Copy JSON-like metadata while removing explicit profiling fields only."""
    if isinstance(value, dict):
        return {key: _strip_timing(item) for key, item in value.items()
                if key not in TIMING_KEYS}
    if isinstance(value, list):
        return [_strip_timing(item) for item in value]
    return value


def _array_bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (left.shape == right.shape and left.dtype == right.dtype
            and np.ascontiguousarray(left).tobytes()
            == np.ascontiguousarray(right).tobytes())


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        needed = {"qpos", "t", "metadata", "task"}
        missing = needed.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        qpos = np.array(archive["qpos"], copy=True)
        times = np.array(archive["t"], copy=True)
        metadata = json.loads(str(archive["metadata"]))
        task = json.loads(str(archive["task"]))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata must decode to an object")
    missing_meta = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing_meta:
        raise ValueError(f"{path}: metadata lacks {missing_meta}")
    return qpos, times, task, metadata


def _pairs(left: Path, right: Path, episodes: list[str] | None = None
           ) -> list[tuple[str, Path, Path]]:
    if left.is_file() and right.is_file():
        if episodes:
            raise ValueError("--episodes can only be used with directory inputs")
        return [(left.stem, left, right)]
    if left.is_dir() and right.is_dir():
        lfiles = {p.name: p for p in left.glob("*.npz")}
        rfiles = {p.name: p for p in right.glob("*.npz")}
        if episodes:
            names = [f"{ep}.npz" for ep in episodes]
            missing_left = sorted(set(names).difference(lfiles))
            missing_right = sorted(set(names).difference(rfiles))
            if missing_left or missing_right:
                raise ValueError(f"requested traces missing: left={missing_left}, right={missing_right}")
            return [(Path(name).stem, lfiles[name], rfiles[name]) for name in names]
        if set(lfiles) != set(rfiles):
            raise ValueError(
                f"trace sets differ: left-only={sorted(set(lfiles)-set(rfiles))}, "
                f"right-only={sorted(set(rfiles)-set(lfiles))}")
        return [(Path(name).stem, lfiles[name], rfiles[name])
                for name in sorted(lfiles)]
    raise ValueError("inputs must both be NPZ files or both directories")


def compare_trace(left: Path, right: Path) -> dict[str, Any]:
    lq, lt, ltask, lmeta = _load(left)
    rq, rt, rtask, rmeta = _load(right)
    checks = {
        "qpos_bit_identical": _array_bytes_equal(lq, rq),
        "t_bit_identical": _array_bytes_equal(lt, rt),
        "task_equal": ltask == rtask,
        "metadata_equal_excluding_timing": _strip_timing(lmeta) == _strip_timing(rmeta),
    }
    return {"left": str(left), "right": str(right), "checks": checks,
            "passed": all(checks.values()),
            "timing_exclusions": sorted(TIMING_KEYS)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="NPZ trace or directory of traces")
    parser.add_argument("right", type=Path, help="matching NPZ trace or trace directory")
    parser.add_argument("--episodes", nargs="+", help="directory mode: compare these episode IDs only")
    args = parser.parse_args(argv)
    try:
        results = [{"episode": ep, **compare_trace(lpath, rpath)}
                   for ep, lpath, rpath in _pairs(args.left, args.right, args.episodes)]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    report = {"passed": bool(results) and all(item["passed"] for item in results),
              "n_traces": len(results), "timing_exclusions": sorted(TIMING_KEYS),
              "episodes": results}
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
