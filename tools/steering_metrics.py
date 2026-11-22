#!/usr/bin/env python3
"""Measure steering, yaw, and path statistics from saved trace NPZ files.

For a directory, every top-level ``*.npz`` is reported. Navigation and
non-navigation episodes receive separate aggregates. Command durations are
integrated from the command timestamps through the trace's final timestamp;
pure yaw means exactly ``vx == vy == 0`` and ``wz != 0``. Turn flips ignore
zero-yaw commands. Path smoothing calls the frozen checker's centred MA30
routine directly, without changing scoring.

Optional ``--sample-report`` freezes a deterministic navigation sample and
creates its fixed list plus a baseline metrics JSON for that subset. The
selection algorithm is documented in the resulting fixed-sample artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav import checker


METRIC_KEYS = (
    "wz_sign_flips", "wz_sign_flips_per_minute", "pure_yaw_s",
    "pure_yaw_fraction", "commanded_abs_yaw_rad", "total_abs_yaw_rad", "net_yaw_rad",
    "total_to_net_yaw_ratio", "sway_rms_m", "path_length_raw_m",
    "path_length_ma30_m", "path_ratio_ma30_geodesic", "spl_term",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def command_metrics(commands: Any, trace_times: Any) -> dict[str, float | int | None]:
    commands = np.asarray(commands, dtype=float)
    trace_times = np.asarray(trace_times, dtype=float)
    if commands.size == 0:
        commands = np.empty((0, 4), dtype=float)
    if commands.ndim != 2 or commands.shape[1] != 4:
        raise ValueError("commands must have columns [t, vx, vy, wz]")
    duration = float(trace_times[-1] - trace_times[0]) if len(trace_times) > 1 else 0.0
    if len(commands):
        command_times = commands[:, 0]
        end_time = float(trace_times[-1]) if len(trace_times) else float(command_times[-1])
        dt = np.maximum(0.0, np.diff(np.r_[command_times, end_time]))
        wz, vx, vy = commands[:, 3], commands[:, 1], commands[:, 2]
        active_signs = np.sign(wz[wz != 0])
        flips = int(np.count_nonzero(active_signs[1:] != active_signs[:-1]))
        pure_yaw_s = float(np.sum(dt[(vx == 0) & (vy == 0) & (wz != 0)]))
        total_abs_yaw = float(np.sum(np.abs(wz) * dt))
    else:
        flips, pure_yaw_s, total_abs_yaw = 0, 0.0, 0.0
    return {
        "wz_sign_flips": flips,
        "wz_sign_flips_per_minute": flips / (duration / 60.0) if duration > 0 else None,
        "pure_yaw_s": pure_yaw_s,
        "pure_yaw_fraction": pure_yaw_s / duration if duration > 0 else None,
        "commanded_abs_yaw_rad": total_abs_yaw,
    }


def path_metrics(xy: Any, yaw: Any, geodesic: float | None,
                 success: bool | None = None) -> dict[str, float | None]:
    xy = np.asarray(xy, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) != len(yaw):
        raise ValueError("xy and yaw must have matching shapes (T,2) and (T,)")
    smooth = checker._smooth_xy(xy)
    raw_length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
    smooth_length = float(np.linalg.norm(np.diff(smooth, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
    sway = float(np.sqrt(np.mean(np.sum((xy - smooth) ** 2, axis=1)))) if len(xy) else 0.0
    yaw_unwrapped = np.unwrap(yaw) if len(yaw) else yaw
    net_yaw_unwrapped = float(yaw_unwrapped[-1] - yaw_unwrapped[0]) if len(yaw_unwrapped) else 0.0
    net_yaw = abs(_wrap(float(yaw[-1] - yaw[0]))) if len(yaw) else 0.0
    total_yaw = float(np.abs(np.diff(yaw_unwrapped)).sum()) if len(yaw_unwrapped) > 1 else 0.0
    ratio = (total_yaw / net_yaw) if net_yaw > 0 else (0.0 if total_yaw == 0 else None)
    if geodesic is None or geodesic <= 0:
        path_ratio = None
        spl_term = None
    else:
        path_ratio = smooth_length / geodesic
        spl_term = None if success is None else float(success) * geodesic / max(geodesic, smooth_length)
    return {
        "net_yaw_rad": net_yaw,
        "net_yaw_unwrapped_rad": net_yaw_unwrapped,
        "total_abs_yaw_rad": total_yaw,
        "total_to_net_yaw_ratio": ratio,
        "sway_rms_m": sway,
        "path_length_raw_m": raw_length,
        "path_length_ma30_m": smooth_length,
        "path_ratio_ma30_geodesic": path_ratio,
        "spl_term": spl_term,
    }


def _trace_xy_yaw(qpos: np.ndarray, model_path: str) -> tuple[np.ndarray, np.ndarray]:
    model = checker.load_model(model_path)
    body = model.body(checker.TRUNK).id
    qadr = checker._free_qadr(model, body)
    xy = np.asarray(qpos[:, qadr:qadr + 2], dtype=float)
    yaw = np.asarray([checker._quat_tilt_yaw(row[qadr + 3:qadr + 7])[1]
                      for row in qpos], dtype=float)
    return xy, yaw


def trace_metrics(trace_path: Path, verdict: dict[str, Any] | None = None) -> dict[str, Any]:
    with np.load(trace_path, allow_pickle=False) as archive:
        qpos = np.array(archive["qpos"], copy=True)
        times = np.array(archive["t"], copy=True)
        metadata = json.loads(str(archive["metadata"]))
        task = json.loads(str(archive["task"]))
    xy, yaw = _trace_xy_yaw(qpos, metadata["model_path"])
    commands = command_metrics(metadata.get("commands", []), times)
    success = verdict.get("success") if verdict is not None else None
    path_stats = path_metrics(xy, yaw, task.get("geodesic"), success)
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    result = {
        "episode": task.get("ep", trace_path.stem),
        "task_type": task.get("type"),
        "duration_s": duration,
        "geodesic_m": task.get("geodesic"),
        "success": success,
        **commands,
        **path_stats,
    }
    if verdict is not None:
        result.update({key: verdict.get(key) for key in
                       ("time_to_goal", "hole_incursion", "fallen", "reasons")})
    return result


def _episode_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(source.glob("*.npz"))
    raise FileNotFoundError(source)


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "nav": [row for row in episodes if row.get("task_type") == "nav"],
        "non_nav": [row for row in episodes if row.get("task_type") != "nav"],
    }
    result: dict[str, Any] = {}
    for name, rows in groups.items():
        metrics: dict[str, Any] = {"n": len(rows)}
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            metrics[key] = {"count": len(values), "mean": mean(values) if values else None,
                             "median": median(values) if values else None}
        result[name] = metrics
    return result


def fixed_sample(report_path: Path, sample_size: int = 60,
                 rng_seed: int = 20261109) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    nav = {ep: data for ep, data in report["episodes"].items()
           if ep.startswith("nav_") and data.get("task_type") == "nav"}
    if len(nav) < sample_size:
        raise ValueError(f"frozen report has only {len(nav)} nav episodes; need {sample_size}")
    failures = sorted(ep for ep, data in nav.items() if not data.get("success"))
    successes = [(ep, data) for ep, data in nav.items() if data.get("success")]
    longest = max(successes, key=lambda pair: (pair[1].get("duration", 0.0), pair[0]))[0]
    mandatory = sorted(set(failures + ["nav_125", longest]))
    if len(mandatory) > sample_size:
        raise ValueError(f"mandatory episodes ({len(mandatory)}) exceed sample size {sample_size}")
    pool = sorted(set(nav).difference(mandatory))
    fill_count = sample_size - len(mandatory)
    fill = sorted(np.random.default_rng(rng_seed).choice(pool, size=fill_count,
                                                         replace=False).tolist())
    chosen = sorted(mandatory + fill)
    return {
        "sample_size": sample_size,
        "selection": "Include all frozen GT nav failures, nav_125, and the successful nav episode with maximum frozen report duration. From remaining nav IDs sorted lexicographically, select the needed number without replacement using numpy.random.default_rng(20261109).choice; sort the final ID list lexicographically.",
        "rng_seed": rng_seed,
        "mandatory_ids": mandatory,
        "gt_failure_ids": failures,
        "nav_125_included": "nav_125" in chosen,
        "longest_success_id": longest,
        "longest_success_duration_s": nav[longest].get("duration"),
        "random_fill_ids": fill,
        "episode_ids": chosen,
        "source_report": str(report_path),
        "source_report_sha256": file_sha256(report_path),
        "source_seeds_sha256": report.get("seeds_sha256"),
        "checker_sha256": file_sha256(Path("nav/checker.py")),
    }


def analyze(source: Path, report_path: Path | None = None) -> dict[str, Any]:
    report = json.loads(report_path.read_text()) if report_path else {}
    verdicts = report.get("episodes", {})
    rows = []
    for trace in _episode_paths(source):
        verdict = verdicts.get(trace.stem)
        rows.append(trace_metrics(trace, verdict))
    rows.sort(key=lambda row: str(row["episode"]))
    return {"source": str(source), "report_source": str(report_path) if report_path else None,
            "report_source_sha256": file_sha256(report_path) if report_path else None,
            "n_traces": len(rows), "aggregates": summarize(rows), "episodes": rows,
            "metric_definitions": {
                "wz_sign_flips": "Adjacent nonzero commanded wz signs change; zero wz rows are skipped.",
                "pure_yaw": "Exact vx=vy=0 and wz!=0, integrated over command timestamp intervals.",
                "commanded_abs_yaw_rad": "Integral of abs(commanded wz) over command timestamp intervals.",
                "total_abs_yaw_rad": "Sum of absolute frame-to-frame trunk-yaw changes after angle unwrapping.",
                "net_yaw_rad": "Absolute shortest wrapped endpoint trunk-yaw difference.",
                "net_yaw_unwrapped_rad": "Signed endpoint trunk-yaw difference after angle unwrapping.",
                "sway_rms_m": "RMS Euclidean trunk-XY displacement from frozen checker's endpoint-padded centred MA30.",
                "path_lengths": "Raw and MA30 path lengths from trunk XY decoded from frozen checker model.",
                "path_ratio_ma30_geodesic": "MA30 path length divided by the task's frozen geodesic.",
            }}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="one trace NPZ or directory of trace NPZ files")
    parser.add_argument("--report", type=Path, help="benchmark JSON for per-episode outcomes")
    parser.add_argument("--out", type=Path, help="write metrics JSON here (default: stdout)")
    parser.add_argument("--sample-report", type=Path,
                        help="also freeze a deterministic nav sample from this benchmark report")
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--sample-seed", type=int, default=20261109)
    parser.add_argument("--sample-out", type=Path)
    parser.add_argument("--sample-metrics-out", type=Path)
    args = parser.parse_args(argv)
    try:
        output = analyze(args.source, args.report or args.sample_report)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(output, indent=2) + "\n")
        else:
            print(json.dumps(output, indent=2))
        if args.sample_report:
            if not (args.sample_out and args.sample_metrics_out):
                parser.error("--sample-report requires --sample-out and --sample-metrics-out")
            sample = fixed_sample(args.sample_report, args.sample_size, args.sample_seed)
            args.sample_out.parent.mkdir(parents=True, exist_ok=True)
            args.sample_out.write_text(json.dumps(sample, indent=2) + "\n")
            chosen = set(sample["episode_ids"])
            selected = [row for row in output["episodes"] if row["episode"] in chosen]
            sample_report = {"source": output["source"], "report_source": str(args.sample_report),
                             "sample_definition": str(args.sample_out), "n_traces": len(selected),
                             "aggregates": summarize(selected), "episodes": selected,
                             "selection_sha256": file_sha256(args.sample_out)}
            args.sample_metrics_out.parent.mkdir(parents=True, exist_ok=True)
            args.sample_metrics_out.write_text(json.dumps(sample_report, indent=2) + "\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
