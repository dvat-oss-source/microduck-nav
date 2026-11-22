#!/usr/bin/env python3
"""Compare A5 steering metrics on the exact frozen development sample.

Inputs are the frozen ``baseline_sample.json``, an after-run
``steering_metrics.py`` JSON, and ``fixed_sample.json``. An optional after
benchmark report supplies outcome fields and benchmark plan-call p95. The
after metrics may contain the full trace directory; this tool selects exactly
the frozen sample IDs, fails on any missing or duplicate ID, and emits matching
JSON and CSV rows.

This reports steering criteria only. Planner and follower gates are external
requirements and are never inferred from this comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "episode",
    "before_path_ratio", "after_path_ratio",
    "before_pure_yaw_fraction", "after_pure_yaw_fraction",
    "before_pure_yaw_s", "after_pure_yaw_s",
    "before_spl_term", "after_spl_term",
    "before_time_to_goal_s", "after_time_to_goal_s",
    "before_falls", "after_falls", "before_holes", "after_holes",
    "before_wz_flips_per_minute", "after_wz_flips_per_minute",
    "before_total_net_yaw_ratio", "after_total_net_yaw_ratio",
    "before_success", "after_success",
]


def _read(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return obj


def _index_metric_rows(doc: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = doc.get("episodes")
    if not isinstance(rows, list):
        raise ValueError(f"{label}: expected episodes array")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("episode"), str):
            raise ValueError(f"{label}: episode row lacks string episode ID")
        ep = row["episode"]
        if ep in indexed:
            raise ValueError(f"{label}: duplicate episode ID {ep}")
        indexed[ep] = row
    return indexed


def _finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    vals = [float(row[field]) for row in rows if row.get(field) is not None
            and math.isfinite(float(row[field]))]
    return vals


def _median(rows: list[dict[str, Any]], field: str) -> float | None:
    vals = _finite_values(rows, field)
    return statistics.median(vals) if vals else None


def _episode_report(report: dict[str, Any] | None, ep: str) -> dict[str, Any] | None:
    if report is None:
        return None
    result = report.get("episodes", {}).get(ep)
    if result is None:
        raise ValueError(f"after benchmark report lacks {ep}")
    return result


def _status(row: dict[str, Any], report_row: dict[str, Any] | None,
            key: str, report_key: str | None = None) -> Any:
    if report_row is not None:
        return report_row.get(report_key or key)
    return row.get(key)


def _aggregate(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    prefix = side + "_"
    n = len(rows)
    successes = sum(bool(row[prefix + "success"]) for row in rows)
    falls = sum(bool(row[prefix + "falls"]) for row in rows)
    holes = sum(bool(row[prefix + "holes"]) for row in rows)
    spl = _finite_values(rows, prefix + "spl_term")
    times = _finite_values(rows, prefix + "time_to_goal_s")
    return {
        "n": n,
        "successes": successes,
        "success_rate": successes / n if n else None,
        "spl": statistics.mean(spl) if spl else None,
        "falls": falls,
        "fall_rate": falls / n if n else None,
        "holes": holes,
        "hole_rate": holes / n if n else None,
        "successful_time_to_goal_median_s": statistics.median(times) if times else None,
    }


def _reduction_flag(before: float | None, after: float | None, factor: float) -> bool:
    if before is None or after is None:
        return False
    return after <= factor * before if before > 0 else after == 0


def compare(baseline: dict[str, Any], after: dict[str, Any], fixed: dict[str, Any],
            after_report: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids = fixed.get("episode_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(ep, str) for ep in ids):
        raise ValueError("fixed sample must contain a nonempty episode_ids list")
    if len(set(ids)) != len(ids):
        raise ValueError("fixed sample contains duplicate episode IDs")
    if ids != sorted(ids):
        raise ValueError("fixed sample episode_ids must be sorted for stable alignment")

    base_rows = _index_metric_rows(baseline, "baseline")
    after_rows = _index_metric_rows(after, "after metrics")
    if set(base_rows) != set(ids):
        raise ValueError("baseline episode IDs do not exactly match fixed sample")
    missing = sorted(set(ids).difference(after_rows))
    if missing:
        raise ValueError(f"after metrics missing fixed sample IDs: {missing}")

    joined: list[dict[str, Any]] = []
    for ep in ids:
        before, post = base_rows[ep], after_rows[ep]
        report_row = _episode_report(after_report, ep)
        after_success = _status(post, report_row, "success")
        if after_success is None:
            raise ValueError(f"after outcome unavailable for {ep}; pass --after-bench or metrics with --report")
        after_fallen = _status(post, report_row, "fallen", "fallen")
        after_hole = _status(post, report_row, "hole_incursion")
        if after_fallen is None or after_hole is None:
            missing = []
            if after_fallen is None:
                missing.append("fallen")
            if after_hole is None:
                missing.append("hole_incursion")
            raise ValueError(f"after safety outcome unavailable for {ep}: {', '.join(missing)}; "
                             "pass --after-bench or metrics with --report")
        after_path = post.get("path_ratio_ma30_geodesic")
        after_spl = post.get("spl_term")
        if after_spl is None:
            geodesic = post.get("geodesic_m")
            path_length = post.get("path_length_ma30_m")
            if geodesic is not None and path_length is not None and geodesic > 0:
                after_spl = float(bool(after_success)) * geodesic / max(geodesic, path_length)
        row = {
            "episode": ep,
            "before_path_ratio": before.get("path_ratio_ma30_geodesic"),
            "after_path_ratio": after_path,
            "before_pure_yaw_fraction": before.get("pure_yaw_fraction"),
            "after_pure_yaw_fraction": post.get("pure_yaw_fraction"),
            "before_spl_term": before.get("spl_term"),
            "after_spl_term": after_spl,
            "before_time_to_goal_s": before.get("time_to_goal"),
            "after_time_to_goal_s": _status(post, report_row, "time_to_goal"),
            "before_falls": bool(before.get("fallen", False)),
            "after_falls": bool(after_fallen),
            "before_holes": bool(before.get("hole_incursion", False)),
            "after_holes": bool(after_hole),
            "before_wz_flips_per_minute": before.get("wz_sign_flips_per_minute"),
            "after_wz_flips_per_minute": post.get("wz_sign_flips_per_minute"),
            "before_total_net_yaw_ratio": before.get("total_to_net_yaw_ratio"),
            "after_total_net_yaw_ratio": post.get("total_to_net_yaw_ratio"),
            "before_success": bool(before["success"]),
            "after_success": bool(after_success),
            "before_pure_yaw_s": before.get("pure_yaw_s"),
            "after_pure_yaw_s": post.get("pure_yaw_s"),
        }
        joined.append(row)

    before_agg = _aggregate(joined, "before")
    after_agg = _aggregate(joined, "after")
    base_nav = [base_rows[ep] for ep in ids]
    post_nav = [after_rows[ep] for ep in ids]
    base_flips = _median(base_nav, "wz_sign_flips_per_minute")
    post_flips = _median(post_nav, "wz_sign_flips_per_minute")
    base_pure = _median(base_nav, "pure_yaw_fraction")
    post_pure = _median(post_nav, "pure_yaw_fraction")
    base_yaw_ratio = _median(base_nav, "total_to_net_yaw_ratio")
    post_yaw_ratio = _median(post_nav, "total_to_net_yaw_ratio")
    nav125 = next((row for row in joined if row["episode"] == "nav_125"), None)
    if nav125 is None:
        raise ValueError("fixed sample must include nav_125")
    flags = {
        "median_flips_per_minute_reduced_by_at_least_75pct": _reduction_flag(base_flips, post_flips, 0.25),
        "median_pure_yaw_fraction_reduced_by_at_least_50pct": _reduction_flag(base_pure, post_pure, 0.5),
        "nav125_flips_per_minute_at_most_30": nav125["after_wz_flips_per_minute"] is not None
            and nav125["after_wz_flips_per_minute"] <= 30,
        "nav125_pure_yaw_s_at_most_19": nav125["after_pure_yaw_s"] is not None
            and nav125["after_pure_yaw_s"] <= 19,
        "median_total_net_yaw_ratio_reduced": base_yaw_ratio is not None and post_yaw_ratio is not None
            and post_yaw_ratio < base_yaw_ratio,
        "nav125_total_net_yaw_ratio_reduced": nav125["after_total_net_yaw_ratio"] is not None
            and nav125["before_total_net_yaw_ratio"] is not None
            and nav125["after_total_net_yaw_ratio"] < nav125["before_total_net_yaw_ratio"],
        "no_increase_falls": after_agg["falls"] <= before_agg["falls"],
        "no_increase_holes": after_agg["holes"] <= before_agg["holes"],
        "all_previously_successful_sample_episodes_remain_successful": all(
            (not row["before_success"]) or row["after_success"] for row in joined),
        "sample_spl_strictly_higher": after_agg["spl"] is not None and before_agg["spl"] is not None
            and after_agg["spl"] > before_agg["spl"],
    }
    plan_p95 = None
    if after_report:
        plan_p95 = (after_report.get("by_task", {}).get("nav", {})
                    .get("plan_time_s", {}).get("p95"))
    result = {
        "sample_ids": ids,
        "sample_size": len(ids),
        "before": before_agg,
        "after": after_agg,
        "deltas": {
            "success_rate": after_agg["success_rate"] - before_agg["success_rate"],
            "spl": None if before_agg["spl"] is None or after_agg["spl"] is None
                else after_agg["spl"] - before_agg["spl"],
            "falls": after_agg["falls"] - before_agg["falls"],
            "holes": after_agg["holes"] - before_agg["holes"],
            "median_wz_flips_per_minute": None if base_flips is None or post_flips is None
                else post_flips - base_flips,
            "median_pure_yaw_fraction": None if base_pure is None or post_pure is None
                else post_pure - base_pure,
            "median_total_net_yaw_ratio": None if base_yaw_ratio is None or post_yaw_ratio is None
                else post_yaw_ratio - base_yaw_ratio,
        },
        "after_benchmark_plan_call_p95_s": plan_p95,
        "steering_criteria": flags,
        "steering_criteria_passed": all(flags.values()),
        "external_gate_status": {
            "planner_100_of_100_zero_lethal_p95_under_0_2s": "not evaluated by this tool",
            "follower_20_of_20_each_walker": "not evaluated by this tool",
            "full_a5_acceptance": "requires steering criteria and external gates",
        },
        "note": "A strictly higher sample SPL is reported as a numeric criterion; this tool does not decide whether the gain is clearly higher.",
    }
    return result, joined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_sample", type=Path)
    parser.add_argument("after_metrics", type=Path)
    parser.add_argument("fixed_sample", type=Path)
    parser.add_argument("--after-bench", type=Path,
                        help="optional benchmark JSON for post-run success, time, falls, holes and plan p95")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        baseline, after, fixed = map(_read, (args.baseline_sample, args.after_metrics, args.fixed_sample))
        bench = _read(args.after_bench) if args.after_bench else None
        result, rows = compare(baseline, after, fixed, bench)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2) + "\n")
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
