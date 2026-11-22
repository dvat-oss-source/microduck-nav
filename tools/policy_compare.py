"""Compare navigation results from same-seed GT benchmark reports.

Usage::

    python -m tools.policy_compare --base reports/bench_selected_gt.json \
        --other reports/bench_walk_gt.json reports/bench_velocity_hard_gt.json \
        --out reports/policy_comparison_v1

The full selected-policy report and nav-only comparison reports are explicitly
projected to matched nav episodes before calling ``nav.bench.compare``. The
benchmark comparator remains responsible for refusing seed, mode, or sample
count mismatches.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from nav import bench, stats

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "policies.lock.json"
REPORTS = ROOT / "reports"
ALPHA = 0.05
NAV_N = 200
RATE_METRICS = (
    ("success_rate", "successes", "success_ci", True),
    ("fall_rate", "falls", "fall_ci", False),
    ("hole_rate", "hole_incursions", "hole_ci", False),
)


def _read_report(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return report, hashlib.sha256(raw).hexdigest()


def _mode(report: dict[str, Any]) -> str | None:
    mode = report.get("mode")
    if mode is not None:
        return mode
    config = report.get("runner_config")
    return config.get("mode") if isinstance(config, dict) else None


def _nav_episodes(report: dict[str, Any], source: Path) -> dict[str, Any]:
    episodes = report.get("episodes")
    if not isinstance(episodes, dict):
        raise ValueError(f"{source} has no episode verdict map")
    selected = {key: value for key, value in episodes.items() if key.startswith("nav_")}
    if not selected:
        raise ValueError(f"{source} contains no nav-only episodes")
    for key, verdict in selected.items():
        if not isinstance(verdict, dict):
            raise ValueError(f"{source}: {key} verdict is not an object")
        task_type = verdict.get("task_type")
        if task_type not in (None, "nav"):
            raise ValueError(f"{source}: {key} is not a navigation episode")
    return selected


def _nav_view(report: dict[str, Any], source: Path) -> dict[str, Any]:
    """Project one source report to just its nav task and nav episode verdicts."""
    by_task = report.get("by_task")
    nav = by_task.get("nav") if isinstance(by_task, dict) else None
    if not isinstance(nav, dict):
        raise ValueError(f"{source} has no by_task.nav metrics")
    episodes = _nav_episodes(report, source)
    count = int(nav.get("n", 0))
    if count != len(episodes):
        raise ValueError(f"{source} nav count {count} does not match {len(episodes)} nav episodes")
    task_counts = report.get("task_counts", {})
    nav_task_count = task_counts.get("nav", count) if isinstance(task_counts, dict) else count
    view = copy.deepcopy(report)
    view["by_task"] = {"nav": copy.deepcopy(nav)}
    view["task_counts"] = {"nav": nav_task_count}
    view["episodes"] = episodes
    view["n_episodes"] = len(episodes)
    return view


def _require_report_shapes(base: dict[str, Any], other: dict[str, Any],
                           base_path: Path, other_path: Path) -> None:
    """Check report roles while leaving seed/mode/count equality to bench.compare."""
    if base.get("comparison_policy"):
        raise ValueError(f"--base must be the full selected-policy report, not {base_path}")
    if base.get("limit") is not None:
        raise ValueError(f"--base must be a full benchmark report, not a limited run: {base_path}")
    base_counts = base.get("task_counts", {})
    if not isinstance(base_counts, dict) or any(base_counts.get(k) != n
                                               for k, n in (("nav", 200), ("kick", 50), ("pick", 50))):
        raise ValueError(f"--base must contain the full frozen 200/50/50 suite: {base_path}")
    if not other.get("comparison_policy"):
        raise ValueError(f"--other must be an explicit nav-only comparison report: {other_path}")
    if other.get("limit") is not None:
        raise ValueError(f"--other must be a full nav comparison report, not a limited run: {other_path}")
    counts = other.get("task_counts", {})
    if not isinstance(counts, dict) or set(counts) != {"nav"}:
        raise ValueError(f"--other must contain only navigation task counts: {other_path}")


def _counts(view: dict[str, Any]) -> dict[str, int]:
    episodes = view["episodes"].values()
    return {
        "success": sum(bool(v.get("success")) for v in episodes),
        "fall": sum(bool(v.get("fallen")) for v in view["episodes"].values()),
        "hole": sum(bool(v.get("hole_incursion")) for v in view["episodes"].values()),
    }


def _validate_nav_view(view: dict[str, Any], source: Path) -> None:
    """Require the agreed frozen navigation suite and internally consistent counts."""
    nav = view["by_task"]["nav"]
    if nav.get("n") != NAV_N or view["task_counts"].get("nav") != NAV_N or len(view["episodes"]) != NAV_N:
        raise ValueError(f"{source} must describe exactly {NAV_N} frozen nav episodes")
    observed = _counts(view)
    expected = {
        "success": nav.get("successes"),
        "fall": round(float(nav.get("fall_rate", 0.0)) * NAV_N),
        "hole": nav.get("hole_incursions"),
    }
    for metric, count in expected.items():
        if count is None or int(count) != observed[metric]:
            raise ValueError(f"{source} {metric} episode count disagrees with by_task.nav")
    if round(float(nav.get("hole_rate", 0.0)) * NAV_N) != observed["hole"]:
        raise ValueError(f"{source} hole_rate disagrees with episode verdicts")


def _rate_deltas(other_view: dict[str, Any], base_view: dict[str, Any]) -> dict[str, Any]:
    other_nav, base_nav = other_view["by_task"]["nav"], base_view["by_task"]["nav"]
    other_counts, base_counts = _counts(other_view), _counts(base_view)
    out = {}
    for rate_key, count_key, ci_key, _higher_is_better in RATE_METRICS:
        label = {"success_rate": "success", "fall_rate": "fall", "hole_rate": "hole"}[rate_key]
        k_other, k_base = other_counts[label], base_counts[label]
        p_value = stats.two_proportion_pvalue(k_other, NAV_N, k_base, NAV_N)
        out[rate_key] = {
            "base_count": k_base,
            "other_count": k_other,
            "n": NAV_N,
            "base_rate": float(base_nav[rate_key]),
            "other_rate": float(other_nav[rate_key]),
            "delta": float(other_nav[rate_key]) - float(base_nav[rate_key]),
            "delta_percentage_points": 100.0 * (float(other_nav[rate_key]) - float(base_nav[rate_key])),
            "base_ci95": list(base_nav[ci_key]),
            "other_ci95": list(other_nav[ci_key]),
            "p_value_two_sided": p_value,
            "significant_at_0_05": p_value < ALPHA,
            "test": "two_proportion_z",
            "alpha": ALPHA,
        }
    return out


def _spl_delta(other_nav: dict[str, Any], base_nav: dict[str, Any]) -> dict[str, Any]:
    base_spl, other_spl = float(base_nav["spl"]), float(other_nav["spl"])
    return {
        "base": base_spl,
        "other": other_spl,
        "delta": other_spl - base_spl,
        "base_ci95": list(base_nav["spl_ci"]),
        "other_ci95": list(other_nav["spl_ci"]),
        "ci_overlap": not (other_nav["spl_ci"][1] < base_nav["spl_ci"][0]
                           or base_nav["spl_ci"][1] < other_nav["spl_ci"][0]),
    }


def _flat_gate_status(policy: str | None, report: dict[str, Any], report_path: Path) -> dict[str, Any] | None:
    if policy != "velstand_flat":
        return None
    path = REPORTS / "follower_sim_velstand_flat.json"
    result: dict[str, Any] = {
        "policy": policy,
        "file": str(path.relative_to(ROOT)),
        "matching_current": False,
        "eligible": False,
        "reasons": [],
        "gate_numbers": None,
    }
    if not path.exists():
        result["reasons"].append("current follower gate is missing")
        return result
    gate, digest = _read_report(path)
    lock = json.loads(LOCK_PATH.read_text())
    expected_md5 = lock.get(policy, {}).get("md5")
    current_follower_sha = hashlib.sha256((ROOT / "nav" / "follower.py").read_bytes()).hexdigest()
    metrics = ("passed", "n_passed", "reached", "n", "falls", "worst_cross_track_rms",
               "median_cross_track_rms")
    result["gate_numbers"] = {key: gate.get(key) for key in metrics}
    result["gate_sha256"] = digest
    runner = report.get("runner_config", {})
    runner_dynamics = runner.get("dynamics") if isinstance(runner, dict) else None
    frozen = report.get("frozen_provenance") or {}
    checks = {
        "policy_md5": gate.get("policy_md5") == expected_md5,
        "dynamics": gate.get("dynamics") == runner_dynamics,
        "dynamics_version": gate.get("dynamics_version") == frozen.get("dynamics_version"),
        "current_follower_source": gate.get("follower_sha256") == current_follower_sha,
    }
    result["matching_checks"] = checks
    result["matching_current"] = all(checks.values())
    result["eligible"] = result["matching_current"] and bool(gate.get("passed"))
    for key, matches in checks.items():
        if not matches:
            result["reasons"].append(f"current follower gate {key} mismatch")
    if not gate.get("passed"):
        result["reasons"].append("current follower gate failed")
    if not result["matching_current"]:
        result["reasons"].append("gate is not a matching current gate")
    if result["eligible"]:
        result["reasons"] = []
    return result


def compare_reports(base_path: Path, other_paths: list[Path]) -> dict[str, Any]:
    base_report, base_sha = _read_report(base_path)
    base_view = _nav_view(base_report, base_path)
    _validate_nav_view(base_view, base_path)
    base_policy = base_report.get("runner_config", {}).get("policy")
    if _mode(base_view) != "gt":
        raise ValueError(f"--base must be a GT report: {base_path}")
    selected_path = REPORTS / "walker_gate.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text()).get("selected")
        if base_policy != selected:
            raise ValueError(f"--base policy {base_policy!r} is not current selected policy {selected!r}")

    comparisons = []
    for other_path in other_paths:
        other_report, other_sha = _read_report(other_path)
        _require_report_shapes(base_report, other_report, base_path, other_path)
        other_view = _nav_view(other_report, other_path)

        # Keep this call authoritative for the frozen seed, mode, and n refusals.
        regressions = bench.compare(other_view, base_view)
        _validate_nav_view(other_view, other_path)
        if _mode(other_view) != "gt":
            raise ValueError(f"--other must be a GT report: {other_path}")
        if set(other_view["episodes"]) != set(base_view["episodes"]):
            raise ValueError(f"navigation episode ids differ between {base_path} and {other_path}")

        config = other_report.get("runner_config")
        policy = config.get("policy") if isinstance(config, dict) else None
        flat_gate = _flat_gate_status(policy, other_report, other_path)
        row: dict[str, Any] = {
            "policy": policy,
            "source_report": str(other_path),
            "source_report_sha256": other_sha,
            "nav_view": {
                "task_count": other_view["task_counts"].get("nav"),
                "episode_count": len(other_view["episodes"]),
                "episode_ids_sha256": hashlib.sha256("\n".join(sorted(other_view["episodes"])).encode()).hexdigest(),
            },
            "eligible": flat_gate["eligible"] if flat_gate is not None else True,
        }
        if flat_gate is not None:
            row["follower_gate"] = flat_gate
        if row["eligible"]:
            row["rate_deltas"] = _rate_deltas(other_view, base_view)
            row["spl"] = _spl_delta(other_view["by_task"]["nav"], base_view["by_task"]["nav"])
            row["significant_regressions_from_bench_compare"] = regressions
        else:
            row["benchmark_metrics"] = None
            row["ineligible_note"] = "Flat walker benchmark metrics are withheld because its matching current follower gate did not pass."
        comparisons.append(row)

    return {
        "schema": "policy_comparison_v1",
        "benchmark_reference_hash": base_sha,
        "reference": {
            "source_report": str(base_path),
            "source_report_sha256": base_sha,
            "run": base_report.get("run"),
            "timestamp": base_report.get("timestamp"),
            "policy": base_policy,
            "mode": _mode(base_view),
            "seeds_sha256": base_view.get("seeds_sha256"),
            "nav_task_count": base_view["task_counts"].get("nav"),
            "nav_episode_count": len(base_view["episodes"]),
            "nav_episode_ids_sha256": hashlib.sha256("\n".join(sorted(base_view["episodes"])).encode()).hexdigest(),
        },
        "method": {
            "nav_only_view": "by_task.nav and nav_* episode verdicts only",
            "rate_test": "two-sided pooled two-proportion z test via nav.stats.two_proportion_pvalue",
            "alpha": ALPHA,
            "spl_delta": "other minus selected-policy SPL; Wilson/bootstrap CIs retained from benchmark reports",
            "benchmark_compare_regressions": "nav.bench.compare; its frozen seed/mode/count refusals are preserved",
        },
        "comparisons": comparisons,
    }


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_markdown(result: dict[str, Any]) -> str:
    ref = result["reference"]
    lines = [
        "# Frozen GT navigation policy comparison",
        "",
        f"Reference policy: **{ref['policy']}** ({ref['timestamp'] or ref['run']})",
        f"Mode: `{ref['mode']}` · frozen seed SHA-256: `{ref['seeds_sha256']}`",
        f"Benchmark reference SHA-256: `{result['benchmark_reference_hash']}`",
        f"Matched navigation episodes: {ref['nav_episode_count']}",
        "",
        "Rate deltas are **other − reference**; p-values are two-sided and test changes in either direction.",
        "",
        "| Policy | Eligible | Success Δ (p) | Fall Δ (p) | Hole Δ (p) | SPL Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["comparisons"]:
        if not row["eligible"]:
            gate = row.get("follower_gate", {}).get("gate_numbers") or {}
            lines.append(f"| {row.get('policy') or 'unknown'} | No (follower gate) | — | — | — | — |")
            lines.append(f"|  |  | gate {gate.get('n_passed')}/{gate.get('n')} passed; "
                         f"worst RMS {_fmt_m(gate.get('worst_cross_track_rms'))}; falls {gate.get('falls')} |  |  |  |")
            continue
        rates = row["rate_deltas"]
        def rate_cell(key: str) -> str:
            metric = rates[key]
            return f"{metric['delta_percentage_points']:+.1f} pp ({metric['p_value_two_sided']:.3g})"
        lines.append(f"| {row.get('policy') or 'unknown'} | Yes | {rate_cell('success_rate')} | "
                     f"{rate_cell('fall_rate')} | {rate_cell('hole_rate')} | {row['spl']['delta']:+.3f} |")
    lines.extend(["", "## Details", ""])
    for row in result["comparisons"]:
        lines.append(f"### {row.get('policy') or 'unknown'}")
        lines.append("")
        if not row["eligible"]:
            gate = row.get("follower_gate", {})
            lines.append("Ineligible for benchmark comparison: " + "; ".join(gate.get("reasons", [])) + ".")
            lines.append("")
            lines.append(f"Current gate result: {gate.get('gate_numbers')}")
            lines.append("")
            continue
        lines.append(f"Source: `{row['source_report']}` · SHA-256 `{row['source_report_sha256']}`")
        lines.append("")
        for key, metric in row["rate_deltas"].items():
            lines.append(f"- **{key}:** reference {_fmt_pct(metric['base_rate'])} {metric['base_ci95']}; "
                         f"other {_fmt_pct(metric['other_rate'])} {metric['other_ci95']}; "
                         f"delta {metric['delta_percentage_points']:+.2f} pp, p={metric['p_value_two_sided']:.5g}.")
        spl = row["spl"]
        lines.append(f"- **SPL:** reference {spl['base']:.3f} {spl['base_ci95']}; "
                     f"other {spl['other']:.3f} {spl['other_ci95']}; delta {spl['delta']:+.3f}.")
        lines.append(f"- Significant regressions flagged by `bench.compare`: {row['significant_regressions_from_bench_compare'] or 'none'}.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _fmt_m(value: Any) -> str:
    return "n/a" if value is None else f"{1000 * float(value):.1f} mm"


def output_paths(out: Path) -> tuple[Path, Path]:
    if out.suffix.lower() == ".json":
        return out, out.with_suffix(".md")
    if out.suffix.lower() == ".md":
        return out.with_suffix(".json"), out
    return Path(str(out) + ".json"), Path(str(out) + ".md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="full selected-policy GT benchmark report")
    parser.add_argument("--other", required=True, nargs="+", type=Path,
                        help="one or more full nav-only GT comparison reports")
    parser.add_argument("--out", required=True, type=Path,
                        help="output basename (writes .json and .md), or either extension")
    args = parser.parse_args(argv)
    result = compare_reports(args.base, args.other)
    json_path, markdown_path = output_paths(args.out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    markdown_path.write_text(render_markdown(result))
    print(f"wrote {json_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
