import csv
import json

import pytest

from tools.compare_steering import compare, main


IDS = ["nav_018", "nav_125", "nav_129"]


def _episode(ep, *, flips, pure, ratio, spl, success, time, falls=False, holes=False):
    return {
        "episode": ep,
        "task_type": "nav",
        "wz_sign_flips_per_minute": flips,
        "pure_yaw_fraction": pure,
        "pure_yaw_s": 38.1 if ep == "nav_125" and pure > 0.1 else 10.0 if ep == "nav_125" else 2.0,
        "total_to_net_yaw_ratio": ratio,
        "path_ratio_ma30_geodesic": 1.4,
        "spl_term": spl,
        "time_to_goal": time,
        "success": success,
        "fallen": falls,
        "hole_incursion": holes,
    }


def _fixture_docs():
    before = {
        "episodes": [
            _episode("nav_018", flips=50, pure=0.1, ratio=10, spl=0.0,
                     success=False, time=None),
            _episode("nav_125", flips=100, pure=0.2, ratio=55, spl=0.65,
                     success=True, time=100),
            _episode("nav_129", flips=100, pure=0.2, ratio=20, spl=0.75,
                     success=True, time=130),
        ]
    }
    after = {
        "episodes": [
            _episode("nav_018", flips=10, pure=0.04, ratio=8, spl=0.0,
                     success=False, time=None),
            _episode("nav_125", flips=20, pure=0.08, ratio=20, spl=0.7,
                     success=True, time=90),
            _episode("nav_129", flips=20, pure=0.08, ratio=15, spl=0.8,
                     success=True, time=120),
            _episode("nav_999", flips=0, pure=0, ratio=0, spl=0,
                     success=True, time=1),
        ]
    }
    fixed = {"episode_ids": IDS}
    bench = {
        "episodes": {ep: {"success": row["success"], "time_to_goal": row["time_to_goal"],
                           "fallen": False, "hole_incursion": False}
                     for ep, row in zip(IDS, after["episodes"])},
        "by_task": {"nav": {"plan_time_s": {"p95": 0.18}}},
    }
    return before, after, fixed, bench


def test_compare_uses_exact_fixed_sample_and_reports_criteria():
    before, after, fixed, bench = _fixture_docs()

    result, rows = compare(before, after, fixed, bench)

    assert [row["episode"] for row in rows] == IDS
    assert result["sample_size"] == 3
    assert result["before"]["successes"] == result["after"]["successes"] == 2
    assert result["before"]["spl"] == pytest.approx((0 + 0.65 + 0.75) / 3)
    assert result["after"]["spl"] == pytest.approx((0 + 0.7 + 0.8) / 3)
    assert result["after_benchmark_plan_call_p95_s"] == 0.18
    assert all(result["steering_criteria"].values())
    assert result["steering_criteria_passed"]


def test_compare_rejects_baseline_sample_id_mismatch():
    before, after, fixed, bench = _fixture_docs()
    before["episodes"].pop()

    with pytest.raises(ValueError, match="baseline episode IDs"):
        compare(before, after, fixed, bench)


def test_compare_rejects_missing_after_sample_episode():
    before, after, fixed, bench = _fixture_docs()
    after["episodes"] = [row for row in after["episodes"] if row["episode"] != "nav_129"]

    with pytest.raises(ValueError, match="after metrics missing"):
        compare(before, after, fixed, bench)


@pytest.mark.parametrize("safety_field", ["fallen", "hole_incursion"])
def test_compare_fails_closed_when_after_safety_field_is_missing(safety_field):
    before, after, fixed, bench = _fixture_docs()
    after["episodes"][1].pop(safety_field)

    with pytest.raises(ValueError, match="after safety outcome unavailable"):
        compare(before, after, fixed)


def test_cli_writes_matching_json_and_csv(tmp_path):
    before, after, fixed, bench = _fixture_docs()
    files = {}
    for name, doc in (("before", before), ("after", after), ("fixed", fixed), ("bench", bench)):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(doc))
    out_json, out_csv = tmp_path / "comparison.json", tmp_path / "per_episode.csv"

    status = main([str(files["before"]), str(files["after"]), str(files["fixed"]),
                   "--after-bench", str(files["bench"]), "--out-json", str(out_json),
                   "--out-csv", str(out_csv)])

    assert status == 0
    assert json.loads(out_json.read_text())["sample_ids"] == IDS
    with out_csv.open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert [row["episode"] for row in csv_rows] == IDS
    assert "before_total_net_yaw_ratio" in csv_rows[0]
