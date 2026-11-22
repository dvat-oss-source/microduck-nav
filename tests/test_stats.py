import math
from nav import stats


def test_wilson_known():
    lo, hi = stats.wilson_ci(8, 10)
    assert abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3
    assert stats.wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = stats.wilson_ci(10, 10)
    assert abs(hi - 1.0) < 1e-9 and abs(lo - 0.7225) < 1e-3
    lo, hi = stats.wilson_ci(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3


def test_spl():
    # ep1: success, optimal -> 1 ; ep2: success, walked 2x -> 0.5 ; ep3: fail -> 0 ; ep4: success, shorter than l -> 1
    v = stats.spl([1, 1, 0, 1], [2.0, 2.0, 2.0, 2.0], [2.0, 4.0, 1.0, 1.5])
    assert abs(v - (1 + 0.5 + 0 + 1) / 4) < 1e-12
    assert stats.spl([], [], []) == 0.0


def test_percentiles():
    p = stats.percentiles(list(range(1, 101)), (50, 90))
    assert abs(p["p50"] - 50.5) < 1e-9 and abs(p["p90"] - 90.1) < 1e-9
    assert stats.percentiles([None])["p50"] is None


def test_bootstrap_contains_mean():
    lo, hi = stats.bootstrap_ci([0, 1] * 50)
    assert lo < 0.5 < hi


def test_rate_regression_detects_overlapping_wilson_intervals():
    # The original CI-overlap heuristic missed this 7-point regression at n=200.
    assert 0.005 < stats.two_proportion_pvalue(176, 200, 190, 200) < 0.05
    assert stats.two_proportion_pvalue(190, 200, 190, 200) == 1.0
    assert stats.two_proportion_pvalue(0, 200, 0, 200) == 1.0
