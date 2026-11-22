"""Benchmark statistics: Wilson CI, SPL (Anderson et al. 2018), percentiles, bootstrap CI."""
import math
from typing import Sequence, Tuple

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def spl(success: Sequence[bool], shortest: Sequence[float], path: Sequence[float]) -> float:
    """SPL = mean_i S_i * l_i / max(p_i, l_i); l_i = geodesic shortest-path length, p_i = walked length."""
    n = len(success)
    if n == 0:
        return 0.0
    return float(np.mean(spl_terms(success, shortest, path)))


def spl_terms(success, shortest, path):
    out = []
    for s, l, p in zip(success, shortest, path):
        m = max(p, l)
        out.append(float(s) * l / m if m > 0 else float(s))
    return np.array(out)


def percentiles(x: Sequence[float], qs=(50, 90, 95, 99)) -> dict:
    x = np.asarray([a for a in x if a is not None and not (isinstance(a, float) and math.isnan(a))], float)
    if len(x) == 0:
        return {"p%g" % q: None for q in qs}
    return {"p%g" % q: float(np.percentile(x, q)) for q in qs}


def bootstrap_ci(x: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return (0.0, 1.0)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def two_proportion_pvalue(k1: float, n1: int, k2: float, n2: int) -> float:
    """Two-sided pooled two-proportion z test (normal approximation)."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("rate comparisons require positive task counts")
    if not (0 <= k1 <= n1 and 0 <= k2 <= n2):
        raise ValueError("rate count outside sample size")
    pooled = (k1 + k2) / (n1 + n2)
    variance = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    if variance == 0:
        return 1.0
    z = abs(k1 / n1 - k2 / n2) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))
