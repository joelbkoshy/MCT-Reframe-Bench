"""Paired statistics. Every arm answers the same items, so all contrasts are paired."""

from __future__ import annotations

import numpy as np
from scipy import stats


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, input order preserved. NaNs pass through."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    n = valid.size
    running = 0.0
    for rank, idx in enumerate(valid[np.argsort(p[valid])]):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar on paired booleans. Returns (only_b, only_a, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_b = int(np.sum(~a & b))
    only_a = int(np.sum(a & ~b))
    n = only_a + only_b
    if n == 0:
        return only_b, only_a, 1.0
    return only_b, only_a, float(stats.binomtest(only_b, n, 0.5).pvalue)


def wilcoxon_with_effect(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Signed-rank p-value and matched-pairs rank-biserial r (positive: y > x)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = y - x
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return 1.0, 0.0
    _, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    ranks = stats.rankdata(np.abs(nonzero))
    pos, neg = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    return float(p), float((pos - neg) / (pos + neg))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, int), np.asarray(b, int)
    if a.size == 0:
        return float("nan")
    observed = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def cramers_v(table: np.ndarray) -> tuple[float, float, int]:
    """Chi-square test of independence on a contingency table: (V, p, dof).

    Rows or columns that are entirely zero are dropped first; a table that
    collapses to a single row or column carries no association to test.
    """
    t = np.asarray(table, float)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return float("nan"), float("nan"), 0
    chi2, p, dof, _ = stats.chi2_contingency(t, correction=False)
    n = t.sum()
    v = np.sqrt(chi2 / (n * (min(t.shape) - 1)))
    return float(v), float(p), int(dof)
