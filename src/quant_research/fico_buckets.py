"""Optimal FICO bucketing by dynamic programming.

Quantisation problem: given borrowers with a FICO score and a default flag, pick
``n_buckets - 1`` boundaries on the score axis so that a bucket-level default-rate
model explains the observed defaults as well as possible. "As well as possible" is the
Bernoulli log-likelihood

    LL = sum_b [ k_b * ln(p_b) + (n_b - k_b) * ln(1 - p_b) ],   p_b = k_b / n_b

with n_b borrowers and k_b defaults in bucket b. Equal-frequency (``pd.qcut``) bins ignore
the defaults entirely; this maximises the likelihood exactly.

The score axis is reduced to its distinct values (at most ~440 for FICO), so the DP is
O(n_buckets * m^2) with m distinct scores: instant, and the optimum is exact, not heuristic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _bucket_loglik(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Log-likelihood of buckets with n obs and k defaults (0*ln0 treated as 0)."""
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, 0.0)
        a = np.where(k > 0, k * np.log(p), 0.0)
        b = np.where(n - k > 0, (n - k) * np.log1p(-p), 0.0)
    return a + b


def optimal_fico_buckets(
    fico: pd.Series | np.ndarray,
    default: pd.Series | np.ndarray,
    n_buckets: int = 5,
    min_bucket_size: int = 1,
) -> tuple[list[float], float]:
    """Return ``(upper_edges, loglik)`` of the log-likelihood-optimal bucketing.

    ``upper_edges`` has ``n_buckets - 1`` interior boundaries; bucket ``i`` holds scores in
    ``(edge[i-1], edge[i]]`` (first bucket from -inf, last to +inf).
    ``min_bucket_size`` stops the optimiser isolating a handful of borrowers.
    """
    fico = np.asarray(fico)
    default = np.asarray(default)
    if fico.shape != default.shape:
        raise ValueError("fico and default must have the same length")
    if not set(np.unique(default)) <= {0, 1}:
        raise ValueError("default must be 0/1")

    scores = np.unique(fico)
    m = len(scores)
    if not 1 <= n_buckets <= m:
        raise ValueError(f"n_buckets must be in [1, {m}] for this data")

    idx = np.searchsorted(scores, fico)
    n_at = np.bincount(idx, minlength=m)
    k_at = np.bincount(idx, weights=default, minlength=m)
    cn = np.concatenate([[0], np.cumsum(n_at)])  # cn[j] = obs with score index < j
    ck = np.concatenate([[0], np.cumsum(k_at)])

    def seg(i: int, j: int) -> float:
        """LL of one bucket covering distinct-score indices [i, j)."""
        n, k = cn[j] - cn[i], ck[j] - ck[i]
        if n < min_bucket_size:
            return -np.inf
        return float(_bucket_loglik(np.array([n]), np.array([k]))[0])

    neg = -np.inf
    # best[b][j]: max LL splitting the first j distinct scores into b buckets
    best = np.full((n_buckets + 1, m + 1), neg)
    cut = np.zeros((n_buckets + 1, m + 1), dtype=int)
    best[0][0] = 0.0
    for b in range(1, n_buckets + 1):
        for j in range(b, m + 1):
            for i in range(b - 1, j):
                if best[b - 1][i] == neg:
                    continue
                v = best[b - 1][i] + seg(i, j)
                if v > best[b][j]:
                    best[b][j], cut[b][j] = v, i
    if best[n_buckets][m] == neg:
        raise ValueError("no feasible bucketing; lower min_bucket_size or n_buckets")

    edges, j = [], m
    for b in range(n_buckets, 1, -1):
        j = cut[b][j]
        edges.append(float(scores[j - 1]))  # upper edge of the previous bucket
    return sorted(edges), float(best[n_buckets][m])


def assign_rating(fico: pd.Series | np.ndarray, edges: list[float]) -> np.ndarray:
    """Map scores to ratings; **rating 1 = best credit** (highest FICO)."""
    bucket = np.searchsorted(np.asarray(edges), np.asarray(fico), side="left")  # 0 = lowest FICO
    return (len(edges) + 1) - bucket
