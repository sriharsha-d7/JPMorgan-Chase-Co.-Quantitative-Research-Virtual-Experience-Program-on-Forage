import itertools

import numpy as np
import pytest

from quant_research.fico_buckets import _bucket_loglik, assign_rating, optimal_fico_buckets


def _ll(fico, default, edges):
    b = np.searchsorted(np.asarray(edges), fico, side="left")
    total = 0.0
    for i in range(len(edges) + 1):
        m = b == i
        total += _bucket_loglik(np.array([m.sum()]), np.array([default[m].sum()]))[0]
    return total


def test_dp_matches_brute_force():
    rng = np.random.default_rng(0)
    fico = rng.integers(500, 520, 400)
    default = (rng.random(400) < (850 - fico) / 400).astype(int)
    scores = np.unique(fico)
    edges, ll = optimal_fico_buckets(fico, default, n_buckets=3)
    brute = max(_ll(fico, default, list(c)) for c in itertools.combinations(scores[:-1], 2))
    assert ll == pytest.approx(brute)
    assert _ll(fico, default, edges) == pytest.approx(ll)


def test_recovers_planted_boundary():
    fico = np.repeat(np.arange(500, 600), 20)
    default = (fico < 550).astype(int)  # perfect step at 549/550
    edges, _ = optimal_fico_buckets(fico, default, n_buckets=2)
    assert edges == [549.0]


def test_more_buckets_never_worse():
    rng = np.random.default_rng(1)
    fico = rng.integers(400, 850, 2000)
    default = (rng.random(2000) < 1 / (1 + np.exp((fico - 600) / 40))).astype(int)
    lls = [optimal_fico_buckets(fico, default, k)[1] for k in (2, 4, 6)]
    assert lls == sorted(lls)


def test_rating_one_is_best_credit():
    edges = [550.0, 650.0]
    assert list(assign_rating([800, 600, 500], edges)) == [1, 2, 3]


def test_min_bucket_size_respected():
    fico = np.array([500] * 3 + [600] * 50 + [700] * 50)
    default = np.array([1, 1, 1] + [0] * 100)
    edges, _ = optimal_fico_buckets(fico, default, n_buckets=2, min_bucket_size=10)
    assert edges != [500.0]


def test_input_validation():
    with pytest.raises(ValueError):
        optimal_fico_buckets([500, 600], [0, 2], 2)
    with pytest.raises(ValueError):
        optimal_fico_buckets([500, 600], [0, 1], 5)
