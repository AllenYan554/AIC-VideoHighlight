"""Directed tests for Stage 7.16 TVSum zero-shot metrics (pure functions)."""

from __future__ import annotations

import numpy as np

from aic_video_highlight.ftnet.tvsum_metrics import (
    average_ranks,
    kendall_tau_b,
    mean_pool_to_segments,
    ndcg_at_fraction,
    spearman,
    video_bootstrap_ci,
)


def test_average_ranks_ties():
    ranks = average_ranks(np.array([10.0, 20.0, 20.0, 30.0]))
    assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_perfect_and_constant():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, x) == 1.0
    assert spearman(x, -x) == -1.0
    assert spearman(np.ones(5), x) is None
    assert spearman(x, np.ones(5)) is None


def test_spearman_matches_pearson_on_ranks():
    x = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0])
    y = np.array([2.0, 7.0, 1.0, 6.0, 3.0, 5.0])
    rx = average_ranks(x)
    ry = average_ranks(y)
    expected = float(np.corrcoef(rx, ry)[0, 1])
    assert abs(spearman(x, y) - expected) < 1e-12


def test_kendall_tau_b_perfect_and_ties():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert kendall_tau_b(x, x) == 1.0
    assert kendall_tau_b(x, -x) == -1.0
    # one tie pair on each side: tau-b denominator shrinks but sign preserved
    xb = np.array([1.0, 1.0, 2.0, 3.0])
    yb = np.array([1.0, 2.0, 3.0, 4.0])
    tau = kendall_tau_b(xb, yb)
    assert tau is not None and 0.0 < tau < 1.0


def test_ndcg_bounds():
    rel = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert ndcg_at_fraction(rel, rel) == 1.0
    assert ndcg_at_fraction(-rel, rel) is not None
    assert 0.0 <= ndcg_at_fraction(-rel, rel) <= 1.0
    assert ndcg_at_fraction(np.arange(10.0), np.zeros(10)) is None


def test_mean_pool_to_segments():
    prob = np.arange(8.0)  # segments of 4
    pooled = mean_pool_to_segments(prob, 2)
    assert pooled.tolist() == [1.5, 5.5]
    # partial tail segment
    pooled2 = mean_pool_to_segments(np.arange(6.0), 2)
    assert pooled2[0] == 1.5
    assert pooled2[1] == 4.5


def test_bootstrap_ci_is_video_level():
    rng = np.random.default_rng(0)
    values = rng.normal(0.3, 0.1, size=50)
    out = video_bootstrap_ci(values, n_boot=2000, seed=1)
    assert out["n_videos"] == 50
    assert out["lo"] < out["mean"] < out["hi"]
    # a single large value cannot explode with only nan handling
    out_none = video_bootstrap_ci(np.array([0.1, 0.2, None, 0.4], dtype=object), n_boot=500)
    assert out_none["n_videos"] == 3
