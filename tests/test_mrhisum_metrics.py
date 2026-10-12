"""Directed tests for the official Mr. HiSum MAP metric reproduction."""

from __future__ import annotations

import numpy as np

from aic_video_highlight.ftnet.mrhisum_metrics import (
    aggregate_2fps_to_1hz,
    map_at_fractions,
    segment_scores,
    softmax,
    top15_summary,
    top50_summary,
)


def _brute_force_ap(y_true, y_score):
    """Naive AP for an independent cross-check of the sklearn-based path."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="stable")
    y = y_true[order]
    P = y_true.sum()
    if P == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / P)


def test_segment_scores_including_partial_last():
    x = np.arange(12.0)
    seg = segment_scores(x, 5)
    assert seg.shape[0] == 3
    assert np.allclose(seg, [2.0, 7.0, 10.5])  # last clip = mean(10,11)


def test_top_counts_floor():
    assert top15_summary(np.arange(20.0)).sum() == int(20 * 0.15)  # 3
    assert top50_summary(np.arange(20.0)).sum() == 10
    assert top15_summary(np.arange(13.0)).sum() == int(13 * 0.15)  # 1
    assert top50_summary(np.arange(13.0)).sum() == 6


def test_top15_selects_highest():
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4])
    binary = top15_summary(scores)  # int(7*0.15)=1 -> argmax = index 1
    assert binary.tolist() == [0, 1, 0, 0, 0, 0, 0]


def test_map_perfect_and_reversed():
    gt = np.linspace(0.05, 0.95, 50)  # 10 clips -> top15=1, top50=5
    perfect = map_at_fractions(gt, gt.copy())
    assert perfect["map15"] == 1.0 and perfect["map50"] == 1.0
    rev = map_at_fractions(gt, -gt)
    assert rev["map15"] < 0.6 and rev["map50"] < 0.6


def test_map_undefined_for_too_few_clips():
    tiny = np.arange(10.0)  # 2 clips -> int(2*0.15)=0 positives
    out = map_at_fractions(tiny, tiny)
    assert out["map15"] is None and out["map50"] is not None


def test_map_matches_brute_force_ap():
    rng = np.random.default_rng(0)
    gt = rng.random(37)
    model = rng.random(37)
    out = map_at_fractions(gt, model)
    gt_seg = segment_scores(gt, 5)
    m_seg = softmax(segment_scores(model, 5))
    expected15 = _brute_force_ap(top15_summary(gt_seg), m_seg)
    assert abs(out["map15"] - expected15) < 1e-9


def test_softmax_is_rank_preserving():
    x = np.array([-2.0, 0.5, 3.0, 1.0, -1.0])
    assert np.array_equal(np.argsort(softmax(x)), np.argsort(x))


def test_aggregate_2fps_to_1hz():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # 2.5 seconds
    out = aggregate_2fps_to_1hz(x, 3)
    assert np.allclose(out[:2], [0.5, 2.5])
    assert out[2] == 4.0  # only one 0.5s sample available
