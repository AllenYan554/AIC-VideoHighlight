"""Mr. HiSum official highlight-detection metric (MAP@15% / MAP@50%), replicated.

This is a faithful, dependency-light reproduction of the official evaluation found
in MRHiSum/MR.HiSum ``model/utils/evaluate_map.py`` + ``model/solver.py``:

1. ``segment_scores``: split a frame-score sequence into uniform clips of length
   ``uniform_clip`` (default 5) and average each clip (last clip may be shorter,
   matching ``torch.split``).
2. GT top-15% / top-50% binary shots via descending sort (floor counts).
3. Model clip scores are normalized with a softmax (rank-invariant) and scored with
   ``sklearn.metrics.average_precision_score``.

TVSum/ODH experiments must NOT reuse this module for their own labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

__all__ = [
    "segment_scores",
    "top15_summary",
    "top50_summary",
    "softmax",
    "aggregate_2fps_to_1hz",
    "map_at_fractions",
]

UNIFORM_CLIP = 5


def segment_scores(frame_scores: np.ndarray, clip: int = UNIFORM_CLIP) -> np.ndarray:
    """Mean-pool a 1-D frame-score sequence into clips of ``clip`` (last may be short)."""

    values = np.asarray(frame_scores, dtype=np.float64)
    n = values.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    out = [values[i : i + clip].mean() for i in range(0, n, clip)]
    return np.asarray(out, dtype=np.float64)


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    order = np.argsort(-scores, kind="stable")
    binary = np.zeros(n, dtype=np.int64)
    if k > 0:
        binary[order[:k]] = 1
    return binary


def top50_summary(scores: np.ndarray) -> np.ndarray:
    """Top 50% of clips as 1s (official: ``len(scores)//2``)."""

    n = np.asarray(scores).shape[0]
    return _top_k(scores, n // 2)


def top15_summary(scores: np.ndarray) -> np.ndarray:
    """Top 15% of clips as 1s (official: ``int(len(scores) * 0.15)``)."""

    n = np.asarray(scores).shape[0]
    return _top_k(scores, int(n * 0.15))


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[0] == 0:
        return scores
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    denom = exp.sum()
    return exp / denom if denom > 0 else exp


def aggregate_2fps_to_1hz(prob_2fps: np.ndarray, n_seconds: int) -> np.ndarray:
    """Mean-pool a 2.0 fps sequence into 1 Hz by averaging the two 0.5s samples of each second."""

    values = np.asarray(prob_2fps, dtype=np.float64)
    out = np.full(n_seconds, np.nan, dtype=np.float64)
    for t in range(n_seconds):
        lo = 2 * t
        hi = min(2 * t + 2, values.shape[0])
        if lo < values.shape[0]:
            out[t] = float(values[lo:hi].mean())
    return out


def map_at_fractions(gt_gtscore: np.ndarray, model_score: np.ndarray,
                     *, clip: int = UNIFORM_CLIP) -> dict:
    """Return MAP@15% and MAP@50% for one video using the official procedure.

    ``gt_gtscore`` is the 1 Hz Most-Replayed importance; ``model_score`` is the
    model's per-second score on the same timeline. ``None`` when undefined.
    """

    gt_seg = segment_scores(gt_gtscore, clip)
    m_seg = segment_scores(model_score, clip)
    length = min(gt_seg.shape[0], m_seg.shape[0])
    gt_seg = gt_seg[:length]
    m_seg = m_seg[:length]
    if length == 0:
        return {"map15": None, "map50": None, "n_segments": 0}
    m_seg = softmax(m_seg)
    gt15 = top15_summary(gt_seg)
    gt50 = top50_summary(gt_seg)
    map15 = float(average_precision_score(gt15, m_seg)) if gt15.sum() > 0 else None
    map50 = float(average_precision_score(gt50, m_seg)) if gt50.sum() > 0 else None
    return {"map15": map15, "map50": map50, "n_segments": int(length)}
