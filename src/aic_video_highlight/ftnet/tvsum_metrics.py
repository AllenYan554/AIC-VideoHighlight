"""Rank-correlation and summarization metrics for the Stage 7.16 TVSum zero-shot evaluation.

Pure NumPy functions with no data access. Run/label alignment lives in the
experiment adapter; this module only computes metrics on two aligned 1-D vectors.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "average_ranks",
    "spearman",
    "kendall_tau_b",
    "dcg_at_k",
    "ndcg_at_fraction",
    "mean_pool_to_segments",
    "video_bootstrap_ci",
]


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, ties sharing the mean rank (Spearman convention)."""

    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    sorted_values = values[order]
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _is_constant(values: np.ndarray) -> bool:
    return values.shape[0] == 0 or np.all(values == values[0])


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rank correlation; ``None`` when a vector is constant or empty."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.shape[0] < 2:
        return None
    if _is_constant(x) or _is_constant(y):
        return None
    rx = average_ranks(x)
    ry = average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return None
    return float((rx * ry).sum() / denom)


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float | None:
    """Kendall tau-b with tie correction; ``None`` when undefined."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.shape[0] < 2:
        return None
    n = x.shape[0]
    concordant = discordant = 0
    for i in range(n - 1):
        sx = np.sign(x[i + 1 :] - x[i])
        sy = np.sign(y[i + 1 :] - y[i])
        prod = sx * sy
        concordant += int((prod > 0).sum())
        discordant += int((prod < 0).sum())
    n0 = n * (n - 1) / 2.0

    def _tie_pairs(v: np.ndarray) -> float:
        _, counts = np.unique(v, return_counts=True)
        return float(np.sum(counts * (counts - 1) / 2.0))

    n1 = _tie_pairs(x)
    n2 = _tie_pairs(y)
    denom = float(np.sqrt((n0 - n1) * (n0 - n2)))
    if denom == 0.0:
        return None
    return float((concordant - discordant) / denom)


def dcg_at_k(gains: np.ndarray, k: int) -> float:
    gains = np.asarray(gains, dtype=np.float64)[:k]
    if gains.shape[0] == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.shape[0] + 2))
    return float((gains / discounts).sum())


def ndcg_at_fraction(scores: np.ndarray, relevance: np.ndarray, fraction: float = 0.15) -> float | None:
    """nDCG at the top ``fraction`` of segments when ranked by ``scores``."""

    scores = np.asarray(scores, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=np.float64)
    if scores.shape != relevance.shape or scores.shape[0] == 0:
        return None
    n = scores.shape[0]
    k = max(1, int(np.ceil(fraction * n)))
    k = min(k, n)
    order = np.argsort(-scores, kind="stable")
    dcg = dcg_at_k(relevance[order], k)
    ideal = np.sort(relevance)[::-1]
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return None
    return float(dcg / idcg)


def mean_pool_to_segments(prob: np.ndarray, n_segments: int, samples_per_segment: int = 4) -> np.ndarray:
    """Mean-pool a 2.0 fps probability sequence into 2.0s segments (grid index j -> floor(j/4))."""

    prob = np.asarray(prob, dtype=np.float64)
    pooled = np.full(n_segments, np.nan, dtype=np.float64)
    for k in range(n_segments):
        lo = k * samples_per_segment
        hi = min(lo + samples_per_segment, prob.shape[0])
        if lo < prob.shape[0]:
            chunk = prob[lo:hi]
            if chunk.size:
                pooled[k] = float(chunk.mean())
    return pooled


def video_bootstrap_ci(
    per_video_values: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 20261008,
    alpha: float = 0.05,
) -> dict:
    """Video-level bootstrap CI of the mean of ``per_video_values`` (None entries dropped)."""

    values = np.asarray([v for v in np.asarray(per_video_values, dtype=object).tolist() if v is not None],
                        dtype=np.float64)
    if values.shape[0] == 0:
        return {"mean": None, "lo": None, "hi": None, "n_videos": 0, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    draws = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[draws].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {"mean": float(values.mean()), "lo": lo, "hi": hi,
            "n_videos": int(n), "n_boot": n_boot}
