"""Stage 6.1 literature frame selection: KTS, knapsack and the FS-0 subset adapter.

What this module is
-------------------
The frozen Stage 6.1 control is::

    Qwen Temporal Retrieval -> Frame Projection -> RT-DETR / LOC v1 -> CMP1
      -> TS5 Revised -> FS0 -> predictions

Stage 6.1 compares three *frame selectors* behind TS5 while keeping every other
stage byte-identical:

* Arm A -- ``FS0`` identity control (all frames kept),
* Arm B -- ``PGL-SUM`` (see :mod:`.pgl_sum_selector`),
* Arm C -- ``VASNet`` (see :mod:`.vasnet_selector`).

This module owns the parts shared by Arms B/C:

* the KTS change-point detector and the 0-1 knapsack post-processing that turn
  per-frame importance scores into a shot-level summary, and
* the **FS-0 subset adapter** that can only KEEP or DROP frames that already
  exist in the frozen FS-0 universe.  It can never introduce a frame outside
  FS-0 and never touches bbox geometry.

Upstream provenance
-------------------
* KTS (``calc_scatters`` / ``cpd_nonlin`` / ``cpd_auto``) is ported from
  ``ok1zjf/VASNet`` @ c3787531486f74789dc5e92758edf51e24f56e6d
  (``cpd_nonlin.py``, ``cpd_auto.py``), which are courtesy of KaiyangZhou
  (pytorch-vsumm-reinforce, MIT).  Mathematics unchanged.
* ``knapSack`` is ported from ``e-apostolidis/PGL-SUM`` @
  81d0d6d0ee0470775ad759087deebbce1ceffec3
  (``inference/knapsack_implementation.py``, author Bhavya Jain).  Unchanged.
* ``generate_summary`` reconstruction follows the upstream inference contract
  (expand per-sample scores to the original frame space via ``positions``,
  average within shots, budget a fraction of the original video).  PGL-SUM and
  VASNet each have their own upstream variant; both are represented here.

Licenses: PGL-SUM is academic/non-commercial (CERTH-ITI); VASNet and the KTS
code are MIT.  Retain notices.  See the Stage 6.1 provenance record.

Open scientific items (must be pinned before the Formal run)
------------------------------------------------------------
The author inference paths consume ``change_points`` that were pre-computed by
an *external* dataset pipeline; neither upstream repository defines the KTS
``ncp``/``vmax`` used to build SumMe/TVSum.  This module therefore requires
``KTSConfig`` explicitly and never guesses those values.  See
``adapter_design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from aic_video_highlight.composition.frame_projection import (
    CFR_FPS,
    PTS_TABLE,
    VideoTiming,
    fps_rational,
)

LITERATURE_FRAME_SELECTION_SCHEMA = "aic.stage6_1.literature-frame-selection/v1"

PGL_SUM = "pgl_sum"
VASNET = "vasnet"
METHODS = (PGL_SUM, VASNET)

#: Upstream 0-1 knapsack value scaling (VASNet's ``knapsack_ortools`` multiplies
#: values by 1000 before solving; PGL-SUM's DP uses raw values).  Preserved so
#: tie-breaking matches each author pipeline.
VALUE_SCALE = {PGL_SUM: 1, VASNET: 1000}


class LiteratureFrameSelectionError(ValueError):
    """Raised for malformed literature-selector inputs."""


@dataclass(frozen=True, slots=True)
class KTSConfig:
    """Explicit KTS parameters (no silent defaults; see module docstring)."""

    ncp_max: int
    vmax: float
    lmin: int = 1

    def __post_init__(self) -> None:
        if self.ncp_max < 1:
            raise LiteratureFrameSelectionError("KTS ncp_max must be >= 1")
        if not np.isfinite(self.vmax):
            raise LiteratureFrameSelectionError("KTS vmax must be finite")
        if self.lmin < 1:
            raise LiteratureFrameSelectionError("KTS lmin must be >= 1")


@dataclass(frozen=True, slots=True)
class LiteratureFrameSelection:
    """Auditable result of one literature selector over one video."""

    method: str
    sample_frames: tuple[int, ...]
    frame_scores: tuple[float, ...]
    shot_bounds: tuple[tuple[int, int], ...]
    selected_shots: tuple[int, ...]
    budget_frames: int
    selected_sample_indices: tuple[int, ...]
    selected_source_frames: tuple[int, ...]
    fs0_frames: tuple[int, ...]
    kept_frames: tuple[int, ...]
    dropped_frames: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise LiteratureFrameSelectionError(f"unknown method: {self.method}")
        fs0 = set(self.fs0_frames)
        kept = set(self.kept_frames)
        dropped = set(self.dropped_frames)
        if not kept <= fs0:
            raise LiteratureFrameSelectionError("KEEP contains frames outside the FS-0 universe")
        if kept & dropped:
            raise LiteratureFrameSelectionError("KEEP and DROP overlap")
        if kept | dropped != fs0:
            raise LiteratureFrameSelectionError("KEEP + DROP must cover the FS-0 universe exactly")
        if not set(self.selected_source_frames) <= fs0:
            raise LiteratureFrameSelectionError("selected source frames escape the FS-0 universe")

    @property
    def is_subset_of_fs0(self) -> bool:
        return set(self.kept_frames) <= set(self.fs0_frames)

    @property
    def drop_ratio(self) -> float:
        return len(self.dropped_frames) / len(self.fs0_frames) if self.fs0_frames else 0.0


# ---------------------------------------------------------------------------
# KTS change-point detection (ported, mathematics unchanged)
# ---------------------------------------------------------------------------

def calc_scatters(K: np.ndarray) -> np.ndarray:
    """Scatter matrix for change-point detection (upstream ``cpd_nonlin``)."""
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n + 1, n + 1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1)

    diagK2 = np.diag(K2)

    i = np.arange(n).reshape((-1, 1))
    j = np.arange(n).reshape((1, -1))
    scatters = (
        K1[1:].reshape((1, -1))
        - K1[:-1].reshape((-1, 1))
        - (diagK2[1:].reshape((1, -1)) + diagK2[:-1].reshape((-1, 1)) - K2[1:, :-1].T - K2[:-1, 1:])
        / ((j - i + 1).astype(float) + (j == i - 1).astype(float))
    )
    scatters[j < i] = 0
    return scatters


def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, verbose=False):
    """Dynamic-programming change-point detection (upstream ``cpd_nonlin``)."""
    m = int(ncp)

    (n, n1) = K.shape
    if n != n1:
        raise LiteratureFrameSelectionError("kernel matrix must be square")
    if n < (m + 1) * lmin or n > (m + 1) * lmax or lmax < lmin or lmin < 1:
        raise LiteratureFrameSelectionError("invalid KTS segment-length constraints")

    J = calc_scatters(K)

    I = 1e101 * np.ones((m + 1, n + 1))
    I[0, lmin:lmax] = J[0, lmin - 1:lmax - 1]

    if backtrack:
        p = np.zeros((m + 1, n + 1), dtype=int)
    else:
        p = np.zeros((1, 1), dtype=int)

    for k in range(1, m + 1):
        for l in range((k + 1) * lmin, n + 1):
            tmin = max(k * lmin, l - lmax)
            tmax = l - lmin + 1
            c = J[tmin:tmax, l - 1].reshape(-1) + I[k - 1, tmin:tmax].reshape(-1)
            I[k, l] = np.min(c)
            if backtrack:
                p[k, l] = np.argmin(c) + tmin

    cps = np.zeros(m, dtype=int)
    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k - 1] = p[k, cur]
            cur = cps[k - 1]

    scores = I[:, n].copy()
    scores[scores > 1e99] = np.inf
    return cps, scores


def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """Automatic change-point count selection (upstream ``cpd_auto``)."""
    m = ncp
    (_, scores) = cpd_nonlin(K, m, backtrack=False, **kwargs)

    N = K.shape[0]
    N2 = N * desc_rate

    penalties = np.zeros(m + 1)
    ncp_arange = np.arange(1, m + 1)
    penalties[1:] = (vmax * ncp_arange / (2.0 * N2)) * (np.log(float(N2) / ncp_arange) + 1)

    costs = scores / float(N) + penalties
    m_best = int(np.argmin(costs))
    (cps, scores2) = cpd_nonlin(K, m_best, **kwargs)
    return cps, scores2


def kts_change_points(features: np.ndarray, config: KTSConfig, *, desc_rate: int = 1) -> np.ndarray:
    """Return KTS boundaries (in sample-index space, sorted, excluding 0 and T)."""
    if features.ndim != 2:
        raise LiteratureFrameSelectionError("KTS expects a 2-D [T, D] feature matrix")
    n = features.shape[0]
    if n < 2:
        return np.zeros(0, dtype=int)
    ncp_max = min(int(config.ncp_max), max(1, n - config.lmin))
    K = np.matmul(features, features.T)
    boundaries, _ = cpd_auto(K, ncp_max, config.vmax, desc_rate=desc_rate, lmin=config.lmin)
    boundaries = np.unique(np.clip(np.asarray(boundaries, dtype=int), 1, n - 1))
    return boundaries


def shot_bounds_from_boundaries(n: int, boundaries: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Half-open sample-index shot intervals from interior KTS boundaries."""
    edges = [0] + [int(b) for b in boundaries if 0 < int(b) < n] + [n]
    return tuple((edges[i], edges[i + 1]) for i in range(len(edges) - 1))


# ---------------------------------------------------------------------------
# 0-1 knapsack (ported from PGL-SUM, unchanged)
# ---------------------------------------------------------------------------

def knapSack(W: int, wt: Sequence[int], val: Sequence[float], n: int) -> list[int]:
    """Maximise value in a capacity-``W`` 0-1 knapsack; return selected indices."""
    K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    selected = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i - 1][w]:
            selected.insert(0, i - 1)
            w -= wt[i - 1]

    return selected


# ---------------------------------------------------------------------------
# Sampling grid (reuses the AIC VideoTiming machinery)
# ---------------------------------------------------------------------------

def sample_frames_uniform(timing: VideoTiming, sample_fps: float) -> tuple[int, ...]:
    """Deterministic uniform frame sample, VFR/PTS aware.

    Fractional-frame timestamps are avoided by walking the timing timeline
    through :func:`fps_rational` for CFR and the PTS table for VFR.
    """
    if not np.isfinite(sample_fps) or sample_fps <= 0:
        raise LiteratureFrameSelectionError("sample_fps must be finite and positive")

    if timing.timestamp_mode == CFR_FPS:
        step = float(fps_rational(timing)) / float(sample_fps)
        if step <= 0:
            raise LiteratureFrameSelectionError("invalid CFR sampling step")
        count = int(np.floor((timing.frame_count - 1) / step)) + 1
        return tuple(sorted({int(round(i * step)) for i in range(count)} & set(range(timing.frame_count))))

    if timing.timestamp_mode == PTS_TABLE:
        pts = timing.pts_timestamps
        if pts is None:
            raise LiteratureFrameSelectionError("PTS_TABLE timing requires pts timestamps")
        interval = 1.0 / float(sample_fps)
        selected: list[int] = []
        next_target = 0.0
        for frame, stamp in enumerate(pts):
            if float(stamp) >= next_target - 1e-12:
                selected.append(frame)
                next_target = float(stamp) + interval
        return tuple(selected)

    raise LiteratureFrameSelectionError(f"unsupported timestamp_mode: {timing.timestamp_mode}")


# ---------------------------------------------------------------------------
# Shot-level post-processing (upstream contract)
# ---------------------------------------------------------------------------

def _shot_source_length(shot: tuple[int, int], sample_frames: Sequence[int], n_frames: int) -> int:
    start, end = shot
    first_frame = int(sample_frames[start])
    if end < len(sample_frames):
        last_frame = int(sample_frames[end]) - 1
    else:
        last_frame = n_frames - 1
    return max(1, last_frame - first_frame + 1)


def shot_scores(frame_scores: np.ndarray, shot_bounds: Sequence[tuple[int, int]]) -> list[float]:
    """Average per-sample score inside each shot (upstream ``seg_score``)."""
    scores: list[float] = []
    for start, end in shot_bounds:
        window = frame_scores[start:end]
        scores.append(float(np.mean(window)) if window.size else 0.0)
    return scores


def summary_budget_frames(method: str, n_frames: int) -> int:
    """Summary length budget in original frames (upstream definition)."""
    proportion = 0.15
    if method == PGL_SUM:
        final_max_length = int((n_frames - 1 + 1) * proportion)
    elif method == VASNET:
        final_max_length = int(np.floor(n_frames * proportion))
    else:
        raise LiteratureFrameSelectionError(f"unknown method: {method}")
    return max(1, final_max_length)


def select_shots(method: str, shot_scores_: Sequence[float], shot_lengths: Sequence[int], budget_frames: int) -> list[int]:
    """Solve the shot-level 0-1 knapsack with the upstream objective."""
    scale = VALUE_SCALE[method]
    values = [float(score) * scale for score in shot_scores_]
    # knapSack needs integer-ish values; keep floats for the DP exactly as the
    # reference does for PGL-SUM and use rounded scaled ints for VASNet.
    if method == PGL_SUM:
        dp_values: Sequence[float] = values
    else:
        dp_values = [int(round(v)) for v in values]
    return knapSack(int(budget_frames), list(shot_lengths), list(dp_values), len(shot_lengths))


def literature_select(
    *,
    method: str,
    sample_frames: Sequence[int],
    frame_scores: Sequence[float],
    n_frames: int,
    fs0_frames: Sequence[int],
    shot_bounds: Sequence[tuple[int, int]] | None = None,
    features: np.ndarray | None = None,
    kts: KTSConfig | None = None,
) -> LiteratureFrameSelection:
    """Turn per-sample importance scores into an FS-0 KEEP/DROP mask.

    ``shot_bounds`` may be supplied directly (as the author inference uses
    pre-computed change points); otherwise ``features`` + ``kts`` are required.
    """
    if method not in METHODS:
        raise LiteratureFrameSelectionError(f"unknown method: {method}")
    samples = tuple(int(frame) for frame in sample_frames)
    if list(samples) != sorted(samples) or len(set(samples)) != len(samples):
        raise LiteratureFrameSelectionError("sample_frames must be strictly increasing")
    if any(frame < 0 or frame >= n_frames for frame in samples):
        raise LiteratureFrameSelectionError("sample frame outside the source video")
    scores = np.asarray(frame_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.shape[0] != len(samples):
        raise LiteratureFrameSelectionError("frame_scores length must equal sample count")

    if shot_bounds is None:
        if features is None or kts is None:
            raise LiteratureFrameSelectionError("shot_bounds or (features and kts) is required")
        boundaries = kts_change_points(features, kts)
        shot_bounds = shot_bounds_from_boundaries(len(samples), boundaries)

    lengths = [_shot_source_length(shot, samples, n_frames) for shot in shot_bounds]
    scores_by_shot = shot_scores(scores, shot_bounds)
    budget = summary_budget_frames(method, n_frames)
    selected_shots = tuple(sorted(select_shots(method, scores_by_shot, lengths, budget)))

    selected_samples: list[int] = []
    for shot_index in selected_shots:
        start, end = shot_bounds[shot_index]
        selected_samples.extend(range(start, end))
    selected_source = tuple(sorted({samples[i] for i in selected_samples}))

    fs0 = tuple(sorted({int(frame) for frame in fs0_frames}))
    fs0_set = set(fs0)
    kept = tuple(frame for frame in selected_source if frame in fs0_set)
    kept_set = set(kept)
    dropped = tuple(frame for frame in fs0 if frame not in kept_set)

    return LiteratureFrameSelection(
        method=method,
        sample_frames=samples,
        frame_scores=tuple(float(score) for score in scores),
        shot_bounds=tuple((int(s), int(e)) for s, e in shot_bounds),
        selected_shots=selected_shots,
        budget_frames=budget,
        selected_sample_indices=tuple(selected_samples),
        selected_source_frames=selected_source,
        fs0_frames=fs0,
        kept_frames=kept,
        dropped_frames=dropped,
    )


def apply_keep_drop(
    predictions_by_video: Mapping[str, Sequence[Mapping[str, object]]],
    selections: Mapping[str, LiteratureFrameSelection],
) -> dict[str, list[Mapping[str, object]]]:
    """Drop rejected frames from frozen predictions without touching geometry.

    Every surviving prediction row is returned unchanged: this function only
    removes rows whose ``frame`` is in ``dropped_frames``.  It never rewrites a
    bbox, never re-runs RT-DETR/CMP-1/TS-5 and never adds a frame.
    """
    output: dict[str, list[Mapping[str, object]]] = {}
    for video_id, rows in predictions_by_video.items():
        selection = selections[video_id]
        dropped = set(selection.dropped_frames)
        kept_rows = [row for row in rows if int(row["frame"]) not in dropped]
        kept_frames = {int(row["frame"]) for row in kept_rows}
        expected = set(selection.kept_frames)
        if kept_frames != expected:
            raise LiteratureFrameSelectionError(
                f"{video_id}: KEEP mask does not match prediction frames"
            )
        output[video_id] = kept_rows
    return output
