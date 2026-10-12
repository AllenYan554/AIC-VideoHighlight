"""Pure-CPU assembly of one Stage 7 FTNet video from staged upstream artifacts.

This module owns the frozen Native Schema v1.1 computations (FTNet.md section 8)
and the Y target projection.  It never touches Qwen, RT-DETR or the GPU, never
reads ground truth beyond the authoritative MTurk soft votes, and never applies
AIC submission geometry: the Generic CMP call is aspect-agnostic
(``target_aspect_ratio=None``) by construction.

The heavy producer stages (retrieval / detection) leave artifacts on disk; this
module turns them into the exact ``VideoUpstream`` contract owned by
``materialize.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aic_video_highlight.composition.generic_focus import (
    geometric_context_retention,
    plan_subject_focus,
)
from aic_video_highlight.composition.generic_stabilization import (
    stabilize_focus,
    stabilized_focus_velocity,
)
from aic_video_highlight.localization.selection_margin import (
    DETECTION_FLOOR,
    subject_selection_margin,
)
from aic_video_highlight.localization.subject_localization import (
    STATUS_PRIMARY,
    SubjectCandidate,
    SubjectPolicyConfig,
    select_primary_subject,
)

from .materialize import VideoRef, VideoUpstream
from .native_schema import NATIVE_FIELDS
from .sampling import ADJACENCY_RULE
from .youtube_highlights import YouTubeTargets
from .youtube_highlights import project_soft_vote_targets

FOCUS_MARGIN_FRACTION = 0.35
STABILIZATION_ALPHA = 0.6
PERSISTENCE_RADIUS_SEC = 0.5
FROZEN_LOC_POLICY = SubjectPolicyConfig(person_priority=True)

# Pre-registered idx0 fallback (YouTube_Highlights_Preprocessing_Protocol_v1 §12):
# Qwen emits multiple candidate intervals per chunk, so the frozen fallback is
# ``retrieval_support_ratio_norm`` = S_t / max_t(S_t), E_t = 0 -> 0.
IDX0_FALLBACK_NONE = "NONE"
IDX0_FALLBACK_NORM = "retrieval_support_ratio_norm"
IDX0_FALLBACK_INDICATOR = "candidate_indicator"
IDX0_FALLBACKS = (IDX0_FALLBACK_NONE, IDX0_FALLBACK_NORM, IDX0_FALLBACK_INDICATOR)


class ProviderCoreError(RuntimeError):
    """Raised when staged artifacts cannot produce a valid VideoUpstream."""


@dataclass(frozen=True)
class ChunkSpan:
    chunk_index: int
    chunk_start_sec: float
    chunk_end_sec: float


@dataclass(frozen=True)
class RawCandidate:
    chunk_index: int
    start_sec: float
    end_sec: float
    score: float


@dataclass(frozen=True)
class MergedCandidate:
    start_sec: float
    end_sec: float
    score: float


@dataclass(frozen=True)
class RetrievalContext:
    chunks: tuple[ChunkSpan, ...]
    raw_candidates: tuple[RawCandidate, ...]
    merged_candidates: tuple[MergedCandidate, ...]


@dataclass(frozen=True)
class DetectionArtifacts:
    offsets: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    class_names: tuple[str, ...]

    def frame_candidates(self, frame_index: int) -> tuple[SubjectCandidate, ...]:
        start = int(self.offsets[frame_index])
        end = int(self.offsets[frame_index + 1])
        result: list[SubjectCandidate] = []
        for row in range(start, end):
            label_id = int(self.labels[row])
            label = (
                self.class_names[label_id]
                if 0 <= label_id < len(self.class_names)
                else str(label_id)
            )
            box = tuple(float(value) for value in self.boxes[row])
            result.append(
                SubjectCandidate(
                    box=box, score=float(self.scores[row]), label_id=label_id, label=label
                )
            )
        return tuple(result)


@dataclass
class VideoArtifacts:
    ref: VideoRef
    width: int
    height: int
    timestamps: np.ndarray
    source_frame_ids: np.ndarray
    adjacency_mask: np.ndarray
    visual: np.ndarray
    retrieval: RetrievalContext
    detections: DetectionArtifacts
    mturk_clips: tuple
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)


def _candidate_flags(timestamps: np.ndarray, merged: tuple[MergedCandidate, ...]) -> np.ndarray:
    flags = np.zeros(timestamps.shape[0], dtype=bool)
    for candidate in merged:
        flags |= (timestamps >= candidate.start_sec) & (timestamps < candidate.end_sec)
    return flags


def _context_indices(timestamps: np.ndarray, merged: tuple[MergedCandidate, ...]) -> np.ndarray:
    """Deterministic single-candidate context: highest score, then earliest, longest."""

    index = np.full(timestamps.shape[0], -1, dtype=np.int64)
    if not merged:
        return index
    order = sorted(
        range(len(merged)),
        key=lambda i: (-merged[i].score, merged[i].start_sec, -(merged[i].end_sec - merged[i].start_sec), i),
    )
    for candidate_index in order:
        candidate = merged[candidate_index]
        mask = (timestamps >= candidate.start_sec) & (timestamps < candidate.end_sec) & (index < 0)
        index[mask] = candidate_index
    return index


def _support_ratio(
    timestamps: np.ndarray,
    retrieval: RetrievalContext,
) -> np.ndarray:
    frame_count = timestamps.shape[0]
    exposure = np.zeros(frame_count, dtype=np.float64)
    support = np.zeros(frame_count, dtype=np.float64)
    for chunk in retrieval.chunks:
        covered = (timestamps >= chunk.chunk_start_sec) & (timestamps < chunk.chunk_end_sec)
        exposure += covered.astype(np.float64)
        segment_covered = np.zeros(frame_count, dtype=bool)
        for raw in retrieval.raw_candidates:
            if raw.chunk_index != chunk.chunk_index:
                continue
            segment_covered |= (
                (timestamps >= raw.start_sec)
                & (timestamps < raw.end_sec)
                & (timestamps >= chunk.chunk_start_sec)
                & (timestamps < chunk.chunk_end_sec)
            )
        support += segment_covered.astype(np.float64)
    ratio = np.zeros(frame_count, dtype=np.float64)
    valid = exposure > 0
    ratio[valid] = support[valid] / exposure[valid]
    return np.clip(ratio, 0.0, 1.0)


def apply_idx0_fallback(support_ratio: np.ndarray, fallback: str) -> np.ndarray:
    """Deterministic pre-registered replacement for a degenerate idx0 column."""

    if fallback == IDX0_FALLBACK_NONE:
        return support_ratio
    if fallback == IDX0_FALLBACK_NORM:
        peak = float(np.max(support_ratio)) if support_ratio.size else 0.0
        if peak <= 0.0:
            return np.zeros_like(support_ratio)
        return np.clip(support_ratio / peak, 0.0, 1.0)
    if fallback == IDX0_FALLBACK_INDICATOR:
        return (support_ratio > 0.0).astype(np.float64)
    raise ProviderCoreError(f"unknown idx0 fallback: {fallback}")


def _timing(stabilized_center: np.ndarray, timestamps: np.ndarray, adjacency: np.ndarray):
    return stabilized_focus_velocity(stabilized_center, timestamps, adjacency_mask=adjacency)


def _run_bounds(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = adjacency.shape[0]
    run_start = np.zeros(length, dtype=np.int64)
    run_end = np.full(length, length - 1, dtype=np.int64)
    start = 0
    for index in range(1, length + 1):
        boundary = index == length or not bool(adjacency[index])
        if boundary:
            run_start[start:index] = start
            run_end[start:index] = index - 1
            start = index
    return run_start, run_end


def _persistence(present: np.ndarray, timestamps: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    length = timestamps.shape[0]
    if length == 0:
        return np.zeros(0, dtype=np.float64)
    run_start, run_end = _run_bounds(adjacency)
    left = np.searchsorted(timestamps, timestamps - PERSISTENCE_RADIUS_SEC, side="left")
    right = np.searchsorted(timestamps, timestamps + PERSISTENCE_RADIUS_SEC, side="right")
    result = np.zeros(length, dtype=np.float64)
    for index in range(length):
        lo = max(int(left[index]), int(run_start[index]))
        hi = min(int(right[index]), int(run_end[index]) + 1)
        result[index] = float(present[lo:hi].mean()) if hi > lo else 0.0
    return result


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def build_video_upstream(
    artifacts: VideoArtifacts,
    *,
    idx0_fallback: str = IDX0_FALLBACK_NONE,
) -> VideoUpstream:
    total_started = time.perf_counter()
    if idx0_fallback not in IDX0_FALLBACKS:
        raise ProviderCoreError(f"unknown idx0 fallback: {idx0_fallback}")
    timestamps = np.asarray(artifacts.timestamps, dtype=np.float64)
    frame_ids = np.asarray(artifacts.source_frame_ids, dtype=np.int64)
    adjacency = np.asarray(artifacts.adjacency_mask, dtype=bool)
    visual = np.asarray(artifacts.visual, dtype=np.float16)
    frame_count = timestamps.shape[0]
    if visual.shape != (frame_count, 256):
        raise ProviderCoreError(f"visual must be [T,256], got {visual.shape}")
    if frame_ids.shape != (frame_count,) or adjacency.shape != (frame_count,):
        raise ProviderCoreError("grid arrays must share the frame count")
    if artifacts.width <= 0 or artifacts.height <= 0:
        raise ProviderCoreError("frame dimensions must be positive")

    width = float(artifacts.width)
    height = float(artifacts.height)

    # --- retrieval context (fields 0-4) ---
    retrieval_started = time.perf_counter()
    in_candidate = _candidate_flags(timestamps, artifacts.retrieval.merged_candidates)
    context_index = _context_indices(timestamps, artifacts.retrieval.merged_candidates)
    support_ratio = _support_ratio(timestamps, artifacts.retrieval)
    support_field0 = apply_idx0_fallback(support_ratio, idx0_fallback)
    time_from_start = np.full(frame_count, np.nan, dtype=np.float64)
    time_to_end = np.full(frame_count, np.nan, dtype=np.float64)
    duration = np.full(frame_count, np.nan, dtype=np.float64)
    position = np.full(frame_count, np.nan, dtype=np.float64)
    for frame_index in range(frame_count):
        candidate_index = int(context_index[frame_index])
        if candidate_index < 0:
            continue
        candidate = artifacts.retrieval.merged_candidates[candidate_index]
        span = candidate.end_sec - candidate.start_sec
        if span <= 0:
            continue
        time_from_start[frame_index] = max(0.0, timestamps[frame_index] - candidate.start_sec)
        time_to_end[frame_index] = max(0.0, candidate.end_sec - timestamps[frame_index])
        duration[frame_index] = span
        position[frame_index] = min(1.0, max(0.0, (timestamps[frame_index] - candidate.start_sec) / span))
    retrieval_context_s = time.perf_counter() - retrieval_started

    # --- frozen LOC primary selection per frame ---
    loc_started = time.perf_counter()
    primary_present = np.zeros(frame_count, dtype=bool)
    primary_score = np.full(frame_count, np.nan, dtype=np.float64)
    margin_values = np.full(frame_count, np.nan, dtype=np.float64)
    boxes = np.full((frame_count, 4), np.nan, dtype=np.float64)
    ambiguous_flags = np.zeros(frame_count, dtype=bool)
    statuses: dict[str, int] = {}
    for frame_index in range(frame_count):
        candidates = artifacts.detections.frame_candidates(frame_index)
        decision = select_primary_subject(
            artifacts.ref.canonical_video_id,
            int(frame_ids[frame_index]),
            artifacts.width,
            artifacts.height,
            candidates,
            FROZEN_LOC_POLICY,
        )
        statuses[decision.status] = statuses.get(decision.status, 0) + 1
        ambiguous_flags[frame_index] = decision.ambiguous
        if decision.status == STATUS_PRIMARY and decision.primary is not None:
            box = np.asarray(decision.primary.box, dtype=np.float64)
            box[[0, 2]] = np.clip(box[[0, 2]], 0.0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0.0, height)
            if box[2] > box[0] and box[3] > box[1]:
                boxes[frame_index] = box
                primary_present[frame_index] = True
                primary_score[frame_index] = float(decision.primary.score)
        selection = subject_selection_margin(candidates, detection_floor=DETECTION_FLOOR)
        if selection.available:
            margin_values[frame_index] = float(selection.margin)
    loc_s = time.perf_counter() - loc_started

    native_started = time.perf_counter()
    box_width = boxes[:, 2] - boxes[:, 0]
    box_height = boxes[:, 3] - boxes[:, 1]
    area = box_width * box_height
    area_ratio = np.full(frame_count, np.nan, dtype=np.float64)
    offset = np.full(frame_count, np.nan, dtype=np.float64)
    valid_area = primary_present & (area > 0)
    frame_area = width * height
    area_ratio[valid_area] = area[valid_area] / frame_area
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0 / width
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0 / height
    offset[valid_area] = np.sqrt(
        (centers_x[valid_area] - 0.5) ** 2 + (centers_y[valid_area] - 0.5) ** 2
    )

    # --- subject scale dynamics + bbox continuity (fields 9, 12, 13) ---
    scale_dynamics = np.full(frame_count, np.nan, dtype=np.float64)
    continuity_iou = np.zeros(frame_count, dtype=np.float64)
    continuity_valid = np.zeros(frame_count, dtype=bool)
    for frame_index in range(1, frame_count):
        if not bool(adjacency[frame_index]):
            continue
        if not (valid_area[frame_index] and valid_area[frame_index - 1]):
            continue
        continuity_valid[frame_index] = True
        continuity_iou[frame_index] = _iou_xyxy(boxes[frame_index], boxes[frame_index - 1])
        dt = timestamps[frame_index] - timestamps[frame_index - 1]
        if dt <= 0:
            scale_dynamics[frame_index] = np.nan
        else:
            ratio = area[frame_index] / area[frame_index - 1]
            if ratio > 0:
                scale_dynamics[frame_index] = abs(np.log(ratio)) / dt

    native_geometry_s = time.perf_counter() - native_started

    # --- composition: generic CMP + generic stabilization (fields 11, 14) ---
    cmp_started = time.perf_counter()
    focus_center = np.zeros((frame_count, 2), dtype=np.float64)
    focus_scale = np.ones(frame_count, dtype=np.float64)
    focus_window = np.zeros((frame_count, 4), dtype=np.float64)
    fallback_flags = np.zeros(frame_count, dtype=bool)
    for frame_index in range(frame_count):
        subject_box = None
        if primary_present[frame_index]:
            x1, y1, x2, y2 = boxes[frame_index]
            subject_box = (x1 / width, y1 / height, x2 / width, y2 / height)
        plan = plan_subject_focus(subject_box, margin_fraction=FOCUS_MARGIN_FRACTION)
        if plan.plan_valid:
            focus_center[frame_index] = plan.focus_center
            focus_scale[frame_index] = plan.focus_scale
            focus_window[frame_index] = plan.focus_window
        else:
            fallback_flags[frame_index] = True
            focus_center[frame_index] = (0.5, 0.5)
            focus_scale[frame_index] = 1.0
            focus_window[frame_index] = (0.0, 0.0, 1.0, 1.0)
    cmp_s = time.perf_counter() - cmp_started
    ts_started = time.perf_counter()
    stabilized = stabilize_focus(
        focus_center,
        focus_scale,
        focus_window,
        adjacency_mask=adjacency,
        alpha=STABILIZATION_ALPHA,
    )
    retention = np.clip(
        stabilized.stabilized_focus_window[:, 2] * stabilized.stabilized_focus_window[:, 3],
        0.0,
        1.0,
    )
    velocity = stabilized_focus_velocity(
        stabilized.stabilized_focus_center, timestamps, adjacency_mask=adjacency
    )
    ts_s = time.perf_counter() - ts_started

    # --- persistence (field 15) ---
    native_tail_started = time.perf_counter()
    persistence = _persistence(primary_present, timestamps, adjacency)

    # --- Y target ---
    targets: YouTubeTargets = project_soft_vote_targets(
        frame_ids, artifacts.mturk_clips, in_candidate=in_candidate
    )

    native_signals = {
        "retrieval_support_ratio": support_field0,
        "candidate_time_from_start_log": time_from_start,
        "candidate_time_to_end_log": time_to_end,
        "candidate_duration_log": duration,
        "normalized_candidate_position": position,
        "subject_track_present": primary_present.astype(np.float64),
        "subject_selection_confidence": primary_score,
        "subject_selection_margin": margin_values,
        "bbox_area_ratio": area_ratio,
        "subject_scale_dynamics": scale_dynamics,
        "subject_frame_offset": offset,
        "geometric_context_retention": retention,
        "bbox_continuity_iou": continuity_iou,
        "continuity_valid": continuity_valid.astype(np.float64),
        "stabilized_focus_velocity": velocity,
        "confidence_persistence": persistence,
    }
    if tuple(native_signals) != NATIVE_FIELDS:
        raise ProviderCoreError("native signal assembly diverged from the frozen field order")
    native_s = retrieval_context_s + native_geometry_s + (time.perf_counter() - native_tail_started)

    missing_by_field = {
        name: int((~np.isfinite(values)).sum()) for name, values in native_signals.items()
    }
    metadata = {
        "frame_count": frame_count,
        "primary_frames": int(primary_present.sum()),
        "fallback_plan_frames": int(fallback_flags.sum()),
        "ambiguous_frames": int(ambiguous_flags.sum()),
        "loc_status_counts": statuses,
        "candidate_frames": int(in_candidate.sum()),
        "raw_candidate_count": len(artifacts.retrieval.raw_candidates),
        "merged_candidate_count": len(artifacts.retrieval.merged_candidates),
        "retrieval_support_degenerate_fraction": float((support_ratio == 1.0).mean()),
        "retrieval_support_ratio": {
            "mean": float(support_ratio.mean()) if frame_count else 0.0,
            "fraction_eq_one": float((support_ratio == 1.0).mean()) if frame_count else 0.0,
        },
        "idx0_fallback": idx0_fallback,
        "native_missing_frames_by_field": missing_by_field,
        "target_stats": targets.stats,
        "loc_policy_version": FROZEN_LOC_POLICY.policy_version,
        "cmp_version": "generic-cmp-contract-v1",
        "ts_version": "generic-focus-stabilizer-v1",
        "focus_margin_fraction": FOCUS_MARGIN_FRACTION,
        "stabilization_alpha": STABILIZATION_ALPHA,
        "adjacency_rule": ADJACENCY_RULE,
        "performance_timing": {
            "loc_s": loc_s,
            "cmp_s": cmp_s,
            "ts_s": ts_s,
            "native_s": native_s,
            "provider_total_s": time.perf_counter() - total_started,
        },
        **artifacts.retrieval_metadata,
    }
    return VideoUpstream(
        ref=artifacts.ref,
        timestamps=timestamps,
        source_frame_id=frame_ids,
        adjacency_mask=adjacency,
        visual=visual,
        native_signals=native_signals,
        target=targets.target,
        loss_mask=targets.loss_mask,
        candidate_missed_positive=targets.missed_positive,
        metadata=metadata,
    )


__all__ = [
    "ChunkSpan",
    "DetectionArtifacts",
    "FROZEN_LOC_POLICY",
    "IDX0_FALLBACK_INDICATOR",
    "IDX0_FALLBACK_NONE",
    "IDX0_FALLBACK_NORM",
    "IDX0_FALLBACKS",
    "MergedCandidate",
    "ProviderCoreError",
    "RawCandidate",
    "RetrievalContext",
    "VideoArtifacts",
    "apply_idx0_fallback",
    "build_video_upstream",
]
