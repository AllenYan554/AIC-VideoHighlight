"""Stage 5.5 frame-level prediction calibration (emit/drop selection).

This module changes ONLY which candidate frames that already own a frozen
Stage 5.4 TS-5 Revised bbox are emitted.  It never changes frame identity,
bbox geometry, crop geometry, subject, segment, the TS-5 trajectory or any
temporal smoothing: the frozen trajectory is always computed in full first and
this layer only produces a final emit mask.

Policies (preregistered, three arms only):

* ``FS-0`` ``all_frames_v1``                 -- identity control (keep all).
* ``FS-1`` ``cross_chunk_singleton_edge_prune_v1``
* ``FS-2`` ``cross_chunk_nonunanimous_edge_prune_v1``

Cross-chunk evidence for a candidate frame ``t``:

* ``E_t`` = number of original temporal inference chunks whose observable
  window covers ``t`` (eligible chunks).
* ``S_t`` = number of those eligible chunks that actually produced a raw
  temporal candidate covering ``t`` (supporting chunks).

All temporal-segment -> frame projections reuse the frozen Stage 5.1 canonical
mapping (:func:`aic_video_highlight.spatial_composition.frame_projection`
``frames_in_segment``); this module never re-implements fps rounding, start
floor or end ceil rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from aic_video_highlight.spatial_composition.frame_projection import (
    VideoTiming,
    frames_in_segment,
)

FS0 = "FS-0"
FS1 = "FS-1"
FS2 = "FS-2"
POLICIES = (FS0, FS1, FS2)

ARM_NAMES = {
    FS0: "all_frames_v1",
    FS1: "cross_chunk_singleton_edge_prune_v1",
    FS2: "cross_chunk_nonunanimous_edge_prune_v1",
}


class FrameSelectionError(ValueError):
    """Raised when Stage 5.5 temporal-evidence inputs are malformed."""


@dataclass(frozen=True, slots=True)
class ChunkWindow:
    """One original temporal inference chunk's observable time window."""

    chunk_index: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class RawCandidateSpan:
    """One raw candidate's global clip-local time span from one chunk."""

    chunk_index: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class FrameSupport:
    """Cross-chunk evidence for a single candidate frame."""

    frame: int
    eligible_chunks: int
    supporting_chunks: int


@dataclass(frozen=True, slots=True)
class FinalSegment:
    """One frozen Stage 4 final segment (identity plus frozen time span)."""

    segment_id: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class SegmentSelection:
    """Per-segment emit/drop projection with auditable evidence."""

    segment_id: str
    frames: tuple[int, ...]
    kept: tuple[int, ...]
    dropped: tuple[int, ...]
    support: tuple[FrameSupport, ...]
    core_exists: bool
    drop_rule: str


@dataclass(frozen=True, slots=True)
class FrameSelection:
    """Whole-video emit/drop projection across all final segments."""

    policy: str
    arm: str
    candidate_frames: tuple[int, ...]
    emitted_frames: tuple[int, ...]
    dropped_frames: tuple[int, ...]
    segments: tuple[SegmentSelection, ...]
    support_by_frame: tuple[FrameSupport, ...]


def _validate_spans(chunk_windows: Sequence[ChunkWindow], raw_spans: Sequence[RawCandidateSpan]) -> None:
    seen: set[int] = set()
    for window in chunk_windows:
        if window.chunk_index in seen:
            raise FrameSelectionError(f"duplicate chunk_index: {window.chunk_index}")
        seen.add(window.chunk_index)
        if window.end_sec <= window.start_sec:
            raise FrameSelectionError(f"invalid chunk window {window.chunk_index}")
    known = {window.chunk_index for window in chunk_windows}
    for span in raw_spans:
        if span.chunk_index not in known:
            raise FrameSelectionError(f"raw span references unknown chunk {span.chunk_index}")
        if span.end_sec <= span.start_sec:
            raise FrameSelectionError("raw span must have positive extent")


def _project_frames(
    start_sec: float, end_sec: float, timing: VideoTiming
) -> frozenset[int]:
    return frozenset(frames_in_segment(start_sec, end_sec, timing))


def compute_support_by_frame(
    candidate_frames: Iterable[int],
    timing: VideoTiming,
    chunk_windows: Sequence[ChunkWindow],
    raw_spans: Sequence[RawCandidateSpan],
) -> dict[int, FrameSupport]:
    """Compute exact ``E_t`` / ``S_t`` for each candidate frame via Stage 5.1."""
    _validate_spans(chunk_windows, raw_spans)
    eligible_frames_by_chunk = {
        window.chunk_index: _project_frames(window.start_sec, window.end_sec, timing)
        for window in chunk_windows
    }
    supported_frames_by_chunk: dict[int, set[int]] = {window.chunk_index: set() for window in chunk_windows}
    for span in raw_spans:
        supported_frames_by_chunk[span.chunk_index] |= set(
            _project_frames(span.start_sec, span.end_sec, timing)
        )

    ordered = sorted(set(int(frame) for frame in candidate_frames))
    profile: dict[int, FrameSupport] = {}
    for frame in ordered:
        eligible = 0
        supporting = 0
        for chunk_index, eligible_frames in eligible_frames_by_chunk.items():
            if frame in eligible_frames:
                eligible += 1
                if frame in supported_frames_by_chunk[chunk_index]:
                    supporting += 1
        profile[frame] = FrameSupport(frame, eligible, supporting)
    return profile


def _edge_connected_runs(flags: Sequence[bool]) -> list[tuple[int, int]]:
    """Return half-open index ranges of maximal runs where ``flags`` is True."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def _edge_runs(flags: Sequence[bool]) -> list[tuple[int, int]]:
    """Maximal True runs that touch the segment's left or right edge only."""
    length = len(flags)
    return [
        (start, end)
        for start, end in _edge_connected_runs(flags)
        if start == 0 or end == length
    ]


def select_segment_frames(
    policy: str,
    frames: Sequence[int],
    support: dict[int, FrameSupport],
) -> SegmentSelection:
    """Apply one preregistered policy to a single contiguous final segment."""
    if policy not in POLICIES:
        raise FrameSelectionError(f"unknown Stage 5.5 policy: {policy}")
    ordered = tuple(sorted(int(frame) for frame in frames))
    supports = tuple(support[frame] for frame in ordered)
    if policy == FS0:
        return SegmentSelection(
            segment_id="",
            frames=ordered,
            kept=ordered,
            dropped=(),
            support=supports,
            core_exists=True,
            drop_rule="fs0.identity_keep",
        )

    if policy == FS1:
        core_exists = any(item.supporting_chunks >= 2 for item in supports)
        droppable = [
            item.eligible_chunks >= 2 and item.supporting_chunks == 1 for item in supports
        ]
        rule = "fs1.edge_singleton_support_prune"
    else:
        core_exists = any(
            item.eligible_chunks >= 2 and item.supporting_chunks == item.eligible_chunks
            for item in supports
        )
        droppable = [
            item.eligible_chunks >= 2 and item.supporting_chunks < item.eligible_chunks
            for item in supports
        ]
        rule = "fs2.edge_nonunanimous_support_prune"

    if not core_exists:
        return SegmentSelection(
            segment_id="",
            frames=ordered,
            kept=ordered,
            dropped=(),
            support=supports,
            core_exists=False,
            drop_rule="keep_all.no_consensus_core",
        )

    drop_indices: set[int] = set()
    for start, end in _edge_runs(droppable):
        drop_indices.update(range(start, end))
    kept = tuple(frame for index, frame in enumerate(ordered) if index not in drop_indices)
    dropped = tuple(frame for index, frame in enumerate(ordered) if index in drop_indices)
    return SegmentSelection(
        segment_id="",
        frames=ordered,
        kept=kept,
        dropped=dropped,
        support=supports,
        core_exists=True,
        drop_rule=rule if dropped else "keep_all.no_edge_run",
    )


def select_emit_frames(
    policy: str,
    segments: Sequence[FinalSegment],
    timing: VideoTiming,
    chunk_windows: Sequence[ChunkWindow],
    raw_spans: Sequence[RawCandidateSpan],
) -> FrameSelection:
    """Project every final segment to frames and emit a whole-video keep mask.

    A candidate frame is emitted when at least one final segment that contains
    it keeps it.  This is deliberately conservative for recall: overlapping
    segments never let one segment's edge decision evict another's core, and no
    video can be emptied by a single over-eager edge run.
    """
    if policy not in POLICIES:
        raise FrameSelectionError(f"unknown Stage 5.5 policy: {policy}")

    segment_frames: list[tuple[FinalSegment, tuple[int, ...]]] = []
    candidate_set: set[int] = set()
    for segment in segments:
        frames = tuple(
            frames_in_segment(segment.start_sec, segment.end_sec, timing)
        )
        segment_frames.append((segment, frames))
        candidate_set.update(frames)

    support = compute_support_by_frame(
        candidate_set, timing, chunk_windows, raw_spans
    )

    selections: list[SegmentSelection] = []
    emitted: set[int] = set()
    for segment, frames in segment_frames:
        selection = select_segment_frames(policy, frames, support)
        selections.append(
            SegmentSelection(
                segment_id=segment.segment_id,
                frames=selection.frames,
                kept=selection.kept,
                dropped=selection.dropped,
                support=selection.support,
                core_exists=selection.core_exists,
                drop_rule=selection.drop_rule,
            )
        )
        emitted.update(selection.kept)

    ordered_candidates = tuple(sorted(candidate_set))
    ordered_emitted = tuple(sorted(emitted))
    ordered_dropped = tuple(frame for frame in ordered_candidates if frame not in emitted)
    return FrameSelection(
        policy=policy,
        arm=ARM_NAMES[policy],
        candidate_frames=ordered_candidates,
        emitted_frames=ordered_emitted,
        dropped_frames=ordered_dropped,
        segments=tuple(selections),
        support_by_frame=tuple(support[frame] for frame in ordered_candidates),
    )
