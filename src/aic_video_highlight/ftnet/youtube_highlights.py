"""YouTube Highlights MTurk annotation adapter for Stage 7 FTNet targets.

Frozen protocol: Y_TARGET_ADAPTER_PROTOCOL_V1 (SOFT_VOTE_TARGET).

    clip unit        = [start_frame, end_frame)  (source frame indices, half-open)
    containing clips = {clip : start_frame <= source_frame_id < end_frame}
    raw target       = mean(soft_vote(clip) for clip in containing_clips) / 5.0
    final y_t        = clip(raw target, 0, 1)
    no-clip coverage -> loss_mask = 0 (unknown), never forced to y=0
    candidate 外 human positive -> UPSTREAM_MISSED_POSITIVE (audit only)
    candidate 内 non-highlight -> ordinary negative

Only ``mturk_label.json`` votes are authoritative.  ``match_label.json`` is a
harvested signal and is never read by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .highlight_targets import HighlightTargetStrategy

ANNOTATOR_COUNT = 5
Y_RANGE = (0.0, 1.0)

# Audit-only: a clip-covered frame counts as a human positive for the
# UPSTREAM_MISSED_POSITIVE report when its projected soft vote is at least
# one half of the annotator mass.  This flag never enters loss_mask or targets.
MISSED_POSITIVE_THRESHOLD = 0.5

MTURK_LABEL_NAME = "mturk_label.json"
CLIP_NAME = "clip.json"


class YouTubeHighlightsAnnotationError(RuntimeError):
    """Raised when authoritative MTurk annotations are missing or malformed."""


@dataclass(frozen=True)
class MturkClip:
    start_frame: int
    end_frame: int
    vote: float

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise YouTubeHighlightsAnnotationError(
                f"invalid clip interval [{self.start_frame}, {self.end_frame})"
            )
        if not math.isfinite(self.vote) or self.vote < 0.0 or self.vote > ANNOTATOR_COUNT:
            raise YouTubeHighlightsAnnotationError(f"invalid soft vote: {self.vote}")


@dataclass(frozen=True)
class YouTubeTargets:
    target: np.ndarray
    loss_mask: np.ndarray
    covered: np.ndarray
    missed_positive: np.ndarray
    stats: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_intervals(value: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not value:
        raise YouTubeHighlightsAnnotationError("clip windows must be a non-empty list")
    windows: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise YouTubeHighlightsAnnotationError(f"clip window must be [start, end]: {item!r}")
        start, end = item
        if isinstance(start, bool) or isinstance(end, bool):
            raise YouTubeHighlightsAnnotationError("clip bounds must be numeric frame indices")
        start_value = float(start)
        end_value = float(end)
        if not start_value.is_integer() or not end_value.is_integer():
            raise YouTubeHighlightsAnnotationError("clip bounds must be integer frame indices")
        windows.append((int(start_value), int(end_value)))
    return tuple(windows)


def _parse_votes(value: Any, expected: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != expected:
        raise YouTubeHighlightsAnnotationError(
            f"vote list must align with {expected} clip windows"
        )
    votes: list[float] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise YouTubeHighlightsAnnotationError(f"vote must be numeric: {raw!r}")
        number = float(raw)
        if not math.isfinite(number) or number < 0.0 or number > ANNOTATOR_COUNT:
            raise YouTubeHighlightsAnnotationError(f"vote out of range [0, {ANNOTATOR_COUNT}]: {raw!r}")
        votes.append(number)
    return tuple(votes)


def load_mturk_clips(annotation_dir: str | Path) -> tuple[MturkClip, ...]:
    """Load and cross-validate ``mturk_label.json`` against ``clip.json``."""

    directory = Path(annotation_dir).expanduser().resolve()
    mturk_path = directory / MTURK_LABEL_NAME
    if not mturk_path.is_file():
        raise YouTubeHighlightsAnnotationError(f"missing authoritative annotation: {mturk_path}")
    payload = json.loads(mturk_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 2:
        raise YouTubeHighlightsAnnotationError("mturk_label.json must be [clips, votes]")
    windows = _parse_intervals(payload[0])
    votes = _parse_votes(payload[1], len(windows))
    clips = tuple(
        MturkClip(start_frame=start, end_frame=end, vote=vote)
        for (start, end), vote in zip(windows, votes)
    )

    clip_path = directory / CLIP_NAME
    if clip_path.is_file():
        clip_payload = json.loads(clip_path.read_text(encoding="utf-8"))
        clip_windows = _parse_intervals(clip_payload)
        if clip_windows != windows:
            raise YouTubeHighlightsAnnotationError(
                "mturk_label.json windows do not match clip.json; refusing ambiguous annotations"
            )
    return clips


def annotation_hashes(annotation_dir: str | Path) -> dict[str, str]:
    directory = Path(annotation_dir).expanduser().resolve()
    hashes: dict[str, str] = {}
    for name in (MTURK_LABEL_NAME, CLIP_NAME):
        path = directory / name
        if path.is_file():
            hashes[name] = _sha256_file(path)
    return hashes


def project_soft_vote_targets(
    source_frame_ids: np.ndarray,
    clips: tuple[MturkClip, ...],
    *,
    in_candidate: np.ndarray,
    strategy: HighlightTargetStrategy = HighlightTargetStrategy.SOFT_VOTE_TARGET,
) -> YouTubeTargets:
    """Project clip-level soft votes onto sampled source frames."""

    if strategy is not HighlightTargetStrategy.SOFT_VOTE_TARGET:
        raise YouTubeHighlightsAnnotationError(
            "Stage 7 YouTube Highlights targets are frozen to SOFT_VOTE_TARGET"
        )
    frame_ids = np.asarray(source_frame_ids, dtype=np.int64)
    if frame_ids.ndim != 1:
        raise YouTubeHighlightsAnnotationError("source_frame_ids must be [T]")
    candidate_flags = np.asarray(in_candidate, dtype=bool)
    if candidate_flags.shape != frame_ids.shape:
        raise YouTubeHighlightsAnnotationError("in_candidate must have shape [T]")

    frame_count = int(frame_ids.shape[0])
    vote_sum = np.zeros(frame_count, dtype=np.float64)
    vote_count = np.zeros(frame_count, dtype=np.int64)
    starts = np.array([clip.start_frame for clip in clips], dtype=np.int64)
    ends = np.array([clip.end_frame for clip in clips], dtype=np.int64)
    votes = np.array([clip.vote for clip in clips], dtype=np.float64)
    for clip_index in range(len(clips)):
        member = (frame_ids >= starts[clip_index]) & (frame_ids < ends[clip_index])
        vote_sum[member] += votes[clip_index]
        vote_count[member] += 1
    covered = vote_count > 0
    raw_target = np.zeros(frame_count, dtype=np.float64)
    raw_target[covered] = (vote_sum[covered] / vote_count[covered]) / float(ANNOTATOR_COUNT)
    target = np.clip(raw_target, Y_RANGE[0], Y_RANGE[1]).astype(np.float32)
    loss_mask = candidate_flags & covered
    missed_positive = (~candidate_flags) & covered & (raw_target >= MISSED_POSITIVE_THRESHOLD)
    stats = {
        "clips": len(clips),
        "frames": frame_count,
        "candidate_frames": int(candidate_flags.sum()),
        "covered_frames": int(covered.sum()),
        "supervised_frames": int(loss_mask.sum()),
        "unknown_frames": int((~covered).sum()),
        "supervised_positives": int(((target >= 0.5) & loss_mask).sum()),
        "missed_positive_frames": int(missed_positive.sum()),
        "annotation_coverage_fraction": float(covered.mean()) if frame_count else 0.0,
        "strategy": strategy.value,
        "annotator_count": ANNOTATOR_COUNT,
        "missed_positive_threshold": MISSED_POSITIVE_THRESHOLD,
    }
    return YouTubeTargets(
        target=target,
        loss_mask=loss_mask,
        covered=covered,
        missed_positive=missed_positive,
        stats=stats,
    )


__all__ = [
    "ANNOTATOR_COUNT",
    "CLIP_NAME",
    "MISSED_POSITIVE_THRESHOLD",
    "MTURK_LABEL_NAME",
    "MturkClip",
    "YouTubeHighlightsAnnotationError",
    "YouTubeTargets",
    "annotation_hashes",
    "load_mturk_clips",
    "project_soft_vote_targets",
]
