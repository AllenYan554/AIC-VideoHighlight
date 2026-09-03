"""Deterministic Temporal-IoU based candidate deduplication."""

from __future__ import annotations

from collections.abc import Iterable

from .metrics import temporal_iou
from .models import HighlightSegment


def _merge_pair(left: HighlightSegment, right: HighlightSegment) -> HighlightSegment:
    reasons = list(dict.fromkeys(reason for reason in (left.reason, right.reason) if reason))
    source_chunk = left.source_chunk if left.source_chunk == right.source_chunk else None
    return HighlightSegment(
        start_sec=min(left.start_sec, right.start_sec),
        end_sec=max(left.end_sec, right.end_sec),
        score=max(left.score, right.score),
        reason=" | ".join(reasons),
        source_chunk=source_chunk,
    )


def merge_segments(
    segments: Iterable[HighlightSegment],
    *,
    tiou_threshold: float = 0.5,
) -> list[HighlightSegment]:
    """Merge sorted candidates when their Temporal IoU meets the threshold.

    The merged interval uses the temporal union, the maximum confidence score,
    stable de-duplicated reasons, and clears ``source_chunk`` when sources differ.
    """
    if not 0 <= tiou_threshold <= 1:
        raise ValueError("tiou_threshold must be in [0, 1]")

    ordered = sorted(segments, key=lambda item: (item.start_sec, item.end_sec, -item.score))
    merged: list[HighlightSegment] = []
    for candidate in ordered:
        if not merged or temporal_iou(merged[-1], candidate) < tiou_threshold:
            merged.append(candidate)
        else:
            merged[-1] = _merge_pair(merged[-1], candidate)
    return merged
