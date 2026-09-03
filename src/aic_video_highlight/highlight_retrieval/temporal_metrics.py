"""Highlight retrieval duration metrics; these are not official competition metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol


class TemporalInterval(Protocol):
    start_sec: float
    end_sec: float


def _bounds(interval: TemporalInterval | Sequence[float]) -> tuple[float, float]:
    if hasattr(interval, "start_sec") and hasattr(interval, "end_sec"):
        start_sec = float(interval.start_sec)
        end_sec = float(interval.end_sec)
    else:
        if len(interval) != 2:
            raise ValueError("an interval sequence must contain exactly two values")
        start_sec, end_sec = map(float, interval)
    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError("intervals require 0 <= start_sec < end_sec")
    return start_sec, end_sec


def temporal_intersection(
    left: TemporalInterval | Sequence[float],
    right: TemporalInterval | Sequence[float],
) -> float:
    left_start, left_end = _bounds(left)
    right_start, right_end = _bounds(right)
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def temporal_union(
    left: TemporalInterval | Sequence[float],
    right: TemporalInterval | Sequence[float],
) -> float:
    left_start, left_end = _bounds(left)
    right_start, right_end = _bounds(right)
    return (left_end - left_start) + (right_end - right_start) - temporal_intersection(left, right)


def temporal_iou(
    left: TemporalInterval | Sequence[float],
    right: TemporalInterval | Sequence[float],
) -> float:
    union_sec = temporal_union(left, right)
    return temporal_intersection(left, right) / union_sec


def _merge_interval_union(
    intervals: Iterable[TemporalInterval | Sequence[float]],
) -> list[tuple[float, float]]:
    ordered = sorted(_bounds(interval) for interval in intervals)
    merged: list[tuple[float, float]] = []
    for start_sec, end_sec in ordered:
        if not merged or start_sec > merged[-1][1]:
            merged.append((start_sec, end_sec))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_sec))
    return merged


def _total_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end_sec - start_sec for start_sec, end_sec in intervals)


def _set_intersection_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    total = 0.0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def duration_based_metrics(
    predicted: Iterable[TemporalInterval | Sequence[float]],
    reference_segments: Iterable[TemporalInterval | Sequence[float]],
) -> dict[str, float]:
    """Compute union-aware duration precision, recall, F1, and Temporal IoU."""
    predicted_union = _merge_interval_union(predicted)
    reference_union = _merge_interval_union(reference_segments)
    predicted_sec = _total_duration(predicted_union)
    reference_sec = _total_duration(reference_union)
    intersection_sec = _set_intersection_duration(predicted_union, reference_union)
    union_sec = predicted_sec + reference_sec - intersection_sec

    precision = intersection_sec / predicted_sec if predicted_sec else float(not reference_sec)
    recall = intersection_sec / reference_sec if reference_sec else float(not predicted_sec)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tiou = intersection_sec / union_sec if union_sec else 1.0
    return {
        "intersection_sec": intersection_sec,
        "union_sec": union_sec,
        "temporal_iou": tiou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
