import pytest

from aic_video_highlight.highlight_retrieval.schemas import HighlightSegment
from aic_video_highlight.highlight_retrieval.temporal_metrics import (
    duration_based_metrics,
    temporal_intersection,
    temporal_iou,
    temporal_union,
)


def segment(start: float, end: float) -> HighlightSegment:
    return HighlightSegment(start, end, 1.0)


def test_temporal_interval_operations() -> None:
    left = segment(0.0, 10.0)
    right = segment(5.0, 15.0)

    assert temporal_intersection(left, right) == 5.0
    assert temporal_union(left, right) == 15.0
    assert temporal_iou(left, right) == pytest.approx(1 / 3)


def test_duration_metrics_use_union_duration_without_double_counting() -> None:
    predicted = [segment(0.0, 6.0), segment(4.0, 10.0)]
    reference_segments = [segment(5.0, 15.0)]

    metrics = duration_based_metrics(predicted, reference_segments)

    assert metrics["intersection_sec"] == 5.0
    assert metrics["union_sec"] == 15.0
    assert metrics["temporal_iou"] == pytest.approx(1 / 3)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_duration_metrics_handle_both_sets_empty() -> None:
    metrics = duration_based_metrics([], [])

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["temporal_iou"] == 1.0


def test_duration_metrics_exclude_zero_duration_weak_references() -> None:
    metrics = duration_based_metrics(
        [segment(2.0, 4.0)],
        [(0.0, 0.0), (2.0, 5.0)],
        zero_duration_epsilon_sec=1e-3,
    )

    assert metrics["zero_duration_reference_count"] == 1
    assert metrics["reference_sec"] == 3.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(0.8)
