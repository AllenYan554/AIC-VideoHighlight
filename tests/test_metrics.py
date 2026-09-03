import pytest

from aic_video_highlight.stage1.metrics import (
    duration_based_metrics,
    temporal_intersection,
    temporal_iou,
    temporal_union,
)
from aic_video_highlight.stage1.models import HighlightSegment


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
    ground_truth = [segment(5.0, 15.0)]

    metrics = duration_based_metrics(predicted, ground_truth)

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
