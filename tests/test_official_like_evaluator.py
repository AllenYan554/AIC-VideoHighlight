import pytest

from aic_video_highlight.evaluation.official_like import (
    evaluate_official_like,
    spatial_iou,
)


def pred(video_id, frames):
    return {"video_id": video_id, "targetRatioWH": [1, 1], "predictions": frames}


def ref(video_id, rois):
    return {"video_id": video_id, "targetRatioWH": [1, 1], "rois": rois}


def metrics(predictions, references):
    return evaluate_official_like(predictions, references)["per_video"][0]


def test_full_match_and_height_derivation():
    row = metrics(
        [pred("v", [{"frame": 1, "bboxes": [0, 0, 10]}])],
        [pred("v", [{"frame": 1, "bboxes": [0, 0, 10]}])],
    )
    assert row["precision"] == row["recall"] == row["f_score"] == 1.0


def test_height_derived_from_target_ratio_wh():
    row = metrics(
        [
            {
                "video_id": "v",
                "targetRatioWH": [9, 16],
                "predictions": [{"frame": 1, "bboxes": [0, 0, 45]}],
            }
        ],
        [ref("v", {"1": [0, 0, 45, 80]})],
    )
    assert row["precision"] == row["recall"] == row["f_score"] == 1.0


def test_missing_gt_frame_penalizes_recall():
    row = metrics(
        [pred("v", [{"frame": 1, "bboxes": [0, 0, 10]}])],
        [ref("v", {"1": [0, 0, 10, 10], "2": [0, 0, 10, 10]})],
    )
    assert row["precision"] == 1.0
    assert row["recall"] == 0.5
    assert row["f_score"] == pytest.approx(2 / 3)


def test_extra_prediction_frame_penalizes_precision_and_no_nearest_match():
    row = metrics(
        [pred("v", [{"frame": 1, "bboxes": [0, 0, 10]}, {"frame": 3, "bboxes": [0, 0, 10]}])],
        [ref("v", {"1": [0, 0, 10, 10], "2": [0, 0, 10, 10]})],
    )
    assert row["exact_common_frames"] == 1
    assert row["precision"] == row["recall"] == row["f_score"] == 0.5


def test_partial_bbox_iou():
    assert spatial_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)
    row = metrics(
        [pred("v", [{"frame": 1, "bboxes": [0, 0, 10]}])],
        [ref("v", {"1": [5, 0, 10, 10]})],
    )
    assert row["precision"] == row["recall"] == row["f_score"] == pytest.approx(1 / 3)


def test_empty_prediction_scores_zero():
    row = metrics([pred("v", [])], [ref("v", {"1": [0, 0, 10, 10]})])
    assert row["precision"] == row["recall"] == row["f_score"] == 0.0


def test_empty_reference_scores_zero():
    row = metrics(
        [pred("v", [{"frame": 1, "bboxes": [0, 0, 10]}])],
        [ref("v", {})],
    )
    assert row["precision"] == row["recall"] == row["f_score"] == 0.0


def test_both_empty_uses_official_per_video_f_score_contract():
    row = metrics([pred("v", [])], [ref("v", {})])
    assert row["prediction_frames"] == row["reference_frames"] == 0
    assert row["precision"] == row["recall"] == 0.0
    assert row["f_score"] == 1.0


def test_macro_mean_over_union_of_exact_video_ids():
    report = evaluate_official_like(
        [
            pred("matched", [{"frame": 1, "bboxes": [0, 0, 10]}]),
            pred("pred-only", [{"frame": 1, "bboxes": [0, 0, 10]}]),
        ],
        [ref("matched", {"1": [0, 0, 10, 10]}), ref("gt-only", {"1": [0, 0, 10, 10]})],
    )
    assert report["video_count"] == 3
    assert report["Official-like Weak-Reference F-score"] == pytest.approx(1 / 3)
    assert report["score_identity"] == "NOT_OFFICIAL_SCORE"
