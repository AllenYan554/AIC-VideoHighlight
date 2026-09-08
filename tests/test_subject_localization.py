import pytest

from aic_video_highlight.spatial_localization import (
    REASON_BOX_TOO_LARGE,
    REASON_BOX_TOO_SMALL,
    REASON_FULL_FRAME_LIKE,
    REASON_INVALID_GEOMETRY,
    REASON_LOW_CONFIDENCE,
    REASON_NO_DETECTION,
    STATUS_CENTER_CROP_FALLBACK,
    STATUS_PRIMARY,
    SubjectCandidate,
    SubjectPolicyConfig,
    detection_iou,
    decision_to_record,
    select_primary_subject,
)


def candidate(x1, y1, x2, y2, score=0.9, label="person"):
    return SubjectCandidate(box=(float(x1), float(y1), float(x2), float(y2)), score=score, label_id=1, label=label)


def decide(candidates, width=534, height=300, **kwargs):
    config = SubjectPolicyConfig(**kwargs)
    return select_primary_subject("v0", 7, width, height, tuple(candidates), config)


def test_no_detection_falls_back_to_center_crop() -> None:
    decision = decide([])

    assert decision.status == STATUS_CENTER_CROP_FALLBACK
    assert REASON_NO_DETECTION in decision.fallback_reasons
    assert decision.primary is None
    assert decision.fallback_box == (183, 0, 168)


def test_low_confidence_detection_falls_back() -> None:
    decision = decide([candidate(100, 50, 200, 250, score=0.4)])

    assert decision.status == STATUS_CENTER_CROP_FALLBACK
    assert REASON_LOW_CONFIDENCE in decision.fallback_reasons


def test_single_reliable_detection_is_primary() -> None:
    decision = decide([candidate(100, 50, 200, 250)])

    assert decision.status == STATUS_PRIMARY
    assert decision.fallback_reasons == ()
    assert decision.primary is not None
    assert decision.primary.box == (100.0, 50.0, 200.0, 250.0)
    assert decision.fallback_box is None


def test_multiple_detections_select_highest_score() -> None:
    decision = decide(
        [
            candidate(10, 10, 60, 60, score=0.6),
            candidate(100, 50, 250, 290, score=0.95),
            candidate(200, 20, 300, 120, score=0.7),
        ]
    )

    assert decision.primary is not None
    assert decision.primary.box == (100.0, 50.0, 250.0, 290.0)
    assert decision.status == STATUS_PRIMARY


def test_tie_break_prefers_larger_area_then_input_order() -> None:
    tie = decide([candidate(10, 10, 110, 110, score=0.9), candidate(200, 50, 350, 250, score=0.9)])

    assert tie.primary is not None
    assert tie.primary.box == (200.0, 50.0, 350.0, 250.0)

    same_area = decide([candidate(10, 10, 60, 60, score=0.9), candidate(300, 50, 350, 100, score=0.9)])

    assert same_area.primary is not None
    assert same_area.primary.box == (10.0, 10.0, 60.0, 60.0)


def test_multi_subject_ambiguity_is_flagged_without_fallback() -> None:
    decision = decide([candidate(10, 10, 110, 110, score=0.9), candidate(300, 50, 400, 150, score=0.88)])

    assert decision.status == STATUS_PRIMARY
    assert decision.ambiguous is True
    assert decision.ambiguous_candidate_count == 1


def test_similar_boxes_are_not_ambiguous() -> None:
    decision = decide([candidate(10, 10, 110, 110, score=0.9), candidate(12, 12, 112, 112, score=0.88)])

    assert decision.status == STATUS_PRIMARY
    assert decision.ambiguous is False


def test_tiny_subject_falls_back() -> None:
    decision = decide([candidate(10, 10, 12, 12, score=0.95)])

    assert decision.status == STATUS_CENTER_CROP_FALLBACK
    assert REASON_BOX_TOO_SMALL in decision.fallback_reasons


def test_full_frame_detection_falls_back() -> None:
    full = decide([candidate(2, 2, 533, 299, score=0.95)])

    assert full.status == STATUS_CENTER_CROP_FALLBACK
    assert REASON_FULL_FRAME_LIKE in full.fallback_reasons

    large = decide([candidate(0, 0, 534, 300, score=0.95)])

    assert large.status == STATUS_CENTER_CROP_FALLBACK
    assert REASON_BOX_TOO_LARGE in large.fallback_reasons


def test_invalid_geometry_is_dropped_and_reported() -> None:
    decision = decide(
        [
            SubjectCandidate(box=(200.0, 100.0, 100.0, 200.0), score=0.99, label_id=1, label="ghost"),
            candidate(100, 50, 200, 250, score=0.8),
        ]
    )

    assert decision.invalid_candidate_count == 1
    assert REASON_INVALID_GEOMETRY not in decision.fallback_reasons
    assert decision.status == STATUS_PRIMARY
    assert decision.primary is not None
    assert decision.primary.score == 0.8


def test_nan_detection_is_invalid() -> None:
    decision = decide(
        [SubjectCandidate(box=(float("nan"), 0, 10, 10), score=0.99, label_id=1, label="x")]
    )

    assert decision.status == STATUS_CENTER_CROP_FALLBACK
    assert REASON_INVALID_GEOMETRY in decision.fallback_reasons
    assert REASON_NO_DETECTION in decision.fallback_reasons


def test_iou_computation() -> None:
    a = candidate(0, 0, 100, 100)
    b = candidate(50, 0, 150, 100)

    assert detection_iou(a, b) == pytest.approx(1 / 3)
    assert detection_iou(a, a) == pytest.approx(1.0)


def test_decision_record_preserves_frame_identity() -> None:
    decision = decide([candidate(100, 50, 200, 250)])
    record = decision_to_record(decision)

    assert record["video_id"] == "v0"
    assert record["frame"] == 7
    assert record["primary"]["xywh_int"] == [100, 50, 100]
    assert record["provenance"] == {}

    with_provenance = select_primary_subject(
        "v1",
        3,
        534,
        300,
        (candidate(10, 10, 60, 60),),
        SubjectPolicyConfig(),
        provenance={"model": "test"},
    )
    assert decision_to_record(with_provenance)["provenance"] == {"model": "test"}


def test_policy_config_validation() -> None:
    with pytest.raises(ValueError):
        SubjectPolicyConfig(reliable_score=0.2, possible_score=0.5)
    with pytest.raises(ValueError):
        SubjectPolicyConfig(min_area_fraction=0.0)


def test_determinism_over_repeated_calls() -> None:
    candidates = (
        candidate(10, 10, 60, 60, score=0.6),
        candidate(100, 50, 250, 290, score=0.95),
        candidate(200, 20, 300, 120, score=0.7),
    )
    first = decide(candidates)
    second = decide(candidates)

    assert first == second
