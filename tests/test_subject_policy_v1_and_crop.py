import pytest

from aic_video_highlight.spatial_localization import (
    REASON_BOX_TOO_SMALL,
    STATUS_PRIMARY,
    SubjectCandidate,
    SubjectPolicyConfig,
    decision_to_record,
    select_primary_subject,
    subject_centered_crop,
)
from aic_video_highlight.spatial_localization.subject_to_crop import (
    STATUS_CONTAINS_SUBJECT,
    STATUS_DEGRADED_MAX_CROP,
)


def candidate(x1, y1, x2, y2, score=0.9, label="person"):
    return SubjectCandidate(box=(float(x1), float(y1), float(x2), float(y2)), score=score, label_id=1, label=label)


def v1_config(**kwargs):
    return SubjectPolicyConfig(person_priority=True, **kwargs)


def v0_config(**kwargs):
    return SubjectPolicyConfig(**kwargs)


def decide_v1(candidates, width=534, height=300, **kwargs):
    return select_primary_subject("v", 3, width, height, tuple(candidates), v1_config(**kwargs))


def test_reliable_person_overrides_higher_score_non_person() -> None:
    candidates = [
        candidate(300, 50, 400, 150, score=0.95, label="potted_plant"),
        candidate(20, 40, 200, 290, score=0.70, label="person"),
    ]
    decision = decide_v1(candidates)

    assert decision.primary is not None
    assert decision.primary.label == "person"
    assert decision.status == STATUS_PRIMARY


def test_multiple_reliable_persons_rank_deterministically() -> None:
    candidates = [
        candidate(0, 0, 100, 100, score=0.8, label="person"),
        candidate(200, 40, 400, 280, score=0.9, label="person"),
        candidate(300, 20, 380, 120, score=0.9, label="person"),
    ]
    decision = decide_v1(candidates)

    assert decision.primary is not None
    assert decision.primary.box == (200.0, 40.0, 400.0, 280.0)

    same_area = decide_v1(
        [
            candidate(10, 10, 60, 60, score=0.9, label="person"),
            candidate(300, 50, 350, 100, score=0.9, label="person"),
        ]
    )

    assert same_area.primary is not None
    assert same_area.primary.box == (10.0, 10.0, 60.0, 60.0)


def test_possible_person_does_not_override_reliable_non_person() -> None:
    candidates = [
        candidate(100, 50, 250, 250, score=0.95, label="car"),
        candidate(10, 10, 90, 110, score=0.45, label="person"),
    ]
    decision = decide_v1(candidates)

    assert decision.primary is not None
    assert decision.primary.label == "car"


def test_no_person_preserves_v0_behavior() -> None:
    candidates = [
        candidate(10, 10, 60, 60, score=0.6, label="chair"),
        candidate(100, 50, 250, 290, score=0.95, label="car"),
    ]
    v1_decision = decide_v1(candidates)
    v0_decision = select_primary_subject(
        "v", 3, 534, 300, tuple(candidates), v0_config()
    )

    assert v1_decision.primary == v0_decision.primary
    assert v1_decision.status == v0_decision.status


def test_invalid_person_candidate_is_ignored() -> None:
    candidates = [
        SubjectCandidate(box=(300.0, 100.0, 100.0, 200.0), score=0.99, label_id=1, label="person"),
        candidate(100, 50, 200, 250, score=0.6, label="chair"),
    ]
    decision = decide_v1(candidates)

    assert decision.primary is not None
    assert decision.primary.label == "chair"
    assert decision.invalid_candidate_count == 1


def test_policy_version_string_reflects_config() -> None:
    assert v0_config().policy_version == "primary_subject_policy_v0"
    assert v1_config().policy_version == "primary_subject_policy_v1_person_priority"


def test_v1_ambiguity_flag_among_persons() -> None:
    decision = decide_v1(
        [
            candidate(10, 10, 110, 110, score=0.9, label="person"),
            candidate(300, 50, 400, 150, score=0.88, label="person"),
        ]
    )

    assert decision.status == STATUS_PRIMARY
    assert decision.ambiguous is True
    assert decision.ambiguous_candidate_count == 1


def test_v1_fallback_taxonomy_unchanged() -> None:
    decision = decide_v1([candidate(10, 10, 12, 12, score=0.95, label="person")])

    assert decision.status != STATUS_PRIMARY
    assert REASON_BOX_TOO_SMALL in decision.fallback_reasons


def test_v1_frame_identity_preserved() -> None:
    decision = decide_v1([candidate(20, 40, 200, 290, score=0.7, label="person")])
    record = decision_to_record(decision)

    assert record["video_id"] == "v"
    assert record["frame"] == 3


def test_crop_transform_contains_subject_with_exact_ratio() -> None:
    crop = subject_centered_crop((100.0, 80.0, 200.0, 220.0), 534, 300, 9, 16)

    assert crop.w * 16 == crop.h * 9
    assert crop.contains_subject
    assert crop.status == STATUS_CONTAINS_SUBJECT
    assert crop.x <= 100 and crop.x + crop.w >= 200
    assert crop.y <= 80 and crop.y + crop.h >= 220
    assert crop.x >= 0 and crop.y >= 0 and crop.x + crop.w <= 534 and crop.y + crop.h <= 300


def test_crop_transform_border_subject_shifts_into_frame() -> None:
    crop = subject_centered_crop((0.0, 250.0, 60.0, 300.0), 534, 300, 9, 16)

    assert crop.x >= 0
    assert crop.y >= 0
    assert crop.x + crop.w <= 534
    assert crop.y + crop.h <= 300
    assert crop.contains_subject


def test_crop_transform_oversized_subject_degrades_to_max_legal() -> None:
    crop = subject_centered_crop((5.0, 5.0, 530.0, 295.0), 534, 300, 9, 16)

    assert crop.degraded
    assert crop.status == STATUS_DEGRADED_MAX_CROP
    assert not crop.contains_subject
    assert crop.w * 16 == crop.h * 9
    assert crop.w <= 534 and crop.h <= 300


def test_crop_transform_is_deterministic() -> None:
    first = subject_centered_crop((100.0, 80.0, 200.0, 220.0), 534, 300, 9, 16)
    second = subject_centered_crop((100.0, 80.0, 200.0, 220.0), 534, 300, 9, 16)

    assert first == second
