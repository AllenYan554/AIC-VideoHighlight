import json
import math

import pytest

from aic_video_highlight.spatial_composition.submission import (
    SubmissionValidationError,
    build_submission_record,
    render_submission_line,
    write_predictions_jsonl,
)
from aic_video_highlight.spatial_composition.validation import (
    validate_submission_file,
)


def frame(frame_id: int, bboxes: tuple[int, int, int] = (0, 0, 100)):
    return {"frame": frame_id, "bboxes": list(bboxes)}


def record(video_id: str = "v0", target=(16, 9), predictions=None):
    return build_submission_record(
        video_id=video_id,
        target_ratio=target,
        predictions=predictions if predictions is not None else [frame(0), frame(1)],
    )


def test_writer_emits_ascending_canonical_line(tmp_path) -> None:
    record_value = record(predictions=[frame(1), frame(2), frame(3)])
    out = tmp_path / "pred.jsonl"

    write_predictions_jsonl([record_value], out)

    line = out.read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert [p["frame"] for p in parsed["predictions"]] == [1, 2, 3]
    assert parsed["video_id"] == "v0"
    assert parsed["targetRatioWH"] == [16, 9]


def test_writer_rejects_duplicate_frames() -> None:
    with pytest.raises(SubmissionValidationError):
        record(predictions=[frame(1), frame(1)])


def test_writer_rejects_nan_bbox() -> None:
    with pytest.raises(SubmissionValidationError):
        record(predictions=[{"frame": 0, "bboxes": [0.0, float("nan"), 100]}])


def test_writer_rejects_out_of_bounds_bbox() -> None:
    with pytest.raises(SubmissionValidationError):
        build_submission_record(
            video_id="v0",
            target_ratio=(16, 9),
            predictions=[frame(0, (1500, 0, 500))],
            frame_size=(1920, 1080),
        )


def test_writer_rejects_ratio_overflow() -> None:
    with pytest.raises(SubmissionValidationError):
        build_submission_record(
            video_id="v0",
            target_ratio=(9, 16),
            predictions=[frame(0, (0, 0, 1920))],
            frame_size=(1920, 1080),
        )


def test_writer_rejects_invalid_frames() -> None:
    with pytest.raises(SubmissionValidationError):
        record(predictions=[frame(-1)])
    with pytest.raises(SubmissionValidationError):
        record(predictions=[{"frame": 1.5, "bboxes": [0, 0, 100]}])
    with pytest.raises(SubmissionValidationError):
        record(predictions=[{"frame": True, "bboxes": [0, 0, 100]}])
    with pytest.raises(SubmissionValidationError):
        build_submission_record(
            video_id="v0",
            target_ratio=(16, 9),
            predictions=[frame(999)],
            frame_count=100,
        )


def test_writer_rejects_non_ascending_predictions() -> None:
    with pytest.raises(SubmissionValidationError):
        record(predictions=[frame(2), frame(1)])


def test_empty_predictions_pass_and_render_official_line(tmp_path) -> None:
    empty_record = record(predictions=[])
    out = tmp_path / "pred.jsonl"

    write_predictions_jsonl([empty_record], out)

    parsed = json.loads(out.read_text(encoding="utf-8").strip())
    assert parsed["predictions"] == []
    assert parsed["video_id"] == "v0"


def test_rendered_line_matches_official_sample_shape() -> None:
    line = render_submission_line(record())

    assert line == (
        '{"video_id": "v0", "targetRatioWH": [16, 9], '
        '"predictions": [{"frame": 0, "bboxes": [0, 0, 100]}, {"frame": 1, "bboxes": [0, 0, 100]}]}'
    )


def test_writer_output_is_byte_deterministic(tmp_path) -> None:
    records = [record("a"), record("b", predictions=[]), record("c")]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    write_predictions_jsonl(records, first)
    write_predictions_jsonl(records, second)

    assert first.read_bytes() == second.read_bytes()


def test_validator_accepts_compliant_submission(tmp_path) -> None:
    records = [
        record("a", predictions=[frame(0, (100, 50, 200)), frame(5, (0, 0, 640))]),
        record("b", target=(9, 16), predictions=[]),
    ]
    path = tmp_path / "pred.jsonl"
    write_predictions_jsonl(records, path)

    report = validate_submission_file(
        path,
        index={"a": (16, 9), "b": (9, 16)},
        metadata={
            "a": {"width": 1920, "height": 1080, "frame_count": 100},
            "b": {"width": 1080, "height": 1920, "frame_count": 100},
        },
        temporal_frames={"a": {0, 5}, "b": set()},
    )

    assert report.is_valid
    assert report.stats["prediction_count"] == 2
    assert report.stats["empty_video_count"] == 1


def test_validator_rejects_duplicate_frames(tmp_path) -> None:
    path = tmp_path / "pred.jsonl"
    path.write_text(
        json.dumps(
            {
                "video_id": "a",
                "targetRatioWH": [16, 9],
                "predictions": [frame(1), frame(1)],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_submission_file(path)

    assert not report.is_valid
    assert report.stats["duplicate_frame_count"] == 1


def test_validator_rejects_nan_and_missing_fields(tmp_path) -> None:
    bad_nan = {"video_id": "a", "predictions": [{"frame": 0, "bboxes": [0, 0, math.nan]}]}
    missing_bboxes = {"video_id": "b", "predictions": [{"frame": 0}]}
    missing_predictions = {"video_id": "c"}
    path = tmp_path / "pred.jsonl"
    path.write_text(
        "\n".join(json.dumps(item) for item in (bad_nan, missing_bboxes, missing_predictions)) + "\n",
        encoding="utf-8",
    )

    report = validate_submission_file(path)

    assert not report.is_valid
    codes = {issue.code for issue in report.issues}
    assert "non_integer_bbox" in codes
    assert "missing_bboxes" in codes
    assert "missing_predictions" in codes


def test_validator_rejects_frame_out_of_bounds(tmp_path) -> None:
    path = tmp_path / "pred.jsonl"
    write_predictions_jsonl(
        [record("a", predictions=[frame(0), frame(99)])], path
    )

    report = validate_submission_file(
        path,
        metadata={"a": {"width": 1920, "height": 1080, "frame_count": 50}},
    )

    assert not report.is_valid
    assert report.stats["invalid_frame_count"] == 1


def test_validator_rejects_out_of_bounds_and_ratio_violation(tmp_path) -> None:
    out_of_bounds = build_submission_record(
        video_id="a",
        target_ratio=(16, 9),
        predictions=[frame(0, (1800, 0, 200))],
    )
    ratio_violation = build_submission_record(
        video_id="b",
        target_ratio=(9, 16),
        predictions=[frame(0, (0, 0, 1920))],
    )
    path = tmp_path / "pred.jsonl"
    write_predictions_jsonl([out_of_bounds, ratio_violation], path)

    report = validate_submission_file(
        path,
        metadata={
            "a": {"width": 1920, "height": 1080, "frame_count": 10},
            "b": {"width": 1920, "height": 1080, "frame_count": 10},
        },
    )

    assert not report.is_valid
    assert report.stats["out_of_bounds_count"] == 1
    assert report.stats["ratio_violation_count"] == 1


def test_validator_rejects_wrong_ratio_shape(tmp_path) -> None:
    path = tmp_path / "pred.jsonl"
    path.write_text(
        json.dumps(
            {
                "video_id": "a",
                "targetRatioWH": [0, 9],
                "predictions": [frame(0)],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_submission_file(path)

    assert not report.is_valid
    assert any(issue.code == "invalid_target_ratio" for issue in report.issues)


def test_validator_rejects_unknown_and_missing_videos(tmp_path) -> None:
    path = tmp_path / "pred.jsonl"
    write_predictions_jsonl([record("a")], path)

    report = validate_submission_file(
        path, index={"a": (16, 9), "b": (16, 9)}
    )

    assert not report.is_valid
    codes = {issue.code for issue in report.issues}
    assert "missing_video_line" in codes


def test_validator_rejects_unknown_video_id(tmp_path) -> None:
    path = tmp_path / "pred.jsonl"
    write_predictions_jsonl([record("ghost")], path)

    report = validate_submission_file(path, index={"a": (16, 9)})

    assert not report.is_valid
    assert any(issue.code == "unknown_video_id" for issue in report.issues)


def test_validator_rejects_frames_outside_temporal_allowlist(tmp_path) -> None:
    path = tmp_path / "pred.jsonl"
    write_predictions_jsonl([record("a", predictions=[frame(0), frame(7)])], path)

    report = validate_submission_file(path, temporal_frames={"a": {0, 1, 2}})

    assert not report.is_valid
    assert report.stats["temporal_traceability_violation_count"] == 1


def test_validator_rejects_float_bbox_and_non_integer_frame(tmp_path) -> None:
    path = tmp_path / "pred.jsonl"
    path.write_text(
        json.dumps(
            {
                "video_id": "a",
                "targetRatioWH": [16, 9],
                "predictions": [
                    {"frame": 0.0, "bboxes": [0.5, 0, 100]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_submission_file(path)

    assert not report.is_valid
    codes = {issue.code for issue in report.issues}
    assert "non_integer_frame" in codes
    assert "non_integer_bbox" in codes


def test_validator_accepts_official_kit_line_shape(tmp_path) -> None:
    official_line = (
        '{"video_id": "0", "targetRatioWH": [16, 9], '
        '"predictions": [{"frame": 10, "bboxes": [120, 80, 40]}, {"frame": 11, "bboxes": [125, 82, 40]}]}'
    )
    path = tmp_path / "pred.jsonl"
    path.write_text(official_line + "\n", encoding="utf-8")

    report = validate_submission_file(path)

    assert report.is_valid
    assert report.stats["prediction_count"] == 2
