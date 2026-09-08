"""Official-format JSONL writer for per-frame crop predictions."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class SubmissionValidationError(ValueError):
    """Raised when a submission record violates the official output contract."""


def _require_plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SubmissionValidationError(f"{name} must be an integer")
    return value


def _validated_target_ratio(
    target_ratio: Sequence[int | float],
) -> tuple[int | float, int | float]:
    if not isinstance(target_ratio, Sequence) or isinstance(
        target_ratio, (str, bytes)
    ):
        raise SubmissionValidationError("target_ratio must be a [w, h] pair")
    if len(target_ratio) != 2:
        raise SubmissionValidationError("target_ratio must be a [w, h] pair")
    tw, th = target_ratio
    for name, value in (("target_w", tw), ("target_h", th)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SubmissionValidationError(f"{name} must be a positive number")
        if value <= 0:
            raise SubmissionValidationError(f"{name} must be a positive number")
    return tw, th


def _ratio_fractions(target_ratio: Sequence[int | float]) -> tuple[Fraction, Fraction]:
    tw, th = _validated_target_ratio(target_ratio)
    return Fraction(str(float(tw))), Fraction(str(float(th)))


def _validated_predictions(
    predictions: Iterable[Mapping[str, object]],
    *,
    target_ratio: Sequence[int | float],
    frame_size: tuple[int, int] | None = None,
    frame_count: int | None = None,
) -> list[dict[str, object]]:
    tw_frac, th_frac = _ratio_fractions(target_ratio)
    width_limit = None if frame_size is None else int(frame_size[0])
    height_limit = None if frame_size is None else int(frame_size[1])
    cleaned: list[dict[str, object]] = []
    previous_frame: int | None = None
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise SubmissionValidationError("each prediction must be an object")
        if "frame" not in prediction:
            raise SubmissionValidationError("prediction is missing frame")
        frame = _require_plain_int(prediction["frame"], "frame")
        if frame < 0:
            raise SubmissionValidationError("frame must be >= 0")
        if frame_count is not None and frame >= frame_count:
            raise SubmissionValidationError("frame exceeds the video frame count")
        if previous_frame is not None:
            if frame == previous_frame:
                raise SubmissionValidationError(
                    f"duplicate frame {frame} in predictions"
                )
            if frame < previous_frame:
                raise SubmissionValidationError(
                    "predictions must be sorted by ascending frame"
                )
        previous_frame = frame
        if "bboxes" not in prediction:
            raise SubmissionValidationError("prediction is missing bboxes")
        bboxes = prediction["bboxes"]
        if (
            not isinstance(bboxes, Sequence)
            or isinstance(bboxes, (str, bytes))
            or len(bboxes) != 3
        ):
            raise SubmissionValidationError("bboxes must be a [x, y, w] triplet")
        values = [_require_plain_int(value, "bbox value") for value in bboxes]
        x, y, w = values
        if x < 0 or y < 0 or w <= 0:
            raise SubmissionValidationError("bboxes must satisfy x >= 0, y >= 0, w > 0")
        if width_limit is not None and x + w > width_limit:
            raise SubmissionValidationError("bbox exceeds the frame width")
        if height_limit is not None:
            if Fraction(w) * th_frac > Fraction(height_limit - y) * tw_frac:
                raise SubmissionValidationError(
                    "derived bbox height exceeds the frame height for targetRatioWH"
                )
        cleaned.append({"frame": frame, "bboxes": [x, y, w]})
    return cleaned


def build_submission_record(
    *,
    video_id: str,
    target_ratio: Sequence[int | float],
    predictions: Iterable[Mapping[str, object]],
    frame_size: tuple[int, int] | None = None,
    frame_count: int | None = None,
) -> dict[str, object]:
    if not isinstance(video_id, str) or not video_id:
        raise SubmissionValidationError("video_id must be a non-empty string")
    tw, th = _validated_target_ratio(target_ratio)
    cleaned = _validated_predictions(
        predictions, target_ratio=(tw, th), frame_size=frame_size, frame_count=frame_count
    )
    return {
        "video_id": video_id,
        "targetRatioWH": [tw, th],
        "predictions": cleaned,
    }


def render_submission_line(record: Mapping[str, object]) -> str:
    return json.dumps(record, ensure_ascii=False)


def write_predictions_jsonl(
    records: Iterable[Mapping[str, object]], path: str | Path
) -> int:
    """Write one canonical JSON line per record with LF newlines."""
    target = Path(path)
    count = 0
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(render_submission_line(record))
            handle.write("\n")
            count += 1
    return count
