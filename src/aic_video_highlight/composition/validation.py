"""Independent validator for official-format prediction JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    line_number: int
    video_id: str | None
    code: str
    detail: str


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.issues


_STAT_KEYS = (
    "video_count",
    "prediction_count",
    "empty_video_count",
    "skipped_blank_line_count",
    "invalid_frame_count",
    "duplicate_frame_count",
    "invalid_bbox_count",
    "out_of_bounds_count",
    "ratio_violation_count",
    "temporal_traceability_violation_count",
    "missing_video_count",
    "unknown_video_count",
)


def _new_stats() -> dict[str, int]:
    return {key: 0 for key in _STAT_KEYS}


def _metadata_field(metadata: object, name: str) -> int | None:
    if metadata is None:
        return None
    if isinstance(metadata, Mapping):
        value = metadata.get(name)
    else:
        value = getattr(metadata, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _target_ratio_from(
    record: Mapping[str, object],
    video_id: str,
    index: Mapping[str, Sequence[int | float]] | None,
) -> tuple[Fraction, Fraction] | None:
    raw = record.get("targetRatioWH")
    if raw is None and index is not None:
        raw = index.get(video_id)
    if raw is None:
        return None
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 2
    ):
        return None
    try:
        tw = float(raw[0])
        th = float(raw[1])
    except (TypeError, ValueError):
        return None
    if tw <= 0 or th <= 0:
        return None
    return Fraction(str(tw)), Fraction(str(th))


def validate_submission_file(
    path: str | Path,
    *,
    index: Mapping[str, Sequence[int | float]] | None = None,
    metadata: Mapping[str, object] | None = None,
    temporal_frames: Mapping[str, set[int] | frozenset[int]] | None = None,
) -> ValidationReport:
    """Re-check a submission against the official contract.

    All optional inputs (index, metadata, temporal allow-list) are independent
    of the writer; when present they enable the corresponding deeper checks.
    """
    report = ValidationReport(stats=_new_stats())
    stats = report.stats
    seen_video_ids: list[str] = []

    text = Path(path).read_text(encoding="utf-8")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            stats["skipped_blank_line_count"] += 1
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            report.issues.append(
                ValidationIssue(line_number, None, "invalid_json", str(exc))
            )
            continue
        if not isinstance(record, dict):
            report.issues.append(
                ValidationIssue(line_number, None, "invalid_record", "line is not an object")
            )
            continue

        video_id = record.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            report.issues.append(
                ValidationIssue(line_number, None, "missing_video_id", "video_id missing or not a string")
            )
            continue
        seen_video_ids.append(video_id)
        stats["video_count"] += 1

        if index is not None and video_id not in index:
            stats["unknown_video_count"] += 1
            report.issues.append(
                ValidationIssue(line_number, video_id, "unknown_video_id", "not present in the index")
            )

        raw_line_ratio = record.get("targetRatioWH")
        ratio = _target_ratio_from(record, video_id, index)
        if raw_line_ratio is not None and ratio is None:
            report.issues.append(
                ValidationIssue(
                    line_number, video_id, "invalid_target_ratio", repr(raw_line_ratio)
                )
            )
        if ratio is not None and index is not None and video_id in index:
            try:
                index_ratio = _target_ratio_from({}, video_id, index)
            except Exception:
                index_ratio = None
            if index_ratio is not None and index_ratio != ratio:
                report.issues.append(
                    ValidationIssue(line_number, video_id, "ratio_mismatch", "line ratio differs from index")
                )

        video_meta = metadata.get(video_id) if metadata is not None else None
        width = _metadata_field(video_meta, "width")
        height = _metadata_field(video_meta, "height")
        frame_count = _metadata_field(video_meta, "frame_count")

        allowed_frames = (
            temporal_frames.get(video_id) if temporal_frames is not None else None
        )

        predictions = record.get("predictions")
        if not isinstance(predictions, list):
            report.issues.append(
                ValidationIssue(line_number, video_id, "missing_predictions", "predictions missing or not a list")
            )
            continue
        if not predictions:
            stats["empty_video_count"] += 1

        seen_frames: set[int] = set()
        previous_frame: int | None = None
        for prediction in predictions:
            if not isinstance(prediction, dict):
                report.issues.append(
                    ValidationIssue(line_number, video_id, "invalid_prediction", "prediction is not an object")
                )
                continue
            stats["prediction_count"] += 1

            raw_frame = prediction.get("frame")
            if isinstance(raw_frame, bool) or not isinstance(raw_frame, int):
                stats["invalid_frame_count"] += 1
                report.issues.append(
                    ValidationIssue(line_number, video_id, "non_integer_frame", repr(raw_frame))
                )
                frame = None
            elif raw_frame < 0:
                stats["invalid_frame_count"] += 1
                report.issues.append(
                    ValidationIssue(line_number, video_id, "negative_frame", str(raw_frame))
                )
                frame = None
            elif frame_count is not None and raw_frame >= frame_count:
                stats["invalid_frame_count"] += 1
                report.issues.append(
                    ValidationIssue(
                        line_number, video_id, "frame_out_of_range", f"{raw_frame} >= {frame_count}"
                    )
                )
                frame = None
            else:
                frame = raw_frame

            if frame is not None:
                if frame in seen_frames:
                    stats["duplicate_frame_count"] += 1
                    report.issues.append(
                        ValidationIssue(line_number, video_id, "duplicate_frame", str(frame))
                    )
                seen_frames.add(frame)
                if previous_frame is not None and frame < previous_frame:
                    report.issues.append(
                        ValidationIssue(
                            line_number, video_id, "non_ascending_order", f"{frame} after {previous_frame}"
                        )
                    )
                previous_frame = frame
                if allowed_frames is not None and frame not in allowed_frames:
                    stats["temporal_traceability_violation_count"] += 1
                    report.issues.append(
                        ValidationIssue(
                            line_number, video_id, "temporal_traceability_violation", str(frame)
                        )
                    )

            if "bboxes" not in prediction:
                report.issues.append(
                    ValidationIssue(line_number, video_id, "missing_bboxes", "bboxes missing")
                )
                continue
            bboxes = prediction["bboxes"]
            if (
                not isinstance(bboxes, Sequence)
                or isinstance(bboxes, (str, bytes))
                or len(bboxes) != 3
            ):
                stats["invalid_bbox_count"] += 1
                report.issues.append(
                    ValidationIssue(line_number, video_id, "invalid_bbox_shape", repr(bboxes))
                )
                continue
            if any(
                isinstance(value, bool) or not isinstance(value, int) for value in bboxes
            ):
                stats["invalid_bbox_count"] += 1
                report.issues.append(
                    ValidationIssue(line_number, video_id, "non_integer_bbox", repr(bboxes))
                )
                continue
            x, y, w = bboxes
            if x < 0 or y < 0 or w <= 0:
                stats["invalid_bbox_count"] += 1
                report.issues.append(
                    ValidationIssue(line_number, video_id, "invalid_bbox_value", repr(bboxes))
                )
                continue
            if width is not None and x + w > width:
                stats["out_of_bounds_count"] += 1
                report.issues.append(
                    ValidationIssue(
                        line_number, video_id, "bbox_out_of_bounds", f"x+w={x + w} > W={width}"
                    )
                )
            if ratio is not None and height is not None:
                tw_frac, th_frac = ratio
                if Fraction(w) * th_frac > Fraction(height - y) * tw_frac:
                    stats["ratio_violation_count"] += 1
                    report.issues.append(
                        ValidationIssue(
                            line_number,
                            video_id,
                            "ratio_height_violation",
                            f"derived h exceeds H={height} for targetRatioWH",
                        )
                    )

    if index is not None:
        submitted = set(seen_video_ids)
        for video_id in index:
            if video_id not in submitted:
                stats["missing_video_count"] += 1
                report.issues.append(
                    ValidationIssue(0, video_id, "missing_video_line", "no JSONL line for index entry")
                )

    return report
