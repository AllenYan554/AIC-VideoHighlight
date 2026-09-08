"""Official-format emission and independent validation for CMP-0 / CMP-1 crops."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from aic_video_highlight.spatial_composition.submission import (
    build_submission_record,
    write_predictions_jsonl,
)
from aic_video_highlight.spatial_composition.validation import validate_submission_file


def build_official_records(
    records: Sequence[Mapping[str, Any]],
    config_key: str,
    target_ratio: Sequence[int | float],
) -> list[dict[str, Any]]:
    """Group composed frames into official per-video prediction lines (ascending)."""
    by_video: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_video.setdefault(record["video_id"], []).append(record)
    lines: list[dict[str, Any]] = []
    for video_id in sorted(by_video):
        predictions = [
            {"frame": int(record["frame"]), "bboxes": [int(record[config_key]["x"]), int(record[config_key]["y"]), int(record[config_key]["w"])]}
            for record in sorted(by_video[video_id], key=lambda item: int(item["frame"]))
        ]
        lines.append(
            build_submission_record(
                video_id=video_id,
                target_ratio=target_ratio,
                predictions=predictions,
                frame_size=None,
                frame_count=None,
            )
        )
    return lines


def write_and_validate_official(
    records: Sequence[Mapping[str, Any]],
    config_key: str,
    target_ratio: Sequence[int | float],
    output_path: Path,
    metadata: Mapping[str, Mapping[str, Any]],
    index: Mapping[str, Mapping[str, Any]],
    temporal_frames: Mapping[str, set[int]],
) -> dict[str, Any]:
    """Write an official-format JSONL and re-validate it with the independent validator.

    The completeness check ("every index entry has a line") is scoped to the videos
    actually emitted, so a subset run (e.g. the smoke) is validated against its own
    video set rather than the full 166-video dev index.
    """
    lines = build_official_records(records, config_key, target_ratio)
    write_predictions_jsonl(lines, output_path)
    emitted = {video_id for video_id in (line["video_id"] for line in lines)}
    validator_index = {
        video_id: record["targetRatioWH"]
        for video_id, record in index.items()
        if video_id in emitted
    }
    validator_metadata = {
        video_id: {
            "width": int(meta["width"]),
            "height": int(meta["height"]),
            "frame_count": int(meta["frame_count"]),
        }
        for video_id, meta in metadata.items()
        if video_id in index
    }
    report = validate_submission_file(
        output_path,
        index=validator_index,
        metadata=validator_metadata,
        temporal_frames=temporal_frames,
    )
    return {
        "lines": len(lines),
        "predictions": report.stats["prediction_count"],
        "is_valid": report.is_valid,
        "issue_count": len(report.issues),
        "issues": [
            {"line": issue.line_number, "video_id": issue.video_id, "code": issue.code, "detail": issue.detail}
            for issue in report.issues[:50]
        ],
        "stats": dict(report.stats),
    }
