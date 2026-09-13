"""Independent official-format ``predictions.jsonl`` contract validator.

Deliberately independent of the inference runner: it re-reads the written JSONL
and re-checks the competition contract.  It reuses the project's canonical
``validate_submission_file`` and adds duplicate-video, non-finite-number,
canonical-ordering and deterministic-serialization checks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from aic_video_highlight.composition.validation import validate_submission_file


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _canonical_line(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_contract(
    predictions_path: Path,
    *,
    role_manifest: Path | None = None,
    metadata_cache: Path | None = None,
    require_full_index: bool = False,
) -> dict[str, Any]:
    raw_lines = [
        line for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for number, line in enumerate(raw_lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append({"line": number, "code": "invalid_json", "detail": str(exc)})
            continue
        if not isinstance(payload, dict):
            issues.append({"line": number, "code": "invalid_record", "detail": "not an object"})
            continue
        records.append(payload)

    # duplicate video detection (not covered by the base validator)
    seen: dict[str, int] = {}
    duplicate_video_count = 0
    for number, record in enumerate(records, start=1):
        video_id = record.get("video_id")
        if isinstance(video_id, str):
            if video_id in seen:
                duplicate_video_count += 1
                issues.append({"line": number, "code": "duplicate_video", "detail": video_id})
            else:
                seen[video_id] = number

    # non-finite number scan + canonical ordering + deterministic serialization
    non_finite_count = 0
    ordering_violations = 0
    serialization_mismatches = 0
    for number, record in enumerate(records, start=1):
        for value in record.get("targetRatioWH", []) or []:
            if not _is_finite_number(value):
                non_finite_count += 1
                issues.append({"line": number, "code": "non_finite_ratio", "detail": repr(value)})
        frames: list[int] = []
        for prediction in record.get("predictions", []) or []:
            frame = prediction.get("frame")
            if not _is_finite_number(frame):
                non_finite_count += 1
            else:
                frames.append(int(frame))
            for value in prediction.get("bboxes", []) or []:
                if not _is_finite_number(value):
                    non_finite_count += 1
                    issues.append({"line": number, "code": "non_finite_bbox", "detail": repr(value)})
        if frames != sorted(frames):
            ordering_violations += 1
            issues.append({"line": number, "code": "frame_order", "detail": "not ascending"})
        if _canonical_line(record) != json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")):
            serialization_mismatches += 1

    index = None
    metadata = None
    present = set(seen)
    if role_manifest is not None:
        role = _read_json(role_manifest)
        index = {
            item["video_id"]: [9, 16]
            for item in role["records"]
            if require_full_index or item["video_id"] in present
        }
    if metadata_cache is not None:
        records_meta = _read_json(metadata_cache)["records"]
        metadata = {
            video_id: {
                "width": int(meta["width"]),
                "height": int(meta["height"]),
                "frame_count": int(meta["frame_count"]),
            }
            for video_id, meta in records_meta.items()
            if require_full_index or video_id in present
        }

    base = validate_submission_file(predictions_path, index=index, metadata=metadata)
    base_issues = [
        {"line": i.line_number, "video_id": i.video_id, "code": i.code, "detail": i.detail}
        for i in base.issues
    ]
    all_issues = issues + base_issues
    return {
        "schema_version": "aic.vhicraft.contract-validation/v1",
        "predictions_path": str(predictions_path),
        "line_count": len(records),
        "unique_video_count": len(seen),
        "duplicate_video_count": duplicate_video_count,
        "non_finite_count": non_finite_count,
        "ordering_violations": ordering_violations,
        "serialization_mismatches": serialization_mismatches,
        "base_stats": dict(base.stats),
        "base_valid": base.is_valid,
        "issue_count": len(all_issues),
        "issues": all_issues[:100],
        "is_valid": base.is_valid and not issues,
    }
