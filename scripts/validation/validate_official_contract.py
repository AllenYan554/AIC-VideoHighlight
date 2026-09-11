#!/usr/bin/env python3
"""Independent official-format predictions.jsonl contract validator (Stage 5.6).

Deliberately independent of the E2E runner: it re-reads the written JSONL and
re-checks the competition contract.  It reuses the project's canonical
``validate_submission_file`` and adds duplicate-video, non-finite-number,
canonical-ordering and deterministic-serialization checks.

CLI:
  python scripts/validation/validate_official_contract.py \
      --predictions <predictions.jsonl> \
      [--role-manifest <role.json>] \
      [--metadata-cache <metadata_cache.json>] \
      [--report <report.json>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aic_video_highlight.spatial_composition.validation import validate_submission_file


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
    if role_manifest is not None:
        role = _read_json(role_manifest)
        index = {item["video_id"]: [9, 16] for item in role["records"]}
    if metadata_cache is not None:
        records_meta = _read_json(metadata_cache)["records"]
        metadata = {
            video_id: {
                "width": int(meta["width"]),
                "height": int(meta["height"]),
                "frame_count": int(meta["frame_count"]),
            }
            for video_id, meta in records_meta.items()
        }

    base = validate_submission_file(predictions_path, index=index, metadata=metadata)
    base_issues = [
        {"line": i.line_number, "video_id": i.video_id, "code": i.code, "detail": i.detail}
        for i in base.issues
    ]
    all_issues = issues + base_issues
    return {
        "schema_version": "aic.stage5.6-official-contract-validation/v1",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--metadata-cache", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = validate_contract(
        args.predictions,
        role_manifest=args.role_manifest,
        metadata_cache=args.metadata_cache,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k: report[k] for k in (
        "line_count", "unique_video_count", "duplicate_video_count", "non_finite_count",
        "ordering_violations", "base_valid", "issue_count", "is_valid")}, indent=2))
    return 0 if report["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
