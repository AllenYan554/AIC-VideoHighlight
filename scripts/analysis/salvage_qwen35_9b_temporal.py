#!/usr/bin/env python3
"""Build a non-official temporal-only diagnostic for 9B schema failures.

The script never writes to either baseline output directory. It reads the
persisted raw responses, reuses the project's union-aware duration metrics,
and writes only small research artifacts under ``docs/experiments``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any


SALVAGE_IDS = {"qvh_000008_9x16", "qvh_000574_9x16"}
FULL_VIDEO_COVERAGE_THRESHOLD = 0.999


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _raw_payload(raw_response: str) -> dict[str, Any]:
    # Import the project's existing strict JSON-envelope loader. Deliberately do
    # not call parse_highlight_response: that formal parser must keep rejecting
    # missing score values.
    from aic_video_highlight.highlight_retrieval.response_parser import _load_json_object

    return _load_json_object(raw_response)


def _all_raw_segments(raw_record: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for chunk in raw_record["chunks"]:
        payload = _raw_payload(str(chunk["raw_response"]))
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise ValueError(f"{raw_record['video_id']}: segments is not an array")
        result.extend(segments)
    return result


def _temporal_only_salvage(raw_record: dict[str, Any]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    raw_segments: list[dict[str, Any]] = []
    chunk_facts: list[dict[str, Any]] = []
    for chunk in sorted(raw_record["chunks"], key=lambda item: item["chunk_index"]):
        payload = _raw_payload(str(chunk["raw_response"]))
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise ValueError("segments must be an array")
        chunk_start = float(chunk["chunk_start_sec"])
        chunk_end = float(chunk["chunk_end_sec"])
        chunk_duration = chunk_end - chunk_start
        for index, item in enumerate(segments):
            if not isinstance(item, dict):
                raise ValueError(f"segments[{index}] must be an object")
            start = item.get("start_sec")
            end = item.get("end_sec")
            if isinstance(start, bool) or not isinstance(start, (int, float)):
                raise ValueError(f"segments[{index}].start_sec must be a number")
            if isinstance(end, bool) or not isinstance(end, (int, float)):
                raise ValueError(f"segments[{index}].end_sec must be a number")
            start = float(start)
            end = float(end)
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError(f"segments[{index}] bounds must be finite")
            if start < 0 or end <= start or start >= chunk_duration:
                raise ValueError(f"segments[{index}] has invalid temporal bounds")

            # This is the only normalization performed by the formal parser.
            normalized_end = min(end, chunk_duration)
            raw_item = {
                "chunk_index": int(chunk["chunk_index"]),
                "start_sec": start,
                "end_sec": end,
                "reason": item.get("reason"),
                "has_start_sec": "start_sec" in item,
                "has_end_sec": "end_sec" in item,
                "has_score": "score" in item,
                "has_reason": "reason" in item,
            }
            raw_segments.append(raw_item)
            normalized.append(
                {
                    "chunk_index": int(chunk["chunk_index"]),
                    "start_sec": chunk_start + start,
                    "end_sec": chunk_start + normalized_end,
                    "reason": item.get("reason"),
                    "end_was_clipped": normalized_end != end,
                }
            )
        chunk_facts.append(
            {
                "chunk_index": int(chunk["chunk_index"]),
                "finish_reason": chunk.get("finish_reason"),
                "parse_success": chunk.get("parse_success"),
                "parse_error": chunk.get("parse_error"),
                "json_syntax_valid": True,
                "segment_count": len(segments),
            }
        )
    return {
        "raw_segments": raw_segments,
        "normalized_segments": normalized,
        "chunk_facts": chunk_facts,
    }


def _sample_from_success(record: dict[str, Any], raw_count: int) -> dict[str, Any]:
    metrics = record["metrics"]
    duration = float(record["experiment_clip_duration_sec"])
    prediction = float(metrics["prediction_duration_sec"])
    intersection = float(metrics["intersection_sec"])
    reference = float(metrics["reference_duration_sec"])
    return {
        "precision": float(metrics["weak_ref_precision"]),
        "recall": float(metrics["weak_ref_recall"]),
        "f1": float(metrics["weak_ref_f1"]),
        "temporal_iou": float(metrics["temporal_iou"]),
        "prediction_duration_sec": prediction,
        "reference_duration_sec": reference,
        "intersection_sec": intersection,
        "duration_union_sec": float(metrics["duration_union_sec"]),
        "coverage": prediction / duration,
        "over_prediction_sec": prediction - intersection,
        "missed_reference_sec": reference - intersection,
        "raw_candidate_count": raw_count,
        "temporal_segment_count": len(record["merged_prediction_segments"]),
        "inference_time_sec": float(record["timing"]["model_inference_sec"]),
    }


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(samples),
        "mean_precision": mean(item["precision"] for item in samples),
        "mean_recall": mean(item["recall"] for item in samples),
        "mean_f1": mean(item["f1"] for item in samples),
        "mean_temporal_iou": mean(item["temporal_iou"] for item in samples),
        "mean_coverage": mean(item["coverage"] for item in samples),
        "full_video_coverage_count": sum(
            item["coverage"] >= FULL_VIDEO_COVERAGE_THRESHOLD for item in samples
        ),
        "coverage_gte_90_count": sum(item["coverage"] >= 0.9 for item in samples),
        "cumulative_over_prediction_sec": sum(
            item["over_prediction_sec"] for item in samples
        ),
        "cumulative_missed_reference_sec": sum(
            item["missed_reference_sec"] for item in samples
        ),
        "raw_candidate_count": sum(item["raw_candidate_count"] for item in samples),
        "temporal_segment_count": sum(
            item["temporal_segment_count"] for item in samples
        ),
        "mean_inference_time_sec": mean(item["inference_time_sec"] for item in samples),
    }


def _fmt(value: Any) -> Any:
    return "" if value is None else value


def build(repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sys.path.insert(0, str(repo_root / "src"))
    from aic_video_highlight.highlight_retrieval.temporal_metrics import (
        duration_based_metrics,
    )

    manifest = _load_json(repo_root / "configs" / "baseline_20.json")
    manifest_by_id = {item["video_id"]: item for item in manifest}
    output_4b = repo_root / "outputs" / "baseline_20_qwen3.5_4b_zero_shot_v0"
    output_9b = repo_root / "outputs" / "baseline_20_qwen3.5_9b_zero_shot_v0"
    predictions_4b = {item["video_id"]: item for item in _load_jsonl(output_4b / "predictions.jsonl")}
    predictions_9b = {item["video_id"]: item for item in _load_jsonl(output_9b / "predictions.jsonl")}
    raw_4b = {item["video_id"]: item for item in (_load_json(path) for path in (output_4b / "raw").glob("*.json"))}
    raw_9b = {item["video_id"]: item for item in (_load_json(path) for path in (output_9b / "raw").glob("*.json"))}

    if set(manifest_by_id) != set(predictions_4b) or set(manifest_by_id) != set(predictions_9b):
        raise AssertionError("manifest and prediction video IDs differ")

    sample_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    all_4b: list[dict[str, Any]] = []
    matched_4b: list[dict[str, Any]] = []
    matched_9b: list[dict[str, Any]] = []
    salvaged_9b: list[dict[str, Any]] = []

    for manifest_item in manifest:
        video_id = manifest_item["video_id"]
        record_4b = predictions_4b[video_id]
        record_9b = predictions_9b[video_id]
        raw_count_4b = len(_all_raw_segments(raw_4b[video_id]))
        raw_count_9b = len(_all_raw_segments(raw_9b[video_id]))
        metrics_4b = _sample_from_success(record_4b, raw_count_4b)
        all_4b.append(metrics_4b)

        official_success = bool(record_9b["success"])
        official_9b: dict[str, Any] | None = None
        if official_success:
            official_9b = _sample_from_success(record_9b, raw_count_9b)
            temporal_9b = official_9b
            matched_4b.append(metrics_4b)
            matched_9b.append(official_9b)
        else:
            if video_id not in SALVAGE_IDS:
                raise AssertionError(f"unexpected failed sample: {video_id}")
            salvage = _temporal_only_salvage(raw_9b[video_id])
            normalized_intervals = [
                (item["start_sec"], item["end_sec"])
                for item in salvage["normalized_segments"]
            ]
            duration_metrics = duration_based_metrics(
                normalized_intervals,
                [
                    (item["start_sec"], item["end_sec"])
                    for item in manifest_item["weak_reference_segments"]
                ],
            )
            duration = float(raw_9b[video_id]["experiment_clip_duration_sec"])
            temporal_9b = {
                "precision": duration_metrics["precision"],
                "recall": duration_metrics["recall"],
                "f1": duration_metrics["f1"],
                "temporal_iou": duration_metrics["temporal_iou"],
                "prediction_duration_sec": duration_metrics["predicted_sec"],
                "reference_duration_sec": duration_metrics["reference_sec"],
                "intersection_sec": duration_metrics["intersection_sec"],
                "duration_union_sec": duration_metrics["union_sec"],
                "coverage": duration_metrics["predicted_sec"] / duration,
                "over_prediction_sec": (
                    duration_metrics["predicted_sec"] - duration_metrics["intersection_sec"]
                ),
                "missed_reference_sec": (
                    duration_metrics["reference_sec"] - duration_metrics["intersection_sec"]
                ),
                "raw_candidate_count": raw_count_9b,
                "temporal_segment_count": len(salvage["normalized_segments"]),
                "inference_time_sec": sum(
                    float(chunk["request_latency_sec"])
                    for chunk in raw_9b[video_id]["chunks"]
                ),
            }
            detail[video_id] = {
                "status_labels": [
                    "SALVAGED",
                    "TEMPORAL_ONLY",
                    "NON_OFFICIAL_PIPELINE_RESULT",
                ],
                "raw_response": [chunk["raw_response"] for chunk in raw_9b[video_id]["chunks"]],
                "raw_vs_normalized": salvage,
                "schema_violations": [
                    "missing score on every segment",
                    f"segment_count={raw_count_9b} exceeds max_segments_per_chunk=5",
                ],
                "metrics": temporal_9b,
            }

        salvaged_9b.append(temporal_9b)
        failure_reason = ""
        if not official_success:
            failure_reason = record_9b["errors"][0]["message"]
        sample_rows.append(
            {
                "video_id": video_id,
                "group": manifest_item["group"],
                "4b_success": "true",
                "9b_official_success": str(official_success).lower(),
                "9b_salvaged": str(not official_success).lower(),
                "salvaged_temporal": str(not official_success).lower(),
                "4b_precision": metrics_4b["precision"],
                "4b_recall": metrics_4b["recall"],
                "4b_f1": metrics_4b["f1"],
                "4b_tiou": metrics_4b["temporal_iou"],
                "9b_official_precision": _fmt(official_9b and official_9b["precision"]),
                "9b_official_recall": _fmt(official_9b and official_9b["recall"]),
                "9b_official_f1": _fmt(official_9b and official_9b["f1"]),
                "9b_official_tiou": _fmt(official_9b and official_9b["temporal_iou"]),
                "9b_salvaged_precision": temporal_9b["precision"],
                "9b_salvaged_recall": temporal_9b["recall"],
                "9b_salvaged_f1": temporal_9b["f1"],
                "9b_salvaged_tiou": temporal_9b["temporal_iou"],
                "4b_coverage": metrics_4b["coverage"],
                "9b_coverage": temporal_9b["coverage"],
                "9b_official_coverage": _fmt(official_9b and official_9b["coverage"]),
                "9b_salvaged_coverage": temporal_9b["coverage"],
                "4b_segment_count": metrics_4b["temporal_segment_count"],
                "9b_segment_count": temporal_9b["temporal_segment_count"],
                "9b_official_segment_count": _fmt(
                    official_9b and official_9b["temporal_segment_count"]
                ),
                "9b_salvaged_segment_count": temporal_9b["temporal_segment_count"],
                "4b_raw_candidate_count": metrics_4b["raw_candidate_count"],
                "9b_raw_candidate_count": temporal_9b["raw_candidate_count"],
                "schema_compliant": str(official_success).lower(),
                "schema_failure_reason": failure_reason,
                "delta_precision": temporal_9b["precision"] - metrics_4b["precision"],
                "delta_recall": temporal_9b["recall"] - metrics_4b["recall"],
                "delta_f1": temporal_9b["f1"] - metrics_4b["f1"],
                "delta_tiou": temporal_9b["temporal_iou"] - metrics_4b["temporal_iou"],
                "notes": (
                    "SALVAGED|TEMPORAL_ONLY|NON_OFFICIAL_PIPELINE_RESULT"
                    if not official_success
                    else "official_equals_temporal_diagnostic"
                ),
            }
        )

    result = {
        "labels": ["SALVAGED", "TEMPORAL_ONLY", "NON_OFFICIAL_PIPELINE_RESULT"],
        "policy": {
            "metric": "project duration_based_metrics (union-aware)",
            "normalization": "formal parser temporal rules only; clip end above chunk duration",
            "full_video_coverage_threshold": FULL_VIDEO_COVERAGE_THRESHOLD,
            "official_outputs_modified": False,
            "score_imputed": False,
            "segments_removed_or_limited": False,
        },
        "official_9b": {
            "processed": 20,
            "success": 18,
            "failed": 2,
            "schema_success_rate": 0.9,
            "failed_video_ids": sorted(SALVAGE_IDS),
        },
        "aggregates": {
            "4b_official_20": _aggregate(all_4b),
            "4b_matched_18": _aggregate(matched_4b),
            "9b_official_matched_18": _aggregate(matched_9b),
            "9b_salvaged_temporal_20": _aggregate(salvaged_9b),
        },
        "salvaged_samples": detail,
    }

    expected = result["aggregates"]
    assert len(sample_rows) == 20
    assert expected["4b_official_20"]["raw_candidate_count"] == 78
    assert expected["4b_official_20"]["temporal_segment_count"] == 74
    assert expected["9b_official_matched_18"]["raw_candidate_count"] == 52
    assert expected["9b_official_matched_18"]["temporal_segment_count"] == 50
    assert expected["9b_salvaged_temporal_20"]["raw_candidate_count"] == 65
    assert expected["9b_salvaged_temporal_20"]["temporal_segment_count"] == 63
    assert math.isclose(expected["4b_matched_18"]["mean_precision"], 0.403331226528718)
    assert math.isclose(expected["9b_official_matched_18"]["mean_precision"], 0.41634374056753903)
    return result, sample_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    result, rows = build(repo_root)
    if not args.check_only:
        data_dir = repo_root / "docs" / "experiments" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        with (data_dir / "qwen35_9b_salvaged_temporal_diagnostic.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        csv_path = repo_root / "docs" / "experiments" / "qwen35_4b_vs_9b_baseline20.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result["aggregates"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
