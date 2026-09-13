"""Official-formula-compatible evaluator for local Dev weak references.

This module intentionally never calls its output an official score.  It uses
the published exact-video/exact-frame matching formula, but Dev references are
weak project references rather than the hidden competition ground truth.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from aic_video_highlight.composition.center_crop import derived_height


class OfficialLikeEvaluationError(ValueError):
    """Raised for ambiguous or malformed prediction/reference data."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise OfficialLikeEvaluationError(f"line {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _ratio(record: Mapping[str, Any]) -> tuple[float, float]:
    raw = record.get("targetRatioWH")
    if not isinstance(raw, list) or len(raw) != 2:
        raise OfficialLikeEvaluationError("targetRatioWH must contain two components")
    tw, th = float(raw[0]), float(raw[1])
    if not all(math.isfinite(value) and value > 0 for value in (tw, th)):
        raise OfficialLikeEvaluationError("targetRatioWH components must be finite and positive")
    return tw, th


def _prediction_frames(record: Mapping[str, Any]) -> dict[int, tuple[float, float, float, float]]:
    tw, th = _ratio(record)
    raw = record.get("predictions")
    if not isinstance(raw, list):
        raise OfficialLikeEvaluationError("predictions must be an array")
    result: dict[int, tuple[float, float, float, float]] = {}
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("bboxes"), list):
            raise OfficialLikeEvaluationError("prediction item requires frame and bboxes")
        frame = int(item["frame"])
        if frame in result:
            raise OfficialLikeEvaluationError(f"duplicate prediction frame: {frame}")
        bbox = item["bboxes"]
        if len(bbox) != 3:
            raise OfficialLikeEvaluationError("official prediction bboxes must be [x, y, w]")
        x, y, width = float(bbox[0]), float(bbox[1]), float(bbox[2])
        if not all(math.isfinite(value) for value in (x, y, width)) or width <= 0:
            raise OfficialLikeEvaluationError("prediction bbox values must be finite and width positive")
        # The official contract omits h; derive it exactly from this video's record's ratio.
        height = float(derived_height(int(width), tw, th))
        result[frame] = (x, y, width, height)
    return result


def _reference_frames(record: Mapping[str, Any]) -> dict[int, tuple[float, float, float, float]]:
    if "rois" in record:
        raw = record["rois"]
        if not isinstance(raw, Mapping):
            raise OfficialLikeEvaluationError("weak-reference rois must be an object")
        items: Iterable[tuple[Any, Any]] = raw.items()
    else:
        # Official-shaped fixtures/references use the same [x,y,w] encoding and
        # therefore the same targetRatioWH-derived height rule as predictions.
        return _prediction_frames(record)
    result: dict[int, tuple[float, float, float, float]] = {}
    for frame_raw, bbox in items:
        frame = int(frame_raw)
        if frame in result:
            raise OfficialLikeEvaluationError(f"duplicate reference frame: {frame}")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise OfficialLikeEvaluationError("weak-reference ROI must be [x, y, w, h]")
        values = tuple(float(value) for value in bbox)
        if not all(math.isfinite(value) for value in values) or values[2] <= 0 or values[3] <= 0:
            raise OfficialLikeEvaluationError("reference ROI values must be finite and dimensions positive")
        result[frame] = values
    return result


def spatial_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = left
    bx, by, bw, bh = right
    intersection_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_w * intersection_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _index(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise OfficialLikeEvaluationError(f"{label} record has invalid video_id")
        if video_id in result:
            raise OfficialLikeEvaluationError(f"duplicate {label} video_id: {video_id}")
        result[video_id] = row
    return result


def evaluate_official_like(
    prediction_rows: Iterable[Mapping[str, Any]],
    reference_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    predictions = _index(prediction_rows, "prediction")
    references = _index(reference_rows, "reference")
    video_ids = sorted(set(predictions) | set(references))
    if not video_ids:
        raise OfficialLikeEvaluationError("evaluation set is empty")
    per_video: list[dict[str, Any]] = []
    for video_id in video_ids:
        pred_frames = _prediction_frames(predictions[video_id]) if video_id in predictions else {}
        ref_frames = _reference_frames(references[video_id]) if video_id in references else {}
        common = sorted(set(pred_frames) & set(ref_frames))
        iou_sum = sum(spatial_iou(pred_frames[frame], ref_frames[frame]) for frame in common)
        n_pred, n_gt = len(pred_frames), len(ref_frames)
        precision = iou_sum / n_pred if n_pred else 0.0
        recall = iou_sum / n_gt if n_gt else 0.0
        if n_pred == 0 and n_gt == 0:
            # The published competition contract assigns perfect per-video F
            # when neither predictions nor ground truth contain any frames.
            f_score = 1.0
        else:
            f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_video.append(
            {
                "video_id": video_id,
                "prediction_frames": n_pred,
                "reference_frames": n_gt,
                "exact_common_frames": len(common),
                "predicted_only_frames": n_pred - len(common),
                "reference_only_frames": n_gt - len(common),
                "sum_spatial_iou": iou_sum,
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
            }
        )
    count = len(per_video)
    return {
        "schema_version": "aic.vhicraft.official-like-weak-reference/v1",
        "score_identity": "NOT_OFFICIAL_SCORE",
        "matching": "exact_video_id_and_exact_frame_only",
        "video_count": count,
        "Official-like Weak-Reference Precision": sum(row["precision"] for row in per_video) / count,
        "Official-like Weak-Reference Recall": sum(row["recall"] for row in per_video) / count,
        "Official-like Weak-Reference F-score": sum(row["f_score"] for row in per_video) / count,
        "per_video": per_video,
    }


def evaluate_files(
    predictions: Path,
    references: Path,
    *,
    video_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    prediction_rows = _read_jsonl(predictions)
    reference_rows = _read_jsonl(references)
    if video_ids is not None:
        selected = {str(value) for value in video_ids}
        prediction_rows = [row for row in prediction_rows if str(row.get("video_id")) in selected]
        reference_rows = [row for row in reference_rows if str(row.get("video_id")) in selected]
        present = {str(row.get("video_id")) for row in prediction_rows + reference_rows}
        missing = selected - present
        if missing:
            raise OfficialLikeEvaluationError(f"requested video_ids absent from both inputs: {sorted(missing)}")
    return evaluate_official_like(prediction_rows, reference_rows)
