"""Stage 5.3 composition pipeline: frozen-input loading, manifest, CMP-0/CMP-1 composition.

CMP-0 = Stage 5.1 Frozen Center Crop (control, taken verbatim from the frozen
predictions artifact). CMP-1 = Primary-Subject Shifted Max-Ratio Crop: the Stage
5.1 maximal legal crop size with position shifted toward the sanitized primary
subject center. No detector inference, no frame reselection, no subject research.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from aic_video_highlight.experiment_runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.spatial_composition.center_crop import compute_center_crop, derived_height
from aic_video_highlight.spatial_composition.composition_metrics import (
    STRATUM_NO_SUBJECT,
    classify_center_stratum,
    crop_rect_from_xywh,
    crop_shift_normalized,
    horizontal_center_offset,
    intersection_area,
    max_overflow_px,
    overflow_distribution,
    overflow_threshold_frames,
    raw_bbox_overflow,
    subject_center_inside_crop,
    subject_visible_fraction,
    summarize,
    visible_fraction_thresholds,
)
from aic_video_highlight.spatial_composition.subject_shifted_crop import (
    PLACEMENT_CENTER_EQUIVALENT,
    PLACEMENT_FALLBACK_CENTER_CROP,
    SANITIZE_INVALID,
    SANITIZE_OK,
    SanitizedSubject,
    compute_subject_shifted_crop,
    sanitize_primary_bbox,
    stage5_1_crop_height,
)
from aic_video_highlight.spatial_localization.subject_localization import (
    REASON_LOW_CONFIDENCE,
    REASON_NO_DETECTION,
    STATUS_PRIMARY,
    SubjectCandidate,
    SubjectPolicyConfig,
    candidate_is_valid,
    detection_iou,
)

NEAR_CENTER_DEFAULT = 0.10
STRONGLY_OFF_CENTER_DEFAULT = 0.25


class FrozenInputError(RuntimeError):
    """Raised when a frozen Stage 5.1/5.2 input is missing, unreadable, or hash-mismatched."""


@dataclass(frozen=True, slots=True)
class InputBinding:
    name: str
    path: Path
    sha256: str = ""
    format: str = "json"


@dataclass(frozen=True, slots=True)
class FrozenInputs:
    stage5_1_predictions: dict[str, dict[str, Any]]
    metadata: dict[str, dict[str, Any]]
    index: dict[str, dict[str, Any]]
    policy_records: tuple[dict[str, Any], ...]
    raw_frames: dict[tuple[str, int], dict[str, Any]]
    weak_reference: dict[str, dict[str, Any]]
    input_hashes: dict[str, str]


def verify_input_bindings(bindings: Sequence[InputBinding]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for binding in bindings:
        if not binding.path.is_file():
            raise FrozenInputError(f"frozen input missing: {binding.name} -> {binding.path}")
        actual = file_sha256(binding.path)
        if binding.sha256 and actual != binding.sha256:
            raise FrozenInputError(
                f"frozen input hash mismatch: {binding.name}: expected {binding.sha256}, got {actual}"
            )
        hashes[binding.name] = actual
    return hashes


def load_policy_shard_dir(path: Path) -> list[dict[str, Any]]:
    """Load frozen Stage 5.2 policy decisions from the canonical per-video shard layout."""
    if not path.is_dir():
        raise FrozenInputError(f"policy shard dir missing: {path}")
    records: list[dict[str, Any]] = []
    for shard in sorted(path.glob("*.jsonl")):
        records.extend(
            json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    if not records:
        raise FrozenInputError(f"policy shard dir empty: {path}")
    return records


def policy_shard_semantic_sha(records: Sequence[Mapping[str, Any]]) -> str:
    """Semantic hash of merged policy records using the frozen full-dev hash convention."""
    from aic_video_highlight.spatial_localization.full_dev import canonical_sha

    return canonical_sha(records)


def load_raw_shard_dir(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load frozen Stage 5.2 raw detector records from the canonical shard layout."""
    if not path.is_dir():
        raise FrozenInputError(f"raw shard dir missing: {path}")
    frames: dict[tuple[str, int], dict[str, Any]] = {}
    for shard in sorted(path.glob("*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                frames[(record["video_id"], int(record["frame"]))] = record
    if not frames:
        raise FrozenInputError(f"raw shard dir empty: {path}")
    return frames


def verify_shard_dir_integrity(
    output_dir: Path,
    expected_keys: Mapping[str, set[int]],
) -> dict[str, Any]:
    """Recheck every frozen Stage 5.2 shard status (frame count + byte hashes)."""
    from aic_video_highlight.spatial_localization.full_dev import shard_is_complete

    broken: list[str] = []
    for video_id, frames in sorted(expected_keys.items()):
        if not shard_is_complete(output_dir, video_id, sorted(frames)):
            broken.append(video_id)
    return {
        "videos_checked": len(expected_keys),
        "broken_shards": broken,
        "intact": not broken,
    }


def load_frozen_inputs(
    bindings: Sequence[InputBinding],
    *,
    include_raw: bool = True,
    policy_semantic_expectation: str | None = None,
) -> FrozenInputs:
    """Load and hash-verify frozen Stage 5.1/5.2 inputs (read-only)."""
    payloads: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    policy_records: tuple[dict[str, Any], ...] = ()
    for binding in bindings:
        if binding.format in ("policy_shard_dir", "raw_shard_dir"):
            if not binding.path.is_dir():
                raise FrozenInputError(f"frozen shard dir missing: {binding.name} -> {binding.path}")
            if binding.format == "raw_shard_dir":
                continue
            policy_records = tuple(load_policy_shard_dir(binding.path))
            semantic = policy_shard_semantic_sha(policy_records)
            if policy_semantic_expectation and semantic != policy_semantic_expectation:
                raise FrozenInputError(
                    f"policy artifact semantic mismatch: expected {policy_semantic_expectation}, got {semantic}"
                )
            hashes[binding.name] = semantic
            continue
        if not binding.path.is_file():
            raise FrozenInputError(f"frozen input missing: {binding.name} -> {binding.path}")
        actual = file_sha256(binding.path)
        if binding.sha256 and actual != binding.sha256:
            raise FrozenInputError(
                f"frozen input hash mismatch: {binding.name}: expected {binding.sha256}, got {actual}"
            )
        hashes[binding.name] = actual
        if binding.format == "jsonl":
            payloads[binding.name] = [
                json.loads(line)
                for line in binding.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payloads[binding.name] = json.loads(binding.path.read_text(encoding="utf-8"))

    predictions = {record["video_id"]: record for record in payloads["stage5_1_predictions"]}
    metadata_payload = payloads["video_metadata_cache"]
    metadata = metadata_payload["records"] if "records" in metadata_payload else metadata_payload
    index = {record["video_id"]: record for record in payloads["dev166_index"]}
    weak_reference = {record["video_id"]: record for record in payloads["weak_spatial_reference"]}
    if "stage5_2_policy_artifact" not in hashes:
        raise FrozenInputError("stage5_2_policy_artifact binding is required")
    if policy_records == ():
        policy_records = tuple(payloads["stage5_2_policy_artifact"])

    raw_frames: dict[tuple[str, int], dict[str, Any]] = {}
    if include_raw:
        raw_binding = next((b for b in bindings if b.name == "stage5_2_raw_detector"), None)
        if raw_binding is None:
            raise FrozenInputError("stage5_2_raw_detector binding is required")
        if raw_binding.format == "raw_shard_dir":
            raw_frames = load_raw_shard_dir(raw_binding.path)
            hashes["stage5_2_raw_detector"] = "SHARD_DIR_VERIFIED_SEPARATELY"
        else:
            raw_frames = {
                (frame["video_id"], int(frame["frame"])): frame
                for frame in payloads["stage5_2_raw_detector"]["frames"]
            }

    _cross_check_inputs(predictions, metadata, index, policy_records, raw_frames, include_raw)

    return FrozenInputs(
        stage5_1_predictions=predictions,
        metadata=metadata,
        index=index,
        policy_records=policy_records,
        raw_frames=raw_frames,
        weak_reference=weak_reference,
        input_hashes=hashes,
    )


def _cross_check_inputs(
    predictions: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    index: Mapping[str, Mapping[str, Any]],
    policy_records: Sequence[Mapping[str, Any]],
    raw_frames: Mapping[tuple[str, int], Mapping[str, Any]],
    include_raw: bool = True,
) -> None:
    for video_id, record in index.items():
        if video_id not in predictions:
            raise FrozenInputError(f"index video missing from Stage 5.1 predictions: {video_id}")
        if video_id not in metadata:
            raise FrozenInputError(f"index video missing from metadata cache: {video_id}")
        if list(map(float, record["targetRatioWH"])) != list(map(float, predictions[video_id]["targetRatioWH"])):
            raise FrozenInputError(f"targetRatioWH disagreement for {video_id}")
    if include_raw:
        for record in policy_records:
            key = (record["video_id"], int(record["frame"]))
            if key not in raw_frames:
                raise FrozenInputError(f"policy frame without raw detector record: {key}")


def build_manifest(
    inputs: FrozenInputs,
    *,
    manifest_id: str,
    strata_thresholds: Mapping[str, float],
    target_ratio: Sequence[int | float],
) -> dict[str, Any]:
    """Deterministic, model-blind manifest from frozen Stage 5.1/5.2 artifacts.

    Frame set = exact frames of the frozen Stage 5.2 formal policy artifact
    (sorted video_id, ascending frames); strata derive only from frozen policy
    decisions and frame geometry, never from CMP-1 output.
    """
    policy_by_video: dict[str, list[dict[str, Any]]] = {}
    for record in inputs.policy_records:
        policy_by_video.setdefault(record["video_id"], []).append(record)

    videos: list[dict[str, Any]] = []
    strata_counts: dict[str, int] = {}
    frame_count = 0
    for video_id in sorted(policy_by_video):
        meta = inputs.metadata[video_id]
        width, height = int(meta["width"]), int(meta["height"])
        ratio = [float(value) for value in inputs.index[video_id]["targetRatioWH"]]
        if ratio != [float(value) for value in target_ratio]:
            raise FrozenInputError(f"target ratio disagreement for {video_id}: {ratio}")
        predicted = {int(pred["frame"]) for pred in inputs.stage5_1_predictions[video_id]["predictions"]}
        entries = []
        for record in sorted(policy_by_video[video_id], key=lambda item: int(item["frame"])):
            frame = int(record["frame"])
            if frame not in predicted:
                raise FrozenInputError(f"Stage 5.2 frame outside the Stage 5.1 frozen set: {(video_id, frame)}")
            stratum, offset = _frame_stratum(record, width, height, strata_thresholds)
            strata_counts[stratum] = strata_counts.get(stratum, 0) + 1
            entries.append(
                {
                    "frame": frame,
                    "stratum": stratum,
                    "stage5_2_status": record["status"],
                    "horizontal_center_offset": _round_or_none(offset),
                }
            )
        frame_count += len(entries)
        videos.append(
            {
                "video_id": video_id,
                "image_width": width,
                "image_height": height,
                "target_ratio_wh": ratio,
                "frame_count": len(entries),
                "frames": entries,
            }
        )
    manifest: dict[str, Any] = {
        "manifest_id": manifest_id,
        "model_blind": True,
        "selection_rule": (
            "frames = exact frame set of the frozen Stage 5.2 formal policy artifact "
            "(sorted video_id, ascending frames); strata = preregistered horizontal "
            "center-offset thresholds on the sanitized Stage 5.2 primary bbox; "
            "no CMP-1 output, no detector rerun, no Heldout access"
        ),
        "strata_thresholds": {
            "near_center_lt": float(strata_thresholds["near_center_lt"]),
            "strongly_off_center_gte": float(strata_thresholds["strongly_off_center_gte"]),
        },
        "target_ratio_wh": [float(value) for value in target_ratio],
        "videos": videos,
        "video_count": len(videos),
        "frame_count": frame_count,
        "strata_counts": dict(sorted(strata_counts.items())),
        "frozen_bindings": {
            "stage5_1_predictions_sha256": inputs.input_hashes["stage5_1_predictions"],
            "video_metadata_cache_sha256": inputs.input_hashes["video_metadata_cache"],
            "stage5_2_policy_artifact_sha256": inputs.input_hashes["stage5_2_policy_artifact"],
            "stage5_2_raw_detector_sha256": inputs.input_hashes.get("stage5_2_raw_detector", "DEFERRED_NOT_LOADED"),
            "weak_spatial_reference_sha256": inputs.input_hashes["weak_spatial_reference"],
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _frame_stratum(
    record: Mapping[str, Any],
    width: int,
    height: int,
    thresholds: Mapping[str, float],
) -> tuple[str, float | None]:
    primary = record.get("primary")
    if record["status"] != STATUS_PRIMARY or not primary:
        return STRATUM_NO_SUBJECT, None
    sanitized = sanitize_primary_bbox(primary["xyxy"], width, height)
    if not sanitized.valid:
        return STRATUM_NO_SUBJECT, None
    offset = horizontal_center_offset(sanitized, width)
    return (
        classify_center_stratum(
            offset,
            float(thresholds["near_center_lt"]),
            float(thresholds["strongly_off_center_gte"]),
        ),
        offset,
    )


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def _fraction_to_float(value: Fraction | float) -> float:
    return float(value)


def _iou_rect(a: Sequence[float], b: Sequence[float]) -> float:
    inter = intersection_area(a, b)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return round(inter / union, 6) if union > 0 else 0.0


def _geometry_gate(
    cmp0_x: int,
    cmp0_y: int,
    cmp0_w: int,
    cmp1_x: int,
    cmp1_y: int,
    cmp1_w: int,
    width: int,
    height: int,
    tw: float,
    th: float,
) -> dict[str, bool]:
    def check(x: int, y: int, w: int) -> dict[str, bool]:
        derived = Fraction(w) * Fraction(str(th)) / Fraction(str(tw))
        return {
            "nonnegative": x >= 0 and y >= 0,
            "width_positive": w > 0,
            "x_within_frame": x + w <= width,
            "derived_height_within_frame": y + derived <= height,
        }

    return {
        f"cmp0_{key}": value for key, value in check(cmp0_x, cmp0_y, cmp0_w).items()
    } | {
        f"cmp1_{key}": value for key, value in check(cmp1_x, cmp1_y, cmp1_w).items()
    }


def compose_frame(
    video_id: str,
    frame: int,
    inputs: FrozenInputs,
    target_ratio: Sequence[int | float],
    strata_thresholds: Mapping[str, float] | None = None,
    frozen_boxes: Mapping[int, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Pure per-frame composition: CMP-0 (frozen) vs CMP-1 (subject-shifted)."""
    thresholds = dict(strata_thresholds or {"near_center_lt": NEAR_CENTER_DEFAULT, "strongly_off_center_gte": STRONGLY_OFF_CENTER_DEFAULT})
    meta = inputs.metadata[video_id]
    width, height = int(meta["width"]), int(meta["height"])
    tw, th = float(target_ratio[0]), float(target_ratio[1])
    policy = next(
        record
        for record in inputs.policy_records
        if record["video_id"] == video_id and int(record["frame"]) == frame
    )
    if frozen_boxes is None:
        frozen_boxes = {
            int(pred["frame"]): list(pred["bboxes"])
            for pred in inputs.stage5_1_predictions[video_id]["predictions"]
        }
    frozen_x, frozen_y, frozen_w = (int(value) for value in frozen_boxes[frame])

    center = compute_center_crop(width, height, tw, th)
    cmp0_regression = (center.x, center.y, center.w) != (frozen_x, frozen_y, frozen_w)

    primary = policy.get("primary")
    raw_bbox = [float(value) for value in primary["xyxy"]] if primary else None
    sanitized = sanitize_primary_bbox(raw_bbox, width, height)
    is_fallback = policy["status"] != STATUS_PRIMARY or raw_bbox is None or not sanitized.valid

    cmp0_h = derived_height(frozen_w, tw, th)
    cmp0_rect = crop_rect_from_xywh(frozen_x, frozen_y, frozen_w, _fraction_to_float(cmp0_h))

    if is_fallback:
        cmp1 = {
            "x": frozen_x,
            "y": frozen_y,
            "w": frozen_w,
            "h": _fraction_to_float(cmp0_h),
            "crop_w": frozen_w,
            "crop_h": stage5_1_crop_height(width, height, tw, th),
            "placement_status": PLACEMENT_FALLBACK_CENTER_CROP,
            "ideal_x": float(frozen_x),
            "ideal_y": float(frozen_y),
            "clamped_x": False,
            "clamped_y": False,
            "fallback": True,
        }
    else:
        shifted = compute_subject_shifted_crop(width, height, tw, th, (sanitized.center_x, sanitized.center_y))
        cmp1 = {
            "x": shifted.x,
            "y": shifted.y,
            "w": shifted.w,
            "h": _fraction_to_float(shifted.h),
            "crop_w": shifted.crop_w,
            "crop_h": shifted.crop_h,
            "placement_status": shifted.placement_status,
            "ideal_x": shifted.ideal_x,
            "ideal_y": shifted.ideal_y,
            "clamped_x": shifted.clamped_x,
            "clamped_y": shifted.clamped_y,
            "fallback": False,
        }
    cmp1_rect = crop_rect_from_xywh(cmp1["x"], cmp1["y"], cmp1["w"], cmp1["h"])

    cmp0_shift = crop_shift_normalized(frozen_x, frozen_w, float(frozen_y), _fraction_to_float(cmp0_h), width, height)
    cmp1_shift = crop_shift_normalized(cmp1["x"], cmp1["w"], float(cmp1["y"]), cmp1["h"], width, height)

    if sanitized.valid:
        cmp0_visible = subject_visible_fraction(sanitized, cmp0_rect)
        cmp1_visible = subject_visible_fraction(sanitized, cmp1_rect)
        cmp0_center_inside = subject_center_inside_crop(sanitized, cmp0_rect)
        cmp1_center_inside = subject_center_inside_crop(sanitized, cmp1_rect)
        center_offset = horizontal_center_offset(sanitized, width)
        stratum = classify_center_stratum(
            center_offset, float(thresholds["near_center_lt"]), float(thresholds["strongly_off_center_gte"])
        )
    else:
        cmp0_visible = cmp1_visible = None
        cmp0_center_inside = cmp1_center_inside = False
        center_offset = None
        stratum = STRATUM_NO_SUBJECT

    geometry = _geometry_gate(frozen_x, frozen_y, frozen_w, cmp1["x"], cmp1["y"], cmp1["w"], width, height, tw, th)

    weak = inputs.weak_reference.get(video_id, {}).get("rois", {}).get(str(frame))
    weak_payload = None
    if weak is not None:
        weak_rect = (
            float(weak[0]),
            float(weak[1]),
            float(weak[0]) + float(weak[2]),
            float(weak[1]) + float(weak[3]),
        )
        weak_payload = {
            "exists": True,
            "iou_cmp0": _iou_rect(cmp0_rect, weak_rect),
            "iou_cmp1": _iou_rect(cmp1_rect, weak_rect),
        }

    return {
        "video_id": video_id,
        "frame": frame,
        "image_width": width,
        "image_height": height,
        "stage5_2_status": policy["status"],
        "fallback_reasons": list(policy.get("fallback_reasons", [])),
        "ambiguous": bool(policy.get("ambiguous")),
        "ambiguous_candidate_count": int(policy.get("ambiguous_candidate_count", 0)),
        "primary_label": primary["label"] if primary else None,
        "primary_score": primary["score"] if primary else None,
        "raw_primary_bbox": raw_bbox,
        "sanitized": {
            "xyxy": list(sanitized.as_xyxy()),
            "status": sanitized.status,
            "clamp_left": sanitized.clamp_left,
            "clamp_top": sanitized.clamp_top,
            "clamp_right": sanitized.clamp_right,
            "clamp_bottom": sanitized.clamp_bottom,
        },
        "stratum": stratum,
        "horizontal_center_offset": _round_or_none(center_offset),
        "cmp0": {
            "x": frozen_x,
            "y": frozen_y,
            "w": frozen_w,
            "h": _fraction_to_float(cmp0_h),
            "subject_visible_fraction": _round_or_none(cmp0_visible),
            "subject_center_inside": cmp0_center_inside,
            "shift_x_norm": round(cmp0_shift[0], 6),
            "shift_y_norm": round(cmp0_shift[1], 6),
            "frozen_regression": cmp0_regression,
        },
        "cmp1": {
            **cmp1,
            "subject_visible_fraction": _round_or_none(cmp1_visible),
            "subject_center_inside": cmp1_center_inside,
            "shift_x_norm": round(cmp1_shift[0], 6),
            "shift_y_norm": round(cmp1_shift[1], 6),
        },
        "geometry_valid": geometry,
        "weak_reference": weak_payload,
    }


def compose_all_frames(
    manifest: Mapping[str, Any],
    inputs: FrozenInputs,
    target_ratio: Sequence[int | float],
    strata_thresholds: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    thresholds = dict(strata_thresholds or {"near_center_lt": NEAR_CENTER_DEFAULT, "strongly_off_center_gte": STRONGLY_OFF_CENTER_DEFAULT})
    for video in manifest["videos"]:
        frozen_boxes = {
            int(pred["frame"]): list(pred["bboxes"])
            for pred in inputs.stage5_1_predictions[video["video_id"]]["predictions"]
        }
        for entry in video["frames"]:
            records.append(
                compose_frame(
                    video["video_id"],
                    int(entry["frame"]),
                    inputs,
                    target_ratio,
                    thresholds,
                    frozen_boxes,
                )
            )
    return records


def engineering_gate(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_keys = {
        (video["video_id"], int(entry["frame"]))
        for video in manifest["videos"]
        for entry in video["frames"]
    }
    produced_keys = [(record["video_id"], int(record["frame"])) for record in records]
    missing = expected_keys - set(produced_keys)
    extra = set(produced_keys) - expected_keys
    duplicates = len(produced_keys) - len(set(produced_keys))
    sanitized_invalid = [r for r in records if r["sanitized"]["status"] == SANITIZE_INVALID]
    sanitized_invalid_unhandled = [
        r
        for r in sanitized_invalid
        if r["cmp1"]["placement_status"] != PLACEMENT_FALLBACK_CENTER_CROP
    ]
    geometry_failures = sum(1 for r in records if not all(r["geometry_valid"].values()))
    ratio_violations = sum(
        1
        for r in records
        if not (
            r["geometry_valid"]["cmp0_derived_height_within_frame"]
            and r["geometry_valid"]["cmp1_derived_height_within_frame"]
        )
    )
    out_of_bounds = sum(
        1
        for r in records
        if not (
            r["geometry_valid"]["cmp0_x_within_frame"]
            and r["geometry_valid"]["cmp1_x_within_frame"]
            and r["geometry_valid"]["cmp0_nonnegative"]
            and r["geometry_valid"]["cmp1_nonnegative"]
        )
    )
    cmp0_regression = sum(1 for r in records if r["cmp0"]["frozen_regression"])
    return {
        "expected_frames": len(expected_keys),
        "produced_frames": len(produced_keys),
        "missing": len(missing),
        "extra": len(extra),
        "duplicates": duplicates,
        "invalid_crop": geometry_failures,
        "out_of_bounds": out_of_bounds,
        "ratio_violations": ratio_violations,
        "sanitized_invalid": len(sanitized_invalid),
        "sanitized_invalid_unhandled": len(sanitized_invalid_unhandled),
        "cmp0_frozen_regression": cmp0_regression,
        "frame_identity_complete": not missing and not extra and not duplicates,
    }


def _records_by_stratum(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["stratum"], []).append(record)
    return grouped


def aggregate_metrics(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    strata_thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    gate = engineering_gate(manifest, records)
    primary_records = [r for r in records if r["sanitized"]["status"] == SANITIZE_OK]
    cmp0_visible = [r["cmp0"]["subject_visible_fraction"] for r in primary_records]
    cmp1_visible = [r["cmp1"]["subject_visible_fraction"] for r in primary_records]
    improvements = [
        r["cmp1"]["subject_visible_fraction"] - r["cmp0"]["subject_visible_fraction"]
        for r in primary_records
    ]
    placement_counts: dict[str, int] = {}
    for record in records:
        status = record["cmp1"]["placement_status"]
        placement_counts[status] = placement_counts.get(status, 0) + 1
    fallback_reasons: dict[str, int] = {}
    for record in records:
        for reason in record["fallback_reasons"]:
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

    strata_payload: dict[str, Any] = {}
    for stratum, grouped in sorted(_records_by_stratum(records).items()):
        valid = [r for r in grouped if r["sanitized"]["status"] == SANITIZE_OK]
        strata_payload[stratum] = {
            "n": len(grouped),
            "with_subject": len(valid),
            "cmp0_visible": summarize(r["cmp0"]["subject_visible_fraction"] for r in valid).as_dict(),
            "cmp1_visible": summarize(r["cmp1"]["subject_visible_fraction"] for r in valid).as_dict(),
            "mean_improvement": _round_or_none(
                sum(
                    r["cmp1"]["subject_visible_fraction"] - r["cmp0"]["subject_visible_fraction"]
                    for r in valid
                )
                / len(valid)
            )
            if valid
            else None,
            "cmp1_shifted_rate": round(
                sum(1 for r in grouped if r["cmp1"]["placement_status"] == "SUBJECT_SHIFTED") / len(grouped), 6
            )
            if grouped
            else 0.0,
        }

    overflow_frames = [
        max_overflow_px(raw_bbox_overflow(r["raw_primary_bbox"], r["image_width"], r["image_height"]))
        for r in records
        if r["raw_primary_bbox"] is not None
    ]
    direction_overflow: dict[str, list[float]] = {"left": [], "top": [], "right": [], "bottom": []}
    for record in records:
        if record["raw_primary_bbox"] is None:
            continue
        for direction, value in raw_bbox_overflow(
            record["raw_primary_bbox"], record["image_width"], record["image_height"]
        ).items():
            direction_overflow[direction].append(value)

    weak_rows = [r for r in records if r["weak_reference"] is not None]
    weak_wins = sum(1 for r in weak_rows if r["weak_reference"]["iou_cmp1"] > r["weak_reference"]["iou_cmp0"] + 1e-9)
    weak_ties = sum(
        1 for r in weak_rows if abs(r["weak_reference"]["iou_cmp1"] - r["weak_reference"]["iou_cmp0"]) <= 1e-9
    )
    weak_losses = sum(1 for r in weak_rows if r["weak_reference"]["iou_cmp1"] < r["weak_reference"]["iou_cmp0"] - 1e-9)

    return {
        "engineering_gate": gate,
        "frames": len(records),
        "frames_with_subject": len(primary_records),
        "composition_counts": {
            "placement_status": dict(sorted(placement_counts.items())),
            "cmp1_fallback": placement_counts.get(PLACEMENT_FALLBACK_CENTER_CROP, 0),
            "cmp1_shifted": placement_counts.get("SUBJECT_SHIFTED", 0),
            "cmp1_center_equivalent": placement_counts.get(PLACEMENT_CENTER_EQUIVALENT, 0),
        },
        "subject_visibility": {
            "cmp0": {**summarize(cmp0_visible).as_dict(), "thresholds": visible_fraction_thresholds(cmp0_visible)},
            "cmp1": {**summarize(cmp1_visible).as_dict(), "thresholds": visible_fraction_thresholds(cmp1_visible)},
            "improvement_cmp1_minus_cmp0": summarize(improvements).as_dict(),
        },
        "subject_center_inside": {
            "cmp0_rate": round(sum(1 for r in primary_records if r["cmp0"]["subject_center_inside"]) / len(primary_records), 6)
            if primary_records
            else None,
            "cmp1_rate": round(sum(1 for r in primary_records if r["cmp1"]["subject_center_inside"]) / len(primary_records), 6)
            if primary_records
            else None,
        },
        "strata": strata_payload,
        "crop_shift": {
            "cmp0_shift_x": summarize(r["cmp0"]["shift_x_norm"] for r in records).as_dict(),
            "cmp1_shift_x": summarize(r["cmp1"]["shift_x_norm"] for r in records).as_dict(),
            "cmp1_shift_y": summarize(r["cmp1"]["shift_y_norm"] for r in records).as_dict(),
        },
        "fallback": {
            "count": placement_counts.get(PLACEMENT_FALLBACK_CENTER_CROP, 0),
            "stage5_2_fallback_reason_counts": dict(sorted(fallback_reasons.items())),
        },
        "selected_primary_overflow_audit": {
            "scope": "frozen Stage 5.2 formal policy artifact frames (selected primary bboxes only)",
            "frames_with_primary_bbox": len(overflow_frames),
            "frames_with_overflow_gt0": sum(1 for value in overflow_frames if value > 1e-9),
            "thresholds": overflow_threshold_frames(overflow_frames),
            "max_overflow_distribution": overflow_distribution(overflow_frames),
            "per_direction_distribution": {
                direction: overflow_distribution(values) for direction, values in sorted(direction_overflow.items())
            },
            "sanitized_invalid_count": sum(1 for r in records if r["sanitized"]["status"] == SANITIZE_INVALID),
        },
        "weak_reference_diagnostic": {
            "tag": "WEAK_REFERENCE_DIAGNOSTIC / NOT OFFICIAL GT / NOT FOR MODEL SELECTION",
            "frames_with_reference": len(weak_rows),
            "iou_cmp0": summarize(r["weak_reference"]["iou_cmp0"] for r in weak_rows).as_dict(),
            "iou_cmp1": summarize(r["weak_reference"]["iou_cmp1"] for r in weak_rows).as_dict(),
            "paired_cmp1_vs_cmp0": {"wins": weak_wins, "ties": weak_ties, "losses": weak_losses},
        },
    }


def mirror_reliable_candidate(
    candidates: Sequence[Mapping[str, Any]],
    config: SubjectPolicyConfig,
) -> tuple[SubjectCandidate | None, int]:
    """Diagnostic mirror of the frozen policy ranking (read-only threshold reuse).

    Returns the candidate the frozen Stage 5.2 policy would rank first among
    reliable detections (with person priority), plus the invalid-candidate count.
    Used ONLY for FULL_FRAME_SUBJECT_DIAGNOSTIC and multi-subject diagnostics;
    frozen Stage 5.2 decisions are never modified.
    """
    pool = [
        (index, SubjectCandidate(tuple(float(v) for v in item["box_xyxy"]), float(item["score"]), int(item["label_id"]), str(item["label"])))
        for index, item in enumerate(candidates)
        if candidate_is_valid(tuple(float(v) for v in item["box_xyxy"]))
    ]
    invalid_count = len(candidates) - len(pool)
    reliable = [(index, candidate) for index, candidate in pool if candidate.score >= config.reliable_score]
    if config.person_priority:
        persons = [(index, candidate) for index, candidate in reliable if candidate.label == config.person_label]
        if persons:
            reliable = persons
    if not reliable:
        return None, invalid_count
    reliable.sort(key=lambda item: (-item[1].score, -item[1].area(), item[0]))
    return reliable[0][1], invalid_count


def crosscheck_mirror_against_artifact(
    inputs: FrozenInputs,
    policy_config: SubjectPolicyConfig,
) -> dict[str, int]:
    """Every frozen PRIMARY decision must be reproduced by the diagnostic mirror."""
    mismatches = 0
    compared = 0
    for record in inputs.policy_records:
        if record["status"] != STATUS_PRIMARY:
            continue
        raw_frame = inputs.raw_frames[(record["video_id"], int(record["frame"]))]
        mirrored, _ = mirror_reliable_candidate(raw_frame["candidates"], policy_config)
        compared += 1
        stored = record["primary"]
        if (
            mirrored is None
            or mirrored.label != stored["label"]
            or mirrored.score != stored["score"]
            or list(mirrored.box) != [float(value) for value in stored["xyxy"]]
        ):
            mismatches += 1
    return {"compared": compared, "mismatches": mismatches}


def _counter(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def box_too_large_diagnostic(
    inputs: FrozenInputs,
    target_ratio: Sequence[int | float],
    policy_config: SubjectPolicyConfig,
) -> dict[str, Any]:
    """FULL_FRAME_SUBJECT_DIAGNOSTIC: observe BOX_TOO_LARGE frames without reclassifying them."""
    rows: list[dict[str, Any]] = []
    for record in inputs.policy_records:
        reasons = list(record.get("fallback_reasons", []))
        if "BOX_TOO_LARGE" not in reasons:
            continue
        video_id = record["video_id"]
        frame = int(record["frame"])
        meta = inputs.metadata[video_id]
        width, height = int(meta["width"]), int(meta["height"])
        raw_frame = inputs.raw_frames[(video_id, frame)]
        trigger, _ = mirror_reliable_candidate(raw_frame["candidates"], policy_config)
        if trigger is None:
            rows.append({"video_id": video_id, "frame": frame, "trigger_candidate_found": False})
            continue
        raw_bbox = list(trigger.box)
        sanitized = sanitize_primary_bbox(raw_bbox, width, height)
        tw, th = float(target_ratio[0]), float(target_ratio[1])
        center = compute_center_crop(width, height, tw, th)
        if sanitized.valid:
            shifted = compute_subject_shifted_crop(width, height, tw, th, (sanitized.center_x, sanitized.center_y))
            crop_rect = crop_rect_from_xywh(shifted.x, shifted.y, shifted.w, _fraction_to_float(shifted.h))
            visible = subject_visible_fraction(sanitized, crop_rect)
            center_inside = subject_center_inside_crop(sanitized, crop_rect)
        else:
            shifted = None
            visible = None
            center_inside = False
        overflow = raw_bbox_overflow(raw_bbox, width, height)
        rows.append(
            {
                "video_id": video_id,
                "frame": frame,
                "trigger_candidate_found": True,
                "class": trigger.label,
                "score": trigger.score,
                "raw_bbox_xyxy": raw_bbox,
                "raw_area_fraction": round(trigger.area() / float(width * height), 6),
                "clamped_area_fraction": round(sanitized.area / float(width * height), 6) if sanitized.valid else None,
                "center_x_norm": round(sanitized.center_x / width, 6),
                "center_y_norm": round(sanitized.center_y / height, 6),
                "overflow_px": {key: round(value, 3) for key, value in overflow.items()},
                "sanitized_status": sanitized.status,
                "hypothetical_cmp1_visible_fraction": _round_or_none(visible),
                "hypothetical_cmp1_center_inside": center_inside,
                "cmp0_crop_w": center.w,
                "cmp1_crop_w": shifted.w if shifted else None,
            }
        )
    valid_rows = [row for row in rows if row.get("hypothetical_cmp1_visible_fraction") is not None]
    return {
        "tag": "FULL_FRAME_SUBJECT_DIAGNOSTIC / observation only / Stage 5.2 decisions NOT modified",
        "box_too_large_frames": len(rows),
        "trigger_candidate_found": sum(1 for row in rows if row.get("trigger_candidate_found")),
        "class_counts": dict(sorted(_counter(row.get("class") for row in rows).items())),
        "hypothetical_cmp1_visible": summarize(
            row["hypothetical_cmp1_visible_fraction"] for row in valid_rows
        ).as_dict(),
        "hypothetical_cmp1_visible_thresholds": visible_fraction_thresholds(
            row["hypothetical_cmp1_visible_fraction"] for row in valid_rows
        ),
        "rows": rows,
    }


def multi_subject_diagnostic(
    inputs: FrozenInputs,
    target_ratio: Sequence[int | float],
    policy_config: SubjectPolicyConfig,
) -> dict[str, Any]:
    """Ambiguous-frame observation only: no union boxes, no fusion, no new policy."""
    rows: list[dict[str, Any]] = []
    for record in inputs.policy_records:
        if not record.get("ambiguous"):
            continue
        video_id = record["video_id"]
        frame = int(record["frame"])
        meta = inputs.metadata[video_id]
        width, height = int(meta["width"]), int(meta["height"])
        raw_frame = inputs.raw_frames[(video_id, frame)]
        primary, _ = mirror_reliable_candidate(raw_frame["candidates"], policy_config)
        if primary is None:
            continue
        contenders = [
            candidate
            for candidate in (
                SubjectCandidate(
                    tuple(float(v) for v in item["box_xyxy"]),
                    float(item["score"]),
                    int(item["label_id"]),
                    str(item["label"]),
                )
                for item in raw_frame["candidates"]
            )
            if candidate_is_valid(candidate.box)
            and candidate.score >= primary.score - policy_config.ambiguity_score_gap
            and detection_iou(primary, candidate) < policy_config.ambiguity_iou_threshold
            and list(candidate.box) != list(primary.box)
        ]
        sanitized = sanitize_primary_bbox(list(primary.box), width, height)
        tw, th = float(target_ratio[0]), float(target_ratio[1])
        center = compute_center_crop(width, height, tw, th)
        cmp0_rect = crop_rect_from_xywh(center.x, center.y, center.w, _fraction_to_float(derived_height(center.w, tw, th)))
        if sanitized.valid:
            shifted = compute_subject_shifted_crop(width, height, tw, th, (sanitized.center_x, sanitized.center_y))
            cmp1_rect = crop_rect_from_xywh(shifted.x, shifted.y, shifted.w, _fraction_to_float(shifted.h))
        else:
            cmp1_rect = cmp0_rect
        secondary_centers = []
        for candidate in contenders:
            center_x = (candidate.box[0] + candidate.box[2]) / 2
            center_y = (candidate.box[1] + candidate.box[3]) / 2
            secondary_centers.append(
                {
                    "label": candidate.label,
                    "center_x_norm": round(center_x / width, 6),
                    "center_y_norm": round(center_y / height, 6),
                    "inside_cmp0_crop": cmp0_rect[0] <= center_x < cmp0_rect[2]
                    and cmp0_rect[1] <= center_y < cmp0_rect[3],
                    "inside_cmp1_crop": cmp1_rect[0] <= center_x < cmp1_rect[2]
                    and cmp1_rect[1] <= center_y < cmp1_rect[3],
                }
            )
        secondary_labels = {item["label"] for item in secondary_centers}
        if primary.label == "person":
            composition = "person+person" if secondary_labels <= {"person"} else "person+other"
        else:
            composition = "nonperson_multi"
        rows.append(
            {
                "video_id": video_id,
                "frame": frame,
                "primary_class": primary.label,
                "primary_score": primary.score,
                "composition": composition,
                "stored_ambiguous_candidate_count": int(record.get("ambiguous_candidate_count", 0)),
                "secondary_count": len(secondary_centers),
                "secondary_centers": secondary_centers,
                "secondary_centers_inside_cmp0_crop": sum(1 for item in secondary_centers if item["inside_cmp0_crop"]),
                "secondary_centers_inside_cmp1_crop": sum(1 for item in secondary_centers if item["inside_cmp1_crop"]),
                "primary_secondary_span_x_norm": round(
                    (
                        max([primary.box[2]] + [c.box[2] for c in contenders])
                        - min([primary.box[0]] + [c.box[0] for c in contenders])
                    )
                    / width,
                    6,
                )
                if contenders
                else None,
            }
        )
    secondary_counts = [row["secondary_count"] for row in rows]

    def ratio(rows_subset, key, condition) -> float | None:
        relevant = [row for row in rows_subset if row["secondary_count"]]
        if not relevant:
            return None
        return round(sum(1 for row in relevant if condition(row)) / len(relevant), 6)

    composition_classes = sorted({row["composition"] for row in rows})
    composition_summary = {}
    for composition_class in composition_classes:
        subset = [row for row in rows if row["composition"] == composition_class]
        composition_summary[composition_class] = {
            "n": len(subset),
            "secondary_count": summarize(row["secondary_count"] for row in subset).as_dict(),
            "all_secondaries_inside_cmp0_rate": ratio(subset, "inside_cmp0_crop", lambda row: row["secondary_centers_inside_cmp0_crop"] == row["secondary_count"]),
            "all_secondaries_inside_cmp1_rate": ratio(subset, "inside_cmp1_crop", lambda row: row["secondary_centers_inside_cmp1_crop"] == row["secondary_count"]),
            "at_least_one_inside_cmp0_rate": ratio(subset, "inside_cmp0_crop", lambda row: row["secondary_centers_inside_cmp0_crop"] > 0),
            "at_least_one_inside_cmp1_rate": ratio(subset, "inside_cmp1_crop", lambda row: row["secondary_centers_inside_cmp1_crop"] > 0),
        }
    return {
        "tag": "MULTI_SUBJECT_DIAGNOSTIC / observation only / no union or fusion policy",
        "ambiguous_frames": len(rows),
        "primary_class_counts": dict(sorted(_counter(row["primary_class"] for row in rows).items())),
        "composition_summary": composition_summary,
        "secondary_count": summarize(secondary_counts).as_dict(),
        "secondary_inside_cmp0_crop": summarize(row["secondary_centers_inside_cmp0_crop"] for row in rows).as_dict(),
        "secondary_inside_cmp1_crop": summarize(row["secondary_centers_inside_cmp1_crop"] for row in rows).as_dict(),
        "all_secondaries_inside_cmp0_frames": sum(
            1 for row in rows if row["secondary_count"] and row["secondary_centers_inside_cmp0_crop"] == row["secondary_count"]
        ),
        "all_secondaries_inside_cmp1_frames": sum(
            1 for row in rows if row["secondary_count"] and row["secondary_centers_inside_cmp1_crop"] == row["secondary_count"]
        ),
        "at_least_one_secondary_inside_cmp1_frames": sum(1 for row in rows if row["secondary_centers_inside_cmp1_crop"] > 0),
        "rows": rows,
    }


def mirror_best_possible_candidate(
    candidates: Sequence[Mapping[str, Any]],
    config: SubjectPolicyConfig,
) -> SubjectCandidate | None:
    """Best possible-tier candidate (possible_score <= score < reliable_score) for LOW_CONFIDENCE diagnostics."""
    pool = [
        SubjectCandidate(tuple(float(v) for v in item["box_xyxy"]), float(item["score"]), int(item["label_id"]), str(item["label"]))
        for item in candidates
        if candidate_is_valid(tuple(float(v) for v in item["box_xyxy"]))
    ]
    possible = [candidate for candidate in pool if config.possible_score <= candidate.score < config.reliable_score]
    if not possible:
        return None
    possible.sort(key=lambda candidate: (-candidate.score, -candidate.area()))
    return possible[0]


def fallback_reason_diagnostic(
    inputs: FrozenInputs,
    target_ratio: Sequence[int | float],
    policy_config: SubjectPolicyConfig,
) -> dict[str, Any]:
    """Per-reason observation of frozen Stage 5.2 fallbacks: count/ratio/class/area.

    Observation only: no zoom, no SAM, no reclassification, no new fallback policy.
    """
    total = len(inputs.policy_records)
    reasons: dict[str, dict[str, Any]] = {}
    for record in inputs.policy_records:
        frame_reasons = list(record.get("fallback_reasons", []))
        if record["status"] == STATUS_PRIMARY or not frame_reasons:
            continue
        video_id = record["video_id"]
        frame = int(record["frame"])
        meta = inputs.metadata[video_id]
        width, height = int(meta["width"]), int(meta["height"])
        raw_frame = inputs.raw_frames[(video_id, frame)]
        for reason in frame_reasons:
            bucket = reasons.setdefault(
                reason,
                {"count": 0, "class_counts": {}, "area_ratios": [], "trigger_found": 0, "rows": []},
            )
            bucket["count"] += 1
            if reason == REASON_NO_DETECTION:
                trigger = None
            elif reason == REASON_LOW_CONFIDENCE:
                trigger = mirror_best_possible_candidate(raw_frame["candidates"], policy_config)
            else:
                trigger, _ = mirror_reliable_candidate(raw_frame["candidates"], policy_config)
            if trigger is None:
                bucket["rows"].append({"video_id": video_id, "frame": frame, "trigger_found": False})
                continue
            bucket["trigger_found"] += 1
            bucket["class_counts"][trigger.label] = bucket["class_counts"].get(trigger.label, 0) + 1
            area_ratio = round(trigger.area() / float(width * height), 6)
            bucket["area_ratios"].append(area_ratio)
            bucket["rows"].append(
                {
                    "video_id": video_id,
                    "frame": frame,
                    "trigger_found": True,
                    "class": trigger.label,
                    "score": trigger.score,
                    "bbox_area_ratio": area_ratio,
                }
            )
    summary: dict[str, Any] = {}
    for reason, bucket in sorted(reasons.items()):
        summary[reason] = {
            "count": bucket["count"],
            "ratio_of_all_frames": round(bucket["count"] / total, 6),
            "trigger_candidate_found": bucket["trigger_found"],
            "class_distribution": dict(sorted(bucket["class_counts"].items())),
            "bbox_area_ratio": summarize(bucket["area_ratios"]).as_dict(),
        }
    return {
        "tag": "FALLBACK_REASON_DIAGNOSTIC / observation only / frozen Stage 5.2 semantics untouched",
        "total_frames": total,
        "fallback_frames": sum(bucket["count"] for bucket in reasons.values()),
        "frames_with_multiple_reasons": sum(
            1
            for record in inputs.policy_records
            if record["status"] != STATUS_PRIMARY and len(list(record.get("fallback_reasons", []))) > 1
        ),
        "reasons": summary,
    }
