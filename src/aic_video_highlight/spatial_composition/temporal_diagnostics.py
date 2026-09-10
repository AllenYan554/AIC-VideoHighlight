"""Stage 5.4 temporal stability evaluation over frozen TS-0 (CMP-1) vs TS-1 records.

Pure evaluation of merged per-frame records produced by the Stage 5.4 runner:
displacement / acceleration distributions, spatial quality guardrails, strata
breakdown and the observation-only multi-subject diagnostic. No promotion gate is
derived here in v1: temporal metrics are DESCRIPTIVE ONLY.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping, Sequence

from aic_video_highlight.spatial_composition.composition_metrics import (
    crop_rect_from_xywh,
    summarize,
    visible_fraction_thresholds,
)
from aic_video_highlight.spatial_composition.temporal_smoothing import (
    LARGE_JUMP_THRESHOLDS,
    TemporalObservation,
    acceleration_norm,
    displacement_norm,
    large_jump_ratios,
    temporal_summary,
    transition_pairs,
    transition_triplets,
)

SANITIZE_OK = "OK"


def crop_center_x(crop_x: int, crop_w: int) -> float:
    return float(crop_x) + crop_w / 2.0


def crop_center_y(crop_y: int, derived_h: float) -> float:
    return float(crop_y) + derived_h / 2.0


def observations_from_records(records: Sequence[Mapping[str, Any]]) -> list[TemporalObservation]:
    """TemporalObservation list (ascending frames) from merged records of ONE video.

    The EMA signal is the sanitized primary subject center (the pre-clamp target
    the frozen Stage 5.3 placement consumed). Fallback frames carry the frozen
    center-crop center as a placeholder; it is never EMA-consumed.
    """
    observations = []
    for record in records:
        fallback = bool(record["ts0"]["fallback"])
        if fallback:
            ideal_center_x = crop_center_x(int(record["ts0"]["x"]), int(record["ts0"]["w"]))
            ideal_center_y = crop_center_y(int(record["ts0"]["y"]), float(record["ts0"]["h"]))
        else:
            xyxy = record["sanitized"]["xyxy"]
            ideal_center_x = (float(xyxy[0]) + float(xyxy[2])) / 2.0
            ideal_center_y = (float(xyxy[1]) + float(xyxy[3])) / 2.0
        observations.append(
            TemporalObservation(
                frame=int(record["frame"]),
                fallback=fallback,
                ideal_center_x=ideal_center_x,
                ideal_center_y=ideal_center_y,
            )
        )
    return sorted(observations, key=lambda item: item.frame)


def _usable(record: Mapping[str, Any]) -> bool:
    return not bool(record["ts0"]["fallback"])


def _frame_gap(record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> int:
    return int(record_b["frame"]) - int(record_a["frame"])


def count_reset_steps(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count TS-1 sequence-reset events over all frames (independent of metric pairs).

    By construction, reset transitions are excluded from the paired displacement
    metrics (the previous frame is a fallback or the gap is > 1), so reset events
    are counted directly over the frozen frame sequence.
    """
    steps: dict[str, int] = {}
    for record in records:
        reason = record["ts1"]["reset_reason"]
        if reason is not None:
            steps[str(reason)] = steps.get(str(reason), 0) + 1
    return dict(sorted(steps.items()))


def temporal_stability_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """TS-0 vs TS-1 displacement / acceleration over paired frozen transitions.

    All metrics are computed over the SAME paired transition / triplet sets (frame
    gap == 1, no fallback frames). TS-1 additionally reports the reset steps
    separately; the head-to-head jitter comparison uses only transitions where the
    EMA was actually inherited (``reset_reason is None``) on the TS-1 side.
    """
    ordered = sorted(records, key=lambda record: int(record["frame"]))
    width_by_video: dict[str, int] = {}
    for record in ordered:
        width_by_video.setdefault(str(record["video_id"]), int(record["image_width"]))
    if len(width_by_video) != 1:
        raise ValueError("temporal metrics require records of exactly one video")
    width = next(iter(width_by_video.values()))

    observations = observations_from_records(ordered)
    pair_indexes = transition_pairs(observations)
    triplet_indexes = transition_triplets(observations)

    ts0_displacements: list[float] = []
    ts1_displacements_all: list[float] = []
    ts1_displacements_smoothed: list[float] = []
    for previous_index, current_index in pair_indexes:
        previous_record = ordered[previous_index]
        current_record = ordered[current_index]
        ts0_displacements.append(
            displacement_norm(
                crop_center_x(int(previous_record["ts0"]["x"]), int(previous_record["ts0"]["w"])),
                crop_center_x(int(current_record["ts0"]["x"]), int(current_record["ts0"]["w"])),
                width,
            )
        )
        ts1_displacements_all.append(
            displacement_norm(
                crop_center_x(int(previous_record["ts1"]["x"]), int(previous_record["ts1"]["w"])),
                crop_center_x(int(current_record["ts1"]["x"]), int(current_record["ts1"]["w"])),
                width,
            )
        )
        if current_record["ts1"]["reset_reason"] is None:
            ts1_displacements_smoothed.append(ts1_displacements_all[-1])

    ts0_accelerations: list[float] = []
    ts1_accelerations: list[float] = []
    ts1_accelerations_smoothed: list[float] = []
    for first_index, second_index, third_index in triplet_indexes:
        first_record, second_record, third_record = ordered[first_index], ordered[second_index], ordered[third_index]

        def center_delta(record_a: Mapping[str, Any], record_b: Mapping[str, Any], side: str) -> float:
            center_a = crop_center_x(int(record_a[side]["x"]), int(record_a[side]["w"]))
            center_b = crop_center_x(int(record_b[side]["x"]), int(record_b[side]["w"]))
            return center_b - center_a

        ts0_accelerations.append(
            acceleration_norm(
                center_delta(first_record, second_record, "ts0"),
                center_delta(second_record, third_record, "ts0"),
                width,
            )
        )
        ts1_accelerations.append(
            acceleration_norm(
                center_delta(first_record, second_record, "ts1"),
                center_delta(second_record, third_record, "ts1"),
                width,
            )
        )
        if second_record["ts1"]["reset_reason"] is None and third_record["ts1"]["reset_reason"] is None:
            ts1_accelerations_smoothed.append(ts1_accelerations[-1])

    return {
        "paired_transitions": len(pair_indexes),
        "paired_triplets": len(triplet_indexes),
        "ts0_displacement": {
            **temporal_summary(ts0_displacements),
            "large_jump_ratios_descriptive_only": large_jump_ratios(ts0_displacements, LARGE_JUMP_THRESHOLDS),
        },
        "ts1_displacement": {
            **temporal_summary(ts1_displacements_all),
            "large_jump_ratios_descriptive_only": large_jump_ratios(ts1_displacements_all, LARGE_JUMP_THRESHOLDS),
        },
        "ts1_displacement_smoothed_only": {
            **temporal_summary(ts1_displacements_smoothed),
            "large_jump_ratios_descriptive_only": large_jump_ratios(ts1_displacements_smoothed, LARGE_JUMP_THRESHOLDS),
        },
        "ts0_acceleration": temporal_summary(ts0_accelerations),
        "ts1_acceleration": temporal_summary(ts1_accelerations),
        "ts1_acceleration_smoothed_only": temporal_summary(ts1_accelerations_smoothed),
        "ts1_reset_steps": count_reset_steps(ordered),
        "metric_role": "DESCRIPTIVE_ONLY / no promotion gate in Stage 5.4 v1",
    }


def crop_geometry_valid(
    crop_x: int, crop_w: int, crop_y: int, derived_h: Fraction | float, width: int, height: int
) -> dict[str, bool]:
    derived = Fraction(derived_h)
    return {
        "nonnegative": crop_x >= 0 and crop_y >= 0,
        "width_positive": crop_w > 0,
        "x_within_frame": crop_x + crop_w <= width,
        "derived_height_within_frame": crop_y + derived <= height,
    }


def spatial_guardrail_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stage 5.3 spatial guardrails recomputed for TS-0 vs TS-1."""
    with_subject = [record for record in records if record["sanitized"]["status"] == SANITIZE_OK]
    fallback_frames = [record for record in records if record["sanitized"]["status"] != SANITIZE_OK]

    def side_payload(side: str) -> dict[str, Any]:
        visible = [record[side]["subject_visible_fraction"] for record in with_subject]
        inside = [record[side]["subject_center_inside"] for record in with_subject]
        return {
            **summarize(visible).as_dict(),
            "thresholds": visible_fraction_thresholds(visible),
            "subject_center_inside_rate": round(sum(1 for value in inside if value) / len(inside), 6)
            if inside
            else None,
        }

    strata: dict[str, dict[str, list[float]]] = {}
    stratum_sizes: dict[str, int] = {}
    for record in records:
        stratum = str(record["stratum"])
        stratum_sizes[stratum] = stratum_sizes.get(stratum, 0) + 1
        bucket = strata.setdefault(stratum, {"ts0": [], "ts1": []})
        if record["sanitized"]["status"] == SANITIZE_OK:
            bucket["ts0"].append(record["ts0"]["subject_visible_fraction"])
            bucket["ts1"].append(record["ts1"]["subject_visible_fraction"])
    strata_payload = {
        stratum: {
            "n": stratum_sizes[stratum],
            "with_subject": len(payload["ts0"]),
            "ts0_visible": summarize(payload["ts0"]).as_dict(),
            "ts1_visible": summarize(payload["ts1"]).as_dict(),
            "delta_mean": _delta_mean(payload["ts0"], payload["ts1"]),
        }
        for stratum, payload in sorted(strata.items())
    }
    return {
        "frames": len(records),
        "frames_with_subject": len(with_subject),
        "fallback_frames": len(fallback_frames),
        "ts0": side_payload("ts0"),
        "ts1": side_payload("ts1"),
        "strata": strata_payload,
    }


def _delta_mean(ts0: Sequence[float], ts1: Sequence[float]) -> float | None:
    if not ts0:
        return None
    return round(sum(ts1) / len(ts1) - sum(ts0) / len(ts0), 6)


def temporal_multi_subject_diagnostic(
    inputs: Any,
    records: Sequence[Mapping[str, Any]],
    target_ratio: Sequence[int | float],
    policy_config: Any,
) -> dict[str, Any]:
    """Observation-only multi-subject diagnostic for TS-0 vs TS-1 crops.

    Mirrors the frozen Stage 5.3 diagnostic semantics: ambiguous frames only,
    secondary containment compared between the TS-0 (CMP-1) and TS-1 crops. No
    union boxes, no fusion, no new policy.
    """
    from aic_video_highlight.spatial_composition.composition_pipeline import (
        mirror_reliable_candidate,
    )
    from aic_video_highlight.spatial_localization.subject_localization import (
        SubjectCandidate,
        candidate_is_valid,
        detection_iou,
    )

    by_key = {(str(record["video_id"]), int(record["frame"])): record for record in records}
    rows: list[dict[str, Any]] = []
    for record in inputs.policy_records:
        if not record.get("ambiguous"):
            continue
        video_id = str(record["video_id"])
        frame = int(record["frame"])
        merged = by_key.get((video_id, frame))
        if merged is None:
            continue
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
                    tuple(float(value) for value in item["box_xyxy"]),
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
        ts0_rect = crop_rect_from_xywh(
            int(merged["ts0"]["x"]), int(merged["ts0"]["y"]), int(merged["ts0"]["w"]), float(merged["ts0"]["h"])
        )
        ts1_rect = crop_rect_from_xywh(
            int(merged["ts1"]["x"]), int(merged["ts1"]["y"]), int(merged["ts1"]["w"]), float(merged["ts1"]["h"])
        )
        primary_center_x = (primary.box[0] + primary.box[2]) / 2
        primary_center_y = (primary.box[1] + primary.box[3]) / 2
        secondary_centers = []
        for candidate in contenders:
            center_x = (candidate.box[0] + candidate.box[2]) / 2
            center_y = (candidate.box[1] + candidate.box[3]) / 2
            secondary_centers.append(
                {
                    "label": candidate.label,
                    "center_x_norm": round(center_x / width, 6),
                    "center_y_norm": round(center_y / height, 6),
                    "inside_ts0_crop": ts0_rect[0] <= center_x < ts0_rect[2]
                    and ts0_rect[1] <= center_y < ts0_rect[3],
                    "inside_ts1_crop": ts1_rect[0] <= center_x < ts1_rect[2]
                    and ts1_rect[1] <= center_y < ts1_rect[3],
                }
            )
        rows.append(
            {
                "video_id": video_id,
                "frame": frame,
                "primary_class": primary.label,
                "primary_center_inside_ts0_crop": ts0_rect[0] <= primary_center_x < ts0_rect[2]
                and ts0_rect[1] <= primary_center_y < ts0_rect[3],
                "primary_center_inside_ts1_crop": ts1_rect[0] <= primary_center_x < ts1_rect[2]
                and ts1_rect[1] <= primary_center_y < ts1_rect[3],
                "secondary_count": len(secondary_centers),
                "secondary_centers_inside_ts0_crop": sum(
                    1 for item in secondary_centers if item["inside_ts0_crop"]
                ),
                "secondary_centers_inside_ts1_crop": sum(
                    1 for item in secondary_centers if item["inside_ts1_crop"]
                ),
            }
        )
    return summarize_multi_subject_rows(rows, candidate_side="ts1")


def summarize_multi_subject_rows(
    rows: Sequence[Mapping[str, Any]], *, candidate_side: str
) -> dict[str, Any]:
    """Summarize TS-0 versus one explicitly named temporal treatment side."""
    if candidate_side not in {"ts1", "ts2", "ts3"}:
        raise ValueError(f"unknown multi-subject candidate side: {candidate_side}")
    candidate_label = f"TS-{candidate_side[2:]}"
    rows = list(rows)
    required_fields = {
        "secondary_count",
        "primary_center_inside_ts0_crop",
        f"primary_center_inside_{candidate_side}_crop",
        "secondary_centers_inside_ts0_crop",
        f"secondary_centers_inside_{candidate_side}_crop",
    }
    for index, row in enumerate(rows):
        missing = sorted(required_fields - row.keys())
        if missing:
            raise ValueError(
                f"missing multi-subject row fields at index {index}: {', '.join(missing)}"
            )
    relevant = [row for row in rows if row["secondary_count"]]

    def primary_and_all(side: str) -> float | None:
        if not relevant:
            return None
        hits = sum(
            1
            for row in relevant
            if row[f"primary_center_inside_{side}_crop"]
            and row[f"secondary_centers_inside_{side}_crop"] == row["secondary_count"]
        )
        return round(hits / len(relevant), 6)

    return {
        "tag": (
            "MULTI_SUBJECT_DIAGNOSTIC / observation only / TS-0 (CMP-1) vs "
            f"{candidate_label} / no union or fusion policy"
        ),
        "ambiguous_frames": len(rows),
        "frames_with_secondaries": len(relevant),
        "all_secondaries_inside_ts0_rate": _rate(
            relevant, "secondary_centers_inside_ts0_crop"
        ),
        f"all_secondaries_inside_{candidate_side}_rate": _rate(
            relevant, f"secondary_centers_inside_{candidate_side}_crop"
        ),
        "at_least_one_inside_ts0_rate": _rate(relevant, "secondary_centers_inside_ts0_crop", at_least_one=True),
        f"at_least_one_inside_{candidate_side}_rate": _rate(
            relevant, f"secondary_centers_inside_{candidate_side}_crop", at_least_one=True
        ),
        "primary_and_all_secondaries_inside_ts0_rate": primary_and_all("ts0"),
        f"primary_and_all_secondaries_inside_{candidate_side}_rate": primary_and_all(
            candidate_side
        ),
        "rows": rows,
    }


def _rate(rows: Sequence[Mapping[str, Any]], key: str, at_least_one: bool = False) -> float | None:
    if not rows:
        return None
    if at_least_one:
        hits = sum(1 for row in rows if row[key] > 0)
    else:
        hits = sum(1 for row in rows if row[key] == row["secondary_count"])
    return round(hits / len(rows), 6)
