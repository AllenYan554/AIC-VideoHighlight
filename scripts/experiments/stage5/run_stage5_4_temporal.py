#!/usr/bin/env python3
"""Stage 5.4 Temporal Composition Stabilization runner (TS-0 through TS-5).

Config-driven runner for Stage 5.4 Smoke/Formal/Amendment configs: consumes the frozen Stage 5.3 CMP-1
composition as TS-0 (temporal control baseline), applies the deterministic TS-1
EMA crop-center smoothing (development baseline alpha = 0.5), and evaluates
temporal stability metrics, Stage 5.3 spatial guardrails, strata and the
observation-only multi-subject diagnostic. Amendment modes add TS-2 and TS-3
while keeping the earlier record contracts intact. CPU-only, frozen-artifact-only,
no RT-DETR, no Qwen, no SAM, no tracking, no Heldout access.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from aic_video_highlight.experiment_runtime.artifacts import build_artifact_manifest
from aic_video_highlight.experiment_runtime.hashing import canonical_sha256, file_sha256
from aic_video_highlight.experiment_runtime.io import atomic_write_json
from aic_video_highlight.experiment_runtime.paths import EnvironmentPaths
from aic_video_highlight.experiment_runtime.progress import ProgressReporter
from aic_video_highlight.experiment_runtime.promotion import (
    adjudicate_ts5_formal,
    build_promotion_marker,
    evaluate_smoke_promotion,
)
from aic_video_highlight.experiment_runtime.raw_report import render_raw_report, write_ai_report_inputs
from aic_video_highlight.experiment_runtime.run_context import RunContext, RunIdentityMismatch
from aic_video_highlight.experiment_runtime.shards import ShardStore
from aic_video_highlight.spatial_composition.center_crop import derived_height
from aic_video_highlight.spatial_composition.composition_metrics import (
    crop_rect_from_xywh,
    subject_center_inside_crop,
    subject_visible_fraction,
)
from aic_video_highlight.spatial_composition.composition_pipeline import (
    FrozenInputError,
    InputBinding,
    compose_frame,
    load_frozen_inputs,
    load_raw_shard_dir_subset,
    verify_input_bindings,
)
from aic_video_highlight.spatial_composition.subject_shifted_crop import (
    PLACEMENT_FALLBACK_CENTER_CROP,
    SANITIZE_OK,
    SanitizedSubject,
)
from aic_video_highlight.spatial_composition.temporal_diagnostics import (
    crop_geometry_valid,
    observations_from_records,
    spatial_guardrail_metrics,
    summarize_multi_subject_rows,
    temporal_multi_subject_diagnostic,
    temporal_stability_metrics,
)
from aic_video_highlight.spatial_composition.temporal_smoothing import (
    DEFAULT_EMA_ALPHA,
    large_jump_ratios,
    smooth_video_sequence,
    smooth_video_sequence_adaptive,
    smooth_video_sequence_guarded,
    smooth_video_sequence_bbox_guarded,
    smooth_video_sequence_projected_state_bbox_guarded,
    temporal_summary,
)
from aic_video_highlight.spatial_localization.subject_localization import SubjectPolicyConfig

BASE_FIELDS = {
    "archive": "archive",
    "outputs": "outputs",
    "datasets": "datasets",
    "repo": "repo",
    "models": "models",
}

MULTI_SUBJECT_COMPARISON_CANDIDATES = {
    "ts0_vs_ts1": "ts1",
    "ts0_vs_ts2": "ts2",
    "ts0_vs_ts3": "ts3",
    "ts0_vs_ts4": "ts4",
    "ts0_vs_ts5": "ts5",
}

TREATMENT_SIDES = ("ts1", "ts2", "ts3", "ts4", "ts5")

# Experiment ids that consume the Stage 5.3 frozen Full-Dev manifest and apply
# the preregistered Formal contract (Full Dev166 + Confirmatory Dev142).
FORMAL_EXPERIMENT_IDS = (
    "stage5_4_formal",
    "stage5_4_amendment2_formal",
    "stage5_4_amendment3_formal",
    "stage5_4_amendment4_formal",
)
AMENDMENT_SMOKE_EXPERIMENT_IDS = (
    "stage5_4_amendment_smoke",
    "stage5_4_amendment2_smoke",
    "stage5_4_amendment3_smoke",
    "stage5_4_amendment4_smoke",
)

AMENDMENT3_FORMAL_REPORT_SECTIONS = (
    "History", "Method", "TS-0~TS-4", "Smoke result", "Full Dev166", "Dev142",
    "Temporal", "Spatial", "Strata", "BBox diagnostics", "Guard diagnostics",
    "Multi-subject", "Fallback", "Engineering gates", "Scientific gates",
    "Mechanism checks", "Limitations", "Conclusion", "Artifact paths",
    "Git/Protocol/Config identities",
)

AMENDMENT4_REPORT_SECTIONS = (
    "Identity", "Method", "TS-0~TS-5", "Smoke promotion", "Full Dev166",
    "Dev142", "Temporal gates", "Spatial gates", "Mechanism checks",
    "Diagnostic-only attribution", "TS-4 comparison", "Determinism",
    "Engineering validation", "Final adjudication", "Artifact paths",
    "Git/Protocol/Config identities",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def resolve_bindings(config: dict, environment: EnvironmentPaths) -> list[InputBinding]:
    bindings = []
    for name, entry in config["inputs"].items():
        base = getattr(environment, BASE_FIELDS[entry["base"]])
        path = Path(entry["path"])
        if not path.is_absolute():
            path = base / path
        bindings.append(
            InputBinding(
                name=name,
                path=path,
                sha256=str(entry.get("sha256", "") or ""),
                format=entry.get("format", "json"),
            )
        )
    manifest_input = config.get("manifest", {}).get("input_name", "stage5_4_smoke_manifest")
    required = {
        manifest_input,
        "stage5_1_predictions",
        "video_metadata_cache",
        "dev166_index",
        "stage5_2_policy_artifact",
        "stage5_2_raw_detector",
        "weak_spatial_reference",
    }
    missing = required - {binding.name for binding in bindings}
    if missing:
        raise FrozenInputError(f"config inputs missing: {sorted(missing)}")
    return bindings


def policy_config_from(config: dict) -> SubjectPolicyConfig:
    thresholds = config["composition"]["stage5_2_policy_thresholds"]
    return SubjectPolicyConfig(
        reliable_score=thresholds["reliable_score"],
        possible_score=thresholds["possible_score"],
        ambiguity_score_gap=thresholds["ambiguity_score_gap"],
        ambiguity_iou_threshold=thresholds["ambiguity_iou_threshold"],
        min_area_fraction=thresholds["min_area_fraction"],
        full_frame_width_fraction=thresholds["full_frame_width_fraction"],
        full_frame_height_fraction=thresholds["full_frame_height_fraction"],
        full_frame_area_fraction=thresholds["full_frame_area_fraction"],
        person_priority=True,
        target_ratio=tuple(config["composition"]["target_ratio"]),
    )


def load_frozen_manifest(config: dict, bindings: list[InputBinding]) -> dict:
    """Load and canonical-content-verify the configured frozen frame manifest."""
    expected_sha = config["manifest"].get("expected_manifest_sha256")
    if not expected_sha:
        raise FrozenInputError(
            f"{config['experiment_id']} frame manifest SHA is not pinned"
        )
    input_name = config.get("manifest", {}).get("input_name", "stage5_4_smoke_manifest")
    binding = next(b for b in bindings if b.name == input_name)
    if not binding.path.is_file():
        raise FrozenInputError(f"frozen manifest missing: {binding.path}")
    manifest = json.loads(binding.path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != expected_sha:
        raise FrozenInputError("frozen manifest internal sha mismatch")
    content_sha = canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if content_sha != expected_sha:
        raise FrozenInputError(
            f"frozen manifest sha mismatch: expected {expected_sha}, got canonical content {content_sha}"
        )
    return manifest


def load_manifest_binding(spec: dict, bindings: list[InputBinding]) -> dict:
    """Load a secondary identity manifest with the same canonical SHA contract."""
    input_name = spec["input_name"]
    binding = next(b for b in bindings if b.name == input_name)
    if not binding.path.is_file():
        raise FrozenInputError(f"frozen manifest missing: {binding.path}")
    manifest = json.loads(binding.path.read_text(encoding="utf-8"))
    expected = spec["expected_manifest_sha256"]
    actual = canonical_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    if manifest.get("manifest_sha256") != expected or actual != expected:
        raise FrozenInputError(
            f"{input_name} canonical identity mismatch: expected {expected}, got {actual}"
        )
    return manifest


def crosscheck_manifest_entry(manifest_entry: dict, composed: dict) -> list[str]:
    """The recomputed frozen TS-0 must reproduce the frozen manifest model-blind fields."""
    mismatches: list[str] = []
    key = f"{composed['video_id']}:{composed['frame']}"
    if manifest_entry["stratum"] != composed["stratum"]:
        mismatches.append(f"{key} stratum")
    if "stage5_2_status" in manifest_entry and manifest_entry["stage5_2_status"] != composed["stage5_2_status"]:
        mismatches.append(f"{key} stage5_2_status")
    if "cmp1_fallback" in manifest_entry and bool(manifest_entry["cmp1_fallback"]) != bool(composed["cmp1"]["fallback"]):
        mismatches.append(f"{key} fallback")
    if "ambiguous" in manifest_entry and bool(manifest_entry["ambiguous"]) != bool(composed["ambiguous"]):
        mismatches.append(f"{key} ambiguous")
    if manifest_entry["horizontal_center_offset"] != composed["horizontal_center_offset"]:
        mismatches.append(f"{key} offset")
    if "subject_center_x" not in manifest_entry or "subject_center_y" not in manifest_entry:
        return mismatches
    if manifest_entry["subject_center_x"] is None or manifest_entry["subject_center_y"] is None:
        # The model-blind builder stores a null subject center exactly on fallback
        # frames (no valid sanitized subject); coordinates are not comparable there.
        if not bool(composed["cmp1"]["fallback"]):
            mismatches.append(f"{key} subject_center_null_vs_nonfallback")
        return mismatches
    xyxy = composed["sanitized"]["xyxy"]
    recomputed_center_x = round((float(xyxy[0]) + float(xyxy[2])) / 2.0, 6)
    recomputed_center_y = round((float(xyxy[1]) + float(xyxy[3])) / 2.0, 6)
    if manifest_entry["subject_center_x"] != recomputed_center_x:
        mismatches.append(f"{key} center_x")
    if manifest_entry["subject_center_y"] != recomputed_center_y:
        mismatches.append(f"{key} center_y")
    return mismatches


def build_video_records(
    video: dict,
    inputs,
    target_ratio: list[float],
    strata_thresholds: dict,
    alpha: float,
    tw: float,
    th: float,
    frozen_ts0_by_frame: dict[int, list[int]] | None = None,
    include_ts2: bool = False,
    include_ts3: bool = False,
    include_ts4: bool = False,
    include_ts5: bool = False,
) -> tuple[list[dict], list[str]]:
    """TS-0 plus optional TS-1/2/3/4/5 treatments for one frozen video."""
    manifest_by_frame = {int(entry["frame"]): entry for entry in video["frames"]}
    frozen_boxes = {
        int(pred["frame"]): list(pred["bboxes"])
        for pred in inputs.stage5_1_predictions[video["video_id"]]["predictions"]
    }
    composed_records: list[dict] = []
    mismatches: list[str] = []
    for entry in sorted(video["frames"], key=lambda item: int(item["frame"])):
        composed = compose_frame(
            video["video_id"],
            int(entry["frame"]),
            inputs,
            target_ratio,
            strata_thresholds,
            frozen_boxes,
        )
        frame_mismatches = crosscheck_manifest_entry(entry, composed)
        if frozen_ts0_by_frame is not None:
            expected_bbox = frozen_ts0_by_frame.get(int(entry["frame"]))
            actual_bbox = [
                int(composed["cmp1"]["x"]),
                int(composed["cmp1"]["y"]),
                int(composed["cmp1"]["w"]),
            ]
            if expected_bbox != actual_bbox:
                frame_mismatches.append(f"{video['video_id']}:{entry['frame']} ts0_crop")
        mismatches.extend(frame_mismatches)
        # TS-0 is the Stage 5.3 FINAL FROZEN CMP-1; its regression evidence is the
        # per-frame cross-check against the SHA-pinned manifest identity (the
        # frozen CMP-1 must reproduce it exactly, model-blind).
        composed["ts0_frozen_regression"] = bool(frame_mismatches)
        composed_records.append(composed)

    # observations_from_records consumes the merged record schema; the TS-1 side
    # does not exist yet at this point, so expose the frozen TS-0 (CMP-1) under
    # the "ts0" key contract expected by the diagnostics module.
    observations = observations_from_records(
        [
            {"frame": composed["frame"], "ts0": composed["cmp1"], "sanitized": composed["sanitized"]}
            for composed in composed_records
        ]
    )
    width, height = int(video["image_width"]), int(video["image_height"])
    smoothed = smooth_video_sequence(width, height, tw, th, observations, alpha=alpha)
    smoothed_by_frame = {item.frame: item for item in smoothed}
    adaptive_by_frame = (
        {item.frame: item for item in smooth_video_sequence_adaptive(width, height, tw, th, observations)}
        if include_ts2
        else {}
    )
    guarded_by_frame = (
        {
            item.frame: item
            for item in smooth_video_sequence_guarded(
                width, height, tw, th, observations, alpha=alpha
            )
        }
        if include_ts3
        else {}
    )
    primary_bboxes = {
        int(composed["frame"]): tuple(float(value) for value in composed["sanitized"]["xyxy"])
        for composed in composed_records
        if not composed["cmp1"]["fallback"]
    }
    bbox_guarded_by_frame = (
        {
            item.frame: item
            for item in smooth_video_sequence_bbox_guarded(
                width,
                height,
                tw,
                th,
                observations,
                primary_bboxes,
                alpha=alpha,
            )
        }
        if include_ts4
        else {}
    )
    projected_state_by_frame = (
        {
            item.frame: item
            for item in smooth_video_sequence_projected_state_bbox_guarded(
                width, height, tw, th, observations, primary_bboxes, alpha=alpha
            )
        }
        if include_ts5
        else {}
    )

    records: list[dict] = []
    for composed in composed_records:
        frame = int(composed["frame"])
        item = smoothed_by_frame[frame]
        ts0 = composed["cmp1"]
        ts1 = {
            "x": item.x,
            "y": item.y,
            "w": item.w,
            "h": float(item.h),
            "crop_w": item.crop_w,
            "crop_h": item.crop_h,
            "placement_status": item.placement_status,
            "reset_reason": item.reset_reason,
            "ema_center_x": item.ema_center_x,
            "ema_center_y": item.ema_center_y,
            "clamped_x": item.clamped_x,
            "clamped_y": item.clamped_y,
            "matches_ts0_placement": item.matches_ts0_placement,
            "subject_visible_fraction": None,
            "subject_center_inside": False,
        }
        if composed["sanitized"]["status"] == SANITIZE_OK:
            xyxy = composed["sanitized"]["xyxy"]
            sanitized = SanitizedSubject(
                float(xyxy[0]),
                float(xyxy[1]),
                float(xyxy[2]),
                float(xyxy[3]),
                SANITIZE_OK,
                composed["sanitized"]["clamp_left"],
                composed["sanitized"]["clamp_top"],
                composed["sanitized"]["clamp_right"],
                composed["sanitized"]["clamp_bottom"],
            )
            ts1_rect = crop_rect_from_xywh(item.x, item.y, item.w, float(item.h))
            ts1["subject_visible_fraction"] = round(subject_visible_fraction(sanitized, ts1_rect), 6)
            ts1["subject_center_inside"] = subject_center_inside_crop(sanitized, ts1_rect)
        ts2 = None
        if include_ts2:
            adaptive = adaptive_by_frame[frame]
            ts2 = {
                "x": adaptive.x,
                "y": adaptive.y,
                "w": adaptive.w,
                "h": float(adaptive.h),
                "crop_w": adaptive.crop_w,
                "crop_h": adaptive.crop_h,
                "placement_status": adaptive.placement_status,
                "reset_reason": adaptive.reset_reason,
                "ema_center_x": adaptive.ema_center_x,
                "ema_center_y": adaptive.ema_center_y,
                "clamped_x": adaptive.clamped_x,
                "clamped_y": adaptive.clamped_y,
                "matches_ts0_placement": adaptive.matches_ts0_placement,
                "motion_norm": adaptive.motion_norm,
                "alpha_t": adaptive.alpha_t,
                "subject_visible_fraction": None,
                "subject_center_inside": False,
            }
            if composed["sanitized"]["status"] == SANITIZE_OK:
                ts2_rect = crop_rect_from_xywh(adaptive.x, adaptive.y, adaptive.w, float(adaptive.h))
                ts2["subject_visible_fraction"] = round(subject_visible_fraction(sanitized, ts2_rect), 6)
                ts2["subject_center_inside"] = subject_center_inside_crop(sanitized, ts2_rect)
        ts3 = None
        if include_ts3:
            guarded = guarded_by_frame[frame]
            ts3 = {
                "x": guarded.x,
                "y": guarded.y,
                "w": guarded.w,
                "h": float(guarded.h),
                "crop_w": guarded.crop_w,
                "crop_h": guarded.crop_h,
                "placement_status": guarded.placement_status,
                "reset_reason": guarded.reset_reason,
                "ema_center_x": guarded.ema_center_x,
                "ema_center_y": guarded.ema_center_y,
                "clamped_x": guarded.clamped_x,
                "clamped_y": guarded.clamped_y,
                "matches_ts0_placement": guarded.matches_ts0_placement,
                "guard_applied": guarded.guard_applied,
                "guard_correction_x": guarded.guard_correction_x,
                "guard_correction_y": guarded.guard_correction_y,
                "safe_x_min": guarded.safe_x_min,
                "safe_x_max": guarded.safe_x_max,
                "safe_y_min": guarded.safe_y_min,
                "safe_y_max": guarded.safe_y_max,
                "subject_visible_fraction": None,
                "subject_center_inside": False,
            }
            if composed["sanitized"]["status"] == SANITIZE_OK:
                ts3_rect = crop_rect_from_xywh(guarded.x, guarded.y, guarded.w, float(guarded.h))
                ts3["subject_visible_fraction"] = round(
                    subject_visible_fraction(sanitized, ts3_rect), 6
                )
                ts3["subject_center_inside"] = subject_center_inside_crop(sanitized, ts3_rect)
        ts4 = None
        if include_ts4:
            bbox_guarded = bbox_guarded_by_frame[frame]
            ts4 = {
                "x": bbox_guarded.x,
                "y": bbox_guarded.y,
                "w": bbox_guarded.w,
                "h": float(bbox_guarded.h),
                "crop_w": bbox_guarded.crop_w,
                "crop_h": bbox_guarded.crop_h,
                "placement_status": bbox_guarded.placement_status,
                "reset_reason": bbox_guarded.reset_reason,
                "ema_center_x": bbox_guarded.ema_center_x,
                "ema_center_y": bbox_guarded.ema_center_y,
                "clamped_x": bbox_guarded.clamped_x,
                "clamped_y": bbox_guarded.clamped_y,
                "matches_ts0_placement": bbox_guarded.matches_ts0_placement,
                "guard_applied": bbox_guarded.guard_applied,
                "guard_correction_x": bbox_guarded.guard_correction_x,
                "guard_correction_y": bbox_guarded.guard_correction_y,
                "safe_x_min": bbox_guarded.safe_x_min,
                "safe_x_max": bbox_guarded.safe_x_max,
                "safe_y_min": bbox_guarded.safe_y_min,
                "safe_y_max": bbox_guarded.safe_y_max,
                "guard_mode_x": bbox_guarded.guard_mode_x,
                "guard_mode_y": bbox_guarded.guard_mode_y,
                "bbox_fully_containable": bbox_guarded.bbox_fully_containable,
                "bbox_larger_than_crop": bbox_guarded.bbox_larger_than_crop,
                "per_axis_infeasible_x": bbox_guarded.per_axis_infeasible_x,
                "per_axis_infeasible_y": bbox_guarded.per_axis_infeasible_y,
                "visible_fraction_before_guard": bbox_guarded.visible_fraction_before_guard,
                "visible_fraction_after_guard": bbox_guarded.visible_fraction_after_guard,
                "visible_gain": bbox_guarded.visible_gain,
                "subject_visible_fraction": None,
                "subject_center_inside": False,
            }
            if composed["sanitized"]["status"] == SANITIZE_OK:
                ts4_rect = crop_rect_from_xywh(
                    bbox_guarded.x, bbox_guarded.y, bbox_guarded.w, float(bbox_guarded.h)
                )
                ts4["subject_visible_fraction"] = round(
                    subject_visible_fraction(sanitized, ts4_rect), 6
                )
                ts4["subject_center_inside"] = subject_center_inside_crop(sanitized, ts4_rect)
        ts5 = None
        if include_ts5:
            projected_state = projected_state_by_frame[frame]
            ts5 = {
                "x": projected_state.x,
                "y": projected_state.y,
                "w": projected_state.w,
                "h": float(projected_state.h),
                "crop_w": projected_state.crop_w,
                "crop_h": projected_state.crop_h,
                "placement_status": projected_state.placement_status,
                "reset_reason": projected_state.reset_reason,
                "ema_center_x": projected_state.ema_center_x,
                "ema_center_y": projected_state.ema_center_y,
                "proposal_center_x": projected_state.proposal_center_x,
                "proposal_center_y": projected_state.proposal_center_y,
                "projected_state_center_x": projected_state.projected_state_center_x,
                "projected_state_center_y": projected_state.projected_state_center_y,
                "state_output_residual_l1": projected_state.state_output_residual_l1,
                "clamped_x": projected_state.clamped_x,
                "clamped_y": projected_state.clamped_y,
                "matches_ts0_placement": projected_state.matches_ts0_placement,
                "guard_applied": projected_state.guard_applied,
                "guard_correction_x": projected_state.guard_correction_x,
                "guard_correction_y": projected_state.guard_correction_y,
                "safe_x_min": projected_state.safe_x_min,
                "safe_x_max": projected_state.safe_x_max,
                "safe_y_min": projected_state.safe_y_min,
                "safe_y_max": projected_state.safe_y_max,
                "guard_mode_x": projected_state.guard_mode_x,
                "guard_mode_y": projected_state.guard_mode_y,
                "bbox_fully_containable": projected_state.bbox_fully_containable,
                "bbox_larger_than_crop": projected_state.bbox_larger_than_crop,
                "per_axis_infeasible_x": projected_state.per_axis_infeasible_x,
                "per_axis_infeasible_y": projected_state.per_axis_infeasible_y,
                "visible_fraction_before_guard": projected_state.visible_fraction_before_guard,
                "visible_fraction_after_guard": projected_state.visible_fraction_after_guard,
                "visible_gain": projected_state.visible_gain,
                "subject_visible_fraction": None,
                "subject_center_inside": False,
            }
            if composed["sanitized"]["status"] == SANITIZE_OK:
                ts5_rect = crop_rect_from_xywh(
                    projected_state.x,
                    projected_state.y,
                    projected_state.w,
                    float(projected_state.h),
                )
                ts5["subject_visible_fraction"] = round(
                    subject_visible_fraction(sanitized, ts5_rect), 6
                )
                ts5["subject_center_inside"] = subject_center_inside_crop(sanitized, ts5_rect)
        geometry = {
            **{
                f"ts0_{key}": value
                for key, value in crop_geometry_valid(
                    int(ts0["x"]),
                    int(ts0["w"]),
                    int(ts0["y"]),
                    derived_height(int(ts0["w"]), tw, th),
                    width,
                    height,
                ).items()
            },
            **{
                f"ts1_{key}": value
                for key, value in crop_geometry_valid(
                    item.x, item.w, item.y, derived_height(item.w, tw, th), width, height
                ).items()
            },
        }
        if include_ts2:
            adaptive = adaptive_by_frame[frame]
            geometry.update({
                f"ts2_{key}": value
                for key, value in crop_geometry_valid(
                    adaptive.x,
                    adaptive.w,
                    adaptive.y,
                    derived_height(adaptive.w, tw, th),
                    width,
                    height,
                ).items()
            })
        if include_ts3:
            guarded = guarded_by_frame[frame]
            geometry.update({
                f"ts3_{key}": value
                for key, value in crop_geometry_valid(
                    guarded.x,
                    guarded.w,
                    guarded.y,
                    derived_height(guarded.w, tw, th),
                    width,
                    height,
                ).items()
            })
        if include_ts4:
            bbox_guarded = bbox_guarded_by_frame[frame]
            geometry.update({
                f"ts4_{key}": value
                for key, value in crop_geometry_valid(
                    bbox_guarded.x,
                    bbox_guarded.w,
                    bbox_guarded.y,
                    derived_height(bbox_guarded.w, tw, th),
                    width,
                    height,
                ).items()
            })
        if include_ts5:
            projected_state = projected_state_by_frame[frame]
            geometry.update({
                f"ts5_{key}": value
                for key, value in crop_geometry_valid(
                    projected_state.x,
                    projected_state.w,
                    projected_state.y,
                    derived_height(projected_state.w, tw, th),
                    width,
                    height,
                ).items()
            })
        record = {
                "video_id": composed["video_id"],
                "frame": frame,
                "image_width": width,
                "image_height": height,
                "stage5_2_status": composed["stage5_2_status"],
                "fallback_reasons": composed["fallback_reasons"],
                "ambiguous": composed["ambiguous"],
                "ambiguous_candidate_count": composed["ambiguous_candidate_count"],
                "stratum": composed["stratum"],
                "horizontal_center_offset": composed["horizontal_center_offset"],
                "sanitized": composed["sanitized"],
                "ts0": {
                    "x": int(ts0["x"]),
                    "y": int(ts0["y"]),
                    "w": int(ts0["w"]),
                    "h": float(ts0["h"]),
                    "crop_w": int(ts0["crop_w"]),
                    "crop_h": int(ts0["crop_h"]),
                    "placement_status": ts0["placement_status"],
                    "fallback": bool(ts0["fallback"]),
                    "subject_visible_fraction": ts0["subject_visible_fraction"],
                    "subject_center_inside": ts0["subject_center_inside"],
                    "frozen_regression": bool(composed["ts0_frozen_regression"]),
                },
                "ts1": ts1,
                "geometry_valid": geometry,
            }
        if ts2 is not None:
            record["ts2"] = ts2
        if ts3 is not None:
            record["ts3"] = ts3
        if ts4 is not None:
            record["ts4"] = ts4
        if ts5 is not None:
            record["ts5"] = ts5
        records.append(record)
    return records, mismatches


def engineering_gate(manifest: dict, records: list[dict], crosscheck_mismatches: list[str]) -> dict:
    expected_keys = {
        (video["video_id"], int(entry["frame"]))
        for video in manifest["videos"]
        for entry in video["frames"]
    }
    produced_keys = [(record["video_id"], int(record["frame"])) for record in records]
    missing = expected_keys - set(produced_keys)
    extra = set(produced_keys) - expected_keys
    duplicates = len(produced_keys) - len(set(produced_keys))
    treatment_sides = [side for side in TREATMENT_SIDES if records and side in records[0]]
    crop_size_unchanged = all(
        record[side]["w"] == record["ts0"]["w"] and record[side]["h"] == record["ts0"]["h"]
        for record in records
        for side in treatment_sides
    )
    fallback_placement_unchanged = all(
        record[side]["placement_status"] == PLACEMENT_FALLBACK_CENTER_CROP
        and (record[side]["x"], record[side]["y"]) == (record["ts0"]["x"], record["ts0"]["y"])
        for record in records
        if record["ts0"]["fallback"]
        for side in treatment_sides
    )
    geometry_failures = [record for record in records if not all(record["geometry_valid"].values())]
    invalid_crop = len(geometry_failures)
    out_of_bounds = sum(
        1
        for record in geometry_failures
        if not all(record["geometry_valid"][f"{side}_{key}"] for side in ["ts0", *treatment_sides] for key in ("nonnegative", "x_within_frame"))
    )
    ratio_violations = sum(
        1
        for record in geometry_failures
        if not all(record["geometry_valid"][f"{side}_derived_height_within_frame"] for side in ["ts0", *treatment_sides])
    )
    ts0_regression = sum(1 for record in records if record["ts0"]["frozen_regression"])
    result = {
        "expected_frames": len(expected_keys),
        "produced_frames": len(produced_keys),
        "missing": len(missing),
        "extra": len(extra),
        "duplicates": duplicates,
        "manifest_crosscheck_mismatches": len(crosscheck_mismatches),
        "crop_size_unchanged": crop_size_unchanged,
        "fallback_placement_unchanged": fallback_placement_unchanged,
        "invalid_crop": invalid_crop,
        "out_of_bounds": out_of_bounds,
        "ratio_violations": ratio_violations,
        "ts0_frozen_regression": ts0_regression,
        "frame_identity_complete": not missing and not extra and not duplicates,
    }
    if len(treatment_sides) > 1:
        result["treatment_sides"] = treatment_sides
    return result


def evaluate_gates(
    gate: dict, deterministic_pass: bool, artifact_modified: dict, multi_subject_generated: bool
) -> dict:
    result = {
        "manifest_sha_pinned": True,
        "frame_identity_100pct": bool(gate["frame_identity_complete"]),
        "manifest_crosscheck_consistent": gate["manifest_crosscheck_mismatches"] == 0,
        "ts1_crop_size_unchanged": bool(gate["crop_size_unchanged"]),
        "fallback_placement_unchanged": bool(gate["fallback_placement_unchanged"]),
        "invalid_crop_zero": gate["invalid_crop"] == 0,
        "out_of_bounds_zero": gate["out_of_bounds"] == 0,
        "ratio_violations_zero": gate["ratio_violations"] == 0,
        "ts0_frozen_regression_zero": gate["ts0_frozen_regression"] == 0,
        "deterministic_pass": deterministic_pass,
        "stage5_2_artifact_unmodified": not any(artifact_modified.values()),
        "heldout_access_zero": True,
        "multi_subject_diagnostic_generated": multi_subject_generated,
    }
    if "ts2" in gate.get("treatment_sides", []):
        result["ts2_crop_size_unchanged"] = bool(gate["crop_size_unchanged"])
        result["ts2_fallback_placement_unchanged"] = bool(gate["fallback_placement_unchanged"])
    if "ts3" in gate.get("treatment_sides", []):
        result["ts3_crop_size_unchanged"] = bool(gate["crop_size_unchanged"])
        result["ts3_fallback_placement_unchanged"] = bool(gate["fallback_placement_unchanged"])
    if "ts4" in gate.get("treatment_sides", []):
        result["ts4_crop_size_unchanged"] = bool(gate["crop_size_unchanged"])
        result["ts4_fallback_placement_unchanged"] = bool(gate["fallback_placement_unchanged"])
    if "ts5" in gate.get("treatment_sides", []):
        result["ts5_crop_size_unchanged"] = bool(gate["crop_size_unchanged"])
        result["ts5_fallback_placement_unchanged"] = bool(gate["fallback_placement_unchanged"])
    return result


def pooled_distributions(records: list[dict]) -> dict[str, list[float]]:
    """Concatenated per-transition displacement / acceleration values across videos."""
    from aic_video_highlight.spatial_composition.temporal_diagnostics import (
        acceleration_norm,
        crop_center_x,
        displacement_norm,
    )

    treatment_sides = [side for side in TREATMENT_SIDES if records and side in records[0]]
    pooled: dict[str, list[float]] = {"ts0_displacement": [], "ts0_acceleration": []}
    for side in treatment_sides:
        pooled[f"{side}_displacement"] = []
        pooled[f"{side}_displacement_smoothed_only"] = []
        pooled[f"{side}_acceleration"] = []
        pooled[f"{side}_acceleration_smoothed_only"] = []
    by_video: dict[str, list[dict]] = {}
    for record in records:
        by_video.setdefault(record["video_id"], []).append(record)
    for video_records in by_video.values():
        ordered = sorted(video_records, key=lambda record: int(record["frame"]))
        width = int(ordered[0]["image_width"])
        for previous, current in zip(ordered, ordered[1:]):
            if previous["ts0"]["fallback"] or current["ts0"]["fallback"]:
                continue
            if int(current["frame"]) - int(previous["frame"]) != 1:
                continue
            pooled["ts0_displacement"].append(
                displacement_norm(
                    crop_center_x(int(previous["ts0"]["x"]), int(previous["ts0"]["w"])),
                    crop_center_x(int(current["ts0"]["x"]), int(current["ts0"]["w"])),
                    width,
                )
            )
            for side in treatment_sides:
                value = displacement_norm(
                    crop_center_x(int(previous[side]["x"]), int(previous[side]["w"])),
                    crop_center_x(int(current[side]["x"]), int(current[side]["w"])),
                    width,
                )
                pooled[f"{side}_displacement"].append(value)
                if current[side]["reset_reason"] is None:
                    pooled[f"{side}_displacement_smoothed_only"].append(value)
        for first, second, third in zip(ordered, ordered[1:], ordered[2:]):
            if any(record["ts0"]["fallback"] for record in (first, second, third)):
                continue
            if int(second["frame"]) - int(first["frame"]) != 1 or int(third["frame"]) - int(second["frame"]) != 1:
                continue

            def delta(record_a: dict, record_b: dict, side: str) -> float:
                return crop_center_x(int(record_a[side]["x"]), int(record_a[side]["w"])) - crop_center_x(
                    int(record_b[side]["x"]), int(record_b[side]["w"])
                )

            pooled["ts0_acceleration"].append(
                acceleration_norm(delta(first, second, "ts0"), delta(second, third, "ts0"), width)
            )
            for side in treatment_sides:
                value = acceleration_norm(delta(first, second, side), delta(second, third, side), width)
                pooled[f"{side}_acceleration"].append(value)
                if second[side]["reset_reason"] is None and third[side]["reset_reason"] is None:
                    pooled[f"{side}_acceleration_smoothed_only"].append(value)
    return pooled


def summary_payload(values: list[float]) -> dict:
    return {
        **temporal_summary(values),
        "large_jump_ratios_descriptive_only": large_jump_ratios(values),
    }


def _project_treatment(records: list[dict], side: str) -> list[dict]:
    """Expose one frozen treatment under the legacy TS-1 diagnostic contract."""
    if side == "ts1":
        return records
    return [{**record, "ts1": record[side]} for record in records]


def _rename_ts1_keys(value, replacement: str):
    if isinstance(value, dict):
        return {
            str(key).replace("ts1", replacement): _rename_ts1_keys(item, replacement)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_ts1_keys(item, replacement) for item in value]
    if isinstance(value, str):
        return value.replace("TS-1", replacement.upper()).replace("ts1", replacement)
    return value


def temporal_metrics_for_all_treatments(records: list[dict]) -> dict:
    metrics = temporal_stability_metrics(records)
    treatment_sides = [side for side in TREATMENT_SIDES[1:] if records and side in records[0]]
    for side in treatment_sides:
        adaptive = _rename_ts1_keys(
            temporal_stability_metrics(_project_treatment(records, side)), side
        )
        metrics.update({key: value for key, value in adaptive.items() if key.startswith(f"{side}_")})
    if treatment_sides:
        metrics["metric_role"] = (
            "DRAFT AMENDMENT SMOKE / TS-0 vs "
            + " vs ".join(side.upper() for side in ("ts1", *treatment_sides))
        )
    return metrics


def spatial_metrics_for_all_treatments(records: list[dict]) -> dict:
    metrics = spatial_guardrail_metrics(records)
    for side in TREATMENT_SIDES[1:]:
        if not records or side not in records[0]:
            continue
        treatment = _rename_ts1_keys(
            spatial_guardrail_metrics(_project_treatment(records, side)), side
        )
        metrics[side] = treatment[side]
        for stratum, payload in metrics["strata"].items():
            payload[f"{side}_visible"] = treatment["strata"][stratum][f"{side}_visible"]
            payload[f"{side}_delta_mean"] = treatment["strata"][stratum]["delta_mean"]
    return metrics


def guard_projection_diagnostics(records: list[dict], side: str = "ts3") -> dict | None:
    """Describe one guard's intervention frequency/magnitude without changing a gate.

    Diagnostic only: activation counts/rates, axis split, continuous guard-run
    length, correction-magnitude distributions and per-stratum breakdowns. None of
    these values feeds a promotion gate in any Stage 5.4 protocol.
    """
    if side not in ("ts3", "ts4", "ts5"):
        raise ValueError(f"unsupported guard side: {side}")
    eligible = [record for record in records if side in record and not record["ts0"]["fallback"]]
    if not eligible:
        return None
    guarded = [record for record in eligible if record[side]["guard_applied"]]

    def correction(axis: str) -> dict:
        return temporal_summary([abs(float(record[side][f"guard_correction_{axis}"])) for record in eligible])

    horizontal_applied = sum(1 for record in guarded if float(record[side]["guard_correction_x"]) != 0.0)
    vertical_applied = sum(1 for record in guarded if float(record[side]["guard_correction_y"]) != 0.0)
    both_axes = sum(
        1
        for record in guarded
        if float(record[side]["guard_correction_x"]) != 0.0
        and float(record[side]["guard_correction_y"]) != 0.0
    )

    guard_run_count = 0
    max_guard_run_length = 0
    by_video: dict[str, list[dict]] = {}
    for record in eligible:
        by_video.setdefault(record["video_id"], []).append(record)
    for video_records in by_video.values():
        ordered = sorted(video_records, key=lambda record: int(record["frame"]))
        active_run_length = 0
        previous_frame = None
        for record in ordered:
            frame = int(record["frame"])
            contiguous = previous_frame is not None and frame - previous_frame == 1
            if record[side]["guard_applied"]:
                if active_run_length > 0 and contiguous:
                    active_run_length += 1
                else:
                    guard_run_count += 1
                    active_run_length = 1
                max_guard_run_length = max(max_guard_run_length, active_run_length)
            else:
                active_run_length = 0
            previous_frame = frame

    strata: dict[str, dict] = {}
    for record in eligible:
        stratum = str(record["stratum"])
        bucket = strata.setdefault(stratum, {"eligible": 0, "applied": 0, "corrections": []})
        bucket["eligible"] += 1
        if record[side]["guard_applied"]:
            bucket["applied"] += 1
            bucket["corrections"].append(
                abs(float(record[side]["guard_correction_x"]))
                + abs(float(record[side]["guard_correction_y"]))
            )
    strata_payload = {
        stratum: {
            "eligible": bucket["eligible"],
            "guard_applied_frames": bucket["applied"],
            "guard_applied_rate": round(bucket["applied"] / bucket["eligible"], 6) if bucket["eligible"] else 0.0,
            "absolute_correction_l1_px": summary_payload(bucket["corrections"]),
        }
        for stratum, bucket in sorted(strata.items())
    }

    return {
        "role": "DIAGNOSTIC_ONLY",
        "treatment_side": side,
        "eligible_nonfallback_frames": len(eligible),
        "guard_applied_frames": len(guarded),
        "guard_applied_rate": round(len(guarded) / len(eligible), 6),
        "absolute_horizontal_correction": correction("x"),
        "absolute_vertical_correction": correction("y"),
        "horizontal_applied_frames": horizontal_applied,
        "vertical_applied_frames": vertical_applied,
        "both_axes_applied_frames": both_axes,
        "absolute_correction_l1_all_eligible": temporal_summary([
            abs(float(record[side]["guard_correction_x"]))
            + abs(float(record[side]["guard_correction_y"]))
            for record in eligible
        ]),
        "absolute_correction_l1_applied": temporal_summary([
            abs(float(record[side]["guard_correction_x"]))
            + abs(float(record[side]["guard_correction_y"]))
            for record in guarded
        ]),
        "guard_run_count": guard_run_count,
        "max_guard_run_length": max_guard_run_length,
        "strata": strata_payload,
    }


def bbox_guard_diagnostics(records: list[dict], side: str = "ts4") -> dict | None:
    """BBox feasibility/visibility diagnostics for TS-4 or TS-5; never a gate."""
    if side not in ("ts4", "ts5"):
        raise ValueError(f"unsupported bbox guard side: {side}")
    eligible = [record for record in records if side in record and not record["ts0"]["fallback"]]
    if not eligible:
        return None

    def summarize(bucket: list[dict]) -> dict:
        gains = [float(record[side]["visible_gain"]) for record in bucket]
        before = [float(record[side]["visible_fraction_before_guard"]) for record in bucket]
        after = [float(record[side]["visible_fraction_after_guard"]) for record in bucket]
        return {
            "eligible": len(bucket),
            "subject_bbox_fully_containable_count": sum(
                bool(record[side]["bbox_fully_containable"]) for record in bucket
            ),
            "bbox_larger_than_crop_count": sum(
                bool(record[side]["bbox_larger_than_crop"]) for record in bucket
            ),
            "full_containment_projection_count": sum(
                bool(record[side]["guard_applied"])
                and bool(record[side]["bbox_fully_containable"])
                for record in bucket
            ),
            "maximum_overlap_projection_count": sum(
                bool(record[side]["guard_applied"])
                and not bool(record[side]["bbox_fully_containable"])
                for record in bucket
            ),
            "no_correction_count": sum(not bool(record[side]["guard_applied"]) for record in bucket),
            "per_axis_infeasible_count": {
                "horizontal": sum(bool(record[side]["per_axis_infeasible_x"]) for record in bucket),
                "vertical": sum(bool(record[side]["per_axis_infeasible_y"]) for record in bucket),
            },
            "visible_fraction_before_guard": temporal_summary(before),
            "visible_fraction_after_guard": temporal_summary(after),
            "visible_gain": temporal_summary(gains),
        }

    payload = summarize(eligible)
    ts3_eligible = [record for record in eligible if "ts3" in record]
    if ts3_eligible:
        ts4_l1 = [
            abs(float(record[side]["guard_correction_x"]))
            + abs(float(record[side]["guard_correction_y"]))
            for record in ts3_eligible
        ]
        ts3_l1 = [
            abs(float(record["ts3"]["guard_correction_x"]))
            + abs(float(record["ts3"]["guard_correction_y"]))
            for record in ts3_eligible
        ]
        payload["relative_to_ts3"] = {
            f"{side}_minus_ts3_activation_count": sum(
                bool(record[side]["guard_applied"]) for record in ts3_eligible
            ) - sum(bool(record["ts3"]["guard_applied"]) for record in ts3_eligible),
            f"{side}_minus_ts3_activation_rate": round(
                (
                    sum(bool(record[side]["guard_applied"]) for record in ts3_eligible)
                    - sum(bool(record["ts3"]["guard_applied"]) for record in ts3_eligible)
                ) / len(ts3_eligible),
                6,
            ),
            "ts3_absolute_correction_l1": temporal_summary(ts3_l1),
            f"{side}_absolute_correction_l1": temporal_summary(ts4_l1),
            f"{side}_minus_ts3_correction_l1": temporal_summary([
                ts4 - ts3 for ts4, ts3 in zip(ts4_l1, ts3_l1)
            ]),
        }
    payload.update({
        "role": "DIAGNOSTIC_ONLY",
        "treatment_side": side,
        "strata": {
            stratum: summarize([record for record in eligible if record["stratum"] == stratum])
            for stratum in ("near_center", "moderately_off_center", "strongly_off_center")
        },
    })
    return payload


def projected_state_attribution_diagnostics(records: list[dict]) -> dict | None:
    """TS-5 guard/acceleration attribution, strictly DIAGNOSTIC_ONLY."""
    from aic_video_highlight.spatial_composition.temporal_diagnostics import (
        acceleration_norm,
        crop_center_x,
    )

    eligible = [
        record for record in records if "ts5" in record and not record["ts0"]["fallback"]
    ]
    if not eligible:
        return None
    acceleration_buckets = {
        "guard_active": [],
        "guard_inactive": [],
        "inactive_to_active": [],
        "active_to_inactive": [],
    }
    boundary_movements: list[float] = []
    by_video: dict[str, list[dict]] = {}
    for record in records:
        if "ts5" in record:
            by_video.setdefault(record["video_id"], []).append(record)
    for video_records in by_video.values():
        ordered = sorted(video_records, key=lambda record: int(record["frame"]))
        for previous, current in zip(ordered, ordered[1:]):
            if previous["ts0"]["fallback"] or current["ts0"]["fallback"]:
                continue
            if int(current["frame"]) - int(previous["frame"]) != 1:
                continue
            boundary_movements.append(sum(
                abs(float(current["ts5"][key]) - float(previous["ts5"][key]))
                for key in ("safe_x_min", "safe_x_max", "safe_y_min", "safe_y_max")
            ))
        for first, second, third in zip(ordered, ordered[1:], ordered[2:]):
            if any(record["ts0"]["fallback"] for record in (first, second, third)):
                continue
            if (
                int(second["frame"]) - int(first["frame"]) != 1
                or int(third["frame"]) - int(second["frame"]) != 1
            ):
                continue
            width = int(third["image_width"])
            first_delta = crop_center_x(second["ts5"]["x"], second["ts5"]["w"]) - crop_center_x(
                first["ts5"]["x"], first["ts5"]["w"]
            )
            second_delta = crop_center_x(third["ts5"]["x"], third["ts5"]["w"]) - crop_center_x(
                second["ts5"]["x"], second["ts5"]["w"]
            )
            value = acceleration_norm(first_delta, second_delta, width)
            second_active = bool(second["ts5"]["guard_applied"])
            third_active = bool(third["ts5"]["guard_applied"])
            acceleration_buckets["guard_active" if third_active else "guard_inactive"].append(value)
            if not second_active and third_active:
                acceleration_buckets["inactive_to_active"].append(value)
            elif second_active and not third_active:
                acceleration_buckets["active_to_inactive"].append(value)

    corrections = [
        abs(float(record["ts5"]["guard_correction_x"]))
        + abs(float(record["ts5"]["guard_correction_y"]))
        for record in eligible
    ]
    return {
        "role": "DIAGNOSTIC_ONLY",
        "decision_inputs": False,
        "pre_projection_correction_l1_px": temporal_summary(corrections),
        "proposal_projected_state_displacement_l1_px": temporal_summary(corrections),
        "guard": guard_projection_diagnostics(records, "ts5"),
        "acceleration_by_guard_state": {
            name: temporal_summary(values) for name, values in acceleration_buckets.items()
        },
        "feasible_boundary_movement_l1_px": temporal_summary(boundary_movements),
        "state_output_residual_l1_px": temporal_summary([
            float(record["ts5"]["state_output_residual_l1"]) for record in eligible
        ]),
    }


def validate_amendment_contract(config: dict, protocol: dict, manifest: dict) -> dict:
    """Static scientific identity checks; executes neither Smoke nor Formal."""
    if config.get("experiment_id") != "stage5_4_amendment_smoke":
        raise FrozenInputError("unexpected Amendment experiment id")
    if protocol.get("status") != "DRAFT_READY_FOR_SMOKE":
        raise FrozenInputError("Amendment protocol is not DRAFT_READY_FOR_SMOKE")
    smoothing = config.get("temporal_smoothing", {})
    ts2 = smoothing.get("ts2", {})
    expected = {
        "method": "motion_adaptive_ema_v1",
        "alpha_min": 0.5,
        "smoothing_ceiling_motion_norm": 0.1,
        "full_response_motion_norm": 0.2,
    }
    if any(ts2.get(key) != value for key, value in expected.items()):
        raise FrozenInputError("Amendment TS-2 schedule differs from the frozen draft")
    candidate_set = protocol.get("parameter_candidate_set")
    if candidate_set not in (None, []) and not (
        isinstance(candidate_set, dict) and candidate_set.get("enabled") is False
    ):
        raise FrozenInputError("Amendment protocol unexpectedly enables a parameter candidate set")
    if manifest.get("video_count") != 24 or manifest.get("frame_count") != 1080:
        raise FrozenInputError("Amendment must reuse the frozen Stage 5.4 Smoke24 manifest")
    manifest_spec = config.get("manifest", {})
    if (
        manifest.get("manifest_id") != manifest_spec.get("manifest_id")
        or manifest.get("manifest_sha256") != manifest_spec.get("expected_manifest_sha256")
    ):
        raise FrozenInputError("Amendment Smoke24 manifest identity mismatch")
    if smoothing.get("reset_rules") != ["NEW_VIDEO", "FRAME_GAP_GT_1", "FALLBACK"]:
        raise FrozenInputError("Amendment reset rules differ from TS-1")
    if int(config.get("heldout_lock", {}).get("allowed_access", -1)) != 0:
        raise FrozenInputError("Heldout access must remain zero")
    forbidden_input_tokens = ("heldout", "hard", "official_test", "official-test")
    for name, entry in config.get("inputs", {}).items():
        candidate = f"{name} {entry.get('path', '')}".lower()
        if any(token in candidate for token in forbidden_input_tokens):
            raise FrozenInputError(f"forbidden dataset input configured: {name}")
    protocol_path = Path(config["protocol"])
    actual_protocol_sha = file_sha256(protocol_path)
    if config.get("protocol_sha256") != actual_protocol_sha:
        raise FrozenInputError("Amendment protocol file SHA does not match config binding")
    return {
        "validation": "PASS",
        "protocol_status": protocol["status"],
        "protocol_sha256": actual_protocol_sha,
        "algorithm": expected,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "videos": manifest["video_count"],
        "frames": manifest["frame_count"],
        "gpu_requirement": "NONE",
        "heldout_access": 0,
    }


def validate_amendment2_contract(config: dict, protocol: dict, manifest: dict) -> dict:
    """Validate the parameter-free TS-3 Smoke draft without executing it."""
    if config.get("experiment_id") != "stage5_4_amendment2_smoke":
        raise FrozenInputError("unexpected Amendment 2 experiment id")
    if protocol.get("status") != "DRAFT_READY_FOR_SMOKE":
        raise FrozenInputError("Amendment 2 protocol is not DRAFT_READY_FOR_SMOKE")
    smoothing = config.get("temporal_smoothing", {})
    if smoothing.get("alpha") != 0.5:
        raise FrozenInputError("Amendment 2 must retain fixed alpha=0.5")
    expected_ts3 = {
        "method": "guarded_constrained_ema_v1",
        "proposal": "ema_crop_center_v1 fixed alpha=0.5",
        "constraint": "current_frozen_primary_subject_center_inside_final_crop",
        "projection": "nearest_point_on_safe_integer_top_left_rectangle",
        "correction_feedback_to_ema_state": False,
        "extra_hyperparameters": [],
    }
    if smoothing.get("ts3") != expected_ts3:
        raise FrozenInputError("Amendment 2 TS-3 definition differs from the frozen draft")
    if smoothing.get("reset_rules") != ["NEW_VIDEO", "FRAME_GAP_GT_1", "FALLBACK"]:
        raise FrozenInputError("Amendment 2 reset rules differ from TS-1")
    if manifest.get("video_count") != 24 or manifest.get("frame_count") != 1080:
        raise FrozenInputError("Amendment 2 must reuse the frozen Stage 5.4 Smoke24 manifest")
    manifest_spec = config.get("manifest", {})
    if (
        manifest.get("manifest_id") != manifest_spec.get("manifest_id")
        or manifest.get("manifest_sha256") != manifest_spec.get("expected_manifest_sha256")
    ):
        raise FrozenInputError("Amendment 2 Smoke24 manifest identity mismatch")
    if int(config.get("heldout_lock", {}).get("allowed_access", -1)) != 0:
        raise FrozenInputError("Heldout access must remain zero")
    forbidden_input_tokens = ("heldout", "hard", "official_test", "official-test")
    for name, entry in config.get("inputs", {}).items():
        candidate = f"{name} {entry.get('path', '')}".lower()
        if any(token in candidate for token in forbidden_input_tokens):
            raise FrozenInputError(f"forbidden dataset input configured: {name}")
    protocol_path = Path(config["protocol"])
    actual_protocol_sha = file_sha256(protocol_path)
    if config.get("protocol_sha256") != actual_protocol_sha:
        raise FrozenInputError("Amendment 2 protocol file SHA does not match config binding")
    return {
        "validation": "PASS",
        "protocol_status": protocol["status"],
        "protocol_sha256": actual_protocol_sha,
        "algorithm": expected_ts3,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "videos": manifest["video_count"],
        "frames": manifest["frame_count"],
        "gpu_requirement": "NONE",
        "heldout_access": 0,
    }


def _expected_ts4_definition() -> dict:
    return {
        "method": "bbox_aware_constrained_ema_v1",
        "proposal": "ema_crop_center_v1 fixed alpha=0.5",
        "primary_objective": "maximize_current_frozen_primary_bbox_visible_fraction",
        "secondary_objective": "minimum_integer_projection_distance_from_ts1_proposal",
        "projection": "nearest_point_on_discrete_maximum_overlap_rectangle",
        "correction_feedback_to_ema_state": False,
        "extra_hyperparameters": [],
    }


def _expected_ts5_definition() -> dict:
    return {
        "method": "projected_state_constrained_ema_v1",
        "proposal": "ema_crop_center_v1 fixed alpha=0.5",
        "projection": "reuse_ts4_nearest_point_on_discrete_maximum_overlap_rectangle",
        "state_update": "continuous_ema_state_plus_exact_integer_projection_correction",
        "output": "placement_of_projected_recursive_state",
        "correction_feedback_to_ema_state": True,
        "extra_hyperparameters": [],
    }


def _protocol_semantic_sha(protocol: dict) -> str:
    return canonical_sha256({
        key: value for key, value in protocol.items() if key != "protocol_semantic_sha256"
    })


def _validate_master_preregistration(config: dict, protocol: dict) -> dict:
    master_path = Path(config["master_preregistration"])
    if not master_path.is_file():
        raise FrozenInputError("Amendment 4 master preregistration is missing")
    master = json.loads(master_path.read_text(encoding="utf-8"))
    role = "formal" if config["experiment_id"].endswith("_formal") else "smoke"
    binding = master.get("artifacts", {}).get(role, {})
    checks = {
        "master_status": master.get("status")
        == "PREREGISTERED_BEFORE_ANY_AMENDMENT4_EXPERIMENT",
        "method": master.get("method") == _expected_ts5_definition(),
        "config_path": binding.get("config") == str(config.get("config_repo_path")),
        "config_sha": binding.get("config_sha256") == file_sha256(Path(config["config_repo_path"])),
        "protocol_path": binding.get("protocol") == config.get("protocol"),
        "protocol_byte_sha": binding.get("protocol_byte_sha256")
        == file_sha256(Path(config["protocol"])),
        "protocol_semantic_sha": binding.get("protocol_semantic_sha256")
        == _protocol_semantic_sha(protocol),
    }
    if not all(checks.values()):
        raise FrozenInputError(f"Amendment 4 master preregistration drift: {checks}")
    return checks


def validate_amendment4_smoke_contract(config: dict, protocol: dict, manifest: dict) -> dict:
    """Validate frozen TS-5 Smoke science and identities without executing it."""
    expected_status = "PREREGISTERED_BEFORE_ANY_AMENDMENT4_EXPERIMENT"
    if config.get("experiment_id") != "stage5_4_amendment4_smoke":
        raise FrozenInputError("unexpected Amendment 4 Smoke experiment id")
    if protocol.get("status") != expected_status:
        raise FrozenInputError("Amendment 4 Smoke protocol is not preregistered")
    smoothing = config.get("temporal_smoothing", {})
    if (
        smoothing.get("alpha") != 0.5
        or smoothing.get("ts4") != _expected_ts4_definition()
        or smoothing.get("ts5") != _expected_ts5_definition()
        or smoothing.get("reset_rules") != ["NEW_VIDEO", "FRAME_GAP_GT_1", "FALLBACK"]
    ):
        raise FrozenInputError("Amendment 4 method, projection, alpha or reset semantics drifted")
    if manifest.get("video_count") != 24 or manifest.get("frame_count") != 1080:
        raise FrozenInputError("Amendment 4 must reuse frozen Smoke24")
    spec = config.get("manifest", {})
    if (
        manifest.get("manifest_id") != spec.get("manifest_id")
        or manifest.get("manifest_sha256") != spec.get("expected_manifest_sha256")
    ):
        raise FrozenInputError("Amendment 4 Smoke24 identity mismatch")
    actual_byte_sha = file_sha256(Path(config["protocol"]))
    actual_semantic_sha = _protocol_semantic_sha(protocol)
    if (
        config.get("protocol_sha256") != actual_byte_sha
        or config.get("protocol_semantic_sha256") != actual_semantic_sha
        or protocol.get("protocol_semantic_sha256") != actual_semantic_sha
    ):
        raise FrozenInputError("Amendment 4 Smoke protocol byte/semantic binding mismatch")
    if config.get("decision_gates") != protocol.get("decision_gates"):
        raise FrozenInputError("Amendment 4 Smoke gates drifted")
    if int(config.get("heldout_lock", {}).get("allowed_access", -1)) != 0:
        raise FrozenInputError("Heldout access must remain zero")
    if int(config.get("official_test_lock", {}).get("allowed_access", -1)) != 0:
        raise FrozenInputError("Official Test access must remain zero")
    master_checks = _validate_master_preregistration(config, protocol)
    return {
        "validation": "PASS",
        "protocol_status": expected_status,
        "protocol_sha256": actual_byte_sha,
        "protocol_semantic_sha256": actual_semantic_sha,
        "algorithm": _expected_ts5_definition(),
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "videos": manifest["video_count"],
        "frames": manifest["frame_count"],
        "master_preregistration": master_checks,
        "gpu_requirement": "NONE",
        "heldout_access": 0,
        "official_test_access": 0,
    }


def validate_amendment3_smoke_contract(config: dict, protocol: dict, manifest: dict) -> dict:
    """Validate jointly preregistered, parameter-free TS-4 Smoke identity."""
    if config.get("experiment_id") != "stage5_4_amendment3_smoke":
        raise FrozenInputError("unexpected Amendment 3 Smoke experiment id")
    expected_status = "PREREGISTERED_BEFORE_ANY_AMENDMENT3_EXPERIMENT"
    if protocol.get("status") != expected_status:
        raise FrozenInputError("Amendment 3 Smoke protocol is not preregistered")
    smoothing = config.get("temporal_smoothing", {})
    if smoothing.get("alpha") != 0.5 or smoothing.get("ts4") != _expected_ts4_definition():
        raise FrozenInputError("Amendment 3 TS-4 definition drifted")
    if smoothing.get("reset_rules") != ["NEW_VIDEO", "FRAME_GAP_GT_1", "FALLBACK"]:
        raise FrozenInputError("Amendment 3 reset rules differ from TS-1")
    if manifest.get("video_count") != 24 or manifest.get("frame_count") != 1080:
        raise FrozenInputError("Amendment 3 must reuse frozen Smoke24")
    spec = config.get("manifest", {})
    if manifest.get("manifest_id") != spec.get("manifest_id") or manifest.get(
        "manifest_sha256"
    ) != spec.get("expected_manifest_sha256"):
        raise FrozenInputError("Amendment 3 Smoke24 identity mismatch")
    if int(config.get("heldout_lock", {}).get("allowed_access", -1)) != 0:
        raise FrozenInputError("Heldout access must remain zero")
    actual_protocol_sha = file_sha256(Path(config["protocol"]))
    if config.get("protocol_sha256") != actual_protocol_sha:
        raise FrozenInputError("Amendment 3 Smoke protocol SHA binding mismatch")
    return {
        "validation": "PASS",
        "protocol_status": expected_status,
        "protocol_sha256": actual_protocol_sha,
        "algorithm": _expected_ts4_definition(),
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "videos": manifest["video_count"],
        "frames": manifest["frame_count"],
        "gpu_requirement": "NONE",
        "heldout_access": 0,
    }


def load_amendment3_promotion_evidence(config: dict, environment: EnvironmentPaths) -> dict:
    """Fail-closed authorization from the exact preregistered Smoke artifacts."""
    spec = config.get("promotion_authorization", {})
    output = environment.outputs / Path(spec.get("smoke_output_directory", ""))
    validation_path = output / "machine/validation.json"
    config_snapshot = output / "snapshots/config.json"
    protocol_snapshot = output / "snapshots/protocol.json"
    required = (validation_path, config_snapshot, protocol_snapshot)
    if not all(path.is_file() for path in required):
        raise FrozenInputError("Amendment 3 Smoke promotion evidence is incomplete")
    identity_ok = (
        file_sha256(config_snapshot) == spec.get("expected_smoke_config_sha256")
        and file_sha256(protocol_snapshot) == spec.get("expected_smoke_protocol_sha256")
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    decision = evaluate_smoke_promotion(validation, identity_ok=identity_ok)
    if not decision["formal_authorized"]:
        raise FrozenInputError(
            f"SMOKE_PASS_TO_FORMAL denied: {json.dumps(decision, sort_keys=True)}"
        )
    return {"decision": decision, "validation": validation, "output": str(output)}


def write_amendment4_promotion_marker(
    output: Path, config: dict, protocol: dict, validation: dict, execution_head: str
) -> dict:
    """Persist the exact fail-closed TS-5 Smoke decision after validation."""
    validation_path = output / "machine/validation.json"
    marker = build_promotion_marker(
        validation,
        identity_ok=True,
        method=_expected_ts5_definition()["method"],
        smoke_manifest_identity=config["manifest"]["expected_manifest_sha256"],
        execution_head=execution_head,
        protocol_byte_sha256=file_sha256(Path(config["protocol"])),
        protocol_semantic_sha256=_protocol_semantic_sha(protocol),
        config_sha256=file_sha256(Path(config["config_repo_path"])),
        validation_sha256=file_sha256(validation_path),
    )
    atomic_write_json(output / "machine/promotion_decision.json", marker)
    return marker


def load_amendment4_promotion_evidence(config: dict, environment: EnvironmentPaths) -> dict:
    """Authorize TS-5 Formal only from an exact immutable Smoke marker."""
    spec = config.get("promotion_authorization", {})
    output = environment.outputs / Path(spec.get("smoke_output_directory", ""))
    paths = {
        "marker": output / "machine/promotion_decision.json",
        "validation": output / "machine/validation.json",
        "config": output / "snapshots/config.json",
        "protocol": output / "snapshots/protocol.json",
    }
    if not all(path.is_file() for path in paths.values()):
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: TS-5 Smoke promotion evidence incomplete")
    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    current_head = RunContext(
        config["experiment_id"], config["stage"], config["run_type"], environment.repo,
        resolve_experiment_paths(environment, config), Path(config["config_repo_path"]),
        Path(config["protocol"]),
    ).identity()["git_head"]
    identity_ok = all((
        file_sha256(paths["config"]) == spec.get("expected_smoke_config_sha256"),
        file_sha256(paths["protocol"]) == spec.get("expected_smoke_protocol_sha256"),
        marker.get("schema_version") == "aic.smoke-promotion/v1",
        marker.get("method") == _expected_ts5_definition()["method"],
        marker.get("smoke_manifest_identity") == spec.get("expected_smoke_manifest_sha256"),
        marker.get("execution_head") == current_head,
        marker.get("protocol_byte_sha256") == spec.get("expected_smoke_protocol_sha256"),
        marker.get("protocol_semantic_sha256") == spec.get("expected_smoke_protocol_semantic_sha256"),
        marker.get("config_sha256") == spec.get("expected_smoke_config_sha256"),
        marker.get("validation_sha256") == file_sha256(paths["validation"]),
        marker.get("override_allowed") is False,
    ))
    decision = evaluate_smoke_promotion(validation, identity_ok=identity_ok)
    if not (
        decision["formal_authorized"]
        and marker.get("all_pass") is True
        and marker.get("decision") == "AUTHORIZE_FORMAL"
    ):
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: SMOKE_PASS_TO_FORMAL denied")
    return {
        "decision": decision,
        "marker": marker,
        "validation": validation,
        "output": str(output),
    }


def build_analysis_set_identity(
    manifest: dict, dataset_name: str, definition: str, include_video_ids: set[str]
) -> dict:
    """Canonical whole-video analysis-set identity derived from a frozen frame manifest."""
    videos_by_id = {str(video["video_id"]): video for video in manifest["videos"]}
    unknown = sorted(include_video_ids - set(videos_by_id))
    if unknown:
        raise ValueError(f"analysis set contains unknown video_ids: {unknown}")
    video_ids = sorted(include_video_ids)
    frame_keys = [
        [video_id, int(frame["frame"])]
        for video_id in video_ids
        for frame in videos_by_id[video_id]["frames"]
    ]
    identity_payload = {
        "dataset": dataset_name,
        "definition": definition,
        "video_ids": video_ids,
        "frame_keys": frame_keys,
    }
    return {
        "name": dataset_name,
        "definition": definition,
        "video_count": len(video_ids),
        "frame_count": len(frame_keys),
        "video_ids": video_ids,
        "identity_sha256": canonical_sha256(identity_payload),
    }


def validate_formal_contract(config: dict, protocol: dict, manifest: dict, smoke_manifest: dict) -> dict:
    """Validate preregistered Formal identities without executing any treatment or producing results."""
    if config.get("experiment_id") not in FORMAL_EXPERIMENT_IDS:
        raise ValueError(f"formal contract validator requires one of {FORMAL_EXPERIMENT_IDS}")
    if protocol.get("protocol_id") != config["experiment_id"]:
        raise ValueError("formal protocol_id mismatch")
    if protocol.get("status") not in {
        "DRAFT",
        "PREREGISTERED_BEFORE_FORMAL",
        "PREREGISTERED_BEFORE_ANY_AMENDMENT3_EXPERIMENT",
        "PREREGISTERED_BEFORE_ANY_AMENDMENT4_EXPERIMENT",
    }:
        raise ValueError("formal protocol status is invalid")
    if config["experiment_id"] == "stage5_4_amendment2_formal":
        expected_output_run_id = protocol.get("runtime_contract", {}).get("output_run_id")
        if not expected_output_run_id or config.get("output_run_id") != expected_output_run_id:
            raise ValueError("Amendment 2 Formal output run identity drifted")
        execution_identity = config.get("execution_identity", {})
        if (
            execution_identity.get("output_run_id") != expected_output_run_id
            or execution_identity.get("old_shard_reuse") is not False
            or execution_identity.get("fresh_run_required") is not True
        ):
            raise ValueError("Amendment 2 Formal execution identity policy drifted")
    if config["experiment_id"] == "stage5_4_amendment3_formal":
        expected_output_run_id = protocol.get("runtime_contract", {}).get("output_run_id")
        if not expected_output_run_id or config.get("output_run_id") != expected_output_run_id:
            raise ValueError("Amendment 3 Formal output run identity drifted")
        if protocol.get("execution_authorization") != "SMOKE_PASS_TO_FORMAL_ONLY":
            raise ValueError("Amendment 3 Formal promotion rule drifted")
    if config["experiment_id"] == "stage5_4_amendment4_formal":
        expected_output_run_id = protocol.get("runtime_contract", {}).get("output_run_id")
        if not expected_output_run_id or config.get("output_run_id") != expected_output_run_id:
            raise ValueError("Amendment 4 Formal output run identity drifted")
        if protocol.get("execution_authorization") != "SMOKE_PASS_TO_FORMAL_ONLY":
            raise ValueError("Amendment 4 Formal promotion rule drifted")
    if float(config["temporal_smoothing"]["alpha"]) != 0.5:
        raise ValueError("Formal alpha must remain exactly 0.5")
    if config["temporal_smoothing"].get("reset_rules") != [
        "NEW_VIDEO", "FRAME_GAP_GT_1", "FALLBACK"
    ]:
        raise ValueError("Formal reset rules drifted")
    if config["experiment_id"] in {
        "stage5_4_amendment2_formal", "stage5_4_amendment3_formal",
        "stage5_4_amendment4_formal",
    }:
        expected_ts2 = {
            "method": "motion_adaptive_ema_v1",
            "alpha_min": 0.5,
            "smoothing_ceiling_motion_norm": 0.1,
            "full_response_motion_norm": 0.2,
        }
        ts2 = config["temporal_smoothing"].get("ts2", {})
        if any(ts2.get(key) != value for key, value in expected_ts2.items()):
            raise ValueError("Formal TS-2 ablation schedule drifted from the frozen amendment draft")
        expected_ts3 = {
            "method": "guarded_constrained_ema_v1",
            "proposal": "ema_crop_center_v1 fixed alpha=0.5",
            "constraint": "current_frozen_primary_subject_center_inside_final_crop",
            "projection": "nearest_point_on_safe_integer_top_left_rectangle",
            "correction_feedback_to_ema_state": False,
        }
        ts3 = config["temporal_smoothing"].get("ts3", {})
        if any(ts3.get(key) != value for key, value in expected_ts3.items()):
            raise ValueError("Formal TS-3 guard definition drifted from the frozen Smoke algorithm")
        if ts3.get("extra_hyperparameters"):
            raise ValueError("Formal TS-3 must remain parameter-free")
    if config["experiment_id"] in {
        "stage5_4_amendment3_formal", "stage5_4_amendment4_formal"
    }:
        if config["temporal_smoothing"].get("ts4") != _expected_ts4_definition():
            raise ValueError("Formal TS-4 bbox guard definition drifted")
    if config["experiment_id"] == "stage5_4_amendment4_formal":
        if config["temporal_smoothing"].get("ts5") != _expected_ts5_definition():
            raise ValueError("Formal TS-5 projected-state definition drifted")
        actual_byte_sha = file_sha256(Path(config["protocol"]))
        actual_semantic_sha = _protocol_semantic_sha(protocol)
        if (
            config.get("protocol_sha256") != actual_byte_sha
            or config.get("protocol_semantic_sha256") != actual_semantic_sha
            or protocol.get("protocol_semantic_sha256") != actual_semantic_sha
            or config.get("decision_gates") != protocol.get("decision_gates")
        ):
            raise ValueError("Amendment 4 Formal protocol/gate identity drifted")
        _validate_master_preregistration(config, protocol)
    if manifest.get("manifest_sha256") != config["manifest"]["expected_manifest_sha256"]:
        raise ValueError("Stage 5.3 Formal manifest identity mismatch")
    if int(manifest.get("video_count", -1)) != int(config["manifest"]["expected_video_count"]):
        raise ValueError("Frozen Dev166 video count mismatch")
    if int(manifest.get("frame_count", -1)) != int(config["manifest"]["expected_frame_count"]):
        raise ValueError("Frozen Dev166 frame count mismatch")
    if smoke_manifest.get("manifest_sha256") != config["smoke_binding"]["expected_manifest_sha256"]:
        raise ValueError("Stage 5.4 Smoke manifest identity mismatch")

    full_spec = config["analysis_sets"]["full_dev166"]
    confirm_spec = config["analysis_sets"]["confirmatory_dev142"]
    all_video_ids = {str(video["video_id"]) for video in manifest["videos"]}
    smoke_video_ids = {str(video["video_id"]) for video in smoke_manifest["videos"]}
    configured_exclusions = set(confirm_spec["excluded_smoke_video_ids"])
    if configured_exclusions != smoke_video_ids:
        raise ValueError("configured Smoke24 video identity does not match frozen smoke manifest")
    if not smoke_video_ids <= all_video_ids:
        raise ValueError("Smoke24 is not a subset of Frozen Dev166")
    confirmatory_ids = all_video_ids - smoke_video_ids
    full = build_analysis_set_identity(
        manifest, full_spec["name"], full_spec["definition"], all_video_ids
    )
    confirmatory = build_analysis_set_identity(
        manifest, confirm_spec["name"], confirm_spec["definition"], confirmatory_ids
    )
    for spec, actual, label in (
        (full_spec, full, "Frozen Dev166"),
        (confirm_spec, confirmatory, "Confirmatory Dev142"),
    ):
        for key in ("video_count", "frame_count"):
            if actual[key] != int(spec[f"expected_{key}"]):
                raise ValueError(f"{label} {key} mismatch")
        if actual["identity_sha256"] != spec["identity_sha256"]:
            raise ValueError(f"{label} identity SHA mismatch")
    if set(confirmatory["video_ids"]) & smoke_video_ids:
        raise ValueError("Confirmatory Dev142 overlaps Smoke24")
    if int(config["heldout_lock"].get("allowed_access", -1)) != 0:
        raise ValueError("Heldout access must be zero")
    if (
        config["experiment_id"] == "stage5_4_amendment4_formal"
        and int(config.get("official_test_lock", {}).get("allowed_access", -1)) != 0
    ):
        raise ValueError("Official Test access must be zero")
    forbidden_input_tokens = ("heldout", "hard", "official_test", "official-test")
    for name, entry in config["inputs"].items():
        candidate = f"{name} {entry.get('path', '')}".lower()
        if any(token in candidate for token in forbidden_input_tokens):
            raise ValueError(f"forbidden dataset input configured: {name}")
    return {
        "validation": "PASS",
        "protocol_status": protocol["status"],
        "formal_manifest_sha256": manifest["manifest_sha256"],
        "smoke_manifest_sha256": smoke_manifest["manifest_sha256"],
        "full_dev166": full,
        "confirmatory_dev142": confirmatory,
        "confirmatory_smoke_overlap": 0,
        "heldout_access": 0,
    }


def _relative_reduction(control: float, treatment: float) -> float | None:
    if control == 0.0:
        return None
    return (control - treatment) / control


def _gate(value: float | None, threshold: float, *, minimum: bool = True) -> dict:
    passed = value is not None and (value >= threshold if minimum else value <= threshold)
    return {"value": None if value is None else round(value, 6), "threshold": threshold, "pass": passed}


def _evaluate_one_analysis_set(metrics: dict, gate_config: dict) -> dict:
    temporal = metrics["temporal_stability_pooled"]
    spatial = metrics["spatial_guardrails"]
    benefit = gate_config["temporal_benefit"]
    guardrail = gate_config["spatial_regression_guardrails"]
    displacement_reduction = _relative_reduction(
        float(temporal["ts0_displacement"]["mean"]),
        float(temporal["ts1_displacement"]["mean"]),
    )
    acceleration_reduction = _relative_reduction(
        float(temporal["ts0_acceleration"]["mean"]),
        float(temporal["ts1_acceleration"]["mean"]),
    )
    p95_control = float(temporal["ts0_displacement"]["p95"])
    p95_treatment = float(temporal["ts1_displacement"]["p95"])
    p95_regression = (p95_treatment - p95_control) / p95_control if p95_control else (
        0.0 if p95_treatment == 0.0 else float("inf")
    )
    jump0 = temporal["ts0_displacement"]["large_jump_ratios_descriptive_only"]
    jump1 = temporal["ts1_displacement"]["large_jump_ratios_descriptive_only"]
    jump_nonincrease = {
        f">{float(threshold):.2f}": {
            "ts0": jump0[f">{float(threshold):.2f}"],
            "ts1": jump1[f">{float(threshold):.2f}"],
            "pass": jump1[f">{float(threshold):.2f}"] <= jump0[f">{float(threshold):.2f}"],
        }
        for threshold in benefit["large_jump_nonincrease_thresholds"]
    }
    gt020_reduction = _relative_reduction(float(jump0[">0.20"]), float(jump1[">0.20"]))
    gt020_pass = (
        float(jump1[">0.20"]) == 0.0
        if float(jump0[">0.20"]) == 0.0
        else gt020_reduction >= float(benefit["minimum_gt_0_20_relative_reduction"])
    )
    temporal_results = {
        "mean_displacement_relative_reduction": _gate(
            displacement_reduction, float(benefit["minimum_mean_displacement_relative_reduction"])
        ),
        "mean_acceleration_relative_reduction": _gate(
            acceleration_reduction, float(benefit["minimum_mean_acceleration_relative_reduction"])
        ),
        "p95_displacement_relative_regression": _gate(
            p95_regression, float(benefit["maximum_p95_displacement_relative_regression"]), minimum=False
        ),
        "large_jump_nonincrease": {
            "values": jump_nonincrease,
            "pass": all(item["pass"] for item in jump_nonincrease.values()),
        },
        "gt_0_20_relative_reduction": {
            "value": None if gt020_reduction is None else round(gt020_reduction, 6),
            "threshold": benefit["minimum_gt_0_20_relative_reduction"],
            "pass": gt020_pass,
        },
    }
    spatial_results = {
        "mean_visible_fraction_delta": _gate(
            float(spatial["ts1"]["mean"]) - float(spatial["ts0"]["mean"]),
            float(guardrail["minimum_mean_visible_fraction_delta"]),
        ),
        "visible_ge_0_90_delta": _gate(
            float(spatial["ts1"]["thresholds"][">=0.90"])
            - float(spatial["ts0"]["thresholds"][">=0.90"]),
            float(guardrail["minimum_visible_ge_0_90_delta"]),
        ),
        "subject_center_containment_delta": _gate(
            float(spatial["ts1"]["subject_center_inside_rate"])
            - float(spatial["ts0"]["subject_center_inside_rate"]),
            float(guardrail["minimum_subject_center_containment_delta"]),
        ),
        "strong_off_center_mean_visible_delta": _gate(
            float(spatial["strata"]["strongly_off_center"]["ts1_visible"]["mean"])
            - float(spatial["strata"]["strongly_off_center"]["ts0_visible"]["mean"]),
            float(guardrail["minimum_strong_off_center_mean_visible_delta"]),
        ),
    }
    return {
        "temporal_benefit": temporal_results,
        "spatial_regression_guardrails": spatial_results,
        "pass": all(item["pass"] for item in temporal_results.values())
        and all(item["pass"] for item in spatial_results.values()),
    }


def evaluate_formal_scientific_gates(analysis_metrics: dict, gate_config: dict) -> dict:
    """Apply the same preregistered gates to Full Dev166 and Confirmatory Dev142."""
    required = ("full_dev166", "confirmatory_dev142")
    results = {name: _evaluate_one_analysis_set(analysis_metrics[name], gate_config) for name in required}
    return {"analysis_sets": results, "all_pass": all(item["pass"] for item in results.values())}


def evaluate_amendment2_scientific_gates(
    records: list[dict],
    pooled: dict,
    guardrails: dict,
    gate_config: dict,
    role: str = "AMENDMENT2_SMOKE_PROMOTION_REVIEW / frozen Stage 5.4 gates applied to TS-3",
) -> dict:
    """Apply frozen TS-3 gates and preregistered TS-3-vs-TS-1/TS-2 checks."""
    ts3_gate_result = _evaluate_one_analysis_set(
        build_analysis_metrics(_project_treatment(records, "ts3")), gate_config
    )
    strong = guardrails["strata"]["strongly_off_center"]
    ts2_displacement = temporal_summary(pooled["ts2_displacement"])["mean"]
    ts3_displacement = temporal_summary(pooled["ts3_displacement"])["mean"]
    ts2_acceleration = temporal_summary(pooled["ts2_acceleration"])["mean"]
    ts3_acceleration = temporal_summary(pooled["ts3_acceleration"])["mean"]
    ts2_jump = large_jump_ratios(pooled["ts2_displacement"])[">0.20"]
    ts3_jump = large_jump_ratios(pooled["ts3_displacement"])[">0.20"]
    eligible_ts3 = [record for record in records if not record["ts0"]["fallback"]]
    mechanism_checks = {
        "every_nonfallback_ts3_contains_primary_center": {
            "eligible": len(eligible_ts3),
            "contained": sum(record["ts3"]["subject_center_inside"] for record in eligible_ts3),
            "pass": bool(eligible_ts3)
            and all(record["ts3"]["subject_center_inside"] for record in eligible_ts3),
        },
        "strong_visible_delta_ts3_strictly_better_than_ts1": {
            "ts1_delta": strong["delta_mean"],
            "ts3_delta": strong["ts3_delta_mean"],
            "pass": strong["ts3_delta_mean"] > strong["delta_mean"],
        },
        "mean_displacement_ts3_no_worse_than_ts2": {
            "ts2": ts2_displacement,
            "ts3": ts3_displacement,
            "pass": ts3_displacement is not None
            and ts2_displacement is not None
            and ts3_displacement <= ts2_displacement,
        },
        "mean_acceleration_ts3_no_worse_than_ts2": {
            "ts2": ts2_acceleration,
            "ts3": ts3_acceleration,
            "pass": ts3_acceleration is not None
            and ts2_acceleration is not None
            and ts3_acceleration <= ts2_acceleration,
        },
        "gt_0_20_jump_ts3_no_worse_than_ts2": {
            "ts2": ts2_jump,
            "ts3": ts3_jump,
            "pass": ts3_jump <= ts2_jump,
        },
    }
    return {
        "role": role,
        "ts3_vs_ts0": ts3_gate_result,
        "amendment2_mechanism_checks": mechanism_checks,
        "all_pass": ts3_gate_result["pass"]
        and all(item["pass"] for item in mechanism_checks.values()),
    }


def evaluate_amendment2_formal_scientific_gates(
    analysis_set_records: dict[str, list[dict]], gate_config: dict
) -> dict:
    """Apply the frozen Stage 5.4 gates and Amendment 2 mechanism checks per analysis set.

    Every gate is evaluated independently on Full Dev166 and Confirmatory Dev142;
    both analysis sets must pass. The gate thresholds and mechanism checks are the
    ones frozen before the Amendment 2 smoke; nothing is tuned from Formal output.
    """
    results = {}
    for name, records in analysis_set_records.items():
        pooled = pooled_distributions(records)
        guardrails = spatial_metrics_for_all_treatments(records)
        results[name] = evaluate_amendment2_scientific_gates(
            records,
            pooled,
            guardrails,
            gate_config,
            role="AMENDMENT2_FORMAL_PREREGISTERED_GATES / frozen Stage 5.4 gates applied to TS-3",
        )
    return {
        "analysis_sets": results,
        "all_pass": all(item["all_pass"] for item in results.values()),
    }


def evaluate_amendment3_scientific_gates(
    records: list[dict],
    pooled: dict,
    guardrails: dict,
    gate_config: dict,
    *,
    role: str = "AMENDMENT3_SMOKE_PREREGISTERED_GATES / TS-4",
) -> dict:
    """Frozen TS-4 gates plus mechanism and temporal-trade-off checks."""
    ts4_gate_result = _evaluate_one_analysis_set(
        build_analysis_metrics(_project_treatment(records, "ts4")), gate_config
    )
    strong = guardrails["strata"]["strongly_off_center"]
    eligible = [record for record in records if not record["ts0"]["fallback"]]
    ts2_displacement = temporal_summary(pooled["ts2_displacement"])["mean"]
    ts4_displacement = temporal_summary(pooled["ts4_displacement"])["mean"]
    ts2_acceleration = temporal_summary(pooled["ts2_acceleration"])["mean"]
    ts4_acceleration = temporal_summary(pooled["ts4_acceleration"])["mean"]
    ts2_jump = large_jump_ratios(pooled["ts2_displacement"])[">0.20"]
    ts4_jump = large_jump_ratios(pooled["ts4_displacement"])[">0.20"]
    ts3_visible90 = float(guardrails["ts3"]["thresholds"][">=0.90"])
    ts4_visible90 = float(guardrails["ts4"]["thresholds"][">=0.90"])
    mechanism_checks = {
        "ts4_bbox_visibility_pointwise_no_worse_than_ts3": {
            "eligible": len(eligible),
            "violations": sum(
                float(record["ts4"]["subject_visible_fraction"]) + 1e-9
                < float(record["ts3"]["subject_visible_fraction"])
                for record in eligible
            ),
            "pass": bool(eligible)
            and all(
                float(record["ts4"]["subject_visible_fraction"]) + 1e-9
                >= float(record["ts3"]["subject_visible_fraction"])
                for record in eligible
            ),
        },
        "strong_visible_delta_ts4_strictly_better_than_ts3": {
            "ts3_delta": strong["ts3_delta_mean"],
            "ts4_delta": strong["ts4_delta_mean"],
            "pass": strong["ts4_delta_mean"] > strong["ts3_delta_mean"],
        },
        "visible_ge_0_90_ts4_no_worse_than_ts3": {
            "ts3": ts3_visible90,
            "ts4": ts4_visible90,
            "pass": ts4_visible90 >= ts3_visible90,
        },
        "bbox_objective_realized_on_every_eligible_frame": {
            "eligible": len(eligible),
            "violations": sum(
                abs(
                    float(record["ts4"]["visible_fraction_after_guard"])
                    - float(record["ts4"]["subject_visible_fraction"])
                ) > 1e-6
                or float(record["ts4"]["visible_gain"]) < -1e-9
                for record in eligible
            ),
            "pass": bool(eligible)
            and all(
                abs(
                    float(record["ts4"]["visible_fraction_after_guard"])
                    - float(record["ts4"]["subject_visible_fraction"])
                ) <= 1e-6
                and float(record["ts4"]["visible_gain"]) >= -1e-9
                for record in eligible
            ),
        },
        "mean_displacement_ts4_no_worse_than_ts2": {
            "ts2": ts2_displacement, "ts4": ts4_displacement,
            "pass": ts4_displacement is not None and ts2_displacement is not None
            and ts4_displacement <= ts2_displacement,
        },
        "mean_acceleration_ts4_no_worse_than_ts2": {
            "ts2": ts2_acceleration, "ts4": ts4_acceleration,
            "pass": ts4_acceleration is not None and ts2_acceleration is not None
            and ts4_acceleration <= ts2_acceleration,
        },
        "gt_0_20_jump_ts4_no_worse_than_ts2": {
            "ts2": ts2_jump, "ts4": ts4_jump, "pass": ts4_jump <= ts2_jump,
        },
    }
    all_pass = ts4_gate_result["pass"] and all(
        item["pass"] for item in mechanism_checks.values()
    )
    return {
        "status": "PASS" if all_pass else "FAIL",
        "role": role,
        "ts4_vs_ts0": ts4_gate_result,
        "amendment3_mechanism_checks": mechanism_checks,
        "all_pass": all_pass,
    }


def evaluate_amendment3_formal_scientific_gates(
    analysis_set_records: dict[str, list[dict]], gate_config: dict
) -> dict:
    results = {}
    for name, records in analysis_set_records.items():
        results[name] = evaluate_amendment3_scientific_gates(
            records,
            pooled_distributions(records),
            spatial_metrics_for_all_treatments(records),
            gate_config,
            role="AMENDMENT3_FORMAL_PREREGISTERED_GATES / TS-4",
        )
    all_pass = all(item["all_pass"] for item in results.values())
    return {
        "status": "PASS" if all_pass else "FAIL",
        "analysis_sets": results,
        "all_pass": all_pass,
    }


def evaluate_amendment4_scientific_gates(
    records: list[dict],
    pooled: dict,
    guardrails: dict,
    gate_config: dict,
    *,
    role: str = "AMENDMENT4_SMOKE_PREREGISTERED_GATES / TS-5",
) -> dict:
    """Frozen 9 gates plus preregistered TS-5 mechanism invariants."""
    ts5_gate_result = _evaluate_one_analysis_set(
        build_analysis_metrics(_project_treatment(records, "ts5")), gate_config
    )
    eligible = [record for record in records if not record["ts0"]["fallback"]]
    def canonical_projection_matches(record: dict) -> bool:
        item = record["ts5"]
        proposal_x = min(max(
            math.floor(float(item["proposal_center_x"]) - int(item["w"]) / 2.0), 0
        ), int(record["image_width"]) - int(item["w"]))
        proposal_y = min(max(
            math.floor(float(item["proposal_center_y"]) - int(item["crop_h"]) / 2.0), 0
        ), math.floor(float(record["image_height"]) - float(item["h"])))
        projected_x = min(max(proposal_x, int(item["safe_x_min"])), int(item["safe_x_max"]))
        projected_y = min(max(proposal_y, int(item["safe_y_min"])), int(item["safe_y_max"]))
        return (projected_x, projected_y) == (int(item["x"]), int(item["y"]))

    checks = {
        "projected_state_placement_inside_ts4_feasible_set_every_frame": {
            "eligible": len(eligible),
            "violations": sum(not (
                int(record["ts5"]["safe_x_min"]) <= int(record["ts5"]["x"]) <= int(record["ts5"]["safe_x_max"])
                and int(record["ts5"]["safe_y_min"]) <= int(record["ts5"]["y"]) <= int(record["ts5"]["safe_y_max"])
            ) for record in eligible),
        },
        "output_placement_equals_projected_state_placement": {
            "eligible": len(eligible),
            "violations": sum(float(record["ts5"]["state_output_residual_l1"]) != 0.0 for record in eligible),
        },
        "bbox_maximum_overlap_objective_realized_every_frame": {
            "eligible": len(eligible),
            "violations": sum(abs(
                float(record["ts5"]["visible_fraction_after_guard"])
                - float(record["ts5"]["subject_visible_fraction"])
            ) > 1e-6 for record in eligible),
        },
        "ts4_projection_operator_reused_exactly": {
            "eligible": len(eligible),
            "violations": sum(not canonical_projection_matches(record) for record in eligible),
        },
        "reset_and_fallback_semantics_equal_ts4": {
            "frames": len(records),
            "violations": sum(
                record["ts5"]["reset_reason"] != record["ts4"]["reset_reason"]
                or (
                    record["ts0"]["fallback"]
                    and (record["ts5"]["x"], record["ts5"]["y"])
                    != (record["ts4"]["x"], record["ts4"]["y"])
                )
                for record in records
            ),
        },
        "bbox_visibility_pointwise_equal_ts4": {
            "eligible": len(eligible),
            "violations": sum(abs(
                float(record["ts5"]["subject_visible_fraction"])
                - float(record["ts4"]["subject_visible_fraction"])
            ) > 1e-6 for record in eligible),
        },
    }
    for check in checks.values():
        if "pass" not in check:
            check["pass"] = bool(check.get("eligible", check.get("frames", 0))) and check["violations"] == 0
    all_pass = ts5_gate_result["pass"] and all(check["pass"] for check in checks.values())
    return {
        "status": "PASS" if all_pass else "FAIL",
        "role": role,
        "ts5_vs_ts0": ts5_gate_result,
        "amendment4_mechanism_checks": checks,
        "all_pass": all_pass,
    }


def evaluate_amendment4_formal_scientific_gates(
    analysis_set_records: dict[str, list[dict]], gate_config: dict
) -> dict:
    results = {
        name: evaluate_amendment4_scientific_gates(
            records,
            pooled_distributions(records),
            spatial_metrics_for_all_treatments(records),
            gate_config,
            role="AMENDMENT4_FORMAL_PREREGISTERED_GATES / TS-5",
        )
        for name, records in analysis_set_records.items()
    }
    all_pass = bool(results) and all(result["all_pass"] for result in results.values())
    return {
        "status": "PASS" if all_pass else "FAIL",
        "analysis_sets": results,
        "all_pass": all_pass,
    }


def validate_formal_frozen_dependencies(
    config: dict, bindings: list[InputBinding], inputs, manifest: dict
) -> dict:
    """Read-only identity checks for frozen Stage 5.1/5.2/5.3 dependencies."""
    by_name = {binding.name: binding for binding in bindings}
    raw_dir = by_name["stage5_2_raw_detector"].path
    if not raw_dir.is_dir():
        raise FrozenInputError(f"raw shard dir missing: {raw_dir}")
    expected_video_ids = {str(video["video_id"]) for video in manifest["videos"]}
    raw_video_ids = {path.stem for path in raw_dir.glob("*.jsonl")}
    if raw_video_ids != expected_video_ids:
        raise FrozenInputError("Stage 5.2 raw shard video identity mismatch vs Frozen Dev166")

    summary = json.loads(by_name["stage5_2_full_dev_summary"].path.read_text(encoding="utf-8"))
    expected_summary = {
        "videos": int(manifest["video_count"]),
        "frames": int(manifest["frame_count"]),
        "missing": 0,
        "extra": 0,
        "duplicates": 0,
        "model_error_frames": 0,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise FrozenInputError(f"Stage 5.2 frozen summary mismatch: {key}")
    semantic = summary.get("semantic_hashes", {})
    if semantic.get("policy_v1_decisions") != config["expected_policy_semantic_sha256"]:
        raise FrozenInputError("Stage 5.2 policy semantic SHA mismatch in frozen summary")
    if semantic.get("raw_candidates") != config["expected_raw_semantic_sha256"]:
        raise FrozenInputError("Stage 5.2 raw semantic SHA mismatch in frozen summary")
    if inputs.input_hashes.get("stage5_2_policy_artifact") != config["expected_policy_semantic_sha256"]:
        raise FrozenInputError("loaded Stage 5.2 policy semantic SHA mismatch")

    stage53 = json.loads(by_name["stage5_3_final_freeze"].path.read_text(encoding="utf-8"))
    stage52 = json.loads(by_name["stage5_2_final_freeze"].path.read_text(encoding="utf-8"))
    if stage53.get("status") != "FINAL_FROZEN" or stage52.get("status") != "FINAL_FROZEN":
        raise FrozenInputError("Stage 5.2/5.3 freeze status mismatch")
    if stage53.get("bindings", {}).get("formal_manifest_semantic_sha256") != manifest["manifest_sha256"]:
        raise FrozenInputError("Stage 5.3 freeze does not bind the configured Formal manifest")
    if stage52.get("artifact_verification", {}).get("semantic_sha256", {}).get("raw_candidates") != config[
        "expected_raw_semantic_sha256"
    ]:
        raise FrozenInputError("Stage 5.2 freeze raw semantic identity mismatch")
    return {
        "stage5_3_freeze_status": stage53["status"],
        "stage5_3_formal_manifest_sha256": manifest["manifest_sha256"],
        "stage5_2_freeze_status": stage52["status"],
        "stage5_2_policy_semantic_sha256": semantic["policy_v1_decisions"],
        "stage5_2_raw_semantic_sha256": semantic["raw_candidates"],
        "stage5_2_raw_shards": len(raw_video_ids),
        "frozen_frames": summary["frames"],
    }


def load_frozen_ts0_predictions(
    bindings: list[InputBinding], manifest: dict, *, exact_identity: bool = True
) -> dict[str, dict[int, list[int]]]:
    """Load byte-pinned Stage 5.3 CMP-1 crops and verify manifest frame identity."""
    binding = next((item for item in bindings if item.name == "stage5_3_frozen_cmp1"), None)
    if binding is None:
        raise FrozenInputError("stage5_3_frozen_cmp1 binding is required for Formal")
    records = [
        json.loads(line)
        for line in binding.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_video: dict[str, dict[int, list[int]]] = {}
    for record in records:
        video_id = str(record["video_id"])
        if video_id in by_video:
            raise FrozenInputError(f"duplicate Stage 5.3 CMP-1 video: {video_id}")
        by_video[video_id] = {
            int(item["frame"]): [int(value) for value in item["bboxes"]]
            for item in record["predictions"]
        }
    expected = {
        (str(video["video_id"]), int(frame["frame"]))
        for video in manifest["videos"]
        for frame in video["frames"]
    }
    actual = {(video_id, frame) for video_id, frames in by_video.items() for frame in frames}
    if (exact_identity and (expected != actual or len(by_video) != int(manifest["video_count"]))) or (
        not exact_identity and not expected <= actual
    ):
        raise FrozenInputError("Stage 5.3 frozen CMP-1 frame identity mismatch")
    if any(len(bbox) != 3 for frames in by_video.values() for bbox in frames.values()):
        raise FrozenInputError("Stage 5.3 frozen CMP-1 bbox contract mismatch")
    expected_video_ids = {video_id for video_id, _frame in expected}
    return {
        video_id: {frame: bbox for frame, bbox in by_video[video_id].items() if (video_id, frame) in expected}
        for video_id in sorted(expected_video_ids)
    }


def build_analysis_metrics(records: list[dict]) -> dict:
    pooled = pooled_distributions(records)
    return {
        "videos": len({record["video_id"] for record in records}),
        "frames": len(records),
        "temporal_stability_pooled": {key: summary_payload(values) for key, values in pooled.items()},
        "spatial_guardrails": spatial_guardrail_metrics(records),
    }


def build_analysis_metrics_all_treatments(records: list[dict]) -> dict:
    """Analysis-set metrics for all configured treatments plus guard diagnostics."""
    pooled = pooled_distributions(records)
    metrics = {
        "videos": len({record["video_id"] for record in records}),
        "frames": len(records),
        "temporal_stability_pooled": {key: summary_payload(values) for key, values in pooled.items()},
        "spatial_guardrails": spatial_metrics_for_all_treatments(records),
        "guard_projection_diagnostics": guard_projection_diagnostics(records, "ts3"),
    }
    if records and "ts4" in records[0]:
        metrics["ts4_guard_projection_diagnostics"] = guard_projection_diagnostics(
            records, "ts4"
        )
        metrics["bbox_guard_diagnostics"] = bbox_guard_diagnostics(records)
    if records and "ts5" in records[0]:
        metrics["ts5_guard_projection_diagnostics"] = guard_projection_diagnostics(
            records, "ts5"
        )
        metrics["ts5_bbox_guard_diagnostics"] = bbox_guard_diagnostics(
            records, side="ts5"
        )
        metrics["projected_state_attribution_diagnostics"] = (
            projected_state_attribution_diagnostics(records)
        )
    return metrics


def aggregate_formal_multi_subject_diagnostics(
    analysis_metrics: dict,
    multi_subject: dict,
    confirmatory_ids: set[str],
    *,
    four_arm: bool,
    candidate_sides: tuple[str, ...] = ("ts1", "ts2", "ts3"),
) -> dict:
    """Attach Full/Confirmatory summaries for explicitly mapped treatment sides."""
    by_comparison = multi_subject if four_arm else {"ts0_vs_ts1": multi_subject}
    for comparison, payload in by_comparison.items():
        if comparison not in MULTI_SUBJECT_COMPARISON_CANDIDATES or (
            MULTI_SUBJECT_COMPARISON_CANDIDATES.get(comparison) not in candidate_sides
        ):
            raise ValueError(f"unknown multi-subject comparison: {comparison}")
        if not isinstance(payload, dict) or "rows" not in payload:
            raise ValueError(f"multi-subject comparison missing rows: {comparison}")
        candidate_side = MULTI_SUBJECT_COMPARISON_CANDIDATES[comparison]
        full_summary = {
            key: value
            for key, value in summarize_multi_subject_rows(
                payload["rows"], candidate_side=candidate_side
            ).items()
            if key != "rows"
        }
        confirmatory_summary = {
            key: value
            for key, value in summarize_multi_subject_rows(
                [row for row in payload["rows"] if row["video_id"] in confirmatory_ids],
                candidate_side=candidate_side,
            ).items()
            if key != "rows"
        }
        if four_arm:
            analysis_metrics["full_dev166"].setdefault(
                "multi_subject_observation_only", {}
            )[comparison] = full_summary
            analysis_metrics["confirmatory_dev142"].setdefault(
                "multi_subject_observation_only", {}
            )[comparison] = confirmatory_summary
        else:
            analysis_metrics["full_dev166"]["multi_subject_observation_only"] = full_summary
            analysis_metrics["confirmatory_dev142"][
                "multi_subject_observation_only"
            ] = confirmatory_summary
    return analysis_metrics


def resolve_experiment_paths(environment: EnvironmentPaths, config: dict):
    """Resolve an evidence-preserving run directory while retaining registry identity."""
    output_run_id = config.get("output_run_id", config["experiment_id"])
    return environment.for_experiment(config["stage"], output_run_id)


def amendment4_execution_preflight(
    config: dict,
    environment: EnvironmentPaths,
    bindings: list[InputBinding],
    paths,
    *,
    resume: bool,
) -> dict:
    """Fail before run creation unless every future TS-5 execution identity closes."""
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(environment.repo), *args], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    expected_head = os.environ.get("AIC_EXPECTED_GIT_HEAD")
    windows_origin_head = os.environ.get("AIC_WINDOWS_ORIGIN_MASTER_HEAD")
    windows_clean = os.environ.get("AIC_WINDOWS_GIT_CLEAN") == "1"
    try:
        head = git("rev-parse", "HEAD")
        origin_head = git("rev-parse", "origin/master")
        branch = git("branch", "--show-current")
        dirty = git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: Git identity unavailable") from exc
    git_checks = {
        "A_windows_repo_head_supplied": bool(expected_head),
        "B_origin_master_head_matches_windows": origin_head == windows_origin_head == expected_head,
        "C_autodl_execution_head_matches_windows": head == expected_head,
        "E_windows_worktree_clean_attested": windows_clean,
        "F_autodl_worktree_clean": not dirty,
        "branch_master": branch == "master",
    }
    if not all(git_checks.values()):
        raise FrozenInputError(f"BLOCKED_BEFORE_EXECUTION: Git preflight failed: {git_checks}")

    manifest_path = paths.output / "run_manifest.json"
    if resume and not manifest_path.is_file():
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: resume requested without run_manifest")
    if paths.output.exists() and not manifest_path.is_file():
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: output directory conflicts with no run identity")
    if manifest_path.is_file() and not resume:
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: existing run requires explicit --resume")

    resume_checks = {
        "D_run_manifest_execution_head": True,
        "W_output_directory_no_conflict": not paths.output.exists() or resume,
        "X_resume_identity_legal": not resume,
    }
    if resume:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_resume = {
            "experiment_id": config["experiment_id"],
            "stage": config["stage"],
            "run_type": config["run_type"],
            "git_head": head,
            "config_sha256": file_sha256(Path(config["config_repo_path"])),
            "protocol_sha256": file_sha256(Path(config["protocol"])),
        }
        identity_ok = all(prior.get(key) == value for key, value in expected_resume.items())
        resume_checks.update({
            "D_run_manifest_execution_head": prior.get("git_head") == head,
            "X_resume_identity_legal": identity_ok,
        })
        if not all(resume_checks.values()):
            raise FrozenInputError(
                f"BLOCKED_BEFORE_EXECUTION: run identity failed: {resume_checks}"
            )

    unrelated_processes_absent = True
    if os.name != "nt":
        try:
            subprocess.check_output(
                ["pgrep", "-af", "[v]llm|[q]wen"], text=True,
                stderr=subprocess.DEVNULL,
            )
            unrelated_processes_absent = False
        except FileNotFoundError:
            unrelated_processes_absent = False
        except subprocess.CalledProcessError:
            pass
    if not unrelated_processes_absent:
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: active Qwen/vLLM process or unavailable process audit")

    protocol = json.loads(Path(config["protocol"]).read_text(encoding="utf-8"))
    manifest = load_frozen_manifest(config, bindings)
    inputs = load_frozen_inputs(
        [binding for binding in bindings if binding.format != "raw_shard_dir"],
        include_raw=False,
        policy_semantic_expectation=config.get("expected_policy_semantic_sha256"),
    )
    if config["experiment_id"] == "stage5_4_amendment4_formal":
        smoke_manifest = load_manifest_binding(config["smoke_binding"], bindings)
        contract = validate_formal_contract(config, protocol, manifest, smoke_manifest)
        promotion = load_amendment4_promotion_evidence(config, environment)
        frozen = validate_formal_frozen_dependencies(config, bindings, inputs, manifest)
        frozen_ts0 = load_frozen_ts0_predictions(bindings, manifest)
        frozen_ts0_frames = sum(len(frames) for frames in frozen_ts0.values())
        frozen["stage5_3_cmp1_videos"] = len(frozen_ts0)
        frozen["stage5_3_cmp1_frames"] = frozen_ts0_frames
        expected_counts = {
            "dev166": contract["full_dev166"]["frame_count"] == 51256,
            "dev142": contract["confirmatory_dev142"]["frame_count"] == 43820,
            "smoke_overlap_zero": contract["confirmatory_smoke_overlap"] == 0,
            "ts0_dev166": len(frozen_ts0) == 166 and frozen_ts0_frames == 51256,
        }
    else:
        contract = validate_amendment4_smoke_contract(config, protocol, manifest)
        promotion = None
        frozen_ts0 = load_frozen_ts0_predictions(
            bindings, manifest, exact_identity=False
        )
        frozen_ts0_frames = sum(len(frames) for frames in frozen_ts0.values())
        frozen = {
            "stage5_3_cmp1_videos": len(frozen_ts0),
            "stage5_3_cmp1_frames": frozen_ts0_frames,
        }
        expected_counts = {
            "smoke24": contract["frames"] == 1080,
            "ts0_smoke24": len(frozen_ts0) == 24 and frozen_ts0_frames == 1080,
        }
    if not all(expected_counts.values()):
        raise FrozenInputError(f"BLOCKED_BEFORE_EXECUTION: dataset identity failed: {expected_counts}")
    master = json.loads(Path(config["master_preregistration"]).read_text(encoding="utf-8"))
    prereg_role = "formal" if config["experiment_id"].endswith("_formal") else "smoke"
    master_binding = master["artifacts"][prereg_role]
    for destination in (paths.output.parent, paths.logs.parent, paths.cache.parent, paths.tmp.parent):
        destination.mkdir(parents=True, exist_ok=True)
    destinations_ready = all(
        destination.is_dir() and os.access(destination, os.W_OK)
        for destination in (paths.output.parent, paths.logs.parent, paths.cache.parent, paths.tmp.parent)
    )
    if not destinations_ready:
        raise FrozenInputError("BLOCKED_BEFORE_EXECUTION: report/artifact destinations unavailable")
    identity_checks = {
        "G_protocol_byte_hash": config["protocol_sha256"] == file_sha256(Path(config["protocol"])),
        "H_protocol_semantic_hash": config["protocol_semantic_sha256"] == _protocol_semantic_sha(protocol),
        "I_config_hash": master_binding["config_sha256"]
        == file_sha256(Path(config["config_repo_path"])),
        "J_method_identity": config["temporal_smoothing"]["ts5"] == _expected_ts5_definition(),
        "K_smoke_promotion_identity": promotion is not None if config["experiment_id"].endswith("_formal") else True,
        "L_input_manifests": True,
        "M_ts0_frozen_baseline": bool(frozen),
        "N_stage5_1_frozen_inputs": True,
        "O_stage5_2_frozen_inputs": True,
        "P_stage5_3_frozen_inputs": True,
        "Q_dev166_identity": expected_counts.get("dev166", True),
        "R_dev142_identity": expected_counts.get("dev142", True),
        "S_smoke24_identity": expected_counts.get("smoke24", expected_counts.get("smoke_overlap_zero", True)),
        "T_heldout_access_zero": True,
        "U_official_test_access_zero": True,
        "V_no_active_qwen_vllm": unrelated_processes_absent,
        "Y_report_artifact_destinations_ready": destinations_ready,
    }
    if not all(identity_checks.values()):
        raise FrozenInputError(f"BLOCKED_BEFORE_EXECUTION: frozen identity failed: {identity_checks}")
    return {
        "validation": "PASS",
        "git": git_checks,
        "run_lifecycle": resume_checks,
        "identity": identity_checks,
        "dataset_counts": expected_counts,
        "frozen_inputs": frozen,
        "promotion": promotion,
        "heldout_access": 0,
        "official_test_access": 0,
        "output_lifecycle": "RESUME_IDENTITY_REQUIRED" if resume else "FRESH_ONLY",
        "report_destinations_ready": True,
    }


def render_amendment3_experiment_report(
    output_path: Path,
    *,
    config: dict,
    summary: dict,
    metrics: dict,
    validation: dict,
    runtime: dict,
    smoke_result: dict | None = None,
) -> None:
    """Write the preregistered Amendment 3 human-report contract in one pass."""
    if config.get("experiment_id") not in {
        "stage5_4_amendment3_smoke", "stage5_4_amendment3_formal"
    }:
        return
    formal = config["experiment_id"].endswith("_formal")
    final_status = (
        "READY_FOR_HUMAN_REVIEW"
        if formal and validation["status"] == "PASS"
        else "NOT_READY_FOR_FORMAL"
        if not formal and validation["status"] != "PASS"
        else "SMOKE_PASS_TO_FORMAL"
        if not formal
        else "NOT_READY_FOR_FREEZE"
    )
    analysis_sets = metrics.get("analysis_sets", {})
    section_payloads = {
        "History": config.get("history", {}),
        "Method": summary.get("ts4", {}),
        "TS-0~TS-4": {side: summary.get(side) for side in ("ts0", "ts1", "ts2", "ts3", "ts4")},
        "Smoke result": smoke_result or ({"this_run": validation} if not formal else {"available": False}),
        "Full Dev166": analysis_sets.get("full_dev166", {}),
        "Dev142": analysis_sets.get("confirmatory_dev142", {}),
        "Temporal": metrics.get("temporal_stability_pooled", {}),
        "Spatial": metrics.get("spatial_guardrails", {}),
        "Strata": metrics.get("spatial_guardrails", {}).get("strata", {}),
        "BBox diagnostics": metrics.get("bbox_guard_diagnostics")
        or analysis_sets.get("full_dev166", {}).get("bbox_guard_diagnostics", {}),
        "Guard diagnostics": {
            "ts3": metrics.get("guard_projection_diagnostics"),
            "ts4": metrics.get("ts4_guard_projection_diagnostics"),
        },
        "Multi-subject": {
            name: payload.get("multi_subject_observation_only", {})
            for name, payload in analysis_sets.items()
        },
        "Fallback": {"rule": "TS-1/2/3/4 fallback placement equals TS-0", "gate": validation.get("gates", {})},
        "Engineering gates": validation.get("gates", {}),
        "Scientific gates": validation.get("scientific_gates", {}),
        "Mechanism checks": validation.get("scientific_gates", {}),
        "Limitations": config.get("limitations", []),
        "Conclusion": {"status": final_status, "automatic_final_freeze": False},
        "Artifact paths": {"output": str(output_path.parent), "runtime": runtime},
        "Git/Protocol/Config identities": {
            "protocol": config.get("protocol"),
            "protocol_sha256": config.get("protocol_sha256"),
            "config_identity_captured_by_run_manifest": True,
        },
    }
    lines = [
        f"# Stage 5.4 Amendment 3 — {config['experiment_id']} 自动实验报告",
        "",
        f"- 状态：**{final_status}**",
        "- 本报告由冻结 runner 一次性生成；不得据此自动 FINAL_FROZEN。",
        "",
    ]
    for section in AMENDMENT3_FORMAL_REPORT_SECTIONS:
        lines.extend(
            [
                f"## {section}",
                "",
                "```json",
                json.dumps(section_payloads[section], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_amendment4_experiment_report(
    output_path: Path,
    *,
    config: dict,
    summary: dict,
    metrics: dict,
    validation: dict,
    runtime: dict,
    smoke_result: dict | None = None,
    promotion_marker: dict | None = None,
) -> None:
    """Write the frozen Amendment 4 Smoke/Formal report without invented results."""
    if config.get("experiment_id") not in {
        "stage5_4_amendment4_smoke", "stage5_4_amendment4_formal"
    }:
        return
    formal = config["experiment_id"].endswith("_formal")
    adjudication = (
        adjudicate_ts5_formal(validation)
        if formal
        else {
            "status": "PROMOTE_TO_FORMAL"
            if validation["status"] == "PASS"
            else "STOP_NO_FORMAL_NO_AMENDMENT5",
            "all_pass": validation["status"] == "PASS",
        }
    )
    analysis_sets = metrics.get("analysis_sets", {})
    payloads = {
        "Identity": {
            "experiment_id": config["experiment_id"],
            "manifest": config.get("manifest"),
            "protocol_byte_sha256": config.get("protocol_sha256"),
            "protocol_semantic_sha256": config.get("protocol_semantic_sha256"),
        },
        "Method": summary.get("ts5", {}),
        "TS-0~TS-5": {side: summary.get(side) for side in ("ts0", "ts1", "ts2", "ts3", "ts4", "ts5")},
        "Smoke promotion": promotion_marker or smoke_result or {"available": False},
        "Full Dev166": analysis_sets.get("full_dev166", {}),
        "Dev142": analysis_sets.get("confirmatory_dev142", {}),
        "Temporal gates": validation.get("scientific_gates", {}),
        "Spatial gates": validation.get("scientific_gates", {}),
        "Mechanism checks": validation.get("scientific_gates", {}),
        "Diagnostic-only attribution": metrics.get("ts5_projected_state_attribution", {}),
        "TS-4 comparison": {
            "ts4_guard": metrics.get("ts4_guard_projection_diagnostics"),
            "ts5_guard": metrics.get("ts5_guard_projection_diagnostics"),
            "ts4_bbox": metrics.get("bbox_guard_diagnostics"),
            "ts5_bbox": metrics.get("ts5_bbox_guard_diagnostics"),
        },
        "Determinism": validation.get("deterministic_replay"),
        "Engineering validation": validation.get("gates", {}),
        "Final adjudication": adjudication,
        "Artifact paths": {"output": str(output_path.parent), "runtime": runtime},
        "Git/Protocol/Config identities": {
            "captured_by_run_manifest": True,
            "protocol": config.get("protocol"),
            "config": config.get("config_repo_path"),
        },
    }
    lines = [
        f"# Stage 5.4 Amendment 4 — {config['experiment_id']} 自动实验报告",
        "",
        f"- 状态：**{adjudication['status']}**",
        "- 诊断性 attribution 不参与 9 项科学 gate；不得据此调参。",
        "- 任一失败均终止 Stage 5.4；禁止 Amendment 5。",
        "",
    ]
    for section in AMENDMENT4_REPORT_SECTIONS:
        lines.extend([
            f"## {section}", "", "```json",
            json.dumps(payloads[section], ensure_ascii=False, indent=2),
            "```", "",
        ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> int:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    environment = EnvironmentPaths.from_json(args.environment)
    paths = resolve_experiment_paths(environment, config)
    if args.dry_run:
        print(json.dumps({
            "experiment_id": config["experiment_id"],
            "paths": {key: str(value) for key, value in vars(paths).items()},
            "action": "NONE",
        }, indent=2))
        return 0

    bindings = resolve_bindings(config, environment)
    context = RunContext(
        config["experiment_id"],
        config["stage"],
        config["run_type"],
        environment.repo,
        paths,
        args.config,
        Path(config["protocol"]),
        {
            f"input:{name}": (entry.get("sha256") or "UNPINNED_VERIFIED_AT_RUNTIME")
            for name, entry in config["inputs"].items()
        },
        {"name": "none", "gpu": False, "note": "frozen Stage 5.3 artifact consumption only; CPU temporal smoothing"},
    )
    if args.validate_only:
        try:
            manifest = load_frozen_manifest(config, bindings)
            inputs = load_frozen_inputs(
                [b for b in bindings if b.format != "raw_shard_dir"],
                include_raw=False,
                policy_semantic_expectation=config.get("expected_policy_semantic_sha256"),
            )
            if config["experiment_id"] in FORMAL_EXPERIMENT_IDS:
                protocol_path = Path(config["protocol"])
                protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
                smoke_manifest = load_manifest_binding(config["smoke_binding"], bindings)
                contract = validate_formal_contract(config, protocol, manifest, smoke_manifest)
                promotion = (
                    load_amendment3_promotion_evidence(config, environment)
                    if config["experiment_id"] == "stage5_4_amendment3_formal"
                    else load_amendment4_promotion_evidence(config, environment)
                    if config["experiment_id"] == "stage5_4_amendment4_formal"
                    else None
                )
                frozen = validate_formal_frozen_dependencies(config, bindings, inputs, manifest)
                frozen_ts0 = load_frozen_ts0_predictions(bindings, manifest)
                frozen["stage5_3_cmp1_videos"] = len(frozen_ts0)
                frozen["stage5_3_cmp1_frames"] = sum(len(frames) for frames in frozen_ts0.values())
                pinned_protocol_sha = config.get("protocol_sha256")
                actual_protocol_sha = file_sha256(protocol_path)
                if pinned_protocol_sha is not None and pinned_protocol_sha != actual_protocol_sha:
                    raise FrozenInputError("formal protocol file SHA does not match config binding")
                result = {
                    **contract,
                    "frozen_inputs": frozen,
                    "protocol_sha256": actual_protocol_sha,
                    "execution_ready": protocol["status"] in {
                        "PREREGISTERED_BEFORE_FORMAL",
                        "PREREGISTERED_BEFORE_ANY_AMENDMENT3_EXPERIMENT",
                        "PREREGISTERED_BEFORE_ANY_AMENDMENT4_EXPERIMENT",
                    }
                    and pinned_protocol_sha == actual_protocol_sha,
                    "gpu_requirement": "NONE",
                    "output_directory": str(paths.output),
                    "smoke_promotion": promotion,
                }
            elif config["experiment_id"] in AMENDMENT_SMOKE_EXPERIMENT_IDS:
                protocol = json.loads(Path(config["protocol"]).read_text(encoding="utf-8"))
                validator = {
                    "stage5_4_amendment_smoke": validate_amendment_contract,
                    "stage5_4_amendment2_smoke": validate_amendment2_contract,
                    "stage5_4_amendment3_smoke": validate_amendment3_smoke_contract,
                    "stage5_4_amendment4_smoke": validate_amendment4_smoke_contract,
                }[config["experiment_id"]]
                result = validator(config, protocol, manifest)
                frozen_ts0 = load_frozen_ts0_predictions(bindings, manifest, exact_identity=False)
                result["stage5_3_cmp1_videos"] = len(frozen_ts0)
                result["stage5_3_cmp1_frames"] = sum(len(frames) for frames in frozen_ts0.values())
                result["output_directory"] = str(paths.output)
            else:
                result = {
                    "manifest_id": manifest["manifest_id"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "videos": manifest["video_count"],
                    "frames": manifest["frame_count"],
                    "validation": "PASS",
                }
        except (FrozenInputError, ValueError, KeyError, json.JSONDecodeError) as exc:
            result = {"validation": "FAIL", "reason": str(exc)}
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("validation") == "PASS" else 2

    if config["experiment_id"] in FORMAL_EXPERIMENT_IDS:
        protocol_path = Path(config["protocol"])
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        actual_protocol_sha = file_sha256(protocol_path)
        if protocol.get("status") not in {
            "PREREGISTERED_BEFORE_FORMAL",
            "PREREGISTERED_BEFORE_ANY_AMENDMENT3_EXPERIMENT",
            "PREREGISTERED_BEFORE_ANY_AMENDMENT4_EXPERIMENT",
        } or config.get(
            "protocol_sha256"
        ) != actual_protocol_sha:
            print(
                "EXPERIMENT FAILED\n\nReason:\nFormal protocol is not SHA-bound as "
                "PREREGISTERED_BEFORE_FORMAL\n\nResume available:\nNO",
                file=sys.stderr,
            )
            return 2

    execution_preflight = None
    if config["experiment_id"] in {
        "stage5_4_amendment4_smoke", "stage5_4_amendment4_formal"
    }:
        try:
            execution_preflight = amendment4_execution_preflight(
                config, environment, bindings, paths, resume=args.resume
            )
        except (FrozenInputError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"BLOCKED_BEFORE_EXECUTION\n\nReason:\n{exc}", file=sys.stderr)
            return 2

    try:
        context.start(resume=args.resume)
    except (RunIdentityMismatch, FileExistsError) as exc:
        print(f"EXPERIMENT FAILED\n\nReason:\n{exc}\n\nResume available:\nNO", file=sys.stderr)
        return 2

    started = time.monotonic()
    identity_hash = canonical_sha256(context.identity())
    shard_store = ShardStore(paths.output / "shards", identity_hash)
    reporter = None

    try:
        smoke_promotion_result = None
        manifest = load_frozen_manifest(config, bindings)
        inputs = load_frozen_inputs(
            [b for b in bindings if b.format != "raw_shard_dir"],
            include_raw=False,
            policy_semantic_expectation=config.get("expected_policy_semantic_sha256"),
        )
        if config["experiment_id"] in FORMAL_EXPERIMENT_IDS:
            smoke_manifest = load_manifest_binding(config["smoke_binding"], bindings)
            formal_contract = validate_formal_contract(
                config,
                json.loads(Path(config["protocol"]).read_text(encoding="utf-8")),
                manifest,
                smoke_manifest,
            )
            if config["experiment_id"] == "stage5_4_amendment3_formal":
                smoke_promotion_result = load_amendment3_promotion_evidence(
                    config, environment
                )
            elif config["experiment_id"] == "stage5_4_amendment4_formal":
                smoke_promotion_result = load_amendment4_promotion_evidence(
                    config, environment
                )
            frozen_dependency_audit = validate_formal_frozen_dependencies(
                config, bindings, inputs, manifest
            )
            frozen_ts0 = load_frozen_ts0_predictions(bindings, manifest)
        elif config["experiment_id"] in AMENDMENT_SMOKE_EXPERIMENT_IDS:
            protocol = json.loads(Path(config["protocol"]).read_text(encoding="utf-8"))
            validator = {
                "stage5_4_amendment_smoke": validate_amendment_contract,
                "stage5_4_amendment2_smoke": validate_amendment2_contract,
                "stage5_4_amendment3_smoke": validate_amendment3_smoke_contract,
                "stage5_4_amendment4_smoke": validate_amendment4_smoke_contract,
            }[config["experiment_id"]]
            validator(config, protocol, manifest)
            formal_contract = None
            frozen_dependency_audit = {
                "stage5_3_cmp1_identity": "byte-pinned Full Dev artifact; exact Smoke24 subset crosschecked per frame",
                "stage5_3_final_freeze": "FINAL_FROZEN",
                "stage5_2_policy_semantic_sha256": config["expected_policy_semantic_sha256"],
            }
            frozen_ts0 = load_frozen_ts0_predictions(bindings, manifest, exact_identity=False)
        else:
            formal_contract = None
            frozen_dependency_audit = None
            frozen_ts0 = None
        manifest_video_ids = {video["video_id"] for video in manifest["videos"]}
        manifest_frame_keys = {
            (video["video_id"], int(entry["frame"]))
            for video in manifest["videos"]
            for entry in video["frames"]
        }
        slim_policy_records = tuple(
            record
            for record in inputs.policy_records
            if (record["video_id"], int(record["frame"])) in manifest_frame_keys
        )
        slim_inputs = dataclasses.replace(inputs, policy_records=slim_policy_records)

        target_ratio = [float(value) for value in config["composition"]["target_ratio"]]
        strata = config["composition"]["strata_thresholds"]
        tw, th = float(target_ratio[0]), float(target_ratio[1])
        alpha = float(config["temporal_smoothing"].get("alpha", DEFAULT_EMA_ALPHA))
        include_ts2 = "ts2" in config["temporal_smoothing"]
        include_ts3 = "ts3" in config["temporal_smoothing"]
        include_ts4 = "ts4" in config["temporal_smoothing"]
        include_ts5 = "ts5" in config["temporal_smoothing"]

        total_frames = int(manifest["frame_count"])
        reporter = ProgressReporter(config["experiment_id"], total_frames, paths.logs)
        frames_done = 0
        videos_done = 0
        invalid_total = 0
        all_mismatches: list[str] = []
        for video in manifest["videos"]:
            video_id = video["video_id"]
            if shard_store.is_complete(video_id):
                frames_done += int(video["frame_count"])
                videos_done += 1
                reporter.update(
                    frames_done,
                    current_video=video_id,
                    current_shard=video_id,
                    errors=len(all_mismatches),
                    display={
                        "videos": f"{videos_done}/{manifest['video_count']}",
                        "frames": f"{frames_done}/{total_frames}",
                        "current_sequence": video_id,
                        "invalid": invalid_total,
                        "resumed": "yes",
                    },
                )
                continue
            video_records, mismatches = build_video_records(
                video,
                slim_inputs,
                target_ratio,
                strata,
                alpha,
                tw,
                th,
                None if frozen_ts0 is None else frozen_ts0[video_id],
                include_ts2,
                include_ts3,
                include_ts4,
                include_ts5,
            )
            all_mismatches.extend(mismatches)
            shard_store.write(video_id, video_records)
            frames_done += len(video_records)
            videos_done += 1
            invalid = sum(1 for record in video_records if not all(record["geometry_valid"].values()))
            invalid_total += invalid
            reporter.update(
                frames_done,
                current_video=video_id,
                current_shard=video_id,
                errors=len(all_mismatches),
                display={
                    "videos": f"{videos_done}/{manifest['video_count']}",
                    "frames": f"{frames_done}/{total_frames}",
                    "current_sequence": video_id,
                    "invalid": invalid_total,
                },
            )

        records: list[dict] = []
        for shard in sorted(shard_store.root.glob("*.json")):
            if shard.name.endswith(".status.json"):
                continue
            records.extend(json.loads(shard.read_text(encoding="utf-8")))
        records.sort(key=lambda record: (record["video_id"], record["frame"]))

        machine = paths.output / "machine"
        diagnostics_dir = machine / "diagnostics"
        snapshots_dir = paths.output / "snapshots"
        machine.mkdir(parents=True, exist_ok=True)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.config, snapshots_dir / "config.json")
        shutil.copyfile(Path(config["protocol"]), snapshots_dir / "protocol.json")
        for progress_name in ("progress.json", "progress.jsonl"):
            source = paths.logs / progress_name
            if source.is_file():
                shutil.copyfile(source, paths.output / progress_name)

        gate = engineering_gate(manifest, records, all_mismatches)
        temporal_by_video = {
            video["video_id"]: temporal_metrics_for_all_treatments(
                [record for record in records if record["video_id"] == video["video_id"]]
            )
            for video in manifest["videos"]
        }
        pooled = pooled_distributions(records)
        guardrails = spatial_metrics_for_all_treatments(records)

        deterministic_records: list[dict] = []
        for video in manifest["videos"]:
            video_records, _ = build_video_records(
                video,
                slim_inputs,
                target_ratio,
                strata,
                alpha,
                tw,
                th,
                None if frozen_ts0 is None else frozen_ts0[video["video_id"]],
                include_ts2,
                include_ts3,
                include_ts4,
                include_ts5,
            )
            deterministic_records.extend(video_records)
        deterministic_records.sort(key=lambda record: (record["video_id"], record["frame"]))
        deterministic_pass = canonical_sha256(records) == canonical_sha256(deterministic_records)

        post_hashes = verify_input_bindings(
            [b for b in bindings if b.format not in ("policy_shard_dir", "raw_shard_dir")]
        )
        artifact_modified = {
            name: post_hashes.get(name) != inputs.input_hashes.get(name) for name in post_hashes
        }

        raw_frames = load_raw_shard_dir_subset(
            next(b for b in bindings if b.name == "stage5_2_raw_detector").path,
            manifest_video_ids,
        )
        inputs_with_raw = dataclasses.replace(slim_inputs, raw_frames=raw_frames)
        multi_subject = temporal_multi_subject_diagnostic(
            inputs_with_raw, records, target_ratio, policy_config_from(config)
        )
        additional_sides = [
            side
            for side, enabled in (
                ("ts2", include_ts2), ("ts3", include_ts3), ("ts4", include_ts4),
                ("ts5", include_ts5),
            )
            if enabled
        ]
        if additional_sides:
            multi_subject = {
                "ts0_vs_ts1": multi_subject,
                **{
                    f"ts0_vs_{side}": _rename_ts1_keys(
                        temporal_multi_subject_diagnostic(
                            inputs_with_raw,
                            _project_treatment(records, side),
                            target_ratio,
                            policy_config_from(config),
                        ),
                        side,
                    )
                    for side in additional_sides
                },
            }

        gates = evaluate_gates(
            gate,
            deterministic_pass,
            artifact_modified,
            (
                multi_subject["ts0_vs_ts1"].get("ambiguous_frames") is not None
                if additional_sides
                else multi_subject.get("ambiguous_frames") is not None
            ),
        )
        if config["experiment_id"] in FORMAL_EXPERIMENT_IDS or additional_sides:
            gates["protocol_config_snapshots_generated"] = all(
                (snapshots_dir / name).is_file() for name in ("config.json", "protocol.json")
            )
        if config["experiment_id"] in FORMAL_EXPERIMENT_IDS:
            confirmatory_ids = set(formal_contract["confirmatory_dev142"]["video_ids"])
            confirmatory_records = [
                record for record in records if record["video_id"] in confirmatory_ids
            ]
            if config["experiment_id"] in {
                "stage5_4_amendment2_formal", "stage5_4_amendment3_formal",
                "stage5_4_amendment4_formal",
            }:
                analysis_set_records = {
                    "full_dev166": records,
                    "confirmatory_dev142": confirmatory_records,
                }
                analysis_metrics = {
                    name: build_analysis_metrics_all_treatments(set_records)
                    for name, set_records in analysis_set_records.items()
                }
                if config["experiment_id"] == "stage5_4_amendment4_formal":
                    scientific_gates = evaluate_amendment4_formal_scientific_gates(
                        analysis_set_records, config["decision_gates"]
                    )
                elif config["experiment_id"] == "stage5_4_amendment3_formal":
                    scientific_gates = evaluate_amendment3_formal_scientific_gates(
                        analysis_set_records, config["decision_gates"]
                    )
                else:
                    scientific_gates = evaluate_amendment2_formal_scientific_gates(
                        analysis_set_records, config["decision_gates"]
                    )
            else:
                analysis_metrics = {
                    "full_dev166": build_analysis_metrics(records),
                    "confirmatory_dev142": build_analysis_metrics(confirmatory_records),
                }
                scientific_gates = evaluate_formal_scientific_gates(
                    analysis_metrics, config["decision_gates"]
                )
            aggregate_formal_multi_subject_diagnostics(
                analysis_metrics,
                multi_subject,
                confirmatory_ids,
                four_arm=config["experiment_id"] in {
                    "stage5_4_amendment2_formal", "stage5_4_amendment3_formal",
                    "stage5_4_amendment4_formal",
                },
                candidate_sides=(
                    TREATMENT_SIDES
                    if config["experiment_id"] in {
                        "stage5_4_amendment3_formal", "stage5_4_amendment4_formal"
                    }
                    else ("ts1", "ts2", "ts3")
                ),
            )
        elif include_ts5:
            analysis_metrics = None
            scientific_gates = evaluate_amendment4_scientific_gates(
                records, pooled, guardrails, config["decision_gates"]
            )
        elif include_ts4:
            analysis_metrics = None
            scientific_gates = evaluate_amendment3_scientific_gates(
                records, pooled, guardrails, config["decision_gates"]
            )
        elif include_ts3:
            analysis_metrics = None
            scientific_gates = evaluate_amendment2_scientific_gates(
                records, pooled, guardrails, config["decision_gates"]
            )
        elif include_ts2:
            analysis_metrics = None
            ts2_gate_metrics = build_analysis_metrics(_project_treatment(records, "ts2"))
            ts2_gate_result = _evaluate_one_analysis_set(
                ts2_gate_metrics, config["decision_gates"]
            )
            strong = guardrails["strata"]["strongly_off_center"]
            mechanism_checks = {
                "strong_visible_delta_ts2_not_worse_than_ts1": {
                    "ts1_delta": strong["delta_mean"],
                    "ts2_delta": strong["ts2_delta_mean"],
                    "pass": strong["ts2_delta_mean"] >= strong["delta_mean"],
                },
                "containment_delta_ts2_not_worse_than_ts1": {
                    "ts1_delta": round(
                        guardrails["ts1"]["subject_center_inside_rate"]
                        - guardrails["ts0"]["subject_center_inside_rate"],
                        6,
                    ),
                    "ts2_delta": round(
                        guardrails["ts2"]["subject_center_inside_rate"]
                        - guardrails["ts0"]["subject_center_inside_rate"],
                        6,
                    ),
                },
                "full_response_alpha_at_m_gte_0_20": {
                    "eligible": sum(
                        1
                        for record in records
                        if record["ts2"]["motion_norm"] is not None
                        and record["ts2"]["motion_norm"] >= 0.2
                    ),
                    "pass": all(
                        record["ts2"]["alpha_t"] == 1.0
                        for record in records
                        if record["ts2"]["motion_norm"] is not None
                        and record["ts2"]["motion_norm"] >= 0.2
                    ),
                },
            }
            containment = mechanism_checks["containment_delta_ts2_not_worse_than_ts1"]
            containment["pass"] = containment["ts2_delta"] >= containment["ts1_delta"]
            scientific_gates = {
                "role": "AMENDMENT_SMOKE_PROMOTION_REVIEW / frozen Stage 5.4 gates applied to TS-2",
                "ts2_vs_ts0": ts2_gate_result,
                "amendment_mechanism_checks": mechanism_checks,
                "all_pass": ts2_gate_result["pass"]
                and all(item["pass"] for item in mechanism_checks.values()),
            }
        else:
            analysis_metrics = None
            scientific_gates = None
        wall_sec = time.monotonic() - started
        engineering_pass = all(gates.values())
        overall_pass = engineering_pass and (
            scientific_gates is None or scientific_gates["all_pass"]
        )

        summary = {
            "schema_version": "aic.machine-summary/v1",
            "experiment_id": config["experiment_id"],
            "manifest_id": manifest["manifest_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "videos": manifest["video_count"],
            "frames": manifest["frame_count"],
            "ts0": "stage5_3_final_frozen_cmp1",
            "ts1": {
                key: value
                for key, value in config["temporal_smoothing"].items()
                if key not in {"ts2", "ts3", "ts4", "ts5"}
            },
            "manifest_coverage": manifest.get("coverage", {}),
            "analysis_set_identities": None if formal_contract is None else {
                "full_dev166": formal_contract["full_dev166"],
                "confirmatory_dev142": formal_contract["confirmatory_dev142"],
            },
            "frozen_dependency_audit": frozen_dependency_audit,
        }
        if include_ts2:
            summary["ts2"] = config["temporal_smoothing"]["ts2"]
        if include_ts3:
            summary["ts3"] = config["temporal_smoothing"]["ts3"]
        if include_ts4:
            summary["ts4"] = config["temporal_smoothing"]["ts4"]
        if include_ts5:
            summary["ts5"] = config["temporal_smoothing"]["ts5"]
        atomic_write_json(machine / "summary.json", summary)
        metrics_payload = {
            "schema_version": "aic.machine-metrics/v1",
            "metric_role": "PREREGISTERED_FORMAL_GATES"
            if analysis_metrics is not None
            else (
                "PREREGISTERED_AMENDMENT4_SMOKE_PROMOTION_GATES"
                if include_ts5
                else "DRAFT_AMENDMENT_SMOKE_PROMOTION_GATES"
                if additional_sides
                else "DESCRIPTIVE_ONLY / no promotion gate in Stage 5.4 smoke"
            ),
            "temporal_stability_by_video": temporal_by_video,
            "temporal_stability_pooled": {key: summary_payload(values) for key, values in pooled.items()},
            "spatial_guardrails": guardrails,
        }
        if analysis_metrics is not None:
            metrics_payload["analysis_sets"] = analysis_metrics
        if include_ts3:
            metrics_payload["guard_projection_diagnostics"] = guard_projection_diagnostics(
                records, "ts3"
            )
        if include_ts4:
            metrics_payload["ts4_guard_projection_diagnostics"] = guard_projection_diagnostics(
                records, "ts4"
            )
            metrics_payload["bbox_guard_diagnostics"] = bbox_guard_diagnostics(records)
        if include_ts5:
            metrics_payload["ts5_guard_projection_diagnostics"] = guard_projection_diagnostics(
                records, "ts5"
            )
            metrics_payload["ts5_bbox_guard_diagnostics"] = bbox_guard_diagnostics(
                records, "ts5"
            )
            metrics_payload["ts5_projected_state_attribution"] = (
                projected_state_attribution_diagnostics(records)
            )
        atomic_write_json(machine / "metrics.json", metrics_payload)
        runtime_payload = {
            "schema_version": "aic.machine-runtime/v1",
            "wall_sec": round(wall_sec, 3),
            "gpu_calls": 0,
            "qwen_vllm_calls": 0,
            "rtdetr_inference": 0,
            "device": "cpu",
        }
        if additional_sides:
            runtime_payload.update({"sam_calls": 0, "heldout_access": 0, "official_test_access": 0})
        atomic_write_json(machine / "runtime.json", runtime_payload)
        validation_payload = {
            "schema_version": "aic.machine-validation/v1",
            "status": "PASS" if overall_pass else "FAIL",
            "gates": gates,
            "scientific_gates": scientific_gates,
            "engineering_gate": gate,
            "deterministic_replay": deterministic_pass,
            "stage5_2_artifact_modified": artifact_modified,
            "preflight": execution_preflight,
        }
        atomic_write_json(machine / "validation.json", validation_payload)
        promotion_marker = None
        if config["experiment_id"] == "stage5_4_amendment4_smoke":
            protocol = json.loads(Path(config["protocol"]).read_text(encoding="utf-8"))
            promotion_marker = write_amendment4_promotion_marker(
                paths.output,
                config,
                protocol,
                validation_payload,
                context.identity()["git_head"],
            )
        atomic_write_json(diagnostics_dir / "multi_subject_diagnostic.json", multi_subject)
        guard_diagnostics = guard_projection_diagnostics(records)
        if guard_diagnostics is not None:
            atomic_write_json(diagnostics_dir / "guard_projection_diagnostic.json", guard_diagnostics)
        ts4_guard_diagnostics = guard_projection_diagnostics(records, "ts4")
        bbox_diagnostics = bbox_guard_diagnostics(records)
        if ts4_guard_diagnostics is not None:
            atomic_write_json(
                diagnostics_dir / "ts4_guard_projection_diagnostic.json",
                ts4_guard_diagnostics,
            )
        if bbox_diagnostics is not None:
            atomic_write_json(diagnostics_dir / "bbox_guard_diagnostic.json", bbox_diagnostics)
        ts5_guard_diagnostics = guard_projection_diagnostics(records, "ts5")
        ts5_bbox_diagnostics = bbox_guard_diagnostics(records, "ts5")
        ts5_attribution = projected_state_attribution_diagnostics(records)
        if ts5_guard_diagnostics is not None:
            atomic_write_json(
                diagnostics_dir / "ts5_guard_projection_diagnostic.json",
                ts5_guard_diagnostics,
            )
        if ts5_bbox_diagnostics is not None:
            atomic_write_json(
                diagnostics_dir / "ts5_bbox_guard_diagnostic.json",
                ts5_bbox_diagnostics,
            )
        if ts5_attribution is not None:
            atomic_write_json(
                diagnostics_dir / "ts5_projected_state_attribution.json",
                ts5_attribution,
            )

        report_path = paths.output / "experiment_report.md"
        render_amendment3_experiment_report(
            report_path,
            config=config,
            summary=summary,
            metrics=metrics_payload,
            validation=validation_payload,
            runtime=runtime_payload,
            smoke_result=smoke_promotion_result,
        )
        render_amendment4_experiment_report(
            report_path,
            config=config,
            summary=summary,
            metrics=metrics_payload,
            validation=validation_payload,
            runtime=runtime_payload,
            smoke_result=smoke_promotion_result,
            promotion_marker=promotion_marker,
        )

        artifact_files = [
            *sorted(shard_store.root.glob("*.json")),
            machine / "summary.json",
            machine / "metrics.json",
            machine / "validation.json",
            machine / "runtime.json",
            diagnostics_dir / "multi_subject_diagnostic.json",
            snapshots_dir / "config.json",
            snapshots_dir / "protocol.json",
            paths.output / "progress.json",
            paths.output / "progress.jsonl",
        ]
        if guard_diagnostics is not None:
            artifact_files.append(diagnostics_dir / "guard_projection_diagnostic.json")
        if ts4_guard_diagnostics is not None:
            artifact_files.append(diagnostics_dir / "ts4_guard_projection_diagnostic.json")
        if bbox_diagnostics is not None:
            artifact_files.append(diagnostics_dir / "bbox_guard_diagnostic.json")
        if promotion_marker is not None:
            artifact_files.append(machine / "promotion_decision.json")
        if ts5_guard_diagnostics is not None:
            artifact_files.append(diagnostics_dir / "ts5_guard_projection_diagnostic.json")
        if ts5_bbox_diagnostics is not None:
            artifact_files.append(diagnostics_dir / "ts5_bbox_guard_diagnostic.json")
        if ts5_attribution is not None:
            artifact_files.append(diagnostics_dir / "ts5_projected_state_attribution.json")
        if report_path.is_file():
            artifact_files.append(report_path)
        build_artifact_manifest(paths.output, artifact_files, machine / "artifact_manifest.json", created_by=config["experiment_id"])
        render_raw_report(machine, paths.output / "experiment_raw_report.md", config["experiment_id"])
        write_ai_report_inputs(paths.output)
        context.set_status(
            "COMPLETED" if overall_pass else "VALIDATION_FAILED",
            validation="PASS" if overall_pass else "FAIL",
        )

        print(f"\n{'=' * 50}\nEXPERIMENT {'COMPLETE' if overall_pass else 'VALIDATION FAILED'}\n{'=' * 50}")
        print(f"Gates: {json.dumps(gates, indent=2)}")
        print(f"Raw report:\n{paths.output / 'experiment_raw_report.md'}")
        return 0 if overall_pass else 2
    except KeyboardInterrupt:
        if reporter is not None:
            reporter.interrupt()
        context.set_status("INTERRUPTED")
        print("\nEXPERIMENT FAILED\n\nReason:\nInterrupted safely\n\nResume available:\nYES\n\nResume command:\nSame command + --resume")
        return 130
    except FrozenInputError as exc:
        context.set_status("FAILED", reason=str(exc))
        print(f"EXPERIMENT FAILED\n\nReason:\n{exc}\n\nResume available:\nNO (frozen input problem)", file=sys.stderr)
        return 2


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
