"""Stage 5.4 temporal diagnostics and runner helper tests (synthetic records)."""

from __future__ import annotations

import pytest

from aic_video_highlight.spatial_composition.temporal_diagnostics import (
    crop_geometry_valid,
    observations_from_records,
    spatial_guardrail_metrics,
    summarize_multi_subject_rows,
    temporal_stability_metrics,
)
from scripts.experiments.stage5.run_stage5_4_temporal import engineering_gate, pooled_distributions

W, H = 1600, 900
TW, TH = 9.0, 16.0
CROP_W = (H * 9) // 16


def merged_record(video_id: str, frame: int, ts0_x: int, ts1_x: int, *, fallback: bool = False, reset=None) -> dict:
    return {
        "video_id": video_id,
        "frame": frame,
        "image_width": W,
        "image_height": H,
        "stratum": "no_subject" if fallback else "near_center",
        "horizontal_center_offset": None if fallback else 0.05,
        "ambiguous": False,
        "ambiguous_candidate_count": 0,
        "fallback_reasons": [],
        "sanitized": {
            "xyxy": [700.0, 400.0, 760.0, 460.0] if not fallback else [0.0, 0.0, 0.0, 0.0],
            "status": "INVALID_SANITIZED_SUBJECT" if fallback else "OK",
            "clamp_left": 0.0,
            "clamp_top": 0.0,
            "clamp_right": 0.0,
            "clamp_bottom": 0.0,
        },
        "ts0": {
            "x": ts0_x,
            "y": 0,
            "w": CROP_W,
            "h": float(CROP_W * 16 / 9),
            "crop_w": CROP_W,
            "crop_h": H,
            "placement_status": "FALLBACK_CENTER_CROP" if fallback else "SUBJECT_SHIFTED",
            "fallback": fallback,
            "subject_visible_fraction": None if fallback else 0.8,
            "subject_center_inside": False,
            "frozen_regression": False,
        },
        "ts1": {
            "x": ts1_x,
            "y": 0,
            "w": CROP_W,
            "h": float(CROP_W * 16 / 9),
            "crop_w": CROP_W,
            "crop_h": H,
            "placement_status": "FALLBACK_CENTER_CROP" if fallback else "TS1_SMOOTHED",
            "reset_reason": reset,
            "ema_center_x": None if fallback else float(ts1_x) + CROP_W / 2.0,
            "ema_center_y": None if fallback else 450.0,
            "clamped_x": False,
            "clamped_y": False,
            "matches_ts0_placement": ts0_x == ts1_x,
            "subject_visible_fraction": None if fallback else 0.7,
            "subject_center_inside": True,
        },
        "geometry_valid": {
            "ts0_nonnegative": True, "ts0_width_positive": True,
            "ts0_x_within_frame": True, "ts0_derived_height_within_frame": True,
            "ts1_nonnegative": True, "ts1_width_positive": True,
            "ts1_x_within_frame": True, "ts1_derived_height_within_frame": True,
        },
    }


def test_observations_use_sanitized_subject_center():
    records = [merged_record("v", 0, 500, 500), merged_record("v", 1, 560, 540)]
    observations = observations_from_records(records)
    assert observations[0].ideal_center_x == pytest.approx(730.0)
    assert observations[1].fallback is False


def test_temporal_stability_metrics_paired_and_smoothed_subset():
    records = [
        merged_record("v", 0, 500, 500),
        merged_record("v", 1, 560, 540),
        merged_record("v", 2, 620, 590),
    ]
    metrics = temporal_stability_metrics(records)
    assert metrics["paired_transitions"] == 2
    assert metrics["paired_triplets"] == 1
    assert metrics["ts0_displacement"]["n"] == 2
    assert metrics["ts1_displacement_smoothed_only"]["n"] == 2
    assert metrics["metric_role"].startswith("DESCRIPTIVE_ONLY")


def test_temporal_stability_metrics_excludes_reset_and_fallback_transitions():
    records = [
        merged_record("v", 0, 500, 500),
        merged_record("v", 1, 560, 560),
        merged_record("v", 2, 620, 620, fallback=True),
        merged_record("v", 3, 700, 700, reset="FALLBACK"),
        merged_record("v", 11, 800, 800, reset="FRAME_GAP"),
    ]
    metrics = temporal_stability_metrics(records)
    # only (0, 1) is a valid paired transition; reset steps are counted over all frames
    assert metrics["paired_transitions"] == 1
    assert metrics["ts1_displacement_smoothed_only"]["n"] == 1
    assert metrics["ts1_reset_steps"] == {"FALLBACK": 1, "FRAME_GAP": 1}
    assert metrics["paired_triplets"] == 0


def test_spatial_guardrail_metrics_ts0_vs_ts1_and_strata():
    records = [
        merged_record("v", 0, 500, 500),
        merged_record("v", 1, 560, 540),
        merged_record("v", 2, 620, 620, fallback=True),
    ]
    payload = spatial_guardrail_metrics(records)
    assert payload["frames"] == 3
    assert payload["frames_with_subject"] == 2
    assert payload["fallback_frames"] == 1
    assert payload["ts0"]["n"] == 2
    assert payload["ts1"]["mean"] == pytest.approx(0.7)
    assert payload["ts0"]["mean"] == pytest.approx(0.8)
    assert payload["strata"]["near_center"]["with_subject"] == 2
    assert payload["strata"]["no_subject"]["with_subject"] == 0


def test_crop_geometry_valid_flags_bounds_and_ratio():
    valid = crop_geometry_valid(0, CROP_W, 0, CROP_W * 16 / 9, W, H)
    assert all(valid.values())
    out_of_bounds = crop_geometry_valid(W - CROP_W + 1, CROP_W, 0, CROP_W * 16 / 9, W, H)
    assert out_of_bounds["x_within_frame"] is False
    ratio_violation = crop_geometry_valid(0, CROP_W, 1, CROP_W * 16 / 9, W, H)
    assert ratio_violation["derived_height_within_frame"] is False


def test_pooled_distributions_concatenate_across_videos():
    records = [
        merged_record("a", 0, 500, 500),
        merged_record("a", 1, 560, 540),
        merged_record("b", 0, 400, 400),
        merged_record("b", 1, 480, 450),
    ]
    pooled = pooled_distributions(records)
    assert len(pooled["ts0_displacement"]) == 2
    assert len(pooled["ts1_displacement"]) == 2
    assert len(pooled["ts1_displacement_smoothed_only"]) == 2
    assert pooled["ts0_acceleration"] == []


def test_engineering_gate_counts_identity_size_and_fallback():
    manifest = {
        "videos": [
            {
                "video_id": "a",
                "frames": [{"frame": 0}, {"frame": 1}, {"frame": 2}],
            }
        ]
    }
    records = [
        merged_record("a", 0, 500, 500),
        merged_record("a", 1, 560, 560),
        merged_record("a", 2, 620, 620, fallback=True),
    ]
    gate = engineering_gate(manifest, records, [])
    assert gate["frame_identity_complete"]
    assert gate["crop_size_unchanged"]
    assert gate["fallback_placement_unchanged"]
    assert gate["invalid_crop"] == 0
    assert gate["ts0_frozen_regression"] == 0

    broken = [merged_record("a", 0, 500, 505), merged_record("a", 1, 560, 560)]
    broken[0]["ts1"]["w"] = broken[0]["ts0"]["w"] + 1
    gate_broken = engineering_gate(manifest, broken, ["a:0 center_x"])
    assert not gate_broken["crop_size_unchanged"]
    assert gate_broken["missing"] == 1
    assert gate_broken["manifest_crosscheck_mismatches"] == 1


def test_multi_subject_summary_includes_primary_plus_all_secondaries_observation():
    rows = [
        {
            "video_id": "a",
            "frame": 0,
            "primary_center_inside_ts0_crop": True,
            "primary_center_inside_ts1_crop": False,
            "secondary_count": 2,
            "secondary_centers_inside_ts0_crop": 2,
            "secondary_centers_inside_ts1_crop": 2,
        },
        {
            "video_id": "a",
            "frame": 1,
            "primary_center_inside_ts0_crop": True,
            "primary_center_inside_ts1_crop": True,
            "secondary_count": 1,
            "secondary_centers_inside_ts0_crop": 0,
            "secondary_centers_inside_ts1_crop": 1,
        },
    ]
    summary = summarize_multi_subject_rows(rows)
    assert summary["primary_and_all_secondaries_inside_ts0_rate"] == 0.5
    assert summary["primary_and_all_secondaries_inside_ts1_rate"] == 0.5
    assert summary["tag"].startswith("MULTI_SUBJECT_DIAGNOSTIC")
