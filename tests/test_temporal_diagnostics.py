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
from scripts.experiments.stage5.run_stage5_4_temporal import (
    engineering_gate,
    evaluate_amendment2_scientific_gates,
    guard_projection_diagnostics,
    pooled_distributions,
    spatial_metrics_for_all_treatments,
    temporal_metrics_for_all_treatments,
)

W, H = 1600, 900
TW, TH = 9.0, 16.0
CROP_W = (H * 9) // 16


def merged_record(video_id: str, frame: int, ts0_x: int, ts1_x: int, *, fallback: bool = False, reset=None, ts2_x: int | None = None, ts3_x: int | None = None) -> dict:
    record = {
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
    if ts2_x is not None:
        record["ts2"] = {
            **record["ts1"],
            "x": ts2_x,
            "placement_status": "FALLBACK_CENTER_CROP" if fallback else "TS2_ADAPTIVE_SMOOTHED",
            "ema_center_x": None if fallback else float(ts2_x) + CROP_W / 2.0,
            "motion_norm": None if reset or fallback else 0.2,
            "alpha_t": None if reset or fallback else 1.0,
        }
        record["geometry_valid"].update({
            "ts2_nonnegative": True,
            "ts2_width_positive": True,
            "ts2_x_within_frame": True,
            "ts2_derived_height_within_frame": True,
        })
    if ts3_x is not None:
        record["ts3"] = {
            **record["ts1"],
            "x": ts3_x,
            "placement_status": "FALLBACK_CENTER_CROP" if fallback else "TS3_GUARDED_SMOOTHED",
            "ema_center_x": None if fallback else float(ts1_x) + CROP_W / 2.0,
            "guard_applied": not fallback and ts3_x != ts1_x,
            "guard_correction_x": None if fallback else ts3_x - ts1_x,
            "guard_correction_y": None if fallback else 0,
        }
        record["geometry_valid"].update({
            "ts3_nonnegative": True,
            "ts3_width_positive": True,
            "ts3_x_within_frame": True,
            "ts3_derived_height_within_frame": True,
        })
    return record


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
    summary = summarize_multi_subject_rows(rows, candidate_side="ts1")
    assert summary == {
        "tag": "MULTI_SUBJECT_DIAGNOSTIC / observation only / TS-0 (CMP-1) vs TS-1 / no union or fusion policy",
        "ambiguous_frames": 2,
        "frames_with_secondaries": 2,
        "all_secondaries_inside_ts0_rate": 0.5,
        "all_secondaries_inside_ts1_rate": 1.0,
        "at_least_one_inside_ts0_rate": 0.5,
        "at_least_one_inside_ts1_rate": 1.0,
        "primary_and_all_secondaries_inside_ts0_rate": 0.5,
        "primary_and_all_secondaries_inside_ts1_rate": 0.5,
        "rows": rows,
    }


@pytest.mark.parametrize("candidate_side", ["ts2", "ts3"])
def test_multi_subject_summary_uses_explicit_candidate_side_with_exact_rates(candidate_side):
    rows = [
        {
            "video_id": "a",
            "frame": 0,
            "primary_center_inside_ts0_crop": True,
            f"primary_center_inside_{candidate_side}_crop": True,
            "secondary_count": 2,
            "secondary_centers_inside_ts0_crop": 2,
            f"secondary_centers_inside_{candidate_side}_crop": 1,
        },
        {
            "video_id": "a",
            "frame": 1,
            "primary_center_inside_ts0_crop": True,
            f"primary_center_inside_{candidate_side}_crop": True,
            "secondary_count": 1,
            "secondary_centers_inside_ts0_crop": 0,
            f"secondary_centers_inside_{candidate_side}_crop": 1,
        },
    ]
    summary = summarize_multi_subject_rows(rows, candidate_side=candidate_side)
    assert summary["all_secondaries_inside_ts0_rate"] == 0.5
    assert summary[f"all_secondaries_inside_{candidate_side}_rate"] == 0.5
    assert summary["at_least_one_inside_ts0_rate"] == 0.5
    assert summary[f"at_least_one_inside_{candidate_side}_rate"] == 1.0
    assert summary["primary_and_all_secondaries_inside_ts0_rate"] == 0.5
    assert summary[f"primary_and_all_secondaries_inside_{candidate_side}_rate"] == 0.5
    assert f"TS-0 (CMP-1) vs TS-{candidate_side[2:]}" in summary["tag"]


def test_multi_subject_summary_rejects_unknown_or_missing_candidate_contract():
    with pytest.raises(ValueError, match="unknown multi-subject candidate side"):
        summarize_multi_subject_rows([], candidate_side="ts4")
    incomplete = {
        "video_id": "a",
        "frame": 0,
        "primary_center_inside_ts0_crop": True,
        "secondary_count": 1,
        "secondary_centers_inside_ts0_crop": 1,
    }
    with pytest.raises(ValueError, match="missing multi-subject row fields"):
        summarize_multi_subject_rows([incomplete], candidate_side="ts2")


def test_multi_subject_summary_empty_and_zero_applicable_rows_are_explicit():
    empty = summarize_multi_subject_rows([], candidate_side="ts3")
    assert empty["ambiguous_frames"] == 0
    assert empty["frames_with_secondaries"] == 0
    assert empty["all_secondaries_inside_ts0_rate"] is None
    assert empty["all_secondaries_inside_ts3_rate"] is None

    zero_applicable = summarize_multi_subject_rows(
        [
            {
                "video_id": "a",
                "frame": 0,
                "primary_center_inside_ts0_crop": True,
                "primary_center_inside_ts3_crop": True,
                "secondary_count": 0,
                "secondary_centers_inside_ts0_crop": 0,
                "secondary_centers_inside_ts3_crop": 0,
            }
        ],
        candidate_side="ts3",
    )
    assert zero_applicable["ambiguous_frames"] == 1
    assert zero_applicable["frames_with_secondaries"] == 0
    assert zero_applicable["primary_and_all_secondaries_inside_ts3_rate"] is None


def test_amendment_helpers_include_ts2_without_changing_ts1_contract():
    records = [
        merged_record("v", 0, 500, 500, ts2_x=500),
        merged_record("v", 1, 820, 660, ts2_x=820),
    ]
    pooled = pooled_distributions(records)
    assert pooled["ts1_displacement"][0] < pooled["ts2_displacement"][0]
    temporal = temporal_metrics_for_all_treatments(records)
    spatial = spatial_metrics_for_all_treatments(records)
    assert temporal["ts2_displacement"]["n"] == temporal["ts1_displacement"]["n"] == 1
    assert spatial["ts2"]["n"] == spatial["ts1"]["n"] == 2
    gate = engineering_gate(
        {"videos": [{"video_id": "v", "frames": [{"frame": 0}, {"frame": 1}]}]},
        records,
        [],
    )
    assert gate["crop_size_unchanged"]
    assert gate["fallback_placement_unchanged"]
    assert gate["treatment_sides"] == ["ts1", "ts2"]


def test_amendment2_helpers_include_ts3_in_four_way_metrics_and_gates():
    records = [
        merged_record("v", 0, 500, 500, ts2_x=500, ts3_x=500),
        merged_record("v", 1, 820, 660, ts2_x=820, ts3_x=700),
    ]
    pooled = pooled_distributions(records)
    temporal = temporal_metrics_for_all_treatments(records)
    spatial = spatial_metrics_for_all_treatments(records)
    gate = engineering_gate(
        {"videos": [{"video_id": "v", "frames": [{"frame": 0}, {"frame": 1}]}]}, records, []
    )
    assert pooled["ts3_displacement"][0] < pooled["ts2_displacement"][0]
    assert temporal["ts3_displacement"]["n"] == 1
    assert spatial["ts3"]["n"] == 2
    assert gate["treatment_sides"] == ["ts1", "ts2", "ts3"]
    guard = guard_projection_diagnostics(records)
    assert guard["eligible_nonfallback_frames"] == 2
    assert guard["guard_applied_frames"] == 1
    assert guard["guard_applied_rate"] == 0.5


def test_amendment2_promotion_checks_are_frozen_and_mechanism_specific():
    records = [
        merged_record("v", 0, 500, 500, ts2_x=500, ts3_x=500),
        merged_record("v", 1, 900, 600, ts2_x=900, ts3_x=650),
        merged_record("v", 2, 500, 550, ts2_x=500, ts3_x=700),
    ]
    for record in records:
        record["stratum"] = "strongly_off_center"
        record["ts3"]["subject_visible_fraction"] = record["ts0"]["subject_visible_fraction"]
        record["ts3"]["subject_center_inside"] = True
    gates = {
        "temporal_benefit": {
            "minimum_mean_displacement_relative_reduction": 0.2,
            "minimum_mean_acceleration_relative_reduction": 0.3,
            "maximum_p95_displacement_relative_regression": 0.0,
            "large_jump_nonincrease_thresholds": [0.1, 0.2, 0.3],
            "minimum_gt_0_20_relative_reduction": 0.25,
        },
        "spatial_regression_guardrails": {
            "minimum_mean_visible_fraction_delta": -0.03,
            "minimum_visible_ge_0_90_delta": -0.05,
            "minimum_subject_center_containment_delta": -0.02,
            "minimum_strong_off_center_mean_visible_delta": -0.05,
        },
    }
    result = evaluate_amendment2_scientific_gates(
        records,
        pooled_distributions(records),
        spatial_metrics_for_all_treatments(records),
        gates,
    )
    checks = result["amendment2_mechanism_checks"]
    assert all(check["pass"] for check in checks.values())
    assert result["ts3_vs_ts0"]["pass"]
    assert result["all_pass"]
