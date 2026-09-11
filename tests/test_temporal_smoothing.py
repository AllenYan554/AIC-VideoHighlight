"""Stage 5.4 TS-1 temporal crop-center smoothing tests (frozen Stage 5.3 contract)."""

from __future__ import annotations

from fractions import Fraction
import random

import pytest

from aic_video_highlight.spatial_composition.center_crop import compute_center_crop, derived_height
from aic_video_highlight.spatial_composition.composition_metrics import (
    crop_rect_from_xywh,
    subject_center_inside_crop,
    subject_visible_fraction,
)
from aic_video_highlight.spatial_composition.subject_shifted_crop import (
    PLACEMENT_FALLBACK_CENTER_CROP,
    sanitize_primary_bbox,
)
from aic_video_highlight.spatial_composition.temporal_smoothing import (
    DEFAULT_EMA_ALPHA,
    MOTION_ADAPTIVE_FULL_RESPONSE,
    MOTION_ADAPTIVE_SMOOTHING_CEILING,
    PLACEMENT_TS2_ADAPTIVE_SMOOTHED,
    PLACEMENT_TS3_GUARDED_SMOOTHED,
    PLACEMENT_TS4_BBOX_GUARDED_SMOOTHED,
    PLACEMENT_TS5_PROJECTED_STATE_SMOOTHED,
    PLACEMENT_TS1_SMOOTHED,
    RESET_FALLBACK,
    RESET_FRAME_GAP,
    SmoothedFrame,
    TemporalObservation,
    acceleration_norm,
    containment_safe_top_left_interval,
    displacement_norm,
    large_jump_ratios,
    motion_adaptive_alpha,
    maximum_overlap_safe_top_left_interval,
    place_crop_from_center,
    smooth_video_sequence,
    smooth_video_sequence_adaptive,
    smooth_video_sequence_guarded,
    smooth_video_sequence_bbox_guarded,
    smooth_video_sequence_projected_state_bbox_guarded,
    temporal_summary,
    transition_pairs,
    transition_triplets,
)

# 16:9-shaped test frame with horizontal slack under a 9:16 target.
W, H = 1600, 900
TW, TH = 9, 16
CENTER_BOX = compute_center_crop(W, H, TW, TH)
CROP_W = CENTER_BOX.w
MAX_X = W - CROP_W


def obs(frame: int, cx: float, cy: float = 450.0, fallback: bool = False) -> TemporalObservation:
    return TemporalObservation(frame=frame, fallback=fallback, ideal_center_x=cx, ideal_center_y=cy)


def assert_geometry_valid(frames: list[SmoothedFrame], width: int = W, height: int = H) -> None:
    for item in frames:
        assert item.w > 0
        assert 0 <= item.x and item.x + item.w <= width
        assert 0 <= item.y and float(item.y + item.h) <= height


# 1. constant crop -> unchanged
def test_constant_crop_unchanged():
    observations = [obs(frame, 800.0) for frame in range(5)]
    smoothed = smooth_video_sequence(W, H, TW, TH, observations)
    assert all(item.matches_ts0_placement for item in smoothed)
    assert [item.x for item in smoothed] == [place_crop_from_center(W, H, TW, TH, (800.0, 450.0))[0]] * 5
    assert [item.ema_center_x for item in smoothed] == [800.0] * 5


# 2. simple linear movement
def test_linear_movement_smooths_toward_lag():
    observations = [obs(0, 500.0), obs(1, 560.0), obs(2, 620.0), obs(3, 680.0)]
    smoothed = smooth_video_sequence(W, H, TW, TH, observations)
    assert [item.ema_center_x for item in smoothed] == [500.0, 530.0, 575.0, 627.5]
    ts0 = [displacement_norm(500.0 + 60.0 * index, 500.0 + 60.0 * (index + 1), W) for index in range(3)]
    ts1 = [displacement_norm(a.ema_center_x, b.ema_center_x, W) for a, b in zip(smoothed, smoothed[1:])]
    assert max(ts1) < max(ts0)
    assert smoothed[1].ema_center_x == DEFAULT_EMA_ALPHA * 560.0 + (1 - DEFAULT_EMA_ALPHA) * 500.0


# 3. alternating jitter
def test_alternating_jitter_is_damped():
    observations = [obs(0, 500.0), obs(1, 700.0), obs(2, 500.0), obs(3, 700.0)]
    smoothed = smooth_video_sequence(W, H, TW, TH, observations)
    ts0 = [displacement_norm(500.0, 700.0, W)] * 3
    ts1 = [displacement_norm(a.ema_center_x, b.ema_center_x, W) for a, b in zip(smoothed, smoothed[1:])]
    assert max(ts1) < max(ts0)
    assert sum(ts1) / len(ts1) < sum(ts0) / len(ts0)


# 4. left/right clamp
def test_clamp_left_and_right_keeps_crop_in_bounds():
    left = smooth_video_sequence(W, H, TW, TH, [obs(0, 100.0)])
    right = smooth_video_sequence(W, H, TW, TH, [obs(0, 1550.0)])
    assert left[0].x == 0 and left[0].clamped_x
    assert right[0].x == MAX_X and right[0].clamped_x
    assert_geometry_valid(left + right)


# 5. video boundary reset (fresh state per call)
def test_video_boundary_reset_no_state_leakage():
    first = smooth_video_sequence(W, H, TW, TH, [obs(0, 500.0), obs(1, 900.0)])
    second = smooth_video_sequence(W, H, TW, TH, [obs(0, 400.0)])
    assert first[-1].ema_center_x == 700.0
    assert second[0].ema_center_x == 400.0
    assert second[0].reset_reason is None


# 6+7. frozen segment discontinuity / frame gap reset
def test_frame_gap_reset_starts_fresh_run():
    observations = [obs(0, 500.0), obs(1, 700.0), obs(2, 900.0), obs(10, 300.0), obs(11, 340.0)]
    smoothed = smooth_video_sequence(W, H, TW, TH, observations)
    assert smoothed[3].reset_reason == RESET_FRAME_GAP
    assert smoothed[3].ema_center_x == 300.0
    assert smoothed[4].ema_center_x == 0.5 * 340.0 + 0.5 * 300.0
    assert all(item.reset_reason != RESET_FRAME_GAP for item in smoothed[:3])


# 8. fallback reset (current or previous frame fallback)
def test_fallback_resets_state_and_emits_frozen_center_crop():
    observations = [obs(0, 500.0), obs(1, 640.0), obs(2, 700.0, fallback=True), obs(3, 800.0), obs(4, 860.0)]
    smoothed = smooth_video_sequence(W, H, TW, TH, observations)
    assert smoothed[2].placement_status == PLACEMENT_FALLBACK_CENTER_CROP
    assert (smoothed[2].x, smoothed[2].y) == (CENTER_BOX.x, CENTER_BOX.y)
    assert smoothed[2].reset_reason == RESET_FALLBACK
    assert smoothed[2].ema_center_x is None
    assert smoothed[3].reset_reason == RESET_FALLBACK
    assert smoothed[3].ema_center_x == 800.0
    assert smoothed[4].reset_reason is None
    assert smoothed[4].ema_center_x == 0.5 * 860.0 + 0.5 * 800.0


def test_leading_fallback_and_consecutive_fallbacks_do_not_double_reset():
    observations = [obs(0, 500.0, fallback=True), obs(1, 600.0, fallback=True), obs(2, 700.0)]
    smoothed = smooth_video_sequence(W, H, TW, TH, observations)
    assert smoothed[0].reset_reason is None
    assert smoothed[1].reset_reason is None
    assert smoothed[2].reset_reason == RESET_FALLBACK
    assert smoothed[2].ema_center_x == 700.0


# 9. one-frame sequence
def test_one_frame_sequence_equals_ts0():
    smoothed = smooth_video_sequence(W, H, TW, TH, [obs(7, 640.0)])
    assert len(smoothed) == 1
    assert smoothed[0].matches_ts0_placement
    assert smoothed[0].reset_reason is None


# 10. two-frame sequence
def test_two_frame_sequence_applies_single_ema_step():
    smoothed = smooth_video_sequence(W, H, TW, TH, [obs(0, 500.0), obs(1, 600.0)])
    assert smoothed[0].ema_center_x == 500.0
    assert smoothed[1].ema_center_x == 550.0
    assert smoothed[1].reset_reason is None


# 11. deterministic replay
def test_deterministic_replay_identical_outputs():
    observations = [obs(0, 500.0), obs(1, 700.0, fallback=True), obs(2, 900.0), obs(3, 300.0), obs(9, 400.0)]
    first = smooth_video_sequence(W, H, TW, TH, observations)
    second = smooth_video_sequence(W, H, TW, TH, list(reversed(observations)))
    assert first == second


# 12. width / height unchanged
def test_crop_width_and_height_unchanged_including_fallback():
    observations = [obs(0, 500.0), obs(1, 700.0, fallback=True), obs(2, 900.0)]
    smoothed = smooth_video_sequence(W, H, TW, TH, observations)
    assert all(item.w == CROP_W for item in smoothed)
    assert all(item.crop_w == CROP_W and item.crop_h == H for item in smoothed)
    assert all(item.h == derived_height(CROP_W, TW, TH) for item in smoothed)


# 13. ratio unchanged
def test_target_ratio_contract_holds_for_every_frame():
    observations = [obs(0, 500.0), obs(1, 1300.0), obs(2, 700.0, fallback=True)]
    smoothed = smooth_video_sequence(W, H, TW, TH, observations)
    for item in smoothed:
        assert item.h == Fraction(item.w) * Fraction(str(float(TH))) / Fraction(str(float(TW)))
        assert float(item.y + item.h) <= H


# 14. frame identity unchanged
def test_frame_identity_preserved_and_sorted():
    observations = [obs(23, 900.0), obs(3, 500.0), obs(11, 700.0, fallback=True)]
    smoothed = smooth_video_sequence(W, H, TW, TH, observations)
    assert [item.frame for item in smoothed] == [3, 11, 23]


def test_duplicate_frame_ids_rejected():
    with pytest.raises(ValueError, match="duplicate frame ids"):
        smooth_video_sequence(W, H, TW, TH, [obs(5, 500.0), obs(5, 600.0)])


# 15. invalid = 0
def test_no_invalid_or_out_of_bounds_crops():
    centers = [100.0, 300.0, 1550.0, 800.0, 640.0, 20.0, 1590.0]
    observations = [obs(index * 2, value) for index, value in enumerate(centers)]
    observations.insert(3, obs(7, 700.0, fallback=True))
    smoothed = smooth_video_sequence(W, H, TW, TH, observations)
    assert len(smoothed) == len(observations)
    assert_geometry_valid(smoothed)
    assert all(item.placement_status in (PLACEMENT_TS1_SMOOTHED, PLACEMENT_FALLBACK_CENTER_CROP) for item in smoothed)


# 16. visibility metric compatibility
def test_visibility_metric_on_smoothed_crops():
    subject = sanitize_primary_bbox((560.0, 400.0, 700.0, 520.0), W, H)
    observations = [obs(0, 630.0), obs(1, 630.0)]
    smoothed = smooth_video_sequence(W, H, TW, TH, observations)
    ts1_rect = crop_rect_from_xywh(smoothed[0].x, smoothed[0].y, smoothed[0].w, float(smoothed[0].h))
    ts0_x, ts0_y, _, _, _, _ = place_crop_from_center(W, H, TW, TH, (630.0, 450.0))
    ts0_rect = crop_rect_from_xywh(ts0_x, ts0_y, CROP_W, float(derived_height(CROP_W, TW, TH)))
    assert subject_visible_fraction(subject, ts1_rect) == subject_visible_fraction(subject, ts0_rect)
    assert subject_center_inside_crop(subject, ts1_rect)


# 17. displacement metric
def test_displacement_metric_and_summary():
    assert displacement_norm(500.0, 620.0, 1600) == pytest.approx(0.075)
    values = [0.05, 0.01, 0.2, 0.31, 0.07]
    summary = temporal_summary(values)
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(0.128, abs=1e-6)
    assert summary["median"] == pytest.approx(0.07)
    assert summary["p90"] == pytest.approx(0.31)
    assert summary["p95"] == pytest.approx(0.31)
    assert summary["max"] == pytest.approx(0.31)
    assert large_jump_ratios(values)[">0.10"] == pytest.approx(0.4)
    assert temporal_summary([])["n"] == 0


# 18. acceleration metric
def test_acceleration_metric_detects_direction_flips():
    assert acceleration_norm(60.0, -120.0, 1600) == pytest.approx(180.0 / 1600.0)
    assert acceleration_norm(60.0, 60.0, 1600) == 0.0


def test_transition_pairs_and_triplets_respect_gap_and_fallback():
    observations = [obs(0, 500.0), obs(1, 560.0), obs(2, 620.0, fallback=True), obs(3, 700.0), obs(10, 800.0)]
    pairs = transition_pairs(observations)
    assert (0, 1) in pairs
    assert all(2 not in pair and 3 not in pair for pair in pairs)
    triplets = transition_triplets(observations)
    assert triplets == []


# 19. multi-subject diagnostic compatibility (rect containment over TS-0 vs TS-1)
def test_multi_subject_containment_computable_for_ts0_and_ts1():
    secondary_center = (1500.0, 450.0)
    ts0_x, ts0_y, _, _, _, _ = place_crop_from_center(W, H, TW, TH, (500.0, 450.0))
    smoothed = smooth_video_sequence(W, H, TW, TH, [obs(0, 500.0), obs(1, 500.0)])
    ts1_rect = crop_rect_from_xywh(smoothed[-1].x, smoothed[-1].y, smoothed[-1].w, float(smoothed[-1].h))
    ts0_rect = crop_rect_from_xywh(ts0_x, ts0_y, CROP_W, float(derived_height(CROP_W, TW, TH)))

    def contains(rect, point) -> bool:
        return rect[0] <= point[0] < rect[2] and rect[1] <= point[1] < rect[3]

    assert contains(ts1_rect, secondary_center) == contains(ts0_rect, secondary_center)


# 20. 9:16 target
def test_9_16_target_crop_size():
    assert CROP_W == (H * TW) // TH == 506
    smoothed = smooth_video_sequence(W, H, TW, TH, [obs(0, 800.0)])
    assert smoothed[0].w == 506 and smoothed[0].h == Fraction(506 * 16, 9)
    assert_geometry_valid(smoothed)


# 21. 16:9 target (vertical slack on a 9:16 frame)
def test_16_9_target_vertical_smoothing():
    fw, fh = 1080, 1920
    observations = [obs(0, 540.0, 700.0), obs(1, 540.0, 1300.0)]
    smoothed = smooth_video_sequence(fw, fh, 16, 9, observations)
    assert smoothed[0].w == fw
    assert smoothed[1].ema_center_y == 0.5 * 1300.0 + 0.5 * 700.0
    assert_geometry_valid(smoothed, width=fw, height=fh)


def test_alpha_one_reproduces_ts0_and_alpha_validated():
    observations = [obs(0, 500.0), obs(1, 700.0), obs(2, 300.0)]
    alpha_one = smooth_video_sequence(W, H, TW, TH, observations, alpha=1.0)
    assert all(item.matches_ts0_placement for item in alpha_one)
    for bad in (0.0, -0.5, 1.5, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            smooth_video_sequence(W, H, TW, TH, observations, alpha=bad)


def test_motion_adaptive_alpha_schedule_contract():
    assert motion_adaptive_alpha(0.0) == DEFAULT_EMA_ALPHA
    assert motion_adaptive_alpha(0.01) == DEFAULT_EMA_ALPHA
    assert motion_adaptive_alpha(MOTION_ADAPTIVE_SMOOTHING_CEILING) == DEFAULT_EMA_ALPHA
    assert motion_adaptive_alpha(0.15) == pytest.approx(0.75)
    assert motion_adaptive_alpha(MOTION_ADAPTIVE_FULL_RESPONSE) == 1.0
    assert motion_adaptive_alpha(0.9) == 1.0
    values = [motion_adaptive_alpha(value) for value in (0.0, 0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.30)]
    assert values == sorted(values)
    assert all(DEFAULT_EMA_ALPHA <= value <= 1.0 for value in values)
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="motion_norm"):
            motion_adaptive_alpha(bad)


def test_motion_adaptive_sequence_uses_raw_normalized_motion_and_preserves_contracts():
    observations = [
        obs(0, 500.0),
        obs(1, 516.0),   # 0.01 W: fixed alpha_min
        obs(2, 756.0),   # 0.15 W: alpha_t = 0.75
        obs(3, 1076.0),  # 0.20 W: alpha_t = 1.0
    ]
    adaptive = smooth_video_sequence_adaptive(W, H, TW, TH, observations)
    assert adaptive[0].motion_norm is None and adaptive[0].alpha_t is None
    assert adaptive[1].motion_norm == pytest.approx(0.01)
    assert adaptive[1].alpha_t == DEFAULT_EMA_ALPHA
    assert adaptive[2].motion_norm == pytest.approx(0.15)
    assert adaptive[2].alpha_t == pytest.approx(0.75)
    assert adaptive[3].motion_norm == pytest.approx(0.20)
    assert adaptive[3].alpha_t == 1.0
    assert adaptive[3].ema_center_x == 1076.0
    assert all(item.placement_status == PLACEMENT_TS2_ADAPTIVE_SMOOTHED for item in adaptive)
    assert [item.frame for item in adaptive] == [item.frame for item in observations]
    assert all(item.w == CROP_W and item.h == derived_height(CROP_W, TW, TH) for item in adaptive)
    assert_geometry_valid(adaptive)


def test_motion_adaptive_inherits_all_ts1_reset_semantics_deterministically():
    observations = [
        obs(0, 500.0),
        obs(1, 540.0),
        obs(2, 700.0, fallback=True),
        obs(3, 900.0),
        obs(10, 300.0),
        obs(11, 620.0),
    ]
    first = smooth_video_sequence_adaptive(W, H, TW, TH, observations)
    second = smooth_video_sequence_adaptive(W, H, TW, TH, list(reversed(observations)))
    assert first == second
    assert first[0].motion_norm is None and first[0].alpha_t is None
    assert first[2].placement_status == PLACEMENT_FALLBACK_CENTER_CROP
    assert first[3].reset_reason == RESET_FALLBACK
    assert first[3].motion_norm is None and first[3].alpha_t is None
    assert first[4].reset_reason == RESET_FRAME_GAP
    assert first[4].motion_norm is None and first[4].alpha_t is None
    assert first[5].alpha_t == 1.0


def test_motion_adaptive_does_not_change_ts0_or_ts1_regression_contract():
    observations = [obs(0, 500.0), obs(1, 620.0), obs(2, 940.0)]
    before = smooth_video_sequence(W, H, TW, TH, observations, alpha=0.5)
    after = smooth_video_sequence(W, H, TW, TH, observations, alpha=0.5)
    adaptive = smooth_video_sequence_adaptive(W, H, TW, TH, observations)
    assert before == after
    assert all(item.motion_norm is None and item.alpha_t is None for item in after)
    assert all(item.matches_ts0_placement for item in smooth_video_sequence(W, H, TW, TH, observations, alpha=1.0))
    assert [item.frame for item in adaptive] == [item.frame for item in before]
    assert [(item.w, item.h, item.crop_w, item.crop_h) for item in adaptive] == [
        (item.w, item.h, item.crop_w, item.crop_h) for item in before
    ]


def test_guarded_ema_safe_proposal_is_identical_to_fixed_ema_placement():
    observations = [obs(0, 700.0), obs(1, 720.0), obs(2, 740.0)]
    fixed = smooth_video_sequence(W, H, TW, TH, observations, alpha=0.5)
    guarded = smooth_video_sequence_guarded(W, H, TW, TH, observations)
    assert [(item.x, item.y, item.w, item.h) for item in guarded] == [
        (item.x, item.y, item.w, item.h) for item in fixed
    ]
    assert [item.ema_center_x for item in guarded] == [item.ema_center_x for item in fixed]
    assert all(item.guard_applied is False for item in guarded)
    assert all(item.placement_status == PLACEMENT_TS3_GUARDED_SMOOTHED for item in guarded)


def test_guarded_ema_slight_violation_moves_only_to_nearest_safe_boundary():
    observations = [obs(0, 700.0), obs(1, 1220.0)]
    fixed = smooth_video_sequence(W, H, TW, TH, observations)
    guarded = smooth_video_sequence_guarded(W, H, TW, TH, observations)
    safe_min, safe_max = containment_safe_top_left_interval(W, CROP_W, 1220.0)
    assert fixed[1].x == safe_min - 8
    assert guarded[1].x == safe_min
    assert guarded[1].safe_x_min == safe_min and guarded[1].safe_x_max == safe_max
    assert guarded[1].guard_correction_x == 8
    assert guarded[1].guard_correction_y == 0


@pytest.mark.parametrize(
    ("previous_center", "current_center", "expected_boundary", "raw_side"),
    [(1300.0, 100.0, 100, "left"), (300.0, 1500.0, 995, "right")],
)
def test_guarded_ema_severe_lag_projects_nearest_without_snapping_to_raw_cmp1(
    previous_center, current_center, expected_boundary, raw_side
):
    observations = [obs(0, previous_center), obs(1, current_center)]
    guarded = smooth_video_sequence_guarded(W, H, TW, TH, observations)
    raw_x = place_crop_from_center(W, H, TW, TH, (current_center, 450.0))[0]
    assert guarded[1].x == expected_boundary
    assert guarded[1].x != raw_x
    assert guarded[1].guard_applied
    assert raw_side in ("left", "right")


def test_guarded_ema_vertical_ratio_projects_with_exact_derived_height():
    fw, fh = 1080, 1920
    observations = [obs(0, 540.0, 1600.0), obs(1, 540.0, 100.0)]
    fixed = smooth_video_sequence(fw, fh, 16, 9, observations)
    guarded = smooth_video_sequence_guarded(fw, fh, 16, 9, observations)
    assert fixed[1].y > 100
    assert guarded[1].y == 100
    assert guarded[1].y <= 100.0 < guarded[1].y + guarded[1].h
    assert guarded[1].h == derived_height(guarded[1].w, 16, 9)
    assert guarded[1].x == 0
    assert_geometry_valid(guarded, width=fw, height=fh)


def test_guarded_ema_inherits_first_frame_gap_and_fallback_reset_semantics():
    observations = [
        obs(0, 300.0),
        obs(1, 500.0, fallback=True),
        obs(2, 1400.0),
        obs(10, 200.0),
    ]
    guarded = smooth_video_sequence_guarded(W, H, TW, TH, observations)
    assert guarded[0].reset_reason is None and guarded[0].matches_ts0_placement
    assert guarded[1].placement_status == PLACEMENT_FALLBACK_CENTER_CROP
    assert guarded[1].guard_applied is None
    assert guarded[2].reset_reason == RESET_FALLBACK and guarded[2].matches_ts0_placement
    assert guarded[3].reset_reason == RESET_FRAME_GAP and guarded[3].matches_ts0_placement


def test_guarded_ema_is_deterministic_and_preserves_ts0_ts1_ts2_and_geometry_contracts():
    observations = [
        obs(5, 1400.0),
        obs(3, 200.0),
        obs(4, 1000.0),
        obs(12, 600.0, fallback=True),
    ]
    ts1_before = smooth_video_sequence(W, H, TW, TH, observations)
    ts2_before = smooth_video_sequence_adaptive(W, H, TW, TH, observations)
    guarded = smooth_video_sequence_guarded(W, H, TW, TH, observations)
    replay = smooth_video_sequence_guarded(W, H, TW, TH, list(reversed(observations)))
    assert guarded == replay
    assert smooth_video_sequence(W, H, TW, TH, observations) == ts1_before
    assert smooth_video_sequence_adaptive(W, H, TW, TH, observations) == ts2_before
    assert [item.frame for item in guarded] == sorted(item.frame for item in observations)
    assert [(item.w, item.h, item.crop_w, item.crop_h) for item in guarded] == [
        (item.w, item.h, item.crop_w, item.crop_h) for item in ts1_before
    ]
    by_frame = {item.frame: item for item in observations}
    for item in guarded:
        assert 0 <= item.x and item.x + item.w <= W
        assert 0 <= item.y and item.y + item.h <= H
        assert item.h == derived_height(item.w, TW, TH)
        observation = by_frame[item.frame]
        if not observation.fallback:
            assert item.x <= observation.ideal_center_x < item.x + item.w
            assert item.y <= observation.ideal_center_y < item.y + item.h


def test_containment_interval_handles_horizontal_crop_and_frame_edges_exactly():
    assert containment_safe_top_left_interval(W, CROP_W, 0.0) == (0, 0)
    assert containment_safe_top_left_interval(W, CROP_W, W - 0.1) == (MAX_X, MAX_X)
    middle_min, middle_max = containment_safe_top_left_interval(W, CROP_W, 800.0)
    assert middle_min == 295
    assert middle_max == 800
    assert 0 <= middle_min <= middle_max <= MAX_X
    with pytest.raises(ValueError, match="half-open frame"):
        containment_safe_top_left_interval(W, CROP_W, float(W))


def test_bbox_maximum_overlap_interval_covers_containable_equal_and_oversized_cases():
    smaller = maximum_overlap_safe_top_left_interval(W, CROP_W, 700.0, 900.0)
    assert (smaller.minimum, smaller.maximum) == (394, 700)
    assert smaller.fully_containable and smaller.maximum_overlap == Fraction(200)

    equal = maximum_overlap_safe_top_left_interval(W, CROP_W, 400.0, 906.0)
    assert (equal.minimum, equal.maximum) == (400, 400)
    assert equal.fully_containable and equal.maximum_overlap == Fraction(CROP_W)

    oversized = maximum_overlap_safe_top_left_interval(W, CROP_W, 300.0, 1000.0)
    assert (oversized.minimum, oversized.maximum) == (300, 494)
    assert not oversized.fully_containable
    assert oversized.maximum_overlap == Fraction(CROP_W)


def test_bbox_maximum_overlap_interval_handles_fractional_integer_infeasibility_and_edges():
    fractional = maximum_overlap_safe_top_left_interval(10, 1, 1.3, 2.2)
    assert (fractional.minimum, fractional.maximum) == (1, 1)
    assert not fractional.fully_containable
    assert fractional.maximum_overlap == Fraction(7, 10)

    left = maximum_overlap_safe_top_left_interval(W, CROP_W, 0.0, 40.5)
    right = maximum_overlap_safe_top_left_interval(W, CROP_W, W - 40.5, float(W))
    assert (left.minimum, left.maximum) == (0, 0)
    assert (right.minimum, right.maximum) == (MAX_X, MAX_X)
    assert left.fully_containable and right.fully_containable


def test_bbox_guarded_ema_projects_to_nearest_maximum_visibility_placement():
    observations = [obs(0, 700.0), obs(1, 1220.0)]
    bboxes = {0: (650.0, 350.0, 750.0, 550.0), 1: (1100.0, 350.0, 1400.0, 550.0)}
    fixed = smooth_video_sequence(W, H, TW, TH, observations)
    guarded = smooth_video_sequence_bbox_guarded(W, H, TW, TH, observations, bboxes)
    x_interval = maximum_overlap_safe_top_left_interval(W, CROP_W, 1100.0, 1400.0)
    assert fixed[1].x < x_interval.minimum
    assert guarded[1].x == x_interval.minimum
    assert guarded[1].guard_applied
    assert guarded[1].placement_status == PLACEMENT_TS4_BBOX_GUARDED_SMOOTHED
    assert guarded[1].bbox_fully_containable
    assert guarded[1].visible_fraction_after_guard == 1.0
    assert guarded[1].visible_gain > 0.0
    assert [item.ema_center_x for item in guarded] == [item.ema_center_x for item in fixed]


def test_bbox_guarded_ema_oversized_bbox_fallback_reset_and_determinism():
    fw, fh = 1000, 600
    observations = [
        obs(0, 800.0, 300.0),
        obs(1, 200.0, 300.0),
        obs(2, 500.0, 300.0, fallback=True),
        obs(3, 900.0, 300.0),
    ]
    bboxes = {
        0: (100.0, 50.0, 900.0, 550.0),
        1: (100.0, 50.0, 900.0, 550.0),
        3: (850.0, 200.0, 950.0, 400.0),
    }
    first = smooth_video_sequence_bbox_guarded(fw, fh, TW, TH, observations, bboxes)
    replay = smooth_video_sequence_bbox_guarded(
        fw, fh, TW, TH, list(reversed(observations)), bboxes
    )
    assert first == replay
    assert first[0].bbox_larger_than_crop
    assert first[0].guard_mode_x == "MAXIMUM_OVERLAP"
    assert first[2].placement_status == PLACEMENT_FALLBACK_CENTER_CROP
    assert first[2].guard_applied is None
    assert first[3].reset_reason == RESET_FALLBACK
    assert all(item.w == first[0].w and item.h == first[0].h for item in first)
    assert_geometry_valid(first, width=fw, height=fh)


def test_bbox_maximum_overlap_closed_form_matches_exhaustive_discrete_geometry():
    rng = random.Random(5403)
    for _ in range(500):
        frame = rng.randint(3, 20)
        crop = Fraction(rng.randint(1, frame * 4), 4)
        if crop > frame:
            crop = Fraction(frame)
        start_tenth = rng.randint(0, frame * 10 - 1)
        end_tenth = rng.randint(start_tenth + 1, frame * 10)
        start = Fraction(start_tenth, 10)
        end = Fraction(end_tenth, 10)
        result = maximum_overlap_safe_top_left_interval(
            frame, crop, float(start), float(end)
        )
        legal = range(0, int(Fraction(frame) - crop) + 1)

        def overlap(q):
            return max(Fraction(0), min(Fraction(q) + crop, end) - max(Fraction(q), start))

        maximum = max(overlap(q) for q in legal)
        maximizers = [q for q in legal if overlap(q) == maximum]
        assert (result.minimum, result.maximum) == (min(maximizers), max(maximizers))
        assert result.maximum_overlap == maximum
        assert result.fully_containable is (maximum == end - start)


def test_projected_state_long_guard_run_keeps_every_state_feasible():
    observations = [obs(0, 300.0)] + [obs(frame, 1300.0) for frame in range(1, 41)]
    bboxes = {0: (200.0, 350.0, 400.0, 550.0)} | {
        frame: (1100.0, 350.0, 1400.0, 550.0) for frame in range(1, 41)
    }

    projected = smooth_video_sequence_projected_state_bbox_guarded(
        W, H, TW, TH, observations, bboxes
    )

    assert all(
        item.safe_x_min <= item.x <= item.safe_x_max
        and item.safe_y_min <= item.y <= item.safe_y_max
        for item in projected
    )
    assert all(item.state_output_residual_l1 == 0.0 for item in projected)
    assert all(
        item.placement_status == PLACEMENT_TS5_PROJECTED_STATE_SMOOTHED
        for item in projected
    )
    assert projected[1].guard_applied
    assert projected[1].ema_center_x == (
        projected[1].proposal_center_x + projected[1].guard_correction_x
    )


def test_projected_state_stationary_and_uniform_motion_are_stable_and_deterministic():
    stationary = [obs(frame, 800.0) for frame in range(20)]
    stationary_bboxes = {frame: (700.0, 350.0, 900.0, 550.0) for frame in range(20)}
    stable = smooth_video_sequence_projected_state_bbox_guarded(
        W, H, TW, TH, stationary, stationary_bboxes
    )
    assert len({(item.x, item.y, item.ema_center_x, item.ema_center_y) for item in stable}) == 1
    assert not any(item.guard_applied for item in stable)

    moving = [obs(frame, 400.0 + 20.0 * frame) for frame in range(20)]
    moving_bboxes = {
        frame: (330.0 + 20.0 * frame, 350.0, 470.0 + 20.0 * frame, 550.0)
        for frame in range(20)
    }
    first = smooth_video_sequence_projected_state_bbox_guarded(
        W, H, TW, TH, moving, moving_bboxes
    )
    replay = smooth_video_sequence_projected_state_bbox_guarded(
        W, H, TW, TH, list(reversed(moving)), moving_bboxes
    )
    assert first == replay
    assert all(a.x <= b.x for a, b in zip(first, first[1:]))


def test_projected_state_sudden_jump_and_oversized_bbox_feed_back_exact_correction():
    fw, fh = 1000, 600
    observations = [obs(0, 200.0, 300.0), obs(1, 850.0, 300.0), obs(2, 850.0, 300.0)]
    bboxes = {
        0: (100.0, 50.0, 900.0, 550.0),
        1: (100.0, 50.0, 900.0, 550.0),
        2: (100.0, 50.0, 900.0, 550.0),
    }
    projected = smooth_video_sequence_projected_state_bbox_guarded(
        fw, fh, TW, TH, observations, bboxes
    )
    assert all(item.bbox_larger_than_crop for item in projected)
    assert all(item.guard_mode_x == "MAXIMUM_OVERLAP" for item in projected)
    assert projected[1].ema_center_x == pytest.approx(
        projected[1].proposal_center_x + projected[1].guard_correction_x
    )
    assert projected[2].proposal_center_x == pytest.approx(
        0.5 * observations[2].ideal_center_x + 0.5 * projected[1].ema_center_x
    )


def test_projected_state_moving_feasible_boundary_remains_feasible_without_sign_flip():
    observations = [obs(frame, 1200.0 + 5.0 * frame) for frame in range(30)]
    bboxes = {
        frame: (1050.0 + 5.0 * frame, 300.0, 1350.0 + 5.0 * frame, 600.0)
        for frame in range(30)
    }
    projected = smooth_video_sequence_projected_state_bbox_guarded(
        W, H, TW, TH, observations, bboxes
    )
    assert all(item.safe_x_min <= item.x <= item.safe_x_max for item in projected)
    deltas = [current.x - previous.x for previous, current in zip(projected, projected[1:])]
    assert all(delta >= 0 for delta in deltas)
    assert_geometry_valid(projected)


def test_projected_state_reuses_ts4_projection_and_reset_fallback_semantics():
    observations = [
        obs(0, 300.0), obs(1, 1300.0), obs(4, 1200.0),
        obs(5, 800.0, fallback=True), obs(6, 400.0),
    ]
    bboxes = {
        0: (200.0, 350.0, 400.0, 550.0),
        1: (1100.0, 350.0, 1400.0, 550.0),
        4: (1000.0, 350.0, 1300.0, 550.0),
        6: (300.0, 350.0, 500.0, 550.0),
    }
    ts4 = smooth_video_sequence_bbox_guarded(W, H, TW, TH, observations, bboxes)
    ts5 = smooth_video_sequence_projected_state_bbox_guarded(W, H, TW, TH, observations, bboxes)
    assert (ts5[0].x, ts5[0].y) == (ts4[0].x, ts4[0].y)
    assert ts5[2].reset_reason == ts4[2].reset_reason == RESET_FRAME_GAP
    assert ts5[3].placement_status == ts4[3].placement_status == PLACEMENT_FALLBACK_CENTER_CROP
    assert ts5[4].reset_reason == ts4[4].reset_reason == RESET_FALLBACK
    for item in ts5:
        if item.placement_status == PLACEMENT_FALLBACK_CENTER_CROP:
            continue
        state_placement = place_crop_from_center(
            W, H, TW, TH, (item.ema_center_x, item.ema_center_y)
        )[:2]
        assert state_placement == (item.x, item.y)
