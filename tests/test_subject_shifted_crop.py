"""Stage 5.3 CMP-1 geometry, sanitization, strata, and metrics tests."""

from __future__ import annotations

import math

import pytest

from aic_video_highlight.spatial_composition.center_crop import compute_center_crop, derived_height
from aic_video_highlight.spatial_composition.composition_metrics import (
    classify_center_stratum,
    crop_rect_from_xywh,
    crop_shift_normalized,
    horizontal_center_offset,
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
    PLACEMENT_SUBJECT_SHIFTED,
    SANITIZE_ABSENT,
    SANITIZE_INVALID,
    SANITIZE_OK,
    compute_subject_shifted_crop,
    sanitize_primary_bbox,
    stage5_1_crop_height,
)

W, H = 534, 300
TW, TH = 9, 16
CROP_W = (H * TW) // TH
CROP_H = H
CENTER_X = (W - CROP_W) // 2


def test_zero_overflow_bbox_sanitizes_unchanged():
    sanitized = sanitize_primary_bbox((100.0, 50.0, 200.0, 250.0), W, H)
    assert sanitized.status == SANITIZE_OK
    assert sanitized.as_xyxy() == (100.0, 50.0, 200.0, 250.0)
    assert (sanitized.clamp_left, sanitized.clamp_top, sanitized.clamp_right, sanitized.clamp_bottom) == (0.0, 0.0, 0.0, 0.0)


def test_slight_overflow_clamps_and_records_amounts():
    sanitized = sanitize_primary_bbox((-2.0, -0.5, 100.0, 301.0), W, H)
    assert sanitized.status == SANITIZE_OK
    assert sanitized.as_xyxy() == (0.0, 0.0, 100.0, 300.0)
    assert sanitized.clamp_left == 2.0
    assert sanitized.clamp_top == 0.5
    assert sanitized.clamp_bottom == 1.0
    assert sanitized.clamp_right == 0.0


def test_severe_overflow_clamps_into_frame():
    sanitized = sanitize_primary_bbox((-100.0, -100.0, 50.0, 60.0), W, H)
    assert sanitized.status == SANITIZE_OK
    assert sanitized.as_xyxy() == (0.0, 0.0, 50.0, 60.0)


def test_clamp_invalid_when_box_degenerates():
    sanitized = sanitize_primary_bbox((600.0, 100.0, 610.0, 200.0), W, H)
    assert sanitized.status == SANITIZE_INVALID
    assert sanitized.x1 == float(W) and sanitized.x2 == float(W)
    zero_width = sanitize_primary_bbox((10.0, 10.0, 10.0, 50.0), W, H)
    assert zero_width.status == SANITIZE_INVALID
    with pytest.raises(ValueError):
        sanitize_primary_bbox((float("nan"), 10.0, 50.0, 60.0), W, H)


def test_absent_primary_is_distinct_from_invalid():
    assert sanitize_primary_bbox(None, W, H).status == SANITIZE_ABSENT


def test_center_primary_reproduces_center_crop():
    shifted = compute_subject_shifted_crop(W, H, TW, TH, (W / 2, H / 2))
    assert (shifted.x, shifted.y, shifted.w) == (CENTER_X, 0, CROP_W)
    assert shifted.placement_status == PLACEMENT_CENTER_EQUIVALENT


def test_left_border_primary_shifts_and_clamps():
    shifted = compute_subject_shifted_crop(W, H, TW, TH, (40.0, H / 2))
    assert shifted.x == 0
    assert shifted.placement_status == PLACEMENT_SUBJECT_SHIFTED
    assert shifted.clamped_x is True


def test_right_border_primary_shifts_and_clamps():
    shifted = compute_subject_shifted_crop(W, H, TW, TH, (520.0, H / 2))
    assert shifted.x == W - CROP_W
    assert shifted.clamped_x is True
    assert shifted.placement_status == PLACEMENT_SUBJECT_SHIFTED


def test_top_and_bottom_shift_with_portrait_16_9():
    width, height = 300, 534
    shifted_top = compute_subject_shifted_crop(width, height, 16, 9, (width / 2, 100.0))
    assert (shifted_top.w, shifted_top.crop_h) == (width, (width * 9) // 16)
    assert shifted_top.y == 100 - shifted_top.crop_h // 2
    assert shifted_top.placement_status == PLACEMENT_SUBJECT_SHIFTED
    shifted_bottom = compute_subject_shifted_crop(width, height, 16, 9, (width / 2, 530.0))
    assert shifted_bottom.y == int(math.floor(height - float(shifted_bottom.h)))
    assert shifted_bottom.y + float(shifted_bottom.h) <= height
    assert shifted_bottom.clamped_y is True


def test_huge_primary_keeps_max_crop_and_partial_visibility():
    sanitized = sanitize_primary_bbox((0.0, 0.0, 500.0, 295.0), W, H)
    shifted = compute_subject_shifted_crop(W, H, TW, TH, (sanitized.center_x, sanitized.center_y))
    assert shifted.crop_w == CROP_W
    rect = crop_rect_from_xywh(shifted.x, shifted.y, shifted.w, float(shifted.h))
    visible = subject_visible_fraction(sanitized, rect)
    assert 0.0 < visible < 1.0
    assert visible == pytest.approx((CROP_W * 295.0) / (500.0 * 295.0))


def test_tiny_primary_fully_visible():
    sanitized = sanitize_primary_bbox((100.0, 100.0, 101.0, 101.0), W, H)
    shifted = compute_subject_shifted_crop(W, H, TW, TH, (sanitized.center_x, sanitized.center_y))
    rect = crop_rect_from_xywh(shifted.x, shifted.y, shifted.w, float(shifted.h))
    assert subject_visible_fraction(sanitized, rect) == pytest.approx(1.0)
    assert subject_center_inside_crop(sanitized, rect)


def test_label_is_irrelevant_to_geometry():
    shifted_person = compute_subject_shifted_crop(W, H, TW, TH, (80.0, 150.0))
    shifted_car = compute_subject_shifted_crop(W, H, TW, TH, (80.0, 150.0))
    assert shifted_person == shifted_car


def test_fallback_crop_matches_center_crop():
    center = compute_center_crop(W, H, TW, TH)
    assert (center.x, center.y, center.w) == (CENTER_X, 0, CROP_W)
    assert PLACEMENT_FALLBACK_CENTER_CROP == "FALLBACK_CENTER_CROP"


def test_crop_size_equals_stage5_1_maximal_crop():
    center = compute_center_crop(W, H, TW, TH)
    for cx in (0.0, 100.0, 267.0, 533.0):
        shifted = compute_subject_shifted_crop(W, H, TW, TH, (cx, H / 2))
        assert shifted.crop_w == center.w
        assert shifted.w == center.w


def test_crop_position_clamped_into_bounds():
    for cx in (-50.0, 0.0, 267.0, 600.0, 10_000.0):
        shifted = compute_subject_shifted_crop(W, H, TW, TH, (cx, H / 2))
        assert 0 <= shifted.x <= W - CROP_W


def test_ratio_contract_exact():
    shifted = compute_subject_shifted_crop(W, H, TW, TH, (100.0, 150.0))
    assert shifted.h == derived_height(shifted.w, TW, TH)
    assert shifted.w * TH == pytest.approx(float(shifted.h) * TW)


def test_bounds_contract_for_all_subject_positions():
    for cx in range(0, W + 1, 37):
        for cy in range(0, H + 1, 29):
            shifted = compute_subject_shifted_crop(W, H, TW, TH, (float(cx), float(cy)))
            assert shifted.x >= 0 and shifted.x + shifted.w <= W
            assert shifted.y >= 0 and shifted.y + float(shifted.h) <= H + 1e-9


def test_deterministic_geometry_output():
    first = compute_subject_shifted_crop(W, H, TW, TH, (123.4, 150.0))
    second = compute_subject_shifted_crop(W, H, TW, TH, (123.4, 150.0))
    assert first == second
    assert sanitize_primary_bbox((-1.0, -1.0, 30.0, 30.0), W, H) == sanitize_primary_bbox((-1.0, -1.0, 30.0, 30.0), W, H)


def test_nine_16_and_16_9_ratio_configs():
    portrait = compute_subject_shifted_crop(300, 534, 16, 9, (150.0, 267.0))
    assert portrait.w == 300 and portrait.crop_h == (300 * 9) // 16
    landscape = compute_subject_shifted_crop(534, 300, 9, 16, (267.0, 150.0))
    assert landscape.w == (300 * 9) // 16 and landscape.crop_h == 300


def test_overflow_direction_amounts_and_thresholds():
    overflow = raw_bbox_overflow((-5.0, -1.0, 600.0, 310.0), W, H)
    assert overflow == {"left": 5.0, "top": 1.0, "right": 66.0, "bottom": 10.0}
    thresholds = overflow_threshold_frames([0.0, 0.5, 1.5, 3.5, 5.5, 10.5])
    assert thresholds == {">0px": 5, ">1px": 4, ">3px": 3, ">5px": 2, ">10px": 1}


def test_overflow_distribution_percentiles():
    distribution = overflow_distribution([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0])
    assert distribution["n"] == 10
    assert distribution["max"] == 100.0
    assert distribution["median"] == 4.5


def test_visibility_summary_and_thresholds():
    values = [1.0] * 6 + [0.8] * 2 + [0.4] * 2
    summary = summarize(values).as_dict()
    assert summary["n"] == 10
    assert summary["mean"] == pytest.approx(0.84)
    thresholds = visible_fraction_thresholds(values)
    assert thresholds[">=0.50"] == pytest.approx(0.8)
    assert thresholds[">=0.75"] == pytest.approx(0.8)
    assert thresholds[">=0.90"] == pytest.approx(0.6)
    assert thresholds[">=1.00"] == pytest.approx(0.6)


def test_center_offset_strata_classification():
    assert classify_center_stratum(0.05, 0.10, 0.25) == "near_center"
    assert classify_center_stratum(0.15, 0.10, 0.25) == "moderately_off_center"
    assert classify_center_stratum(0.30, 0.10, 0.25) == "strongly_off_center"
    assert horizontal_center_offset(sanitize_primary_bbox((0.0, 0.0, 53.4, 300.0), W, H), W) == pytest.approx(0.45)


def test_crop_shift_normalization():
    shift_x, shift_y = crop_shift_normalized(0, 168, 0.0, 300.0, W, H)
    assert shift_x == pytest.approx(abs((0 + 84) - 267) / W)
    assert shift_y == pytest.approx(0.0)


def test_geometry_is_pure_and_finite():
    shifted = compute_subject_shifted_crop(W, H, TW, TH, (267.0, 150.0))
    assert all(math.isfinite(value) for value in (shifted.ideal_x, shifted.ideal_y, float(shifted.h)))


def test_internal_crop_height_matches_frozen_centering_convention():
    cases = [(534, 300), (1920, 1080), (1080, 1920), (270, 480), (300, 534), (1280, 720), (720, 1280), (640, 360)]
    for width, height in cases:
        for tw, th in ((9, 16), (16, 9), (1, 1), (3, 4)):
            center = compute_center_crop(width, height, tw, th)
            internal_h = stage5_1_crop_height(width, height, tw, th)
            assert (height - internal_h) // 2 == center.y
            assert (width - center.w) // 2 == center.x
            assert 0 < internal_h <= height
