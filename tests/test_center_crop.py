import math

import pytest

from aic_video_highlight.spatial_composition.center_crop import (
    CenterCropBox,
    compute_center_crop,
    derived_height,
)


def test_center_crop_source_matches_target_returns_full_frame() -> None:
    box = compute_center_crop(1920, 1080, 16, 9)

    assert box == CenterCropBox(x=0, y=0, w=1920)
    assert derived_height(box.w, 16, 9) == 1080


def test_center_crop_wide_source_to_portrait_target_stays_in_bounds() -> None:
    box = compute_center_crop(1920, 1080, 9, 16)

    assert box.x == 656
    assert box.y == 0
    assert box.w == 607
    assert box.x + box.w <= 1920
    assert derived_height(box.w, 9, 16) <= 1080


def test_center_crop_portrait_source_to_landscape_target_is_centered() -> None:
    box = compute_center_crop(1080, 1920, 16, 9)

    assert box.x == 0
    assert box.y == 656
    assert box.w == 1080
    assert derived_height(box.w, 16, 9) == 607.5
    assert box.y + derived_height(box.w, 16, 9) <= 1920


def test_center_crop_square_source_to_portrait_target() -> None:
    box = compute_center_crop(1000, 1000, 9, 16)

    assert box.w == 562
    assert box.x == 219
    assert box.y == 0
    assert box.x + box.w <= 1000
    assert derived_height(box.w, 9, 16) <= 1000


def test_center_crop_odd_dimensions_use_floor_policy() -> None:
    box = compute_center_crop(1081, 1921, 9, 16)

    assert box.w == 1080
    assert box.x == 0
    assert box.y == 0
    assert box.x + box.w <= 1081
    assert derived_height(box.w, 9, 16) <= 1921


def test_center_crop_tiny_frame_remains_valid() -> None:
    box = compute_center_crop(2, 2, 9, 16)

    assert box.w == 1
    assert box.x == 0
    assert box.y == 0
    assert box.x + box.w <= 2
    assert derived_height(box.w, 9, 16) <= 2


def test_center_crop_infeasible_geometry_raises() -> None:
    with pytest.raises(ValueError):
        compute_center_crop(1, 1, 9, 16)


@pytest.mark.parametrize(
    "target_ratio",
    [(0, 9), (16, 0), (-16, 9), (16, -9)],
)
def test_center_crop_rejects_invalid_target_ratio(target_ratio: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        compute_center_crop(1920, 1080, *target_ratio)


def test_center_crop_rejects_non_finite_target_ratio() -> None:
    with pytest.raises(ValueError):
        compute_center_crop(1920, 1080, float("nan"), 9)
    with pytest.raises(ValueError):
        compute_center_crop(1920, 1080, 16, math.inf)


@pytest.mark.parametrize(
    "width,height,target",
    [
        (1920, 1080, (16, 9)),
        (1920, 1080, (9, 16)),
        (1080, 1920, (16, 9)),
        (1080, 1920, (9, 16)),
        (1280, 720, (9, 16)),
        (720, 1280, (16, 9)),
        (854, 480, (9, 16)),
        (1000, 1000, (16, 9)),
        (1000, 1000, (9, 16)),
        (640, 360, (9, 16)),
        (1920, 1080, (4, 3)),
        (1440, 1080, (16, 9)),
    ],
)
def test_center_crop_never_overflows_frame(width: int, height: int, target: tuple[int, int]) -> None:
    tw, th = target
    box = compute_center_crop(width, height, tw, th)

    assert box.x >= 0
    assert box.y >= 0
    assert box.w > 0
    assert box.x + box.w <= width
    assert derived_height(box.w, tw, th) <= height


def test_center_crop_is_deterministic() -> None:
    first = compute_center_crop(1920, 1080, 9, 16)
    second = compute_center_crop(1920, 1080, 9, 16)

    assert first == second
