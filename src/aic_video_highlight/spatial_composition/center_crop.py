"""Deterministic center-crop geometry for the official output contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

_RATIONAL_LIMIT = 1_000_000


@dataclass(frozen=True, slots=True)
class CenterCropBox:
    x: int
    y: int
    w: int


def _positive_ratio_component(value: int | float, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return Fraction(number).limit_denominator(_RATIONAL_LIMIT)


def compute_center_crop(
    width: int, height: int, target_w: int | float, target_h: int | float
) -> CenterCropBox:
    """Largest target-ratio rectangle centred inside the frame, integer-safe.

    The emitted width is floored in the constrained dimension so that the
    evaluator-derived height ``h = w * target_h / target_w`` never exceeds the
    frame height, per the official out-of-bounds rule.
    """
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    tw = _positive_ratio_component(target_w, "target_w")
    th = _positive_ratio_component(target_h, "target_h")
    target_ratio = tw / th

    if Fraction(width, height) >= target_ratio:
        crop_w = (height * tw) // th
        if crop_w < 1:
            raise ValueError(
                "no target-ratio crop of width >= 1 fits inside the frame height"
            )
        crop_h = height
    else:
        crop_w = width
        crop_h = (width * th) // tw
        if crop_h < 1:
            crop_h = 1

    x = (width - crop_w) // 2
    y = (height - crop_h) // 2
    return CenterCropBox(x=int(x), y=int(y), w=int(crop_w))


def derived_height(
    bbox_width: int, target_w: int | float, target_h: int | float
) -> Fraction:
    """Exact evaluator-derived height ``h = w * target_h / target_w``."""
    if isinstance(bbox_width, bool) or not isinstance(bbox_width, int) or bbox_width <= 0:
        raise ValueError("bbox_width must be a positive integer")
    tw = _positive_ratio_component(target_w, "target_w")
    th = _positive_ratio_component(target_h, "target_h")
    return Fraction(bbox_width) * th / tw
