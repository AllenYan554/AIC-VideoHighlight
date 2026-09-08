"""Diagnostic-only subject box -> target-ratio crop transform.

Not the Stage 5.3 production algorithm: a minimal deterministic transform that
fully contains the primary subject, keeps the exact target ratio, and stays
strictly inside the frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

STATUS_CONTAINS_SUBJECT = "SUBJECT_CENTERED_CONTAINING"
STATUS_DEGRADED_MAX_CROP = "DEGRADED_MAX_LEGAL_CROP"


@dataclass(frozen=True, slots=True)
class DiagnosticCrop:
    x: int
    y: int
    w: int
    h: int
    status: str
    contains_subject: bool
    degraded: bool


def _integer_ratio_components(
    target_w: int | float, target_h: int | float
) -> tuple[int, int]:
    tw = Fraction(str(float(target_w)))
    th = Fraction(str(float(target_h)))
    if tw.denominator != 1 or th.denominator != 1:
        raise ValueError("diagnostic transform requires integer targetRatioWH components")
    tw_i, th_i = int(tw), int(th)
    if tw_i <= 0 or th_i <= 0:
        raise ValueError("targetRatioWH components must be positive")
    return tw_i, th_i


def subject_centered_crop(
    subject_box: tuple[float, float, float, float],
    width: int,
    height: int,
    target_w: int | float,
    target_h: int | float,
) -> DiagnosticCrop:
    """Minimal enclosing exact-ratio crop around the subject, shifted into frame."""
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    x1, y1, x2, y2 = (float(v) for v in subject_box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError("subject box must be finite with positive extent")

    tw_i, th_i = _integer_ratio_components(target_w, target_h)

    k_max = min(width // tw_i, height // th_i)
    if k_max < 1:
        raise ValueError("no target-ratio crop fits inside the frame")

    sub_w = x2 - x1
    sub_h = y2 - y1
    k_needed = max(
        math.ceil(Fraction(str(sub_w)) / tw_i),
        math.ceil(Fraction(str(sub_h)) / th_i),
    )
    contains = k_needed <= k_max
    k = k_needed if contains else k_max
    crop_w = tw_i * int(k)
    crop_h = th_i * int(k)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    x = int(math.floor(cx - crop_w / 2))
    y = int(math.floor(cy - crop_h / 2))
    x = min(max(x, 0), width - crop_w)
    y = min(max(y, 0), height - crop_h)

    contains_subject = (
        contains
        and x <= x1
        and y <= y1
        and x + crop_w >= x2
        and y + crop_h >= y2
    )
    status = STATUS_CONTAINS_SUBJECT if contains else STATUS_DEGRADED_MAX_CROP
    return DiagnosticCrop(
        x=x,
        y=y,
        w=crop_w,
        h=crop_h,
        status=status,
        contains_subject=contains_subject,
        degraded=not contains,
    )
