"""Stage 5.3 CMP-1 geometry: primary-bbox sanitization and subject-shifted crop.

CMP-1 keeps the Stage 5.1 frozen maximal legal target-ratio crop SIZE and only
changes the crop POSITION (single-variable control vs CMP-0 Center Crop).
The Stage 5.1 ``compute_center_crop`` function is reused verbatim for the crop
size; no alternative ratio logic is allowed in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from aic_video_highlight.spatial_composition.center_crop import (
    CenterCropBox,
    _positive_ratio_component,
    compute_center_crop,
    derived_height,
)

RAW_BBOX_LEN = 4

PLACEMENT_FALLBACK_CENTER_CROP = "FALLBACK_CENTER_CROP"
PLACEMENT_CENTER_EQUIVALENT = "CENTER_EQUIVALENT"
PLACEMENT_SUBJECT_SHIFTED = "SUBJECT_SHIFTED"

SANITIZE_OK = "OK"
SANITIZE_INVALID = "INVALID_SANITIZED_SUBJECT"
SANITIZE_ABSENT = "PRIMARY_ABSENT"


def _require_frame(width: int, height: int) -> None:
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")


def _require_raw_bbox(raw_bbox: Sequence[float]) -> tuple[float, float, float, float]:
    if len(raw_bbox) != RAW_BBOX_LEN:
        raise ValueError("raw bbox must be an xyxy 4-tuple")
    values = tuple(float(value) for value in raw_bbox)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("raw bbox must be finite")
    return values  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class SanitizedSubject:
    """Clamped primary bbox; the raw bbox is never modified or replaced."""

    x1: float
    y1: float
    x2: float
    y2: float
    status: str
    clamp_left: float
    clamp_top: float
    clamp_right: float
    clamp_bottom: float

    @property
    def valid(self) -> bool:
        return self.status == SANITIZE_OK

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def sanitize_primary_bbox(
    raw_bbox: Sequence[float] | None, width: int, height: int
) -> SanitizedSubject:
    """Deterministic first-version sanitization: clamp into [0, W] x [0, H].

    ``raw_bbox`` is the frozen Stage 5.2 primary xyxy box. If the clamped box
    has zero (or negative) extent, the subject is marked INVALID_SANITIZED_SUBJECT
    and CMP-1 must fall back to CMP-0. Clamp amounts are per-side positive values
    describing how much of the raw box was cut.
    """
    _require_frame(width, height)
    if raw_bbox is None:
        return SanitizedSubject(0.0, 0.0, 0.0, 0.0, SANITIZE_ABSENT, 0.0, 0.0, 0.0, 0.0)
    x1, y1, x2, y2 = _require_raw_bbox(raw_bbox)
    cx1 = min(max(x1, 0.0), float(width))
    cy1 = min(max(y1, 0.0), float(height))
    cx2 = min(max(x2, 0.0), float(width))
    cy2 = min(max(y2, 0.0), float(height))
    clamp_left = cx1 - x1
    clamp_top = cy1 - y1
    clamp_right = x2 - cx2
    clamp_bottom = y2 - cy2
    if cx2 <= cx1 or cy2 <= cy1:
        return SanitizedSubject(
            cx1, cy1, cx2, cy2, SANITIZE_INVALID, clamp_left, clamp_top, clamp_right, clamp_bottom
        )
    return SanitizedSubject(
        cx1, cy1, cx2, cy2, SANITIZE_OK, clamp_left, clamp_top, clamp_right, clamp_bottom
    )


def stage5_1_crop_height(width: int, height: int, target_w: int | float, target_h: int | float) -> int:
    """Stage 5.1 internal crop height implied by the frozen maximal legal crop.

    The frozen ``compute_center_crop`` constrains exactly one dimension using the
    exact condition ``W/H >= tw/th``: when true, the internal height is the frame
    height; otherwise it is ``floor(W * th / tw)``. This helper recovers that
    internal value with the same frozen formulas; the official derived height is
    computed separately via ``derived_height``. A consistency test pins this
    helper to the frozen centering convention.
    """
    center: CenterCropBox = compute_center_crop(width, height, target_w, target_h)
    tw = _positive_ratio_component(target_w, "target_w")
    th = _positive_ratio_component(target_h, "target_h")
    if Fraction(width, height) >= tw / th:
        return height
    return int((Fraction(width) * th) // tw)


@dataclass(frozen=True, slots=True)
class ShiftedCrop:
    """CMP-1 crop with official integer x/y/w plus the exact derived height."""

    x: int
    y: int
    w: int
    h: Fraction
    crop_w: int
    crop_h: int
    placement_status: str
    ideal_x: float
    ideal_y: float
    clamped_x: bool
    clamped_y: bool


def compute_subject_shifted_crop(
    width: int,
    height: int,
    target_w: int | float,
    target_h: int | float,
    subject_center: tuple[float, float],
) -> ShiftedCrop:
    """Maximal legal target-ratio crop re-centred on the subject, then clamped.

    The crop size (crop_w, official derived height) is exactly the Stage 5.1
    frozen maximal legal crop from ``compute_center_crop``; only x/y placement
    changes. Ideal placement uses the Stage 5.1 internal centering convention
    (floor(center - crop/2)) and is clamped into strict bounds, with the vertical
    bound tightened by the exact evaluator-derived height ``w * th / tw``.
    """
    _require_frame(width, height)
    center: CenterCropBox = compute_center_crop(width, height, target_w, target_h)
    crop_w = center.w
    crop_h = stage5_1_crop_height(width, height, target_w, target_h)
    derived_h = derived_height(crop_w, target_w, target_h)
    cx, cy = float(subject_center[0]), float(subject_center[1])
    if not math.isfinite(cx) or not math.isfinite(cy):
        raise ValueError("subject center must be finite")
    ideal_x = math.floor(cx - crop_w / 2.0)
    ideal_y = math.floor(cy - crop_h / 2.0)
    max_x = width - crop_w
    max_y = int(math.floor(Fraction(height) - derived_h))
    if max_x < 0 or max_y < 0:
        raise ValueError("no target-ratio crop of the frozen size fits inside the frame")
    x = min(max(ideal_x, 0), max_x)
    y = min(max(ideal_y, 0), max_y)
    clamped_x = x != ideal_x
    clamped_y = y != ideal_y
    center_equivalent = x == center.x and y == center.y
    placement = PLACEMENT_CENTER_EQUIVALENT if center_equivalent else PLACEMENT_SUBJECT_SHIFTED
    return ShiftedCrop(
        x=int(x),
        y=int(y),
        w=int(crop_w),
        h=derived_h,
        crop_w=int(crop_w),
        crop_h=int(crop_h),
        placement_status=placement,
        ideal_x=float(ideal_x),
        ideal_y=float(ideal_y),
        clamped_x=clamped_x,
        clamped_y=clamped_y,
    )
