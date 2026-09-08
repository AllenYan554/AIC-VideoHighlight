"""Stage 5.3 composition behaviour metrics: visibility, shift, strata, overflow.

All metrics are diagnostic behaviour evidence; none of them is an official score.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

from aic_video_highlight.spatial_composition.subject_shifted_crop import SanitizedSubject

STRATUM_NEAR_CENTER = "near_center"
STRATUM_MODERATELY_OFF_CENTER = "moderately_off_center"
STRATUM_STRONGLY_OFF_CENTER = "strongly_off_center"
STRATUM_NO_SUBJECT = "no_subject"

VISIBILITY_THRESHOLDS = (0.50, 0.75, 0.90, 0.95, 1.00)
OVERFLOW_THRESHOLDS_PX = (0, 1, 3, 5, 10)


def crop_rect_from_xywh(x: int, y: int, w: int, derived_h: float) -> tuple[float, float, float, float]:
    """Official crop as an xyxy rectangle using the evaluator-derived height."""
    return (float(x), float(y), float(x + w), float(y) + float(derived_h))


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def subject_visible_fraction(subject: SanitizedSubject, crop_rect: Sequence[float]) -> float:
    """area(subject intersect crop) / area(subject) on the sanitized bbox."""
    area = subject.area
    if area <= 0.0:
        return 0.0
    return intersection_area(subject.as_xyxy(), crop_rect) / area


def subject_center_inside_crop(subject: SanitizedSubject, crop_rect: Sequence[float]) -> bool:
    return (
        crop_rect[0] <= subject.center_x < crop_rect[2]
        and crop_rect[1] <= subject.center_y < crop_rect[3]
    )


def crop_shift_normalized(
    crop_x: int,
    crop_w: int,
    crop_y: float,
    crop_h: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """|crop center - image center| normalized by frame width / height."""
    shift_x = abs((crop_x + crop_w / 2.0) - width / 2.0) / width
    shift_y = abs((crop_y + crop_h / 2.0) - height / 2.0) / height
    return (shift_x, shift_y)


def horizontal_center_offset(subject: SanitizedSubject, width: int) -> float:
    """Model-blind geometry feature: |cx/W - 0.5| of the sanitized subject."""
    if width <= 0:
        raise ValueError("width must be positive")
    return abs(subject.center_x / width - 0.5)


def classify_center_stratum(offset: float, near_center: float, strongly_off_center: float) -> str:
    """Preregistered thresholds: <near -> near_center; <strong -> moderate; else strong."""
    if offset < near_center:
        return STRATUM_NEAR_CENTER
    if offset < strongly_off_center:
        return STRATUM_MODERATELY_OFF_CENTER
    return STRATUM_STRONGLY_OFF_CENTER


def raw_bbox_overflow(
    raw_bbox: Sequence[float], width: int, height: int
) -> dict[str, float]:
    """Per-direction positive overflow of the raw bbox beyond the frame."""
    x1, y1, x2, y2 = (float(value) for value in raw_bbox)
    return {
        "left": max(0.0, -x1),
        "top": max(0.0, -y1),
        "right": max(0.0, x2 - width),
        "bottom": max(0.0, y2 - height),
    }


def max_overflow_px(overflow: dict[str, float]) -> float:
    return max(overflow.values()) if overflow else 0.0


@dataclass(frozen=True, slots=True)
class Summary:
    n: int
    mean: float | None
    median: float | None
    p10: float | None
    p50: float | None
    p90: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "mean": _round(self.mean),
            "median": _round(self.median),
            "p10": _round(self.p10),
            "p50": _round(self.p50),
            "p90": _round(self.p90),
        }


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    index = min(int(q * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


def summarize(values: Iterable[float]) -> Summary:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return Summary(0, None, None, None, None, None)
    return Summary(
        n=len(ordered),
        mean=statistics.fmean(ordered),
        median=statistics.median(ordered),
        p10=_percentile(ordered, 0.10),
        p50=_percentile(ordered, 0.50),
        p90=_percentile(ordered, 0.90),
    )


def visible_fraction_thresholds(values: Iterable[float]) -> dict[str, float]:
    """Share of values at or above each preregistered visibility threshold."""
    ordered = [float(value) for value in values]
    return {
        f">={threshold:.2f}": (round(sum(1 for value in ordered if value + 1e-9 >= threshold) / len(ordered), 6) if ordered else 0.0)
        for threshold in VISIBILITY_THRESHOLDS
    }


def overflow_threshold_frames(max_overflows: Iterable[float]) -> dict[str, int]:
    """Frame counts whose max per-direction overflow exceeds each px threshold."""
    ordered = [float(value) for value in max_overflows]
    return {
        f">{threshold}px": sum(1 for value in ordered if value > threshold + 1e-9)
        for threshold in OVERFLOW_THRESHOLDS_PX
    }


def overflow_distribution(max_overflows: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in max_overflows)
    if not ordered:
        return {"n": 0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

    def p(q: float) -> float:
        return round(_percentile(ordered, q), 3)

    return {
        "n": len(ordered),
        "median": round(statistics.median(ordered), 3),
        "p90": p(0.90),
        "p95": p(0.95),
        "p99": p(0.99),
        "max": round(ordered[-1], 3),
    }


def is_finite_point(x: float, y: float) -> bool:
    return math.isfinite(x) and math.isfinite(y)
