"""Aspect-ratio-agnostic subject-focus composition contract.

This is the minimal GENERIC_CMP_OUTPUT_CONTRACT_V1 needed by FTNet
preprocessing.  It is deliberately independent of AIC 9:16 / 16:9 submission
geometry and never decides highlight retention.
"""

from __future__ import annotations

from dataclasses import dataclass

CMP_VERSION = "generic-cmp-contract-v1"

RENDER_MODE_SUBJECT_FOCUS = "subject_focus"
RENDER_MODE_ORIGINAL_VIEW = "original_view"

FALLBACK_NONE = "NONE"
FALLBACK_ORIGINAL_VIEW = "ORIGINAL_VIEW"

REASON_NO_SUBJECT = "NO_SUBJECT"
REASON_INVALID_SUBJECT = "INVALID_SUBJECT"


class GenericFocusError(ValueError):
    """Raised when focus planning inputs violate the contract."""


@dataclass(frozen=True, slots=True)
class FocusPlan:
    cmp_version: str
    render_mode: str
    subject_center: tuple[float, float] | None
    focus_center: tuple[float, float]
    focus_scale: float
    focus_window: tuple[float, float, float, float]
    geometric_context_retention: float
    fallback_state: str
    fallback_reason: str | None
    plan_valid: bool


def _valid_box(box: tuple[float, float, float, float] | None) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return all(0.0 <= value <= 1.0 for value in box) and x2 > x1 and y2 > y1


def _original_view(reason: str) -> FocusPlan:
    return FocusPlan(
        cmp_version=CMP_VERSION,
        render_mode=RENDER_MODE_ORIGINAL_VIEW,
        subject_center=None,
        focus_center=(0.5, 0.5),
        focus_scale=1.0,
        focus_window=(0.0, 0.0, 1.0, 1.0),
        geometric_context_retention=1.0,
        fallback_state=FALLBACK_ORIGINAL_VIEW,
        fallback_reason=reason,
        plan_valid=False,
    )


def plan_subject_focus(
    subject_box_xyxy: tuple[float, float, float, float] | None,
    *,
    margin_fraction: float = 0.35,
    target_aspect_ratio: float | None = None,
) -> FocusPlan:
    """Plan a normalized source-coordinate focus window around the subject.

    ``subject_box_xyxy`` is normalized to ``[0,1]`` in the original frame.
    No legal subject -> conservative original-view fallback (never a DROP).
    """

    if margin_fraction < 0.0:
        raise GenericFocusError("margin_fraction cannot be negative")
    if target_aspect_ratio is not None and target_aspect_ratio <= 0.0:
        raise GenericFocusError("target_aspect_ratio must be positive")
    if subject_box_xyxy is None:
        return _original_view(REASON_NO_SUBJECT)
    if not _valid_box(subject_box_xyxy):
        return _original_view(REASON_INVALID_SUBJECT)

    x1, y1, x2, y2 = subject_box_xyxy
    subject_w = x2 - x1
    subject_h = y2 - y1
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    width = subject_w * (1.0 + 2.0 * margin_fraction)
    height = subject_h * (1.0 + 2.0 * margin_fraction)
    if target_aspect_ratio is not None:
        if width / height < target_aspect_ratio:
            width = height * target_aspect_ratio
        else:
            height = width / target_aspect_ratio

    width = min(width, 1.0)
    height = min(height, 1.0)
    left = min(max(center_x - width / 2.0, 0.0), 1.0 - width)
    top = min(max(center_y - height / 2.0, 0.0), 1.0 - height)
    focus_window = (left, top, width, height)
    return FocusPlan(
        cmp_version=CMP_VERSION,
        render_mode=RENDER_MODE_SUBJECT_FOCUS,
        subject_center=(center_x, center_y),
        focus_center=(left + width / 2.0, top + height / 2.0),
        focus_scale=float(width * height),
        focus_window=focus_window,
        geometric_context_retention=float(width * height),
        fallback_state=FALLBACK_NONE,
        fallback_reason=None,
        plan_valid=True,
    )


def geometric_context_retention(focus_window: tuple[float, float, float, float]) -> float:
    _x, _y, w, h = focus_window
    return float(max(0.0, w) * max(0.0, h))
