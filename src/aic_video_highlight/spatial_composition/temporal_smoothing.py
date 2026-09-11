"""Stage 5.4 TS-1 fixed, TS-2 adaptive, TS-3 center-constrained, and
TS-4 bbox-aware and TS-5 projected-state constrained EMA methods.

Consumes the Stage 5.3 FINAL FROZEN CMP-1 geometry as TS-0 (temporal control
baseline) and changes ONLY the temporal continuity of the crop center / placement:
crop width, crop height, target ratio, frame selection, primary subject and
fallback semantics are all inherited unchanged from the frozen Stage 5.3 contract.

TS-1 uses a fixed exponential moving average (EMA) over the pre-clamp ideal crop center;
TS-2 changes only that coefficient to a frozen function of normalized raw horizontal
primary-subject motion. TS-3 retains the fixed TS-1 state and projects only an
unsafe output placement to the nearest crop containing the current frozen subject
center. All are placed with the exact Stage 5.3 frozen floor/clamp convention.
TS-4 replaces only TS-3's point constraint with the parameter-free set of
integer placements maximizing current frozen primary-bbox visible area, then
projects the unchanged TS-1 proposal to the nearest member of that set.
TS-5 reuses that exact projection and feeds only its integer correction back
into the continuous fixed-EMA state before the next frame.
Reset rules (no cross-sequence smoothing):
new video (one video per call), frozen temporal discontinuity (frame gap > 1 in
the frozen frame identity), and any fallback frame (FALLBACK_CENTER_CROP on the
current or the previous frame). No scene-cut detector, no tracking, no new pixels.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Callable, Sequence

from aic_video_highlight.spatial_composition.center_crop import (
    CenterCropBox,
    compute_center_crop,
    derived_height,
)
from aic_video_highlight.spatial_composition.subject_shifted_crop import (
    PLACEMENT_FALLBACK_CENTER_CROP,
    stage5_1_crop_height,
)

DEFAULT_EMA_ALPHA = 0.5
MOTION_ADAPTIVE_SMOOTHING_CEILING = 0.10
MOTION_ADAPTIVE_FULL_RESPONSE = 0.20

PLACEMENT_TS1_SMOOTHED = "TS1_SMOOTHED"
PLACEMENT_TS2_ADAPTIVE_SMOOTHED = "TS2_ADAPTIVE_SMOOTHED"
PLACEMENT_TS3_GUARDED_SMOOTHED = "TS3_GUARDED_SMOOTHED"
PLACEMENT_TS4_BBOX_GUARDED_SMOOTHED = "TS4_BBOX_GUARDED_SMOOTHED"
PLACEMENT_TS5_PROJECTED_STATE_SMOOTHED = "TS5_PROJECTED_STATE_SMOOTHED"

RESET_NEW_VIDEO = "NEW_VIDEO"
RESET_FRAME_GAP = "FRAME_GAP"
RESET_FALLBACK = "FALLBACK"

MAX_FRAME_GAP = 1

LARGE_JUMP_THRESHOLDS = (0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True, slots=True)
class TemporalObservation:
    """One frozen frame's temporal-relevant geometry (TS-0 CMP-1, pre-clamp center)."""

    frame: int
    fallback: bool
    ideal_center_x: float
    ideal_center_y: float


@dataclass(frozen=True, slots=True)
class SmoothedFrame:
    """One temporal treatment crop with frozen size and deterministic placement."""

    frame: int
    reset_reason: str | None
    ema_center_x: float | None
    ema_center_y: float | None
    x: int
    y: int
    w: int
    h: Fraction
    crop_w: int
    crop_h: int
    placement_status: str
    clamped_x: bool
    clamped_y: bool
    matches_ts0_placement: bool
    motion_norm: float | None = None
    alpha_t: float | None = None
    guard_applied: bool | None = None
    guard_correction_x: int | None = None
    guard_correction_y: int | None = None
    safe_x_min: int | None = None
    safe_x_max: int | None = None
    safe_y_min: int | None = None
    safe_y_max: int | None = None
    guard_mode_x: str | None = None
    guard_mode_y: str | None = None
    bbox_fully_containable: bool | None = None
    bbox_larger_than_crop: bool | None = None
    per_axis_infeasible_x: bool | None = None
    per_axis_infeasible_y: bool | None = None
    visible_fraction_before_guard: float | None = None
    visible_fraction_after_guard: float | None = None
    visible_gain: float | None = None
    proposal_center_x: float | None = None
    proposal_center_y: float | None = None
    projected_state_center_x: float | None = None
    projected_state_center_y: float | None = None
    state_output_residual_l1: float | None = None


@dataclass(frozen=True, slots=True)
class MaximumOverlapInterval:
    """Contiguous integer argmax set for one-axis bbox/crop overlap."""

    minimum: int
    maximum: int
    fully_containable: bool
    maximum_overlap: Fraction

    @property
    def mode(self) -> str:
        return "FULL_CONTAINMENT" if self.fully_containable else "MAXIMUM_OVERLAP"


def _require_frame(width: int, height: int) -> None:
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")


def _require_alpha(alpha: float) -> float:
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("alpha must be a finite value in (0, 1]")
    return value


def motion_adaptive_alpha(
    motion_norm: float,
    *,
    alpha_min: float = DEFAULT_EMA_ALPHA,
    smoothing_ceiling: float = MOTION_ADAPTIVE_SMOOTHING_CEILING,
    full_response_motion: float = MOTION_ADAPTIVE_FULL_RESPONSE,
) -> float:
    """Deterministic monotone Motion-Adaptive EMA v1 schedule.

    ``motion_norm`` is the frozen Stage 5.4 horizontal motion signal
    ``abs(X_t - X_(t-1)) / frame_width`` over consecutive, non-fallback raw
    primary-subject centers. The frozen 0.10 and 0.20 large-jump boundaries
    define the smoothing and full-response anchors; no learned or GT signal is
    consumed.
    """
    motion = float(motion_norm)
    minimum = _require_alpha(alpha_min)
    low = float(smoothing_ceiling)
    high = float(full_response_motion)
    if not math.isfinite(motion) or motion < 0.0:
        raise ValueError("motion_norm must be a finite nonnegative value")
    if minimum < DEFAULT_EMA_ALPHA or minimum > 1.0:
        raise ValueError("alpha_min must be in [0.5, 1.0]")
    if not math.isfinite(low) or not math.isfinite(high) or not 0.0 <= low < high:
        raise ValueError("motion thresholds must be finite and satisfy 0 <= low < high")
    transition = min(max((motion - low) / (high - low), 0.0), 1.0)
    return minimum + (1.0 - minimum) * transition


def containment_safe_top_left_interval(
    frame_extent: int,
    crop_extent: int | float | Fraction,
    subject_coordinate: float,
) -> tuple[int, int]:
    """Legal integer top-left interval whose half-open crop contains a point.

    For crop interval ``[q, q + C)`` and subject coordinate ``p``, containment
    requires ``p - C < q <= p``. This is intersected with the frozen legal
    placement range ``0 <= q <= floor(F - C)`` using exact ``Fraction``
    arithmetic for the derived vertical crop extent.
    """
    if isinstance(frame_extent, bool) or not isinstance(frame_extent, int) or frame_extent <= 0:
        raise ValueError("frame_extent must be a positive integer")
    crop = Fraction(crop_extent)
    coordinate = float(subject_coordinate)
    if crop <= 0 or crop > frame_extent:
        raise ValueError("crop_extent must be positive and fit inside frame_extent")
    if not math.isfinite(coordinate) or not 0.0 <= coordinate < float(frame_extent):
        raise ValueError("subject_coordinate must be finite and inside the half-open frame")
    point = Fraction(str(coordinate))
    lower = max(0, math.floor(point - crop) + 1)
    upper = min(math.floor(Fraction(frame_extent) - crop), math.floor(point))
    if lower > upper:
        raise ValueError("no legal crop placement contains the subject coordinate")
    return int(lower), int(upper)


def maximum_overlap_safe_top_left_interval(
    frame_extent: int,
    crop_extent: int | float | Fraction,
    bbox_start: float,
    bbox_end: float,
) -> MaximumOverlapInterval:
    """Integer placements maximizing half-open 1-D bbox/crop overlap.

    The continuous maximizer plateau is ``[b2-C,b1]`` when the bbox is no
    larger than the crop, and ``[b1,b2-C]`` when the bbox is larger.  We
    intersect that plateau with legal frame placements. If the plateau lies
    beyond the stricter integer legal bound ``floor(F-C)``, the nearest legal
    boundary maximizes overlap. If a nonempty clipped plateau contains no
    integer (possible for fractional bboxes), only its two adjacent legal
    integers can maximize the concave overlap function; those are evaluated
    exactly. When every legal placement has zero overlap, the complete legal
    integer domain is the tie set. The returned argmax is deterministic and
    parameter-free.
    """
    if isinstance(frame_extent, bool) or not isinstance(frame_extent, int) or frame_extent <= 0:
        raise ValueError("frame_extent must be a positive integer")
    crop = Fraction(crop_extent)
    start_value, end_value = float(bbox_start), float(bbox_end)
    if crop <= 0 or crop > frame_extent:
        raise ValueError("crop_extent must be positive and fit inside frame_extent")
    if not all(math.isfinite(value) for value in (start_value, end_value)):
        raise ValueError("bbox coordinates must be finite")
    if not 0.0 <= start_value < end_value <= float(frame_extent):
        raise ValueError("bbox must have positive extent inside the closed frame boundary")

    start = Fraction(str(start_value))
    end = Fraction(str(end_value))
    bbox_extent = end - start
    legal_max = math.floor(Fraction(frame_extent) - crop)

    def overlap(q: int) -> Fraction:
        return max(Fraction(0), min(Fraction(q) + crop, end) - max(Fraction(q), start))

    raw_plateau_start, raw_plateau_end = (
        (end - crop, start) if bbox_extent <= crop else (start, end - crop)
    )
    if raw_plateau_end < 0:
        plateau_start = plateau_end = Fraction(0)
    elif raw_plateau_start > legal_max:
        if overlap(legal_max) == 0:
            return MaximumOverlapInterval(0, legal_max, False, Fraction(0))
        plateau_start = plateau_end = Fraction(legal_max)
    else:
        plateau_start = max(Fraction(0), raw_plateau_start)
        plateau_end = min(Fraction(legal_max), raw_plateau_end)

    integer_min = math.ceil(plateau_start)
    integer_max = math.floor(plateau_end)

    if integer_min <= integer_max:
        maximum_overlap = overlap(integer_min)
        return MaximumOverlapInterval(
            int(integer_min),
            int(integer_max),
            maximum_overlap == bbox_extent,
            maximum_overlap,
        )

    candidates = {
        min(max(math.floor(plateau_start), 0), legal_max),
        min(max(math.ceil(plateau_end), 0), legal_max),
    }
    maximum_overlap = max(overlap(candidate) for candidate in candidates)
    if maximum_overlap == 0:
        return MaximumOverlapInterval(0, legal_max, False, Fraction(0))
    maximizers = sorted(candidate for candidate in candidates if overlap(candidate) == maximum_overlap)
    return MaximumOverlapInterval(
        maximizers[0],
        maximizers[-1],
        maximum_overlap == bbox_extent,
        maximum_overlap,
    )


def place_crop_from_center(
    width: int,
    height: int,
    target_w: int | float,
    target_h: int | float,
    center: tuple[float, float],
) -> tuple[int, int, int, Fraction, bool, bool]:
    """Frozen Stage 5.3 placement of the maximal legal crop at an arbitrary center.

    Identical floor/clamp convention as ``compute_subject_shifted_crop``: the crop
    size is the Stage 5.1 maximal legal target-ratio crop, the ideal placement uses
    the internal centering convention (floor(center - crop/2)) and is clamped into
    strict bounds with the vertical bound tightened by the exact derived height.
    Returns (x, y, crop_w, derived_h, clamped_x, clamped_y).
    """
    _require_frame(width, height)
    center_box: CenterCropBox = compute_center_crop(width, height, target_w, target_h)
    crop_w = center_box.w
    crop_h = stage5_1_crop_height(width, height, target_w, target_h)
    derived_h = derived_height(crop_w, target_w, target_h)
    cx, cy = float(center[0]), float(center[1])
    if not math.isfinite(cx) or not math.isfinite(cy):
        raise ValueError("center must be finite")
    ideal_x = math.floor(cx - crop_w / 2.0)
    ideal_y = math.floor(cy - crop_h / 2.0)
    max_x = width - crop_w
    max_y = int(math.floor(Fraction(height) - derived_h))
    if max_x < 0 or max_y < 0:
        raise ValueError("no target-ratio crop of the frozen size fits inside the frame")
    x = min(max(ideal_x, 0), max_x)
    y = min(max(ideal_y, 0), max_y)
    return int(x), int(y), int(crop_w), derived_h, x != ideal_x, y != ideal_y


def smooth_video_sequence(
    width: int,
    height: int,
    target_w: int | float,
    target_h: int | float,
    observations: Sequence[TemporalObservation],
    alpha: float = DEFAULT_EMA_ALPHA,
) -> list[SmoothedFrame]:
    """Deterministic TS-1 EMA smoothing over ONE video's frozen frame sequence.

    ``observations`` must be the frozen frame sequence of a single video in any
    order (it is sorted internally; duplicate frame ids are rejected). The first
    observation of the call starts a fresh run (NEW_VIDEO is the caller's
    responsibility: pass one video at a time). Reset reasons recorded per frame
    describe why the frame does not inherit smoothing from its predecessor:
    ``RESET_FALLBACK`` (this frame or the previous one is a frozen fallback) and
    ``RESET_FRAME_GAP`` (frozen frame identity discontinuity, gap > 1). Fallback
    frames always emit the frozen center crop and clear the EMA state.
    """
    ema_alpha = _require_alpha(alpha)
    return _smooth_video_sequence(
        width,
        height,
        target_w,
        target_h,
        observations,
        alpha_for_motion=lambda _motion: ema_alpha,
        placement_status=PLACEMENT_TS1_SMOOTHED,
        record_transition_parameters=False,
    )


def smooth_video_sequence_adaptive(
    width: int,
    height: int,
    target_w: int | float,
    target_h: int | float,
    observations: Sequence[TemporalObservation],
) -> list[SmoothedFrame]:
    """Motion-Adaptive EMA v1 with TS-1-identical state/reset semantics."""
    return _smooth_video_sequence(
        width,
        height,
        target_w,
        target_h,
        observations,
        alpha_for_motion=motion_adaptive_alpha,
        placement_status=PLACEMENT_TS2_ADAPTIVE_SMOOTHED,
        record_transition_parameters=True,
    )


def smooth_video_sequence_guarded(
    width: int,
    height: int,
    target_w: int | float,
    target_h: int | float,
    observations: Sequence[TemporalObservation],
    alpha: float = DEFAULT_EMA_ALPHA,
) -> list[SmoothedFrame]:
    """TS-3: fixed EMA proposal projected minimally to center containment.

    The unconstrained fixed-EMA state remains the TS-1 state and is never fed
    back from the projection. Only the current frame's final integer placement
    is projected into the legal rectangle that contains the current frozen
    primary-subject center.
    """
    fixed = smooth_video_sequence(width, height, target_w, target_h, observations, alpha=alpha)
    observations_by_frame = {item.frame: item for item in observations}
    guarded: list[SmoothedFrame] = []
    for proposal in fixed:
        observation = observations_by_frame[proposal.frame]
        if observation.fallback:
            guarded.append(proposal)
            continue
        safe_x_min, safe_x_max = containment_safe_top_left_interval(
            width, proposal.w, observation.ideal_center_x
        )
        safe_y_min, safe_y_max = containment_safe_top_left_interval(
            height, proposal.h, observation.ideal_center_y
        )
        x = min(max(proposal.x, safe_x_min), safe_x_max)
        y = min(max(proposal.y, safe_y_min), safe_y_max)
        ts0_x, ts0_y, _, _, _, _ = place_crop_from_center(
            width,
            height,
            target_w,
            target_h,
            (observation.ideal_center_x, observation.ideal_center_y),
        )
        guarded.append(
            replace(
                proposal,
                x=x,
                y=y,
                placement_status=PLACEMENT_TS3_GUARDED_SMOOTHED,
                matches_ts0_placement=(x, y) == (ts0_x, ts0_y),
                guard_applied=(x, y) != (proposal.x, proposal.y),
                guard_correction_x=x - proposal.x,
                guard_correction_y=y - proposal.y,
                safe_x_min=safe_x_min,
                safe_x_max=safe_x_max,
                safe_y_min=safe_y_min,
                safe_y_max=safe_y_max,
            )
        )
    return guarded


def smooth_video_sequence_bbox_guarded(
    width: int,
    height: int,
    target_w: int | float,
    target_h: int | float,
    observations: Sequence[TemporalObservation],
    primary_bboxes_by_frame: dict[int, tuple[float, float, float, float]],
    alpha: float = DEFAULT_EMA_ALPHA,
) -> list[SmoothedFrame]:
    """TS-4: maximize current bbox visibility, then minimize TS-1 correction.

    The fixed-alpha EMA state, reset rules, crop size, and frame clamp are
    exactly TS-1.  The bbox guard is output-only and is skipped on fallback
    frames.  Since 2-D intersection area is the product of independent positive
    one-axis overlaps, coordinatewise maximization is exactly equivalent to
    maximizing bbox visible fraction; clipping to each integer argmax interval
    then gives the nearest placement to the integer TS-1 proposal.
    """
    ema_alpha = _require_alpha(alpha)
    return _smooth_video_sequence(
        width,
        height,
        target_w,
        target_h,
        observations,
        alpha_for_motion=lambda _motion: ema_alpha,
        placement_status=PLACEMENT_TS1_SMOOTHED,
        record_transition_parameters=False,
        frame_projector=lambda proposal, observation: project_bbox_maximum_visibility(
            width,
            height,
            target_w,
            target_h,
            proposal,
            observation,
            primary_bboxes_by_frame,
            placement_status=PLACEMENT_TS4_BBOX_GUARDED_SMOOTHED,
        ),
    )


def smooth_video_sequence_projected_state_bbox_guarded(
    width: int,
    height: int,
    target_w: int | float,
    target_h: int | float,
    observations: Sequence[TemporalObservation],
    primary_bboxes_by_frame: dict[int, tuple[float, float, float, float]],
    alpha: float = DEFAULT_EMA_ALPHA,
) -> list[SmoothedFrame]:
    """TS-5: TS-4 projection with its correction fed into recursive state.

    The state remains in the continuous crop-center coordinate system.  The
    exact integer TS-4 correction ``(projected - proposal)`` is added to that
    state, so a zero correction leaves TS-1/TS-4 state semantics byte-for-byte
    unchanged and the projected placement is reproduced exactly next frame.
    """
    ema_alpha = _require_alpha(alpha)
    return _smooth_video_sequence(
        width,
        height,
        target_w,
        target_h,
        observations,
        alpha_for_motion=lambda _motion: ema_alpha,
        placement_status=PLACEMENT_TS1_SMOOTHED,
        record_transition_parameters=False,
        frame_projector=lambda proposal, observation: project_bbox_maximum_visibility(
            width,
            height,
            target_w,
            target_h,
            proposal,
            observation,
            primary_bboxes_by_frame,
            placement_status=PLACEMENT_TS5_PROJECTED_STATE_SMOOTHED,
        ),
        feedback_projected_state=True,
    )


def project_bbox_maximum_visibility(
    width: int,
    height: int,
    target_w: int | float,
    target_h: int | float,
    proposal: SmoothedFrame,
    observation: TemporalObservation,
    primary_bboxes_by_frame: dict[int, tuple[float, float, float, float]],
    *,
    placement_status: str,
) -> SmoothedFrame:
    """Canonical TS-4 ``P_t`` shared without alteration by TS-4 and TS-5."""
    if proposal.frame not in primary_bboxes_by_frame:
        raise ValueError(f"missing frozen primary bbox for non-fallback frame {proposal.frame}")
    bbox = tuple(float(value) for value in primary_bboxes_by_frame[proposal.frame])
    if len(bbox) != 4:
        raise ValueError("primary bbox must be an xyxy 4-tuple")
    x1, y1, x2, y2 = bbox
    safe_x = maximum_overlap_safe_top_left_interval(width, proposal.w, x1, x2)
    safe_y = maximum_overlap_safe_top_left_interval(height, proposal.h, y1, y2)
    x = min(max(proposal.x, safe_x.minimum), safe_x.maximum)
    y = min(max(proposal.y, safe_y.minimum), safe_y.maximum)

    def visible_fraction(px: int, py: int) -> float:
        bx1, by1, bx2, by2 = (Fraction(str(value)) for value in bbox)
        overlap_x = max(Fraction(0), min(Fraction(px) + proposal.w, bx2) - max(Fraction(px), bx1))
        overlap_y = max(Fraction(0), min(Fraction(py) + proposal.h, by2) - max(Fraction(py), by1))
        return float((overlap_x * overlap_y) / ((bx2 - bx1) * (by2 - by1)))

    before = visible_fraction(proposal.x, proposal.y)
    after = visible_fraction(x, y)
    ts0_x, ts0_y, _, _, _, _ = place_crop_from_center(
        width,
        height,
        target_w,
        target_h,
        (observation.ideal_center_x, observation.ideal_center_y),
    )
    return replace(
        proposal,
        x=x,
        y=y,
        placement_status=placement_status,
        matches_ts0_placement=(x, y) == (ts0_x, ts0_y),
        guard_applied=(x, y) != (proposal.x, proposal.y),
        guard_correction_x=x - proposal.x,
        guard_correction_y=y - proposal.y,
        safe_x_min=safe_x.minimum,
        safe_x_max=safe_x.maximum,
        safe_y_min=safe_y.minimum,
        safe_y_max=safe_y.maximum,
        guard_mode_x=safe_x.mode,
        guard_mode_y=safe_y.mode,
        bbox_fully_containable=safe_x.fully_containable and safe_y.fully_containable,
        bbox_larger_than_crop=(x2 - x1) > proposal.w or (y2 - y1) > float(proposal.h),
        per_axis_infeasible_x=not safe_x.fully_containable,
        per_axis_infeasible_y=not safe_y.fully_containable,
        visible_fraction_before_guard=round(before, 6),
        visible_fraction_after_guard=round(after, 6),
        visible_gain=round(after - before, 6),
    )


def _smooth_video_sequence(
    width: int,
    height: int,
    target_w: int | float,
    target_h: int | float,
    observations: Sequence[TemporalObservation],
    *,
    alpha_for_motion: Callable[[float], float],
    placement_status: str,
    record_transition_parameters: bool,
    frame_projector: Callable[[SmoothedFrame, TemporalObservation], SmoothedFrame] | None = None,
    feedback_projected_state: bool = False,
) -> list[SmoothedFrame]:
    """Shared TS-1/TS-2 EMA state machine; only transition alpha varies."""
    _require_frame(width, height)
    tw, th = float(target_w), float(target_h)
    center_box: CenterCropBox = compute_center_crop(width, height, tw, th)
    crop_w = center_box.w
    crop_h = stage5_1_crop_height(width, height, tw, th)

    ordered = sorted(observations, key=lambda item: item.frame)
    if len({item.frame for item in ordered}) != len(ordered):
        raise ValueError("duplicate frame ids in one video sequence")

    smoothed: list[SmoothedFrame] = []
    state_x: float | None = None
    state_y: float | None = None
    previous_frame: int | None = None
    previous_fallback = False
    previous_raw_center_x: float | None = None

    for observation in ordered:
        if observation.fallback:
            reset_reason = RESET_FALLBACK if previous_frame is not None and not previous_fallback else None
            derived_h = derived_height(crop_w, tw, th)
            smoothed.append(
                SmoothedFrame(
                    frame=observation.frame,
                    reset_reason=reset_reason,
                    ema_center_x=None,
                    ema_center_y=None,
                    x=center_box.x,
                    y=center_box.y,
                    w=crop_w,
                    h=derived_h,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    placement_status=PLACEMENT_FALLBACK_CENTER_CROP,
                    clamped_x=False,
                    clamped_y=False,
                    matches_ts0_placement=True,
                )
            )
            state_x = None
            state_y = None
            previous_frame = observation.frame
            previous_fallback = True
            previous_raw_center_x = None
            continue

        motion_norm: float | None = None
        alpha_t: float | None = None
        if previous_frame is None:
            reset_reason: str | None = None
            state_x = observation.ideal_center_x
            state_y = observation.ideal_center_y
        elif previous_fallback:
            reset_reason = RESET_FALLBACK
            state_x = observation.ideal_center_x
            state_y = observation.ideal_center_y
        elif observation.frame - previous_frame > MAX_FRAME_GAP:
            reset_reason = RESET_FRAME_GAP
            state_x = observation.ideal_center_x
            state_y = observation.ideal_center_y
        else:
            reset_reason = None
            if previous_raw_center_x is None:
                raise RuntimeError("missing previous raw center for a continuous EMA transition")
            motion_norm = displacement_norm(previous_raw_center_x, observation.ideal_center_x, width)
            alpha_t = _require_alpha(alpha_for_motion(motion_norm))
            state_x = alpha_t * observation.ideal_center_x + (1.0 - alpha_t) * state_x
            state_y = alpha_t * observation.ideal_center_y + (1.0 - alpha_t) * state_y

        x, y, _, derived_h, clamped_x, clamped_y = place_crop_from_center(
            width, height, tw, th, (state_x, state_y)
        )
        ts0_x, ts0_y, _, _, _, _ = place_crop_from_center(
            width, height, tw, th, (observation.ideal_center_x, observation.ideal_center_y)
        )
        proposal = SmoothedFrame(
                frame=observation.frame,
                reset_reason=reset_reason,
                ema_center_x=state_x,
                ema_center_y=state_y,
                x=x,
                y=y,
                w=crop_w,
                h=derived_h,
                crop_w=crop_w,
                crop_h=crop_h,
                placement_status=placement_status,
                clamped_x=clamped_x,
                clamped_y=clamped_y,
                matches_ts0_placement=(x, y) == (ts0_x, ts0_y),
                motion_norm=motion_norm if record_transition_parameters else None,
                alpha_t=alpha_t if record_transition_parameters else None,
            )
        emitted = frame_projector(proposal, observation) if frame_projector else proposal
        if feedback_projected_state:
            if emitted.guard_correction_x is None or emitted.guard_correction_y is None:
                raise RuntimeError("projected-state feedback requires an integer projection correction")
            proposal_center_x, proposal_center_y = state_x, state_y
            state_x += emitted.guard_correction_x
            state_y += emitted.guard_correction_y
            state_x_placement, state_y_placement, _, _, _, _ = place_crop_from_center(
                width, height, tw, th, (state_x, state_y)
            )
            residual = abs(state_x_placement - emitted.x) + abs(state_y_placement - emitted.y)
            if residual != 0:
                raise RuntimeError("projected EMA state does not reproduce projected output")
            emitted = replace(
                emitted,
                ema_center_x=state_x,
                ema_center_y=state_y,
                proposal_center_x=proposal_center_x,
                proposal_center_y=proposal_center_y,
                projected_state_center_x=state_x,
                projected_state_center_y=state_y,
                state_output_residual_l1=float(residual),
            )
        smoothed.append(emitted)
        previous_frame = observation.frame
        previous_fallback = False
        previous_raw_center_x = observation.ideal_center_x
    return smoothed


# ---------------------------------------------------------------------------
# Temporal metrics (descriptive; no promotion gate is derived from them in v1)
# ---------------------------------------------------------------------------

def transition_pairs(observations: Sequence[TemporalObservation]) -> list[tuple[int, int]]:
    """Index pairs (previous, current) usable for temporal displacement metrics.

    A pair is usable when both frames are in the same frozen sequence, the frame
    gap is exactly 1, and neither frame is a frozen fallback frame. Indices refer
    to the input ``observations`` order.
    """
    order = sorted(range(len(observations)), key=lambda index: observations[index].frame)
    pairs: list[tuple[int, int]] = []
    for previous, current in zip(order, order[1:]):
        if observations[previous].fallback or observations[current].fallback:
            continue
        if observations[current].frame - observations[previous].frame != MAX_FRAME_GAP:
            continue
        pairs.append((previous, current))
    return pairs


def transition_triplets(observations: Sequence[TemporalObservation]) -> list[tuple[int, int, int]]:
    """Index triples (t-2, t-1, t) usable for temporal acceleration metrics."""
    order = sorted(range(len(observations)), key=lambda index: observations[index].frame)
    triplets: list[tuple[int, int, int]] = []
    for first, second, third in zip(order, order[1:], order[2:]):
        if any(observations[index].fallback for index in (first, second, third)):
            continue
        if (
            observations[second].frame - observations[first].frame != MAX_FRAME_GAP
            or observations[third].frame - observations[second].frame != MAX_FRAME_GAP
        ):
            continue
        triplets.append((first, second, third))
    return triplets


def displacement_norm(center_x_previous: float, center_x_current: float, width: int) -> float:
    """Normalized one-step crop-center displacement |cx_t - cx_(t-1)| / W."""
    if width <= 0:
        raise ValueError("width must be positive")
    return abs(center_x_current - center_x_previous) / float(width)


def acceleration_norm(
    delta_previous: float, delta_current: float, width: int
) -> float:
    """Normalized second-order crop-center change |d_t - d_(t-1)| / W."""
    if width <= 0:
        raise ValueError("width must be positive")
    return abs(delta_current - delta_previous) / float(width)


def _round6(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def temporal_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """mean / median / P90 / P95 / max of a normalized displacement-like sample."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}

    def percentile(q: float) -> float:
        index = min(int(q * len(ordered)), len(ordered) - 1)
        return ordered[index]

    return {
        "n": len(ordered),
        "mean": _round6(statistics.fmean(ordered)),
        "median": _round6(statistics.median(ordered)),
        "p90": _round6(percentile(0.90)),
        "p95": _round6(percentile(0.95)),
        "max": _round6(ordered[-1]),
    }


def large_jump_ratios(
    values: Sequence[float], thresholds: Sequence[float] = LARGE_JUMP_THRESHOLDS
) -> dict[str, float]:
    """Share of transitions above each descriptive threshold (DESCRIPTIVE ONLY).

    These ratios are diagnostic observations of temporal jitter magnitude; in the
    Stage 5.4 v1 protocol they must not be turned into a promotion gate.
    """
    ordered = [float(value) for value in values]
    if not ordered:
        return {f">{threshold:.2f}": 0.0 for threshold in thresholds}
    return {
        f">{threshold:.2f}": round(
            sum(1 for value in ordered if value > threshold + 1e-9) / len(ordered), 6
        )
        for threshold in thresholds
    }
