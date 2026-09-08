"""Subject-aware spatial localization: candidate schema, primary policy, fallbacks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from aic_video_highlight.spatial_composition.center_crop import compute_center_crop

STATUS_PRIMARY = "PRIMARY"
STATUS_CENTER_CROP_FALLBACK = "CENTER_CROP_FALLBACK"

REASON_NO_DETECTION = "NO_DETECTION"
REASON_LOW_CONFIDENCE = "LOW_CONFIDENCE"
REASON_MULTI_SUBJECT_AMBIGUOUS = "MULTI_SUBJECT_AMBIGUOUS"
REASON_BOX_TOO_SMALL = "BOX_TOO_SMALL"
REASON_BOX_TOO_LARGE = "BOX_TOO_LARGE"
REASON_FULL_FRAME_LIKE = "FULL_FRAME_LIKE"
REASON_INVALID_GEOMETRY = "INVALID_GEOMETRY"
REASON_MODEL_ERROR = "MODEL_ERROR"

FALLBACK_REASONS = frozenset(
    {
        REASON_NO_DETECTION,
        REASON_LOW_CONFIDENCE,
        REASON_BOX_TOO_SMALL,
        REASON_BOX_TOO_LARGE,
        REASON_FULL_FRAME_LIKE,
        REASON_INVALID_GEOMETRY,
        REASON_MODEL_ERROR,
    }
)


@dataclass(frozen=True, slots=True)
class SubjectCandidate:
    """Raw detector output for one frame; box is xyxy in source pixels."""

    box: tuple[float, float, float, float]
    score: float
    label_id: int
    label: str

    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True, slots=True)
class SubjectPolicyConfig:
    reliable_score: float = 0.5
    possible_score: float = 0.3
    ambiguity_score_gap: float = 0.05
    ambiguity_iou_threshold: float = 0.5
    min_area_fraction: float = 0.002
    full_frame_width_fraction: float = 0.98
    full_frame_height_fraction: float = 0.98
    full_frame_area_fraction: float = 0.995
    target_ratio: tuple[int | float, int | float] = (9, 16)

    def __post_init__(self) -> None:
        if not 0 <= self.possible_score <= self.reliable_score <= 1:
            raise ValueError("score thresholds must satisfy 0 <= possible <= reliable <= 1")
        if self.min_area_fraction <= 0 or self.min_area_fraction >= 1:
            raise ValueError("min_area_fraction must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class SubjectDecision:
    video_id: str
    frame: int
    image_width: int
    image_height: int
    candidates: tuple[SubjectCandidate, ...]
    invalid_candidate_count: int
    primary: SubjectCandidate | None
    status: str
    fallback_reasons: tuple[str, ...]
    ambiguous: bool
    ambiguous_candidate_count: int
    fallback_box: tuple[int, int, int] | None
    provenance: dict[str, str] = field(default_factory=dict)


def clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )


def candidate_is_valid(box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return all(math.isfinite(v) for v in box) and x2 > x1 and y2 > y1


def detection_iou(a: SubjectCandidate, b: SubjectCandidate) -> float:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def _ordered_pool(candidates: tuple[SubjectCandidate, ...]) -> tuple[tuple[int, SubjectCandidate], ...]:
    ranked = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate_is_valid(candidate.box)
    ]
    ranked.sort(key=lambda item: (-item[1].score, -item[1].area(), item[0]))
    return tuple(ranked)


def select_primary_subject(
    video_id: str,
    frame: int,
    image_width: int,
    image_height: int,
    candidates: tuple[SubjectCandidate, ...],
    config: SubjectPolicyConfig,
    provenance: dict[str, str] | None = None,
) -> SubjectDecision:
    """Fixed, explainable primary-subject policy v0 (frozen for Stage 5.2 smoke)."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    pool = _ordered_pool(candidates)
    invalid_count = len(candidates) - len(pool)

    possible = [(index, c) for index, c in pool if c.score >= config.possible_score]
    reliable = [(index, c) for index, c in pool if c.score >= config.reliable_score]

    primary: SubjectCandidate | None = None
    ambiguous = False
    ambiguous_count = 0
    fallback_box = None
    fallback_reasons: list[str] = []

    if not possible:
        fallback_reasons.append(REASON_NO_DETECTION)
        if invalid_count:
            fallback_reasons.append(REASON_INVALID_GEOMETRY)
    elif not reliable:
        fallback_reasons.append(REASON_LOW_CONFIDENCE)
        if invalid_count:
            fallback_reasons.append(REASON_INVALID_GEOMETRY)
    else:
        primary_index, primary = reliable[0]
        frame_area = float(image_width * image_height)
        if primary.area() < config.min_area_fraction * frame_area:
            fallback_reasons.append(REASON_BOX_TOO_SMALL)
        elif primary.area() >= config.full_frame_area_fraction * frame_area:
            fallback_reasons.append(REASON_BOX_TOO_LARGE)
        elif (
            primary.box[2] - primary.box[0] >= config.full_frame_width_fraction * image_width
            and primary.box[3] - primary.box[1] >= config.full_frame_height_fraction * image_height
        ):
            fallback_reasons.append(REASON_FULL_FRAME_LIKE)
        else:
            contenders = [
                (index, c)
                for index, c in reliable[1:]
                if primary.score - c.score <= config.ambiguity_score_gap
                and detection_iou(primary, c) < config.ambiguity_iou_threshold
            ]
            ambiguous_count = len(contenders)
            ambiguous = bool(contenders)

    if fallback_reasons:
        box = compute_center_crop(image_width, image_height, *config.target_ratio)
        fallback_box = (box.x, box.y, box.w)
        status = STATUS_CENTER_CROP_FALLBACK
        primary = None
    else:
        status = STATUS_PRIMARY

    return SubjectDecision(
        video_id=video_id,
        frame=frame,
        image_width=image_width,
        image_height=image_height,
        candidates=tuple(candidates),
        invalid_candidate_count=invalid_count,
        primary=primary,
        status=status,
        fallback_reasons=tuple(fallback_reasons),
        ambiguous=ambiguous,
        ambiguous_candidate_count=ambiguous_count,
        fallback_box=fallback_box,
        provenance=dict(provenance or {}),
    )


def decision_to_record(decision: SubjectDecision) -> dict[str, object]:
    primary_box = None
    if decision.primary is not None:
        x1, y1, x2, y2 = decision.primary.box
        primary_box = {
            "xyxy": [x1, y1, x2, y2],
            "xywh_int": [
                int(math.floor(x1)),
                int(math.floor(y1)),
                int(math.ceil(x2)) - int(math.floor(x1)),
            ],
            "score": decision.primary.score,
            "label_id": decision.primary.label_id,
            "label": decision.primary.label,
        }
    return {
        "video_id": decision.video_id,
        "frame": decision.frame,
        "image_width": decision.image_width,
        "image_height": decision.image_height,
        "candidate_count": len(decision.candidates),
        "invalid_candidate_count": decision.invalid_candidate_count,
        "candidates": [
            {
                "box_xyxy": list(c.box),
                "score": c.score,
                "label_id": c.label_id,
                "label": c.label,
                "valid_geometry": candidate_is_valid(c.box),
            }
            for c in decision.candidates
        ],
        "primary": primary_box,
        "status": decision.status,
        "fallback_reasons": list(decision.fallback_reasons),
        "ambiguous": decision.ambiguous,
        "ambiguous_candidate_count": decision.ambiguous_candidate_count,
        "fallback_box": list(decision.fallback_box) if decision.fallback_box is not None else None,
        "provenance": dict(decision.provenance),
    }
