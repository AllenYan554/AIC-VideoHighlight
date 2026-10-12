"""Traceable top-1/top-2 selection margin for the frozen primary-subject policy.

Adds an observable margin signal for the Stage 7 native schema.  It does not
change the frozen primary-subject selection algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

from .subject_localization import SubjectCandidate, candidate_is_valid


DETECTION_FLOOR = 0.0


class SelectionMarginError(ValueError):
    """Raised when margin inputs are invalid."""


@dataclass(frozen=True, slots=True)
class SelectionMargin:
    top1_score: float
    top2_score: float
    detection_floor: float
    margin: float
    candidate_count: int
    available: bool


def subject_selection_margin(
    candidates: tuple[SubjectCandidate, ...],
    *,
    detection_floor: float = DETECTION_FLOOR,
) -> SelectionMargin:
    valid = [candidate for candidate in candidates if candidate_is_valid(candidate.box)]
    valid.sort(key=lambda candidate: -candidate.score)
    if not valid:
        return SelectionMargin(
            top1_score=0.0,
            top2_score=0.0,
            detection_floor=float(detection_floor),
            margin=0.0,
            candidate_count=0,
            available=False,
        )
    top1 = float(valid[0].score)
    top2 = float(valid[1].score) if len(valid) > 1 else 0.0
    margin = top1 - max(top2, float(detection_floor))
    return SelectionMargin(
        top1_score=top1,
        top2_score=top2,
        detection_floor=float(detection_floor),
        margin=float(margin),
        candidate_count=len(valid),
        available=True,
    )
