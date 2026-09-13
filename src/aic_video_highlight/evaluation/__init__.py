"""Release evaluation utilities.

``official_like`` implements the published exact-video/exact-frame spatial-IoU
formula; it is always reported as a weak-reference metric (``NOT_OFFICIAL_SCORE``).
``contract`` re-checks the written ``predictions.jsonl`` against the official
competition contract independently from the inference pipeline.
"""

from .official_like import (
    OfficialLikeEvaluationError,
    evaluate_files,
    evaluate_official_like,
    spatial_iou,
)

__all__ = [
    "OfficialLikeEvaluationError",
    "evaluate_files",
    "evaluate_official_like",
    "spatial_iou",
]
