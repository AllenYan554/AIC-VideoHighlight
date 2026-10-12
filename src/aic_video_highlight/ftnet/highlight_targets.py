"""Explicit MTurk target policies for YouTube Highlights."""

from __future__ import annotations

from enum import Enum
from typing import Any


class GroundTruthPolicyError(ValueError):
    """Raised when a prediction or non-authoritative signal is used as GT."""


class HighlightTargetStrategy(str, Enum):
    SOFT_VOTE_TARGET = "soft_vote_target"
    PAPER_CONSENSUS_BINARY_TARGET = "paper_consensus_binary_target"


class HighlightTargetAdapter:
    """Convert authoritative MTurk votes using an explicitly selected policy."""

    def __init__(self, strategy: HighlightTargetStrategy) -> None:
        if not isinstance(strategy, HighlightTargetStrategy):
            raise TypeError("strategy must be an explicit HighlightTargetStrategy")
        self.strategy = strategy

    def adapt(self, record: dict[str, Any]) -> tuple[float, ...]:
        source = str(record.get("source", "")).strip().lower()
        if source != "mturk_label":
            shown = source or "<missing>"
            raise GroundTruthPolicyError(
                f"Ground-truth source must be mturk_label; {shown} is forbidden"
            )

        raw_votes = record.get("vote_counts")
        annotator_count = record.get("annotator_count")
        if not isinstance(raw_votes, list) or not raw_votes:
            raise GroundTruthPolicyError("MTurk record requires non-empty vote_counts")
        if not isinstance(annotator_count, int) or annotator_count <= 0:
            raise GroundTruthPolicyError("MTurk record requires positive annotator_count")
        votes = tuple(float(value) for value in raw_votes)
        if any(value < 0.0 or value > annotator_count for value in votes):
            raise GroundTruthPolicyError("vote_counts must be within annotator_count")

        if self.strategy is HighlightTargetStrategy.SOFT_VOTE_TARGET:
            return tuple(value / annotator_count for value in votes)

        category = str(record.get("category", "")).strip().lower()
        threshold = 3.0 if category in {"parkour", "skiing"} else 2.0
        return tuple(1.0 if value > threshold else 0.0 for value in votes)
