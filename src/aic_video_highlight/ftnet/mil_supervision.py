"""Clip-level multiple-instance supervision for FTNet (Probe B).

The frozen YouTube Highlights adapter broadcasts clip-level MTurk votes onto
2 FPS sampled frames (see ``youtube_highlights.project_soft_vote_targets``).
This module implements the pre-registered alternative: one MTurk annotation
unit (window) is one bag; bag members are the sampled frames inside the window
that are also inside the supervised (``loss_mask``) universe; the bag label is
``vote / annotator_count`` and the loss aggregates the frame predictions of one
bag only.

Isolation guarantee: aggregation never crosses a bag boundary, a video or a
split.  Bags with zero members are dropped (no fabricated supervision).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .youtube_highlights import ANNOTATOR_COUNT, MturkClip

MIL_MODES = ("mean", "top_k", "logsumexp")
DEFAULT_TOP_K_FRACTION = 0.25
DEFAULT_LSE_TAU = 0.25
BCE_CLAMP = 1e-6


class MilSupervisionError(ValueError):
    """Raised when MIL bags or aggregation violate the frozen contract."""


@dataclass(frozen=True)
class MilBag:
    """One annotation unit (MTurk window) restricted to supervised frames."""

    video_id: str
    bag_index: int
    member_indices: torch.Tensor  # int64 [n], n >= 1, sorted, unique
    target: float  # vote / ANNOTATOR_COUNT in [0, 1]

    def __post_init__(self) -> None:
        if self.member_indices.ndim != 1 or self.member_indices.numel() == 0:
            raise MilSupervisionError("bag members must be a non-empty [n] tensor")
        if self.member_indices.dtype is not torch.int64:
            raise MilSupervisionError("bag member indices must be int64")
        if not math.isfinite(self.target) or not 0.0 <= self.target <= 1.0:
            raise MilSupervisionError(f"bag target must be in [0,1], got {self.target!r}")


def build_video_bags(
    video_id: str,
    source_frame_ids,
    loss_mask,
    clips: tuple[MturkClip, ...],
) -> tuple[MilBag, ...]:
    """Build one bag per MTurk window from the sampled-frame timeline.

    ``source_frame_ids`` and ``loss_mask`` describe the sampled frames of one
    video (the same arrays stored in the materialized v2 safetensors).
    """

    frame_ids = torch.as_tensor(source_frame_ids, dtype=torch.int64)
    mask = torch.as_tensor(loss_mask, dtype=torch.bool)
    if frame_ids.ndim != 1 or mask.shape != frame_ids.shape:
        raise MilSupervisionError("source_frame_ids and loss_mask must share shape [T]")
    bags: list[MilBag] = []
    for bag_index, clip in enumerate(clips):
        member = (
            (frame_ids >= clip.start_frame) & (frame_ids < clip.end_frame) & mask
        )
        indices = torch.nonzero(member, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            continue
        bags.append(
            MilBag(
                video_id=video_id,
                bag_index=bag_index,
                member_indices=indices.to(torch.int64),
                target=float(clip.vote) / float(ANNOTATOR_COUNT),
            )
        )
    return tuple(bags)


def aggregate_members(
    values: torch.Tensor,
    members: torch.Tensor,
    *,
    mode: str,
    top_k_fraction: float = DEFAULT_TOP_K_FRACTION,
    lse_tau: float = DEFAULT_LSE_TAU,
) -> torch.Tensor:
    """Aggregate one bag's member values into a single prediction.

    ``values`` is [T] (probabilities in [0, 1] for the pre-registered arms).
    """

    if mode not in MIL_MODES:
        raise MilSupervisionError(f"unknown MIL mode: {mode!r}")
    if values.ndim != 1:
        raise MilSupervisionError("values must be shape [T]")
    selected = values[members]
    if selected.numel() == 0:
        raise MilSupervisionError("a bag must contain at least one member")
    if mode == "mean":
        return selected.mean()
    if mode == "top_k":
        if not 0.0 < top_k_fraction <= 1.0:
            raise MilSupervisionError("top_k_fraction must be in (0, 1]")
        k = max(1, math.ceil(top_k_fraction * selected.numel()))
        return selected.topk(k, largest=True, sorted=False).values.mean()
    # log-sum-exp on probabilities, numerically stable via max shift
    if lse_tau <= 0.0:
        raise MilSupervisionError("lse_tau must be positive")
    shifted = selected / lse_tau
    max_value = shifted.max().detach()
    lse = lse_tau * (torch.log(torch.exp(shifted - max_value).mean()) + max_value)
    return lse


def bag_binary_cross_entropy(
    aggregated_probability: torch.Tensor,
    target: float,
) -> torch.Tensor:
    """BCE for one bag prediction against the soft MTurk bag label."""

    clamped = aggregated_probability.clamp(BCE_CLAMP, 1.0 - BCE_CLAMP)
    target_tensor = torch.as_tensor(
        target, dtype=clamped.dtype, device=clamped.device
    )
    return F.binary_cross_entropy(clamped, target_tensor)


def mil_bag_loss(
    probabilities: torch.Tensor,
    bags: tuple[MilBag, ...],
    *,
    mode: str,
    top_k_fraction: float = DEFAULT_TOP_K_FRACTION,
    lse_tau: float = DEFAULT_LSE_TAU,
) -> torch.Tensor:
    """Equal-weight mean BCE over the bags of one video.

    ``probabilities`` is the [T] frame prediction for this video.
    """

    if not bags:
        raise MilSupervisionError("at least one bag is required to compute a MIL loss")
    terms = []
    for bag in bags:
        aggregated = aggregate_members(
            probabilities,
            bag.member_indices.to(probabilities.device),
            mode=mode,
            top_k_fraction=top_k_fraction,
            lse_tau=lse_tau,
        )
        terms.append(bag_binary_cross_entropy(aggregated, bag.target))
    return torch.stack(terms).mean()


__all__ = [
    "BCE_CLAMP",
    "DEFAULT_LSE_TAU",
    "DEFAULT_TOP_K_FRACTION",
    "MIL_MODES",
    "MilBag",
    "MilSupervisionError",
    "aggregate_members",
    "bag_binary_cross_entropy",
    "build_video_bags",
    "mil_bag_loss",
]
