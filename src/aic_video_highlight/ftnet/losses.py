"""Minimal, mask-safe losses for FTNet training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    positive_class_weight: float | None = None,
) -> torch.Tensor:
    """Compute BCEWithLogits over explicitly trainable timesteps only.

    ``positive_class_weight`` is an optional training configuration value.  It
    is never estimated from validation/test data by this function.
    """

    if logits.shape != targets.shape or logits.shape != loss_mask.shape:
        raise ValueError("logits, targets, and loss_mask must have identical shapes")
    if loss_mask.dtype is not torch.bool:
        raise TypeError("loss_mask must have dtype torch.bool")
    if not bool(loss_mask.any()):
        raise ValueError("loss_mask must select at least one timestep")
    valid_targets = targets[loss_mask]
    valid_logits = logits[loss_mask]
    if not bool(torch.isfinite(valid_logits).all()):
        raise ValueError("logits must be finite in the loss region")
    if not bool(torch.isfinite(valid_targets).all()):
        raise ValueError("targets must be finite in the loss region")
    if not bool(((valid_targets >= 0.0) & (valid_targets <= 1.0)).all()):
        raise ValueError("targets must be in [0, 1]")

    pos_weight = None
    if positive_class_weight is not None:
        if positive_class_weight <= 0.0:
            raise ValueError("positive_class_weight must be positive")
        pos_weight = torch.as_tensor(
            positive_class_weight,
            dtype=valid_logits.dtype,
            device=valid_logits.device,
        )
    return F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets.to(dtype=valid_logits.dtype),
        pos_weight=pos_weight,
    )
