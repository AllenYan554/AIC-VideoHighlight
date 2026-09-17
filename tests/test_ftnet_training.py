from __future__ import annotations

import math

import torch

from aic_video_highlight.ftnet.losses import masked_binary_cross_entropy
from aic_video_highlight.ftnet.model import FTNet, FTNetConfig
from aic_video_highlight.ftnet.trainer import (
    FTNetExample,
    collate_ftnet_examples,
    train_step,
)


def _example(length: int, native_dim: int = 0) -> FTNetExample:
    return FTNetExample(
        visual_features=torch.randn(length, 256),
        native_features=(
            torch.randn(length, native_dim) if native_dim > 0 else None
        ),
        targets=torch.tensor([(index % 2) for index in range(length)]).float(),
        adjacency_mask=torch.tensor([False] + [True] * (length - 1)),
    )


def test_masked_bce_forward_excludes_padding() -> None:
    logits = torch.tensor([[0.0, 1.5, -9.0]])
    targets = torch.tensor([[0.0, 1.0, 1.0]])
    loss_mask = torch.tensor([[True, True, False]])

    loss = masked_binary_cross_entropy(logits, targets, loss_mask)
    changed_padding = masked_binary_cross_entropy(
        logits,
        torch.tensor([[0.0, 1.0, 0.0]]),
        loss_mask,
    )

    assert math.isfinite(loss.item())
    assert torch.equal(loss, changed_padding)


def test_optional_positive_class_weight_is_supported() -> None:
    logits = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0]])
    loss_mask = torch.tensor([[True, True]])

    unweighted = masked_binary_cross_entropy(logits, targets, loss_mask)
    weighted = masked_binary_cross_entropy(
        logits,
        targets,
        loss_mask,
        positive_class_weight=3.0,
    )

    assert weighted > unweighted


def test_dynamic_padding_builds_sequence_and_loss_masks() -> None:
    batch = collate_ftnet_examples([_example(2), _example(5)])

    assert batch.visual_features.shape == (2, 5, 256)
    assert torch.equal(
        batch.sequence_mask,
        torch.tensor([[True, True, False, False, False], [True] * 5]),
    )
    assert torch.equal(batch.loss_mask, batch.sequence_mask)
    assert not batch.adjacency_mask[0, 2:].any()


def test_train_step_has_finite_gradient_and_updates_parameters() -> None:
    torch.manual_seed(11)
    model = FTNet(FTNetConfig(native_dim=4, dropout=0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = collate_ftnet_examples([_example(4, 4), _example(7, 4)])

    result = train_step(model, batch, optimizer)

    assert math.isfinite(result.loss)
    assert result.gradients_finite
    assert result.parameters_updated
