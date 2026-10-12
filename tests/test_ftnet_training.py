from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

from aic_video_highlight.ftnet.losses import masked_binary_cross_entropy
from aic_video_highlight.ftnet.model import FTNet, FTNetConfig
from aic_video_highlight.ftnet.trainer import (
    FTNetExample,
    build_adamw_optimizer,
    build_cosine_scheduler,
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


def test_adamw_parameter_groups_exclude_bias_and_layer_norm_from_decay() -> None:
    model = FTNet(FTNetConfig(native_dim=0))

    optimizer = build_adamw_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    groups = {group["weight_decay"]: group for group in optimizer.param_groups}
    decay_ids = {id(parameter) for parameter in groups[1e-2]["params"]}
    no_decay_ids = {id(parameter) for parameter in groups[0.0]["params"]}
    assert id(model.y_head.weight) in decay_ids
    assert id(model.y_head.bias) in no_decay_ids
    assert id(model.visual_branch[1].weight) in no_decay_ids
    assert id(model.visual_branch[1].bias) in no_decay_ids
    assert optimizer.defaults["lr"] == 3e-4
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8


def test_cosine_scheduler_matches_frozen_epoch_protocol() -> None:
    model = FTNet(FTNetConfig(native_dim=0))
    optimizer = build_adamw_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = build_cosine_scheduler(
        optimizer,
        max_epochs=40,
        eta_min=1e-6,
    )

    assert scheduler.T_max == 40
    assert scheduler.eta_min == 1e-6


def test_train_step_can_apply_global_norm_clipping() -> None:
    torch.manual_seed(23)
    model = FTNet(FTNetConfig(native_dim=0, dropout=0.0))
    optimizer = build_adamw_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    batch = collate_ftnet_examples([_example(5), _example(8)])

    result = train_step(model, batch, optimizer, max_grad_norm=0.01)
    clipped_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                parameter.grad.detach().norm(2)
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
        ),
        2,
    )

    assert result.gradient_norm is not None
    assert result.gradient_norm > 0.01
    assert clipped_norm <= 0.01001


def test_reference_config_contains_complete_frozen_training_protocol() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "models"
        / "ftnet_reference.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = config["training"]

    assert training["protocol_status"] == "FROZEN_BASELINE"
    assert training["seed"] == 20260917
    assert training["batch_size_videos"] == 4
    assert training["gradient_accumulation_steps"] == 1
    assert training["max_epochs"] == 40
    assert training["optimizer"] == {
        "name": "AdamW",
        "learning_rate": 3e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "no_weight_decay": ["bias", "layer_norm_weight", "layer_norm_bias"],
    }
    assert training["scheduler"] == {
        "name": "CosineAnnealingLR",
        "step_unit": "epoch",
        "warmup_epochs": 0,
        "t_max_epochs": 40,
        "eta_min": 1e-6,
    }
    assert training["gradient_clipping"] == {
        "enabled": True,
        "type": "global_norm",
        "max_norm": 1.0,
        "norm_type": 2.0,
        "error_if_nonfinite": True,
    }
