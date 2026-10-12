"""Small training-ready primitives for FTNet smoke runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .losses import masked_binary_cross_entropy
from .model import FTNet


@dataclass(frozen=True)
class FTNetExample:
    visual_features: torch.Tensor
    targets: torch.Tensor
    adjacency_mask: torch.Tensor
    native_features: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class FTNetBatch:
    visual_features: torch.Tensor
    targets: torch.Tensor
    sequence_mask: torch.Tensor
    adjacency_mask: torch.Tensor
    loss_mask: torch.Tensor
    native_features: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "FTNetBatch":
        return FTNetBatch(
            visual_features=self.visual_features.to(device),
            targets=self.targets.to(device),
            sequence_mask=self.sequence_mask.to(device),
            adjacency_mask=self.adjacency_mask.to(device),
            loss_mask=self.loss_mask.to(device),
            native_features=(
                self.native_features.to(device)
                if self.native_features is not None
                else None
            ),
        )


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    gradients_finite: bool
    parameters_updated: bool
    gradient_norm: float | None


def build_adamw_optimizer(
    model: FTNet,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> torch.optim.AdamW:
    """Build AdamW with decay only on non-bias, non-LayerNorm weights."""

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for module in model.modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if parameter_name == "bias" or isinstance(module, torch.nn.LayerNorm):
                no_decay.append(parameter)
            else:
                decay.append(parameter)

    classified = {id(parameter) for parameter in decay + no_decay}
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if classified != trainable:
        raise RuntimeError("AdamW parameter grouping did not classify every trainable parameter")

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=0.0,
    )


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    max_epochs: int,
    eta_min: float,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Build the epoch-stepped cosine schedule used by the frozen baseline."""

    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if eta_min < 0.0:
        raise ValueError("eta_min cannot be negative")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=eta_min,
    )


def _validate_example(example: FTNetExample) -> int:
    if example.visual_features.ndim != 2:
        raise ValueError("example visual_features must have shape [time, channels]")
    length = example.visual_features.shape[0]
    if length == 0:
        raise ValueError("examples must contain at least one timestep")
    if example.targets.shape != (length,):
        raise ValueError("example targets must have shape [time]")
    if example.adjacency_mask.shape != (length,):
        raise ValueError("example adjacency_mask must have shape [time]")
    if example.adjacency_mask.dtype is not torch.bool:
        raise TypeError("example adjacency_mask must have dtype torch.bool")
    if example.loss_mask is not None:
        if example.loss_mask.shape != (length,):
            raise ValueError("example loss_mask must have shape [time]")
        if example.loss_mask.dtype is not torch.bool:
            raise TypeError("example loss_mask must have dtype torch.bool")
    if example.native_features is not None:
        if example.native_features.ndim != 2 or example.native_features.shape[0] != length:
            raise ValueError("example native_features must have shape [time, native_dim]")
    return length


def collate_ftnet_examples(examples: list[FTNetExample]) -> FTNetBatch:
    """Dynamically pad variable-length examples and construct explicit masks."""

    if not examples:
        raise ValueError("at least one example is required")
    lengths = [_validate_example(example) for example in examples]
    batch_size = len(examples)
    max_length = max(lengths)
    visual_dim = examples[0].visual_features.shape[1]
    if any(example.visual_features.shape[1] != visual_dim for example in examples):
        raise ValueError("all examples must use the same visual feature dimension")

    native_dims = {
        example.native_features.shape[1]
        for example in examples
        if example.native_features is not None
    }
    if native_dims and (
        len(native_dims) != 1
        or any(example.native_features is None for example in examples)
    ):
        raise ValueError("native features must be present with one shared dimension")

    prototype = examples[0].visual_features
    visual = prototype.new_zeros((batch_size, max_length, visual_dim))
    targets = examples[0].targets.new_zeros((batch_size, max_length))
    sequence = torch.zeros(
        batch_size, max_length, dtype=torch.bool, device=prototype.device
    )
    adjacency = torch.zeros_like(sequence)
    loss = torch.zeros_like(sequence)
    native = None
    if native_dims:
        native_dim = next(iter(native_dims))
        native_prototype = examples[0].native_features
        assert native_prototype is not None
        native = native_prototype.new_zeros((batch_size, max_length, native_dim))

    for index, (example, length) in enumerate(zip(examples, lengths)):
        visual[index, :length] = example.visual_features
        targets[index, :length] = example.targets
        sequence[index, :length] = True
        adjacency[index, :length] = example.adjacency_mask
        adjacency[index, 0] = False
        if example.loss_mask is None:
            loss[index, :length] = True
        else:
            loss[index, :length] = example.loss_mask
        if native is not None:
            assert example.native_features is not None
            native[index, :length] = example.native_features

    return FTNetBatch(
        visual_features=visual,
        native_features=native,
        targets=targets,
        sequence_mask=sequence,
        adjacency_mask=adjacency,
        loss_mask=loss & sequence,
    )


def train_step(
    model: FTNet,
    batch: FTNetBatch,
    optimizer: torch.optim.Optimizer,
    *,
    positive_class_weight: float | None = None,
    max_grad_norm: float | None = None,
) -> TrainStepResult:
    """Run exactly one forward/backward/update step for smoke verification."""

    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer.zero_grad(set_to_none=True)
    output = model(
        batch.visual_features,
        native_features=batch.native_features,
        sequence_mask=batch.sequence_mask,
        adjacency_mask=batch.adjacency_mask,
    )
    loss = masked_binary_cross_entropy(
        output.logits,
        batch.targets,
        batch.loss_mask,
        positive_class_weight=positive_class_weight,
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("training loss is not finite")
    loss.backward()
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    if not gradients_finite:
        raise FloatingPointError("training gradients are missing or non-finite")
    gradient_norm = None
    if max_grad_norm is not None:
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive when clipping is enabled")
        total_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=max_grad_norm,
            norm_type=2.0,
            error_if_nonfinite=True,
        )
        gradient_norm = float(total_norm.detach().cpu())
    optimizer.step()
    parameters_updated = any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, trainable)
    )
    return TrainStepResult(
        loss=float(loss.detach().cpu()),
        gradients_finite=gradients_finite,
        parameters_updated=parameters_updated,
        gradient_norm=gradient_norm,
    )
