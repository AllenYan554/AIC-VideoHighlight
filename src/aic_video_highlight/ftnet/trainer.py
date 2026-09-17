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
    optimizer.step()
    parameters_updated = any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, trainable)
    )
    return TrainStepResult(
        loss=float(loss.detach().cpu()),
        gradients_finite=gradients_finite,
        parameters_updated=parameters_updated,
    )
