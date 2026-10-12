"""Reference fine-grained temporal highlight selection network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .temporal_blocks import DilatedDepthwiseSeparableBlock


@dataclass(frozen=True)
class FTNetConfig:
    """Architecture-only configuration for the Stage 7.1 reference model."""

    visual_dim: int = 256
    branch_dim: int = 96
    native_dim: int = 0
    native_branch_dim: int = 32
    temporal_channels: int = 128
    temporal_kernel_size: int = 5
    temporal_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    dropout: float = 0.1
    use_delta: bool = True

    def __post_init__(self) -> None:
        positive = {
            "visual_dim": self.visual_dim,
            "branch_dim": self.branch_dim,
            "native_branch_dim": self.native_branch_dim,
            "temporal_channels": self.temporal_channels,
            "temporal_kernel_size": self.temporal_kernel_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.native_dim < 0:
            raise ValueError("native_dim cannot be negative")
        if not self.temporal_dilations or any(value <= 0 for value in self.temporal_dilations):
            raise ValueError("temporal_dilations must contain positive integers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class FTNetOutput:
    """Mask-safe FTNet outputs; logits remain available for stable BCE."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    features: torch.Tensor


def _validate_bool_mask(mask: torch.Tensor, expected: tuple[int, int], name: str) -> None:
    if mask.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(mask.shape)}")
    if mask.dtype is not torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")


def _require_finite_in_valid_region(
    tensor: torch.Tensor,
    sequence_mask: torch.Tensor,
    name: str,
) -> None:
    if not bool(torch.isfinite(tensor[sequence_mask]).all()):
        raise ValueError(f"{name} must be finite in every valid timestep")


def compute_signed_temporal_difference(
    visual_features: torch.Tensor,
    adjacency_mask: torch.Tensor,
    *,
    sequence_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return signed adjacent differences, resetting discontinuities to zero."""

    if visual_features.ndim != 3:
        raise ValueError("visual_features must have shape [batch, time, channels]")
    batch, length, _ = visual_features.shape
    _validate_bool_mask(adjacency_mask, (batch, length), "adjacency_mask")
    if sequence_mask is None:
        sequence_mask = torch.ones(
            batch, length, dtype=torch.bool, device=visual_features.device
        )
    _validate_bool_mask(sequence_mask, (batch, length), "sequence_mask")

    difference = torch.zeros_like(visual_features)
    if length > 1:
        difference[:, 1:] = visual_features[:, 1:] - visual_features[:, :-1]
    effective_adjacency = adjacency_mask & sequence_mask
    if length:
        effective_adjacency = effective_adjacency.clone()
        effective_adjacency[:, 0] = False
    if length > 1:
        effective_adjacency[:, 1:] &= sequence_mask[:, :-1]
    return difference.masked_fill(~effective_adjacency.unsqueeze(-1), 0.0)


def _projection(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.GELU(),
    )


class FTNet(nn.Module):
    """Predict ``p_t`` for retaining each frame inside candidate intervals.

    No Q head is present.  The reference configuration uses both visual and
    signed-difference branches; ``use_delta=False`` exists only for a clean
    visual-only ablation.
    """

    def __init__(self, config: FTNetConfig | None = None) -> None:
        super().__init__()
        self.config = FTNetConfig() if config is None else config
        config = self.config

        self.visual_branch = _projection(config.visual_dim, config.branch_dim)
        self.delta_branch = (
            _projection(config.visual_dim, config.branch_dim)
            if config.use_delta
            else None
        )
        self.native_branch = (
            _projection(config.native_dim, config.native_branch_dim)
            if config.native_dim > 0
            else None
        )

        fusion_dim = config.branch_dim
        if self.delta_branch is not None:
            fusion_dim += config.branch_dim
        if self.native_branch is not None:
            fusion_dim += config.native_branch_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.temporal_channels),
            nn.GELU(),
        )
        self.temporal_blocks = nn.ModuleList(
            DilatedDepthwiseSeparableBlock(
                config.temporal_channels,
                kernel_size=config.temporal_kernel_size,
                dilation=dilation,
                dropout=config.dropout,
            )
            for dilation in config.temporal_dilations
        )
        self.y_head = nn.Linear(config.temporal_channels, 1)

    def forward(
        self,
        visual_features: torch.Tensor,
        *,
        sequence_mask: torch.Tensor,
        adjacency_mask: torch.Tensor,
        native_features: torch.Tensor | None = None,
    ) -> FTNetOutput:
        if visual_features.ndim != 3:
            raise ValueError("visual_features must have shape [batch, time, channels]")
        batch, length, channels = visual_features.shape
        if length == 0:
            raise ValueError("FTNet requires at least one timestep")
        if channels != self.config.visual_dim:
            raise ValueError(
                f"visual_features last dimension must be {self.config.visual_dim}"
            )
        _validate_bool_mask(sequence_mask, (batch, length), "sequence_mask")
        _validate_bool_mask(adjacency_mask, (batch, length), "adjacency_mask")
        _require_finite_in_valid_region(visual_features, sequence_mask, "visual_features")

        mask = sequence_mask.unsqueeze(-1)
        sanitized_visual = visual_features.masked_fill(~mask, 0.0)
        branches = [self.visual_branch(sanitized_visual).masked_fill(~mask, 0.0)]

        if self.delta_branch is not None:
            difference = compute_signed_temporal_difference(
                sanitized_visual,
                adjacency_mask,
                sequence_mask=sequence_mask,
            )
            branches.append(self.delta_branch(difference).masked_fill(~mask, 0.0))

        if self.native_branch is None:
            if native_features is not None:
                raise ValueError("native_features were provided while native_dim=0")
        else:
            if native_features is None:
                raise ValueError(
                    f"native_features are required when native_dim={self.config.native_dim}"
                )
            expected = (batch, length, self.config.native_dim)
            if native_features.shape != expected:
                raise ValueError(
                    f"native_features must have shape {expected}, got {tuple(native_features.shape)}"
                )
            _require_finite_in_valid_region(native_features, sequence_mask, "native_features")
            sanitized_native = native_features.masked_fill(~mask, 0.0)
            branches.append(
                self.native_branch(sanitized_native).masked_fill(~mask, 0.0)
            )

        temporal = self.fusion(torch.cat(branches, dim=-1)).masked_fill(~mask, 0.0)
        for block in self.temporal_blocks:
            temporal = block(temporal, sequence_mask)

        logits = self.y_head(temporal).squeeze(-1)
        logits = logits.masked_fill(~sequence_mask, 0.0)
        probabilities = torch.sigmoid(logits).masked_fill(~sequence_mask, 0.0)
        return FTNetOutput(
            logits=logits,
            probabilities=probabilities,
            features=temporal,
        )
