"""Frozen-encoder visual representation probe model (Probe C).

``VisualRepresentationFTNet`` mirrors the frozen B0 visual pathway:

    [T, D] -> Linear(D -> 96) + LayerNorm + GELU
           -> Linear(96 -> 128) + GELU
           -> the SAME 5-block dilated TCN (kernel 5, dilations 1..16)
           -> Y head

The only structural difference from B0 is the input representation and its
first projection width: B0 uses Visual256 -> 96 through a 224-wide fusion whose
two extra slots are literal zeros; this class consumes a [T, D] representation
directly.  The TCN, dropout and Y head are unchanged (no per-encoder TCN).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .model import (
    FTNetOutput,
    _projection,
    _require_finite_in_valid_region,
    _validate_bool_mask,
)
from .temporal_blocks import DilatedDepthwiseSeparableBlock


@dataclass(frozen=True)
class VisualRepresentationConfig:
    """Architecture-only configuration for a frozen-representation probe."""

    input_dim: int
    projector_dim: int = 96
    temporal_channels: int = 128
    temporal_kernel_size: int = 5
    temporal_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    dropout: float = 0.1

    def __post_init__(self) -> None:
        positive = {
            "input_dim": self.input_dim,
            "projector_dim": self.projector_dim,
            "temporal_channels": self.temporal_channels,
            "temporal_kernel_size": self.temporal_kernel_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.temporal_dilations or any(
            value <= 0 for value in self.temporal_dilations
        ):
            raise ValueError("temporal_dilations must contain positive integers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class VisualRepresentationFTNet(nn.Module):
    """Frame representation -> shared TCN -> Y head (Probe C trained head only)."""

    def __init__(self, config: VisualRepresentationConfig) -> None:
        super().__init__()
        self.config = config
        self.representation_projection = _projection(
            config.input_dim, config.projector_dim
        )
        self.temporal_projection = nn.Sequential(
            nn.Linear(config.projector_dim, config.temporal_channels),
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
    ) -> FTNetOutput:
        if visual_features.ndim != 3:
            raise ValueError("visual_features must have shape [batch, time, channels]")
        batch, length, channels = visual_features.shape
        if length == 0:
            raise ValueError("VisualRepresentationFTNet requires at least one timestep")
        if channels != self.config.input_dim:
            raise ValueError(
                f"visual_features last dimension must be {self.config.input_dim}"
            )
        _validate_bool_mask(sequence_mask, (batch, length), "sequence_mask")
        _validate_bool_mask(adjacency_mask, (batch, length), "adjacency_mask")
        _require_finite_in_valid_region(visual_features, sequence_mask, "visual_features")

        mask = sequence_mask.unsqueeze(-1)
        sanitized = visual_features.masked_fill(~mask, 0.0)
        projected = self.representation_projection(sanitized).masked_fill(~mask, 0.0)
        temporal = self.temporal_projection(projected).masked_fill(~mask, 0.0)
        for block in self.temporal_blocks:
            temporal = block(temporal, sequence_mask)
        logits = self.y_head(temporal).squeeze(-1).masked_fill(~sequence_mask, 0.0)
        probabilities = torch.sigmoid(logits).masked_fill(~sequence_mask, 0.0)
        return FTNetOutput(logits=logits, probabilities=probabilities, features=temporal)


__all__ = ["VisualRepresentationConfig", "VisualRepresentationFTNet"]
