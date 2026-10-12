"""LocalVideo (VideoMAE) representation arm for Stage 7.8.

``LocalVideoFTNet`` is V0 ("SemanticConditionedFTNet(semantic_mode='off')")
with exactly one scientific change: the visual representation entering the
network is the 384-d frozen VideoMAE LocalVideo feature instead of the 256-d
single-frame RT-DETR Visual256.  The first projection input width follows the
representation; everything else is bit-for-bit structurally identical to V0:

    [T, 384] -> Linear(384 -> 96) + LayerNorm(96) + GELU
             -> fusion(224 -> 128) + GELU          (same module as V0;
                the 96+32 non-visual slots are literal zeros, exactly as in V0)
             -> 5-block dilated depthwise-separable TCN (k=5, dilations 1..16)
             -> Y head (Linear(128 -> 1))

No delta, native, semantic, attention or gate paths exist.
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
class LocalVideoConfig:
    """Architecture-only configuration for the V1 LocalVideo arm."""

    input_dim: int = 384
    branch_dim: int = 96
    fusion_input_dim: int = 224
    temporal_channels: int = 128
    temporal_kernel_size: int = 5
    temporal_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    dropout: float = 0.1

    def __post_init__(self) -> None:
        positive = {
            "input_dim": self.input_dim,
            "branch_dim": self.branch_dim,
            "fusion_input_dim": self.fusion_input_dim,
            "temporal_channels": self.temporal_channels,
            "temporal_kernel_size": self.temporal_kernel_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.fusion_input_dim < self.branch_dim:
            raise ValueError("fusion_input_dim must be >= branch_dim")
        if not self.temporal_dilations or any(
            value <= 0 for value in self.temporal_dilations
        ):
            raise ValueError("temporal_dilations must contain positive integers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class LocalVideoFTNet(nn.Module):
    """V1: frozen VideoMAE LocalVideo384 -> V0 fusion/TCN/Y head."""

    def __init__(self, config: LocalVideoConfig | None = None) -> None:
        super().__init__()
        self.config = LocalVideoConfig() if config is None else config
        config = self.config
        self.visual_branch = _projection(config.input_dim, config.branch_dim)
        self.fusion = nn.Sequential(
            nn.Linear(config.fusion_input_dim, config.temporal_channels),
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
            raise ValueError("LocalVideoFTNet requires at least one timestep")
        if channels != self.config.input_dim:
            raise ValueError(
                f"visual_features last dimension must be {self.config.input_dim}"
            )
        _validate_bool_mask(sequence_mask, (batch, length), "sequence_mask")
        _validate_bool_mask(adjacency_mask, (batch, length), "adjacency_mask")
        _require_finite_in_valid_region(visual_features, sequence_mask, "visual_features")

        mask = sequence_mask.unsqueeze(-1)
        sanitized = visual_features.masked_fill(~mask, 0.0)
        visual_slot = self.visual_branch(sanitized).masked_fill(~mask, 0.0)
        dead_slots = torch.zeros(
            batch,
            length,
            self.config.fusion_input_dim - self.config.branch_dim,
            device=visual_features.device,
            dtype=visual_features.dtype,
        )
        concatenated = torch.cat([visual_slot, dead_slots], dim=-1)
        temporal = self.fusion(concatenated).masked_fill(~mask, 0.0)
        for block in self.temporal_blocks:
            temporal = block(temporal, sequence_mask)
        logits = self.y_head(temporal).squeeze(-1).masked_fill(~sequence_mask, 0.0)
        probabilities = torch.sigmoid(logits).masked_fill(~sequence_mask, 0.0)
        return FTNetOutput(logits=logits, probabilities=probabilities, features=temporal)


__all__ = ["LocalVideoConfig", "LocalVideoFTNet"]
