"""Mask-aware temporal blocks used by FTNet."""

from __future__ import annotations

import torch
from torch import nn


class DilatedDepthwiseSeparableBlock(nn.Module):
    """A non-causal residual depthwise-separable temporal convolution block."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.normalization = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        mask = sequence_mask.unsqueeze(-1)
        residual = features
        temporal = features.masked_fill(~mask, 0.0).transpose(1, 2)
        temporal = self.depthwise(temporal)
        temporal = self.pointwise(temporal).transpose(1, 2)
        temporal = self.normalization(temporal)
        temporal = self.activation(temporal)
        temporal = self.dropout(temporal)
        return (residual + temporal).masked_fill(~mask, 0.0)
