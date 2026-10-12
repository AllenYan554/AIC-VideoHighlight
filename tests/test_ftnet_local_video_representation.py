from __future__ import annotations

import pytest
import torch

from aic_video_highlight.ftnet.local_video_representation import (
    LocalVideoConfig,
    LocalVideoFTNet,
)


def _masks(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = torch.ones(batch, length, dtype=torch.bool)
    adjacency = torch.ones(batch, length, dtype=torch.bool)
    adjacency[:, 0] = False
    return sequence, adjacency


def test_parameter_count_matches_v0_minus_unused_modules_plus_projection() -> None:
    model = LocalVideoFTNet(LocalVideoConfig())
    count = sum(parameter.numel() for parameter in model.parameters())
    # visual 384->96: 384*96+96 + LN 192 = 37_152
    # fusion 224->128: 28_800; TCN 5 x 17_536 = 87_680; y_head 129
    assert count == 153_761
    assert model.fusion[0].in_features == 224
    assert model.visual_branch[0].in_features == 384


def test_forward_shapes_and_zeroed_invalid_timesteps() -> None:
    model = LocalVideoFTNet(LocalVideoConfig()).eval()
    sequence = torch.tensor([[True, True, True, False, False]])
    adjacency = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        output = model(
            torch.randn(1, 5, 384),
            sequence_mask=sequence,
            adjacency_mask=adjacency,
        )
    assert output.logits.shape == (1, 5)
    assert output.features.shape == (1, 5, 128)
    assert torch.equal(output.logits[:, 3:], torch.zeros(1, 2))
    assert torch.equal(output.probabilities[:, 3:], torch.zeros(1, 2))


def test_wrong_input_dim_is_rejected() -> None:
    model = LocalVideoFTNet(LocalVideoConfig()).eval()
    sequence, adjacency = _masks(1, 4)
    with pytest.raises(ValueError):
        model(
            torch.randn(1, 4, 256),
            sequence_mask=sequence,
            adjacency_mask=adjacency,
        )


def test_dead_fusion_columns_are_zeros_and_receive_no_gradient() -> None:
    model = LocalVideoFTNet(LocalVideoConfig())
    captured = {}

    def hook(module, args, output):
        captured["fusion_input"] = args[0].detach()

    handle = model.fusion.register_forward_hook(hook)
    sequence, adjacency = _masks(2, 6)
    output = model(
        torch.randn(2, 6, 384),
        sequence_mask=sequence,
        adjacency_mask=adjacency,
    )
    handle.remove()
    fusion_input = captured["fusion_input"]
    assert fusion_input.shape == (2, 6, 224)
    assert torch.equal(fusion_input[..., 96:], torch.zeros(2, 6, 128))
    loss = output.logits.mean()
    loss.backward()
    gradient = model.fusion[0].weight.grad
    assert gradient is not None
    assert torch.equal(gradient[:, 96:], torch.zeros_like(gradient[:, 96:]))
