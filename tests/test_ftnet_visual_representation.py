from __future__ import annotations

import pytest
import torch

from aic_video_highlight.ftnet.visual_representation import (
    VisualRepresentationConfig,
    VisualRepresentationFTNet,
)


def _masks(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = torch.ones(batch, length, dtype=torch.bool)
    adjacency = torch.ones(batch, length, dtype=torch.bool)
    adjacency[:, 0] = False
    return sequence, adjacency


@pytest.mark.parametrize("input_dim", [512, 384, 256])
def test_forward_shapes_and_tcn_identity(input_dim: int) -> None:
    config = VisualRepresentationConfig(input_dim=input_dim)
    model = VisualRepresentationFTNet(config).eval()
    sequence, adjacency = _masks(2, 9)

    output = model(
        torch.randn(2, 9, input_dim),
        sequence_mask=sequence,
        adjacency_mask=adjacency,
    )

    assert output.logits.shape == (2, 9)
    assert output.features.shape == (2, 9, 128)
    assert len(model.temporal_blocks) == 5
    assert model.temporal_blocks[-1].depthwise.dilation == (16,)


def test_wrong_input_dim_is_rejected() -> None:
    model = VisualRepresentationFTNet(VisualRepresentationConfig(input_dim=512)).eval()
    sequence, adjacency = _masks(1, 4)

    with pytest.raises(ValueError):
        model(
            torch.randn(1, 4, 384),
            sequence_mask=sequence,
            adjacency_mask=adjacency,
        )


def test_invalid_timesteps_do_not_leak_into_valid_outputs() -> None:
    model = VisualRepresentationFTNet(VisualRepresentationConfig(input_dim=512)).eval()
    sequence = torch.tensor([[True, True, True, False, False]])
    adjacency = torch.ones(1, 5, dtype=torch.bool)
    first = torch.randn(1, 5, 512)
    second = first.clone()
    second[:, 3:, :] = 100.0

    with torch.no_grad():
        out_first = model(first, sequence_mask=sequence, adjacency_mask=adjacency)
        out_second = model(second, sequence_mask=sequence, adjacency_mask=adjacency)

    assert torch.allclose(out_first.logits[:, :3], out_second.logits[:, :3])
    assert torch.equal(out_first.logits[:, 3:], torch.zeros(1, 2))


def test_projection_stages_are_pointwise_across_time() -> None:
    model = VisualRepresentationFTNet(VisualRepresentationConfig(input_dim=384)).eval()
    features = torch.randn(1, 6, 384)
    modified = features.clone()
    modified[:, 2, :] += 5.0

    with torch.no_grad():
        base = model.temporal_projection(model.representation_projection(features))
        changed = model.temporal_projection(
            model.representation_projection(modified)
        )

    assert not torch.allclose(base[:, 2], changed[:, 2])
    keep = [0, 1, 3, 4, 5]
    assert torch.allclose(base[:, keep], changed[:, keep], atol=1e-6)


def test_tcn_stage_mixes_across_time_by_design() -> None:
    model = VisualRepresentationFTNet(VisualRepresentationConfig(input_dim=384)).eval()
    sequence, adjacency = _masks(1, 6)
    features = torch.randn(1, 6, 384)
    modified = features.clone()
    modified[:, 2, :] += 5.0

    with torch.no_grad():
        base = model(features, sequence_mask=sequence, adjacency_mask=adjacency)
        changed = model(modified, sequence_mask=sequence, adjacency_mask=adjacency)

    assert not torch.allclose(base.logits[:, 0], changed.logits[:, 0])


def test_parameter_count_is_reported_shape_stable() -> None:
    model = VisualRepresentationFTNet(VisualRepresentationConfig(input_dim=512))
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == 512 * 96 + 96 + 2 * 96 + 96 * 128 + 128 + 5 * (128 * 5 + 128 + 128 * 128 + 128 + 256) + 129
