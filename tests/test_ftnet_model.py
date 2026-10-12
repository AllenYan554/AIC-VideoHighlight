from __future__ import annotations

import pytest
import torch

from aic_video_highlight.ftnet.model import (
    FTNet,
    FTNetConfig,
    compute_signed_temporal_difference,
)


def _masks(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = torch.ones(batch, length, dtype=torch.bool)
    adjacency = torch.ones(batch, length, dtype=torch.bool)
    adjacency[:, 0] = False
    return sequence, adjacency


def test_signed_temporal_difference_resets_at_candidate_boundary() -> None:
    features = torch.tensor([[[1.0, 2.0], [4.0, 8.0], [9.0, 7.0]]])
    adjacency = torch.tensor([[False, True, False]])

    delta = compute_signed_temporal_difference(features, adjacency)

    assert torch.equal(
        delta,
        torch.tensor([[[0.0, 0.0], [3.0, 6.0], [0.0, 0.0]]]),
    )


def test_visual_only_ablation_has_dynamic_fusion_width() -> None:
    config = FTNetConfig(use_delta=False, native_dim=0)
    model = FTNet(config).eval()
    sequence, adjacency = _masks(2, 7)

    output = model(
        torch.randn(2, 7, 256),
        sequence_mask=sequence,
        adjacency_mask=adjacency,
    )

    assert model.fusion[0].in_features == 96
    assert output.logits.shape == (2, 7)


def test_reference_visual_and_delta_branches_are_enabled_by_default() -> None:
    model = FTNet(FTNetConfig(native_dim=0)).eval()
    sequence, adjacency = _masks(1, 4)

    output = model(
        torch.randn(1, 4, 256),
        sequence_mask=sequence,
        adjacency_mask=adjacency,
    )

    assert model.fusion[0].in_features == 192
    assert output.features.shape == (1, 4, 128)


@pytest.mark.parametrize("native_dim", [3, 16, 41])
def test_native_dimension_is_configurable(native_dim: int) -> None:
    model = FTNet(FTNetConfig(native_dim=native_dim)).eval()
    sequence, adjacency = _masks(2, 5)

    output = model(
        torch.randn(2, 5, 256),
        native_features=torch.randn(2, 5, native_dim),
        sequence_mask=sequence,
        adjacency_mask=adjacency,
    )

    assert model.fusion[0].in_features == 224
    assert output.probabilities.shape == (2, 5)


def test_native_branch_is_absent_when_native_dim_is_zero() -> None:
    model = FTNet(FTNetConfig(native_dim=0))

    assert model.native_branch is None
    assert all("native_branch" not in name for name, _ in model.named_parameters())


def test_reference_model_has_no_q_head_parameters() -> None:
    model = FTNet(FTNetConfig(native_dim=8))

    assert not hasattr(model, "q_head")
    assert all("q_head" not in name.lower() for name, _ in model.named_parameters())


@pytest.mark.parametrize("length", [1, 31, 257])
def test_sequence_length_is_preserved(length: int) -> None:
    model = FTNet(FTNetConfig(native_dim=0)).eval()
    sequence, adjacency = _masks(1, length)

    output = model(
        torch.randn(1, length, 256),
        sequence_mask=sequence,
        adjacency_mask=adjacency,
    )

    assert output.logits.shape == (1, length)
    assert output.probabilities.shape == (1, length)
    assert output.features.shape == (1, length, 128)


def test_mixed_lengths_mask_padding_and_prevent_padding_contamination() -> None:
    torch.manual_seed(7)
    model = FTNet(FTNetConfig(native_dim=0, dropout=0.0)).eval()
    sequence = torch.tensor([[True] * 6, [True] * 3 + [False] * 3])
    adjacency = torch.tensor(
        [[False, True, True, True, True, True], [False, True, True, False, False, False]]
    )
    features = torch.randn(2, 6, 256)
    changed_padding = features.clone()
    changed_padding[1, 3:] = 10_000.0

    first = model(
        features,
        sequence_mask=sequence,
        adjacency_mask=adjacency,
    )
    second = model(
        changed_padding,
        sequence_mask=sequence,
        adjacency_mask=adjacency,
    )

    assert torch.allclose(first.logits[1, :3], second.logits[1, :3])
    assert torch.equal(first.logits[1, 3:], torch.zeros(3))
    assert torch.equal(first.probabilities[1, 3:], torch.zeros(3))
    assert torch.equal(first.features[1, 3:], torch.zeros(3, 128))


def test_non_finite_value_in_valid_region_is_rejected() -> None:
    model = FTNet(FTNetConfig(native_dim=0))
    features = torch.randn(1, 3, 256)
    features[0, 1, 5] = float("nan")
    sequence, adjacency = _masks(1, 3)

    with pytest.raises(ValueError, match="finite"):
        model(
            features,
            sequence_mask=sequence,
            adjacency_mask=adjacency,
        )


def test_non_finite_padding_is_safely_ignored() -> None:
    model = FTNet(FTNetConfig(native_dim=0)).eval()
    features = torch.randn(1, 3, 256)
    features[0, 2, 0] = float("inf")
    sequence = torch.tensor([[True, True, False]])
    adjacency = torch.tensor([[False, True, False]])

    output = model(
        features,
        sequence_mask=sequence,
        adjacency_mask=adjacency,
    )

    assert torch.isfinite(output.logits).all()
