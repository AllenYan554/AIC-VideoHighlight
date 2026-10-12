from __future__ import annotations

import torch

from aic_video_highlight.ftnet.model import FTNetConfig
from aic_video_highlight.ftnet.semantic_conditioning import SemanticConditionedFTNet


B0_PARAMS = 154369
A1_PARAMS = 151809


def _b0_config() -> FTNetConfig:
    return FTNetConfig(
        visual_dim=256,
        branch_dim=96,
        native_dim=16,
        native_branch_dim=32,
        temporal_channels=128,
        temporal_kernel_size=5,
        temporal_dilations=(1, 2, 4, 8, 16),
        dropout=0.1,
        use_delta=True,
    )


def _a1_config() -> FTNetConfig:
    base = _b0_config()
    return FTNetConfig(
        visual_dim=base.visual_dim,
        branch_dim=base.branch_dim,
        native_dim=base.native_dim,
        native_branch_dim=base.native_branch_dim,
        temporal_channels=base.temporal_channels,
        temporal_kernel_size=1,
        temporal_dilations=base.temporal_dilations,
        dropout=base.dropout,
        use_delta=base.use_delta,
    )


def _model(config: FTNetConfig) -> SemanticConditionedFTNet:
    return SemanticConditionedFTNet(config, semantic_mode="off", reason_vector_dim=384).eval()


def _masks(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = torch.ones(batch, length, dtype=torch.bool)
    adjacency = torch.ones(batch, length, dtype=torch.bool)
    adjacency[:, 0] = False
    return sequence, adjacency


def test_b0_parameter_count_matches_frozen_reference() -> None:
    model = _model(_b0_config())
    assert sum(parameter.numel() for parameter in model.parameters()) == B0_PARAMS


def test_a1_parameter_count_matches_preregistration() -> None:
    model = _model(_a1_config())
    assert sum(parameter.numel() for parameter in model.parameters()) == A1_PARAMS


def test_a1_is_frame_independent() -> None:
    model = _model(_a1_config())
    sequence, adjacency = _masks(2, 11)
    features = torch.randn(2, 11, 256)
    modified = features.clone()
    modified[:, 5, :] += 7.0

    with torch.no_grad():
        base = model(features, sequence_mask=sequence, adjacency_mask=adjacency)
        changed = model(modified, sequence_mask=sequence, adjacency_mask=adjacency)

    assert not torch.allclose(base.logits[:, 5], changed.logits[:, 5])
    others = [index for index in range(11) if index != 5]
    assert torch.equal(base.logits[:, others], changed.logits[:, others])


def test_a0_tcn_mixes_across_timesteps() -> None:
    model = _model(_b0_config())
    sequence, adjacency = _masks(1, 11)
    features = torch.randn(1, 11, 256)
    modified = features.clone()
    modified[:, 5, :] += 7.0

    with torch.no_grad():
        base = model(features, sequence_mask=sequence, adjacency_mask=adjacency)
        changed = model(modified, sequence_mask=sequence, adjacency_mask=adjacency)

    neighbors = torch.cat([base.logits[:, :5], base.logits[:, 6:]], dim=1)
    changed_neighbors = torch.cat([changed.logits[:, :5], changed.logits[:, 6:]], dim=1)
    assert not torch.allclose(neighbors, changed_neighbors)


def test_b0_zero_semantic_slots_are_exact_zeros() -> None:
    model = _model(_b0_config())
    model.capture_semantic_slots = True
    sequence, adjacency = _masks(1, 6)

    with torch.no_grad():
        model(
            torch.randn(1, 6, 256),
            sequence_mask=sequence,
            adjacency_mask=adjacency,
        )

    assert model.last_semantic_slots is not None
    assert torch.equal(
        model.last_semantic_slots["score"], torch.zeros(1, 6, 96)
    )
    assert torch.equal(
        model.last_semantic_slots["reason"], torch.zeros(1, 6, 32)
    )


def test_invalid_timesteps_are_zeroed_for_both_arms() -> None:
    sequence = torch.tensor([[True, True, True, False, False]])
    adjacency = torch.ones(1, 5, dtype=torch.bool)
    for config in (_b0_config(), _a1_config()):
        model = _model(config)
        with torch.no_grad():
            output = model(
                torch.randn(1, 5, 256),
                sequence_mask=sequence,
                adjacency_mask=adjacency,
            )
        assert torch.equal(output.logits[:, 3:], torch.zeros(1, 2))
        assert torch.equal(output.probabilities[:, 3:], torch.zeros(1, 2))
