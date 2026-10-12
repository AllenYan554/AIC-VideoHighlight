from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from aic_video_highlight.ftnet.global_context import (
    GlobalContextFTNet,
    build_initialized_pair,
    copy_shared_modules,
    derive_seed,
    initialize_relative_projection,
    masked_temporal_mean,
)
from aic_video_highlight.ftnet.model import FTNet, FTNetConfig

EXPECTED_BASELINE_PARAMS = 166945
EXPECTED_CANDIDATE_PARAMS = 144641


def _config() -> FTNetConfig:
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


def _masks(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = torch.ones(batch, length, dtype=torch.bool)
    adjacency = torch.ones(batch, length, dtype=torch.bool)
    adjacency[:, 0] = False
    return sequence, adjacency


class _ZeroBranch(nn.Module):
    """Reference helper: reproduces the frozen branch_mask=(1,0,0) zeros."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.new_zeros(features.shape[0], features.shape[1], self.output_dim)


def _masked_baseline(model: FTNet) -> FTNet:
    """Turn a plain FTNet into the frozen visual-only (1,0,0) forward reference."""

    model.delta_branch = _ZeroBranch(model.config.branch_dim)  # type: ignore[assignment]
    model.native_branch = _ZeroBranch(model.config.native_branch_dim)  # type: ignore[assignment]
    return model


def test_masked_temporal_mean_ignores_padding_values() -> None:
    torch.manual_seed(0)
    features = torch.randn(2, 5, 3)
    sequence = torch.tensor(
        [[True, True, True, False, False], [False, True, True, True, False]]
    )
    pooled = masked_temporal_mean(features, sequence)

    manual = torch.stack(
        [
            features[0, :3].mean(dim=0),
            features[1, 1:4].mean(dim=0),
        ]
    )
    assert torch.allclose(pooled, manual, atol=1e-6, rtol=1e-6)

    perturbed = features.clone()
    perturbed[0, 3:] = 1e4
    perturbed[1, 4] = -1e4
    perturbed_pooled = masked_temporal_mean(perturbed, sequence)
    assert torch.equal(pooled, perturbed_pooled)


def test_fully_invalid_batch_uses_safe_denominator() -> None:
    features = torch.randn(1, 4, 3)
    sequence = torch.zeros(1, 4, dtype=torch.bool)
    pooled = masked_temporal_mean(features, sequence)
    assert torch.equal(pooled, torch.zeros(1, 3))


def test_variable_length_batch_does_not_leak_padding() -> None:
    torch.manual_seed(1)
    config = _config()
    model = GlobalContextFTNet(config, context_mode="global_relative").eval()

    short = torch.randn(1, 6, 256)
    long = torch.randn(1, 11, 256)
    sequence_short, adjacency_short = _masks(1, 6)

    with torch.no_grad():
        alone = model(
            short,
            sequence_mask=sequence_short,
            adjacency_mask=adjacency_short,
        )

    visual = torch.zeros(2, 11, 256)
    visual[0, :6] = short[0]
    visual[1] = long[0]
    sequence = torch.zeros(2, 11, dtype=torch.bool)
    sequence[0, :6] = True
    sequence[1] = True
    adjacency = torch.ones(2, 11, dtype=torch.bool)
    adjacency[:, 0] = False
    with torch.no_grad():
        batched = model(visual, sequence_mask=sequence, adjacency_mask=adjacency)

    max_diff = float((batched.logits[0, :6] - alone.logits[0]).abs().max())
    assert max_diff <= 1e-6, max_diff

    pooled_short = masked_temporal_mean(
        model.visual_branch(short).masked_fill(~sequence_short.unsqueeze(-1), 0.0),
        sequence_short,
    )
    with torch.no_grad():
        local = model.visual_branch(visual).masked_fill(~sequence.unsqueeze(-1), 0.0)
    pooled_batched = masked_temporal_mean(local, sequence)
    assert torch.equal(pooled_short[0], pooled_batched[0])


def test_context_off_reproduces_masked_baseline_logits() -> None:
    torch.manual_seed(20260918)
    config = _config()
    baseline = FTNet(config).eval()
    candidate = GlobalContextFTNet(config, context_mode="off").eval()
    copy_shared_modules(baseline, candidate)
    _masked_baseline(baseline)

    visual = torch.randn(3, 9, 256)
    native = torch.randn(3, 9, 16)
    sequence, adjacency = _masks(3, 9)
    sequence[2, 7:] = False

    with torch.no_grad():
        reference = baseline(
            visual,
            native_features=native,
            sequence_mask=sequence,
            adjacency_mask=adjacency,
        )
        challenged = candidate(visual, sequence_mask=sequence, adjacency_mask=adjacency)

    assert torch.equal(reference.logits, challenged.logits)
    assert torch.equal(reference.probabilities, challenged.probabilities)


def test_paired_initialization_is_bitwise_equivalent_and_rng_safe() -> None:
    seed = 20260918
    config = _config()

    torch.manual_seed(seed)
    baseline_only = FTNet(config)
    state_expected = torch.get_rng_state().clone()

    torch.manual_seed(seed)
    baseline, candidate, evidence = build_initialized_pair(config, seed=seed)
    state_actual = torch.get_rng_state().clone()

    assert evidence["shared_all_zero_diff"] is True
    assert evidence["shared_max_abs_diff"] == 0.0
    assert evidence["rng_state_restored"] is True
    assert torch.equal(state_expected, state_actual)

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    shared_names = [
        name
        for name in candidate_state
        if name.startswith(("visual_branch.", "fusion.", "temporal_blocks.", "y_head."))
    ]
    assert shared_names
    assert all(torch.equal(baseline_state[name], candidate_state[name]) for name in shared_names)

    _, candidate_again, _ = build_initialized_pair(
        config, seed=seed, context_mode="global_relative"
    )
    assert torch.equal(
        candidate.relative_projection[0].weight, candidate_again.relative_projection[0].weight
    )


def test_relative_projection_subseed_is_derived_and_deterministic() -> None:
    first = derive_seed(20260918, "relative_projection")
    second = derive_seed(20260918, "relative_projection")
    other = derive_seed(20260919, "relative_projection")
    assert first == second
    assert first != other

    module_a = nn.Sequential(nn.Linear(96, 32), nn.LayerNorm(32), nn.GELU())
    module_b = nn.Sequential(nn.Linear(96, 32), nn.LayerNorm(32), nn.GELU())
    initialize_relative_projection(module_a, seed=first)
    initialize_relative_projection(module_b, seed=first)
    assert torch.equal(module_a[0].weight, module_b[0].weight)
    assert torch.equal(module_a[0].bias, module_b[0].bias)
    bound = 1.0 / math.sqrt(96)
    assert float(module_a[0].weight.detach().abs().max()) <= bound


def test_parameter_counts() -> None:
    config = _config()
    baseline = FTNet(config)
    candidate = GlobalContextFTNet(config, context_mode="global_relative")
    assert sum(p.numel() for p in baseline.parameters()) == EXPECTED_BASELINE_PARAMS
    assert sum(p.numel() for p in candidate.parameters()) == EXPECTED_CANDIDATE_PARAMS


def test_context_diagnostics_capture_shapes() -> None:
    torch.manual_seed(3)
    config = _config()
    model = GlobalContextFTNet(config, context_mode="global_relative").eval()
    model.capture_context = True
    visual = torch.randn(2, 7, 256)
    sequence, adjacency = _masks(2, 7)
    sequence[0, 5:] = False
    with torch.no_grad():
        model(visual, sequence_mask=sequence, adjacency_mask=adjacency)
    assert model.last_context_tensors is not None
    assert model.last_context_tensors["local"].shape == (2, 7, 96)
    assert model.last_context_tensors["global"].shape == (2, 7, 96)
    assert model.last_context_tensors["relative"].shape == (2, 7, 32)
    assert torch.equal(
        model.last_context_tensors["global"][0, 5:], torch.zeros(2, 96)
    )

    model.capture_context = False
    frozen_tensors = model.last_context_tensors
    with torch.no_grad():
        model(visual, sequence_mask=sequence, adjacency_mask=adjacency)
    assert model.last_context_tensors is frozen_tensors


def test_invalid_context_mode_rejected() -> None:
    with pytest.raises(ValueError):
        GlobalContextFTNet(_config(), context_mode="attention")


def test_fully_padded_sequence_forward_is_zero() -> None:
    torch.manual_seed(4)
    config = _config()
    model = GlobalContextFTNet(config, context_mode="global_relative").eval()
    visual = torch.randn(1, 5, 256)
    sequence = torch.zeros(1, 5, dtype=torch.bool)
    adjacency = torch.zeros(1, 5, dtype=torch.bool)
    with torch.no_grad():
        output = model(visual, sequence_mask=sequence, adjacency_mask=adjacency)
    assert torch.equal(output.logits, torch.zeros(1, 5))
    assert torch.equal(output.probabilities, torch.zeros(1, 5))
