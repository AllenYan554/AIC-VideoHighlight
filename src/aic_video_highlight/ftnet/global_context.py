"""Global-context visual modeling for FTNet (research source, not stage-named).

``GlobalContextFTNet`` replaces the frozen FTNet's zero-masked Delta/Native slots
with an explicit sequence-level context representation while keeping the fusion
input width at 224 and every downstream module (Fusion 224->128, TCN, Y head)
unchanged:

    local_t    = h_t                      # visual projection output, 96-d
    g          = masked temporal mean of h over valid timesteps, 96-d
    global_t   = broadcast(g), 96-d
    relative_t = RelativeProjection(h_t - g), 32-d
    z_t        = concat(local_t, global_t, relative_t), 224-d

Pooling contract: only timesteps valid under ``sequence_mask`` contribute to
the mean; padded timesteps never leak into the global statistic.  With
``context_mode="off"`` the global/relative slots are exact zeros, which
reproduces the frozen visual-only (branch_mask=(1,0,0)) forward exactly when
the shared modules carry the same weights.

Paired-initialization contract: ``build_initialized_pair`` constructs the
baseline FTNet first (consuming the torch global RNG exactly like the frozen
baseline trainer), snapshots the RNG state, constructs the candidate, copies
every shared module tensor, initializes the Relative Projection from an
independent derived sub-seed, and restores the RNG state.  Result: for the same
seed, shared modules are bitwise identical to the baseline and the candidate
run sees the same RNG stream as the baseline run.
"""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn

from .model import (
    FTNet,
    FTNetConfig,
    FTNetOutput,
    _projection,
    _require_finite_in_valid_region,
    _validate_bool_mask,
)

CONTEXT_MODES = ("off", "global_relative")
SHARED_MODULE_PREFIXES = ("visual_branch.", "fusion.", "temporal_blocks.", "y_head.")
RELATIVE_PROJECTION_SEED_TAG = "relative_projection"


def derive_seed(global_seed: int, tag: str) -> int:
    """Deterministic sub-seed derived from the global seed and a stable tag."""

    digest = hashlib.sha256(f"{int(global_seed)}:{tag}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def masked_temporal_mean(features: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
    """Masked temporal global mean pooling over valid timesteps only.

    ``g_b = sum_t m_bt h_bt / max(sum_t m_bt, 1)`` with ``m = sequence_mask``.
    Padded (invalid) timesteps contribute nothing to numerator or denominator.
    """

    if features.ndim != 3:
        raise ValueError("features must have shape [batch, time, channels]")
    if sequence_mask.shape != features.shape[:2] or sequence_mask.dtype is not torch.bool:
        raise ValueError("sequence_mask must be a bool tensor with shape [batch, time]")
    mask = sequence_mask.unsqueeze(-1).to(dtype=features.dtype)
    summed = (features * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def initialize_relative_projection(module: nn.Sequential, *, seed: int) -> None:
    """Initialize RelativeProjection's Linear from an independent sub-seed.

    Replicates the default ``nn.Linear`` reset (kaiming_uniform_ a=sqrt(5)
    weight, uniform bias over 1/sqrt(fan_in)) using a private generator so the
    torch global RNG stream is never consumed.
    """

    linear = module[0]
    if not isinstance(linear, nn.Linear):
        raise TypeError("relative projection must start with nn.Linear")
    generator = torch.Generator(device=linear.weight.device)
    generator.manual_seed(int(seed))
    fan_in = linear.weight.shape[1]
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        linear.weight.uniform_(-bound, bound, generator=generator)
        linear.bias.uniform_(-bound, bound, generator=generator)


def copy_shared_modules(source: nn.Module, target: nn.Module) -> dict[str, float]:
    """Copy every shared-module tensor source -> target; return per-tensor max abs diff after copy."""

    source_state = source.state_dict()
    target_params = dict(target.named_parameters())
    report: dict[str, float] = {}
    with torch.no_grad():
        for name, parameter in target_params.items():
            if name.startswith(SHARED_MODULE_PREFIXES):
                if name not in source_state:
                    raise KeyError(f"source model has no shared parameter {name!r}")
                parameter.copy_(source_state[name].to(dtype=parameter.dtype, device=parameter.device))
                report[name] = float((parameter - source_state[name].to(parameter.device)).abs().max().item())
    if not report:
        raise RuntimeError("no shared parameters were copied")
    return report


class GlobalContextFTNet(FTNet):
    """FTNet visual-only variant with Local + Global + Relative context slots."""

    def __init__(
        self,
        config: FTNetConfig | None = None,
        *,
        context_mode: str = "global_relative",
        relative_dim: int = 32,
        relative_seed: int | None = None,
    ) -> None:
        super().__init__(config)
        if context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {CONTEXT_MODES}, got {context_mode!r}")
        if relative_dim <= 0:
            raise ValueError("relative_dim must be positive")
        # The candidate replaces the zero-masked Delta/Native slots entirely.
        self.delta_branch = None
        self.native_branch = None
        self.context_mode = context_mode
        self.relative_dim = int(relative_dim)
        self.relative_projection = _projection(self.config.branch_dim, self.relative_dim)
        if relative_seed is not None:
            initialize_relative_projection(self.relative_projection, seed=int(relative_seed))
        self.capture_context = False
        self.last_context_tensors: dict[str, torch.Tensor] | None = None

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
        if native_features is not None:
            raise ValueError("GlobalContextFTNet is visual-only; native_features are not accepted")

        mask = sequence_mask.unsqueeze(-1)
        sanitized_visual = visual_features.masked_fill(~mask, 0.0)
        local = self.visual_branch(sanitized_visual).masked_fill(~mask, 0.0)

        if self.context_mode == "off":
            global_context = torch.zeros_like(local)
            relative = local.new_zeros((batch, length, self.relative_dim))
        else:
            pooled = masked_temporal_mean(local, sequence_mask).unsqueeze(1)
            global_context = pooled.expand(-1, length, -1).masked_fill(~mask, 0.0)
            relative = self.relative_projection(local - global_context).masked_fill(~mask, 0.0)

        if self.capture_context:
            self.last_context_tensors = {
                "local": local.detach(),
                "global": global_context.detach(),
                "relative": relative.detach(),
            }

        concatenated = torch.cat([local, global_context, relative], dim=-1)
        temporal = self.fusion(concatenated).masked_fill(~mask, 0.0)
        for block in self.temporal_blocks:
            temporal = block(temporal, sequence_mask)

        logits = self.y_head(temporal).squeeze(-1)
        logits = logits.masked_fill(~sequence_mask, 0.0)
        probabilities = torch.sigmoid(logits).masked_fill(~sequence_mask, 0.0)
        return FTNetOutput(logits=logits, probabilities=probabilities, features=temporal)


def build_initialized_pair(
    config: FTNetConfig,
    *,
    seed: int,
    context_mode: str = "global_relative",
    relative_dim: int = 32,
) -> tuple[FTNet, GlobalContextFTNet, dict]:
    """Construct (baseline, candidate) with identical shared-module initialization.

    Caller must have seeded the torch global RNG with ``seed`` beforehand
    (same convention as the frozen baseline trainer).  The candidate's Relative
    Projection uses ``derive_seed(seed, "relative_projection")`` and the torch
    global RNG state is restored to the post-baseline-construction snapshot, so
    the candidate run consumes exactly the same RNG stream as a baseline run.
    """

    baseline = FTNet(config)
    rng_snapshot = torch.get_rng_state()
    candidate = GlobalContextFTNet(
        config,
        context_mode=context_mode,
        relative_dim=relative_dim,
        relative_seed=derive_seed(seed, RELATIVE_PROJECTION_SEED_TAG),
    )
    shared_report = copy_shared_modules(baseline, candidate)
    shared_max_abs_diff = max(shared_report.values()) if shared_report else float("inf")
    torch.set_rng_state(rng_snapshot)
    evidence = {
        "seed": int(seed),
        "context_mode": context_mode,
        "relative_dim": int(relative_dim),
        "relative_projection_seed": derive_seed(seed, RELATIVE_PROJECTION_SEED_TAG),
        "shared_parameter_count": len(shared_report),
        "shared_max_abs_diff": shared_max_abs_diff,
        "shared_all_zero_diff": bool(shared_max_abs_diff == 0.0),
        "rng_state_restored": True,
    }
    return baseline, candidate, evidence


__all__ = [
    "CONTEXT_MODES",
    "GlobalContextFTNet",
    "build_initialized_pair",
    "copy_shared_modules",
    "derive_seed",
    "initialize_relative_projection",
    "masked_temporal_mean",
]
