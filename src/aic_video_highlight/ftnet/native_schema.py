"""Stage 7 Native Schema v1.1: 16-D upstream structured features.

This module is the single source of truth for the native feature contract
defined in FTNet.md v1.1 section 8.  It never reads labels or ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


NATIVE_DIM = 16
NATIVE_SCHEMA_VERSION = "stage7-native-v1.1"

NATIVE_FIELDS: tuple[str, ...] = (
    "retrieval_support_ratio",
    "candidate_time_from_start_log",
    "candidate_time_to_end_log",
    "candidate_duration_log",
    "normalized_candidate_position",
    "subject_track_present",
    "subject_selection_confidence",
    "subject_selection_margin",
    "bbox_area_ratio",
    "subject_scale_dynamics",
    "subject_frame_offset",
    "geometric_context_retention",
    "bbox_continuity_iou",
    "continuity_valid",
    "stabilized_focus_velocity",
    "confidence_persistence",
)

NATIVE_GROUPS: dict[str, tuple[int, ...]] = {
    "retrieval_context": (0, 1, 2, 3, 4),
    "subject_geometry": (5, 6, 7, 8, 9, 10),
    "composition": (11,),
    "temporal_stability": (12, 13, 14, 15),
}

FIELD_INDEX = {name: index for index, name in enumerate(NATIVE_FIELDS)}

LOG1P_Z_FIELDS: tuple[int, ...] = (1, 2, 3, 9, 14)
Z_FIELDS: tuple[int, ...] = (1, 2, 3, 6, 7, 9, 10, 14)
BOUNDED_FIELDS: tuple[int, ...] = (0, 4, 8, 11, 12, 13, 15)
BOOL_FIELDS: tuple[int, ...] = (5, 13)

SUPPORT_RATIO_INDEX = 0
CONTINUITY_VALID_INDEX = 13
SCALE_DYNAMICS_INDEX = 9
FOCUS_VELOCITY_INDEX = 14

VARIANCE_GATE_THRESHOLD = 0.95
Z_CLIP = 4.0


class NativeSchemaError(ValueError):
    """Raised when native inputs violate the frozen schema."""


@dataclass(frozen=True)
class VarianceGateDecision:
    status: str
    degenerate_fraction: float
    needs_replacement: bool


def empty_raw(frame_count: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if frame_count <= 0:
        raise NativeSchemaError("frame_count must be positive")
    return torch.zeros((frame_count, NATIVE_DIM), dtype=dtype)


def validate_native(native: torch.Tensor) -> None:
    if native.ndim != 2 or native.shape[1] != NATIVE_DIM:
        raise NativeSchemaError(
            f"native features must have shape [T,{NATIVE_DIM}], got {tuple(native.shape)}"
        )
    if not bool(torch.isfinite(native).all()):
        raise NativeSchemaError("native features must be finite after standardization")


def variance_gate_decision(
    support_ratio: torch.Tensor,
    *,
    threshold: float = VARIANCE_GATE_THRESHOLD,
) -> VarianceGateDecision:
    if support_ratio.numel() == 0:
        raise NativeSchemaError("support_ratio must be non-empty")
    degenerate = float((support_ratio == 1.0).to(torch.float64).mean())
    needs_replacement = degenerate > threshold
    return VarianceGateDecision(
        status="REPLACE" if needs_replacement else "KEEP",
        degenerate_fraction=degenerate,
        needs_replacement=needs_replacement,
    )


@dataclass(frozen=True)
class NormalizationStats:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    log1p_fields: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "aic.stage7.native-normalization/v1",
            "native_schema_version": NATIVE_SCHEMA_VERSION,
            "fields": list(NATIVE_FIELDS),
            "mean": list(self.mean),
            "std": list(self.std),
            "log1p_fields": list(self.log1p_fields),
            "z_fields": list(Z_FIELDS),
            "z_clip": Z_CLIP,
        }


def compute_normalization_stats(raw_train: torch.Tensor) -> NormalizationStats:
    """Compute TRAIN-only per-field mean/std on the log1p-transformed space."""

    if raw_train.ndim != 2 or raw_train.shape[1] != NATIVE_DIM:
        raise NativeSchemaError("raw_train must have shape [T,16]")
    if raw_train.shape[0] == 0:
        raise NativeSchemaError("raw_train must contain at least one frame")
    if not bool(torch.isfinite(raw_train).all()):
        raise NativeSchemaError("raw_train must be finite")
    transformed = raw_train.to(torch.float64).clone()
    for index in LOG1P_Z_FIELDS:
        transformed[:, index] = torch.log1p(transformed[:, index].clamp_min(0.0))
    mean = transformed.mean(dim=0)
    std = transformed.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return NormalizationStats(
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
        log1p_fields=LOG1P_Z_FIELDS,
    )


def standardize_native(
    raw: torch.Tensor,
    stats: NormalizationStats,
    missing_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply log1p -> TRAIN z-score -> clip, then forced-standardized-zero missing.

    Fields that are bounded (e.g. ratios) or boolean are passed through unchanged.
    """

    if raw.ndim != 2 or raw.shape[1] != NATIVE_DIM:
        raise NativeSchemaError("raw must have shape [T,16]")
    if missing_mask is not None and missing_mask.shape != raw.shape:
        raise NativeSchemaError("missing_mask must match raw shape")
    result = raw.to(torch.float64).clone()
    mean = torch.tensor(stats.mean, dtype=torch.float64)
    std = torch.tensor(stats.std, dtype=torch.float64)

    for index in LOG1P_Z_FIELDS:
        result[:, index] = torch.log1p(result[:, index].clamp_min(0.0))
    for index in Z_FIELDS:
        result[:, index] = torch.clamp(
            (result[:, index] - mean[index]) / std[index], -Z_CLIP, Z_CLIP
        )
    if missing_mask is not None:
        result = result.masked_fill(missing_mask, 0.0)
    for index in BOOL_FIELDS:
        result[:, index] = (result[:, index] > 0.5).to(torch.float64)
    return result.to(raw.dtype)


def build_native_matrix(signals: dict[str, torch.Tensor]) -> torch.Tensor:
    """Assemble raw [T,16] from named upstream signal tensors (no transforms)."""

    if not signals:
        raise NativeSchemaError("signals must not be empty")
    lengths = {int(value.shape[0]) for value in signals.values()}
    if len(lengths) != 1:
        raise NativeSchemaError("all native signals must share the same frame count")
    frame_count = lengths.pop()
    raw = empty_raw(frame_count)
    for name, value in signals.items():
        if name not in FIELD_INDEX:
            raise NativeSchemaError(f"unknown native field: {name}")
        if value.ndim != 1 or value.shape[0] != frame_count:
            raise NativeSchemaError(f"{name} must have shape [T]")
        raw[:, FIELD_INDEX[name]] = value.to(torch.float32)
    for index in BOOL_FIELDS:
        raw[:, index] = (raw[:, index] > 0.5).to(torch.float32)
    return raw


def assert_native_contract(native: torch.Tensor) -> None:
    if native.shape[-1] != NATIVE_DIM:
        raise NativeSchemaError(f"native last dim must be {NATIVE_DIM}")
    if not bool(torch.isfinite(native).all()):
        raise NativeSchemaError("native must be finite")
