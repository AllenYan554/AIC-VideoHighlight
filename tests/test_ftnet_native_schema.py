from __future__ import annotations

import math

import pytest
import torch

from aic_video_highlight.ftnet.native_schema import (
    CONTINUITY_VALID_INDEX,
    FOCUS_VELOCITY_INDEX,
    LOG1P_Z_FIELDS,
    NATIVE_DIM,
    NATIVE_FIELDS,
    NATIVE_GROUPS,
    NativeSchemaError,
    Z_FIELDS,
    assert_native_contract,
    build_native_matrix,
    compute_normalization_stats,
    empty_raw,
    standardize_native,
    validate_native,
    variance_gate_decision,
)


def test_schema_has_16_fields_in_frozen_group_layout() -> None:
    assert len(NATIVE_FIELDS) == NATIVE_DIM == 16
    assert NATIVE_GROUPS["retrieval_context"] == (0, 1, 2, 3, 4)
    assert NATIVE_GROUPS["subject_geometry"] == (5, 6, 7, 8, 9, 10)
    assert NATIVE_GROUPS["composition"] == (11,)
    assert NATIVE_GROUPS["temporal_stability"] == (12, 13, 14, 15)
    flat = [index for group in NATIVE_GROUPS.values() for index in group]
    assert flat == list(range(16))
    assert NATIVE_FIELDS[10] == "subject_frame_offset"
    assert NATIVE_FIELDS[11] == "geometric_context_retention"


def test_build_native_matrix_orders_fields_and_binarizes_bool() -> None:
    signals = {name: torch.full((4,), float(index)) for index, name in enumerate(NATIVE_FIELDS)}
    signals["subject_track_present"] = torch.tensor([0.2, 0.9, 0.4, 0.7])
    signals["continuity_valid"] = torch.tensor([1.0, 0.0, 1.0, 1.0])

    raw = build_native_matrix(signals)

    assert raw.shape == (4, NATIVE_DIM)
    assert raw[0, 0].item() == 0.0
    assert raw[3, 15].item() == 15.0
    assert torch.equal(raw[:, 5], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.equal(raw[:, CONTINUITY_VALID_INDEX], torch.tensor([1.0, 0.0, 1.0, 1.0]))


def test_build_native_matrix_rejects_unknown_field() -> None:
    with pytest.raises(NativeSchemaError, match="unknown native field"):
        build_native_matrix({"subject_visibility": torch.zeros(3)})


def test_log1p_is_applied_exactly_once_for_log_fields() -> None:
    raw = empty_raw(2)
    value = 9.0
    raw[:, 1] = value
    raw[:, FOCUS_VELOCITY_INDEX] = value
    stats = compute_normalization_stats(torch.zeros(8, NATIVE_DIM))

    out = standardize_native(raw, stats)
    expected_log1p = math.log1p(value)
    mean = math.log1p(0.0)
    assert abs(out[0, 1].item() - (expected_log1p - mean)) < 1e-5
    assert abs(out[0, FOCUS_VELOCITY_INDEX].item() - (expected_log1p - mean)) < 1e-5
    assert not torch.allclose(out[0, 1], torch.tensor(expected_log1p * 2.0))


def test_missing_values_become_standardized_zero_not_raw_zero_recomputed() -> None:
    raw = empty_raw(3)
    raw[:, 6] = torch.tensor([10.0, 20.0, 30.0])
    stats = compute_normalization_stats(torch.zeros(8, NATIVE_DIM))
    missing = torch.zeros_like(raw, dtype=torch.bool)
    missing[:, 6] = torch.tensor([True, False, True])

    out = standardize_native(raw, stats, missing)

    assert out[0, 6].item() == 0.0
    assert out[2, 6].item() == 0.0
    assert out[1, 6].item() != 0.0


def test_z_scores_are_clipped_to_four_sigma() -> None:
    raw = empty_raw(4)
    raw[:, 6] = torch.tensor([0.0, 1.0, 2.0, 1000.0])
    stats = compute_normalization_stats(raw)

    out = standardize_native(raw, stats)

    assert float(out.max()) <= 4.0 + 1e-6
    assert float(out.min()) >= -4.0 - 1e-6


def test_bounded_fields_are_passed_through_untouched() -> None:
    raw = empty_raw(3)
    raw[:, 0] = torch.tensor([0.1, 0.55, 0.99])
    raw[:, 11] = torch.tensor([1.0, 0.5, 0.25])
    stats = compute_normalization_stats(raw)

    out = standardize_native(raw, stats)

    assert torch.allclose(out[:, 0], raw[:, 0], atol=1e-6)
    assert torch.allclose(out[:, 11], raw[:, 11], atol=1e-6)
    assert 0 not in Z_FIELDS
    assert 11 not in Z_FIELDS


def test_stats_are_computed_on_log1p_space_for_log_fields() -> None:
    raw = empty_raw(4)
    raw[:, 3] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    stats = compute_normalization_stats(raw)

    expected = sum(math.log1p(v) for v in [0.0, 1.0, 2.0, 3.0]) / 4
    assert abs(stats.mean[3] - expected) < 1e-6
    assert 3 in LOG1P_Z_FIELDS and 3 in stats.log1p_fields


def test_variance_gate_flags_degenerate_support_ratio() -> None:
    keep = variance_gate_decision(torch.tensor([1.0, 0.5, 0.0, 1.0]))
    assert keep.status == "KEEP" and keep.needs_replacement is False

    replace = variance_gate_decision(torch.ones(100))
    assert replace.status == "REPLACE" and replace.needs_replacement is True
    assert replace.degenerate_fraction == 1.0


def test_assert_native_contract_rejects_bad_dim_and_nonfinite() -> None:
    with pytest.raises(NativeSchemaError, match="last dim"):
        assert_native_contract(torch.zeros(5, 15))
    bad = torch.zeros(5, 16)
    bad[0, 0] = float("nan")
    with pytest.raises(NativeSchemaError, match="finite"):
        assert_native_contract(bad)
    with pytest.raises(NativeSchemaError, match="finite"):
        validate_native(bad)
