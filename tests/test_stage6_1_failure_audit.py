"""Stage 6.1 failure-mechanism audit contracts (CPU-only, synthetic)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO / "scripts" / "experiments" / "analyze_stage6_1_failure.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("aic_test_stage61_failure_audit", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- mapping ---------------------------------------------------------------

def test_frame_to_sample_index_covers_interval(audit):
    sample_frames = (0, 10, 20)
    assert audit.frame_to_sample_index(sample_frames, 0) == 0
    assert audit.frame_to_sample_index(sample_frames, 9) == 0
    assert audit.frame_to_sample_index(sample_frames, 10) == 1
    assert audit.frame_to_sample_index(sample_frames, 19) == 1
    assert audit.frame_to_sample_index(sample_frames, 20) == 2
    assert audit.frame_to_sample_index(sample_frames, 999) == 2  # clamps to last


def test_nearest_sample_distance(audit):
    sample_frames = (0, 10, 20)
    assert audit.nearest_sample_distance(sample_frames, 0) == 0
    assert audit.nearest_sample_distance(sample_frames, 4) == 4
    assert audit.nearest_sample_distance(sample_frames, 6) == 4
    assert audit.nearest_sample_distance(sample_frames, 15) == 5
    assert audit.nearest_sample_distance(sample_frames, 100) == 80


def test_mapping_rejects_empty_grid(audit):
    with pytest.raises(audit.FailureAuditError):
        audit.frame_to_sample_index((), 0)


# --- ranking statistics ----------------------------------------------------

def test_auroc_perfect_and_inverse(audit):
    assert audit.auroc([3, 2, 1], [1, 1, 0]) == pytest.approx(1.0)
    assert audit.auroc([1, 2, 3], [1, 1, 0]) == pytest.approx(0.0)


def test_auroc_handles_ties(audit):
    two_pos_two_neg_tied = audit.auroc([1, 1, 1, 1], [1, 1, 0, 0])
    assert two_pos_two_neg_tied == pytest.approx(0.5)
    assert np.isnan(audit.auroc([1, 2], [1, 1]))


def test_average_precision_upper_and_lower(audit):
    assert audit.average_precision([1.0, 0.9, 0.8], [1, 0, 0]) == pytest.approx(1.0)
    assert audit.average_precision([0.1, 0.2, 0.3], [1, 0, 0]) < 1.0


def test_cohens_d_sign(audit):
    assert audit.cohens_d([2, 3, 4], [0, 1, 2]) > 0
    assert audit.cohens_d([0, 1, 2], [2, 3, 4]) < 0


def test_retention_curve_monotone_recall(audit):
    curve = audit.retention_curve([3, 2, 1], [1, 1, 0], [1 / 3, 2 / 3, 1.0])
    assert [round(point["tp_recall"], 6) for point in curve] == [0.5, 1.0, 1.0]
    assert curve[0]["tp_kept"] == 1


def test_oracle_tp_upper_bound(audit):
    assert audit.oracle_tp_upper_bound([5, 5], [3, 10]) == 8
    assert audit.oracle_tp_upper_bound([2], [0]) == 0


def test_expected_random_retention_full_coverage(audit):
    rng = np.random.default_rng(0)
    result = audit.expected_random_retention(list(range(10)), list(range(10)), 10, 10, rng, 5)
    assert result["fs0_retention"] == pytest.approx(1.0)
    assert result["tp_retention"] == pytest.approx(1.0)


def test_expected_random_retention_zero_coverage(audit):
    rng = np.random.default_rng(0)
    result = audit.expected_random_retention([0, 1, 2], [0], 0, 10, rng, 5)
    assert result["fs0_retention"] == 0.0


# --- module surface --------------------------------------------------------

def test_audit_identity_is_posthoc_only(audit):
    assert audit.IDENTITY == "POSTHOC_DIAGNOSTIC_ONLY"


def test_audit_cli_parses(audit, tmp_path):
    args = audit.parse_args([
        "--protocol", str(tmp_path / "p.json"),
        "--model-manifest", str(tmp_path / "m.json"),
        "--control-run", str(tmp_path / "c"),
        "--feature-cache", str(tmp_path / "f"),
        "--output-root", str(tmp_path / "o"),
    ])
    assert args.random_trials == 100
    assert args.seed == 20260915
