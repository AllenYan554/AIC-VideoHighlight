"""Stage 6.2A within-video temporal-signal audit contracts (CPU-only, synthetic)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
T61_DIR = REPO / "scripts" / "experiments"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return _load("aic_test_stage61_failure_audit_tmp", T61_DIR / "analyze_stage6_1_failure.py")


@pytest.fixture(scope="module")
def m(audit):
    return _load("aic_test_stage62a", T61_DIR / "analyze_stage6_2a_temporal_signal.py")


def test_identity_is_posthoc_only(m):
    assert m.IDENTITY == "POSTHOC_DIAGNOSTIC_ONLY"


def test_percentile_rank_monotone_and_constant(m):
    assert list(m.percentile_rank([10, 20, 30])) == pytest.approx([0.0, 0.5, 1.0])
    assert list(m.percentile_rank([5, 5, 5])) == pytest.approx([0.5, 0.5, 0.5])
    assert m.percentile_rank([42]).tolist() == [0.5]


def test_zscore_and_min_max(m):
    z = m.zscore([1, 2, 3])
    assert abs(z.mean()) < 1e-12 and abs(z.std() - 1.0) < 1e-9
    assert m.zscore([4, 4, 4]).tolist() == [0.0, 0.0, 0.0]
    assert m.min_max([0, 5, 10]).tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert m.min_max([7, 7]).tolist() == [0.0, 0.0]


def test_windows_of(m):
    assert m.windows_of(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert m.windows_of(0, 4) == []
    assert m.windows_of(5, 0) == []


def test_tp_temporal_runs(m):
    assert m.tp_temporal_runs([1, 2, 3, 7, 8]) == [3, 2]
    assert m.tp_temporal_runs([]) == []
    assert m.tp_temporal_runs([5, 5, 5]) == [1]


def test_quintile_densities_requires_five_windows(m):
    assert m._quintile_densities([(0.1, 0.5), (0.2, 0.4)]) is None
    result = m._quintile_densities([(i / 10, i / 10) for i in range(10)])
    assert set(result) == {"q1", "q2", "q3", "q4", "q5"}
    assert result["q5"] > result["q1"]


def test_quintile_enrichment(m):
    means = {"q1": 0.10, "q2": 0.10, "q3": 0.10, "q4": 0.10, "q5": 0.20}
    enrichment = m._quintile_enrichment(means)
    assert enrichment["q5_over_q1"] == pytest.approx(2.0)
    assert enrichment["q5_over_overall"] == pytest.approx(0.20 / 0.12)


def test_drop_curve_aggregation(m):
    rows = [
        {"fraction": 0.1, "tp_retained": 8, "fp_removed": 2, "total_tp": 10, "total_fp": 5},
        {"fraction": 0.1, "tp_retained": 9, "fp_removed": 4, "total_tp": 10, "total_fp": 5},
    ]
    curve = m._drop_curve(rows)
    assert len(curve) == 1
    point = curve[0]
    assert point["tp_recall"] == pytest.approx(0.85)
    assert point["fp_removed_fraction"] == pytest.approx(0.6)
    assert point["tp_loss_fraction"] == pytest.approx(0.15)
    assert point["fp_minus_tp_removal"] == pytest.approx(0.45)


def test_window_metrics_detects_enrichment(m, audit):
    rows = [
        {"score": 0.9, "tp_density": 0.8, "has_tp": 1},
        {"score": 0.8, "tp_density": 0.6, "has_tp": 1},
        {"score": 0.2, "tp_density": 0.1, "has_tp": 0},
        {"score": 0.1, "tp_density": 0.0, "has_tp": 0},
    ]
    metrics = m._window_metrics(audit, rows)
    assert metrics["window_count"] == 4
    assert metrics["auroc_has_tp"] == pytest.approx(1.0)
    assert metrics["top_enrichment"]["top_10pct"]["tp_density"] == pytest.approx(0.8)
    assert metrics["baseline_tp_density"] == pytest.approx(0.375)


def test_cli_defaults(m, tmp_path):
    args = m.parse_args([
        "--protocol", str(tmp_path / "p.json"),
        "--model-manifest", str(tmp_path / "m.json"),
        "--control-run", str(tmp_path / "c"),
        "--feature-cache", str(tmp_path / "f"),
        "--output-root", str(tmp_path / "o"),
    ])
    assert args.seed == 20260915
    assert args.max_videos == 0
    assert m.WINDOW_SECONDS == (2.0, 5.0)
    assert m.DROP_FRACTIONS[-1] == 0.5
