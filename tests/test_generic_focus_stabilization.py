from __future__ import annotations

import numpy as np
import pytest

from aic_video_highlight.composition.generic_focus import (
    CMP_VERSION,
    FALLBACK_ORIGINAL_VIEW,
    RENDER_MODE_ORIGINAL_VIEW,
    RENDER_MODE_SUBJECT_FOCUS,
    geometric_context_retention,
    plan_subject_focus,
)
from aic_video_highlight.composition.generic_stabilization import (
    stabilize_focus,
    stabilized_focus_velocity,
)
from aic_video_highlight.localization.selection_margin import subject_selection_margin
from aic_video_highlight.localization.subject_localization import SubjectCandidate


def test_focus_plan_is_normalized_and_aspect_agnostic() -> None:
    plan = plan_subject_focus((0.4, 0.4, 0.6, 0.6))
    assert plan.cmp_version == CMP_VERSION
    assert plan.render_mode == RENDER_MODE_SUBJECT_FOCUS
    assert plan.plan_valid is True
    x, y, w, h = plan.focus_window
    assert 0.0 <= x and x + w <= 1.0 + 1e-9
    assert 0.0 <= y and y + h <= 1.0 + 1e-9
    assert plan.geometric_context_retention == pytest.approx(w * h)
    assert plan.focus_center == pytest.approx((x + w / 2, y + h / 2))


def test_focus_plan_with_target_aspect_ratio_stays_in_frame() -> None:
    plan = plan_subject_focus((0.45, 0.45, 0.55, 0.55), target_aspect_ratio=9 / 16)
    _x, _y, w, h = plan.focus_window
    assert w / h == pytest.approx(9 / 16, rel=1e-6)
    assert w <= 1.0 + 1e-9 and h <= 1.0 + 1e-9


def test_no_subject_falls_back_to_original_view_never_drop() -> None:
    for box in (None, (0.5, 0.5, 0.5, 0.6), (0.1, 0.1, 1.5, 0.5)):
        plan = plan_subject_focus(box)
        assert plan.render_mode == RENDER_MODE_ORIGINAL_VIEW
        assert plan.fallback_state == FALLBACK_ORIGINAL_VIEW
        assert plan.plan_valid is False
        assert plan.geometric_context_retention == 1.0
        assert plan.focus_window == (0.0, 0.0, 1.0, 1.0)


def test_geometric_context_retention_helper() -> None:
    assert geometric_context_retention((0.0, 0.0, 0.5, 0.5)) == pytest.approx(0.25)


def test_stabilization_resets_on_discontinuity_and_never_crosses_it() -> None:
    center = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    scale = np.array([0.1, 0.1, 0.9, 0.9])
    window = np.array([[0, 0, 0.1, 0.1], [0, 0, 0.1, 0.1], [0, 0, 0.9, 0.9], [0, 0, 0.9, 0.9]], dtype=float)
    adjacency = np.array([False, True, False, True])

    out = stabilize_focus(center, scale, window, adjacency_mask=adjacency, alpha=0.6)

    assert out.reset_flag.tolist() == [True, False, True, False]
    assert out.stabilized_focus_center[1].tolist() == [0.0, 0.0]
    assert out.stabilized_focus_center[3].tolist() == [1.0, 1.0]
    assert out.stabilized_focus_center[2].tolist() == [1.0, 1.0]


def test_stabilized_velocity_is_zero_across_discontinuity() -> None:
    center = np.array([[0.0, 0.0], [0.2, 0.0], [1.0, 1.0]])
    ts = np.array([0.0, 0.5, 1.0])
    adjacency = np.array([False, True, False])
    velocity = stabilized_focus_velocity(center, ts, adjacency_mask=adjacency)
    assert velocity[0] == 0.0 and velocity[2] == 0.0
    assert velocity[1] == pytest.approx(0.2 / 0.5)


def _cand(score: float, box=(0, 0, 10, 10)) -> SubjectCandidate:
    return SubjectCandidate(box=box, score=score, label_id=1, label="person")


def test_selection_margin_uses_top1_minus_max_top2_floor() -> None:
    margin = subject_selection_margin((_cand(0.9), _cand(0.4), _cand(0.2)), detection_floor=0.15)
    assert margin.top1_score == pytest.approx(0.9)
    assert margin.top2_score == pytest.approx(0.4)
    assert margin.margin == pytest.approx(0.5)
    assert margin.available is True


def test_selection_margin_clamps_below_floor() -> None:
    margin = subject_selection_margin((_cand(0.5), _cand(0.2)), detection_floor=0.45)
    assert margin.margin == pytest.approx(0.05)


def test_selection_margin_unavailable_when_no_valid_candidate() -> None:
    margin = subject_selection_margin(())
    assert margin.available is False and margin.margin == 0.0
