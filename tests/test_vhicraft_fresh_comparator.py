"""Preregistered fresh-vs-frozen comparator: macro frame-Jaccard and macro bbox.

Frozen definitions (Stage 5.6 corrective preregistration):
- frame-set Jaccard per video = |F_fresh n F_frozen| / |F_fresh u F_frozen|;
  gate metric = macro mean over videos.
- bbox exact match per video = exact common bbox / common frame count;
  gate metric = macro mean over videos sharing >=1 common frame.
- micro bbox exact match is reported as a diagnostic only, never the gate.
"""

from __future__ import annotations

from aic_video_highlight.spatial_composition.vhicraft_pipeline import (
    FRESH_REPRODUCTION_POLICY,
    compare_replays,
    evaluate_fresh_reproduction,
)


def _line(video_id: str, frames: dict[int, tuple[int, int, int]]) -> dict:
    return {
        "video_id": video_id,
        "targetRatioWH": [9, 16],
        "predictions": [
            {"frame": frame, "bboxes": list(box)} for frame, box in sorted(frames.items())
        ],
    }


def _scenario():
    cached = [
        _line("a", {0: (0, 0, 168), 1: (0, 0, 168)}),
        _line("b", {frame: (5, 5, 168) for frame in range(10)}),
    ]
    fresh = [
        _line("a", {0: (0, 0, 168), 1: (9, 0, 168)}),
        _line("b", {frame: (5, 5, 168) for frame in range(10)}),
    ]
    return compare_replays(cached, fresh)


def test_macro_and_micro_bbox_differ():
    comparison = _scenario()
    assert comparison["frame_set_jaccard_macro"] == 1.0
    assert comparison["bbox_exact_match_rate_macro"] == 0.75
    assert abs(comparison["bbox_exact_match_rate"] - 11 / 12) < 1e-9
    assert comparison["common_frame_count"] == 12
    assert comparison["fresh_only_frame_count"] == 0
    assert comparison["frozen_only_frame_count"] == 0
    rates = {row["video_id"]: row["bbox_exact_rate"] for row in comparison["per_video"]}
    assert rates == {"a": 0.5, "b": 1.0}


def test_gate_uses_macro_bbox_not_micro():
    comparison = _scenario()
    gate = evaluate_fresh_reproduction(
        comparison, schema_success_rate=1.0, contract_valid=True
    )
    assert not gate["all_pass"]
    assert "bbox_exact_match_rate" in gate["failed_checks"]
    check = gate["checks"]["bbox_exact_match_rate"]
    assert check["observed"] == 0.75
    assert abs(check["observed_micro"] - 11 / 12) < 1e-9
    assert check["metric"] == "macro_mean_per_video_exact_bbox_rate"

    relaxed = dict(FRESH_REPRODUCTION_POLICY, bbox_exact_match_rate_min=0.75)
    gate2 = evaluate_fresh_reproduction(
        comparison, schema_success_rate=1.0, contract_valid=True, policy=relaxed
    )
    assert "bbox_exact_match_rate" not in gate2["failed_checks"]


def test_macro_bbox_equals_one_when_all_common_bboxes_exact():
    cached = [_line("a", {0: (1, 2, 168), 3: (4, 5, 168)})]
    fresh = [_line("a", {0: (1, 2, 168), 3: (4, 5, 168)})]
    comparison = compare_replays(cached, fresh)
    assert comparison["bbox_exact_match_rate_macro"] == 1.0
    assert comparison["frame_set_jaccard_macro"] == 1.0


def test_fresh_only_and_frozen_only_counts_and_jaccard():
    cached = [_line("a", {0: (0, 0, 168), 1: (0, 0, 168)})]
    fresh = [_line("a", {1: (0, 0, 168), 2: (0, 0, 168)})]
    comparison = compare_replays(cached, fresh)
    assert comparison["fresh_only_frame_count"] == 1
    assert comparison["frozen_only_frame_count"] == 1
    assert comparison["common_frame_count"] == 1
    assert abs(comparison["frame_set_jaccard_macro"] - 1 / 3) < 1e-9
    assert comparison["bbox_exact_match_rate_macro"] == 1.0


def test_missing_video_counts_against_macro_jaccard():
    cached = [_line("a", {0: (0, 0, 168)}), _line("b", {0: (0, 0, 168)})]
    fresh = [_line("a", {0: (0, 0, 168)})]
    comparison = compare_replays(cached, fresh)
    # video b: union={0}, inter={} -> jaccard 0.0; video a -> 1.0; macro = 0.5
    assert comparison["frame_set_jaccard_macro"] == 0.5
    assert comparison["video_completion_rate"] == 0.5
