"""Empty-video engineering semantics for the true-fresh VHiCraft pipeline.

A fresh Qwen video may legitimately return no highlight (``has_highlight:
false``).  Such a video is a valid member of the Dev166 membership with an
empty frame set: it must emit ``predictions: []`` (never be dropped, never be
filled from the frozen cache) and must be distinguishable from a missing
artifact.
"""

from __future__ import annotations

from aic_video_highlight.spatial_composition.vhicraft_pipeline import (
    FrameCrop,
    assemble_prediction_lines,
    compare_replays,
    validate_prediction_lines,
)
from aic_video_highlight.spatial_composition.submission import write_predictions_jsonl
from aic_video_highlight.spatial_localization.full_dev import (
    shard_is_complete,
    write_shard,
)

RATIO = (9, 16)
SIZE = (534, 300)


def _line(video_id: str, frames: dict[int, tuple[int, int, int]]) -> dict:
    return {
        "video_id": video_id,
        "targetRatioWH": [9, 16],
        "predictions": [
            {"frame": frame, "bboxes": list(box)} for frame, box in sorted(frames.items())
        ],
    }


def test_empty_video_emits_empty_predictions_and_validates(tmp_path):
    lines = assemble_prediction_lines(
        {"a": {}, "b": {0: FrameCrop(0, 0, 0, 168)}},
        target_ratio=RATIO,
        frame_size_by_video={"a": SIZE, "b": SIZE},
        frame_count_by_video={"a": 100, "b": 100},
    )
    by_video = {line["video_id"]: line for line in lines}
    assert by_video["a"]["predictions"] == []
    assert by_video["b"]["predictions"] == [{"frame": 0, "bboxes": [0, 0, 168]}]

    path = tmp_path / "predictions.jsonl"
    write_predictions_jsonl(lines, path)
    report = validate_prediction_lines(
        path,
        index={"a": [9, 16], "b": [9, 16]},
        metadata={"a": {"width": 534, "height": 300, "frame_count": 100},
                  "b": {"width": 534, "height": 300, "frame_count": 100}},
    )
    assert report.is_valid, report.issues
    assert report.stats["video_count"] == 2
    assert report.stats["empty_video_count"] == 1


def test_empty_shard_is_complete_and_distinct_from_missing(tmp_path):
    write_shard(tmp_path, "empty", [], [], [])
    assert shard_is_complete(tmp_path, "empty", [])
    assert not shard_is_complete(tmp_path, "missing", [])


def test_comparator_counts_empty_fresh_video_in_macro_jaccard():
    frozen = [
        _line("a", {0: (0, 0, 168)}),
        _line("b", {0: (0, 0, 168)}),
    ]
    fresh = [
        _line("a", {}),
        _line("b", {0: (0, 0, 168)}),
    ]
    comparison = compare_replays(frozen, fresh)
    # empty fresh vs non-empty frozen -> Jaccard 0, must stay in the macro mean.
    assert comparison["frame_set_jaccard_macro"] == 0.5
    assert comparison["bbox_exact_match_rate_macro"] == 1.0
    rates = {row["video_id"]: row["bbox_exact_rate"] for row in comparison["per_video"]}
    assert rates["a"] is None and rates["b"] == 1.0
    assert comparison["fresh_only_frame_count"] == 0
    assert comparison["frozen_only_frame_count"] == 1
