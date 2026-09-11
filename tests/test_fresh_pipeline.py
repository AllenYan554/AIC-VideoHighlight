"""Deterministic fresh-core tests (no GPU, no frozen Stage 4 cache)."""

from __future__ import annotations

import pytest

from aic_video_highlight.spatial_composition.fresh_pipeline import (
    FRESH_SHARD_SCHEMA_VERSION,
    FreshPipelineError,
    audit_fresh_shard,
    build_fresh_index,
    build_shard_records,
    build_ts5_by_frame,
    shared_upstream_identity,
)

WIDTH = 534
HEIGHT = 300
CROP_W = 168
CROP_H = CROP_W * 16 / 9
TARGET_RATIO = (9, 16)
ALPHA = 0.5


def _composed(frame: int, center_x: int, center_y: int) -> dict:
    return {
        "frame": frame,
        "cmp1": {
            "x": 183,
            "y": 0,
            "w": CROP_W,
            "h": CROP_H,
            "fallback": False,
            "placement_status": "SUBJECT_SHIFTED",
        },
        "sanitized": {
            "xyxy": [center_x - 20, center_y - 40, center_x + 20, center_y + 40],
        },
    }


def _sequence():
    return [
        _composed(0, 250, 150),
        _composed(1, 260, 150),
        _composed(2, 270, 150),
    ]


def test_ts5_is_deterministic_and_preserves_frames():
    records = _sequence()
    first = build_ts5_by_frame(
        records, width=WIDTH, height=HEIGHT, target_ratio=TARGET_RATIO, alpha=ALPHA
    )
    second = build_ts5_by_frame(
        records, width=WIDTH, height=HEIGHT, target_ratio=TARGET_RATIO, alpha=ALPHA
    )
    assert sorted(first) == [0, 1, 2]
    assert {frame: (item.x, item.y, item.w) for frame, item in first.items()} == {
        frame: (item.x, item.y, item.w) for frame, item in second.items()
    }


def test_shard_records_are_schema_stable_and_reference_ts0():
    records = _sequence()
    by_frame = build_ts5_by_frame(
        records, width=WIDTH, height=HEIGHT, target_ratio=TARGET_RATIO, alpha=ALPHA
    )
    shard = build_shard_records(records, by_frame)
    assert [item["frame"] for item in shard] == [0, 1, 2]
    for item in shard:
        assert item["schema_version"] == FRESH_SHARD_SCHEMA_VERSION
        assert item["ts0"]["w"] == CROP_W
        assert item["ts0"]["x"] == 183
        assert item["ts5"]["w"] == CROP_W
    again = build_shard_records(records, by_frame)
    assert shard == again


def test_ts5_requires_nonempty_sequence():
    with pytest.raises(FreshPipelineError):
        build_ts5_by_frame([], width=WIDTH, height=HEIGHT, target_ratio=TARGET_RATIO)


def test_fresh_index_projects_frozen_membership_in_role_order():
    role = [{"video_id": "b"}, {"video_id": "a"}, {"video_id": "c"}]
    frozen = [
        {"video_id": "a", "targetRatioWH": [9, 16], "video_path": "a.mp4"},
        {"video_id": "b", "targetRatioWH": [9, 16], "video_path": "b.mp4"},
        {"video_id": "c", "targetRatioWH": [9, 16], "video_path": "c.mp4"},
        {"video_id": "z", "targetRatioWH": [9, 16], "video_path": "z.mp4"},
    ]
    index = build_fresh_index(role, frozen)
    assert [entry["video_id"] for entry in index] == ["b", "a", "c"]


def test_fresh_index_supports_video_subset_and_rejects_missing():
    role = [{"video_id": "a"}, {"video_id": "b"}]
    frozen = [{"video_id": "a", "targetRatioWH": [9, 16], "video_path": "a.mp4"}]
    assert [e["video_id"] for e in build_fresh_index(role, frozen, video_ids=["a"])] == ["a"]
    with pytest.raises(FreshPipelineError):
        build_fresh_index(role, frozen)


def test_shared_upstream_identity_is_deterministic(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    identity = shared_upstream_identity({"cache": first, "stage5_1": second})
    assert identity["shared"] is True
    assert identity["shared_upstream_sha256"] == shared_upstream_identity(
        {"cache": first, "stage5_1": second}
    )["shared_upstream_sha256"]
    assert set(identity["artifacts"]) == {"cache", "stage5_1"}


def test_audit_fresh_shard_detects_duplicates():
    records = _sequence()
    by_frame = build_ts5_by_frame(
        records, width=WIDTH, height=HEIGHT, target_ratio=TARGET_RATIO, alpha=ALPHA
    )
    shard = build_shard_records(records, by_frame)
    audit = audit_fresh_shard("v", shard)
    assert audit.frame_count == 3
    assert audit.frames_sha256 == audit_fresh_shard("v", build_shard_records(records, by_frame)).frames_sha256
    with pytest.raises(FreshPipelineError):
        audit_fresh_shard("v", shard + [shard[0]])
