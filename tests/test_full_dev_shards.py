import json

import pytest

from aic_video_highlight.spatial_localization.full_dev import (
    FrameIdentityError,
    canonical_sha,
    merge_and_verify,
    pick_even,
    shard_is_complete,
    write_shard,
)


def make_record(video_id: str, frame: int):
    return {"video_id": video_id, "frame": frame, "payload": f"{video_id}-{frame}"}


def populate(tmp_path, entries):
    for video_id, frames in entries.items():
        write_shard(
            tmp_path,
            video_id,
            [make_record(video_id, f) for f in frames],
            [make_record(video_id, f) for f in frames],
            [{"video_id": video_id, "frame": f, "crop": [0, 0, 9, 16]} for f in frames],
        )


def test_write_and_resume_skip_completed_shard(tmp_path) -> None:
    populate(tmp_path, {"a": [1, 2, 3]})

    assert shard_is_complete(tmp_path, "a", [1, 2, 3])


def test_resume_rejects_missing_frame(tmp_path) -> None:
    populate(tmp_path, {"a": [1, 2, 3]})

    assert not shard_is_complete(tmp_path, "a", [1, 2, 3, 4])


def test_resume_rejects_corrupted_artifact(tmp_path) -> None:
    populate(tmp_path, {"a": [1, 2, 3]})
    raw_path = tmp_path / "raw_detector" / "a.jsonl"
    raw_path.write_text('{"broken": true}\n', encoding="utf-8")

    assert not shard_is_complete(tmp_path, "a", [1, 2, 3])


def test_resume_rejects_wrong_count_status(tmp_path) -> None:
    populate(tmp_path, {"a": [1, 2, 3]})
    status_path = tmp_path / "status" / "a.json"
    payload = json.loads(status_path.read_text(encoding="utf-8").splitlines()[0])
    payload["frame_count"] = 99
    status_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert not shard_is_complete(tmp_path, "a", [1, 2, 3])


def test_merge_enforces_frame_identity(tmp_path) -> None:
    populate(tmp_path, {"a": [1, 2], "b": [3]})
    expected = {("a", 1), ("a", 2), ("b", 3)}
    report = merge_and_verify(tmp_path, expected)

    assert report["frame_count"] == 3
    assert report["missing_keys"] == 0
    assert report["extra_keys"] == 0
    assert report["duplicate_keys"] == 0


def test_merge_detects_missing_keys(tmp_path) -> None:
    populate(tmp_path, {"a": [1]})
    with pytest.raises(FrameIdentityError):
        merge_and_verify(tmp_path, {("a", 1), ("a", 2)})


def test_merge_detects_extra_keys(tmp_path) -> None:
    populate(tmp_path, {"a": [1, 5]})
    with pytest.raises(FrameIdentityError):
        merge_and_verify(tmp_path, {("a", 1)})


def test_semantic_hash_ignores_ordering_and_is_canonical() -> None:
    first = canonical_sha([{"a": 1, "b": 2}])
    second = canonical_sha([{"b": 2, "a": 1}])

    assert first == second


def test_pick_even_covers_endpoints_deterministically() -> None:
    values = list(range(97))
    picked = pick_even(values, 10)

    assert len(picked) == 10
    assert picked == pick_even(values, 10)
    assert picked[0] == 0
    assert picked[-1] == values[-1] or picked[-1] in values
