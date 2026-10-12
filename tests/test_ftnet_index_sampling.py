from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aic_video_highlight.composition.frame_projection import CFR_FPS, PTS_TABLE, VideoTiming
from aic_video_highlight.ftnet.index import (
    DECODE_OK,
    DECODE_PENDING,
    audit_index,
    build_index_entries,
    load_index,
    resolve_video_path,
    write_index,
)
from aic_video_highlight.ftnet.sampling import build_uniform_grid


def _manifest_payload() -> dict:
    return {
        "schema": "aic.stage7.ftnet.split-manifest/v1",
        "protocol_name": "VHiCraFTNet-Stage7.1",
        "protocol_version": "1.0",
        "seed": 20260917,
        "manifest_sha256": "m" * 64,
        "content_sha256": "c" * 64,
        "entries": [
            {
                "canonical_video_id": "vid-a",
                "realized_video_id": "vid-a",
                "category": "dog",
                "stage7_split": "TRAIN",
                "relative_video_path": "raw/dog/vid-a.mp4",
                "source_sha256": "a" * 64,
                "annotation_identity": "dog/vid-a",
            },
            {
                "canonical_video_id": "vid-b",
                "realized_video_id": "vid-b-replacement",
                "category": "skiing",
                "stage7_split": "VALIDATION",
                "relative_video_path": "raw/skiing/vid-b.mp4",
                "source_sha256": "b" * 64,
                "annotation_identity": "skiing/vid-b",
            },
        ],
    }


def test_index_entries_are_relative_and_portable(tmp_path: Path) -> None:
    entries, metadata = build_index_entries(_manifest_payload())
    assert [entry.video_id for entry in entries] == ["vid-a", "vid-b"]
    assert all(not Path(entry.relative_video_path).is_absolute() for entry in entries)
    assert all(entry.decode_status == DECODE_PENDING for entry in entries)

    index_path = tmp_path / "index" / "frozen_index.json"
    write_index(entries, metadata=metadata, output_path=index_path)
    sidecar = index_path.with_suffix(index_path.suffix + ".sha256")
    assert sidecar.is_file()
    loaded, loaded_metadata = load_index(index_path)
    assert [entry.video_id for entry in loaded] == ["vid-a", "vid-b"]
    assert loaded_metadata["counts"] == {"TRAIN": 1, "VALIDATION": 1, "CALIBRATION": 0}
    assert resolve_video_path(tmp_path, loaded[0]).as_posix().endswith("youtube_highlights/raw/dog/vid-a.mp4")


def test_index_rejects_absolute_or_escaping_paths() -> None:
    payload = _manifest_payload()
    payload["entries"][0]["relative_video_path"] = "E:/data/vid-a.mp4"
    with pytest.raises(Exception):
        build_index_entries(payload)
    payload = _manifest_payload()
    payload["entries"][0]["relative_video_path"] = "../vid-a.mp4"
    with pytest.raises(Exception):
        build_index_entries(payload)


def test_index_audit_counts_decode_status() -> None:
    entries, _ = build_index_entries(_manifest_payload())
    entries[0] = entries[0].__class__(**{**entries[0].__dict__, "decode_status": DECODE_OK})
    audit = audit_index(entries)
    assert audit["total"] == 2
    assert audit["decode_status"][DECODE_OK] == 1
    assert audit["all_ok"] is False


def _timing_cfr(fps: float, frame_count: int) -> VideoTiming:
    return VideoTiming(video_id="v", fps=fps, frame_count=frame_count, timestamp_mode=CFR_FPS)


def test_uniform_grid_cfr_is_two_fps_with_real_adjacency() -> None:
    sample = build_uniform_grid(_timing_cfr(30.0, 360), sample_fps=2.0)
    assert sample.frames.tolist() == list(range(0, 360, 15))[:24]
    assert sample.timestamps[1] == pytest.approx(0.5)
    assert sample.adjacency_mask[1:].all()
    assert not sample.adjacency_mask[0]
    assert np.all(np.diff(sample.frames) == 15)


def test_uniform_grid_duplicate_frames_break_adjacency() -> None:
    sample = build_uniform_grid(_timing_cfr(1.0, 6), sample_fps=2.0)
    assert sample.frames.tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    # dt=1.0s exceeds 1.5 * 0.5s and duplicates repeat the same source frame
    assert not sample.adjacency_mask.any()


def test_uniform_grid_regular_samples_stay_adjacent() -> None:
    sample = build_uniform_grid(_timing_cfr(5.0, 6), sample_fps=2.0)
    assert sample.frames.tolist() == [0, 3, 5]
    assert sample.adjacency_mask.tolist() == [False, True, True]


def test_uniform_grid_pts_uses_nearest_source_frame() -> None:
    from fractions import Fraction

    pts = tuple(Fraction(value) for value in ("0.0", "0.4", "0.9", "1.33", "2.0"))
    timing = VideoTiming(
        video_id="v", fps=1.0, frame_count=5, timestamp_mode=PTS_TABLE, pts_timestamps=pts
    )
    sample = build_uniform_grid(timing, sample_fps=2.0)
    assert sample.frames.tolist() == [0, 1, 2, 3, 4]
    assert sample.timestamps[1] == pytest.approx(0.4)
    assert sample.frames[3] == 3
    metadata = json.dumps({"frames": sample.frames.tolist()})
    assert "0" in metadata
