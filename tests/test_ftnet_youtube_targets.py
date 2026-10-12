from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aic_video_highlight.ftnet.youtube_highlights import (
    MISSED_POSITIVE_THRESHOLD,
    YouTubeHighlightsAnnotationError,
    annotation_hashes,
    load_mturk_clips,
    project_soft_vote_targets,
)


def _write_annotations(directory: Path, windows, votes, *, clip_windows=None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mturk_label.json").write_text(
        json.dumps([windows, votes]), encoding="utf-8"
    )
    (directory / "clip.json").write_text(
        json.dumps(clip_windows if clip_windows is not None else windows), encoding="utf-8"
    )


def test_load_mturk_clips_cross_checks_clip_json(tmp_path: Path) -> None:
    _write_annotations(tmp_path, [[0, 10], [10, 20]], [5, 2])
    clips = load_mturk_clips(tmp_path)
    assert [(c.start_frame, c.end_frame, c.vote) for c in clips] == [
        (0, 10, 5.0),
        (10, 20, 2.0),
    ]
    hashes = annotation_hashes(tmp_path)
    assert set(hashes) == {"mturk_label.json", "clip.json"}
    assert all(len(digest) == 64 for digest in hashes.values())


def test_load_mturk_clips_rejects_mismatched_clip_json(tmp_path: Path) -> None:
    _write_annotations(tmp_path, [[0, 10], [10, 20]], [5, 2], clip_windows=[[0, 11], [11, 20]])
    with pytest.raises(YouTubeHighlightsAnnotationError):
        load_mturk_clips(tmp_path)


def test_projection_half_open_mean_and_loss_mask() -> None:
    from aic_video_highlight.ftnet.youtube_highlights import MturkClip

    clips = (MturkClip(2, 6, 5.0), MturkClip(4, 8, 3.0))
    frame_ids = np.arange(10, dtype=np.int64)
    in_candidate = np.zeros(10, dtype=bool)
    in_candidate[[2, 6, 7]] = True
    result = project_soft_vote_targets(frame_ids, clips, in_candidate=in_candidate)

    assert result.target[1] == pytest.approx(0.0)  # outside clip
    assert result.target[2] == pytest.approx(1.0)  # clip [2,6)
    assert result.target[4] == pytest.approx(0.8)  # mean(5,3)/5
    assert result.target[5] == pytest.approx(0.8)
    assert result.target[6] == pytest.approx(0.6)
    assert result.target[8] == pytest.approx(0.0)  # half-open end
    assert result.loss_mask.tolist() == [False, False, True, False, False, False, True, True, False, False]
    # covered but outside candidate and >= threshold -> audit-only missed positive
    assert result.missed_positive.tolist() == [
        False, False, False, True, True, True, False, False, False, False
    ]
    assert result.stats["missed_positive_frames"] == 3
    assert result.stats["supervised_frames"] == 3
    assert MISSED_POSITIVE_THRESHOLD == 0.5


def test_projection_half_open_boundary_belongs_to_later_clip() -> None:
    from aic_video_highlight.ftnet.youtube_highlights import MturkClip

    clips = (MturkClip(0, 5, 5.0), MturkClip(5, 10, 1.0))
    frame_ids = np.array([4, 5], dtype=np.int64)
    result = project_soft_vote_targets(
        frame_ids, clips, in_candidate=np.ones(2, dtype=bool)
    )
    assert result.target.tolist() == pytest.approx([1.0, 0.2])
