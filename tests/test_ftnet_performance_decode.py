from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from aic_video_highlight.ftnet.performance_decode import (
    DecodeCounters,
    iter_one_pass_batches,
    iter_windowed_prefix_batches,
)


class _Capture:
    def __init__(self, frames, opens):
        self.frames = frames
        self.index = -1
        self.opens = opens
        self.opens.append(self)

    def isOpened(self):
        return True

    def grab(self):
        self.index += 1
        return self.index < len(self.frames)

    def retrieve(self):
        if 0 <= self.index < len(self.frames):
            return True, self.frames[self.index].copy()
        return False, None

    def release(self):
        return None


@pytest.fixture
def fake_cv2(monkeypatch):
    frames = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(12)]
    opens = []
    module = SimpleNamespace(
        COLOR_BGR2RGB=1,
        VideoCapture=lambda _path: _Capture(frames, opens),
        cvtColor=lambda image, _code: image[..., ::-1],
    )
    monkeypatch.setitem(sys.modules, "cv2", module)
    return frames, opens


def _flatten(batches):
    ids = []
    images = []
    for batch_ids, batch_images in batches:
        ids.extend(batch_ids)
        images.extend(batch_images)
    return ids, images


def test_one_pass_matches_windowed_order_and_duplicate_identity(tmp_path, fake_cv2):
    requested = [0, 2, 2, 5, 8, 11]
    baseline_stats = DecodeCounters()
    optimized_stats = DecodeCounters()
    baseline = list(
        iter_windowed_prefix_batches(
            tmp_path / "video.mp4",
            requested,
            batch_size=2,
            decode_window=2,
            counters=baseline_stats,
        )
    )
    optimized = list(
        iter_one_pass_batches(
            tmp_path / "video.mp4",
            requested,
            batch_size=2,
            counters=optimized_stats,
        )
    )

    baseline_ids, baseline_images = _flatten(baseline)
    optimized_ids, optimized_images = _flatten(optimized)
    assert baseline_ids == optimized_ids == requested
    assert all(np.array_equal(left, right) for left, right in zip(baseline_images, optimized_images))
    assert baseline_stats.capture_open_count == 3
    assert optimized_stats.capture_open_count == 1
    assert baseline_stats.grabbed_frame_count > optimized_stats.grabbed_frame_count
    assert baseline_stats.repeated_prefix_decoded_frame_estimate > 0
    assert optimized_stats.repeated_prefix_decoded_frame_estimate == 0
    assert baseline_stats.emitted_frame_count == optimized_stats.emitted_frame_count == len(requested)


def test_one_pass_propagates_missing_frame_error(tmp_path, fake_cv2):
    with pytest.raises(RuntimeError, match="frames missing"):
        list(
            iter_one_pass_batches(
                tmp_path / "short.mp4",
                [0, 20],
                batch_size=4,
            )
        )


def test_decoder_rejects_unsorted_frame_ids(tmp_path, fake_cv2):
    with pytest.raises(ValueError, match="monotonically"):
        list(
            iter_one_pass_batches(
                tmp_path / "video.mp4",
                [3, 1],
                batch_size=2,
            )
        )
