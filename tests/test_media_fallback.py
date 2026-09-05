import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from aic_video_highlight.highlight_retrieval.video_clip import extract_video_clip
from aic_video_highlight.highlight_retrieval.video_metadata import probe_video


def _write_fixture(path: Path, *, seconds: float = 2.0, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(round(seconds * fps)):
        frame = np.full((48, 64, 3), index * 5, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_probe_video_falls_back_to_opencv_when_ffprobe_is_missing(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    _write_fixture(source)

    meta = probe_video(source, ffprobe_bin="definitely-missing-ffprobe")

    assert meta.duration_sec == 2.0
    assert meta.fps == 10.0
    assert (meta.width, meta.height, meta.frame_count) == (64, 48, 20)


def test_extract_video_clip_falls_back_to_opencv_with_local_coordinates(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "clip.mp4"
    _write_fixture(source)

    extract_video_clip(
        source,
        output,
        start_sec=0.5,
        end_sec=1.5,
        ffmpeg_bin="definitely-missing-ffmpeg",
    )
    meta = probe_video(output, ffprobe_bin="definitely-missing-ffprobe")

    assert meta.duration_sec == 1.0
    assert meta.frame_count == 10


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_extract_video_clip_with_ffmpeg_starts_at_requested_offset(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "clip.mp4"
    _write_fixture(source, seconds=6.0)

    extract_video_clip(source, output, start_sec=2.0, end_sec=5.0, ffmpeg_bin="ffmpeg")
    meta = probe_video(output, ffprobe_bin="ffprobe")

    assert source.stat().st_size > 0
    assert meta.duration_sec == pytest.approx(3.0, abs=0.5)
    assert meta.duration_sec < 6.0
