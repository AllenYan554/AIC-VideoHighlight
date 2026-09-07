import shutil
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from aic_video_highlight.highlight_retrieval.video_clip import extract_video_clip
from aic_video_highlight.highlight_retrieval.video_metadata import probe_video

_SHOWINFO_PTS = re.compile(r"\bpts_time:([-+0-9.eE]+)")
_SHOWINFO_CHECKSUM = re.compile(r"\bchecksum:([0-9A-Fa-f]+)")


def _write_fixture(path: Path, *, seconds: float = 2.0, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(round(seconds * fps)):
        frame = np.full((48, 64, 3), (index * 5) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_h264_fixture(path: Path, *, gop: int, with_audio: bool) -> None:
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=30:duration=6.2",
    ]
    if with_audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=6.2"]
    command += [
        "-c:v", "libx264", "-g", str(gop), "-keyint_min", str(gop),
        "-sc_threshold", "0", "-pix_fmt", "yuv420p",
    ]
    if with_audio:
        command += ["-c:a", "aac", "-shortest"]
    else:
        command += ["-an"]
    command.append(str(path))
    subprocess.run(command, check=True, capture_output=True)


def _independent_source_frame_evidence(source: Path, origin: float) -> tuple[float, str]:
    command = [
        "ffmpeg", "-v", "info", "-ss", f"{origin:.9f}", "-copyts", "-i", str(source),
        "-map", "0:v:0", "-vf", f"trim=start={origin:.9f},showinfo", "-frames:v", "1",
        "-f", "null", "-",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    pts = _SHOWINFO_PTS.search(completed.stderr)
    checksum = _SHOWINFO_CHECKSUM.search(completed.stderr)
    assert pts and checksum
    return float(pts.group(1)), checksum.group(1).upper()


def _first_frame_perceptual_hash(path: Path, *, frame_index: int = 0) -> int:
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    assert capture.isOpened()
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes())


def _first_video_pts(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "frame=best_effort_timestamp_time", "-of", "csv=p=0", "-read_intervals", "%+0.2",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.splitlines()[0].strip().rstrip(","))


def _has_audio(path: Path) -> bool:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return any(stream["codec_type"] == "audio" for stream in json.loads(completed.stdout)["streams"])


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

    timing = extract_video_clip(source, output, start_sec=2.0, end_sec=5.0, ffmpeg_bin="ffmpeg")
    meta = probe_video(output, ffprobe_bin="ffprobe")

    assert timing["coordinate_contract"] == "source_sec = actual_source_origin_sec + local_sec"
    assert timing["requested_source_start_sec"] == 2.0
    assert timing["actual_source_origin_sec"] == pytest.approx(2.0, abs=0.11)
    assert source.stat().st_size > 0
    assert meta.duration_sec == pytest.approx(3.0, abs=0.5)
    assert meta.duration_sec < 6.0


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are not installed",
)
@pytest.mark.parametrize(
    ("gop", "with_audio", "starts"),
    [
        (15, False, (0.03, 1.271, 5.80)),
        (60, True, (0.03, 3.271, 5.80)),
    ],
)
def test_frame_accurate_clip_contract_across_gop_audio_and_edges(
    tmp_path, gop, with_audio, starts
) -> None:
    source = tmp_path / f"source_g{gop}_{with_audio}.mp4"
    _write_h264_fixture(source, gop=gop, with_audio=with_audio)

    for index, requested_start in enumerate(starts):
        requested_end = min(6.2, requested_start + 0.35)
        output = tmp_path / f"clip_{index}.mp4"
        timing = extract_video_clip(
            source,
            output,
            start_sec=requested_start,
            end_sec=requested_end,
            ffmpeg_bin="ffmpeg",
        )
        origin = float(timing["actual_source_origin_sec"])
        source_pts, source_checksum = _independent_source_frame_evidence(source, origin)
        source_frame_index = round(origin * 30.0)

        assert requested_start - 1e-6 <= origin <= requested_start + 1 / 30 + 1e-6
        assert source_pts == pytest.approx(origin, abs=1e-6)
        assert timing["first_source_frame_checksum"] == source_checksum
        assert _first_video_pts(output) == pytest.approx(0.0, abs=1e-6)
        source_hash = _first_frame_perceptual_hash(source, frame_index=source_frame_index)
        clip_hash = _first_frame_perceptual_hash(output)
        assert (source_hash ^ clip_hash).bit_count() <= 3
        assert _has_audio(output) is with_audio
