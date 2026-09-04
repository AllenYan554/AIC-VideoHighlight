"""Temporary video clip extraction with an OpenCV fallback."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path


class VideoClipError(RuntimeError):
    """Raised when an experiment clip cannot be created."""


def _extract_with_opencv(source: Path, output: Path, start_sec: float, end_sec: float) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise VideoClipError("ffmpeg is unavailable and OpenCV is not installed") from exc

    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise VideoClipError(f"OpenCV could not open source video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        capture.release()
        raise VideoClipError(f"OpenCV returned invalid source metadata: {source}")

    start_frame = max(0, round(start_sec * fps))
    requested_frames = max(1, round((end_sec - start_sec) * fps))
    end_frame = min(frame_count, start_frame + requested_frames)
    if end_frame <= start_frame:
        capture.release()
        raise VideoClipError("requested clip contains no decodable frames")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise VideoClipError(f"OpenCV could not create output video: {output}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    try:
        for _ in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        capture.release()
    if written != end_frame - start_frame or not output.is_file() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise VideoClipError(
            f"OpenCV decoded {written} of {end_frame - start_frame} requested frames"
        )


def extract_video_clip(
    source_path: str | Path,
    output_path: str | Path,
    *,
    start_sec: float,
    end_sec: float,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    """Extract ``[start_sec, end_sec]`` without modifying the source video."""
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source video does not exist: {source}")
    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError("clip requires 0 <= start_sec < end_sec")

    command = [
        ffmpeg_bin,
        "-v",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(source),
        "-t",
        f"{end_sec - start_sec:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        _extract_with_opencv(source, output, start_sec, end_sec)
        return
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffmpeg error"
        raise VideoClipError(f"ffmpeg could not extract clip: {detail}")
    if not output.is_file() or output.stat().st_size == 0:
        raise VideoClipError("ffmpeg returned success without a non-empty clip")
