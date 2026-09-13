"""Frame-accurate temporary clips with an explicit local/source time contract."""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from pathlib import Path


class VideoClipError(RuntimeError):
    """Raised when an experiment clip cannot be created."""


COORDINATE_CONTRACT = "source_sec = actual_source_origin_sec + local_sec"
_SHOWINFO_PTS = re.compile(r"\bpts_time:([-+0-9.eE]+)")
_SHOWINFO_CHECKSUM = re.compile(r"\bchecksum:([0-9A-Fa-f]+)")


def _timing_payload(
    *,
    backend: str,
    requested_start: float,
    requested_end: float,
    actual_origin: float,
    clip_duration: float,
    actual_frame_index: int | None,
    source_fps: float | None,
    first_source_frame_checksum: str | None,
) -> dict[str, float | int | str | None]:
    return {
        "coordinate_contract": COORDINATE_CONTRACT,
        "extraction_backend": backend,
        "requested_source_start_sec": requested_start,
        "requested_source_end_sec": requested_end,
        "actual_source_origin_sec": actual_origin,
        "clip_duration_sec": clip_duration,
        "actual_source_frame_index": actual_frame_index,
        "source_fps": source_fps,
        "first_source_frame_checksum": first_source_frame_checksum,
    }


def _extract_with_opencv(
    source: Path, output: Path, start_sec: float, end_sec: float
) -> dict[str, float | int | str | None]:
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

    # A clip starts at the first decodable frame whose timestamp is not earlier
    # than the requested source time.  This avoids silently mapping local t=0
    # to a preceding frame.
    start_frame = max(0, math.ceil(start_sec * fps - 1e-9))
    end_frame = min(frame_count, math.ceil(end_sec * fps - 1e-9))
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
    first_checksum: str | None = None
    actual_start_frame: int | None = None
    try:
        for _ in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if first_checksum is None:
                actual_start_frame = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                if actual_start_frame != start_frame:
                    raise VideoClipError(
                        "OpenCV did not seek to the requested first source frame"
                    )
                first_checksum = hashlib.sha256(frame.tobytes()).hexdigest()
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
    if actual_start_frame is None:  # pragma: no cover - covered by written-count failure
        raise VideoClipError("OpenCV did not decode a first frame")
    actual_origin = actual_start_frame / fps
    return _timing_payload(
        backend="opencv_reencode",
        requested_start=start_sec,
        requested_end=end_sec,
        actual_origin=actual_origin,
        clip_duration=written / fps,
        actual_frame_index=actual_start_frame,
        source_fps=fps,
        first_source_frame_checksum=first_checksum,
    )


def extract_video_clip(
    source_path: str | Path,
    output_path: str | Path,
    *,
    start_sec: float,
    end_sec: float,
    ffmpeg_bin: str = "ffmpeg",
) -> dict[str, float | int | str | None]:
    """Decode/re-encode ``[start_sec, end_sec)`` and return its time mapping.

    The first encoded video frame is selected by source timestamp rather than
    GOP position.  Its source timestamp is recorded as
    ``actual_source_origin_sec`` and the output timeline is reset to zero.
    """
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source video does not exist: {source}")
    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError("clip requires 0 <= start_sec < end_sec")

    video_filter = (
        f"trim=start={start_sec:.9f}:end={end_sec:.9f},"
        "showinfo,setpts=PTS-STARTPTS"
    )
    audio_filter = f"atrim=start={start_sec:.9f}:end={end_sec:.9f},asetpts=PTS-STARTPTS"
    command = [
        ffmpeg_bin,
        "-v",
        "info",
        "-y",
        "-ss",
        f"{start_sec:.9f}",
        "-copyts",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        video_filter,
        "-af",
        audio_filter,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        str(output),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return _extract_with_opencv(source, output, start_sec, end_sec)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffmpeg error"
        raise VideoClipError(f"ffmpeg could not extract clip: {detail}")
    if not output.is_file() or output.stat().st_size == 0:
        raise VideoClipError("ffmpeg returned success without a non-empty clip")
    match = _SHOWINFO_PTS.search(completed.stderr)
    if match is None:
        output.unlink(missing_ok=True)
        raise VideoClipError("ffmpeg did not expose the first decoded source timestamp")
    actual_origin = float(match.group(1))
    checksum_match = _SHOWINFO_CHECKSUM.search(completed.stderr)
    if actual_origin < start_sec - 1e-6 or actual_origin >= end_sec:
        output.unlink(missing_ok=True)
        raise VideoClipError(
            "ffmpeg first-frame source timestamp violates the requested interval"
        )
    return _timing_payload(
        backend="ffmpeg_reencode",
        requested_start=start_sec,
        requested_end=end_sec,
        actual_origin=actual_origin,
        clip_duration=end_sec - actual_origin,
        actual_frame_index=None,
        source_fps=None,
        first_source_frame_checksum=(
            None if checksum_match is None else checksum_match.group(1).upper()
        ),
    )
