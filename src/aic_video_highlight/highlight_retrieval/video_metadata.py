"""FFprobe wrapper for reading video metadata without frame extraction."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .schemas import VideoMeta


class VideoProbeError(RuntimeError):
    """Raised when FFprobe cannot provide usable video metadata."""


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _parse_rate(value: Any) -> float | None:
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = _positive_float(denominator)
        numerator_value = _positive_float(numerator)
        if numerator_value is not None and denominator_value is not None:
            return numerator_value / denominator_value
        return None
    return _positive_float(value)


def probe_video(
    video_path: str | Path,
    *,
    ffprobe_bin: str = "ffprobe",
    timeout_sec: float = 30.0,
) -> VideoMeta:
    """Return metadata from the first video stream, preferring average FPS for VFR."""
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"video file does not exist: {path}")

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        raise VideoProbeError(f"ffprobe executable not found: {ffprobe_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoProbeError(f"ffprobe timed out after {timeout_sec:g}s: {path}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffprobe error"
        raise VideoProbeError(f"ffprobe could not decode {path}: {detail}")

    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise VideoProbeError(f"ffprobe returned no usable video stream: {path}") from exc

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise VideoProbeError(f"video dimensions are missing or invalid: {path}")

    duration_sec = _positive_float(stream.get("duration"))
    if duration_sec is None:
        duration_sec = _positive_float(payload.get("format", {}).get("duration"))

    # avg_frame_rate is more representative for variable-frame-rate inputs.
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    raw_frame_count = _positive_float(stream.get("nb_frames"))
    if fps is None and raw_frame_count is not None and duration_sec is not None:
        fps = raw_frame_count / duration_sec
    if duration_sec is None and raw_frame_count is not None and fps is not None:
        duration_sec = raw_frame_count / fps
    if fps is None:
        raise VideoProbeError(f"video FPS is missing and cannot be inferred: {path}")
    if duration_sec is None:
        raise VideoProbeError(f"video duration is missing and cannot be inferred: {path}")

    frame_count = int(raw_frame_count) if raw_frame_count is not None else round(duration_sec * fps)
    return VideoMeta(
        video_id=path.stem,
        path=path,
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
    )
